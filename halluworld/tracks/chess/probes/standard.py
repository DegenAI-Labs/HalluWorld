from __future__ import annotations

import random
from typing import Optional

import chess

from halluworld.tracks.chess.rules.stockfish import StockfishHelper
from halluworld.probe import Probe, ProbeResult
from halluworld.tracks.chess.serializers import (
    FenDisplayConfig,
    derive_display_fen,
    effective_fen_display,
    piece_name,
    render_full,
    render_history,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


_ALL_PIECE_LABELS = [
    "white pawn", "white knight", "white bishop", "white rook", "white queen", "white king",
    "black pawn", "black knight", "black bishop", "black rook", "black queen", "black king",
]


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _piece_type_name(pt: chess.PieceType) -> str:
    return {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king",
    }[pt]


# --------------------------------------------------------------------------- #
# Existing probes (perceptual V + single-step legality)                        #
# --------------------------------------------------------------------------- #


class ChessPiecePresenceProbe(Probe):
    """Ask whether a specific piece is on a specific square."""

    def __init__(self, positive_rate: float = 0.5, rng: Optional[random.Random] = None):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
        empty = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]

        ask_positive = bool(occupied) and self.rng.random() < self.positive_rate
        if ask_positive:
            square = self.rng.choice(occupied)
            piece = board.piece_at(square)
            assert piece is not None
            target_piece_name = piece_name(piece)
            gt = True
            source = "occupied_square"
        else:
            if empty:
                square = self.rng.choice(empty)
                sample_square = self.rng.choice(occupied) if occupied else chess.E4
                sample_piece = board.piece_at(sample_square)
                target_piece_name = (
                    piece_name(sample_piece)
                    if sample_piece is not None
                    else "white queen"
                )
            else:
                square = self.rng.choice(occupied)
                actual = board.piece_at(square)
                assert actual is not None
                wrong = [lbl for lbl in _ALL_PIECE_LABELS if lbl != piece_name(actual)]
                target_piece_name = self.rng.choice(wrong)
            gt = False
            source = "foil"

        square_name = chess.square_name(square)
        question = (
            f"Is there a {target_piece_name} on square {square_name}?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_piece_presence",
            question=question,
            ground_truth=gt,
            metadata={
                "square": square_name,
                "piece": target_piece_name,
                "source": source,
            },
        )


class ChessLegalMoveProbe(Probe):
    """Ask whether a UCI move is legal in the current position."""

    def __init__(self, legal_rate: float = 0.5, rng: Optional[random.Random] = None):
        self.legal_rate = legal_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        legal_moves = [m.uci() for m in board.legal_moves]
        ask_legal = bool(legal_moves) and self.rng.random() < self.legal_rate

        if ask_legal:
            move_uci = self.rng.choice(legal_moves)
            gt = True
            source = "legal"
        else:
            move_uci = self._sample_illegal_uci(board, legal_moves)
            gt = False
            source = "illegal_foil"

        question = (
            f"Is the move {move_uci} legal for the side to move in this position?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_move_legality",
            question=question,
            ground_truth=gt,
            metadata={"move_uci": move_uci, "source": source},
        )

    def _sample_illegal_uci(self, board: chess.Board, legal_moves: list[str]) -> str:
        for _ in range(200):
            from_sq = self.rng.choice(chess.SQUARES)
            to_sq = self.rng.choice(chess.SQUARES)
            if from_sq == to_sq:
                continue
            candidate = chess.Move(from_sq, to_sq).uci()
            if candidate not in legal_moves:
                return candidate
        for candidate in ("a1a8", "h1h8", "e1e8", "a8a1"):
            if candidate not in legal_moves:
                return candidate
        return "a1a1"


class ChessBestMoveProbe(Probe):
    """Ask for Stockfish best move in UCI (optional; skips if unavailable)."""

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 12,
        helper: Optional[StockfishHelper] = None,
    ):
        self.stockfish_path = stockfish_path
        self.depth = depth
        self._helper = helper

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        if board.is_game_over():
            return ProbeResult(
                probe_type="chess_best_move",
                question="What is the best move in this position?",
                ground_truth=None,
                metadata={"skipped": True, "reason": "game_over"},
            )

        helper = self._helper
        created_local = False
        try:
            if helper is None:
                helper = StockfishHelper(stockfish_path=self.stockfish_path, depth=self.depth)
                created_local = True
            best_move = helper.best_move_uci(board)
        except Exception as exc:
            return ProbeResult(
                probe_type="chess_best_move",
                question="What is the best move in this position?",
                ground_truth=None,
                metadata={"skipped": True, "reason": "stockfish_unavailable", "error": str(exc)},
            )
        finally:
            if created_local and helper is not None:
                helper.close()

        return ProbeResult(
            probe_type="chess_best_move",
            question=(
                "What is the best move for the side to move in this position?\n"
                "Respond with one UCI move only (for example: e2e4)."
            ),
            ground_truth=best_move,
            metadata={"stockfish_depth": self.depth},
        )


# --------------------------------------------------------------------------- #
# Perceptual (V) — additional read-out probes                                 #
# --------------------------------------------------------------------------- #


class ChessAttackerProbe(Probe):
    """Ask whether a given square is attacked by a given color."""

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        square_scope: str = "all",
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.square_scope = square_scope

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        attacked_by_white = [sq for sq in chess.SQUARES if board.is_attacked_by(chess.WHITE, sq)]
        attacked_by_black = [sq for sq in chess.SQUARES if board.is_attacked_by(chess.BLACK, sq)]
        unattacked_by_white = [sq for sq in chess.SQUARES if sq not in set(attacked_by_white)]
        unattacked_by_black = [sq for sq in chess.SQUARES if sq not in set(attacked_by_black)]

        color = self.rng.choice([chess.WHITE, chess.BLACK])
        positives = _filter_square_scope(
            attacked_by_white if color == chess.WHITE else attacked_by_black,
            self.square_scope,
        )
        negatives = _filter_square_scope(
            unattacked_by_white if color == chess.WHITE else unattacked_by_black,
            self.square_scope,
        )

        ask_positive = bool(positives) and self.rng.random() < self.positive_rate
        if ask_positive:
            square = self.rng.choice(positives)
            gt = True
        elif negatives:
            square = self.rng.choice(negatives)
            gt = False
        else:
            square = self.rng.choice(positives) if positives else chess.E4
            gt = bool(board.is_attacked_by(color, square))

        sq_name = chess.square_name(square)
        question = (
            f"Is square {sq_name} currently attacked by {_color_name(color)}?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_attacker",
            question=question,
            ground_truth=gt,
            metadata={"square": sq_name, "attacker_color": _color_name(color)},
        )


class ChessDefendedProbe(Probe):
    """Ask whether a piece on a given square is defended by its own side."""

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        square_scope: str = "all",
        under_attack_only: bool = False,
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.square_scope = square_scope
        self.under_attack_only = under_attack_only

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        occupied = _filter_square_scope(
            [sq for sq in chess.SQUARES if board.piece_at(sq) is not None],
            self.square_scope,
        )
        if not occupied:
            return ProbeResult(
                probe_type="chess_defended",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "empty_board"},
            )

        if self.under_attack_only:
            threatened = [
                sq for sq in occupied
                if (p := board.piece_at(sq)) is not None and board.attackers(not p.color, sq)
            ]
            piece_squares = threatened if threatened else occupied
        else:
            piece_squares = occupied

        defended, undefended = [], []
        for sq in piece_squares:
            piece = board.piece_at(sq)
            assert piece is not None
            attackers = set(board.attackers(piece.color, sq))
            attackers.discard(sq)
            (defended if attackers else undefended).append(sq)

        ask_positive = bool(defended) and self.rng.random() < self.positive_rate
        if ask_positive:
            square = self.rng.choice(defended)
            gt = True
        elif undefended:
            square = self.rng.choice(undefended)
            gt = False
        else:
            square = self.rng.choice(defended) if defended else self.rng.choice(piece_squares)
            piece = board.piece_at(square)
            assert piece is not None
            attackers = set(board.attackers(piece.color, square))
            attackers.discard(square)
            gt = bool(attackers)

        piece = board.piece_at(square)
        assert piece is not None
        sq_name = chess.square_name(square)
        q = f"Is the {piece_name(piece)} on {sq_name} defended by another piece of its own side?"
        if self.under_attack_only and board.attackers(not piece.color, square):
            q = (
                f"The {piece_name(piece)} on {sq_name} is attacked by the opponent. "
                "Is it also defended by another piece of its own side?"
            )
        question = q + "\nAnswer with exactly 'yes' or 'no'."
        return ProbeResult(
            probe_type="chess_defended",
            question=question,
            ground_truth=gt,
            metadata={
                "square": sq_name,
                "piece": piece_name(piece),
                "under_attack_only": self.under_attack_only,
            },
        )


class ChessPinProbe(Probe):
    """Ask whether a piece is pinned to its own king."""

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        square_scope: str = "all",
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.square_scope = square_scope

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        # Exclude kings — "is the king pinned" is a degenerate question.
        non_king = _filter_square_scope(
            [
                sq for sq in chess.SQUARES
                if (p := board.piece_at(sq)) is not None and p.piece_type != chess.KING
            ],
            self.square_scope,
        )
        if not non_king:
            return ProbeResult(
                probe_type="chess_pinned",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_non_king_pieces"},
            )

        pinned = [sq for sq in non_king if board.is_pinned(board.piece_at(sq).color, sq)]  # type: ignore[union-attr]
        unpinned = [sq for sq in non_king if sq not in set(pinned)]

        ask_positive = bool(pinned) and self.rng.random() < self.positive_rate
        if ask_positive:
            square = self.rng.choice(pinned)
            gt = True
        elif unpinned:
            square = self.rng.choice(unpinned)
            gt = False
        else:
            square = self.rng.choice(pinned) if pinned else self.rng.choice(non_king)
            piece = board.piece_at(square)
            assert piece is not None
            gt = board.is_pinned(piece.color, square)

        piece = board.piece_at(square)
        assert piece is not None
        sq_name = chess.square_name(square)
        question = (
            f"Is the {piece_name(piece)} on {sq_name} pinned to its own king?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_pinned",
            question=question,
            ground_truth=gt,
            metadata={"square": sq_name, "piece": piece_name(piece)},
        )


class ChessPieceCountProbe(Probe):
    """Ask: 'How many <color> <piece_type>s are on the board?'"""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        color = self.rng.choice([chess.WHITE, chess.BLACK])
        piece_type = self.rng.choice([
            chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING
        ])
        count = len(board.pieces(piece_type, color))

        plural = "s" if piece_type != chess.KING else ""
        # Kings: ground truth is always 0 or 1; phrasing handled below.
        if piece_type == chess.KING:
            question = (
                f"How many {_color_name(color)} kings are currently on the board?\n"
                "Reply with a single integer."
            )
        else:
            question = (
                f"How many {_color_name(color)} {_piece_type_name(piece_type)}{plural} are currently on the board?\n"
                "Reply with a single integer."
            )

        return ProbeResult(
            probe_type="chess_piece_count",
            question=question,
            ground_truth=count,
            metadata={
                "color": _color_name(color),
                "piece_type": _piece_type_name(piece_type),
            },
        )


# --------------------------------------------------------------------------- #
# Memory / persistence (H) — history-only probes                              #
# --------------------------------------------------------------------------- #


def _replay_random(
    start_board: chess.Board,
    n_plies: int,
    rng: random.Random,
) -> tuple[chess.Board, list[str]]:
    """Push up to ``n_plies`` random legal moves and return (final_board, uci_moves)."""
    board = start_board.copy(stack=False)
    moves: list[str] = []
    for _ in range(n_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = rng.choice(legal)
        moves.append(mv.uci())
        board.push(mv)
    return board, moves


def _replay_random_capture_biased(
    start_board: chess.Board,
    n_plies: int,
    rng: random.Random,
    capture_bias: float,
) -> tuple[chess.Board, list[str]]:
    """Like ``_replay_random`` but skews toward captures when ``capture_bias`` is high.

    Makes SAN / history reconstruction harder (changing material and tactics).
    """
    board = start_board.copy(stack=False)
    moves: list[str] = []
    bias = max(0.0, min(1.0, capture_bias))
    for _ in range(n_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        captures = [m for m in legal if board.is_capture(m)]
        if captures and rng.random() < bias:
            mv = rng.choice(captures)
        else:
            mv = rng.choice(legal)
        moves.append(mv.uci())
        board.push(mv)
    return board, moves


def _legal_moves_for_reply_constraint(board: chess.Board, constraint: str | None) -> list[chess.Move]:
    """Filter legal moves for SAN / UCI reply constraints (still fully determined by the board)."""
    legal = list(board.legal_moves)
    c = (constraint or "any").strip().lower()
    if c in ("", "any", "none"):
        return legal
    if c == "non_capture":
        return [m for m in legal if not board.is_capture(m)]
    if c == "non_promotion":
        return [m for m in legal if m.promotion is None]
    if c == "quiet":
        # Non-capture and does not give check (strict “quiet” move).
        out: list[chess.Move] = []
        for m in legal:
            if board.is_capture(m):
                continue
            tmp = board.copy(stack=False)
            tmp.push(m)
            if tmp.is_check():
                continue
            out.append(m)
        return out
    if c in ("quiet_non_promotion", "quiet_np"):
        return [m for m in _legal_moves_for_reply_constraint(board, "quiet") if m.promotion is None]
    raise ValueError(
        f"Unknown reply constraint {constraint!r}; use any, non_capture, non_promotion, quiet, "
        "or quiet_non_promotion"
    )


def _filter_square_scope(squares: list[int], scope: str) -> list[int]:
    """Optionally keep only inner-board squares (files b–g, ranks 3–6)."""
    if scope != "midboard":
        return squares
    inner = [
        sq for sq in squares
        if 2 <= chess.square_rank(sq) <= 5 and 1 <= chess.square_file(sq) <= 6
    ]
    return inner if inner else squares


class ChessHistoryReadoutProbe(Probe):
    """Hide the board, show only the move history, ask what is on a square.

    The observation override is a 'history-only' rendering (start FEN +
    move list, no grid). Tests whether the LM can maintain ``H`` instead
    of reading off ``V``.
    """

    def __init__(self, n_plies: int = 6, rng: Optional[random.Random] = None):
        self.n_plies = n_plies
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        start_fen = env.board.fen()
        final_board, moves = _replay_random(env.board, self.n_plies, self.rng)
        if not moves:
            return ProbeResult(
                probe_type="chess_history_readout",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_legal_moves"},
            )

        # Bias toward squares that changed during the replay so the question
        # is non-trivial; fall back to any square if nothing changed.
        changed = [
            sq for sq in chess.SQUARES
            if (chess.Board(start_fen).piece_at(sq) != final_board.piece_at(sq))
        ]
        square = self.rng.choice(changed) if changed else self.rng.choice(chess.SQUARES)
        piece = final_board.piece_at(square)
        gt = piece_name(piece) if piece is not None else "empty"

        observation = render_history(start_fen, moves)
        question = (
            f"After replaying the moves listed above from the starting FEN, "
            f"what is on square {chess.square_name(square)}?\n"
            "Reply with the canonical piece name (e.g. 'white knight') or 'empty'."
        )
        return ProbeResult(
            probe_type="chess_history_readout",
            question=question,
            ground_truth=gt,
            metadata={
                "n_plies": len(moves),
                "square": chess.square_name(square),
                "moves_uci": moves,
            },
            observation_override=observation,
        )


class ChessCaptureCountProbe(Probe):
    """Hide the board, show move history, ask how many captures occurred."""

    def __init__(
        self,
        n_plies: int = 8,
        rng: Optional[random.Random] = None,
        capture_bias: float = 0.0,
    ):
        self.n_plies = n_plies
        self.rng = rng or random.Random()
        self.capture_bias = max(0.0, min(1.0, capture_bias))

    def generate(self, env) -> ProbeResult:
        start_fen = env.board.fen()
        board = env.board.copy(stack=False)
        moves: list[str] = []
        captures = 0
        for _ in range(self.n_plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            cap_moves = [m for m in legal if board.is_capture(m)]
            if cap_moves and self.rng.random() < self.capture_bias:
                mv = self.rng.choice(cap_moves)
            else:
                mv = self.rng.choice(legal)
            if board.is_capture(mv):
                captures += 1
            board.push(mv)
            moves.append(mv.uci())

        if not moves:
            return ProbeResult(
                probe_type="chess_capture_count",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_legal_moves"},
            )

        observation = render_history(start_fen, moves)
        question = (
            "How many captures (including en passant) occurred during the moves listed above?\n"
            "Reply with a single integer."
        )
        return ProbeResult(
            probe_type="chess_capture_count",
            question=question,
            ground_truth=captures,
            metadata={"n_plies": len(moves), "moves_uci": moves},
            observation_override=observation,
        )


def _undefended_piece_count(board: chess.Board, color: chess.Color) -> int:
    """Non-king pieces of *color* with no friendly piece attacking their square."""
    n = 0
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p is None or p.color != color or p.piece_type == chess.KING:
            continue
        if not board.attackers(color, sq):
            n += 1
    return n


def _has_friendly_defender(board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
    p = board.piece_at(sq)
    if p is None or p.color != color:
        return False
    return bool(board.attackers(color, sq))


def _castling_rook_from_to(board: chess.Board, move: chess.Move) -> tuple[chess.Square, chess.Square] | None:
    if not board.is_castling(move):
        return None
    r = chess.square_rank(move.from_square)
    df = chess.square_file(move.to_square) - chess.square_file(move.from_square)
    if df == 2:
        return chess.square(7, r), chess.square(5, r)
    if df == -2:
        return chess.square(0, r), chess.square(3, r)
    return None


def _origins_non_king(board: chess.Board, color: chess.Color) -> dict[chess.Square, chess.Square]:
    return {
        sq: sq
        for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None
        and p.color == color
        and p.piece_type != chess.KING
    }


def _update_origins_after_move(
    loc: dict[chess.Square, chess.Square],
    b_before: chess.Board,
    mv: chess.Move,
    color_track: chess.Color,
) -> None:
    """Mutate *loc* (current_square -> origin_square on *b_before*) for *color_track* non-king pieces."""
    mover = b_before.turn
    from_sq, to_sq = mv.from_square, mv.to_square
    if mover == color_track:
        cr = _castling_rook_from_to(b_before, mv)
        if cr is not None:
            rf, rt = cr
            if from_sq in loc:
                loc[to_sq] = loc.pop(from_sq)
            if rf in loc:
                loc[rt] = loc.pop(rf)
        else:
            if from_sq in loc:
                loc[to_sq] = loc.pop(from_sq)
    else:
        if to_sq in loc:
            loc.pop(to_sq)


def _newly_undefended_after_hypothetical_moves(
    b0: chess.Board, moves: list[chess.Move], color: chess.Color
) -> int:
    """Pieces of *color* (non-king) that had a friendly defender on *b0* at their start square but not after *moves*."""
    loc = _origins_non_king(b0, color)
    b = b0.copy(stack=False)
    for mv in moves:
        _update_origins_after_move(loc, b, mv, color)
        b.push(mv)
    n = 0
    for sq_cur, origin_sq in loc.items():
        if _has_friendly_defender(b0, origin_sq, color) and not _has_friendly_defender(b, sq_cur, color):
            n += 1
    return n


def _is_hanging(board: chess.Board, sq: chess.Square) -> bool:
    p = board.piece_at(sq)
    if p is None or p.piece_type == chess.KING:
        return False
    if not board.attackers(not p.color, sq):
        return False
    defenders = set(board.attackers(p.color, sq))
    defenders.discard(sq)
    return len(defenders) == 0


class ChessHiddenSideCaptureStatsProbe(Probe):
    """Long hidden SAN/UCI history; ask for a side-specific capture / material-loss count (integer)."""

    def __init__(
        self,
        min_plies: int = 25,
        max_plies: int = 40,
        rng: Optional[random.Random] = None,
        capture_bias: float = 0.48,
    ):
        lo = min(min_plies, max_plies)
        hi = max(min_plies, max_plies)
        self.min_plies = max(4, lo)
        self.max_plies = max(self.min_plies, hi)
        self.rng = rng or random.Random()
        self.capture_bias = max(0.0, min(1.0, capture_bias))

    def generate(self, env) -> ProbeResult:
        start_fen = env.board.fen()
        board = chess.Board(start_fen)
        target = self.rng.randint(self.min_plies, self.max_plies)
        moves: list[str] = []
        white_captures = 0
        black_pieces_lost = 0
        white_pieces_lost = 0

        for _ in range(target):
            legal = list(board.legal_moves)
            if not legal:
                break
            cap_moves = [m for m in legal if board.is_capture(m)]
            if cap_moves and self.rng.random() < self.capture_bias:
                mv = self.rng.choice(cap_moves)
            else:
                mv = self.rng.choice(legal)

            if board.is_capture(mv):
                if board.turn == chess.WHITE:
                    white_captures += 1
                if board.is_en_passant(mv):
                    victim_sq = mv.to_square + (-8 if board.turn == chess.WHITE else 8)
                    victim = board.piece_at(victim_sq)
                else:
                    victim = board.piece_at(mv.to_square)
                if victim is not None:
                    if victim.color == chess.BLACK:
                        black_pieces_lost += 1
                    else:
                        white_pieces_lost += 1

            board.push(mv)
            moves.append(mv.uci())

        if len(moves) < self.min_plies // 2:
            return ProbeResult(
                probe_type="chess_hidden_side_capture_stats",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "sequence_too_short"},
            )

        ask_white_caps = self.rng.random() < 0.5
        if ask_white_caps:
            gt = white_captures
            q = (
                "How many **capturing moves did white make** during the sequence above "
                "(each capture by white counts once; promotions that capture count)?\n"
                "Reply with a single non-negative integer."
            )
            q_kind = "white_captures"
        else:
            gt = black_pieces_lost
            q = (
                "How many **black pieces were removed by captures** during the sequence above "
                "(each time a black piece is taken off the board counts once)?\n"
                "Reply with a single non-negative integer."
            )
            q_kind = "black_pieces_lost"

        observation = render_history(start_fen, moves, companion_board=True)
        return ProbeResult(
            probe_type="chess_hidden_side_capture_stats",
            question=q,
            ground_truth=gt,
            metadata={
                "n_plies": len(moves),
                "moves_uci": moves,
                "question_kind": q_kind,
                "white_pieces_lost_debug": white_pieces_lost,
            },
            observation_override=observation,
            prepend_env_observation=True,
        )


class ChessAfterMoveUndefendedCountProbe(Probe):
    """Hypothetical move(s): count undefended pieces (``post``) or newly undefended (``delta``)."""

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        *,
        mode: str = "post",
        hypothesis_plies: int = 1,
        max_tries: int = 60,
    ):
        if mode not in ("post", "delta"):
            raise ValueError(f"mode must be post or delta, got {mode!r}")
        if hypothesis_plies not in (1, 2):
            raise ValueError("hypothesis_plies must be 1 or 2")
        self.rng = rng or random.Random()
        self.mode = mode
        self.hypothesis_plies = hypothesis_plies
        self.max_tries = max_tries

    def generate(self, env) -> ProbeResult:
        board0 = env.board
        color0 = board0.turn

        if self.hypothesis_plies == 1:
            legal1 = list(board0.legal_moves)
            if not legal1:
                return ProbeResult(
                    probe_type="chess_after_move_undefended_count",
                    question="(skipped)",
                    ground_truth=None,
                    metadata={"skipped": True, "reason": "no_legal_moves"},
                )
            mv = self.rng.choice(legal1)
            moves = [mv]
            ucis = [mv.uci()]
        else:
            moves = []
            for _ in range(self.max_tries):
                legal1 = list(board0.legal_moves)
                if not legal1:
                    break
                m1 = self.rng.choice(legal1)
                b1 = board0.copy(stack=False)
                b1.push(m1)
                legal2 = list(b1.legal_moves)
                if not legal2:
                    continue
                m2 = self.rng.choice(legal2)
                moves = [m1, m2]
                ucis = [m1.uci(), m2.uci()]
                break
            if len(moves) != 2:
                return ProbeResult(
                    probe_type="chess_after_move_undefended_count",
                    question="(skipped)",
                    ground_truth=None,
                    metadata={"skipped": True, "reason": "no_two_legal_plies"},
                )

        b_final = board0.copy(stack=False)
        for mv in moves:
            b_final.push(mv)

        if self.mode == "post":
            gt = _undefended_piece_count(b_final, color0)
        else:
            gt = _newly_undefended_after_hypothetical_moves(board0, moves, color0)

        side = _color_name(color0)
        if self.hypothesis_plies == 1:
            if self.mode == "post":
                q_body = (
                    f"how many **{side}** pieces **excluding kings** have **no** friendly defender "
                    "(no friendly piece attacks their square)?"
                )
            else:
                q_body = (
                    f"how many **{side}** pieces **excluding kings** had at least one friendly defender "
                    "in the starting position **before** that move, but have **no** friendly defender **after** it?"
                )
            question = (
                f"After **{side}** plays `{ucis[0]}` from the starting FEN above, {q_body}\n"
                "Reply with a single non-negative integer."
            )
        else:
            b_mid = board0.copy(stack=False)
            b_mid.push(moves[0])
            side2 = _color_name(b_mid.turn)
            if self.mode == "post":
                q_body = (
                    f"how many **{side}** pieces **excluding kings** have **no** friendly defender "
                    "after **both** moves?"
                )
            else:
                q_body = (
                    f"how many **{side}** pieces **excluding kings** had a friendly defender in the starting position, "
                    "but have **none** after **both** moves?"
                )
            question = (
                f"If **{side}** plays `{ucis[0]}` and then **{side2}** plays `{ucis[1]}` from the starting FEN above, "
                f"{q_body}\n"
                "Reply with a single non-negative integer."
            )

        return ProbeResult(
            probe_type="chess_after_move_undefended_count",
            question=question,
            ground_truth=gt,
            metadata={
                "setup_uci": ucis[0] if len(ucis) == 1 else None,
                "moves_uci": ucis,
                "side": side,
                "mode": self.mode,
                "hypothesis_plies": self.hypothesis_plies,
            },
        )


class ChessTwoStepHangingProbe(Probe):
    """Two legal plies from the current position; ask whether a chosen piece is hanging after both."""

    def __init__(self, rng: Optional[random.Random] = None, max_tries: int = 60):
        self.rng = rng or random.Random()
        self.max_tries = max_tries

    def generate(self, env) -> ProbeResult:
        board0 = env.board
        for _ in range(self.max_tries):
            legal1 = list(board0.legal_moves)
            if not legal1:
                break
            m1 = self.rng.choice(legal1)
            b1 = board0.copy(stack=False)
            b1.push(m1)
            legal2 = list(b1.legal_moves)
            if not legal2:
                continue
            m2 = self.rng.choice(legal2)
            b2 = b1.copy(stack=False)
            b2.push(m2)

            candidates = [
                sq
                for sq in chess.SQUARES
                if (p := b2.piece_at(sq)) is not None and p.piece_type != chess.KING
            ]
            if not candidates:
                continue
            sq = self.rng.choice(candidates)
            piece = b2.piece_at(sq)
            assert piece is not None
            gt = _is_hanging(b2, sq)
            sq_name = chess.square_name(sq)
            question = (
                f"If {_color_name(board0.turn)} plays `{m1.uci()}` and then "
                f"{_color_name(b1.turn)} plays `{m2.uci()}` from the starting FEN above, "
                f"is the {piece_name(piece)} on **{sq_name}** hanging "
                "(attacked by the opponent and not defended by any friendly piece)?\n"
                "Answer with exactly 'yes' or 'no'."
            )
            return ProbeResult(
                probe_type="chess_two_step_hanging",
                question=question,
                ground_truth=gt,
                metadata={"moves_uci": [m1.uci(), m2.uci()], "square": sq_name},
            )

        return ProbeResult(
            probe_type="chess_two_step_hanging",
            question="(skipped)",
            ground_truth=None,
            metadata={"skipped": True, "reason": "no_two_legal_plies"},
        )


class ChessMateStalemateConfusionProbe(Probe):
    """Random play from standard start until mate or stalemate; mis-ask the other terminal (board hidden)."""

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        max_attempts: int = 120,
        max_plies: int = 120,
        capture_bias: float = 0.42,
    ):
        self.rng = rng or random.Random()
        self.max_attempts = max_attempts
        self.max_plies = max_plies
        self.capture_bias = max(0.0, min(1.0, capture_bias))

    def generate(self, env) -> ProbeResult:
        for _ in range(self.max_attempts):
            b = chess.Board()
            moves: list[chess.Move] = []
            for _ in range(self.max_plies):
                if b.is_game_over(claim_draw=False):
                    break
                legal = list(b.legal_moves)
                if not legal:
                    break
                caps = [m for m in legal if b.is_capture(m)]
                if caps and self.rng.random() < self.capture_bias:
                    mv = self.rng.choice(caps)
                else:
                    mv = self.rng.choice(legal)
                moves.append(mv)
                b.push(mv)

            if not b.is_game_over(claim_draw=False):
                continue
            if b.is_checkmate():
                wrong_label = "stalemate"
                gt = False
                q = "Is the position **stalemate**?\nAnswer with exactly 'yes' or 'no'."
            elif b.is_stalemate():
                wrong_label = "checkmate"
                gt = False
                q = "Is the position **checkmate**?\nAnswer with exactly 'yes' or 'no'."
            else:
                continue

            observation = render_history(chess.STARTING_FEN, [m.uci() for m in moves])
            return ProbeResult(
                probe_type="chess_mate_stalemate_confusion",
                question=q,
                ground_truth=gt,
                metadata={
                    "n_plies": len(moves),
                    "moves_uci": [m.uci() for m in moves],
                    "true_terminal": "checkmate" if b.is_checkmate() else "stalemate",
                    "misleading_question": wrong_label,
                },
                observation_override=observation,
            )

        return ProbeResult(
            probe_type="chess_mate_stalemate_confusion",
            question="(skipped)",
            ground_truth=None,
            metadata={"skipped": True, "reason": "no_mate_or_stalemate_sampled"},
        )


class ChessCastlingRightsHistoryProbe(Probe):
    """Hide the board, show move history, ask whether a castling right still exists."""

    def __init__(self, n_plies: int = 6, rng: Optional[random.Random] = None):
        self.n_plies = n_plies
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        start_fen = env.board.fen()
        final_board, moves = _replay_random(env.board, self.n_plies, self.rng)
        if not moves:
            return ProbeResult(
                probe_type="chess_castling_rights_history",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_legal_moves"},
            )

        color = self.rng.choice([chess.WHITE, chess.BLACK])
        side = self.rng.choice(["kingside", "queenside"])
        if side == "kingside":
            gt = final_board.has_kingside_castling_rights(color)
        else:
            gt = final_board.has_queenside_castling_rights(color)

        observation = render_history(start_fen, moves)
        question = (
            f"After replaying the moves listed above, does {_color_name(color)} "
            f"still have {side} castling rights?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_castling_rights_history",
            question=question,
            ground_truth=gt,
            metadata={
                "n_plies": len(moves),
                "color": _color_name(color),
                "side": side,
                "moves_uci": moves,
            },
            observation_override=observation,
        )


# --------------------------------------------------------------------------- #
# Causal / dynamics — hypothetical and rule-driven probes                     #
# --------------------------------------------------------------------------- #


_HYPOTHETICAL_PREDICATES = (
    "in_check",
    "is_capture",
    "en_passant_available",
    "checkmate",
)


class ChessHypotheticalProbe(Probe):
    """Parametric 'after move X, is predicate P true?' probe.

    Supported predicates:
      * ``"in_check"``: is the side to move (after the setup move) in check?
      * ``"is_capture"``: was the setup move a capture?
      * ``"en_passant_available"``: is en passant legal in the resulting position?
      * ``"checkmate"``: is the resulting position checkmate?
    """

    def __init__(
        self,
        predicate: str = "in_check",
        rng: Optional[random.Random] = None,
        setup_capture_bias: float = 0.0,
        setup_pool: str = "all",
    ):
        if predicate not in _HYPOTHETICAL_PREDICATES:
            raise ValueError(
                f"Unknown predicate {predicate!r}; choose from {_HYPOTHETICAL_PREDICATES}"
            )
        if setup_pool not in ("all", "non_capture"):
            raise ValueError(f"setup_pool must be 'all' or 'non_capture', got {setup_pool!r}")
        self.predicate = predicate
        self.rng = rng or random.Random()
        self.setup_capture_bias = max(0.0, min(1.0, setup_capture_bias))
        self.setup_pool = setup_pool

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        legal = list(board.legal_moves)
        if not legal:
            return ProbeResult(
                probe_type=f"chess_hypothetical_{self.predicate}",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_legal_moves"},
            )

        pool = legal
        if self.setup_pool == "non_capture" and self.predicate != "is_capture":
            nc = [m for m in legal if not board.is_capture(m)]
            if nc:
                pool = nc

        cap_legal = [m for m in pool if board.is_capture(m)]
        if cap_legal and self.rng.random() < self.setup_capture_bias:
            setup = self.rng.choice(cap_legal)
        else:
            setup = self.rng.choice(pool)
        is_capture = board.is_capture(setup)
        future = board.copy(stack=False)
        future.push(setup)

        if self.predicate == "in_check":
            gt = future.is_check()
            phrase = "the side to move would be in check"
        elif self.predicate == "is_capture":
            gt = bool(is_capture)
            phrase = "this move would be a capture"
        elif self.predicate == "en_passant_available":
            gt = future.has_legal_en_passant()
            phrase = "en passant would be legal on the next move"
        elif self.predicate == "checkmate":
            gt = future.is_checkmate()
            phrase = "the position would be checkmate"
        else:  # defensive — should be unreachable due to constructor check
            raise ValueError(self.predicate)

        question = (
            f"If {_color_name(board.turn)} plays {setup.uci()} from the position above, "
            f"would it be true that {phrase}?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type=f"chess_hypothetical_{self.predicate}",
            question=question,
            ground_truth=gt,
            metadata={
                "setup_move": setup.uci(),
                "predicate": self.predicate,
                "setup_pool": self.setup_pool,
            },
        )


class ChessEnPassantProbe(Probe):
    """Force an EP-eligible setup, then ask whether en passant is currently legal.

    Looks for any legal 2-square pawn push for the side to move; if one exists,
    it is played as the setup move and the resulting position is shown via an
    observation override. The question asks whether en passant is legal.
    """

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        include_legal_moves: bool = True,
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.include_legal_moves = include_legal_moves

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        double_pushes = []
        for m in board.legal_moves:
            piece = board.piece_at(m.from_square)
            if piece is None or piece.piece_type != chess.PAWN:
                continue
            if abs(chess.square_rank(m.from_square) - chess.square_rank(m.to_square)) == 2:
                double_pushes.append(m)

        if not double_pushes:
            return ProbeResult(
                probe_type="chess_en_passant",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_double_pawn_push"},
            )

        # Prefer EP-creating pushes if the user wants a positive question.
        ep_creators = []
        non_creators = []
        for m in double_pushes:
            tmp = board.copy(stack=False)
            tmp.push(m)
            (ep_creators if tmp.has_legal_en_passant() else non_creators).append(m)

        ask_positive = bool(ep_creators) and self.rng.random() < self.positive_rate
        if ask_positive:
            setup = self.rng.choice(ep_creators)
        elif non_creators:
            setup = self.rng.choice(non_creators)
        else:
            setup = self.rng.choice(double_pushes)

        future = board.copy(stack=False)
        future.push(setup)
        gt = future.has_legal_en_passant()
        ep_square = chess.square_name(future.ep_square) if future.ep_square is not None else None

        observation = render_full(
            future,
            include_legal_moves=self.include_legal_moves,
        )
        observation = (
            f"Hypothetical position after {_color_name(board.turn)} plays {setup.uci()}:\n"
            + observation
        )
        question = (
            "In the position shown above, is en passant legal for the side to move?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_en_passant",
            question=question,
            ground_truth=gt,
            metadata={
                "setup_move": setup.uci(),
                "ep_square": ep_square,
            },
            observation_override=observation,
        )


class ChessCastlingLegalityProbe(Probe):
    """Ask whether a specific castling move is legal *right now*.

    Catches the classic 'castling through / into check' failure mode: the
    castling rights flag may still be set, but the move is illegal because
    the king passes through an attacked square.
    """

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        side = board.turn
        candidates = {
            "kingside": chess.Move.from_uci("e1g1" if side == chess.WHITE else "e8g8"),
            "queenside": chess.Move.from_uci("e1c1" if side == chess.WHITE else "e8c8"),
        }
        legal_candidates = [(d, m) for d, m in candidates.items() if m in board.legal_moves]
        illegal_candidates = [(d, m) for d, m in candidates.items() if m not in board.legal_moves]

        ask_positive = bool(legal_candidates) and self.rng.random() < self.positive_rate
        if ask_positive:
            direction, move = self.rng.choice(legal_candidates)
            gt = True
        elif illegal_candidates:
            direction, move = self.rng.choice(illegal_candidates)
            gt = False
        elif legal_candidates:
            direction, move = self.rng.choice(legal_candidates)
            gt = True
        else:
            direction, move = self.rng.choice(list(candidates.items()))
            gt = move in board.legal_moves

        question = (
            f"Can {_color_name(side)} legally castle {direction} right now?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_castling_legality",
            question=question,
            ground_truth=gt,
            metadata={
                "side": _color_name(side),
                "direction": direction,
                "move_uci": move.uci(),
            },
        )


class ChessMateInOneProbe(Probe):
    """Ask whether there exists a mate-in-1 for the side to move."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        if board.is_game_over():
            return ProbeResult(
                probe_type="chess_mate_in_one",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "game_over"},
            )

        gt = False
        winning_move = None
        for mv in board.legal_moves:
            board.push(mv)
            if board.is_checkmate():
                gt = True
                winning_move = mv.uci()
                board.pop()
                break
            board.pop()

        question = (
            f"Does {_color_name(board.turn)} have a mate-in-1 in this position?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_mate_in_one",
            question=question,
            ground_truth=gt,
            metadata={"mate_move": winning_move},
        )


class ChessHangingProbe(Probe):
    """Ask whether the piece on a given square is hanging (attacked + undefended)."""

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        square_scope: str = "all",
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.square_scope = square_scope

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        # Skip kings — being "hanging" doesn't apply.
        candidates = _filter_square_scope(
            [
                sq for sq in chess.SQUARES
                if (p := board.piece_at(sq)) is not None and p.piece_type != chess.KING
            ],
            self.square_scope,
        )
        if not candidates:
            return ProbeResult(
                probe_type="chess_hanging",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_non_king_pieces"},
            )

        hanging, safe = [], []
        for sq in candidates:
            piece = board.piece_at(sq)
            assert piece is not None
            attackers = bool(board.attackers(not piece.color, sq))
            defenders = set(board.attackers(piece.color, sq))
            defenders.discard(sq)
            (hanging if attackers and not defenders else safe).append(sq)

        ask_positive = bool(hanging) and self.rng.random() < self.positive_rate
        if ask_positive:
            square = self.rng.choice(hanging)
            gt = True
        elif safe:
            square = self.rng.choice(safe)
            gt = False
        else:
            square = self.rng.choice(candidates)
            piece = board.piece_at(square)
            assert piece is not None
            attackers = bool(board.attackers(not piece.color, square))
            defenders = set(board.attackers(piece.color, square))
            defenders.discard(square)
            gt = attackers and not defenders

        piece = board.piece_at(square)
        assert piece is not None
        sq_name = chess.square_name(square)
        question = (
            f"Is the {piece_name(piece)} on {sq_name} hanging "
            f"(attacked by the opponent and not defended by any of its own pieces)?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_hanging",
            question=question,
            ground_truth=bool(gt),
            metadata={"square": sq_name, "piece": piece_name(piece)},
        )


class ChessCanCaptureProbe(Probe):
    """Ask whether the side to move can capture the piece on a given square."""

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        square_scope: str = "all",
        prefer_ghost_captures: bool = False,
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.square_scope = square_scope
        self.prefer_ghost_captures = prefer_ghost_captures

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        opp_squares = _filter_square_scope(
            [
                sq for sq in chess.SQUARES
                if (p := board.piece_at(sq)) is not None and p.color != board.turn
            ],
            self.square_scope,
        )
        if not opp_squares:
            return ProbeResult(
                probe_type="chess_can_capture",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_opponent_pieces"},
            )

        capturable = set()
        for mv in board.legal_moves:
            if board.is_capture(mv):
                # En passant: target square is empty on the board, but the
                # captured pawn lives next to it. Use mv.to_square anyway —
                # it's the canonical "destination" of the capture for this probe.
                capturable.add(mv.to_square)

        positives = [sq for sq in opp_squares if sq in capturable]
        negatives = [sq for sq in opp_squares if sq not in capturable]

        ghost_negatives: list[int] = []
        if self.prefer_ghost_captures:
            pseudo_cap_to: set[int] = set()
            for mv in board.generate_pseudo_legal_moves():
                if board.is_capture(mv):
                    pseudo_cap_to.add(mv.to_square)
            legal_cap_to = set(capturable)
            ghost_to = pseudo_cap_to - legal_cap_to
            ghost_negatives = [sq for sq in opp_squares if sq in ghost_to]

        ask_positive = bool(positives) and self.rng.random() < self.positive_rate
        if ask_positive:
            square = self.rng.choice(positives)
            gt = True
        elif ghost_negatives:
            square = self.rng.choice(ghost_negatives)
            gt = False
        elif negatives:
            square = self.rng.choice(negatives)
            gt = False
        else:
            square = self.rng.choice(opp_squares)
            gt = square in capturable

        piece = board.piece_at(square)
        assert piece is not None
        sq_name = chess.square_name(square)
        question = (
            f"Can {_color_name(board.turn)} legally capture the {piece_name(piece)} "
            f"on {sq_name} on the next move?\n"
            "Answer with exactly 'yes' or 'no'."
        )
        return ProbeResult(
            probe_type="chess_can_capture",
            question=question,
            ground_truth=gt,
            metadata={
                "square": sq_name,
                "target_piece": piece_name(piece),
                "prefer_ghost_captures": self.prefer_ghost_captures,
                "ghost_negative": bool(ghost_negatives) and square in ghost_negatives,
            },
        )


# --------------------------------------------------------------------------- #
# Belief / uncertainty (P)                                                    #
# --------------------------------------------------------------------------- #


class ChessHiddenSquareProbe(Probe):
    """Mask one square from the rendered board and ask about it.

    The model should respond ``idk`` (rather than guess yes or no) when the
    answer depends on a hidden square. With probability ``unmasked_rate`` the
    question targets a non-masked square instead, and a definite yes/no is
    expected — this prevents the model from learning to always say idk.
    """

    def __init__(
        self,
        n_hidden: int = 1,
        unmasked_rate: float = 0.5,
        rng: Optional[random.Random] = None,
    ):
        self.n_hidden = max(1, n_hidden)
        self.unmasked_rate = unmasked_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        squares = list(chess.SQUARES)
        self.rng.shuffle(squares)
        hidden = set(squares[: self.n_hidden])

        ask_unmasked = self.rng.random() < self.unmasked_rate

        if ask_unmasked:
            visible = [sq for sq in chess.SQUARES if sq not in hidden]
            target = self.rng.choice(visible)
        else:
            target = self.rng.choice(list(hidden))

        # Build a presence-style question; foil with 50% probability.
        actual_piece = board.piece_at(target)
        ask_correct = self.rng.random() < 0.5
        if ask_correct and actual_piece is not None:
            piece_label = piece_name(actual_piece)
        elif actual_piece is not None:
            piece_label = self.rng.choice(
                [lbl for lbl in _ALL_PIECE_LABELS if lbl != piece_name(actual_piece)]
            )
        else:
            piece_label = self.rng.choice(_ALL_PIECE_LABELS)

        if target in hidden:
            gt: object = "idk"
        else:
            gt = (actual_piece is not None and piece_name(actual_piece) == piece_label)

        observation = render_full(board, include_legal_moves=False, mask=hidden)
        question = (
            f"Is there a {piece_label} on square {chess.square_name(target)}?\n"
            "Answer with exactly 'yes', 'no', or 'idk' (when the relevant square is hidden)."
        )
        return ProbeResult(
            probe_type="chess_hidden_square",
            question=question,
            ground_truth=gt,
            metadata={
                "target_square": chess.square_name(target),
                "hidden_squares": [chess.square_name(s) for s in sorted(hidden)],
                "asked_about_hidden": target in hidden,
                "piece_label": piece_label,
            },
            observation_override=observation,
        )


class ChessSanLegalContinuationProbe(Probe):
    """SAN-only movetext (no board grid): ask for one legal next move in UCI.

    The model must track material and side-to-move from algebraic notation
    alone — a failure mode where LMs often propose pseudo-legal moves (wrong
    piece still on a square, etc.). Ground truth is the set of legal UCIs in the
    terminal position that satisfy ``reply_constraint`` (default: any legal move).

    Use ``capture_bias`` > 0 and large ``n_plies`` for a severe stress test
    (tactical chaos + long horizon). ``max_replay_attempts`` retries random
    replays until enough moves exist and the game is not already over.

    ``observation_mode``:
      * ``"fen_and_san"`` — include ``Starting FEN:`` plus the SAN line (default).
      * ``"san_only"`` — SAN line only, no FEN. Requires ``replay_from="startpos"``
        so the line is a complete game prefix from the standard start; a midgame
        FEN cannot be reconstructed from SAN alone.
    ``replay_from``:
      * ``"env"`` — random replay from the current environment board (Lichess FEN, etc.).
      * ``"startpos"`` — random replay always from the standard starting position.

    ``reply_constraint`` (answer must still be one legal UCI in the final position):
      * ``"any"`` — any legal move (default).
      * ``"non_capture"`` — must not capture.
      * ``"non_promotion"`` — must not promote a pawn.
      * ``"quiet"`` — non-capture and does not give check.
      * ``"quiet_non_promotion"`` (alias ``quiet_np``) — quiet and not a promotion.

    ``max_constrained_replies``: if set, reject terminal positions where more than this
    many moves satisfy ``reply_constraint`` (retries another replay). Use to force
    tactically tight positions with a small answer set.

    ``omit_side_to_move``: if True and ``replay_from="startpos"``, do not print which
    side is to move (infer from the move count from the initial position).

    ``prepend_terminal_board_grid``: if True, prepend :func:`render_full` for the **terminal**
    position (ASCII grid + optional ``FEN:`` line using the same ``fen_display`` rules as the
    serializer) before the SAN text. ``chess_full_probes`` sets this from ``INCLUDE_FEN`` so
    ``INCLUDE_FEN=1`` shows a diagram alongside ``san_only`` / ``fen_and_san`` movetext.

    If the final position has no moves satisfying the constraint, the probe retries
    another random replay (same as insufficient plies).
    """

    def __init__(
        self,
        n_plies: int = 32,
        min_plies: int = 12,
        rng: Optional[random.Random] = None,
        capture_bias: float = 0.25,
        max_replay_attempts: int = 16,
        observation_mode: str = "fen_and_san",
        replay_from: str = "env",
        reply_constraint: str = "any",
        max_constrained_replies: Optional[int] = None,
        omit_side_to_move: bool = False,
        prepend_terminal_board_grid: bool = False,
    ):
        if observation_mode not in ("fen_and_san", "san_only"):
            raise ValueError(f"observation_mode must be 'fen_and_san' or 'san_only', got {observation_mode!r}")
        if replay_from not in ("env", "startpos"):
            raise ValueError(f"replay_from must be 'env' or 'startpos', got {replay_from!r}")
        if observation_mode == "san_only" and replay_from != "startpos":
            raise ValueError(
                "observation_mode='san_only' requires replay_from='startpos' "
                "(SAN without a FEN cannot fix an arbitrary midgame position)."
            )
        self.reply_constraint = (reply_constraint or "any").strip().lower()
        _legal_moves_for_reply_constraint(chess.Board(), self.reply_constraint)  # validate
        self.n_plies = n_plies
        self.min_plies = max(1, min_plies)
        self.rng = rng or random.Random()
        self.capture_bias = max(0.0, min(1.0, capture_bias))
        self.max_replay_attempts = max(1, max_replay_attempts)
        self.observation_mode = observation_mode
        self.replay_from = replay_from
        self.max_constrained_replies = max_constrained_replies
        self.omit_side_to_move = bool(omit_side_to_move)
        self.prepend_terminal_board_grid = bool(prepend_terminal_board_grid)

    def generate(self, env) -> ProbeResult:
        replay_root = chess.Board() if self.replay_from == "startpos" else env.board
        start_fen = replay_root.fen()
        final_board: chess.Board | None = None
        moves: list[str] = []
        attempts_used = 0

        for attempt in range(self.max_replay_attempts):
            attempts_used = attempt + 1
            if self.capture_bias > 0.0:
                fb, mv = _replay_random_capture_biased(
                    replay_root, self.n_plies, self.rng, self.capture_bias
                )
            else:
                fb, mv = _replay_random(replay_root, self.n_plies, self.rng)
            if len(mv) < self.min_plies or fb.is_game_over():
                continue
            legal_try = list(fb.legal_moves)
            if not legal_try:
                continue
            constrained_try = _legal_moves_for_reply_constraint(fb, self.reply_constraint)
            if not constrained_try:
                continue
            if (
                self.max_constrained_replies is not None
                and len(constrained_try) > self.max_constrained_replies
            ):
                continue
            final_board, moves = fb, mv
            break

        if final_board is None or not moves:
            return ProbeResult(
                probe_type="chess_san_legal_move",
                question="(skipped)",
                ground_truth=None,
                metadata={
                    "skipped": True,
                    "reason": "insufficient_moves_after_retries",
                    "n_moves": len(moves),
                    "attempts": attempts_used,
                },
            )

        legal = _legal_moves_for_reply_constraint(final_board, self.reply_constraint)
        legal_ucis = sorted({m.uci() for m in legal})
        start = chess.Board(start_fen)
        move_objs = [chess.Move.from_uci(u) for u in moves]
        try:
            san_line = start.variation_san(move_objs)
        except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
            return ProbeResult(
                probe_type="chess_san_legal_move",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "san_render_failed"},
            )

        side_line = (
            ""
            if (
                self.omit_side_to_move
                and self.replay_from == "startpos"
            )
            else f"It is now {_color_name(final_board.turn)}'s turn.\n"
        )
        fen_cfg = effective_fen_display(None)
        grid_prefix = ""
        if self.prepend_terminal_board_grid:
            grid_prefix = (
                render_full(
                    final_board,
                    include_legal_moves=False,
                    include_fen=True,
                    fen_display=fen_cfg,
                ).rstrip()
                + "\n\n"
            )

        if self.observation_mode == "san_only":
            if self.prepend_terminal_board_grid:
                observation = (
                    grid_prefix
                    + "SAN move list that reaches the diagram above (from the standard start):\n"
                    f"Moves played: {san_line}\n"
                    + side_line
                ).rstrip() + "\n"
            else:
                observation = (
                    "You are given a move sequence in standard algebraic notation (SAN) only, "
                    "starting from the usual initial position. There is no board diagram and no FEN; "
                    "reconstruct the position from the moves (including captures, promotions, and "
                    "castling as written in SAN).\n"
                    f"Moves played: {san_line}\n"
                    + side_line
                ).rstrip() + "\n"
        else:
            if self.prepend_terminal_board_grid:
                observation = (
                    grid_prefix
                    + "SAN fragment for the same terminal position as the diagram above.\n"
                    f"Starting FEN: {derive_display_fen(start_fen, fen_cfg)}\n"
                    f"Moves played: {san_line}\n"
                    + side_line
                ).rstrip() + "\n"
            else:
                observation = (
                    "You are given a chess fragment as standard algebraic notation (SAN) only. "
                    "There is no board diagram; you must infer the full position from the moves "
                    "(including captures, promotions, and castling as written in SAN).\n"
                    f"Starting FEN: {derive_display_fen(start_fen, fen_cfg)}\n"
                    f"Moves played: {san_line}\n"
                    + side_line
                ).rstrip() + "\n"
        if fen_cfg.mode not in ("truth", "none", "") and self.observation_mode == "san_only":
            ref = derive_display_fen(final_board.fen(), fen_cfg)
            observation = (
                observation.rstrip()
                + "\n\nReference FEN (secondary source; may disagree with the SAN line above): "
                f"{ref}\n"
            )
        _reply_suffix = {
            "any": "",
            "none": "",
            "non_capture": " The move must not capture any opposing piece.",
            "non_promotion": " The move must not be a pawn promotion.",
            "quiet": " The move must be quiet: no capture and it must not give check.",
            "quiet_np": (
                " The move must be quiet (no capture, no check) and must not be a pawn promotion."
            ),
            "quiet_non_promotion": (
                " The move must be quiet (no capture, no check) and must not be a pawn promotion."
            ),
        }
        question = (
            "From this position, reply with exactly one legal move for the side to move, "
            "in UCI format (e.g. g1f3 or e7e8q). One token only — no explanation."
            + _reply_suffix.get(self.reply_constraint, "")
        )
        return ProbeResult(
            probe_type="chess_san_legal_move",
            question=question,
            ground_truth=legal_ucis,
            metadata={
                "legal_ucis": legal_ucis,
                "n_plies": len(moves),
                "moves_uci": moves,
                "san_line": san_line,
                "capture_bias": self.capture_bias,
                "replay_attempts_used": attempts_used,
                "observation_mode": self.observation_mode,
                "replay_from": self.replay_from,
                "reply_constraint": self.reply_constraint,
                "max_constrained_replies": self.max_constrained_replies,
                "omit_side_to_move": self.omit_side_to_move,
                "start_fen": start_fen,
                "prepend_terminal_board_grid": self.prepend_terminal_board_grid,
            },
            observation_override=observation,
        )


class ChessConflictingPromptProbe(Probe):
    """Inject a false claim into the prompt; ground truth comes from the board.

    Tests whether the model anchors to V (the rendered board) when V conflicts
    with an authoritative-sounding assertion in the question text.

    ``lie_kind``:
      - ``piece`` — false piece label on a square (original behavior).
      - ``fen_transpose`` — inject a plausible wrong FEN (two non-king pieces swapped).
      - ``fen_startpos`` — inject the standard start FEN as if it were official.
      - ``fen_override`` — inject ``false_fen_override`` (must be set).
    """

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        lie_kind: str = "piece",
        false_fen_override: Optional[str] = None,
    ):
        if lie_kind not in ("piece", "fen_transpose", "fen_startpos", "fen_override"):
            raise ValueError(
                f"lie_kind must be piece, fen_transpose, fen_startpos, or fen_override, got {lie_kind!r}"
            )
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self.lie_kind = lie_kind
        self.false_fen_override = false_fen_override

    def generate(self, env) -> ProbeResult:
        board: chess.Board = env.board
        occupied = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
        if not occupied:
            return ProbeResult(
                probe_type="chess_conflicting_prompt",
                question="(skipped)",
                ground_truth=None,
                metadata={"skipped": True, "reason": "empty_board"},
            )

        # Pick a real occupied square and lie about it.
        target = self.rng.choice(occupied)
        actual = board.piece_at(target)
        assert actual is not None
        false_label = self.rng.choice(
            [lbl for lbl in _ALL_PIECE_LABELS if lbl != piece_name(actual)]
        )
        sq_name = chess.square_name(target)

        # Half the time we ask about the false claim (gt=False), half the time
        # about the true contents (gt=True), so over-trusting the prompt
        # produces clearly wrong answers in both directions.
        ask_about_lie = self.rng.random() < self.positive_rate
        if ask_about_lie:
            asked_label = false_label
            gt = False
        else:
            asked_label = piece_name(actual)
            gt = True

        true_fen = board.fen()
        bogus_fen: Optional[str] = None
        if self.lie_kind == "piece":
            prefix = f"A message claims there is a {false_label} on {sq_name}.\n"
        elif self.lie_kind == "fen_transpose":
            bogus_fen = derive_display_fen(
                true_fen,
                FenDisplayConfig(
                    mode="transpose",
                    override_fen=None,
                    seed=self.rng.randint(0, 2**31 - 1),
                ),
            )
            prefix = (
                "The tournament bulletin lists this position as:\n"
                f"FEN: {bogus_fen}\n\n"
                "Use the **rendered board diagram** as authoritative for the next question.\n"
            )
        elif self.lie_kind == "fen_startpos":
            bogus_fen = chess.STARTING_FEN
            prefix = (
                "The scoresheet header records this position as:\n"
                f"FEN: {bogus_fen}\n\n"
                "Use the **rendered board diagram** as authoritative for the next question.\n"
            )
        else:
            if not (self.false_fen_override and self.false_fen_override.strip()):
                return ProbeResult(
                    probe_type="chess_conflicting_prompt",
                    question="(skipped)",
                    ground_truth=None,
                    metadata={"skipped": True, "reason": "fen_override_missing"},
                )
            bogus_fen = self.false_fen_override.strip()
            prefix = (
                "An imported PGN comment claims the position is:\n"
                f"FEN: {bogus_fen}\n\n"
                "Use the **rendered board diagram** as authoritative for the next question.\n"
            )

        question = (
            prefix + f"Is there a {asked_label} on {sq_name}?\nAnswer with exactly 'yes' or 'no'."
        )
        meta = {
            "square": sq_name,
            "asked_label": asked_label,
            "actual_piece": piece_name(actual),
            "false_label": false_label,
            "lie_kind": self.lie_kind,
            "true_fen": true_fen,
        }
        if bogus_fen is not None:
            meta["bogus_fen"] = bogus_fen
        return ProbeResult(
            probe_type="chess_conflicting_prompt",
            question=question,
            ground_truth=gt,
            metadata=meta,
        )
