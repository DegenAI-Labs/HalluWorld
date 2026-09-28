"""Short-range-bishop variant probes — tuned for adversarial failure modes."""

from __future__ import annotations

import random
from typing import Optional

import chess
from chess import square_file, square_rank

from halluworld.tracks.chess.rules.old_bishop_board import OldBishopBoard, old_bishop_attack_mask
from halluworld.probe import Probe, ProbeResult

_FULL_RULES = (
    "Variant patch (all other rules are standard FIDE): each bishop slides along diagonals "
    "but may travel at most two squares along one diagonal direction from its starting square "
    "on that move (blocking and captures stop the ray as usual). Queens are not limited this way."
)

_MINIMAL_RULES = (
    "Patch only: bishops ≤2 squares per diagonal step-chain; queens unchanged."
)

_FIDE_RED_HERRING = (
    "Reminder text (incorrect for this benchmark): under classical FIDE, bishops slide any "
    "distance along a diagonal.\n"
    ">>> Ignore that reminder. The ONLY operative rule change is the bishop patch stated next.\n\n"
)


def _rule_block(rng: random.Random, *, minimal_rules_rate: float) -> str:
    if rng.random() < minimal_rules_rate:
        return _MINIMAL_RULES
    return _FULL_RULES


def _maybe_fake_fide(rng: random.Random, *, fake_fide_rate: float) -> str:
    if rng.random() < fake_fide_rate:
        return _FIDE_RED_HERRING
    return ""


def _sample_long_diagonal_bishop_uci(board: OldBishopBoard, rng: random.Random) -> Optional[str]:
    std = chess.Board(board.fen())
    side = board.turn
    bishops = [
        sq
        for sq in chess.SQUARES
        if (p := board.piece_at(sq))
        and p.piece_type == chess.BISHOP
        and p.color == side
    ]
    rng.shuffle(bishops)
    for from_sq in bishops:
        r0, f0 = square_rank(from_sq), square_file(from_sq)
        for dr, df in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            for dist in range(3, 8):
                r, f = r0 + dr * dist, f0 + df * dist
                if not (0 <= r < 8 and 0 <= f < 8):
                    break
                to_sq = chess.square(f, r)
                if board.color_at(to_sq) == side:
                    break
                mid = chess.between(from_sq, to_sq)
                blocked = False
                for s in chess.scan_reversed(mid):
                    if board.piece_at(s) is not None:
                        blocked = True
                        break
                if blocked:
                    break
                mv = chess.Move(from_sq, to_sq)
                if std.is_pseudo_legal(mv) and not board.is_pseudo_legal(mv):
                    return mv.uci()
    return None


def _sample_long_queen_uci(board: OldBishopBoard, rng: random.Random) -> Optional[str]:
    moves = list(board.legal_moves)
    rng.shuffle(moves)
    for mv in moves:
        if board.piece_type_at(mv.from_square) != chess.QUEEN:
            continue
        if mv.promotion:
            continue
        r0, f0 = square_rank(mv.from_square), square_file(mv.from_square)
        r1, f1 = square_rank(mv.to_square), square_file(mv.to_square)
        dr, df = abs(r1 - r0), abs(f1 - f0)
        if dr == 0 and df == 0:
            continue
        if dr != 0 and df != 0 and dr != df:
            continue
        if max(dr, df) < 3:
            continue
        return mv.uci()
    return None


_RULES_QUIZ = [
    (
        "In this variant, can a bishop move three vacant squares along a single diagonal in one bishop move?",
        False,
    ),
    (
        "In this variant, do queens follow the same two-square diagonal cap as bishops?",
        False,
    ),
    (
        "In this variant, can a knight still move in its usual L-shape?",
        True,
    ),
    (
        "In this variant, may a bishop capture an enemy piece on the second square of a diagonal slide if the first square is empty?",
        True,
    ),
]


class OldBishopLegalMoveProbe(Probe):
    def __init__(
        self,
        legal_rate: float = 0.35,
        negate_rate: float = 0.45,
        fake_fide_rate: float = 0.5,
        minimal_rules_rate: float = 0.35,
        rng: Optional[random.Random] = None,
    ):
        self.legal_rate = legal_rate
        self.negate_rate = negate_rate
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_move_legality",
                question="(misconfigured env — need OldBishopBoard)",
                ground_truth=None,
                metadata={"skipped": True},
            )

        legal_moves = [m.uci() for m in board.legal_moves]
        use_legal = bool(legal_moves) and self.rng.random() < self.legal_rate
        if use_legal:
            move_uci = self.rng.choice(legal_moves)
            base_gt = True
            source = "legal"
        else:
            foil = _sample_long_diagonal_bishop_uci(board, self.rng)
            if foil is not None and foil not in legal_moves:
                move_uci, base_gt, source = foil, False, "long_bishop_slide_foil"
            else:
                move_uci = self._sample_illegal_uci(board, legal_moves)
                base_gt = False
                source = "illegal_foil"

        rules = _maybe_fake_fide(self.rng, fake_fide_rate=self.fake_fide_rate) + _rule_block(
            self.rng, minimal_rules_rate=self.minimal_rules_rate
        )
        negate = self.rng.random() < self.negate_rate
        if negate:
            q = (
                rules
                + "\nAnswer **yes** if and only if the following statement is **false**; otherwise answer **no**.\n"
                + f"Statement: The move {move_uci} is legal for the side to move under the variant rules.\n"
                + "Respond with exactly `yes` or exactly `no` (no other words)."
            )
            gt = not base_gt
        else:
            q = (
                rules
                + f"\nIs the move {move_uci} legal for the side to move under the variant rules?\n"
                + "Respond with exactly `yes` or exactly `no` (no other words)."
            )
            gt = base_gt

        return ProbeResult(
            probe_type="chess_old_bishop_move_legality",
            question=q,
            ground_truth=gt,
            metadata={
                "move_uci": move_uci,
                "source": source,
                "negated_prompt": negate,
                "base_legal": base_gt,
            },
        )

    def _sample_illegal_uci(self, board: OldBishopBoard, legal_moves: list[str]) -> str:
        for _ in range(200):
            a, b = self.rng.choice(chess.SQUARES), self.rng.choice(chess.SQUARES)
            if a == b:
                continue
            c = chess.Move(a, b).uci()
            if c not in legal_moves:
                return c
        return "a1h8" if "a1h8" not in legal_moves else "a1a2"


class OldBishopBishopAttackSquareProbe(Probe):
    def __init__(
        self,
        positive_rate: float = 0.25,
        negate_rate: float = 0.4,
        fake_fide_rate: float = 0.45,
        minimal_rules_rate: float = 0.35,
        rng: Optional[random.Random] = None,
    ):
        self.positive_rate = positive_rate
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.negate_rate = negate_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_bishop_attacks",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )

        bishops = [
            sq
            for sq in chess.SQUARES
            if (p := board.piece_at(sq)) and p.piece_type == chess.BISHOP
        ]
        if not bishops:
            return ProbeResult(
                probe_type="chess_old_bishop_bishop_attacks",
                question="no bishop",
                ground_truth=None,
                metadata={"skipped": True},
            )

        from_sq = self.rng.choice(bishops)
        piece = board.piece_at(from_sq)
        assert piece is not None
        occ = board.occupied
        reach = old_bishop_attack_mask(from_sq, occ)

        def _attack_dests(b: chess.Board, src: chess.Square) -> set[chess.Square]:
            mask = b.attacks_mask(src)
            return {
                sq
                for sq in chess.SQUARES
                if (chess.BB_SQUARES[sq] & mask) and b.color_at(sq) != piece.color
            }

        old_dests = _attack_dests(board, from_sq)
        std = chess.Board(board.fen())
        std_dests = _attack_dests(std, from_sq)
        orthodox_only = sorted(std_dests - old_dests, key=lambda s: s)

        if old_dests and self.rng.random() < self.positive_rate:
            to_sq = self.rng.choice(sorted(old_dests))
            base_gt = True
            source = "variant_reach"
        elif orthodox_only:
            to_sq = self.rng.choice(orthodox_only)
            base_gt = False
            source = "orthodox_long_diagonal"
        elif old_dests:
            to_sq = self.rng.choice(sorted(old_dests))
            base_gt = True
            source = "forced_positive"
        else:
            pool = [sq for sq in chess.SQUARES if sq != from_sq and sq not in std_dests]
            to_sq = self.rng.choice(pool) if pool else from_sq
            base_gt = False
            source = "offboard_foil"

        a, b = chess.square_name(from_sq), chess.square_name(to_sq)
        rules = _maybe_fake_fide(self.rng, fake_fide_rate=self.fake_fide_rate) + _rule_block(
            self.rng, minimal_rules_rate=self.minimal_rules_rate
        )
        stmt = (
            f"The bishop on {a} attacks square {b} in one bishop move under the variant "
            f"(i.e., could capture on {b} from {a} in one bishop move right now)."
        )
        negate = self.rng.random() < self.negate_rate
        if negate:
            q = (
                rules
                + "\nAnswer **yes** if and only if the following statement is **false**; otherwise answer **no**.\n"
                + f"Statement: {stmt}\n"
                + "Respond with exactly `yes` or exactly `no` (no other words)."
            )
            gt = not base_gt
        else:
            q = (
                rules + "\n" + stmt + "\nRespond with exactly `yes` or exactly `no` (no other words)."
            )
            gt = base_gt

        return ProbeResult(
            probe_type="chess_old_bishop_bishop_attacks",
            question=q,
            ground_truth=gt,
            metadata={"from": a, "to": b, "source": source, "negated_prompt": negate},
        )


class OldBishopQueenLongSlideLegalProbe(Probe):
    """Trap: queens are not bishop-capped; long queen slides must still be legal if path clear."""

    def __init__(
        self,
        fake_fide_rate: float = 0.35,
        minimal_rules_rate: float = 0.25,
        rng: Optional[random.Random] = None,
    ):
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_queen_long_slide",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        uci = _sample_long_queen_uci(board, self.rng)
        fallback = False
        if uci is None:
            fb = OldBishopBoard("8/8/8/8/8/8/8/Q3K3 w - - 0 1")
            uci = _sample_long_queen_uci(fb, self.rng)
            fallback = True
        if uci is None:
            return ProbeResult(
                probe_type="chess_old_bishop_queen_long_slide",
                question="(internal generator error)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        mv = chess.Move.from_uci(uci)
        eval_board = OldBishopBoard("8/8/8/8/8/8/8/Q3K3 w - - 0 1") if fallback else board
        gt = eval_board.is_legal(mv)
        rules = _maybe_fake_fide(self.rng, fake_fide_rate=self.fake_fide_rate) + _rule_block(
            self.rng, minimal_rules_rate=self.minimal_rules_rate
        )
        pos_note = (
            "The diagram below is the position for this question (it may differ from the warm-up).\n"
            if fallback
            else ""
        )
        from halluworld.tracks.chess.serializers import board_to_grid_text

        grid = board_to_grid_text(eval_board)
        q = (
            rules
            + ("\n" + pos_note if pos_note else "\n")
            + "Board:\n"
            + grid
            + f"\nConsider the **queen** move {uci} for the side to move in the diagram above. "
            "Is that move **legal** under the variant rules "
            "(bishop cap does **not** apply to queens)?\n"
            "Respond with exactly `yes` or exactly `no` (no other words)."
        )
        return ProbeResult(
            probe_type="chess_old_bishop_queen_long_slide",
            question=q,
            ground_truth=gt,
            metadata={"uci": uci, "fallback_board": fallback},
        )


class OldBishopRulesFactProbe(Probe):
    """Static rule questions — forces careful reading of bishop vs queen scope."""

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_rules_fact",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        text, gt = self.rng.choice(_RULES_QUIZ)
        rules = _maybe_fake_fide(self.rng, fake_fide_rate=0.55) + _FULL_RULES
        q = (
            rules
            + "\nTrue/false question about the variant only (not about the diagram):\n"
            + text
            + "\nAnswer `yes` if true, `no` if false. Exactly one token."
        )
        return ProbeResult(
            probe_type="chess_old_bishop_rules_fact",
            question=q,
            ground_truth=gt,
            metadata={"quiz_id": text[:48]},
        )


def _uci_preamble(
    rng: random.Random,
    *,
    fake_fide_rate: float,
    minimal_rules_rate: float,
) -> str:
    return _maybe_fake_fide(rng, fake_fide_rate=fake_fide_rate) + _rule_block(
        rng, minimal_rules_rate=minimal_rules_rate
    )


class OldBishopLegalAnyUciProbe(Probe):
    """Reply with any one legal UCI for the side to move (variant board)."""

    def __init__(
        self,
        fake_fide_rate: float = 0.45,
        minimal_rules_rate: float = 0.35,
        rng: Optional[random.Random] = None,
    ):
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_legal_any_uci",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        legal = sorted({m.uci() for m in board.legal_moves})
        if not legal:
            return ProbeResult(
                probe_type="chess_old_bishop_legal_any_uci",
                question="(no legal moves)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        rules = _uci_preamble(
            self.rng,
            fake_fide_rate=self.fake_fide_rate,
            minimal_rules_rate=self.minimal_rules_rate,
        )
        q = (
            rules
            + "\nFrom the **Current game** diagram, reply with **one** legal move for the side to move, "
            "as a **single UCI token** (e.g. `e2e4` or `e7e8q`). Do not use SAN. Do not add commentary.\n"
            "Your entire answer must be parseable as that one token."
        )
        return ProbeResult(
            probe_type="chess_old_bishop_legal_any_uci",
            question=q,
            ground_truth=legal,
            metadata={"legal_ucis": legal},
        )


class OldBishopLegalBishopUciProbe(Probe):
    """Legal move that starts by moving a bishop (from-square is a bishop)."""

    def __init__(
        self,
        fake_fide_rate: float = 0.45,
        minimal_rules_rate: float = 0.35,
        rng: Optional[random.Random] = None,
    ):
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_legal_bishop_uci",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        legal = sorted(
            {
                m.uci()
                for m in board.legal_moves
                if board.piece_type_at(m.from_square) == chess.BISHOP
            }
        )
        if not legal:
            return ProbeResult(
                probe_type="chess_old_bishop_legal_bishop_uci",
                question="(no bishop moves)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        rules = _uci_preamble(
            self.rng,
            fake_fide_rate=self.fake_fide_rate,
            minimal_rules_rate=self.minimal_rules_rate,
        )
        q = (
            rules
            + "\nFrom the **Current game** diagram, reply with **one** legal move for the side to move "
            "that **starts on a bishop** (the piece you move first must be a bishop), as a **single UCI token**. "
            "No SAN, no commentary."
        )
        return ProbeResult(
            probe_type="chess_old_bishop_legal_bishop_uci",
            question=q,
            ground_truth=legal,
            metadata={"legal_ucis": legal},
        )


class OldBishopLegalQuietUciProbe(Probe):
    """Legal non-capture, non-castling UCI (quiet in the usual sense)."""

    def __init__(
        self,
        fake_fide_rate: float = 0.45,
        minimal_rules_rate: float = 0.35,
        rng: Optional[random.Random] = None,
    ):
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_legal_quiet_uci",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        legal = sorted(
            {
                m.uci()
                for m in board.legal_moves
                if not board.is_capture(m) and not board.is_castling(m)
            }
        )
        if not legal:
            return ProbeResult(
                probe_type="chess_old_bishop_legal_quiet_uci",
                question="(no quiet moves)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        rules = _uci_preamble(
            self.rng,
            fake_fide_rate=self.fake_fide_rate,
            minimal_rules_rate=self.minimal_rules_rate,
        )
        q = (
            rules
            + "\nFrom the **Current game** diagram, reply with **one** legal **quiet** move for the side to move: "
            "not a capture and not castling, as a **single UCI token**. No SAN, no commentary."
        )
        return ProbeResult(
            probe_type="chess_old_bishop_legal_quiet_uci",
            question=q,
            ground_truth=legal,
            metadata={"legal_ucis": legal},
        )


def _fide_legal_variant_illegal_bishop_ucis(board: OldBishopBoard) -> list[str]:
    std = chess.Board(board.fen())
    out: list[str] = []
    for mv in std.legal_moves:
        if board.piece_type_at(mv.from_square) != chess.BISHOP:
            continue
        if std.is_legal(mv) and not board.is_legal(mv):
            out.append(mv.uci())
    return sorted(set(out))


class OldBishopFideIllegalBishopUciProbe(Probe):
    """UCI that is FIDE-legal but illegal under the short bishop — trips orthodox intuition."""

    def __init__(
        self,
        fake_fide_rate: float = 0.25,
        minimal_rules_rate: float = 0.25,
        rng: Optional[random.Random] = None,
    ):
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, OldBishopBoard):
            return ProbeResult(
                probe_type="chess_old_bishop_fide_illegal_bishop_uci",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        foils = _fide_legal_variant_illegal_bishop_ucis(board)
        if not foils:
            return ProbeResult(
                probe_type="chess_old_bishop_fide_illegal_bishop_uci",
                question="(no FIDE-only bishop slides in this position)",
                ground_truth=None,
                metadata={"skipped": True},
            )
        rules = _uci_preamble(
            self.rng,
            fake_fide_rate=self.fake_fide_rate,
            minimal_rules_rate=self.minimal_rules_rate,
        )
        q = (
            rules
            + "\nFrom the **Current game** diagram, reply with **one** UCI token for a **bishop move** that is "
            "**legal under normal FIDE rules** from this position but **illegal** under the short-bishop variant "
            "(bishops capped at two squares along a diagonal per move). "
            "Only the UCI token — no SAN, no explanation."
        )
        return ProbeResult(
            probe_type="chess_old_bishop_fide_illegal_bishop_uci",
            question=q,
            ground_truth=foils,
            metadata={"legal_ucis": foils, "source": "fide_legal_variant_illegal_bishop"},
        )
