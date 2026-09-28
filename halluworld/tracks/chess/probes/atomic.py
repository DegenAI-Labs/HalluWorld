"""Atomic chess — UCI-named hypotheticals (Lichess atomic rules)."""

from __future__ import annotations

import random
from typing import Optional

import chess

from halluworld.tracks.chess.rules.atomic_board import AtomicBoard
from halluworld.probe import Probe, ProbeResult
from halluworld.tracks.chess.serializers import piece_name

_ATOMIC_FULL = (
    "Rules: Lichess **atomic** chess. Captures cause explosions: the capturing piece is removed, "
    "and non-pawn pieces on adjacent squares to the capture square are removed; pawns in the blast ring "
    "are not removed by the blast. Kings may not be in blast range in the usual atomic way; "
    "castling and checks follow that variant."
)

_ATOMIC_MINIMAL = (
    "Atomic (Lichess): captures explode adjacent non-pawn pieces; special king connectivity rules."
)

_FIDE_RED_HERRING_ATOMIC = (
    "Reminder (incorrect here): under classical FIDE, captures do not remove neighboring pieces.\n"
    ">>> Ignore that. Use **atomic** rules for the hypothetical.\n\n"
)


def _atomic_rule_block(rng: random.Random, *, minimal_rules_rate: float) -> str:
    if rng.random() < minimal_rules_rate:
        return _ATOMIC_MINIMAL
    return _ATOMIC_FULL


def _maybe_fake_fide_atomic(rng: random.Random, *, fake_fide_rate: float) -> str:
    if rng.random() < fake_fide_rate:
        return _FIDE_RED_HERRING_ATOMIC
    return ""


def _atomic_question_preamble(
    rng: random.Random,
    *,
    fake_fide_rate: float,
    minimal_rules_rate: float,
) -> str:
    return _maybe_fake_fide_atomic(rng, fake_fide_rate=fake_fide_rate) + _atomic_rule_block(
        rng, minimal_rules_rate=minimal_rules_rate
    )


def _uci_chain_with_san_labels(board: AtomicBoard, chain: list[chess.Move]) -> str:
    """Human-readable ``UCI (SAN)`` fragments for a move sequence from *board*'s position."""
    b = board.copy(stack=False)
    parts: list[str] = []
    for mv in chain:
        san = b.san(mv)
        parts.append(f"`{mv.uci()}` ({san})")
        b.push(mv)
    return ", ".join(parts)


class AtomicHypotheticalPieceAfterUciProbe(Probe):
    """After one or more legal UCIs in order, ask which piece (or empty) occupies a chosen square."""

    def __init__(
        self,
        fake_fide_rate: float = 0.45,
        minimal_rules_rate: float = 0.35,
        rng: Optional[random.Random] = None,
        max_move_tries: int = 48,
        hypothesis_plies_min: int = 1,
        hypothesis_plies_max: int = 3,
    ):
        self.fake_fide_rate = fake_fide_rate
        self.minimal_rules_rate = minimal_rules_rate
        self.rng = rng or random.Random()
        self.max_move_tries = max_move_tries
        lo = min(hypothesis_plies_min, hypothesis_plies_max)
        hi = max(hypothesis_plies_min, hypothesis_plies_max)
        self.hypothesis_plies_lo = max(1, lo)
        self.hypothesis_plies_hi = max(self.hypothesis_plies_lo, hi)

    def generate(self, env) -> ProbeResult:
        board = env.board
        if not isinstance(board, AtomicBoard):
            return ProbeResult(
                probe_type="chess_atomic_hypothetical_piece_after_uci",
                question="(misconfigured env)",
                ground_truth=None,
                metadata={"skipped": True},
            )

        legal = list(board.legal_moves)
        if not legal:
            return ProbeResult(
                probe_type="chess_atomic_hypothetical_piece_after_uci",
                question="(no legal moves)",
                ground_truth=None,
                metadata={"skipped": True},
            )

        self.rng.shuffle(legal)
        rules = _atomic_question_preamble(
            self.rng,
            fake_fide_rate=self.fake_fide_rate,
            minimal_rules_rate=self.minimal_rules_rate,
        )

        for first in legal[: self.max_move_tries]:
            target_len = self.rng.randint(self.hypothesis_plies_lo, self.hypothesis_plies_hi)
            chain: list[chess.Move] = [first]
            b_chain = board.copy(stack=False)
            b_chain.push(first)
            while len(chain) < target_len:
                nxt = list(b_chain.legal_moves)
                if not nxt:
                    break
                mv = self.rng.choice(nxt)
                chain.append(mv)
                b_chain.push(mv)
            if len(chain) < self.hypothesis_plies_lo:
                continue

            b_final = board.copy(stack=False)
            for mv in chain:
                b_final.push(mv)

            b_pre_last = board.copy(stack=False)
            for mv in chain[:-1]:
                b_pre_last.push(mv)
            last = chain[-1]
            candidates = {last.from_square, last.to_square}
            if b_pre_last.is_capture(last):
                candidates.update(chess.scan_forward(chess.BB_KING_ATTACKS[last.to_square]))
            ask_sq = self.rng.choice(list(candidates))
            sym = chess.square_name(ask_sq)
            p = b_final.piece_at(ask_sq)
            gt = piece_name(p) if p else "empty"

            move_desc = _uci_chain_with_san_labels(board, chain)
            n_moves = len(chain)
            if n_moves == 1:
                hypo_line = (
                    f"\nHypothetical: from the **Current game** position, suppose this move is played: "
                    f"{move_desc}.\n"
                )
            else:
                hypo_line = (
                    f"\nHypothetical: from the **Current game** position, suppose these moves are played "
                    f"in order ({n_moves} plies): {move_desc}.\n"
                )
            q = (
                rules
                + hypo_line
                + f"What occupies square **{sym}** after the entire sequence (under atomic rules)?\n"
                + "Reply with **one** English phrase from this closed list only: "
                "empty, white pawn, white knight, white bishop, white rook, white queen, white king, "
                "black pawn, black knight, black bishop, black rook, black queen, black king.\n"
                "No SAN, no UCI, no extra words."
            )
            return ProbeResult(
                probe_type="chess_atomic_hypothetical_piece_after_uci",
                question=q,
                ground_truth=gt,
                metadata={
                    "hypothesis_ucis": [m.uci() for m in chain],
                    "hypothesis_plies": n_moves,
                    "target_square": sym,
                    "last_was_capture": b_pre_last.is_capture(last),
                },
            )

        return ProbeResult(
            probe_type="chess_atomic_hypothetical_piece_after_uci",
            question="(could not sample hypothetical)",
            ground_truth=None,
            metadata={"skipped": True},
        )
