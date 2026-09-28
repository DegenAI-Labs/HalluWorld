"""Lichess-style atomic chess via python-chess :class:`chess.variant.AtomicBoard`.

Blast rules, king connectivity, and legality match the implementation shipped with
``python-chess`` (same variant engines like Lichess use for atomic).
"""

from __future__ import annotations

import chess
from chess.variant import AtomicBoard as _ChessAtomicBoard


class AtomicBoard(_ChessAtomicBoard):
    """Thin subclass so ``isinstance(board, AtomicBoard)`` is stable for HalluWorld envs/probes."""

    halluworld_variant = "atomic"


def blast_ring_mask(center: chess.Square) -> chess.Bitboard:
    """Capture square plus king-neighborhood (Lichess atomic uses this ring for non-pawn blast pieces)."""
    return chess.BB_SQUARES[center] | chess.BB_KING_ATTACKS[center]
