from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import chess

from halluworld.tracks.chess.rules.atomic_board import AtomicBoard
from halluworld.tracks.chess.envs.chess_env import DEFAULT_CHESS_BENCHMARK_FENS


class _NoOpActionSpace:
    def sample(self) -> None:
        return None


# Extra positions beyond ``DEFAULT_CHESS_BENCHMARK_FENS`` (6 FENs) — four extras here ⇒ **10** builtin FENs
# unless you pass ``fens=`` or load from Hugging Face (``ATOMIC_USE_HF_FENS`` in ``chess_full_probes``).
_ATOMIC_EXTRA_FENS = [
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "8/8/8/3q4/4P3/8/8/4K3 w - - 0 1",
    "8/8/8/3n4/3P4/8/8/4K3 w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 4",
]


@dataclass
class AtomicChessEnv:
    fens: Optional[list[str]] = None
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.fens = self.fens or (list(DEFAULT_CHESS_BENCHMARK_FENS) + _ATOMIC_EXTRA_FENS)
        self.board = AtomicBoard()
        self.current_fen = chess.STARTING_FEN
        self.action_space = _NoOpActionSpace()
        self._episode_root_fen: str = chess.STARTING_FEN
        self._episode_moves_uci: list[str] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
        self.current_fen = self._rng.choice(self.fens)
        self.board = AtomicBoard(self.current_fen)
        self._episode_root_fen = self.board.fen()
        self._episode_moves_uci = []
        return {"fen": self.board.fen()}

    def step(self, _action):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            mv = self._rng.choice(legal_moves)
            self._episode_moves_uci.append(mv.uci())
            self.board.push(mv)
        terminated = self.board.is_game_over()
        truncated = False
        return {"fen": self.board.fen()}, 0.0, terminated, truncated, {}


def make_atomic_chess_env(
    fens: Optional[list[str]] = None,
    seed: int = 42,
) -> AtomicChessEnv:
    return AtomicChessEnv(fens=fens, seed=seed)
