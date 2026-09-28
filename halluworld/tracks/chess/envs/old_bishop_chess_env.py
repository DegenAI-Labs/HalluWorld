from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import chess

from halluworld.tracks.chess.rules.old_bishop_board import OldBishopBoard
from halluworld.tracks.chess.envs.chess_env import DEFAULT_CHESS_BENCHMARK_FENS


class _NoOpActionSpace:
    def sample(self) -> None:
        return None


_OLD_BISHOP_EXTRA_FENS = [
    "8/8/8/3B4/8/8/8/4K3 w - - 0 1",
    "8/8/4b3/8/3B4/8/8/4K3 w - - 0 1",
    "8/5k2/8/3B4/8/8/8/4K3 w - - 0 1",
    "8/8/8/8/8/8/8/Q3K3 w - - 0 1",
]


@dataclass
class OldBishopChessEnv:
    fens: Optional[list[str]] = None
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.fens = self.fens or (list(DEFAULT_CHESS_BENCHMARK_FENS) + _OLD_BISHOP_EXTRA_FENS)
        self.board = OldBishopBoard()
        self.current_fen = chess.STARTING_FEN
        self.action_space = _NoOpActionSpace()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
        self.current_fen = self._rng.choice(self.fens)
        self.board = OldBishopBoard(self.current_fen)
        return {"fen": self.board.fen()}

    def step(self, _action):
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            self.board.push(self._rng.choice(legal_moves))
        terminated = self.board.is_game_over()
        truncated = False
        return {"fen": self.board.fen()}, 0.0, terminated, truncated, {}


def make_old_bishop_chess_env(
    fens: Optional[list[str]] = None,
    seed: int = 42,
) -> OldBishopChessEnv:
    return OldBishopChessEnv(fens=fens, seed=seed)
