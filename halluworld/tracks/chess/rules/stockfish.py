from __future__ import annotations

import os
import platform
import shutil
from typing import Optional

import chess
import chess.engine


def find_stockfish_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """Return a Stockfish binary path if available, else None."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path

    system = platform.system().lower()
    candidates = {
        "darwin": ["/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"],
        "linux": ["/usr/local/bin/stockfish", "/usr/bin/stockfish", "/usr/games/stockfish"],
        "windows": [
            r"C:\Program Files\Stockfish\stockfish.exe",
            "stockfish.exe",
        ],
    }
    for path in candidates.get(system, []):
        if os.path.isfile(path):
            return path

    which_path = shutil.which("stockfish")
    return which_path


class StockfishHelper:
    """Small helper for deterministic Stockfish best-move lookups."""

    def __init__(
        self,
        stockfish_path: Optional[str] = None,
        depth: int = 12,
        threads: int = 1,
        hash_mb: int = 64,
    ):
        self.depth = depth
        self.stockfish_path = find_stockfish_binary(stockfish_path)
        if not self.stockfish_path:
            raise RuntimeError(
                "Stockfish binary not found. Install Stockfish or set STOCKFISH_PATH."
            )

        self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self._engine.configure({"Threads": threads, "Hash": hash_mb})

    def best_move_uci(self, board: chess.Board) -> str:
        result = self._engine.play(board, chess.engine.Limit(depth=self.depth))
        if result.move is None:
            raise RuntimeError("Stockfish did not return a move.")
        return result.move.uci()

    def close(self) -> None:
        if hasattr(self, "_engine") and self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None

    def __enter__(self) -> "StockfishHelper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
