from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import chess


_DEFAULT_FENS = [
    chess.STARTING_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/2P2N2/PP2PPPP/RNBQKB1R b KQkq - 0 3",
    "r3k2r/pppq1ppp/2n1bn2/3p4/3P4/2N1PN2/PPQ2PPP/R3KB1R w KQkq - 2 10",
    "2r2rk1/pp2qppp/2n1bn2/2bp4/3P4/2P1PN2/PP1NBPPP/R2Q1RK1 w - - 3 11",
    "r1bq1rk1/pppp1ppp/2n2n2/4p3/1b1PP3/2N2N2/PPP2PPP/R1BQKB1R w KQ - 4 6",
    "r2q1rk1/ppp2ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
]

DEFAULT_CHESS_BENCHMARK_FENS: list[str] = list(_DEFAULT_FENS)


class _NoOpActionSpace:
    def sample(self) -> None:
        return None


@dataclass
class ChessEnv:
    """Minimal benchmark environment around python-chess board state."""

    fens: Optional[list[str]] = None
    seed: int = 42

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.fens = self.fens or list(_DEFAULT_FENS)
        self.board = chess.Board()
        self.current_fen = chess.STARTING_FEN
        self.action_space = _NoOpActionSpace()

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)
        self.current_fen = self._rng.choice(self.fens)
        self.board = chess.Board(self.current_fen)
        return {"fen": self.board.fen()}

    def step(self, _action):
        """Warm-up transition: apply a random legal move if possible."""
        legal_moves = list(self.board.legal_moves)
        if legal_moves:
            self.board.push(self._rng.choice(legal_moves))
        terminated = self.board.is_game_over()
        truncated = False
        return {"fen": self.board.fen()}, 0.0, terminated, truncated, {}


def make_chess_env(
    fens: Optional[list[str]] = None,
    seed: int = 42,
) -> ChessEnv:
    return ChessEnv(fens=fens, seed=seed)


# --------------------------------------------------------------------------- #
# Lichess puzzle loader with on-disk caching                                  #
# --------------------------------------------------------------------------- #


def _default_cache_dir() -> Path:
    base = os.environ.get("HALLUWORLD_CACHE")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".cache" / "halluworld"


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _coerce_record(row: dict) -> Optional[dict]:
    """Normalize one Lichess puzzle row to a stable schema; return None if invalid."""
    fen = row.get("FEN") or row.get("fen")
    if not fen:
        return None
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    if board.is_game_over():
        return None

    moves = row.get("Moves") or row.get("moves") or ""
    if isinstance(moves, list):
        moves_list = [str(m) for m in moves]
    else:
        moves_list = [m for m in str(moves).split() if m]

    themes = row.get("Themes") or row.get("themes") or ""
    if isinstance(themes, list):
        themes_list = [str(t) for t in themes]
    else:
        themes_list = [t for t in str(themes).split() if t]

    rating = row.get("Rating") or row.get("rating") or row.get("PuzzleRating")
    try:
        rating_int = int(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating_int = None

    return {
        "fen": board.fen(),
        "moves": moves_list,
        "themes": themes_list,
        "rating": rating_int,
        "puzzle_id": row.get("PuzzleId") or row.get("puzzle_id"),
    }


def load_lichess_puzzles(
    num_samples: int = 100,
    seed: int = 42,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    themes: Optional[list[str]] = None,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> list[dict]:
    """Sample puzzle records from Hugging Face ``Lichess/chess-puzzles``.

    Returns a list of dicts with keys ``fen``, ``moves``, ``themes``, ``rating``,
    ``puzzle_id``. Results are cached on disk keyed by the call signature so
    subsequent identical calls skip the (slow) download + filter pass.

    Set ``HALLUWORLD_CACHE`` to override the default cache root
    (``~/.cache/halluworld``). Pass ``force_refresh=True`` to bypass the cache.
    """
    cache_root = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)

    key = _cache_key({
        "n": num_samples,
        "seed": seed,
        "min": min_rating,
        "max": max_rating,
        "themes": sorted(themes) if themes else None,
        "schema": 2,
    })
    cache_path = cache_root / f"lichess_puzzles_{key}.json"

    if cache_path.exists() and not force_refresh:
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass  # fall through and rebuild

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Loading Lichess puzzles requires `datasets`. Install with: pip install datasets"
        ) from exc

    ds = load_dataset("Lichess/chess-puzzles", split="train")

    rating_col = None
    for name in ("Rating", "rating", "PuzzleRating", "puzzle_rating"):
        if name in ds.column_names:
            rating_col = name
            break

    if rating_col and (min_rating is not None or max_rating is not None):
        lo = -(10**9) if min_rating is None else min_rating
        hi = 10**9 if max_rating is None else max_rating
        ds = ds.filter(lambda x: lo <= int(x[rating_col]) <= hi)

    if themes:
        themes_col = None
        for name in ("Themes", "themes"):
            if name in ds.column_names:
                themes_col = name
                break
        if themes_col is not None:
            wanted = set(themes)

            def _has_theme(row: dict) -> bool:
                raw = row.get(themes_col, "")
                tags = raw if isinstance(raw, list) else str(raw).split()
                return any(t in wanted for t in tags)

            ds = ds.filter(_has_theme)

    buffer_size = max(num_samples * 3, num_samples)
    ds = ds.shuffle(seed=seed).select(range(min(len(ds), buffer_size)))

    out: list[dict] = []
    for row in ds:
        rec = _coerce_record(dict(row))
        if rec is None:
            continue
        out.append(rec)
        if len(out) >= num_samples:
            break

    try:
        cache_path.write_text(json.dumps(out))
    except OSError:
        pass  # caching is best-effort

    return out


def load_lichess_puzzle_fens(
    num_samples: int = 100,
    seed: int = 42,
    min_rating: Optional[int] = None,
    max_rating: Optional[int] = None,
    themes: Optional[list[str]] = None,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> list[str]:
    """Sample valid FENs from Lichess puzzles. Thin wrapper over
    :func:`load_lichess_puzzles` for backwards compatibility."""
    records = load_lichess_puzzles(
        num_samples=num_samples,
        seed=seed,
        min_rating=min_rating,
        max_rating=max_rating,
        themes=themes,
        cache_dir=cache_dir,
        force_refresh=force_refresh,
    )
    return [r["fen"] for r in records]
