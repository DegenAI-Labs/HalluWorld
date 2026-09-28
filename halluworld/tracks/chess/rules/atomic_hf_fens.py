"""Sample midgame FENs from `Lichess/atomic-chess-games` on Hugging Face (streaming).

Requires the optional ``datasets`` package. Movetext is replayed under python-chess
:class:`chess.variant.AtomicBoard` rules (``[Variant "Atomic"]`` PGN header).
"""

from __future__ import annotations

import io
import json
import random
import re
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
from chess.variant import AtomicBoard


def _strip_braced_segments(movetext: str) -> str:
    return re.sub(r"\{[^}]*\}", "", movetext)


def _strip_trailing_result(movetext: str) -> str:
    s = movetext.strip()
    for suf in (" 1-0", " 0-1", " 1/2-1/2", " *"):
        if s.endswith(suf):
            s = s[: -len(suf)].strip()
            break
    return s


def _fen_from_movetext(movetext: str, rng: random.Random, min_plies: int, max_plies: int) -> Optional[str]:
    mt = _strip_trailing_result(_strip_braced_segments(movetext))
    if not mt or len(mt) < 8:
        return None
    pgn = '[Event "?"]\n[Site "?"]\n[Date "2000.01.01"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n[Variant "Atomic"]\n\n' + mt + " *\n"
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
    except Exception:
        return None
    if game is None:
        return None
    try:
        mainline = list(game.mainline_moves())
    except Exception:
        return None
    if len(mainline) < min_plies:
        return None
    n = rng.randint(min_plies, min(max_plies, len(mainline)))
    b = game.board()
    if not isinstance(b, AtomicBoard):
        b = AtomicBoard(b.fen())
    for mv in mainline[:n]:
        if not b.is_legal(mv):
            return None
        b.push(mv)
    if b.is_variant_end() or b.king(b.turn) is None:
        return None
    return b.fen()


def load_atomic_hf_fen_pool(
    *,
    target: int = 400,
    max_scan: int = 20000,
    seed: int = 42,
    min_plies: int = 6,
    max_plies: int = 40,
    cache_path: Optional[Path] = None,
) -> list[str]:
    """Return up to *target* FEN strings from streamed atomic games.

    If *cache_path* is set and the file exists, load JSON list from disk instead of HF.
    """
    if cache_path is not None and cache_path.is_file():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw[:target]]

    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "Install datasets to use ATOMIC_USE_HF_FENS=1: pip install datasets"
        ) from e

    rng = random.Random(seed)
    out: list[str] = []
    ds = load_dataset("Lichess/atomic-chess-games", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= max_scan:
            break
        mt = row.get("movetext")
        if not isinstance(mt, str):
            continue
        fen = _fen_from_movetext(mt, rng, min_plies, max_plies)
        if fen:
            out.append(fen)
        if len(out) >= target:
            break

    if cache_path is not None and out:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out), encoding="utf-8")

    return out
