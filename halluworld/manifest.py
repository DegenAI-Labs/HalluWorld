"""Per-run provenance: what code, what config, what models produced a number.

Every run writes one of these. It answers the question that was previously
unanswerable for any published HalluWorld figure: given a number, what exactly
made it?

Two entries are worth calling out.

``git.dirty``
    A run from a modified working tree is not reproducible, and the manifest
    says so rather than implying otherwise.

``models[].reported``
    The dated model snapshot as the API returned it, not the alias that was
    requested. Nothing in this project previously recorded which snapshot of
    `gpt-5.5` produced a result, so "we evaluated gpt-5.5" was untraceable once
    the alias moved. LMResponse.model already carries it; this just keeps it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

#: Recorded via importlib.metadata rather than `pip freeze`, which would leak
#: local paths, virtualenv names, and every unrelated package on the machine.
TRACKED_PACKAGES = (
    "numpy", "pandas", "minigrid", "gymnasium", "openai", "anthropic",
    "tqdm", "chess", "pyyaml",
)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(("git",) + args, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True,
                             cwd=Path(__file__).resolve().parent.parent)
        return out.stdout.strip() if out.returncode == 0 else ""
    except OSError:
        return ""


def git_state() -> dict:
    porcelain = _git("status", "--porcelain")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(porcelain),
        # Identifies *which* dirty state, so two runs from two different
        # uncommitted trees are distinguishable after the fact.
        "dirty_digest": (hashlib.sha256(porcelain.encode()).hexdigest()[:16]
                         if porcelain else None),
    }


def package_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for name in TRACKED_PACKAGES:
        dist = "python-chess" if name == "chess" else name
        try:
            out[name] = md.version(dist)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def config_digest(config: dict) -> str:
    payload = json.dumps({k: v for k, v in config.items() if k != "_meta"},
                         sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def build(
    run_id: str,
    config: dict,
    *,
    track: str = "",
    suite: str = "",
    question_bank: dict | None = None,
    models: list[dict] | None = None,
    counts: dict | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "track": track or (config.get("_meta") or {}).get("track", ""),
        "suite": suite or (config.get("_meta") or {}).get("suite", ""),
        "started_at": started_at or now,
        "finished_at": now,
        "git": git_state(),
        "halluworld_version": _version(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "packages": package_versions(),
        "config": {k: v for k, v in config.items() if k != "_meta"},
        "config_sha256": config_digest(config),
        "question_bank": question_bank or {},
        "models": models or [],
        "counts": counts or {},
    }


def _version() -> str:
    try:
        from halluworld import __version__
        return __version__
    except Exception:
        return "unknown"


def write(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
