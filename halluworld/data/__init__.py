"""Packaged benchmark data: levels, terminal tasks, and the question bank.

Levels and configs are shipped inside the wheel, so `pip install halluworld`
yields an install that can construct a level without a repository checkout.
Before this, level paths were hardcoded relative to the current working
directory, which meant the package only worked when invoked from the
repository root.

The question bank is different. It is deliberately *not* in the wheel or the
public source tree: it is distributed as a gated Hugging Face dataset so the
items are one click-through away from anyone evaluating a model but are not
sitting in a GitHub tree that gets scraped into training corpora.
``questions_dir`` resolves it, in this order:

1. ``HALLUWORLD_QUESTIONS_DIR`` -- a local copy (air-gapped machines, CI
   caches). A full download of the dataset, a folder holding ``<version>/``
   bank directories, or a bank directory itself.
2. ``halluworld/data/questions/<version>/`` in a source checkout. Gitignored,
   so this only exists on machines that built or downloaded the bank there.
3. The Hugging Face dataset ``HALLUWORLD_HF_REPO`` (default below), fetched
   with ``huggingface_hub`` into its cache. Only the version's own folder is
   downloaded; ``hub_bank_path`` says which. Needs an account that has accepted
   the gate and is logged in (``hf auth login`` or ``HF_TOKEN``).

Use the helpers below rather than building paths by hand:

    from halluworld.data import level_path, LEVELS_DIR, questions_dir

    spec = load_level(level_path("u1_hub.txt"))
    bank = questions_dir("v0.1") / "terminal.jsonl.gz"
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "DATA_DIR",
    "LEVELS_DIR",
    "PROBE_BANKS_DIR",
    "TERMINAL_TASKS_DIR",
    "HF_QUESTIONS_REPO",
    "QuestionBankUnavailable",
    "level_path",
    "probe_bank_dir",
    "hub_bank_path",
    "questions_dir",
    "list_levels",
]

DATA_DIR = Path(__file__).resolve().parent

#: ASCII level definitions (`*.txt`) and InNav configs (`*.innav.json`).
LEVELS_DIR = DATA_DIR / "levels"

#: Raw probe-generation inputs, one directory per generation. Local only:
#: gitignored, not in the wheel. The released ones also ship under
#: `<version>/raw/` in the Hugging Face dataset.
PROBE_BANKS_DIR = DATA_DIR / "probe_banks"

#: Our own hand-authored terminal tasks (as opposed to the vendored upstream ones).
TERMINAL_TASKS_DIR = DATA_DIR / "terminal_tasks"

#: The gated Hugging Face dataset the frozen question bank is distributed
#: from. Override with the HALLUWORLD_HF_REPO environment variable.
HF_QUESTIONS_REPO = "DegenAI-Labs/HalluWorld"


class QuestionBankUnavailable(RuntimeError):
    """The frozen question bank could not be found locally or downloaded."""


def level_path(name: str) -> Path:
    """Absolute path to a packaged level file.

    Accepts a bare name ("u1_hub.txt"), a name without extension ("u1_hub"),
    or a legacy "levels/"-prefixed path, so existing call sites keep working.
    """
    rel = name[len("levels/"):] if name.startswith("levels/") else name
    candidate = LEVELS_DIR / rel
    if not candidate.suffix:
        candidate = candidate.with_suffix(".txt")
    return candidate


def probe_bank_dir(name: str) -> Path:
    """Absolute path to a local probe bank directory."""
    return PROBE_BANKS_DIR / name


def list_levels(pattern: str = "*.txt") -> list[Path]:
    """Every packaged level matching *pattern*, sorted for reproducibility."""
    return sorted(LEVELS_DIR.glob(pattern))


def hub_bank_path(version: str) -> str:
    """Where a bank version sits inside the Hugging Face dataset.

    The dataset has one folder per release, each holding its own questions,
    configs and raw inputs, so a version's bank is ``<version>/questions``.
    Extra observation conditions (v0.2's incorrect-FEN chess) live in that same
    folder and manifest, not in a version of their own.
    """
    return f"{version}/questions"


def questions_dir(version: str = "v0.1", *, download: bool = True) -> Path:
    """The directory holding ``manifest.json`` and the bank files for *version*.

    See the module docstring for the resolution order. Raises
    ``QuestionBankUnavailable`` with instructions when nothing resolves; every
    caller that needs the bank lets that propagate or rewraps it, because there
    is nothing useful to do without the items.
    """
    override = os.environ.get("HALLUWORLD_QUESTIONS_DIR", "").strip()
    if override:
        root = Path(override).expanduser()
        # A bank directory itself, a folder of versions, or a full download
        # of the Hugging Face dataset.
        for candidate in (root / version, root, root / hub_bank_path(version)):
            if (candidate / "manifest.json").is_file():
                return candidate
        raise QuestionBankUnavailable(
            f"HALLUWORLD_QUESTIONS_DIR={override!r} holds no manifest.json for {version}"
        )

    local = DATA_DIR / "questions" / version
    if (local / "manifest.json").is_file():
        return local

    if not download:
        raise QuestionBankUnavailable(_help(version, "not downloaded"))
    return _download(version)


def _download(version: str) -> Path:
    repo_id = os.environ.get("HALLUWORLD_HF_REPO", "").strip() or HF_QUESTIONS_REPO
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise QuestionBankUnavailable(
            _help(version, "huggingface_hub is not installed (pip install huggingface_hub)")
        ) from exc
    try:
        # Falls back to the local cache when offline, so a bank fetched once
        # keeps working without a network connection.
        root = snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            allow_patterns=[f"{hub_bank_path(version)}/*"],
        )
    except Exception as exc:  # gated, missing, offline with an empty cache, ...
        raise QuestionBankUnavailable(
            _help(version, f"could not fetch {repo_id}: {type(exc).__name__}: {exc}")
        ) from exc
    path = Path(root) / hub_bank_path(version)
    if not (path / "manifest.json").is_file():
        raise QuestionBankUnavailable(f"{repo_id} has no {hub_bank_path(version)}/manifest.json")
    return path


def _help(version: str, reason: str) -> str:
    repo_id = os.environ.get("HALLUWORLD_HF_REPO", "").strip() or HF_QUESTIONS_REPO
    return (
        f"question bank {version} is unavailable ({reason}).\n"
        f"The bank is a gated Hugging Face dataset. To get it:\n"
        f"  1. request access at https://huggingface.co/datasets/{repo_id}\n"
        f"  2. log in once: `hf auth login` (or export HF_TOKEN=...)\n"
        f"  3. rerun; the bank is downloaded into the Hugging Face cache.\n"
        f"Or point HALLUWORLD_QUESTIONS_DIR at a local copy of the dataset "
        f"(or of its {hub_bank_path(version)}/ folder)."
    )
