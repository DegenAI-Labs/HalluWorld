"""The frozen question bank: one schema for all four tracks.

A *question bank* is the set of probes a benchmark version asks, frozen so that
every model is evaluated on identical items and so that a published number can
be traced back to the exact question that produced it.

Versions
--------
``v0.1`` is public: the questions actually used in the reported experiments.
Later versions may be held out, in which case only the manifest -- counts and
checksums -- is published, so that set membership and size can be verified
without revealing the items.

Why one schema for four tracks
------------------------------
The tracks differ enormously in what an "observation" is: a tmux pane capture
for terminal, a serialized grid for gridworld, a FEN or SAN move list for
chess. They agree on everything that matters for scoring and aggregation --
there is a question, a ground truth, a way to grade it, and a level or task it
belongs to. Those are the fields below; anything track-specific goes in
``extra``.

Three fields deserve explanation:

``cognitive_tier``
    The cross-track comparison axis, and the reason a single bank is worth
    building. The tracks turn out to already share one taxonomy under two
    names: gridworld encodes it in the level-key prefix (P1_dense_array,
    M2_witness_stand, ...), terminal encodes it in ``probe_type``
    (perceptual, memory, causal, uncertainty, cross-tier compound), and the
    chess battery documents the same P/C/M/X split per probe. Both mappings
    below are *derived*, not guessed:

        P  perceptual     read the observation as given
        M  memory         recall state no longer visible
        C  causal         apply a rule or dynamic to reach a new state
        U  uncertainty    decline to answer beyond the evidence
        X  compound       two or more of the above in one item

``failure_mode_target``
    A finer-grained label for *how* the item is expected to fail. The terminal
    bank carries explicit per-probe labels from its generator. Gridworld does
    not: it has free-text ``trap`` prose written for humans ("one-shot
    trigger; gate stays open; model's continuous prior says 'closed'"), which
    does not map onto the taxonomy without inventing a classification. So grid
    records are ``unclassified`` and keep the original prose in
    ``extra.failure_mode_original``. Slice by ``cognitive_tier`` when
    comparing across tracks; ``failure_mode_target`` is terminal-only for now.

``kind``
    ``fixed`` means the question text and ground truth are literals, frozen
    here. ``generated`` means the probe is constructed from a seeded RNG at run
    time; the bank stores the class and its arguments instead, and freezing is
    guaranteed by the seed rather than by the text. Mixing the two in one file
    is deliberate -- a consumer that just wants to ask questions reads
    ``question``/``ground_truth``, and a consumer that wants to regenerate the
    bank reads ``probe_class``/``probe_kwargs``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = "1.0"

TRACKS = ("grid", "chess", "innav", "terminal")

#: The cross-track cognitive taxonomy. See the module docstring.
TIERS = ("P", "M", "C", "U", "X", "unknown")

#: The shared failure-mode taxonomy. Terminal used these five directly;
#: gridworld's free-text `trap` annotations are mapped onto them.
FAILURE_MODES = (
    "stale_memory",
    "cross_tier_reasoning",
    "uncertainty_overclaim",
    "causal_shortcut",
    "version_api_hallucination",
    "unclassified",
)


@dataclass
class QuestionRecord:
    """One frozen benchmark item."""

    # identity
    question_id: str                 # stable, namespaced: "terminal/000498", "grid/P4_harder_array/q07"
    track: str                       # one of TRACKS
    suite: str                       # e.g. "perception", "chess_standard", "terminal_llm_generated"
    task_or_level: str               # level key or terminal task name

    # the item
    kind: str                        # "fixed" | "generated"
    probe_type: str                  # "presence", "count", "uncertainty", ...
    question: str = ""               # verbatim; empty for kind="generated"
    ground_truth: Any = None
    answer_schema: str = ""          # e.g. "status=<created|already_existed|undetermined>"
    context: str = ""                # the observation shown to the model, when frozen with the item

    # for kind="generated": how to rebuild it
    probe_class: str = ""
    probe_kwargs: dict = field(default_factory=dict)

    # analysis axes
    cognitive_tier: str = "unknown"   # P | M | C | U | X
    failure_mode_target: str = "unclassified"
    difficulty: int | None = None
    answerability: int | None = None

    # provenance and track-specific payload
    provenance: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    schema_version: str = SCHEMA_VERSION

    def __post_init__(self):
        if self.track not in TRACKS:
            raise ValueError("unknown track %r (expected one of %r)" % (self.track, TRACKS))
        if self.kind not in ("fixed", "generated"):
            raise ValueError("kind must be 'fixed' or 'generated', got %r" % (self.kind,))
        if self.cognitive_tier not in TIERS:
            raise ValueError("unknown cognitive tier %r (expected one of %r)"
                             % (self.cognitive_tier, TIERS))
        if self.failure_mode_target not in FAILURE_MODES:
            raise ValueError("unknown failure mode %r" % (self.failure_mode_target,))
        if self.kind == "fixed" and not self.question:
            raise ValueError("%s: kind='fixed' requires a question" % self.question_id)
        if self.kind == "generated" and not self.probe_class:
            raise ValueError("%s: kind='generated' requires probe_class" % self.question_id)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


def write_bank(records: list[QuestionRecord], path: Path) -> dict:
    """Write records as gzipped JSONL, sorted by question_id for reproducibility.

    Returns a manifest entry: count, sha256 of the *uncompressed* bytes, and the
    file name. Hashing before compression keeps the digest stable across gzip
    implementations and compression levels.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda r: r.question_id)
    payload = "\n".join(r.to_json() for r in ordered).encode("utf-8")
    # mtime=0 so repeated builds of identical content produce identical files.
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as fh:
        fh.write(payload)
    return {
        "file": path.name,
        "count": len(ordered),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def read_bank(path: Path) -> Iterator[QuestionRecord]:
    """Stream records back out of a bank file (gzipped or plain JSONL)."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield QuestionRecord(**json.loads(line))


def verify_bank(path: Path, expected: dict) -> tuple[bool, str]:
    """Check a bank file against its manifest entry.

    Returns (ok, message). Used by `halluworld questions verify` and by the
    test suite, so a bank that drifts from its published checksum is caught
    rather than silently shipped.
    """
    path = Path(path)
    if not path.exists():
        return False, "missing file: %s" % path
    with gzip.open(path, "rb") as fh:
        payload = fh.read()
    digest = hashlib.sha256(payload).hexdigest()
    # split("\n"), never splitlines(). Terminal contexts are raw tmux pane
    # captures containing control characters. JSON escapes \n and \r, but
    # passes \x0b, \x0c, \x1c-\x1e, \x85, U+2028 and U+2029 through literally,
    # and str.splitlines() treats every one of those as a line break -- which
    # counted a 529-record bank as 554.
    count = sum(1 for line in payload.decode("utf-8").split("\n") if line.strip())
    if digest != expected["sha256"]:
        return False, "%s: sha256 mismatch (expected %s, got %s)" % (
            path.name, expected["sha256"][:12], digest[:12])
    if count != expected["count"]:
        return False, "%s: expected %d records, found %d" % (path.name, expected["count"], count)
    return True, "%s: %d records, sha256 ok" % (path.name, count)
