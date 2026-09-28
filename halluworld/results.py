"""One result schema for all four tracks.

Before this there were eight incompatible per-probe CSV layouts, differing in
column set, column names for the same quantity, and whether a column existed at
all. Nothing could be aggregated across tracks without bespoke glue per file.

THE UNIT OF RECORD IS ONE PROBE TRIAL

One question, asked of one model, about one environment state. Episodes,
levels, and runs are aggregation keys over that, not separate row types. This
is already the shape of EpisodeResult in benchmark.py and of the chess JSONL,
so v1 is a widening of what existed rather than an invention.

TWO MODELLING DECISIONS WORTH THE WORDS

Track-specific fields live in one JSON `extra` column, not as optional columns.
The union across tracks is 27+ fields and grows with every probe class, while
any given row populates three to six. As columns that is a wide CSV that is
>80% empty and a schema migration every time someone adds a probe. As a blob it
is one column, and pd.json_normalize recovers the wide view in a line. The 30
canonical columns are exactly the ones you group by; nothing you group by lives
in `extra`.

InNav paired trials become TWO rows, not one. The legacy format put both arms
on a single row (egocentric_response/egocentric_score alongside
controlled_static_response/controlled_static_score) and additionally repeated
per-episode aggregates (egocentric_accuracy) on every probe row -- which
silently double-counts in any groupby().mean(). Splitting makes InNav rows
structurally identical to every other track's, so one loader and one aggregator
serve all four, and turns the paired analysis into a self-join on pair_id.

score=None MEANS EXCLUDED, NOT WRONG

0.0 is a wrong answer. None is a trial that should not count -- an API failure,
a probe that could not generate. Conflating them is how an infrastructure
outage becomes a reported hallucination rate.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"

TRACKS = ("grid", "chess", "innav", "terminal")

#: Populated only when a trial was excluded rather than scored.
ERRORS = (
    None,
    "rate_limit",
    "bad_request",
    "empty_response",
    "refusal",
    "api_error",
    "dry_run",
    "probe_skipped",
    "legacy_unrecoverable",
)


def classify_error(exc: Exception) -> str:
    """Map a provider exception onto the ERRORS vocabulary.

    Lives beside ERRORS because that is the vocabulary it has to produce:
    ProbeRecord rejects anything else, so a track passing a raw exception name
    crashes while building the record and loses the original error with it.
    """
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "ratelimit" in name or "rate limit" in text or "429" in text:
        return "rate_limit"
    if "badrequest" in name or "bad request" in text or "400" in text:
        return "bad_request"
    return "api_error"


def exclusion_reason(response) -> str | None:
    """Why a response is excluded from scoring, or None if it should be graded.

    One rule, used by every track, so "refusal" means the same thing in every
    table:

    refusal         STRICTLY zero output tokens, AND the provider flagged the
                    response as a refusal. Both are required. A provider-flagged
                    refusal that still emitted text is graded as that text: the
                    model engaged, however unhelpfully.
    empty_response  no answer text at all (a model that spent its whole budget
                    on internal reasoning). Judged on the text, not the token
                    count: a provider that omits usage reports zero tokens for
                    every answer, and that must not exclude answers that exist.

    Excluded means score=None: out of both numerator and denominator. It is not
    a wrong answer, because no answer was given.
    """
    if (response.completion_tokens or 0) == 0 and response.stop_reason == "refusal":
        return "refusal"
    if not (response.text or "").strip():
        return "empty_response"
    return None


@dataclass
class ProbeRecord:
    """One probe trial. The 30 canonical columns."""

    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    track: str = ""
    suite: str = ""
    level: str = ""
    variant: str = ""            # condition axis: serializer, fen mode, or innav arm
    serializer: str | None = None
    model: str = ""              # as reported by the API when known
    model_requested: str = ""
    provider: str = ""
    episode: int | None = None
    trial: int | None = None
    seed: int | None = None
    base_seed: int | None = None
    steps_before_probe: int | None = None
    timestep: int | None = None
    pair_id: str | None = None   # joins the two arms of an InNav trial
    question_id: str = ""        # links to the frozen question bank when known
    probe_type: str = ""
    question: str = ""
    ground_truth: Any = None
    response: str = ""
    is_correct: bool | None = None
    score: float | None = None   # None == excluded; see module docstring
    parse_note: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.track and self.track not in TRACKS:
            raise ValueError(f"unknown track {self.track!r}")
        if self.error not in ERRORS:
            raise ValueError(f"unknown error {self.error!r} (expected one of {ERRORS!r})")
        if self.score is not None and not (0.0 <= float(self.score) <= 1.0):
            raise ValueError(f"score must be in [0,1] or None, got {self.score!r}")


COLUMNS = tuple(ProbeRecord().__dict__.keys())


def write_jsonl(records: Iterable[ProbeRecord], path: str | Path) -> int:
    """JSONL is canonical.

    Not CSV: `question` and `response` routinely contain embedded newlines, and
    round-tripping those through CSV is exactly what produced the "corrupted
    CSVs" that had to be preserved and recovered in this project's history.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[ProbeRecord]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield ProbeRecord(**json.loads(line))


def write_csv(records: Iterable[ProbeRecord], path: str | Path) -> int:
    """A convenience view, regenerable from the JSONL. `extra` is JSON-encoded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        w.writeheader()
        for rec in records:
            row = asdict(rec)
            row["extra"] = json.dumps(row["extra"], sort_keys=True)
            row["ground_truth"] = json.dumps(row["ground_truth"])
            w.writerow(row)
            n += 1
    return n


def summarize(records: Iterable[ProbeRecord]) -> dict:
    """Per (track, level, variant, probe_type) counts and rates.

    Excluded trials (score is None) are counted separately and kept out of the
    mean, so an outage lowers n rather than lowering the score.
    """
    groups: dict[tuple, dict] = {}
    for rec in records:
        key = (rec.track, rec.level, rec.variant, rec.probe_type)
        g = groups.setdefault(key, {"n": 0, "scored": 0, "excluded": 0, "total": 0.0})
        g["n"] += 1
        if rec.score is None:
            g["excluded"] += 1
        else:
            g["scored"] += 1
            g["total"] += float(rec.score)
    out = {}
    for key, g in sorted(groups.items()):
        mean = (g["total"] / g["scored"]) if g["scored"] else None
        out["|".join(str(k) for k in key)] = {
            "n": g["n"],
            "scored": g["scored"],
            "excluded": g["excluded"],
            "accuracy": mean,
            "hallucination_rate": (1.0 - mean) if mean is not None else None,
        }
    return out
