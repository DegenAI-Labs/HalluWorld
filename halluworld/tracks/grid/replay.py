"""Score a model against the frozen grid question bank.

`eval grid` regenerates questions by running perception.make_probes(), which
works for v0.1 because v0.1's bank was extracted from exactly that function. It
does not generalise: an authored set (grid v0.2) exists only as records, and
make_probes can produce none of it, so regenerating would score v0.1's
questions under a v0.2 label.

This module replays the bank instead. Each record carries its own rendered
model-visible context and, in extra, the per-level system prompt it was
authored against -- grid uses ten different rule preambles, so that is
per-record rather than one constant.
"""

from __future__ import annotations

import re
import time

from halluworld.data import QuestionBankUnavailable, questions_dir
from halluworld.questions import QuestionRecord, read_bank
from halluworld.results import ProbeRecord, classify_error, exclusion_reason

#: Answers that count as an explicit refusal to answer, for `cannot_determine`
#: items. The bank's epistemic probes are scored right only when the model
#: declines, so the wording it declines with must not decide the score.
_ABSTENTIONS = {
    "cannot_determine", "cannot determine", "can't determine", "cant determine",
    "unknown", "undetermined", "indeterminate", "not determinable",
    "insufficient information", "not enough information", "unanswerable",
}

_INT_RE = re.compile(r"-?\d+")


class GridReplayError(RuntimeError):
    """The requested replay cannot be run."""


def load_grid_bank(version: str) -> list[QuestionRecord]:
    try:
        path = questions_dir(version) / "grid.jsonl.gz"
    except QuestionBankUnavailable as exc:
        raise GridReplayError(str(exc)) from exc
    if not path.exists():
        raise GridReplayError(f"no grid question bank for version {version!r} at {path}")
    records = list(read_bank(path))
    generated = [r for r in records if r.kind != "fixed"]
    if generated:
        raise GridReplayError(
            f"grid bank {version} holds {len(generated)} generated record(s), which "
            "store a probe recipe rather than a question and cannot be replayed; "
            "use --regenerate for that bank"
        )
    return records


def select_records(args, version: str) -> list[QuestionRecord]:
    records = load_grid_bank(version)
    wanted = set(args.levels) if args.levels else None
    if wanted:
        unknown = wanted - {r.task_or_level for r in records}
        if unknown:
            raise GridReplayError("unknown grid level(s): %s" % ", ".join(sorted(unknown)))
        records = [r for r in records if r.task_or_level in wanted]

    limit = getattr(args, "limit", None)
    episodes = getattr(args, "episodes", None)
    per_level = episodes if limit is None else min(episodes, limit)
    if not per_level or per_level < 1:
        return records

    seen: dict[str, int] = {}
    kept = []
    for record in records:
        n = seen.get(record.task_or_level, 0)
        if n >= per_level:
            continue
        seen[record.task_or_level] = n + 1
        kept.append(record)
    return kept


def _normalize(text: str) -> str:
    text = text.strip().strip(".").strip()
    text = re.sub(r"^(the\s+answer\s+is|answer)\s*[:\-]?\s*", "", text, flags=re.I)
    # Strip markdown emphasis but NOT underscores: several ground truths are
    # snake_case ("cannot_determine"), and removing the underscore turned a
    # correct answer into an unrecognised one.
    text = re.sub(r"[*`]", "", text)
    return " ".join(text.split()).strip().strip(".").lower()


def grade(expected, actual: str, answer_schema: str) -> tuple[bool, str]:
    """Grade one grid answer. Returns (correct, parse_note).

    The schema decides the comparison, not the Python type of `expected`:
    "one integer" is numeric, "yes|no" is boolean, everything else is a closed
    set of literal strings.
    """
    normalized = _normalize(actual)
    if not normalized:
        return False, "empty response"

    schema = (answer_schema or "").strip().lower()

    if schema == "one integer":
        # The coauthor's package flags the original scorer for taking the FIRST
        # integer anywhere in the reply, so "there are 3 doors, so the answer is
        # 2" scored as 3. Taking the last integer reads a reasoned answer
        # correctly and is identical for a bare "2".
        found = _INT_RE.findall(normalized)
        if not found:
            return False, "no integer in response"
        value = int(found[-1])
        note = "" if len(found) == 1 else "multiple integers; used the last (%s)" % found
        return value == int(expected), note

    if schema == "yes|no":
        first = normalized.split()[0].strip(",;:")
        if first in ("yes", "true"):
            return bool(expected) is True, ""
        if first in ("no", "false"):
            return bool(expected) is False, ""
        return False, "not a yes/no answer"

    if schema == "cannot_determine":
        # Scored right only when the model declines; any confident answer is the
        # overclaim these items are built to catch.
        declined = normalized in _ABSTENTIONS or any(
            normalized.startswith(a) for a in _ABSTENTIONS
        )
        return declined, "" if declined else "answered instead of declining"

    options = [o.strip().lower() for o in (answer_schema or "").split("|") if o.strip()]
    expected_text = _normalize(str(expected))
    if len(options) > 1:
        # Closed set: accept the option the reply names, so trailing prose does
        # not fail an otherwise-correct answer. Ambiguous replies naming more
        # than one option are wrong rather than generously resolved.
        named = [o for o in options if o in normalized]
        if len(named) == 1:
            return named[0] == expected_text, ""
        if len(named) > 1:
            return False, "response names several options: %s" % named
        return False, "response matches no option in the schema"

    return normalized == expected_text, "" if normalized == expected_text else "free-text mismatch"


def evaluate_record(record: QuestionRecord, args, lm, version: str) -> ProbeRecord:
    base = dict(
        run_id=getattr(args, "run_id", ""),
        track="grid",
        suite=record.suite,
        level=record.task_or_level,
        model_requested=args.model,
        provider=args.provider,
        question_id=record.question_id,
        probe_type=record.probe_type,
        question=record.question,
        ground_truth=record.ground_truth,
    )
    system = (record.extra or {}).get("system_prompt") or ""
    user = "%s\n\n%s" % (record.context, record.question)

    started = time.monotonic()
    try:
        response = lm.query(system=system, user=user)
    except Exception as exc:  # noqa: BLE001 -- provider SDKs expose unrelated hierarchies
        return ProbeRecord(
            **base,
            latency_s=time.monotonic() - started,
            error=classify_error(exc),
            parse_note=f"{type(exc).__name__}: {exc}",
            extra={"bank_version": version, "answer_schema": record.answer_schema},
        )
    latency = time.monotonic() - started

    if response.error:
        return ProbeRecord(
            **base,
            model=response.model,
            response=response.text,
            latency_s=latency,
            error=response.error,
            extra={**{"bank_version": version, "answer_schema": record.answer_schema}, "stop_reason": response.stop_reason},
        )

    # See the note in chess/replay.py: an empty answer is an exclusion, not a
    # wrong answer. grade() would return False for it and score it 0.
    reason = exclusion_reason(response)
    if reason:
        return ProbeRecord(
            **base,
            model=response.model,
            response=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=latency,
            error=reason,
            parse_note="excluded: %s (stop_reason=%s, output_tokens=%s)"
                       % (reason, response.stop_reason or "unreported",
                          response.completion_tokens),
            extra={"bank_version": version, "answer_schema": record.answer_schema,
                   "stop_reason": response.stop_reason},
        )

    correct, note = grade(record.ground_truth, response.text, record.answer_schema)
    return ProbeRecord(
        **base,
        model=response.model,
        response=response.text,
        is_correct=correct,
        score=1.0 if correct else 0.0,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_s=latency,
        parse_note=note,
        extra={"bank_version": version, "answer_schema": record.answer_schema,
               "stop_reason": response.stop_reason, "attempts": response.attempts},
    )


def run_grid_replay(args, version: str,
                    skip_ids: set[str] | None = None) -> list[ProbeRecord]:
    """Replay the selected bank records, optionally across threads."""
    from halluworld.lm.factory import LMConfigError, make_lm

    selected = select_records(args, version)
    if skip_ids:
        selected = [r for r in selected if r.question_id not in skip_ids]
        if not selected:
            return []
    try:
        lm = make_lm(args)
    except LMConfigError as exc:
        raise GridReplayError(str(exc)) from exc
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    verbose = getattr(args, "verbose", False)
    total = len(selected)

    if workers == 1:
        out = []
        for index, record in enumerate(selected, start=1):
            out.append(evaluate_record(record, args, lm, version))
            if verbose:
                print("[%d/%d] %s" % (index, total, record.question_id), flush=True)
        return out

    from concurrent.futures import ThreadPoolExecutor, as_completed

    ordered: list[ProbeRecord | None] = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_record, record, args, lm, version): position
            for position, record in enumerate(selected)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
            done += 1
            if verbose:
                print("[%d/%d] done" % (done, total), flush=True)
    return [r for r in ordered if r is not None]
