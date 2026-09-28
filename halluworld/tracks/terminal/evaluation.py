"""Evaluate models against the packaged frozen Terminal probe bank."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from halluworld import manifest as manmod
from halluworld.data import QuestionBankUnavailable, questions_dir
from halluworld.questions import QuestionRecord, read_bank
from halluworld.results import ProbeRecord, read_jsonl, summarize, write_jsonl, classify_error, exclusion_reason

SYSTEM_PROMPT = (
    "Answer the Terminal runtime probe using only the supplied evidence. "
    "Return only the exact requested answer string and no explanation."
)

USER_PROMPT = """You are answering a HalluWorld Terminal runtime probe.
Use only the frozen terminal evidence below. Do not assume filesystem state
that is not present in the evidence.

TERMINAL EVIDENCE:
<<<CONTEXT
{context}
CONTEXT>>>

QUESTION:
{question}

REQUIRED ANSWER SCHEMA:
{answer_schema}
"""

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "xai": "XAI_API_KEY",
}


class TerminalEvaluationError(RuntimeError):
    """The Terminal evaluation cannot be started safely."""


@dataclass(frozen=True)
class Grade:
    correct: bool
    match_kind: str
    expected_assignments: dict[str, str]
    actual_assignments: dict[str, str]
    key_matches: dict[str, bool]


def schema_keys(schema: str) -> list[str]:
    """Return declared assignment keys once, preserving schema order."""
    keys: list[str] = []
    for key in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", schema or ""):
        if key not in keys:
            keys.append(key)
    return keys


def _strip_outer_markup(text: str) -> str:
    value = (text or "").strip()
    fenced = re.fullmatch(
        r"```(?:[A-Za-z0-9_-]+)?\s*\n?(.*?)\n?```", value, re.DOTALL
    )
    if fenced:
        value = fenced.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"`":
        value = value[1:-1].strip()
    return value


def normalize_answer(text: str) -> str:
    """Normalize presentation only; never discard structured punctuation."""
    value = _strip_outer_markup(text).lower()
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*(->|[=,;|])\s*", r"\1", value)
    return value


def _parse_declared_assignments(text: str, keys: list[str]) -> dict[str, str] | None:
    """Parse values by locating the next declared key, not by value punctuation."""
    if not keys:
        return None
    value = _strip_outer_markup(text)
    key_pattern = "|".join(re.escape(key) for key in sorted(keys, key=len, reverse=True))
    matches = list(re.finditer(rf"(?<![A-Za-z0-9_])({key_pattern})\s*=", value))
    if not matches or value[: matches[0].start()].strip():
        return None

    parsed: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1)
        if key in parsed:
            return None
        end = matches[index + 1].start() if index + 1 < len(matches) else len(value)
        raw = value[match.end() : end].strip()
        if index + 1 < len(matches):
            raw = raw.rstrip()
            if raw.endswith((";", ",")):
                raw = raw[:-1].rstrip()
        if not raw:
            return None
        parsed[key] = raw
    return parsed


def grade_answer(*, expected: str, actual: str, answer_schema: str) -> Grade:
    """Conservatively score one answer without truncating structured values."""
    if normalize_answer(actual) == normalize_answer(expected):
        return Grade(True, "exact_normalized", {}, {}, {})

    keys = schema_keys(answer_schema)
    expected_values = _parse_declared_assignments(expected, keys)
    actual_values = _parse_declared_assignments(actual, keys)
    if expected_values is None or actual_values is None or set(actual_values) != set(keys):
        return Grade(False, "exact_fallback", expected_values or {}, actual_values or {}, {})

    key_matches = {
        key: normalize_answer(actual_values[key]) == normalize_answer(expected_values[key])
        for key in keys
        if key in expected_values
    }
    correct = set(expected_values) == set(keys) and all(key_matches.values())
    return Grade(correct, "schema_key_values", expected_values, actual_values, key_matches)


def load_terminal_bank(version: str = "v0.1") -> list[QuestionRecord]:
    try:
        path = questions_dir(version) / "terminal.jsonl.gz"
    except QuestionBankUnavailable as exc:
        raise TerminalEvaluationError(str(exc)) from exc
    if not path.is_file():
        raise TerminalEvaluationError(f"missing Terminal bank: {path}")
    records = list(read_bank(path))
    if not records:
        raise TerminalEvaluationError(f"packaged Terminal bank is empty: {path}")
    return records


def select_records(args) -> list[QuestionRecord]:
    records = load_terminal_bank(args.version)
    tasks = set(args.tasks or [])
    question_ids = set(args.question_ids or [])
    probe_types = set(args.probe_types or [])
    if question_ids:
        known = {record.question_id for record in records}
        unknown = question_ids - known
        if unknown:
            raise TerminalEvaluationError(
                f"unknown Terminal question ID(s): {sorted(unknown)}"
            )
        records = [record for record in records if record.question_id in question_ids]
    if tasks:
        known = {record.task_or_level for record in records}
        unknown = tasks - known
        if unknown:
            raise TerminalEvaluationError(f"unknown Terminal task(s): {sorted(unknown)}")
        records = [record for record in records if record.task_or_level in tasks]
    if probe_types:
        known = {record.probe_type for record in records}
        unknown = probe_types - known
        if unknown:
            raise TerminalEvaluationError(
                f"unknown Terminal probe type(s): {sorted(unknown)}"
            )
        records = [record for record in records if record.probe_type in probe_types]
    if args.limit is not None:
        records = records[: args.limit]
    return records


def build_prompt(record: QuestionRecord, max_context_chars: int = 0) -> tuple[str, int]:
    context = record.context
    omitted = 0
    if max_context_chars and len(context) > max_context_chars:
        omitted = len(context) - max_context_chars
        context = (
            f"[Context truncated: omitted {omitted} leading characters.]\n"
            + context[-max_context_chars:]
        )
    return USER_PROMPT.format(
        context=context,
        question=record.question,
        answer_schema=record.answer_schema,
    ), omitted


def _make_lm(args):
    """Delegates to the shared factory.

    The local copy routed every non-openai/anthropic provider through
    BasetenLM using --api-base, whose default is Baseten's endpoint -- so an
    xai run would have been sent to Baseten.
    """
    from halluworld.lm.factory import LMConfigError, make_lm

    try:
        return make_lm(args)
    except LMConfigError as exc:
        raise TerminalEvaluationError(str(exc)) from exc


def _classify_error(exc: Exception) -> str:
    """Kept as a name here; the implementation moved beside the ERRORS vocabulary."""
    return classify_error(exc)


def _dry_run_record(record: QuestionRecord, args, trial: int) -> ProbeRecord:
    return ProbeRecord(
        run_id=args.run_id,
        track="terminal",
        suite=record.suite,
        level=record.task_or_level,
        model_requested=args.model,
        provider=args.provider,
        trial=trial,
        question_id=record.question_id,
        probe_type=record.probe_type,
        question=record.question,
        ground_truth=record.ground_truth,
        parse_note="dry_run",
        error="dry_run",
        extra={"answer_schema": record.answer_schema},
    )


def evaluate_record(record: QuestionRecord, args, lm, trial: int) -> ProbeRecord:
    prompt, omitted = build_prompt(record, args.max_context_chars)
    started = time.monotonic()
    try:
        response = lm.query(system=SYSTEM_PROMPT, user=prompt)
    except Exception as exc:  # noqa: BLE001 -- provider SDKs expose unrelated hierarchies
        return ProbeRecord(
            run_id=args.run_id,
            track="terminal",
            suite=record.suite,
            level=record.task_or_level,
            model_requested=args.model,
            provider=args.provider,
            trial=trial,
            question_id=record.question_id,
            probe_type=record.probe_type,
            question=record.question,
            ground_truth=record.ground_truth,
            latency_s=time.monotonic() - started,
            error=_classify_error(exc),
            parse_note=f"{type(exc).__name__}: {exc}",
            extra={"answer_schema": record.answer_schema, "context_chars_omitted": omitted},
        )

    latency = time.monotonic() - started
    reason = exclusion_reason(response)
    if reason:
        return ProbeRecord(
            run_id=args.run_id,
            track="terminal",
            suite=record.suite,
            level=record.task_or_level,
            model=response.model,
            model_requested=args.model,
            provider=args.provider,
            trial=trial,
            question_id=record.question_id,
            probe_type=record.probe_type,
            question=record.question,
            ground_truth=record.ground_truth,
            response=response.text or "",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            latency_s=latency,
            error=reason,
            parse_note="excluded: %s (stop_reason=%s, output_tokens=%s)"
                       % (reason, response.stop_reason or "unreported",
                          response.completion_tokens),
            extra={"answer_schema": record.answer_schema, "context_chars_omitted": omitted,
                   "stop_reason": response.stop_reason},
        )

    grade = grade_answer(
        expected=str(record.ground_truth),
        actual=response.text,
        answer_schema=record.answer_schema,
    )
    return ProbeRecord(
        run_id=args.run_id,
        track="terminal",
        suite=record.suite,
        level=record.task_or_level,
        model=response.model,
        model_requested=args.model,
        provider=args.provider,
        trial=trial,
        question_id=record.question_id,
        probe_type=record.probe_type,
        question=record.question,
        ground_truth=record.ground_truth,
        response=response.text,
        is_correct=grade.correct,
        score=1.0 if grade.correct else 0.0,
        parse_note=grade.match_kind,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_s=latency,
        extra={
            "answer_schema": record.answer_schema,
            "cognitive_tier": record.cognitive_tier,
            "failure_mode_target": record.failure_mode_target,
            "context_chars_omitted": omitted,
            # Structural refusal marker, so counting refusals never has to
            # string-match the response text.
            "stop_reason": response.stop_reason,
            "attempts": response.attempts,
            "grade": asdict(grade),
        },
    )


def _write_outputs(records: list[ProbeRecord], selected: list[QuestionRecord], args) -> None:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    write_jsonl(records, results_path)
    grouped = summarize(records)
    scored = [record for record in records if record.score is not None]
    excluded = [record for record in records if record.score is None]
    summary = {
        "provider": args.provider,
        "model_requested": args.model,
        "selected": len(selected),
        "records": len(records),
        "scored": len(scored),
        "excluded": len(excluded),
        "correct": sum(record.score == 1.0 for record in scored),
        "accuracy": (
            sum(float(record.score) for record in scored) / len(scored) if scored else None
        ),
        "groups": grouped,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = {
        "provider": args.provider,
        "model": args.model,
        "version": args.version,
        "tasks": args.tasks,
        "question_ids": args.question_ids,
        "probe_types": args.probe_types,
        "limit": args.limit,
        "max_tokens": args.max_tokens,
        "max_context_chars": args.max_context_chars,
        "dry_run": args.dry_run,
    }
    reported = sorted({record.model for record in records if record.model})
    manifest = manmod.build(
        args.run_id,
        config,
        track="terminal",
        suite="terminal_llm_generated",
        question_bank={"version": args.version, "selected": len(selected)},
        models=[
            {"requested": args.model, "reported": model, "provider": args.provider}
            for model in (reported or [""])
        ],
        counts={
            "records": len(records),
            "scored": len(scored),
            "excluded": len(excluded),
        },
    )
    manmod.write(manifest, out / "manifest.json")


def run_terminal(args) -> int:
    """Run or dry-run Terminal evaluation and write canonical artifacts."""
    selected = select_records(args)
    if not selected:
        raise TerminalEvaluationError("Terminal selection is empty")
    out = Path(args.out).expanduser().resolve()
    manifest_path = out / "manifest.json"
    resume_manifest = None
    if args.resume and manifest_path.is_file():
        resume_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.run_id = (
        getattr(args, "run_id", None)
        or (resume_manifest or {}).get("run_id")
        or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_terminal"
    )

    existing: list[ProbeRecord] = []
    results_path = out / "results.jsonl"
    if args.resume and results_path.is_file():
        existing = list(read_jsonl(results_path))
        mismatched = [
            record.question_id
            for record in existing
            if record.model_requested != args.model or record.provider != args.provider
        ]
        if mismatched:
            raise TerminalEvaluationError(
                "resume output belongs to a different provider/model; choose a new --out"
            )
    completed = {
        record.question_id for record in existing if record.score is not None
    }
    records_by_id = {record.question_id: record for record in existing}

    lm = None if args.dry_run else _make_lm(args)
    total = len(selected)
    pending = [
        (trial, record)
        for trial, record in enumerate(selected, start=1)
        if record.question_id not in completed
    ]
    if args.verbose:
        for trial, record in enumerate(selected, start=1):
            if record.question_id in completed:
                print(f"[{trial}/{total}] resume {record.question_id}")

    def _one(trial: int, record):
        return (
            _dry_run_record(record, args, trial)
            if args.dry_run
            else evaluate_record(record, args, lm, trial)
        )

    def _report(trial: int, record, result) -> None:
        status = (
            "correct" if result.score == 1.0
            else "wrong" if result.score == 0.0
            else result.error
        )
        print(f"[{trial}/{total}] {status} {record.question_id} {record.task_or_level}")

    workers = max(1, int(getattr(args, "workers", 1) or 1))
    if workers == 1 or args.dry_run:
        for trial, record in pending:
            result = _one(trial, record)
            records_by_id[record.question_id] = result
            _report(trial, record, result)
    else:
        # Records are independent, so they fan out. Output is keyed by
        # question_id and sorted below, so completion order does not leak in.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, trial, record): (trial, record)
                       for trial, record in pending}
            for future in as_completed(futures):
                trial, record = futures[future]
                result = future.result()
                records_by_id[record.question_id] = result
                _report(trial, record, result)

    ordered = [records_by_id[key] for key in sorted(records_by_id)]
    _write_outputs(ordered, selected, args)
    scored = [record for record in ordered if record.score is not None]
    excluded = [record for record in ordered if record.score is None]
    accuracy = sum(float(record.score) for record in scored) / len(scored) if scored else None
    print(f"Terminal results: scored={len(scored)} excluded={len(excluded)} accuracy={accuracy}")
    print(f"Saved: {out / 'results.jsonl'}")
    return 1 if (not args.dry_run and excluded) else 0
