"""Score a model against the frozen chess question bank.

`halluworld eval chess` used to call battery.main(), which *regenerates* the
questions from a config and relies on determinism to land on the same ones the
bank holds. That is fragile in a way that matters for a held-out set: any drift
between the eval config and the config the bank was built under -- a different
BENCHMARK_SEED, a different LICHESS_* draw, a probe knob -- silently scores the
model on different questions than the frozen artifact contains, with nothing in
the output saying so.

This module replays instead: it sends each record's stored prompt verbatim and
grades the stored ground truth, so the questions asked are the questions frozen.
Regeneration stays available behind `--regenerate` because the bank freezes one
observation configuration, and conditions that were never exported (the paper's
fen-transpose arm, say) can only be produced by running the battery.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from halluworld.benchmark import SYSTEM_PROMPT
from halluworld.data import QuestionBankUnavailable, questions_dir
from halluworld.probe import ProbeResult
from halluworld.questions import QuestionRecord, read_bank
from halluworld.results import ProbeRecord, classify_error, exclusion_reason

# Imported, not redefined: neither battery.py nor question_set.py passes a
# system_prompt to run_benchmark, so every chess question in the bank was asked
# under run_benchmark's default. That default is worded for gridworld, which is
# odd for chess -- but replay has to reproduce what the bank was built with, and
# rewording it here would silently change what is being measured.

class ChessReplayError(RuntimeError):
    """The requested replay cannot be run."""


def _evaluator_for(probe_type: str):
    """Map a probe type onto the evaluator the battery pairs it with.

    Kept in step with build_probes_and_evaluators() in battery.py; an unknown
    probe type raises rather than defaulting, because silently grading a
    capture-count question with a yes/no evaluator would score every answer
    wrong and look like a model failure.
    """
    from halluworld.tracks.chess import (
        ChessIntegerEvaluator,
        ChessLegalUciSetEvaluator,
        ChessYesNoEvaluator,
    )

    table = {
        "chess_can_capture": ChessYesNoEvaluator,
        "chess_defended": ChessYesNoEvaluator,
        "chess_hanging": ChessYesNoEvaluator,
        "chess_hypothetical_in_check": ChessYesNoEvaluator,
        "chess_conflicting_prompt": ChessYesNoEvaluator,
        "chess_hidden_side_capture_stats": ChessIntegerEvaluator,
        "chess_after_move_undefended_count": ChessIntegerEvaluator,
        "chess_san_legal_move": ChessLegalUciSetEvaluator,
    }
    if probe_type not in table:
        raise ChessReplayError(
            f"no evaluator known for chess probe type {probe_type!r}; "
            "add it to _evaluator_for() in halluworld/tracks/chess/replay.py"
        )
    return table[probe_type]()


def chess_bank_file(fen_mode: str | None = "off") -> str:
    """The bank file holding one FEN observation condition.

    ``chess.jsonl.gz`` is the No-FEN condition, the default in every version.
    A version that also freezes another condition carries it alongside as
    ``chess_fen_<mode>.jsonl.gz`` under the same manifest (v0.2 freezes
    ``transpose``, the incorrect-FEN arm). Same questions, same positions;
    only the rendered observation differs.
    """
    mode = (fen_mode or "off").strip().lower()
    return "chess.jsonl.gz" if mode == "off" else f"chess_fen_{mode}.jsonl.gz"


def load_chess_bank(version: str = "v0.1", fen_mode: str | None = "off") -> list[QuestionRecord]:
    name = chess_bank_file(fen_mode)
    try:
        path = questions_dir(version) / name
    except QuestionBankUnavailable as exc:
        raise ChessReplayError(str(exc)) from exc
    if not path.exists() and name != "chess.jsonl.gz":
        # Never fall back to the No-FEN file: that would score the wrong
        # condition under the requested label.
        raise ChessReplayError(
            f"question bank {version} has no frozen fen-{fen_mode} chess condition "
            f"({name}). Pass --regenerate to run the battery under "
            f"halluworld/data/configs/chess/fen_{fen_mode}.yaml instead."
        )
    if not path.exists():
        raise ChessReplayError(
            f"no chess question bank for version {version!r} at {path}"
        )
    return list(read_bank(path))


def select_records(args, version: str) -> list[QuestionRecord]:
    """Bank records for the requested probe types, capped by --episodes."""
    records = load_chess_bank(version, getattr(args, "fen_mode", "off"))
    wanted = set(args.levels) if args.levels else None
    if wanted:
        unknown = wanted - {r.probe_type for r in records}
        if unknown:
            raise ChessReplayError(
                "unknown chess probe type(s): %s" % ", ".join(sorted(unknown))
            )
        records = [r for r in records if r.probe_type in wanted]

    limit = getattr(args, "limit", None)
    per_type = args.episodes if limit is None else min(args.episodes, limit)

    seen: dict[str, int] = {}
    kept = []
    for record in records:
        n = seen.get(record.probe_type, 0)
        if n >= per_type:
            continue
        seen[record.probe_type] = n + 1
        kept.append(record)
    return kept


def _probe_result(record: QuestionRecord) -> ProbeResult:
    """Rebuild the ProbeResult the evaluators expect from a frozen record.

    metadata carries fields some evaluators read directly --
    ChessLegalUciSetEvaluator looks for `legal_ucis` there -- so `extra`, which
    is where the bank stores the original probe metadata, has to travel with it.
    """
    return ProbeResult(
        probe_type=record.probe_type,
        question=record.question,
        ground_truth=record.ground_truth,
        metadata=dict(record.extra or {}),
    )


def _make_lm(args):
    """Delegates to the shared factory; kept as a name for tests that patch it."""
    from halluworld.lm.factory import LMConfigError, make_lm

    try:
        return make_lm(args)
    except LMConfigError as exc:
        raise ChessReplayError(str(exc)) from exc


def evaluate_record(record: QuestionRecord, args, lm, version: str) -> ProbeRecord:
    """Ask one frozen question and grade the answer."""
    base = dict(
        run_id=getattr(args, "run_id", ""),
        track="chess",
        suite=record.suite,
        level=record.task_or_level,
        model_requested=args.model,
        provider=args.provider,
        question_id=record.question_id,
        probe_type=record.probe_type,
        question=record.question,
        ground_truth=record.ground_truth,
    )
    started = time.monotonic()
    try:
        # record.context is the exact model-visible prompt, question included.
        response = lm.query(system=SYSTEM_PROMPT, user=record.context)
    except Exception as exc:  # noqa: BLE001 -- provider SDKs expose unrelated hierarchies
        return ProbeRecord(
            **base,
            latency_s=time.monotonic() - started,
            error=classify_error(exc),
            parse_note=f"{type(exc).__name__}: {exc}",
            extra={"bank_version": version},
        )
    latency = time.monotonic() - started

    # A provider-side failure is an exclusion, not a wrong answer: score stays
    # None so it never lands in a denominator. Mirrors run_benchmark().
    if response.error:
        return ProbeRecord(
            **base,
            model=response.model,
            response=response.text,
            latency_s=latency,
            error=response.error,
            extra={**{"bank_version": version}, "stop_reason": response.stop_reason},
        )

    # No answer is not a wrong answer. A model that spends its whole budget on
    # internal reasoning returns empty text; grading that as 0 records a
    # hallucination the model never made. Terminal already excludes these, and
    # the project settled the principle with BadRequestError: score=None means
    # excluded, score=0 means wrong.
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
            extra={"bank_version": version, "stop_reason": response.stop_reason},
        )

    evaluation = _evaluator_for(record.probe_type).evaluate(response, _probe_result(record))
    return ProbeRecord(
        **base,
        model=response.model,
        response=response.text,
        is_correct=evaluation.correct,
        score=evaluation.score,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        latency_s=latency,
        parse_note=(evaluation.details or {}).get("parse_note", ""),
        extra={"bank_version": version, "stop_reason": response.stop_reason,
               "attempts": response.attempts, **(evaluation.details or {})},
    )


def run_chess_replay(args, version: str, out: Path,
                     skip_ids: set[str] | None = None) -> list[ProbeRecord]:
    """Replay the selected bank records and return the graded results.

    Records are independent -- each is one prompt graded against its own frozen
    ground truth -- so --workers fans them out across threads. Results are
    returned in bank order regardless of completion order, so a run is
    byte-comparable to a sequential one.

    The provider client is shared across threads; the OpenAI and Anthropic SDKs
    document their clients as thread-safe. Concurrency here multiplies the
    request rate against your account's rate limit, so the useful ceiling is
    the account's, not the machine's.
    """
    selected = select_records(args, version)
    if skip_ids:
        selected = [r for r in selected if r.question_id not in skip_ids]
        if not selected:
            return []
    lm = _make_lm(args)
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    total = len(selected)
    verbose = getattr(args, "verbose", False)

    if workers == 1:
        results = []
        for index, record in enumerate(selected, start=1):
            results.append(evaluate_record(record, args, lm, version))
            if verbose:
                print("[%d/%d] %s" % (index, total, record.question_id), flush=True)
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    ordered: list[ProbeRecord | None] = [None] * total
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(evaluate_record, record, args, lm, version): position
            for position, record in enumerate(selected)
        }
        for future in as_completed(futures):
            position = futures[future]
            ordered[position] = future.result()
            done += 1
            if verbose:
                print("[%d/%d] %s" % (done, total, selected[position].question_id), flush=True)
    return [r for r in ordered if r is not None]
