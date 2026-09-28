"""Offline release gates for the frozen Grid, Chess, InNav, and Terminal domains.

``questions verify`` proves that bytes match the manifest.  This module goes
one step further: it proves that the installed runtime can still interpret
those bytes.  Each domain has a different frozen unit, so its wiring check is
necessarily domain-specific:

* Grid compares every banked probe with ``perception.make_probes``.
* Chess sends every golden answer through the evaluator used by the battery.
* InNav compares the generated-probe schedule with ``make_canonical_probes``
  and validates that every released trajectory contains enough actions to
  reconstruct every state selected by the released probing policy.
* Terminal self-grades every golden answer with the conservative production
  scorer and checks that structured answers cannot be accepted by prefix.

All checks are offline.  They do not instantiate a provider client or make a
model request, which makes this command suitable for CI and wheel smoke tests.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halluworld.data import questions_dir
from halluworld.questions import QuestionRecord, read_bank, verify_bank

DOMAINS = ("grid", "chess", "innav", "terminal")


class ReleaseContractError(RuntimeError):
    """A frozen artifact no longer agrees with its runtime implementation."""


@dataclass(frozen=True)
class DomainReport:
    domain: str
    questions: int
    levels: int
    checks: tuple[str, ...] = field(default_factory=tuple)
    details: dict[str, Any] = field(default_factory=dict)


def bank_dir(version: str = "v0.1") -> Path:
    return questions_dir(version)


def _manifest(version: str) -> dict:
    path = bank_dir(version) / "manifest.json"
    if not path.exists():
        raise ReleaseContractError(f"missing release manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("bank_version") != version:
        raise ReleaseContractError(
            "{} declares bank_version={!r}".format(path, manifest.get("bank_version"))
        )
    return manifest


def _records(domain: str, version: str, manifest: dict) -> list[QuestionRecord]:
    try:
        entry = manifest["tracks"][domain]
    except KeyError as exc:
        raise ReleaseContractError(
            f"{version} does not contain the {domain} domain"
        ) from exc
    path = bank_dir(version) / entry["file"]
    ok, message = verify_bank(path, entry)
    if not ok:
        raise ReleaseContractError(message)
    records = list(read_bank(path))
    if len(records) != entry["count"]:
        raise ReleaseContractError(
            f"{domain}: manifest declares {entry['count']} records, "
            f"loader returned {len(records)}"
        )
    return records


def _by_level(records: Iterable[QuestionRecord]) -> dict[str, list[QuestionRecord]]:
    grouped: dict[str, list[QuestionRecord]] = {}
    for record in records:
        grouped.setdefault(record.task_or_level, []).append(record)
    for level_records in grouped.values():
        level_records.sort(key=lambda record: record.question_id)
    return grouped


def check_grid(version: str = "v0.1") -> DomainReport:
    """Check frozen Grid records against the live level and probe dispatch."""
    manifest = _manifest(version)
    records = _records("grid", version, manifest)

    from halluworld.tracks.grid import perception
    from halluworld.tracks.grid.probes.visibility import FixedProbe

    grouped = _by_level(records)
    if set(grouped) != set(perception.LEVELS):
        bank_only = sorted(set(grouped) - set(perception.LEVELS))
        runtime_only = sorted(set(perception.LEVELS) - set(grouped))
        raise ReleaseContractError(
            f"grid level mismatch: bank-only={bank_only} runtime-only={runtime_only}"
        )

    fixed = generated = 0
    for level, banked in sorted(grouped.items()):
        level_path = Path(perception.LEVELS[level])
        if not level_path.is_file():
            raise ReleaseContractError(f"grid level file is missing: {level_path}")
        live = perception.make_probes(level, random.Random(0))
        if len(live) != len(banked):
            raise ReleaseContractError(
                f"{level}: bank has {len(banked)} probes, "
                f"runtime produced {len(live)}"
            )
        for probe, record in zip(live, banked):
            if isinstance(probe, FixedProbe):
                expected = (record.probe_type, record.question, record.ground_truth)
                actual = (probe._probe_type, probe._question, probe._ground_truth)
                if record.kind != "fixed" or actual != expected:
                    raise ReleaseContractError(
                        f"{record.question_id} no longer matches the live FixedProbe"
                    )
                fixed += 1
            else:
                if (
                    record.kind != "generated"
                    or type(probe).__name__ != record.probe_class
                ):
                    raise ReleaseContractError(
                        f"{record.question_id} expects {record.probe_class}, "
                        f"runtime produced {type(probe).__name__}"
                    )
                generated += 1

    return DomainReport(
        domain="grid",
        questions=len(records),
        levels=len(grouped),
        checks=(
            "manifest",
            "level files",
            "live probe dispatch",
            "fixed probe contents",
        ),
        details={"fixed": fixed, "generated": generated},
    )


_CHESS_EVALUATORS = {
    "chess_can_capture": "yes_no",
    "chess_defended": "yes_no",
    "chess_hanging": "yes_no",
    "chess_hypothetical_in_check": "yes_no",
    "chess_after_move_undefended_count": "integer",
    "chess_hidden_side_capture_stats": "integer",
    "chess_san_legal_move": "legal_uci",
}


def check_chess(version: str = "v0.1") -> DomainReport:
    """Check every frozen Chess item with the battery's production evaluator."""
    manifest = _manifest(version)
    records = _records("chess", version, manifest)

    try:
        from halluworld.lm.base import LMResponse
        from halluworld.probe import ProbeResult
        from halluworld.tracks.chess.evaluators import (
            ChessIntegerEvaluator,
            ChessLegalUciSetEvaluator,
            ChessYesNoEvaluator,
        )
    except ImportError as exc:
        raise ReleaseContractError(
            "the Chess release check requires the chess extra: "
            "pip install 'halluworld[chess]'"
        ) from exc

    evaluators = {
        "yes_no": ChessYesNoEvaluator(),
        "integer": ChessIntegerEvaluator(),
        "legal_uci": ChessLegalUciSetEvaluator(),
    }
    counts: dict[str, int] = {}
    for record in records:
        evaluator_name = _CHESS_EVALUATORS.get(record.probe_type)
        if evaluator_name is None:
            raise ReleaseContractError(
                f"{record.probe_type} has no released evaluator mapping"
            )
        if not record.context or record.question not in record.context:
            raise ReleaseContractError(
                f"{record.question_id} does not contain its question in the frozen context"
            )
        if evaluator_name == "legal_uci":
            legal = record.extra.get("legal_ucis") or record.ground_truth
            if not legal:
                raise ReleaseContractError(
                    f"{record.question_id} has no legal UCI answer"
                )
            golden = str(legal[0])
        elif evaluator_name == "yes_no":
            golden = "yes" if record.ground_truth else "no"
        else:
            golden = str(record.ground_truth)

        probe_result = ProbeResult(
            probe_type=record.probe_type,
            question=record.question,
            ground_truth=record.ground_truth,
            metadata=record.extra,
        )
        result = evaluators[evaluator_name].evaluate(
            LMResponse(text=golden, model="release-contract-golden"), probe_result
        )
        if not result.correct or result.score != 1.0:
            raise ReleaseContractError(
                f"{record.question_id} golden answer failed {evaluator_name}"
            )
        counts[record.probe_type] = counts.get(record.probe_type, 0) + 1

    if set(counts) != set(_CHESS_EVALUATORS):
        raise ReleaseContractError(
            f"chess probe mismatch: expected {sorted(_CHESS_EVALUATORS)}, "
            f"found {sorted(counts)}"
        )

    return DomainReport(
        domain="chess",
        questions=len(records),
        levels=len(counts),
        checks=(
            "manifest",
            "frozen contexts",
            "evaluator mapping",
            "golden self-grade",
        ),
        details={"by_probe": counts},
    )


def read_released_trajectories(version: str = "v0.1") -> list[dict]:
    """Load the packaged InNav trajectory artifact and verify its checksum."""
    manifest = _manifest(version)
    try:
        entry = manifest["trajectories"]
    except KeyError as exc:
        raise ReleaseContractError(f"{version} has no trajectory artifact") from exc
    path = bank_dir(version) / entry["file"]
    ok, message = verify_bank(path, entry)
    if not ok:
        raise ReleaseContractError(message)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if len(records) != entry["count"]:
        raise ReleaseContractError(
            f"trajectory manifest declares {entry['count']} records, "
            f"loader returned {len(records)}"
        )
    return records


def released_trajectory_actions(record: dict) -> list[int]:
    """Extract the replayable action prefix from either released trace shape.

    Three early traces carry a rich ``trajectory`` list.  Later traces carry
    the navigation dialogue.  A historical snapshot bug omitted the final
    assistant turn from some dialogues; that final action is not recoverable,
    but the released probe policy skips the endpoint.  The contract below
    verifies that every selected probe state remains reconstructable.
    """
    from halluworld.utils.action_parsing import parse_action

    saved_actions = record.get("action_sequence")
    trajectory = record.get("trajectory") or []
    if saved_actions is not None:
        raw_actions = saved_actions
    elif trajectory:
        raw_actions = [
            step.get("action", step.get("action_text", step.get("action_chosen")))
            for step in trajectory
        ]
    else:
        raw_actions = [
            message.get("content", "")
            for message in (record.get("messages") or [])
            if message.get("role") == "assistant"
        ]

    actions: list[int] = []
    for raw in raw_actions:
        if isinstance(raw, int) and 0 <= raw <= 5:
            action = raw
        else:
            action = parse_action(str(raw), use_llm_fallback=False)
        if action is None:
            raise ReleaseContractError(
                "{} contains an unparseable action: {!r}".format(
                    record.get("file"), raw
                )
            )
        actions.append(action)
    return actions


def selected_probe_timesteps(total_steps: int, probe_timesteps: int = 5) -> list[int]:
    """The released InNav endpoint-skipping timestep policy, without an engine."""
    if total_steps <= 0:
        return []
    n_probes = min(probe_timesteps, total_steps // 3)
    return [
        int(total_steps * index / (n_probes + 1)) for index in range(1, n_probes + 1)
    ]


def check_innav(version: str = "v0.1") -> DomainReport:
    """Check InNav's probe dispatch, runner levels, configs, and trace replay."""
    manifest = _manifest(version)
    records = _records("innav", version, manifest)

    from halluworld.tracks.innav.canonical_probes import make_canonical_probes
    from halluworld.tracks.innav.runner import LEVELS as RUNNER_LEVELS

    grouped = _by_level(records)
    if set(grouped) != set(RUNNER_LEVELS):
        bank_only = sorted(set(grouped) - set(RUNNER_LEVELS))
        runtime_only = sorted(set(RUNNER_LEVELS) - set(grouped))
        raise ReleaseContractError(
            f"innav level mismatch: bank-only={bank_only} runtime-only={runtime_only}"
        )

    for level, banked in sorted(grouped.items()):
        path = Path(RUNNER_LEVELS[level])
        if not path.is_file():
            raise ReleaseContractError(f"innav level file is missing: {path}")
        config_path = path.with_suffix(".innav.json")
        if not config_path.is_file():
            raise ReleaseContractError(f"innav config is missing: {config_path}")
        live = make_canonical_probes(level, random.Random(0))
        actual = [type(probe).__name__ for probe in live]
        expected = [record.probe_class for record in banked]
        if actual != expected:
            raise ReleaseContractError(
                f"{level}: bank expects {expected}, runtime produced {actual}"
            )

    trajectories = read_released_trajectories(version)
    complete = endpoint_omitted = 0
    for record in trajectories:
        actions = released_trajectory_actions(record)
        steps = record.get("steps_taken")
        if not isinstance(steps, int) or steps < 0:
            raise ReleaseContractError(
                "{} has invalid steps_taken={!r}".format(record.get("file"), steps)
            )
        missing = steps - len(actions)
        if missing == 0:
            complete += 1
        elif missing == 1:
            endpoint_omitted += 1
        else:
            raise ReleaseContractError(
                f"{record.get('file')} declares {steps} steps but only "
                f"{len(actions)} actions are replayable"
            )
        selected = selected_probe_timesteps(steps)
        if selected and max(selected) >= len(actions):
            raise ReleaseContractError(
                f"{record.get('file')} cannot reconstruct selected timestep "
                f"{max(selected)} from {len(actions)} actions"
            )

    return DomainReport(
        domain="innav",
        questions=len(records),
        levels=len(grouped),
        checks=(
            "manifest",
            "level files and configs",
            "live probe dispatch",
            "released trajectory replay",
        ),
        details={
            "trajectories": len(trajectories),
            "complete_action_sequences": complete,
            "endpoint_action_omitted": endpoint_omitted,
        },
    )


def _terminal_truncations(answer: str) -> list[str]:
    """Generate known-dangerous structured prefixes for release regression."""
    candidates: list[str] = []
    for delimiter in (",", ";", "->", "|"):
        if delimiter in answer:
            prefix = answer.split(delimiter, 1)[0]
            if prefix and prefix != answer:
                candidates.append(prefix)
    return candidates


def check_terminal(version: str = "v0.1") -> DomainReport:
    """Check the packaged Terminal bank against its production answer grader."""
    manifest = _manifest(version)
    records = _records("terminal", version, manifest)

    from halluworld.tracks.terminal.evaluation import grade_answer, schema_keys

    tasks = {record.task_or_level for record in records}
    ids = {record.question_id for record in records}
    if len(ids) != len(records):
        raise ReleaseContractError("terminal question IDs are not unique")

    by_probe: dict[str, int] = {}
    risky = 0
    for record in records:
        if record.kind != "fixed" or not record.question or not record.context:
            raise ReleaseContractError(
                f"{record.question_id} is not a complete fixed Terminal probe"
            )
        if not schema_keys(record.answer_schema):
            raise ReleaseContractError(
                f"{record.question_id} has no machine-readable answer schema"
            )
        golden = grade_answer(
            expected=str(record.ground_truth),
            actual=str(record.ground_truth),
            answer_schema=record.answer_schema,
        )
        if not golden.correct:
            raise ReleaseContractError(
                f"{record.question_id} golden answer failed Terminal self-grade"
            )
        for truncated in _terminal_truncations(str(record.ground_truth)):
            risky += 1
            result = grade_answer(
                expected=str(record.ground_truth),
                actual=truncated,
                answer_schema=record.answer_schema,
            )
            if result.correct:
                raise ReleaseContractError(
                    f"{record.question_id} accepted truncated answer {truncated!r}"
                )
        by_probe[record.probe_type] = by_probe.get(record.probe_type, 0) + 1

    if len(records) != 529 or len(tasks) != 110:
        raise ReleaseContractError(
            f"terminal v0.1 expects 529 probes across 110 tasks, "
            f"found {len(records)} across {len(tasks)}"
        )

    return DomainReport(
        domain="terminal",
        questions=len(records),
        levels=len(tasks),
        checks=(
            "manifest",
            "frozen contexts",
            "golden self-grade",
            "structured-prefix rejection",
        ),
        details={"tasks": len(tasks), "by_probe": by_probe, "truncations": risky},
    )


def check_release(
    domains: Iterable[str] = DOMAINS, version: str = "v0.1"
) -> list[DomainReport]:
    """Run the selected offline domain checks, in canonical domain order."""
    requested = tuple(dict.fromkeys(domains))
    unknown = set(requested) - set(DOMAINS)
    if unknown:
        raise ReleaseContractError(f"unknown release domain(s): {sorted(unknown)}")
    checks = {
        "grid": check_grid,
        "chess": check_chess,
        "innav": check_innav,
        "terminal": check_terminal,
    }
    return [checks[domain](version) for domain in DOMAINS if domain in requested]
