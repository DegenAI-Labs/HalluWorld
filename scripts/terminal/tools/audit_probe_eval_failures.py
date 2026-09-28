#!/usr/bin/env python3
"""
Audit probe eval failures against the visible extracted context.

This is a heuristic triage pass. It does not ask another model. For each
incorrect result, it tries to re-derive the answer from the visible terminal
context and classifies the failure as likely genuine, likely a golden/context
issue, a container-state probe, or ambiguous.

Example:
  python3 tools/audit_probe_eval_failures.py \
    --probe-dir runs/all-tasks__20260503_015318/extracted_probes \
    --result-dir runs/all-tasks__20260503_015318/gpt-5.4_probe_eval \
    --out-json /tmp/probe_failure_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import shlex
from collections import Counter
from pathlib import Path
from typing import Any


ASSIGNMENT_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z0-9_./:+-]+)"
)
PROMPT_RE = re.compile(r"(?:^|\n)[^\n#]*# ")
LIVE_CONTAINER_INJECTORS = {
    "file_exists",
    "content_contains",
    "program_available",
    "port_listening",
    "http_status",
    "periodic_task_exists",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_assignments(text: str | None) -> dict[str, str]:
    return {
        key: value.strip("`'\".,;").lower()
        for key, value in ASSIGNMENT_RE.findall(text or "")
    }


def first_assignment_value(text: str | None) -> str | None:
    assignments = parse_assignments(text)
    if not assignments:
        return None
    return next(iter(assignments.values()))


def command_program(command: str) -> str | None:
    command = command.replace("; tmux wait -S done", "").strip()
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return None
    if tokens[0] in {"sudo", "env", "time", "timeout"} and len(tokens) > 1:
        return Path(tokens[1]).name
    if tokens[0] in {"cd", "exit", "clear", "asciinema", "tmux"}:
        return None
    if tokens[0].startswith(("#", "-", ">", "<")):
        return None
    return Path(tokens[0]).name


def visible_records(context: str) -> list[tuple[str, str]]:
    matches = list(PROMPT_RE.finditer(context or ""))
    records: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        command_start = match.end()
        command_end = context.find("\n", command_start)
        if command_end < 0:
            command_end = len(context)
        next_prompt = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        command = context[command_start:command_end].strip()
        output = context[command_end:next_prompt]
        if command:
            records.append((command, output))
    return records


def visible_previous_program(
    *, context: str, program: str, limit: int | None
) -> str:
    records = visible_records(context)
    if limit:
        records = records[-limit:]
    seen = any(command_program(command) == program for command, _ in records)
    return "yes" if seen else "no"


def visible_previous_output_contains(
    *, context: str, needle: str, limit: int | None
) -> str:
    records = visible_records(context)
    if limit:
        records = records[-limit:]
    seen = any(needle in output for _, output in records)
    return "yes" if seen else "no"


def visible_program_availability_evidence(
    *, context: str, program: str
) -> str:
    for command, output in reversed(visible_records(context)):
        if command_program(command) != program:
            continue
        if "command not found" in output:
            return "known_no"
        return "known_yes"
    return "unknown"


def visible_error_source_program(
    *, context: str, needle: str
) -> str | None:
    for command, output in reversed(visible_records(context)):
        if needle in output:
            return command_program(command) or "unknown"
    return None


def visible_compound_previous_program_and_error(
    *, context: str, program: str, needle: str
) -> str:
    saw_program = visible_previous_program(
        context=context,
        program=program,
        limit=None,
    ) == "yes"
    saw_error = visible_previous_output_contains(
        context=context,
        needle=needle,
        limit=None,
    ) == "yes"
    return "yes" if saw_program and saw_error else "no"


def extract_backtick_after(pattern: str, question: str) -> str | None:
    match = re.search(pattern, question)
    return match.group(1) if match else None


def derive_visible_value(probe: dict[str, Any]) -> tuple[str | None, str]:
    metadata = probe.get("metadata") or {}
    injector = metadata.get("injector")
    question = probe.get("question") or ""
    context = probe.get("context") or ""

    if injector in LIVE_CONTAINER_INJECTORS:
        return None, "container_state_probe"

    if injector in {"memory_previous_program", "memory_previous_program_last_10"}:
        program = extract_backtick_after(r"invoking `([^`]+)`", question)
        if not program:
            return None, "unparsed_question"
        limit = 10 if injector.endswith("_last_10") else None
        return visible_previous_program(
            context=context,
            program=program,
            limit=limit,
        ), "visible_command_history"

    if injector in {"memory_previous_output", "memory_previous_output_last_10"}:
        needle = extract_backtick_after(r"exact text `([^`]+)`", question)
        if not needle:
            return None, "unparsed_question"
        limit = 10 if injector.endswith("_last_10") else None
        return visible_previous_output_contains(
            context=context,
            needle=needle,
            limit=limit,
        ), "visible_output_history"

    if injector == "uncertainty_program_availability_evidence":
        program = extract_backtick_after(r"whether `([^`]+)`", question)
        if not program:
            return None, "unparsed_question"
        return visible_program_availability_evidence(
            context=context,
            program=program,
        ), "visible_program_evidence"

    if injector == "causal_error_source_program":
        needle = extract_backtick_after(r"containing `([^`]+)`", question)
        if not needle:
            return None, "unparsed_question"
        return visible_error_source_program(
            context=context,
            needle=needle,
        ), "visible_latest_error_output"

    if injector == "compound_previous_program_and_error":
        program = extract_backtick_after(r"invoking `([^`]+)`", question)
        needle = extract_backtick_after(r"exact text `([^`]+)`", question)
        if not program or not needle:
            return None, "unparsed_question"
        return visible_compound_previous_program_and_error(
            context=context,
            program=program,
            needle=needle,
        ), "visible_compound_history"

    return None, "unsupported_injector"


def classify_failure(
    *, visible_value: str | None, expected_value: str | None, actual_value: str | None, basis: str
) -> str:
    if basis == "container_state_probe":
        return "container_state_probe"
    if visible_value is None:
        return "no_visible_derivation"
    if expected_value is None or actual_value is None:
        return "unparsed_assignment"
    if visible_value == expected_value and actual_value != expected_value:
        return "likely_genuine_model_error"
    if visible_value == actual_value and expected_value != actual_value:
        return "likely_golden_or_context_issue"
    if visible_value != expected_value and visible_value != actual_value:
        return "visible_disagrees_with_both"
    return "ambiguous"


def matching_probe_file(probe_dir: Path, result_file: Path) -> Path:
    name = result_file.name
    if not name.endswith(".result.json"):
        raise ValueError(f"not a result file: {result_file}")
    return probe_dir / f"{name[:-len('.result.json')]}.json"


def audit(probe_dir: Path, result_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    by_injector: Counter[tuple[str, str]] = Counter()
    total = 0
    incorrect = 0

    for result_file in sorted(result_dir.glob("*.result.json")):
        total += 1
        result = load_json(result_file)
        if result.get("correct") is not False:
            continue
        incorrect += 1

        probe_file = matching_probe_file(probe_dir, result_file)
        probe = load_json(probe_file)
        metadata = probe.get("metadata") or {}
        injector = metadata.get("injector") or "unknown"

        visible_value, basis = derive_visible_value(probe)
        expected_value = first_assignment_value(probe.get("answer"))
        actual_value = first_assignment_value(result.get("model_answer"))
        classification = classify_failure(
            visible_value=visible_value,
            expected_value=expected_value,
            actual_value=actual_value,
            basis=basis,
        )

        counts[classification] += 1
        by_injector[(classification, injector)] += 1
        rows.append(
            {
                "result_file": result_file.name,
                "probe_file": probe_file.name,
                "classification": classification,
                "basis": basis,
                "injector": injector,
                "probe_type": metadata.get("probe_type"),
                "question": probe.get("question"),
                "golden": probe.get("answer"),
                "model_answer": result.get("model_answer"),
                "visible_value": visible_value,
                "expected_value": expected_value,
                "actual_value": actual_value,
                "trigger_command": metadata.get("trigger_command"),
                "task_name": metadata.get("task_name"),
            }
        )

    return {
        "result_dir": str(result_dir),
        "probe_dir": str(probe_dir),
        "total_results": total,
        "incorrect_results": incorrect,
        "counts": dict(counts),
        "by_injector": [
            {
                "classification": classification,
                "injector": injector,
                "count": count,
            }
            for (classification, injector), count in sorted(
                by_injector.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
        "failures": rows,
    }


def export_classification(
    *,
    report: dict[str, Any],
    probe_dir: Path,
    result_dir: Path,
    out_dir: Path,
    classification: str,
    overwrite: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in out_dir.glob("*.json"):
            path.unlink()

    count = 0
    for row in report["failures"]:
        if row["classification"] != classification:
            continue

        count += 1
        stem = f"{count:04d}_{Path(row['probe_file']).stem}"
        probe_out = out_dir / f"{stem}.probe.json"
        result_out = out_dir / f"{stem}.result.json"
        audit_out = out_dir / f"{stem}.audit.json"

        if not overwrite and (
            probe_out.exists() or result_out.exists() or audit_out.exists()
        ):
            raise FileExistsError(
                f"output exists for {stem}; pass --overwrite-export to replace"
            )

        shutil.copy2(probe_dir / row["probe_file"], probe_out)
        shutil.copy2(result_dir / row["result_file"], result_out)
        write_json(audit_out, row)

    summary = {
        "classification": classification,
        "count": count,
        "probe_dir": str(probe_dir),
        "result_dir": str(result_dir),
    }
    write_json(out_dir / "summary.json", summary)
    return count


def export_probe_benchmark(
    *,
    report: dict[str, Any],
    probe_dir: Path,
    out_dir: Path,
    classification: str,
    overwrite: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in out_dir.glob("*.json"):
            path.unlink()

    count = 0
    selected_probe_files: list[str] = []

    for row in report["failures"]:
        if row["classification"] != classification:
            continue

        count += 1
        source_name = row["probe_file"]
        target_name = f"{count:06d}_{source_name}"
        target = out_dir / target_name

        if target.exists() and not overwrite:
            raise FileExistsError(
                f"output exists: {target}; pass --overwrite-export to replace"
            )

        shutil.copy2(probe_dir / source_name, target)
        selected_probe_files.append(target_name)

    summary = {
        "classification": classification,
        "count": count,
        "source_probe_dir": str(probe_dir),
        "probe_files": selected_probe_files,
    }
    write_json(out_dir / "summary.json", summary)
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Optional directory to export failures matching --export-classification.",
    )
    parser.add_argument(
        "--export-probe-dir",
        type=Path,
        help=(
            "Optional eval-compatible probe directory containing only probes "
            "matching --export-classification."
        ),
    )
    parser.add_argument(
        "--export-classification",
        default="likely_genuine_model_error",
        help="Failure classification to export when --export-dir is set.",
    )
    parser.add_argument(
        "--overwrite-export",
        action="store_true",
        help="Overwrite files in --export-dir if they already exist.",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of examples to print per classification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args.probe_dir, args.result_dir)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out_json, report)
    if args.export_dir:
        exported = export_classification(
            report=report,
            probe_dir=args.probe_dir,
            result_dir=args.result_dir,
            out_dir=args.export_dir,
            classification=args.export_classification,
            overwrite=args.overwrite_export,
        )
        print(
            f"exported_{args.export_classification}={exported} "
            f"export_dir={args.export_dir}"
        )
    if args.export_probe_dir:
        exported = export_probe_benchmark(
            report=report,
            probe_dir=args.probe_dir,
            out_dir=args.export_probe_dir,
            classification=args.export_classification,
            overwrite=args.overwrite_export,
        )
        print(
            f"exported_probe_{args.export_classification}={exported} "
            f"export_probe_dir={args.export_probe_dir}"
        )

    print(f"total_results={report['total_results']}")
    print(f"incorrect_results={report['incorrect_results']}")
    for classification, count in sorted(
        report["counts"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"{classification}={count}")

    if args.show_examples > 0:
        seen: Counter[str] = Counter()
        print("\nexamples:")
        for row in report["failures"]:
            classification = row["classification"]
            if seen[classification] >= args.show_examples:
                continue
            seen[classification] += 1
            print(
                f"- {classification}: {row['result_file']} "
                f"injector={row['injector']} visible={row['visible_value']} "
                f"gold={row['expected_value']} model={row['actual_value']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
