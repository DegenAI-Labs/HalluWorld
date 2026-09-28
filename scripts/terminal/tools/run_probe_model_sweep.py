#!/usr/bin/env python3
"""Legacy sweep driver for historical extracted-probe artifacts.

For released-bank runs use ``halluworld terminal eval``. This script remains for reproducing the
old artifact layout only.

The script invokes tools/evaluate_probe_model.py once per model, writes each
model's normal eval output into its own directory, then aggregates the
summary.json and result usage into JSON, CSV, and Markdown tables.

Example:
  python3 tools/run_probe_model_sweep.py \
    --probe-dir clean_probes_from_4o-mini \
    --out-root runs/clean_probes_from_4o-mini_model_sweep \
    -n 8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_CONFIGS: list[dict[str, Any]] = [
    {"model": "gpt-5.4", "reasoning_effort": "none", "max_output_tokens": 256},
    {"model": "gpt-5.2", "reasoning_effort": "none", "max_output_tokens": 256},
    {"model": "gpt-5.1", "reasoning_effort": "none", "max_output_tokens": 256},
    {"model": "gpt-5", "reasoning_effort": "minimal", "max_output_tokens": 1024},
    {"model": "gpt-5-mini", "reasoning_effort": "minimal", "max_output_tokens": 512},
    {"model": "gpt-5-nano", "reasoning_effort": "minimal", "max_output_tokens": 512},
    {"model": "gpt-4.1", "max_output_tokens": 256},
    {"model": "gpt-4.1-mini", "max_output_tokens": 256},
    {"model": "gpt-4o", "max_output_tokens": 256},
    {"model": "gpt-4o-mini", "max_output_tokens": 256},
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"


def selected_model_configs(models: list[str] | None) -> list[dict[str, Any]]:
    if not models:
        return [dict(config) for config in DEFAULT_MODEL_CONFIGS]

    defaults = {config["model"]: dict(config) for config in DEFAULT_MODEL_CONFIGS}
    selected: list[dict[str, Any]] = []
    for model in models:
        selected.append(defaults.get(model, {"model": model, "max_output_tokens": 512}))
    return selected


def result_path_for_model(out_root: Path, model: str) -> Path:
    return out_root / f"{safe_slug(model)}_eval"


def build_eval_command(
    *,
    python: str,
    evaluate_script: Path,
    probe_dir: Path,
    out_dir: Path,
    config: dict[str, Any],
    concurrent_n: int,
    limit: int | None,
    api_base: str | None,
    timeout_sec: float | None,
    retries: int | None,
    resume: bool,
    dry_run: bool,
    fail_fast: bool,
) -> list[str]:
    command = [
        python,
        str(evaluate_script),
        "--probe-dir",
        str(probe_dir),
        "--out-dir",
        str(out_dir),
        "--model",
        str(config["model"]),
        "-n",
        str(concurrent_n),
        "--max-output-tokens",
        str(config.get("max_output_tokens", 256)),
    ]
    reasoning_effort = config.get("reasoning_effort")
    if reasoning_effort:
        command.extend(["--reasoning-effort", str(reasoning_effort)])
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if api_base:
        command.extend(["--api-base", api_base])
    if timeout_sec is not None:
        command.extend(["--timeout-sec", str(timeout_sec)])
    if retries is not None:
        command.extend(["--retries", str(retries)])
    if resume:
        command.append("--resume")
    if dry_run:
        command.append("--dry-run")
    if fail_fast:
        command.append("--fail-fast")
    return command


def summarize_result_files(out_dir: Path) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "empty_answers": 0,
    }
    for result_file in out_dir.glob("*.result.json"):
        result = load_json(result_file)
        if result.get("status") == "ok" and not str(result.get("model_answer") or "").strip():
            totals["empty_answers"] += 1

        usage = result.get("usage") or {}
        totals["input_tokens"] += int(usage.get("input_tokens") or 0)
        totals["output_tokens"] += int(usage.get("output_tokens") or 0)
        totals["total_tokens"] += int(usage.get("total_tokens") or 0)

        input_details = usage.get("input_tokens_details") or {}
        output_details = usage.get("output_tokens_details") or {}
        totals["cached_tokens"] += int(input_details.get("cached_tokens") or 0)
        totals["reasoning_tokens"] += int(output_details.get("reasoning_tokens") or 0)
    return totals


def row_from_summary(
    *,
    model: str,
    out_dir: Path,
    returncode: int | None,
) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    usage = summarize_result_files(out_dir)
    row: dict[str, Any] = {
        "model": model,
        "out_dir": str(out_dir),
        "returncode": returncode,
        "total": None,
        "completed": None,
        "errors": None,
        "correct": None,
        "accuracy": None,
        **usage,
    }
    if summary_path.exists():
        summary = load_json(summary_path)
        for key in ["total", "completed", "errors", "correct", "accuracy"]:
            row[key] = summary.get(key)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "total",
        "completed",
        "errors",
        "correct",
        "accuracy",
        "empty_answers",
        "input_tokens",
        "cached_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
        "returncode",
        "out_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        ("model", "model", "left"),
        ("correct", "correct", "right"),
        ("total", "total", "right"),
        ("accuracy", "accuracy", "right"),
        ("errors", "errors", "right"),
        ("empty", "empty_answers", "right"),
        ("input", "input_tokens", "right"),
        ("cached", "cached_tokens", "right"),
        ("output", "output_tokens", "right"),
        ("reasoning", "reasoning_tokens", "right"),
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        accuracy = row.get("accuracy")
        values = {
            "accuracy": "" if accuracy is None else f"{float(accuracy):.3f}",
            **{key: "" if row.get(key) is None else str(row.get(key)) for _, key, _ in columns},
        }
        table_rows.append([values[key] for _, key, _ in columns])

    widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        if table_rows
        else len(header)
        for index, (header, _, _) in enumerate(columns)
    ]

    def format_cell(value: str, width: int, align: str) -> str:
        if align == "right":
            return value.rjust(width)
        return value.ljust(width)

    header = "| " + " | ".join(
        format_cell(label, widths[index], "left")
        for index, (label, _, _) in enumerate(columns)
    ) + " |"
    divider = "| " + " | ".join(
        ("-" * (widths[index] - 1) + ":" if align == "right" else "-" * widths[index])
        for index, (_, _, align) in enumerate(columns)
    ) + " |"
    lines = [header, divider]
    for row in table_rows:
        lines.append(
            "| "
            + " | ".join(
                format_cell(row[index], widths[index], columns[index][2])
                for index in range(len(columns))
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-dir",
        type=Path,
        default=Path("clean_probes_from_4o-mini"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs/clean_probes_from_4o-mini_model_sweep"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Optional model list. Defaults to 10 representative GPT models: "
            + ", ".join(config["model"] for config in DEFAULT_MODEL_CONFIGS)
        ),
    )
    parser.add_argument("-n", "--concurrent-n", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--api-base")
    parser.add_argument("--timeout-sec", type=float)
    parser.add_argument("--retries", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run the legacy evaluator.",
    )
    parser.add_argument(
        "--evaluate-script",
        type=Path,
        default=Path("scripts/terminal/tools/evaluate_probe_model.py"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.probe_dir.is_dir():
        print(f"error: probe directory does not exist: {args.probe_dir}", file=sys.stderr)
        return 2
    if not args.evaluate_script.is_file():
        print(
            f"error: evaluate script does not exist: {args.evaluate_script}",
            file=sys.stderr,
        )
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    configs = selected_model_configs(args.models)
    rows: list[dict[str, Any]] = []

    for config in configs:
        model = str(config["model"])
        out_dir = result_path_for_model(args.out_root, model)
        command = build_eval_command(
            python=args.python,
            evaluate_script=args.evaluate_script,
            probe_dir=args.probe_dir,
            out_dir=out_dir,
            config=config,
            concurrent_n=args.concurrent_n,
            limit=args.limit,
            api_base=args.api_base,
            timeout_sec=args.timeout_sec,
            retries=args.retries,
            resume=args.resume,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
        )
        print("\n==>", " ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
        row = row_from_summary(
            model=model,
            out_dir=out_dir,
            returncode=completed.returncode,
        )
        rows.append(row)
        print(
            "summary model={model} completed={completed} errors={errors} "
            "correct={correct} accuracy={accuracy}".format(**row),
            flush=True,
        )
        if completed.returncode != 0 and args.fail_fast:
            break

    summary = {
        "probe_dir": str(args.probe_dir),
        "out_root": str(args.out_root),
        "model_count": len(rows),
        "rows": rows,
    }
    write_json(args.out_root / "model_sweep_summary.json", summary)
    write_csv(args.out_root / "model_sweep_summary.csv", rows)
    (args.out_root / "model_sweep_summary.md").write_text(
        markdown_table(rows),
        encoding="utf-8",
    )

    print("\n" + markdown_table(rows), end="")
    print(f"\nwrote summary files under {args.out_root}")
    return 0 if all((row.get("returncode") or 0) == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
