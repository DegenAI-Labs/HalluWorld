#!/usr/bin/env python3
"""
Extract runtime probes from Terminal-Bench trajectory files.

For each runtime probe, write one JSON file containing:
  - question: model-facing probe question
  - answer: golden answer
  - context: terminal pane text from immediately before the trigger command
  - metadata: source/task/probe fields for traceability

Examples:
  python3 tools/extract_run_probes.py \
    --run-dir runs/all-tasks__20260502_201427 \
    --out-dir runs/all-tasks__20260502_201427/extracted_probes \
    --llm-out-dir runs/all-tasks__20260502_201427/extracted_llm_probes

  python3 tools/extract_run_probes.py \
    --trajectory runs/.../trajectory.json \
    --out-dir /tmp/probes_one_task
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


PROBE_REQUIRED_KEYS = {"injector", "question", "golden"}
SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def safe_slug(value: Any, *, fallback: str = "unknown", max_len: int = 80) -> str:
    text = str(value or fallback).strip()
    text = SLUG_RE.sub("-", text).strip("-._")
    if not text:
        text = fallback
    return text[:max_len].strip("-._") or fallback


def find_trajectory_files(run_dir: Path) -> list[Path]:
    if run_dir.is_file():
        return [run_dir]
    return sorted(run_dir.rglob("trajectory.json"))


def iter_step_probes(step: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """
    Yield runtime probes from a trajectory step.

    Recent trajectories store all probes in step["probes"] and may duplicate the
    first runtime probe in step["probe"]. Older trajectories may only have
    step["probe"]. Prefer step["probes"] when present to avoid duplicate output.
    """
    probes = step.get("probes")
    if isinstance(probes, list):
        for probe in probes:
            if isinstance(probe, dict) and PROBE_REQUIRED_KEYS <= probe.keys():
                yield probe
        return

    probe = step.get("probe")
    if isinstance(probe, dict) and PROBE_REQUIRED_KEYS <= probe.keys():
        yield probe


def iter_trajectory_probes(
    trajectory: dict[str, Any], trajectory_path: Path
) -> Iterable[tuple[dict[str, Any], dict[str, Any]]]:
    task_meta = {
        "task_name": trajectory.get("task_name"),
        "run_id": trajectory.get("run_id"),
        "trial_name": trajectory.get("trial_name"),
        "model": trajectory.get("model"),
        "is_resolved": trajectory.get("is_resolved"),
        "failure_mode": trajectory.get("failure_mode"),
        "trajectory_path": str(trajectory_path),
    }

    steps = trajectory.get("steps", [])
    if not isinstance(steps, list):
        return

    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        step_meta = {
            "trajectory_step_index": step_index,
            "step_src": step.get("src"),
            "step_session": step.get("session"),
            "step_command_index": step.get("session_command_index"),
            "step_command": step.get("msg"),
            "step_cwd": step.get("cwd"),
        }

        for probe in iter_step_probes(step):
            yield probe, {**task_meta, **step_meta}


def pane_before_text(raw_context: str) -> tuple[str, str]:
    marker = "\npane_before:\n"
    if marker in raw_context:
        return raw_context.split(marker, 1)[1], "pane_before"
    if raw_context.startswith("pane_before:\n"):
        return raw_context.split("\n", 1)[1], "pane_before"
    return raw_context, "raw_context_missing_pane_before"


def select_context_path(probe: dict[str, Any]) -> tuple[str | None, str]:
    if probe.get("context_path"):
        return str(probe["context_path"]), "context_path"
    if (
        probe.get("context_scope") == "through_trigger"
        and probe.get("post_context_path")
    ):
        return str(probe["post_context_path"]), "post_context_path"
    if probe.get("pre_context_path"):
        return str(probe["pre_context_path"]), "pre_context_path"
    if probe.get("post_context_path"):
        return str(probe["post_context_path"]), "post_context_path"
    return None, "missing"


def read_context(
    probe: dict[str, Any],
) -> tuple[str, str | None, str | None, str, str]:
    context_path, context_kind = select_context_path(probe)
    if not context_path:
        return "", None, "missing_context_path", "missing", context_kind

    path = Path(str(context_path))
    try:
        raw_context = path.read_text(encoding="utf-8", errors="replace")
        context, context_section = pane_before_text(raw_context)
        return context, str(path), None, context_section, context_kind
    except OSError as exc:
        return "", str(path), f"{type(exc).__name__}: {exc}", "missing", context_kind


def build_record(
    probe: dict[str, Any],
    source_meta: dict[str, Any],
    sequence: int,
) -> dict[str, Any]:
    context, selected_context_path, context_error, context_section, context_kind = (
        read_context(probe)
    )

    metadata_keys = [
        "probe_id",
        "injector",
        "probe_type",
        "context_scope",
        "answer_schema",
        "golden_source",
        "golden_command_or_heuristic",
        "session",
        "trigger_step_index",
        "trigger_command",
        "cwd_before",
        "cwd_after",
        "created_at",
        "runtime_probe_path",
        "context_path",
        "pre_context_path",
        "post_context_path",
        "episode_index",
        "defer_model_answer",
        "llm_probe_model",
        "llm_probe_reasoning_effort",
        "llm_probe_prompt_path",
        "llm_probe_response_path",
        "difficulty_rationale",
        "usefulness",
        "difficulty_score",
        "answerability_score",
        "failure_mode_target",
        "locality_risk",
    ]

    metadata = {
        "sequence": sequence,
        **source_meta,
        **{key: probe.get(key) for key in metadata_keys if key in probe},
        "selected_context_path": selected_context_path,
        "selected_context_kind": context_kind,
        "selected_context_section": context_section,
        "context_contains_trigger_metadata": False,
        "model_answer": probe.get("a"),
    }
    if context_error is not None:
        metadata["context_error"] = context_error

    return {
        "question": probe.get("question") or probe.get("q"),
        "answer": probe.get("golden"),
        "context": context,
        "metadata": metadata,
    }


def output_name(record: dict[str, Any]) -> str:
    metadata = record["metadata"]
    task = safe_slug(metadata.get("task_name"), max_len=48)
    step = metadata.get("trigger_step_index")
    step_text = f"step-{int(step):04d}" if isinstance(step, int) else "step-unknown"
    injector = safe_slug(metadata.get("injector"), max_len=48)
    probe_id = safe_slug(metadata.get("probe_id"), max_len=64)
    return f"{metadata['sequence']:06d}_{task}_{step_text}_{injector}_{probe_id}.json"


def write_record(out_dir: Path, record: dict[str, Any]) -> Path:
    out_path = out_dir / output_name(record)
    out_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out_path


def is_llm_generated_probe(record: dict[str, Any]) -> bool:
    metadata = record.get("metadata") or {}
    return metadata.get("injector") == "llm_generated"


def extract(
    run_or_trajectory: Path,
    out_dir: Path,
    llm_out_dir: Path | None = None,
) -> dict[str, Any]:
    trajectories = find_trajectory_files(run_or_trajectory)
    out_dir.mkdir(parents=True, exist_ok=True)
    if llm_out_dir is not None:
        llm_out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "input": str(run_or_trajectory),
        "output_dir": str(out_dir),
        "llm_output_dir": str(llm_out_dir) if llm_out_dir is not None else None,
        "trajectory_count": len(trajectories),
        "probe_count": 0,
        "llm_probe_count": 0,
        "context_error_count": 0,
        "llm_context_error_count": 0,
        "trajectories": [],
    }

    sequence = 0
    for trajectory_path in trajectories:
        try:
            trajectory = load_json(trajectory_path)
        except Exception as exc:
            summary["trajectories"].append(
                {
                    "trajectory_path": str(trajectory_path),
                    "error": f"{type(exc).__name__}: {exc}",
                    "probe_count": 0,
                }
            )
            continue

        trajectory_count = 0
        trajectory_llm_count = 0
        for probe, source_meta in iter_trajectory_probes(trajectory, trajectory_path):
            sequence += 1
            trajectory_count += 1
            record = build_record(probe, source_meta, sequence)
            is_llm_probe = is_llm_generated_probe(record)
            has_context_error = "context_error" in record["metadata"]
            if has_context_error:
                summary["context_error_count"] += 1
            if is_llm_probe:
                trajectory_llm_count += 1
                summary["llm_probe_count"] += 1
                if has_context_error:
                    summary["llm_context_error_count"] += 1

            write_record(out_dir, record)
            if is_llm_probe and llm_out_dir is not None:
                write_record(llm_out_dir, record)

        summary["probe_count"] += trajectory_count
        summary["trajectories"].append(
            {
                "trajectory_path": str(trajectory_path),
                "task_name": trajectory.get("task_name"),
                "trial_name": trajectory.get("trial_name"),
                "probe_count": trajectory_count,
                "llm_probe_count": trajectory_llm_count,
            }
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if llm_out_dir is not None:
        llm_summary = {
            **summary,
            "output_dir": str(llm_out_dir),
            "all_probe_output_dir": str(out_dir),
            "probe_count": summary["llm_probe_count"],
            "context_error_count": summary["llm_context_error_count"],
        }
        (llm_out_dir / "summary.json").write_text(
            json.dumps(llm_summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def prepare_output_dir(out_dir: Path, *, preserve_existing: bool) -> None:
    if not out_dir.exists():
        return
    existing_json = list(out_dir.glob("*.json"))
    if existing_json and preserve_existing:
        raise SystemExit(
            "error: output directory already contains JSON files. "
            "Use a fresh directory or omit --preserve-existing."
        )
    for path in existing_json:
        path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract runtime probes from a full run into per-probe JSON files."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory to scan recursively for trajectory.json files.",
    )
    source.add_argument(
        "--trajectory",
        type=Path,
        help="Single trajectory.json file to extract.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where per-probe JSON files will be written.",
    )
    parser.add_argument(
        "--llm-out-dir",
        "--llm-generated-out-dir",
        dest="llm_out_dir",
        type=Path,
        default=None,
        help=(
            "Optional second directory that receives only llm_generated probes "
            "in the same per-probe JSON format."
        ),
    )
    parser.add_argument(
        "--preserve-existing",
        action="store_true",
        help="Fail instead of deleting existing JSON files in --out-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.run_dir or args.trajectory
    if not source.exists():
        print(f"error: input path does not exist: {source}", file=sys.stderr)
        return 2

    prepare_output_dir(args.out_dir, preserve_existing=args.preserve_existing)
    if args.llm_out_dir is not None:
        prepare_output_dir(args.llm_out_dir, preserve_existing=args.preserve_existing)

    summary = extract(source, args.out_dir, llm_out_dir=args.llm_out_dir)
    print(
        "extracted "
        f"{summary['probe_count']} probes from {summary['trajectory_count']} "
        f"trajectories into {summary['output_dir']}"
    )
    if args.llm_out_dir is not None:
        print(
            "extracted "
            f"{summary['llm_probe_count']} llm_generated probes into "
            f"{summary['llm_output_dir']}"
        )
    if summary["context_error_count"]:
        print(
            f"warning: {summary['context_error_count']} probes had missing/unreadable context",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
