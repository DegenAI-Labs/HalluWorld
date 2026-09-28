#!/usr/bin/env python3
"""
Post-hoc hallucination probe evaluator.

Trajectory mode (--mode trajectory, default): flatten trajectory JSON/JSONL and
apply regex rules from tools/probe_rubrics.json (command/behavior heuristics).

QA mode (--mode qa): looks for a line PROBE_ANSWER: <text> in the trajectory
and checks answers against ground truth in tools/probe_questions.json. Tasks
should instruct the agent to emit that line when answering a factual question
(see each task instruction). Wrong or missing answers count as probe failure
(hallucination / ungrounded response).

Both modes (--mode both): run trajectory + QA; combined_pass is true only if both pass.

Examples:
  python3 tools/eval_probes.py --trial-dir ./job_outputs/trial_01 --task string-scavenger-hunt
  python3 tools/eval_probes.py --mode qa --trial-dir ./trial --task ghost-port --strict
  python3 tools/eval_probes.py --mode both --trajectory ./trajectory.json --task ghost-branch
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_rubric_path() -> Path:
    return _repo_root() / "tools" / "probe_rubrics.json"


def default_questions_path() -> Path:
    return _repo_root() / "tools" / "probe_questions.json"


def load_json_file(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return json.loads(raw)


def load_jsonl_file(path: Path) -> list[Any]:
    out: list[Any] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def load_trajectory_file(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl_file(path)
    data = load_json_file(path)
    if isinstance(data, list):
        return data
    return data


def flatten_strings(obj: Any, out: list[str] | None = None) -> list[str]:
    """Collect every string found in nested dict/list structures."""
    if out is None:
        out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            flatten_strings(v, out)
    elif isinstance(obj, list):
        for item in obj:
            flatten_strings(item, out)
    return out


def trajectory_to_search_text(data: Any) -> str:
    """
    Build a single searchable string. Prefer explicit keys when present in
    nested dicts (terminal-bench style), then fall back to full flatten.
    """
    chunks: list[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for key in (
                "keystrokes",
                "command",
                "commands",
                "analysis",
                "plan",
                "observation",
                "stdout",
                "stderr",
                "content",
                "text",
                "message",
            ):
                if key in o and isinstance(o[key], str):
                    chunks.append(o[key])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for item in o:
                walk(item)

    walk(data)
    blob = "\n".join(chunks)
    if len(blob.strip()) < 50:
        blob = "\n".join(flatten_strings(data))
    return blob


def find_trajectory_files(trial_dir: Path) -> list[Path]:
    """Pick likely trajectory artifacts under a trial or job directory."""
    preferred = [
        "trajectory.json",
        "trajectory.jsonl",
        "messages.jsonl",
        "session.jsonl",
    ]
    found: list[Path] = []
    if not trial_dir.is_dir():
        return found
    for name in preferred:
        p = trial_dir / name
        if p.is_file():
            found.append(p)
    for p in sorted(trial_dir.rglob("*.jsonl")):
        if p not in found:
            found.append(p)
    for p in sorted(trial_dir.rglob("trajectory.json")):
        if p not in found:
            found.append(p)
    return found


def load_rubrics(path: Path) -> dict[str, Any]:
    data = load_json_file(path)
    return data.get("tasks", {})


def load_probe_questions(path: Path) -> tuple[str, dict[str, Any]]:
    data = load_json_file(path)
    marker = str(data.get("marker", "PROBE_ANSWER:"))
    return marker, data.get("tasks", {})


def extract_probe_answer_line(full_text: str, marker: str) -> str | None:
    """Last PROBE_ANSWER: line wins (in case the model echoed the marker earlier)."""
    esc = re.escape(marker.rstrip())
    pat = esc + r"\s*(.+)$"
    matches = list(re.finditer(pat, full_text, re.MULTILINE))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def evaluate_qa_probes(
    full_text: str,
    task_id: str,
    marker: str,
    tasks_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = tasks_cfg.get(task_id)
    if not cfg:
        return {
            "task": task_id,
            "mode": "qa",
            "qa_pass": None,
            "error": f"No QA probes for task {task_id!r}. Known: {sorted(tasks_cfg)}",
        }

    answer = extract_probe_answer_line(full_text, marker)
    probes_def = cfg.get("probes", [])
    results: dict[str, Any] = {
        "task": task_id,
        "mode": "qa",
        "marker": marker,
        "probe_answer_extracted": answer,
        "qa_pass": True,
        "probes": [],
    }

    if answer is None:
        results["qa_pass"] = False
        results["failure_reason"] = "no_probe_answer_line"
        return results

    def check_one(p: dict[str, Any]) -> dict[str, Any]:
        pid = p.get("id", "?")
        match_kind = p.get("match", "contains")
        gt = p.get("ground_truth", "")
        out: dict[str, Any] = {"id": pid, "question": p.get("question", ""), "match": match_kind}

        if match_kind == "contains":
            ok = gt in answer
            out["ground_truth"] = gt
            out["matched"] = ok
        elif match_kind == "equals":
            ok = answer.strip() == str(gt).strip()
            out["ground_truth"] = gt
            out["matched"] = ok
        elif match_kind == "regex":
            try:
                ok = re.search(str(gt), answer) is not None
            except re.error as e:
                ok = False
                out["regex_error"] = str(e)
            out["pattern"] = gt
            out["matched"] = ok
        else:
            out["matched"] = False
            out["error"] = f"unknown match kind {match_kind!r}"

        return out

    for p in probes_def:
        r = check_one(p)
        results["probes"].append(r)
        if not r.get("matched", False):
            results["qa_pass"] = False

    return results


def _re_search(pat: str, text: str) -> tuple[bool, str | None]:
    try:
        return re.search(pat, text, re.DOTALL | re.MULTILINE) is not None, None
    except re.error as e:
        return False, str(e)


def evaluate_task(
    text: str,
    task_id: str,
    rubrics: dict[str, Any],
) -> dict[str, Any]:
    rule = rubrics.get(task_id)
    if not rule:
        return {
            "task": task_id,
            "probe_pass": None,
            "error": f"No rubric for task {task_id!r}. Known: {sorted(rubrics)}",
        }

    results: dict[str, Any] = {
        "task": task_id,
        "description": rule.get("description", ""),
        "probe_pass": True,
        "checks": {},
    }
    probe_pass = True

    detail: list[dict[str, Any]]

    for pat in rule.get("require_all_regex") or []:
        ok, err = _re_search(pat, text)
        results["checks"].setdefault("require_all_regex", []).append(
            {"pattern": pat, "matched": ok, **({"error": err} if err else {})}
        )
        if err or not ok:
            probe_pass = False

    for sub in rule.get("require_all_substrings") or []:
        ok = sub in text
        results["checks"].setdefault("require_all_substrings", []).append(
            {"substring": sub, "matched": ok}
        )
        if not ok:
            probe_pass = False

    any_patterns = rule.get("require_any_regex") or []
    if any_patterns:
        detail = []
        matched_any = False
        for pat in any_patterns:
            ok, err = _re_search(pat, text)
            detail.append({"pattern": pat, "matched": ok, **({"error": err} if err else {})})
            if err:
                probe_pass = False
            elif ok:
                matched_any = True
        results["checks"]["require_any_regex"] = detail
        if not matched_any:
            probe_pass = False

    any_subs = rule.get("require_any_substrings") or []
    if any_subs:
        detail = [{"substring": s, "matched": s in text} for s in any_subs]
        results["checks"]["require_any_substrings"] = detail
        if not any(s in text for s in any_subs):
            probe_pass = False

    for pat in rule.get("forbidden_regex") or []:
        ok, err = _re_search(pat, text)
        results["checks"].setdefault("forbidden_regex", []).append(
            {"pattern": pat, "matched": ok, **({"error": err} if err else {})}
        )
        if err:
            probe_pass = False
        elif ok:
            probe_pass = False

    results["probe_pass"] = probe_pass
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trial-dir",
        type=Path,
        help="Directory containing trajectory.json / .jsonl (searched recursively)",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        help="Explicit trajectory file (.json or .jsonl)",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task id matching keys in probe_rubrics.json (e.g. string-scavenger-hunt)",
    )
    parser.add_argument(
        "--mode",
        choices=("trajectory", "qa", "both"),
        default="trajectory",
        help="trajectory=regex rubrics; qa=PROBE_ANSWER ground-truth check; both=both",
    )
    parser.add_argument(
        "--rubric",
        type=Path,
        default=None,
        help="Path to probe_rubrics.json (default: tools/probe_rubrics.json)",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=None,
        help="Path to probe_questions.json (default: tools/probe_questions.json)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if probe_pass / qa_pass / combined_pass is false",
    )
    parser.add_argument(
        "--text-dump",
        type=Path,
        default=None,
        help="Write flattened trajectory text to this file (for debugging rubrics)",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.trajectory:
        paths = [args.trajectory]
    elif args.trial_dir:
        paths = find_trajectory_files(args.trial_dir)
    else:
        print("Error: provide --trajectory or --trial-dir", file=sys.stderr)
        return 2

    if not paths:
        print(
            json.dumps(
                {
                    "task": args.task,
                    "probe_pass": None,
                    "error": "No trajectory file found in trial directory",
                    "trial_dir": str(args.trial_dir) if args.trial_dir else None,
                },
                indent=2,
            )
        )
        return 1 if args.strict else 0

    combined_text_parts: list[str] = []
    loaded_from: list[str] = []
    for p in paths:
        data = load_trajectory_file(p)
        combined_text_parts.append(trajectory_to_search_text(data))
        loaded_from.append(str(p))

    text = "\n\n---\n\n".join(combined_text_parts)
    if args.text_dump:
        args.text_dump.parent.mkdir(parents=True, exist_ok=True)
        args.text_dump.write_text(text, encoding="utf-8")

    out: dict[str, Any] = {"task": args.task, "sources": loaded_from, "text_chars": len(text)}

    if args.mode in ("trajectory", "both"):
        rubric_path = args.rubric or default_rubric_path()
        rubrics = load_rubrics(rubric_path)
        traj_result = evaluate_task(text, args.task, rubrics)
        traj_result["sources"] = loaded_from
        traj_result["text_chars"] = len(text)
        if args.mode == "trajectory":
            print(json.dumps(traj_result, indent=2))
            if traj_result.get("error"):
                return 1 if args.strict else 0
            if args.strict and traj_result.get("probe_pass") is False:
                return 1
            return 0
        out["trajectory"] = traj_result

    if args.mode in ("qa", "both"):
        qpath = args.questions or default_questions_path()
        marker, qtasks = load_probe_questions(qpath)
        qa_result = evaluate_qa_probes(text, args.task, marker, qtasks)
        qa_result["sources"] = loaded_from
        if args.mode == "qa":
            print(json.dumps(qa_result, indent=2))
            if qa_result.get("error"):
                return 1 if args.strict else 0
            if args.strict and qa_result.get("qa_pass") is False:
                return 1
            return 0
        out["qa"] = qa_result

    if args.mode == "both":
        tp = out["trajectory"].get("probe_pass")
        qp = out["qa"].get("qa_pass")
        combined = (
            tp is True
            and qp is True
            and not out["trajectory"].get("error")
            and not out["qa"].get("error")
        )
        out["combined_pass"] = combined
        print(json.dumps(out, indent=2))
        if args.strict and not combined:
            return 1
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
