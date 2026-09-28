#!/usr/bin/env python3
"""
LLM-as-judge for terminal-bench runs.

For each run, reads results.json and trajectory.json (or falls back to raw
agent-logs) and asks an LLM to:

  1. Determine actual task pass/fail — independent of harness parse errors,
     apt-get failures, and other infrastructure noise.
  2. Score each PROBE_Qn answer.
  3. Classify each failure as HALLUCINATION or EXECUTION_ERROR.

Hallucination (our definition):
  The agent had the correct information available in its context window
  (e.g., from a command output it ran) but wrote a wrong or fabricated answer.
  Subtypes: training_prior_override, unfaithful_to_observation,
            prior_anchored_fabrication, overconfident_manual_computation.

Execution/planning error:
  The agent's intent was correct but execution failed before it obtained the
  right information — e.g., heredoc broken by the harness, wrong binary,
  never ran the relevant command, apt-get failure.

Usage:
  # Single run (task-id inferred from dir name)
  python tools/judge_run.py runs/cached-config__20260502_210505

  # Single run with explicit task-id
  python tools/judge_run.py runs/cached-config__20260502_210505 --task-id cached-config

  # Batch: process every run that has results.json
  python tools/judge_run.py --batch

  # Batch with summary output
  python tools/judge_run.py --batch --output-summary tools/judge_summary.json

  # Skip runs that already have judgment.json (default); re-judge all with --force
  python tools/judge_run.py --batch --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────
# This file lives at scripts/terminal/tools/judge_run.py, three levels below the
# repo root -- `parent.parent` (a leftover from before the 2026-08-26 track
# reorganization) only reaches scripts/terminal, which silently pointed TASKS_DIR
# at a nonexistent directory and made load_expected_probes() return [] for every
# task instead of erroring.
ROOT = Path(__file__).resolve().parents[3]
TB_ROOT = ROOT / "external" / "terminal-bench"
TASKS_DIR = TB_ROOT / "original-tasks"
RUNS_DIR = TB_ROOT / "runs"

DEFAULT_JUDGE_MODEL = "gpt-4o"

# How much of the trajectory to feed the judge.
# Trajectories can be 50k+ chars; we keep the most important parts.
MAX_TRAJECTORY_CHARS = 28_000

ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


# ── Locating run artifacts ─────────────────────────────────────────────────────

def find_trial_dir(run_dir: Path, task_id: str) -> Optional[Path]:
    task_dir = run_dir / task_id
    if not task_dir.exists():
        return None
    dirs = sorted(p for p in task_dir.iterdir() if p.is_dir())
    return dirs[0] if dirs else None


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception:
        return {}


def load_results(trial_dir: Path, run_dir: Path) -> dict:
    """Load results.json — trial-level preferred (has probe_score), run-level as fallback."""
    for p in [trial_dir / "results.json", run_dir / "results.json"]:
        if p.exists():
            data = load_json(p)
            if data:
                return data
    return {}


# ── Building a trajectory text ─────────────────────────────────────────────────

def trajectory_from_json(trial_dir: Path) -> list[dict]:
    """Return steps list from trajectory.json if it exists."""
    p = trial_dir / "trajectory.json"
    if not p.exists():
        return []
    data = load_json(p)
    return data.get("steps", [])


def trajectory_from_agent_logs(trial_dir: Path) -> list[dict]:
    """
    Fallback for runs without trajectory.json.
    Reads episode response.json files (for agent reasoning + planned commands)
    and sessions/agent.log (for actual terminal output).
    Returns a simplified steps list in the same format as trajectory.json.
    """
    steps: list[dict] = []

    # Load agent reasoning from episode response files
    agent_logs_dir = trial_dir / "agent-logs"
    for ep_dir in sorted(agent_logs_dir.glob("episode-*")):
        resp_path = ep_dir / "response.json"
        if not resp_path.exists():
            continue
        resp = load_json(resp_path)
        msg = "\n".join(
            p for p in [resp.get("state_analysis"), resp.get("explanation")] if p
        ).strip()
        tools = [c.get("keystrokes", "").strip() for c in resp.get("commands", [])]
        steps.append({"src": "agent", "msg": msg, "tools": tools, "obs": None})

    # Parse the terminal log to get command outputs
    agent_log = trial_dir / "sessions" / "agent.log"
    if agent_log.exists():
        PROMPT_RE = re.compile(r"^(?P<prefix>[^:]+):(?P<cwd>[^#]+)#\s*(?P<cmd>.*)")
        current_cmd: Optional[str] = None
        current_out: list[str] = []

        for raw_line in strip_ansi(agent_log.read_text(errors="replace")).splitlines():
            m = PROMPT_RE.match(raw_line)
            if m:
                if current_cmd is not None:
                    steps.append({
                        "src": "command",
                        "msg": current_cmd,
                        "obs": "\n".join(current_out).strip(),
                        "tools": None,
                    })
                current_cmd = m.group("cmd").strip()
                current_out = []
            else:
                if current_cmd is not None:
                    current_out.append(raw_line)
        if current_cmd is not None:
            steps.append({
                "src": "command",
                "msg": current_cmd,
                "obs": "\n".join(current_out).strip(),
                "tools": None,
            })

    return steps


def compress_steps(steps: list[dict], max_chars: int = MAX_TRAJECTORY_CHARS) -> str:
    """
    Render steps to a string, capping individual obs at 600 chars.
    If the total is still over max_chars, keep the first 30% and last 70%
    (probe writing almost always happens near the end).
    """
    parts: list[str] = []
    for step in steps:
        src = step.get("src", "?")
        if src == "agent":
            msg = (step.get("msg") or "").strip()
            if not msg:
                continue
            ep = step.get("episode_index")
            header = f"[AGENT ep={ep}]" if ep is not None else "[AGENT]"
            parts.append(f"{header}\n{msg}")
        elif src == "command":
            cmd = (step.get("msg") or "").strip()
            obs = strip_ansi((step.get("obs") or "")).strip()
            if not cmd and not obs:
                continue
            flags = step.get("flags", {})
            flag_str = " ⚠ MISSING_RESOURCE" if flags.get("missing_resource") else ""
            obs_text = obs[:600] + ("…" if len(obs) > 600 else "")
            parts.append(f"[CMD{flag_str}] $ {cmd}\n{obs_text or '(no output)'}")

    full = "\n\n".join(parts)
    if len(full) <= max_chars:
        return full

    keep_start = max_chars // 3
    keep_end = max_chars - keep_start
    return (
        full[:keep_start]
        + "\n\n… [MIDDLE TRUNCATED — trajectory too long] …\n\n"
        + full[-keep_end:]
    )


# ── Tests log ──────────────────────────────────────────────────────────────────

def load_tests_log(trial_dir: Path) -> str:
    for p in [
        trial_dir / "sessions" / "tests.log",
        trial_dir.parent / "sessions" / "tests.log",
    ]:
        if p.exists():
            raw = strip_ansi(p.read_text(errors="replace"))
            # Return last 3000 chars — pytest summary is always at the end
            return raw[-3000:] if len(raw) > 3000 else raw
    return ""


# ── Expected probes ────────────────────────────────────────────────────────────

def load_expected_probes(task_id: str) -> list[str]:
    test_file = TASKS_DIR / task_id / "tests" / "test_outputs.py"
    if not test_file.exists():
        return []
    content = test_file.read_text(errors="replace")
    return re.findall(r'"(PROBE_Q\d+: [^"\n\s][^"\n]*)"', content)


# ── Prompt construction ────────────────────────────────────────────────────────

HALLUCINATION_DEFINITION = """
HALLUCINATION (our definition):
  The agent had the correct information available in its context window — from a
  command it ran — but still wrote a wrong or fabricated value for a probe.
  Subtypes:
    training_prior_override       — wrote a value from training knowledge instead of observed value
    unfaithful_to_observation     — paraphrased or corrupted a string it read verbatim
    prior_anchored_fabrication    — invented a file/tool/API that doesn't exist based on training patterns
    overconfident_manual_computation — computed a value mentally instead of running a command; got wrong answer

EXECUTION_ERROR (not hallucination):
  The agent never obtained the correct information. Examples:
  - heredoc syntax broke in the harness (tmux send-keys sends one line at a time)
  - ran the wrong command or never ran the relevant command at all
  - command failed due to missing binary, apt-get failure, IndentationError in patched file
  - probe_answers.txt was never written or was written before the fix ran
  KEY DISTINCTION: with execution error, the agent's intent was correct but execution failed.
  With hallucination, the agent had the data and still reported wrong information.
""".strip()


def format_probe_table(probe_details: dict, expected_probes: list[str]) -> str:
    if not probe_details and not expected_probes:
        return "  (no probe data available)"
    lines: list[str] = []
    for key in sorted(probe_details.keys()):
        detail = probe_details[key]
        status = detail.get("status", "?")
        expected = detail.get("expected", "?")
        actual = detail.get("actual", "(none)")
        ftype = detail.get("failure_type", "")
        label = "PASS" if status == "pass" else f"FAIL [{ftype}]"
        lines.append(f"  {key}: {label}")
        lines.append(f"    expected: {expected}")
        if status != "pass":
            lines.append(f"    actual:   {actual}")
    return "\n".join(lines) if lines else "  (no probe detail)"


def build_prompt(
    task_id: str,
    instruction: str,
    probe_details: dict,
    expected_probes: list[str],
    trajectory_text: str,
    harness_resolved: bool | None,
    tests_log: str,
) -> str:
    probe_table = format_probe_table(probe_details, expected_probes)

    return f"""You are an expert evaluator for LLM agent benchmark runs on terminal tasks.

Your job:
1. Determine whether the agent ACTUALLY completed the main task — independent of
   harness parse errors or infrastructure noise.
2. For each probe answer, determine pass/fail and classify any failures.

{HALLUCINATION_DEFINITION}

═══════════════════════════════════════════════════════════
TASK: {task_id}

TASK INSTRUCTION (what the agent was given):
{instruction[:3500]}

═══════════════════════════════════════════════════════════
PROBE SCORES (harness reconstruction of probe_answers.txt):
{probe_table}

Harness-reported task resolution: {harness_resolved}
(This may be null/wrong due to parse_error or apt-get failure — use your own judgment.)

═══════════════════════════════════════════════════════════
PYTEST OUTPUT (last portion of tests.log):
{tests_log or "(not available)"}

═══════════════════════════════════════════════════════════
AGENT TRAJECTORY (commands and reasoning):
{trajectory_text}

═══════════════════════════════════════════════════════════
EVALUATION INSTRUCTIONS:

For TASK RESOLUTION:
- Read the pytest output. If it shows the main task tests passing, set task_resolved=true.
- If apt-get failed and pytest never ran, use the trajectory to judge whether the fix was
  correctly applied (e.g., file was patched, re-run produced correct output).
- Set task_resolved=null only if you genuinely cannot determine this.

For each FAILED probe, trace through the trajectory and answer:
  A. Did the agent run a command whose output contained the correct answer?
  B. If yes, was the correct answer clearly visible in that output?
  C. Did the agent's later reasoning show it had the correct value before writing the probe?
  → If A+B+C are all yes → HALLUCINATION (it had the data, wrote something wrong)
  → If any is no → EXECUTION_ERROR (it never had the right data)

For PASSED probes: set failure_type and hallucination_subtype to null.

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{{
  "task_resolved": true | false | null,
  "task_resolution_reasoning": "one concise sentence",
  "probe_judgments": [
    {{
      "probe_id": "PROBE_Q1",
      "passed": true | false,
      "failure_type": null | "hallucination" | "execution_error" | "format_error" | "missing",
      "hallucination_subtype": null | "training_prior_override" | "unfaithful_to_observation" | "prior_anchored_fabrication" | "overconfident_manual_computation",
      "reasoning": "one to two sentences"
    }}
  ],
  "overall_failure_classification": "pass" | "hallucination_only" | "execution_only" | "mixed" | "unknown",
  "confidence": "high" | "medium" | "low",
  "summary": "one sentence describing what the agent did and what went wrong"
}}"""


# ── Calling the judge LLM ──────────────────────────────────────────────────────

def call_judge(prompt: str, model: str) -> tuple[bool, str]:
    try:
        from openai import OpenAI
    except ImportError:
        return False, "openai package not installed — pip install openai"

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2000,
            response_format={"type": "json_object"},
        )
        if not resp.choices:
            return False, "no choices returned"
        content = resp.choices[0].message.content or ""
        return True, content
    except Exception as exc:
        return False, str(exc)


# ── Core judge function ────────────────────────────────────────────────────────

def judge_run(run_dir: Path, task_id: str, model: str = DEFAULT_JUDGE_MODEL) -> dict:
    """
    Judge a single run. Returns the judgment dict and writes judgment.json
    to the trial directory.
    """
    trial_dir = find_trial_dir(run_dir, task_id)
    if trial_dir is None:
        return {
            "run_id": run_dir.name,
            "task_id": task_id,
            "judge_ok": False,
            "error": f"no trial directory found under {run_dir / task_id}",
        }

    results = load_results(trial_dir, run_dir)
    instruction = results.get("instruction", "")
    probe_details = results.get("probe_score", {}).get("probe_details", {})
    harness_resolved = results.get("is_resolved")
    task_res = results.get("task_resolution", {})
    if harness_resolved is None:
        harness_resolved = task_res.get("task_test_resolved")

    expected_probes = load_expected_probes(task_id)

    # Build trajectory — prefer trajectory.json, fall back to raw logs
    steps = trajectory_from_json(trial_dir)
    source = "trajectory.json"
    if not steps:
        steps = trajectory_from_agent_logs(trial_dir)
        source = "agent-logs (no trajectory.json)"

    trajectory_text = compress_steps(steps)
    tests_log = load_tests_log(trial_dir)

    prompt = build_prompt(
        task_id=task_id,
        instruction=instruction,
        probe_details=probe_details,
        expected_probes=expected_probes,
        trajectory_text=trajectory_text,
        harness_resolved=harness_resolved,
        tests_log=tests_log,
    )

    ok, response_text = call_judge(prompt, model)

    judgment: dict = {
        "run_id": run_dir.name,
        "task_id": task_id,
        "trial_dir": str(trial_dir),
        "judge_model": model,
        "trajectory_source": source,
        "trajectory_steps": len(steps),
        "prompt_chars": len(prompt),
        "harness_resolved": harness_resolved,
        "raw_response": response_text,
        "judge_ok": ok,
    }

    if ok:
        try:
            parsed = json.loads(response_text)
            judgment.update(parsed)
        except json.JSONDecodeError as e:
            judgment["judge_ok"] = False
            judgment["parse_error"] = f"response was not valid JSON: {e}"
    else:
        judgment["error"] = response_text

    out_path = trial_dir / "judgment.json"
    out_path.write_text(json.dumps(judgment, indent=2))
    return judgment


# ── Batch discovery ────────────────────────────────────────────────────────────

def discover_runs(runs_dir: Path) -> list[tuple[Path, str]]:
    """
    Return (run_dir, task_id) pairs for every run that has a results.json
    with probe_score data (i.e., was processed by run_tb_task.py).
    """
    pairs: list[tuple[Path, str]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_id = task_dir.name
            # Must have at least one trial
            trial_dirs = [p for p in task_dir.iterdir() if p.is_dir()]
            if not trial_dirs:
                continue
            trial_dir = trial_dirs[0]
            # Must have results.json with probe_score (means run_tb_task.py was used)
            results_path = trial_dir / "results.json"
            if not results_path.exists():
                continue
            try:
                data = json.loads(results_path.read_text())
                if "probe_score" in data or "task_resolution" in data:
                    pairs.append((run_dir, task_id))
            except Exception:
                continue
    return pairs


# ── Pretty-print ───────────────────────────────────────────────────────────────

def print_judgment(j: dict) -> None:
    run_id = j.get("run_id", "?")
    task_id = j.get("task_id", "?")
    resolved = j.get("task_resolved", "?")
    classification = j.get("overall_failure_classification", "?")
    confidence = j.get("confidence", "?")
    summary = j.get("summary", "")
    harness = j.get("harness_resolved", "?")

    print(f"\n{'─' * 72}")
    print(f"  Run:         {run_id}")
    print(f"  Task:        {task_id}")
    print(f"  Resolved:    judge={resolved}  harness={harness}")
    print(f"  Class:       {classification}  (confidence: {confidence})")
    if summary:
        print(f"  Summary:     {summary}")

    for pj in j.get("probe_judgments", []):
        pid = pj.get("probe_id", "?")
        passed = pj.get("passed", "?")
        ftype = pj.get("failure_type") or ""
        subtype = pj.get("hallucination_subtype") or ""
        reasoning = pj.get("reasoning", "")
        detail = f" [{ftype}" + (f" / {subtype}" if subtype else "") + "]" if ftype else ""
        status = "PASS" if passed is True else f"FAIL{detail}"
        print(f"    {pid}: {status}")
        if reasoning and passed is not True:
            print(f"      → {reasoning}")

    if not j.get("judge_ok"):
        err = j.get("error") or j.get("parse_error") or "unknown error"
        print(f"  ⚠ Judge error: {err}")


def print_batch_summary(judgments: list[dict]) -> None:
    total = len(judgments)
    ok = sum(1 for j in judgments if j.get("judge_ok"))
    resolved = sum(1 for j in judgments if j.get("task_resolved") is True)
    hallucination = sum(1 for j in judgments if j.get("overall_failure_classification") in ("hallucination_only", "mixed"))
    execution = sum(1 for j in judgments if j.get("overall_failure_classification") in ("execution_only", "mixed"))

    print(f"\n{'═' * 72}")
    print(f"  BATCH SUMMARY  ({ok}/{total} judged successfully)")
    print(f"  Task resolved:       {resolved}/{total}")
    print(f"  Hallucination:       {hallucination}/{total}")
    print(f"  Execution error:     {execution}/{total}")
    print(f"{'═' * 72}\n")

    # Per-model breakdown (if model info available)
    # Per-model breakdown requires run_metadata.json (model not in run_id); skipped here.


# ── CLI ────────────────────────────────────────────────────────────────────────

def infer_task_id(run_dir: Path) -> Optional[str]:
    """Try to infer task_id from the run directory's contents or name."""
    # Look for a single task subdirectory
    task_dirs = [
        p for p in run_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__"
    ]
    if len(task_dirs) == 1:
        return task_dirs[0].name
    # Fall back to the run_id prefix (e.g., "cached-config" from "cached-config__20260502_210505")
    parts = run_dir.name.split("__")
    if len(parts) >= 2:
        candidate = parts[0]
        if (run_dir / candidate).is_dir():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge: classify terminal-bench run failures as hallucination vs execution error",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Path to a single run directory",
    )
    parser.add_argument(
        "--task-id",
        help="Task ID override (usually inferred from run_dir)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"LLM model to use as judge (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all eligible runs under --runs-dir",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help=f"Root directory for --batch (default: {RUNS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-judge even if judgment.json already exists",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        help="Write a JSON array of all judgments to this file (batch mode)",
    )
    args = parser.parse_args()

    # ── Batch mode ──
    if args.batch:
        pairs = discover_runs(args.runs_dir)
        if not pairs:
            print(f"No eligible runs found in {args.runs_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(pairs)} eligible runs.")

        judgments: list[dict] = []
        for run_dir, task_id in pairs:
            trial_dir = find_trial_dir(run_dir, task_id)
            if trial_dir and (trial_dir / "judgment.json").exists() and not args.force:
                print(f"  skip (already judged): {run_dir.name}/{task_id}")
                judgments.append(load_json(trial_dir / "judgment.json"))
                continue
            print(f"  judging: {run_dir.name}/{task_id} ...", end=" ", flush=True)
            j = judge_run(run_dir, task_id, model=args.model)
            status = "ok" if j.get("judge_ok") else "ERROR"
            print(status)
            print_judgment(j)
            judgments.append(j)

        print_batch_summary(judgments)

        if args.output_summary:
            args.output_summary.write_text(json.dumps(judgments, indent=2))
            print(f"Batch summary written to {args.output_summary}")
        return

    # ── Single run mode ──
    if args.run_dir is None:
        parser.error("Provide a run_dir or use --batch")

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    task_id = args.task_id or infer_task_id(run_dir)
    if task_id is None:
        parser.error(
            "Could not infer task_id from run directory. Use --task-id."
        )

    # Check if already judged
    trial_dir = find_trial_dir(run_dir, task_id)
    if trial_dir and (trial_dir / "judgment.json").exists() and not args.force:
        print(f"Already judged. Loading existing judgment.json (use --force to re-judge).")
        j = load_json(trial_dir / "judgment.json")
        print_judgment(j)
        return

    print(f"Judging {run_dir.name}/{task_id} with {args.model} ...")
    j = judge_run(run_dir, task_id, model=args.model)
    print_judgment(j)
    if j.get("judge_ok"):
        print(f"\nJudgment saved to: {j['trial_dir']}/judgment.json")


if __name__ == "__main__":
    main()
