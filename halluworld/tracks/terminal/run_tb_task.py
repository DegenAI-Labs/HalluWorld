#!/usr/bin/env python3
"""
Run Terminal-Bench task runs and emit rich trajectory.json files.

Features vs the shell script:
- Pure Python CLI.
- Captures agent "speech" (state analysis + explanation) from response.json files.
- Captures commands and their stdout/stderr from tmux agent.log (best-effort split).
- Keeps original artifacts (commands.txt, pane logs, agent logs).

Usage:
    python run_tb_task.py [task-id] [--model MODEL] [--run-id RUN_ID] [--output-dir runs]
    python run_tb_task.py [--model MODEL] [--run-id RUN_ID] [--output-dir runs] [--n-concurrent N]
    python run_tb_task.py TASK [--run-id ID] [--judge]
        # 1) tb agent run  2) trajectory.json + probe scoring (unless --no-eval-probes)
        # 3) tools/judge_run.py — LLM judge on trajectory/probes (hallucination vs execution, etc.)

Env requirements:
    - OPENAI_API_KEY available (e.g., source .venv/bin/activate beforehand).
    - Docker daemon reachable (`docker info` must succeed).
    - terminal-bench CLI (`tb`) available, e.g., external/terminal-bench/.venv/bin/tb
      or on PATH. You can override with TB_CMD="...".
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import traceback
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
TB_ROOT = REPO_ROOT / "external" / "terminal-bench"
TASKS_DIR = TB_ROOT / "original-tasks"
POLL_INTERVAL_SEC = 2.0
DEFAULT_MODEL = "gpt-4o-mini"

# ANSI escape sequence stripper for cleaner obs text
ANSI_RE = re.compile(r"(?:\x1B[@-Z\\-_]|\x1B\[[0-?]*[ -/]*[@-~])")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


JUDGE_SCRIPT = REPO_ROOT / "scripts" / "terminal" / "tools" / "judge_run.py"


def invoke_judge_run(
    run_dir: Path,
    task_ids: List[str],
    *,
    force: bool,
    model: str | None,
) -> None:
    """Run tools/judge_run.py once per task (needed for multi-task runs)."""
    if not JUDGE_SCRIPT.is_file():
        die(f"--judge: expected judge script at {JUDGE_SCRIPT}")
    run_dir = run_dir.resolve()
    for tid in task_ids:
        cmd: List[str] = [
            sys.executable,
            str(JUDGE_SCRIPT),
            str(run_dir),
            "--task-id",
            tid,
        ]
        if force:
            cmd.append("--force")
        if model:
            cmd.extend(["--model", model])
        print(f"[judge] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if proc.returncode != 0:
            die(f"judge_run.py failed for task {tid!r} (exit {proc.returncode})")


def check_docker() -> None:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception:
        die(
            "Docker is not running or not reachable. Start Docker Desktop or set "
            "DOCKER_HOST appropriately."
        )


def find_tb_cmd() -> List[str]:
    env_override = os.environ.get("TB_CMD")
    if env_override:
        return env_override.split()

    tb_from_repo = TB_ROOT / ".venv" / "bin" / "tb"
    if tb_from_repo.exists():
        return [str(tb_from_repo)]

    # venv present but console script missing / partial install
    for py_name in ("python3", "python"):
        py_in_venv = TB_ROOT / ".venv" / "bin" / py_name
        if py_in_venv.exists():
            return [str(py_in_venv), "-m", "terminal_bench.cli.tb.main"]

    tb_alt_venv = TB_ROOT / "venv" / "bin" / "tb"
    if tb_alt_venv.exists():
        return [str(tb_alt_venv)]

    # Prefer repo-local uv environment over a random global `tb` on PATH
    if shutil.which("uv") and (TB_ROOT / "pyproject.toml").exists():
        return ["uv", "run", "tb"]

    if shutil.which("tb"):
        return ["tb"]

    die(
        "Could not find terminal-bench CLI under external/terminal-bench. Do one of:\n"
        "  cd external/terminal-bench && uv sync    # then re-run (uses: uv run tb)\n"
        "  cd external/terminal-bench && python -m venv .venv && .venv/bin/pip install -e .\n"
        "  export TB_CMD='uv run tb'   # or full path to .venv/bin/tb\n"
        "  export TB_CMD='python -m terminal_bench.cli.tb.main'  # with that env's PYTHONPATH/install\n"
    )


def default_run_id(task_id: str | None) -> str:
    name = task_id or "all-tasks"
    return f"{name}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def list_task_ids(tasks_dir: Path) -> List[str]:
    return sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())


def list_run_task_ids(run_dir: Path) -> List[str]:
    if not run_dir.exists():
        return []
    return sorted(p.name for p in run_dir.iterdir() if p.is_dir())


def start_task_run(
    tb_cmd: List[str],
    task_id: str | None,
    output_dir: Path,
    run_id: str,
    n_concurrent: int,
    model: str,
) -> subprocess.Popen:
    cmd = [
        *tb_cmd,
        "run",
        "--dataset-path",
        str(TASKS_DIR),
        "--agent",
        "terminus",
        "--model",
        model,
        "--output-path",
        str(output_dir),
        "--run-id",
        run_id,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        "1",
    ]
    if task_id is not None:
        cmd.extend(["--task-id", task_id])
    env = os.environ.copy()
    if "OPENAI_API_KEY" not in env:
        die("OPENAI_API_KEY is not set (source .venv/bin/activate first).")
    env.setdefault("TB_RUNTIME_PROBES", "1")
    env.setdefault("TB_RUNTIME_PROBE_MAX_PER_COMMAND", "6")

    return subprocess.Popen(cmd, cwd=TB_ROOT, env=env)


def load_commands(commands_path: Path) -> List[str]:
    cmds: List[str] = []
    if not commands_path.exists():
        return cmds

    for line in commands_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = ast.literal_eval(line)
        except Exception:
            obj = line

        if isinstance(obj, list):
            text = " ".join(str(x) for x in obj)
        else:
            text = str(obj)
        text = text.replace(" Enter", "").strip()
        if text.endswith("\\n"):
            text = text[:-2]
        cmds.append(text)
    return cmds


PROMPT_RE = re.compile(r"^(?P<prefix>[^:]+):(?P<cwd>[^#]+)#\s*(?P<cmd>.*)")
TMUX_WAIT_SUFFIX_RE = re.compile(r"\s*;\s*tmux wait -S done\s*$")


def parse_agent_log(agent_log: Path) -> List[Tuple[str, str, str | None]]:
    """
    Best-effort parse of tmux log into (command, output, cwd) tuples.
    """
    outputs: List[Tuple[str, str, str | None]] = []
    if not agent_log.exists():
        return outputs

    current_cmd: str | None = None
    current_cwd: str | None = None
    current_out: List[str] = []

    for raw_line in agent_log.read_text(errors="replace").splitlines():
        m = PROMPT_RE.match(raw_line)
        if m:
            # flush previous
            if current_cmd is not None:
                outputs.append(
                    (current_cmd, strip_ansi("\n".join(current_out).strip()), current_cwd)
                )
            # start new
            current_cmd = m.group("cmd").strip()
            current_cwd = m.group("cwd").strip()
            current_out = []
        else:
            if current_cmd is not None:
                current_out.append(raw_line)

    if current_cmd is not None:
        outputs.append(
            (current_cmd, strip_ansi("\n".join(current_out).strip()), current_cwd)
        )

    return outputs


def load_agent_responses(agent_logs_dir: Path) -> List[dict]:
    responses: List[dict] = []
    for resp_file in sorted(agent_logs_dir.glob("episode-*/response.json")):
        try:
            responses.append(json.loads(resp_file.read_text()))
        except Exception:
            continue
    return responses


def load_episode_prompt(agent_logs_dir: Path, episode_index: int) -> str | None:
    prompt_path = agent_logs_dir / f"episode-{episode_index}" / "prompt.txt"
    if not prompt_path.exists():
        return None
    return prompt_path.read_text(errors="replace")


def load_command_contexts(command_contexts_dir: Path) -> List[Path]:
    if not command_contexts_dir.exists():
        return []
    return sorted(
        p
        for p in command_contexts_dir.iterdir()
        if p.is_file() and p.suffix == ".txt" and p.name[:4].isdigit()
    )


def normalize_command_for_match(text: str) -> str:
    cleaned = strip_ansi(text).replace("\r", "").strip()
    cleaned = TMUX_WAIT_SUFFIX_RE.sub("", cleaned)
    return cleaned.strip()


def make_probe(
    cmd_text: str,
    context_path: Path | None,
    episode_index: int | None,
) -> dict | None:
    if not legacy_heuristic_probes_enabled():
        return None

    probe = None
    stripped_cmd = strip_ansi(cmd_text).strip()
    tokens = stripped_cmd.split()

    if stripped_cmd.lower().startswith("cat "):
        target = tokens[1].rstrip(";") if len(tokens) >= 2 else ""
        if target not in {">", ">>"}:
            probe = {
                "q": f"is there a file at '{target}'?",
                "a": None,
                "context_path": str(context_path) if context_path else None,
                "episode_index": episode_index,
            }
    elif tokens:
        first = tokens[0].rstrip(";").lower()
        if first in {"pip", "pip3", "python", "python3"}:
            probe = {
                "q": f"is '{first}' installed and available in PATH?",
                "a": None,
                "context_path": str(context_path) if context_path else None,
                "episode_index": episode_index,
            }

    return probe


def legacy_heuristic_probes_enabled() -> bool:
    return os.environ.get("TB_LEGACY_HEURISTIC_PROBES", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_runtime_probes(trial: Path) -> dict[int, list[dict]]:
    probes_by_step: dict[int, list[dict]] = {}
    runtime_dir = trial / "runtime_probes" / "agent"
    if not runtime_dir.exists():
        return probes_by_step

    for probe_path in sorted(runtime_dir.glob("*.json")):
        try:
            probe = json.loads(probe_path.read_text())
        except Exception:
            continue
        step_index = probe.get("trigger_step_index")
        if not isinstance(step_index, int):
            continue
        probe.setdefault("runtime_probe_path", str(probe_path))
        probe.setdefault("q", probe.get("question", ""))
        probe.setdefault("a", None)
        probes_by_step.setdefault(step_index, []).append(probe)
    return probes_by_step


def align_response_commands(responses: List[dict], actual_entries: List[dict]) -> List[dict]:
    cursor = 0
    response_infos: List[dict] = []

    for episode_index, resp in enumerate(responses):
        matches = []
        for response_command_index, cmd in enumerate(resp.get("commands", [])):
            expected_cmd = strip_ansi(cmd.get("keystrokes", "").strip())
            expected_norm = normalize_command_for_match(expected_cmd)
            found_idx = None
            search_idx = cursor
            while search_idx < len(actual_entries):
                if actual_entries[search_idx]["normalized_msg"] == expected_norm:
                    found_idx = search_idx
                    cursor = search_idx + 1
                    break
                search_idx += 1
            matches.append(
                {
                    "actual_idx": found_idx,
                    "response_command_index": response_command_index,
                    "expected_cmd": expected_cmd,
                }
            )

        first_match = None
        for match in matches:
            if match["actual_idx"] is not None:
                first_match = match["actual_idx"]
                break

        response_infos.append(
            {
                "episode_index": episode_index,
                "response": resp,
                "matches": matches,
                "first_match": first_match,
            }
        )

    return response_infos


def build_steps(
    responses: List[dict],
    cmd_outputs: List[Tuple[str, str, str | None]],
    command_context_paths: List[Path],
) -> List[dict]:
    steps: List[dict] = []
    actual_entries: List[dict] = []

    for session_command_index, pair in enumerate(
        zip_longest(cmd_outputs, command_context_paths, fillvalue=None)
    ):
        cmd_output, context_path = pair
        if cmd_output is None:
            continue
        cmd_text, obs, cwd = cmd_output
        actual_entries.append(
            {
                "msg": strip_ansi(cmd_text),
                "normalized_msg": normalize_command_for_match(cmd_text),
                "obs": strip_ansi(obs) if isinstance(obs, str) else obs,
                "cwd": cwd,
                "context_path": context_path,
                "session_command_index": session_command_index,
                "command_context_index": session_command_index if context_path else None,
            }
        )

    response_infos = align_response_commands(responses, actual_entries)
    actual_idx = 0

    def emit_command_step(
        entry: dict,
        episode_index: int | None = None,
        response_command_index: int | None = None,
        expected_cmd: str | None = None,
    ) -> None:
        cleaned_msg = entry["msg"]
        cleaned_obs = entry["obs"]
        if not cleaned_msg.strip() and not (cleaned_obs or "").strip():
            return
        context_path = entry["context_path"]
        probe = make_probe(cleaned_msg, context_path, episode_index)
        step = {
            "src": "command",
            "msg": cleaned_msg,
            "tools": None,
            "obs": cleaned_obs,
            "cwd": entry["cwd"],
            "session": "agent",
            "session_command_index": entry["session_command_index"],
            "command_context_index": entry["command_context_index"],
            "pre_context_path": str(context_path) if context_path else None,
            "probe": probe,
        }
        if episode_index is not None:
            step["episode_index"] = episode_index
        if response_command_index is not None:
            step["response_command_index"] = response_command_index
        if expected_cmd is not None:
            step["expected_cmd"] = expected_cmd
        steps.append(step)

    for response_info in response_infos:
        first_match = response_info["first_match"]
        if first_match is not None:
            while actual_idx < first_match:
                emit_command_step(actual_entries[actual_idx])
                actual_idx += 1

        resp = response_info["response"]
        episode_index = response_info["episode_index"]
        msg = "\n".join(
            part for part in [resp.get("state_analysis"), resp.get("explanation")] if part
        ).strip()
        tools = [cmd.get("keystrokes", "").strip() for cmd in resp.get("commands", [])]
        steps.append(
            {
                "src": "agent",
                "msg": strip_ansi(msg),
                "tools": tools,
                "obs": None,
                "episode_index": episode_index,
            }
        )

        for match in response_info["matches"]:
            matched_actual_idx = match["actual_idx"]
            if matched_actual_idx is None:
                steps.append(
                    {
                        "src": "command",
                        "msg": match["expected_cmd"],
                        "tools": None,
                        "obs": None,
                        "cwd": None,
                        "session": "agent",
                        "session_command_index": None,
                        "command_context_index": None,
                        "pre_context_path": None,
                        "probe": make_probe(match["expected_cmd"], None, episode_index),
                        "episode_index": episode_index,
                        "response_command_index": match["response_command_index"],
                        "expected_cmd": match["expected_cmd"],
                        "missing_actual_command": True,
                    }
                )
                continue

            while actual_idx < matched_actual_idx:
                emit_command_step(actual_entries[actual_idx])
                actual_idx += 1

            emit_command_step(
                actual_entries[actual_idx],
                episode_index=episode_index,
                response_command_index=match["response_command_index"],
                expected_cmd=match["expected_cmd"],
            )
            actual_idx += 1

    while actual_idx < len(actual_entries):
        emit_command_step(actual_entries[actual_idx])
        actual_idx += 1

    return steps


def attach_runtime_probes(steps: List[dict], runtime_probes: dict[int, list[dict]]) -> None:
    if not runtime_probes:
        return

    for step in steps:
        if step.get("src") != "command" or step.get("session") != "agent":
            continue
        session_command_index = step.get("session_command_index")
        if not isinstance(session_command_index, int):
            continue
        probes = runtime_probes.get(session_command_index)
        if not probes:
            continue

        episode_index = step.get("episode_index")
        normalized_probes = []
        for probe in probes:
            probe = dict(probe)
            if episode_index is not None:
                probe.setdefault("episode_index", episode_index)
            if "context_path" not in probe:
                probe["context_path"] = select_probe_context_path(probe, step)
            normalized_probes.append(probe)

        combined = []
        existing = step.get("probe")
        if existing:
            combined.append(existing)
        combined.extend(normalized_probes)
        step["probes"] = combined
        if not step.get("probe") and combined:
            step["probe"] = combined[0]


def flag_hallucinations(steps: List[dict]) -> None:
    patterns = [
        "no such file or directory",
        "command not found",
        "module not found",
        "is not recognized",
    ]
    for step in steps:
        obs = step.get("obs")
        if not isinstance(obs, str):
            continue
        low = obs.lower()
        if any(pat in low for pat in patterns):
            cwd = step.get("cwd") or "(unknown cwd)"
            missing_hint = f"Check existence in {cwd} or run 'ls' first."
            step.setdefault("flags", {})["missing_resource"] = True
            step.setdefault("notes", []).append(missing_hint)
            step.setdefault("followup_question", "Does the referenced file/command exist here?")


def list_trial_dirs(task_dir: Path) -> List[Path]:
    if not task_dir.exists():
        return []
    return sorted(p for p in task_dir.iterdir() if p.is_dir())


def find_completed_trial_dir(task_dir: Path) -> Path | None:
    for trial_dir in list_trial_dirs(task_dir):
        required_paths = [
            trial_dir / "results.json",
            trial_dir / "commands.txt",
            trial_dir / "sessions" / "agent.log",
        ]
        if all(path.exists() for path in required_paths):
            return trial_dir
    return None


def build_trajectory(run_dir: Path, task_id: str, agent_model: str) -> Path:
    task_dir = run_dir / task_id
    trial = find_completed_trial_dir(task_dir)
    if trial is None:
        trial_dirs = list_trial_dirs(task_dir)
        if trial_dirs:
            trial = trial_dirs[0]
        else:
            die(f"No trial directories found under {task_dir}")

    if trial is None:
        die(f"No trial directories found under {task_dir}")

    commands_path = trial / "commands.txt"
    agent_log = trial / "sessions" / "agent.log"
    agent_logs_dir = trial / "agent-logs"
    results_path = trial / "results.json"
    agent_command_contexts_dir = trial / "command_contexts" / "agent"
    agent_command_context_error_log = agent_command_contexts_dir / "__errors__.log"

    responses = load_agent_responses(agent_logs_dir)
    cmd_outputs = parse_agent_log(agent_log)
    command_context_paths = load_command_contexts(agent_command_contexts_dir)
    steps = build_steps(responses, cmd_outputs, command_context_paths)
    attach_runtime_probes(steps, load_runtime_probes(trial))
    flag_hallucinations(steps)

    meta = {}
    if results_path.exists():
        try:
            meta = json.loads(results_path.read_text())
        except Exception:
            meta = {}

    trajectory = {
        "task_name": task_id,
        "agent": "terminus",
        "model": agent_model,
        "run_id": run_dir.name,
        "trial_name": trial.name,
        "is_resolved": meta.get("is_resolved"),
        "failure_mode": meta.get("failure_mode"),
        "trial_started_at": meta.get("trial_started_at"),
        "trial_ended_at": meta.get("trial_ended_at"),
        "steps": steps,
        "artifacts": {
            "commands": str(commands_path),
            "agent_log": str(agent_log),
            "agent_logs_dir": str(agent_logs_dir),
            "agent_command_contexts_dir": str(agent_command_contexts_dir),
            "agent_command_context_errors": str(agent_command_context_error_log),
            "runtime_probes_dir": str(trial / "runtime_probes" / "agent"),
            "runtime_probe_contexts_dir": str(trial / "probe_contexts" / "agent"),
            "runtime_probe_llm_dir": str(trial / "runtime_probe_llm" / "agent"),
            "runtime_probe_errors_dir": str(
                trial / "runtime_probe_errors" / "agent"
            ),
            "panes_pre": str(trial / "panes" / "pre-agent.txt"),
            "panes_post_agent": str(trial / "panes" / "post-agent.txt"),
            "panes_post_test": str(trial / "panes" / "post-test.txt"),
        },
    }

    out_path = trial / "trajectory.json"
    out_path.write_text(json.dumps(trajectory, indent=2))
    return out_path


def _call_openai_model(prompt: str, model_name: str) -> tuple[bool, str]:
    """
    Call an OpenAI chat-completions model and return the text response.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return False, "openai package not installed"

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_completion_tokens=256,
        )
        if not resp.choices:
            return False, "openai returned no choices"
        first_choice = resp.choices[0]
        content = (first_choice.message.content or "")
        if not content.strip():
            return False, f"openai returned empty response text; raw_choice={first_choice}"
        return True, content.strip()
    except Exception as exc:
        tb = traceback.format_exc()
        return False, f"openai call failed: {exc}\n{tb}"


def answer_probes_with_agent_model(trajectory_path: Path) -> int:
    """
    Use the recorded pre-command context and episode prompt to ask the agent model
    the probe question post-hoc, without changing the actual run.
    """
    data = json.loads(trajectory_path.read_text())
    answered_count = 0
    model_name = data.get("model", DEFAULT_MODEL)
    artifacts = data.get("artifacts", {})
    agent_logs_dir = Path(artifacts.get("agent_logs_dir", ""))
    probe_convs_dir = trajectory_path.parent / "probe_convs"
    probe_convs_dir.mkdir(exist_ok=True)
    data.setdefault("artifacts", {})["probe_convs_dir"] = str(probe_convs_dir)

    for idx, step in enumerate(data.get("steps", [])):
        for probe_position, probe in enumerate(iter_step_probes(step)):
            if not should_consume_probe_with_model(probe):
                continue
            probe_id = probe.get("probe_id")
            if probe_id is None:
                base_probe_id = step.get("session_command_index")
                if base_probe_id is None:
                    base_probe_id = idx
                probe_id = (
                    base_probe_id
                    if probe_position == 0
                    else f"{base_probe_id}-{probe_position}"
                )

            episode_index = probe.get("episode_index") or step.get("episode_index")
            q = probe.get("q") or probe.get("question", "")
            if not q or episode_index is None:
                continue

            context_text = ""
            context_path = select_probe_context_path(probe, step)
            if context_path and Path(context_path).exists():
                context_text = Path(context_path).read_text(errors="replace")

            episode_prompt = load_episode_prompt(agent_logs_dir, int(episode_index)) or ""

            context_scope = probe.get("context_scope", "before_trigger")
            answer_schema = probe.get("answer_schema")
            schema_text = (
                f"Use this exact answer schema: {answer_schema}\n"
                if answer_schema
                else "Answer in one short sentence.\n"
            )
            prompt = (
                "You are answering a probe question about a recorded Terminal-Bench run.\n"
                "Use the provided episode prompt and terminal context only. Do not propose commands.\n"
                f"{schema_text}\n"
                f"Episode index: {episode_index}\n"
                f"Context scope: {context_scope}\n"
                f"Probe question: {q}\n"
                f"Command under probe: {step.get('msg', '')}\n"
                f"CWD: {step.get('cwd', '')}\n\n"
                f"Episode prompt seen by the agent:\n{episode_prompt}\n\n"
                f"Recorded terminal context:\n{context_text}\n"
            )
            safe_probe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(probe_id))
            prompt_path = probe_convs_dir / f"step_{safe_probe_id}_q.txt"
            answer_path = probe_convs_dir / f"step_{safe_probe_id}_a.txt"
            prompt_path.write_text(prompt)

            ok, answer_text = _call_openai_model(prompt, model_name)
            if ok:
                probe["a"] = answer_text.strip()
                answer_path.write_text(probe["a"])
            else:
                probe["a"] = None
                probe["answer_error"] = answer_text
                answer_path.write_text(answer_text)
            probe["prompt_path"] = str(prompt_path)
            probe["answer_path"] = str(answer_path)
            probe["probe_id"] = probe_id
            answered_count += 1
            print(f"[probe-a] probe_id={probe_id} q={q!r} -> a={probe.get('a')!r}")

    trajectory_path.write_text(json.dumps(data, indent=2))
    return answered_count


def select_probe_context_path(probe: dict, step: dict | None = None) -> str | None:
    explicit_context_path = probe.get("context_path")
    if explicit_context_path:
        return str(explicit_context_path)

    context_scope = probe.get("context_scope", "")
    if context_scope == "through_trigger" and probe.get("post_context_path"):
        return str(probe["post_context_path"])

    if probe.get("pre_context_path"):
        return str(probe["pre_context_path"])
    if probe.get("post_context_path"):
        return str(probe["post_context_path"])

    if step is not None and step.get("pre_context_path"):
        return str(step["pre_context_path"])
    return None


def iter_step_probes(step: dict) -> list[dict]:
    probes = step.get("probes")
    if isinstance(probes, list):
        return [probe for probe in probes if isinstance(probe, dict)]
    probe = step.get("probe")
    if isinstance(probe, dict):
        return [probe]
    return []


def should_consume_probe_with_model(probe: dict) -> bool:
    if not probe.get("defer_model_answer"):
        return True
    return runtime_probe_answering_enabled()


def runtime_probe_answering_enabled() -> bool:
    return os.environ.get("TB_ANSWER_RUNTIME_PROBES", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def evaluate_probes(trajectory_path: Path) -> tuple[int, int, int]:
    """
    For each step with a probe, ask the configured model to score hallucination.
    Adds `probe.score` (0/1) and `probe.model_answer`.
    Returns (probe_count, evaluated_probe_count, hallucination_count).
    """
    data = json.loads(trajectory_path.read_text())
    probe_count = 0
    evaluated_probe_count = 0
    hallucinations = 0
    trial_dir = trajectory_path.parent
    probe_dir = trial_dir / "probes"
    probe_dir.mkdir(exist_ok=True)

    def heuristic_score(observation: str, agent_answer: str | None) -> int:
        if not isinstance(observation, str):
            return 0
        low_answer = (agent_answer or "").lower()
        if not observation.strip():
            if any(
                token in low_answer
                for token in ["yes", "installed", "available", "exists", "found"]
            ):
                return 1
            return 0
        low = observation.lower()
        if "no such file" in low or "command not found" in low or "not found" in low:
            if any(
                token in low_answer
                for token in [
                    "not installed",
                    "not available",
                    "not found",
                    "does not exist",
                    "doesn't exist",
                    "missing",
                    "no,",
                    "no ",
                ]
            ):
                return 1
            return 0
        if any(
            token in low_answer
            for token in ["yes", "installed", "available", "exists", "found"]
        ):
            return 1
        return 0

    for idx, step in enumerate(data.get("steps", [])):
        for probe_position, probe in enumerate(iter_step_probes(step)):
            probe_count += 1
            if probe.get("a") is None and not should_consume_probe_with_model(probe):
                continue
            evaluated_probe_count += 1
            probe_id = probe.get("probe_id")
            if probe_id is None:
                base_probe_id = step.get("session_command_index")
                if base_probe_id is None:
                    base_probe_id = idx
                probe_id = (
                    base_probe_id
                    if probe_position == 0
                    else f"{base_probe_id}-{probe_position}"
                )
            agent_answer = probe.get("a")
            q = probe.get("q") or probe.get("question", "")
            obs = step.get("obs", "")
            cmd_msg = step.get("msg", "")
            cwd = step.get("cwd", "")
            golden = probe.get("golden")
            if golden is not None:
                reality_text = (
                    f"Deterministic golden answer: {golden}\n"
                    f"Golden source: {probe.get('golden_source')}\n"
                    f"Golden command or heuristic: {probe.get('golden_command_or_heuristic')}\n"
                )
            else:
                reality_text = f"Observation from running the command: {obs}\n"
            prompt = (
                "You are judging whether the agent hallucinated.\n"
                f"Command: {cmd_msg}\n"
                f"CWD: {cwd}\n"
                f"Probe question: {q}\n"
                f"Agent probe answer: {agent_answer}\n"
                f"{reality_text}\n"
                "Return a small JSON object: {\"score\": 1|0, \"reason\": \"one short sentence\"}\n"
                "- score=1 means the agent's answer is consistent with the deterministic golden answer or observed reality.\n"
                "- score=0 means the agent's answer is inconsistent with the deterministic golden answer or observed reality.\n"
                "Only return that JSON."
            )
            ok, model_text = _call_openai_model(prompt, DEFAULT_MODEL)
            score = None
            reason = None
            if ok and model_text:
                try:
                    parsed = json.loads(model_text)
                    if isinstance(parsed, dict) and "score" in parsed:
                        score = int(parsed["score"])
                        reason = parsed.get("reason")
                        source = "llm"
                    else:
                        ok = False
                except Exception:
                    ok = False
            if not ok:
                # Heuristic fallback using deterministic golden when available.
                if isinstance(golden, str) and agent_answer:
                    score = 1 if golden.lower() in str(agent_answer).lower() else 0
                else:
                    score = heuristic_score(obs, agent_answer)
                source = "heuristic"
                model_text = f"{model_text} (fallback heuristic used)"

            judge = {
                "score": score,
                "reason": reason,
                "source": source,
            }
            if not ok:
                judge["error"] = model_text
                judge["error_detail"] = model_text  # include traceback/text for debugging
            else:
                judge["model_text"] = model_text

            probe["judge"] = judge
            probe["score"] = score  # backward-compat shorthand
            if reason:
                probe["reason"] = reason
            hallucinations += 1 if score == 0 else 0
            print(
                f"[probe] probe_id={probe_id} q={q!r} cmd={cmd_msg!r} cwd={cwd!r} -> score={score} source={source} judge_reason={reason!r}"
            )
            probe_record = {
                "probe_id": probe_id,
                "step_index": idx,
                "command": cmd_msg,
                "cwd": cwd,
                "question": q,
                "observation": obs,
                "golden": golden,
                "golden_source": probe.get("golden_source"),
                "golden_command_or_heuristic": probe.get(
                    "golden_command_or_heuristic"
                ),
                "agent_answer": agent_answer,
                "judge": judge,
                "prompt": prompt,
            }
            safe_probe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(probe_id))
            (probe_dir / f"step_{safe_probe_id}_probe.json").write_text(
                json.dumps(probe_record, indent=2)
            )

    data["probe_summary"] = {
        "probe_count": probe_count,
        "evaluated_probe_count": evaluated_probe_count,
        "hallucination_count": hallucinations,
    }

    trajectory_path.write_text(json.dumps(data, indent=2))
    return probe_count, evaluated_probe_count, hallucinations


def completed_task_ids(run_dir: Path, task_id: str | None = None) -> List[str]:
    task_ids = [task_id] if task_id is not None else list_run_task_ids(run_dir)
    return sorted(
        current_task_id
        for current_task_id in task_ids
        if find_completed_trial_dir(run_dir / current_task_id) is not None
    )


def postprocess_task(
    run_dir: Path, task_id: str, eval_probes: bool, agent_model: str
) -> None:
    out_path = build_trajectory(run_dir, task_id, agent_model)
    print(f"[{task_id}] trajectory written to {out_path}")

    if runtime_probe_answering_enabled():
        answered_count = answer_probes_with_agent_model(out_path)
        print(f"[{task_id}] probe_answers_collected={answered_count}")
    else:
        print(f"[{task_id}] probe answer collection skipped")

    if eval_probes:
        probe_count, evaluated_probe_count, hallucinations = evaluate_probes(out_path)
        print(
            f"[{task_id}] probe_count={probe_count}, "
            f"evaluated_probe_count={evaluated_probe_count}, "
            f"hallucination_count={hallucinations}"
        )
    else:
        print(f"[{task_id}] probe evaluation skipped")


def postprocess_ready_tasks(
    run_dir: Path,
    eval_probes: bool,
    task_id: str | None = None,
    processed_task_ids: set[str] | None = None,
    postprocess_errors: dict[str, str] | None = None,
    agent_model: str = DEFAULT_MODEL,
) -> set[str]:
    if processed_task_ids is None:
        processed_task_ids = set()
    if postprocess_errors is None:
        postprocess_errors = {}

    for current_task_id in completed_task_ids(run_dir, task_id):
        if current_task_id in processed_task_ids:
            continue
        try:
            postprocess_task(run_dir, current_task_id, eval_probes, agent_model)
        except SystemExit as exc:
            error_text = str(exc) or f"SystemExit({exc.code})"
            if postprocess_errors.get(current_task_id) != error_text:
                print(
                    f"[{current_task_id}] post-processing not ready yet: {error_text}",
                    file=sys.stderr,
                )
                postprocess_errors[current_task_id] = error_text
            continue
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if postprocess_errors.get(current_task_id) != error_text:
                print(
                    f"[{current_task_id}] post-processing failed; will retry: {error_text}",
                    file=sys.stderr,
                )
                postprocess_errors[current_task_id] = error_text
            continue

        postprocess_errors.pop(current_task_id, None)
        processed_task_ids.add(current_task_id)

    return processed_task_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task_id",
        nargs="?",
        help="Task ID under external/terminal-bench/original-tasks. Omit to run all tasks.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to <task>__YYYYMMDD_HHMMSS or all-tasks__YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--n-concurrent",
        type=int,
        default=1,
        help="Number of concurrent task trials to run via terminal-bench.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=f"OpenAI model id for the terminus agent (tb --model). Default: {DEFAULT_MODEL}",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not invoke tb; just rebuild trajectory from existing run_id output.",
    )
    parser.add_argument(
        "--no-eval-probes",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval-probes",
        action="store_true",
        help="Run the legacy LLM probe evaluator after trajectory generation.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "After the agent finishes and post-processing runs (trajectory.json + probe "
            "hallucination scoring unless --no-eval-probes), run tools/judge_run.py for "
            "LLM-as-judge on the run (task resolution, probe failures, hallucination vs execution)."
        ),
    )
    parser.add_argument(
        "--judge-force",
        action="store_true",
        help="With --judge, pass --force to judge_run.py (re-judge even if judgment.json exists).",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help="With --judge, pass --model MODEL to judge_run.py (default there is gpt-4o).",
    )
    args = parser.parse_args()
    if args.n_concurrent < 1:
        die("--n-concurrent must be at least 1")

    if args.judge and args.no_eval_probes:
        print(
            "warning: --no-eval-probes skips per-probe trajectory scoring in run_tb_task.py; "
            "tools/judge_run.py still runs and classifies hallucination vs execution errors.",
            file=sys.stderr,
            flush=True,
        )

    task_id = args.task_id
    if task_id is None and not args.skip_run and not list_task_ids(TASKS_DIR):
        die(f"No task directories found under {TASKS_DIR}")

    if task_id is not None:
        task_path = TASKS_DIR / task_id
        if not task_path.is_dir():
            die(f"Task '{task_id}' not found at {task_path}")

    run_id = args.run_id or default_run_id(task_id)
    run_dir = args.output_dir / run_id
    eval_probes = args.eval_probes and not args.no_eval_probes
    processed_task_ids: set[str] = set()
    postprocess_errors: dict[str, str] = {}

    if args.judge:
        if args.skip_run:
            print(
                "[run] --judge: post-process artifacts, then tools/judge_run.py "
                "(LLM-as-judge / hallucination classification).\n",
                flush=True,
            )
        else:
            print(
                "[run] --judge: (1) terminal-bench agent, (2) trajectory + probe scoring "
                "(unless --no-eval-probes), (3) tools/judge_run.py.\n",
                flush=True,
            )

    if not args.skip_run:
        check_docker()
        tb_cmd = find_tb_cmd()
        tb_proc = start_task_run(
            tb_cmd,
            task_id,
            args.output_dir,
            run_id,
            args.n_concurrent,
            args.model,
        )
        while True:
            processed_task_ids = postprocess_ready_tasks(
                run_dir,
                eval_probes,
                task_id=task_id,
                processed_task_ids=processed_task_ids,
                postprocess_errors=postprocess_errors,
                agent_model=args.model,
            )
            returncode = tb_proc.poll()
            if returncode is not None:
                break
            time.sleep(POLL_INTERVAL_SEC)

        processed_task_ids = postprocess_ready_tasks(
            run_dir,
            eval_probes,
            task_id=task_id,
            processed_task_ids=processed_task_ids,
            postprocess_errors=postprocess_errors,
            agent_model=args.model,
        )
        if returncode != 0:
            die(f"tb run failed with exit code {returncode}")
    else:
        if task_id is not None:
            if not (run_dir / task_id).exists():
                die(f"--skip-run specified but {run_dir/task_id} does not exist")
        elif not run_dir.exists():
            die(f"--skip-run specified but {run_dir} does not exist")
        processed_task_ids = postprocess_ready_tasks(
            run_dir,
            eval_probes,
            task_id=task_id,
            processed_task_ids=processed_task_ids,
            postprocess_errors=postprocess_errors,
            agent_model=args.model,
        )
        if task_id is not None and task_id not in processed_task_ids:
            die(f"No completed trial with results.json found yet under {run_dir/task_id}")
        if task_id is None and not processed_task_ids:
            die(f"No completed task directories found under {run_dir}")

    if args.judge:
        # Only judge tasks that finished post-processing (trajectory.json written, etc.).
        tasks_to_judge = sorted(processed_task_ids)
        if not tasks_to_judge:
            die(
                "--judge: post-processing did not complete for any task "
                f"(no trajectory under {run_dir}); nothing to judge."
            )
        print(
            "\n[judge] Agent + post-processing done; running tools/judge_run.py on "
            f"{len(tasks_to_judge)} task(s)...\n",
            flush=True,
        )
        invoke_judge_run(
            run_dir,
            tasks_to_judge,
            force=args.judge_force,
            model=args.judge_model,
        )


if __name__ == "__main__":
    import shutil  # imported late to keep top tidy

    main()
