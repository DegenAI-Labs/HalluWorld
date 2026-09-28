#!/usr/bin/env python3
"""NON-FUNCTIONAL -- kept for reference, not wired into the benchmark.

This script has never been importable. It does:

    from halluworld.serializers.omniscient import OmniscientSerializer

and no module or class by that name has ever existed on any branch of this
repository -- it was broken in the commit that introduced it (00d1f5c, "add
level editor and trajectory recorder tools"). It is retained rather than
deleted because the surrounding logic is real work and someone may want to
finish it, but it is deliberately outside the installable package so that
`import halluworld` cannot fail because of it.

To revive it, implement OmniscientSerializer (a serializer exposing full
ground-truth world state rather than the agent's field of view) and re-test
against scripts/record_trajectory.py output.
"""

"""Replay-based eval for hand-recorded trajectories.

Loads a trajectory JSON produced by record_trajectory.py, replays the actions
step-by-step to build a multi-observation prompt, then queries an LM on each
probe question.

Trajectory JSON format (from record_trajectory.py):
    {
      "segments": [
        {"level_file": str(LEVELS_DIR / "h1_foyer.txt"), "seed": 42, "actions": [1, 2, ...]},
        {"level_file": str(LEVELS_DIR / "h2_kitchen.txt"), "seed": 42, "actions": [...]}
      ],
      "probes": [
        {"segment": 0, "step": 18, "probe_type": "presence",
         "question": "Was there a key in this room?",
         "ground_truth": "yes", "metadata": {}}
      ]
    }

Old single-room format (level_file/seed/actions at top level) is auto-upgraded.

Usage
-----
# Single trajectory file, all probes:
conda run -n halluworld python run_trajectory_eval.py \\
    --traj trajectories/h1_foyer_s42.json \\
    --models gpt-4o-mini gpt-4o

# All trajectory files in a folder:
conda run -n halluworld python run_trajectory_eval.py \\
    --traj trajectories/ \\
    --models gpt-4o claude-sonnet-4-6

# Verbose output to see Q/A:
conda run -n halluworld python run_trajectory_eval.py \\
    --traj trajectories/h1_foyer_s42.json \\
    --models gpt-4o-mini --verbose

Results are appended to results/trajectory_results.csv (or --out).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.memory import MemorySerializer
from halluworld.serializers.omniscient import OmniscientSerializer

# Reuse LM routing and scoring from the main eval script
from halluworld.tracks.grid.perception import _make_lm, score_response  # noqa: E402
from halluworld.data import LEVELS_DIR

_ser = MemorySerializer()
_ser_omni = OmniscientSerializer()


def _normalize_gt(probe_type: str, gt):
    """Convert trajectory JSON ground-truth strings to the types expected by score_response.

    run_perception_eval.py uses Python bool for presence, int for count.
    Trajectory JSONs store everything as strings ("yes"/"no"/"3"/etc.).
    """
    if probe_type == "presence":
        if isinstance(gt, bool):
            return gt
        return str(gt).strip().lower() in ("yes", "true", "1")
    if probe_type == "count":
        if isinstance(gt, int):
            return gt
        try:
            return int(str(gt).strip())
        except ValueError:
            return gt
    return gt


_ZERO_PATTERNS = re.compile(
    r"\b(no|none|zero|there (were|are|was) no|not (any|mentioned)|0)\b",
    re.IGNORECASE,
)


def _is_zero_response(text: str) -> bool:
    """Return True if text clearly expresses a count of zero."""
    return bool(_ZERO_PATTERNS.search(text))


# ── Action labels for readable prompt headers ─────────────────────────────────
_ACTION_LABEL: dict[int, str] = {
    0: "turned left",
    1: "turned right",
    2: "moved forward",
    3: "picked up",
    4: "dropped",
    5: "toggled",
}

SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld manor. "
    "You are shown a sequence of observations as you moved through the room. "
    "Each observation shows what you can see at that point in time. "
    "Answer each question precisely based only on what is explicitly stated "
    "across all your observations. "
    "If something was visible in an earlier observation but not the current one, "
    "it may still exist in the room — you just can't see it from your current position.\n\n"
    "For yes/no questions, end your response with exactly 'yes' or 'no' on its own line.\n"
    "For cannot_determine questions, end with exactly 'cannot_determine' on its own line.\n"
    "For count questions, end with just the number on its own line."
)

# ── Trajectory loading ────────────────────────────────────────────────────────

def load_trajectory(path: str) -> dict:
    """Load trajectory JSON; upgrade old single-room format to segments format."""
    with open(path) as f:
        traj = json.load(f)
    if "segments" not in traj:
        traj = {
            "segments": [{
                "level_file": traj["level_file"],
                "seed":       traj["seed"],
                "actions":    traj.get("actions", []),
            }],
            "probes": [
                {"segment": 0, **{k: v for k, v in p.items() if k != "segment"}}
                for p in traj.get("probes", [])
            ],
        }
    return traj


# ── Observation builder ───────────────────────────────────────────────────────

def build_obs_sequence(
    segments: list[dict],
    up_to_segment: int,
    up_to_step: int,
    serializer=None,
) -> str:
    """Replay actions and collect serialized observations.

    Replays all segments up to (segment=up_to_segment, step=up_to_step),
    taking a snapshot after every action.  Returns a formatted multi-step
    prompt string.

    Observations are labelled:
        [Step 0 — Foyer — facing north]
        ...
        [Step 3 — Foyer — moved forward — facing east]

    Segment transitions are marked with a header:
        --- Entering: Kitchen ---
    """
    blocks: list[str] = []
    global_step = 0
    ser = serializer if serializer is not None else _ser

    for seg_idx, seg in enumerate(segments):
        if seg_idx > up_to_segment:
            break

        level_file = seg["level_file"]
        seed       = seg["seed"]
        actions    = seg["actions"]
        room_name  = Path(level_file).stem.replace("_", " ").title()

        env = AsciiEnv.from_file(level_file)
        env.reset(seed=seed)

        # Label the room transition
        if seg_idx == 0:
            blocks.append(f"=== Room: {room_name} ===\n")
        else:
            blocks.append(f"\n=== Entered new room: {room_name} ===\n")

        # Snapshot at step 0 of this segment
        n_actions_this_seg = (
            up_to_step if seg_idx == up_to_segment else len(actions)
        )

        obs0 = ser.serialize(env, step=global_step)
        action_label = "starting position"
        blocks.append(f"[Step {global_step} — {action_label}]\n{obs0}")

        for act_idx, act in enumerate(actions[:n_actions_this_seg]):
            env.step(act)
            global_step += 1
            action_label = _ACTION_LABEL.get(act, f"action {act}")
            obs = ser.serialize(env, step=global_step)
            blocks.append(f"[Step {global_step} — {action_label}]\n{obs}")

    return "\n\n".join(blocks)


# ── Eval runner ───────────────────────────────────────────────────────────────

def run_trajectory(
    traj_path: str,
    model: str,
    verbose: bool,
    max_tokens: int,
    reasoning_effort: str | None,
    thinking_effort: str | None,
    serializer_name: str = "memory",
) -> pd.DataFrame:
    traj = load_trajectory(traj_path)
    segments = traj["segments"]
    probes   = traj["probes"]
    traj_name = Path(traj_path).stem

    ser = _ser_omni if serializer_name == "omniscient" else _ser

    lm = _make_lm(model, max_tokens, reasoning_effort, thinking_effort)
    rows = []

    for probe in probes:
        seg_idx    = probe.get("segment", 0)
        step       = probe["step"]
        probe_type = probe["probe_type"]
        question   = probe["question"]
        ground_truth = probe["ground_truth"]
        metadata   = probe.get("metadata", {})

        # Build the observation sequence up to this probe's position
        obs_text = build_obs_sequence(segments, seg_idx, step, serializer=ser)
        user_msg = f"{obs_text}\n\n{question}"

        # Count tokens hint for count probes
        _mt = 1024 if probe_type == "count" else None

        resp = lm.query(system=SYSTEM_PROMPT, user=user_msg, max_tokens=_mt)

        gt_norm = _normalize_gt(probe_type, ground_truth)

        # For count=0, models often say "there were no X" — nudge the scorer
        # by appending "0" so _parse_int can find it.
        scored_text = resp.text
        if (
            probe_type == "count"
            and gt_norm == 0
            and resp.text
            and _is_zero_response(resp.text)
        ):
            scored_text = "0\n" + resp.text

        score = (
            0.0
            if not resp.text or not resp.text.strip()
            else score_response(probe_type, gt_norm, scored_text)
        )

        if verbose:
            print(f"[{traj_name}] seg={seg_idx} step={step} type={probe_type}")
            print(f"  Q: {question[:120]}")
            print(f"  GT: {ground_truth}")
            print(f"  A:  {resp.text[:120]}")
            print(f"  score: {score}")
            print()

        rows.append({
            "model":             model,
            "serializer":        serializer_name,
            "trajectory":        traj_name,
            "segment":           seg_idx,
            "step":              step,
            "probe_type":        probe_type,
            "question":          question,
            "ground_truth":      str(ground_truth),
            "response":          resp.text,
            "score":             score,
            "prompt_tokens":     resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "metadata":          json.dumps(metadata),
        })

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval on hand-recorded HalluWorld trajectories."
    )
    parser.add_argument(
        "--traj", required=True,
        help="Path to a trajectory JSON file, or a directory of JSON files.",
    )
    parser.add_argument(
        "--models", nargs="+", default=["gpt-4o-mini"],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--out", default="results/trajectory_results.csv",
        help="Output CSV path (appended if it already exists).",
    )
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--thinking_effort", default=None)
    parser.add_argument(
        "--serializer", default="memory", choices=["memory", "omniscient"],
        help="memory = MemorySerializer (default); omniscient = full-room ASCII map + object list",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Collect trajectory file(s)
    traj_path = Path(args.traj)
    if traj_path.is_dir():
        traj_files = sorted(traj_path.glob("*.json"))
    else:
        traj_files = [traj_path]

    if not traj_files:
        print(f"No trajectory JSON files found at {args.traj}")
        sys.exit(1)

    all_rows: list[pd.DataFrame] = []

    for model in args.models:
        for tf in traj_files:
            print(f"  {model}  ←  {tf}")
            df = run_trajectory(
                str(tf), model,
                verbose=args.verbose,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                thinking_effort=args.thinking_effort,
                serializer_name=args.serializer,
            )
            all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_header = not os.path.exists(args.out)
    combined.to_csv(args.out, mode="a", index=False, header=write_header)
    print(f"\nWrote {len(combined)} rows → {args.out}")

    # Quick summary
    scoreable = combined[combined["score"].notna()].copy()
    if not scoreable.empty:
        scoreable["score"] = scoreable["score"].astype(float)
        print("\n=== Summary ===")
        summary = (
            scoreable.groupby(["model", "serializer", "trajectory", "probe_type"])["score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "accuracy", "count": "n"})
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
