#!/usr/bin/env python3
"""Evaluate LMs on DynamicsProbe (windy gridworld).

Runs models against the windy_room level in multi-turn mode.  Each episode the
oracle policy walks the agent through the level for several steps; absolute
position is prepended to every step observation.  The LM must *induce* the wind
rule from seeing position jumps in the trajectory, then apply it to predict the
final landing cell.

Three wind-hint conditions are compared:
  none  — question says nothing about wind; purely observational
  hint  — question says "this grid has wind" but no spec
  full  — question gives the complete column→offset spec (ablation baseline)

Usage:
    conda run -n halluworld python run_dynamics_eval.py
    conda run -n halluworld python run_dynamics_eval.py --models gpt-4o-mini gpt-4o o3-mini --episodes 30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from halluworld.tracks.grid.envs.ascii_env import make_env_from_ascii
from halluworld.serializer import Serializer
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.tracks.grid import DynamicsProbe
from halluworld.tracks.grid import DynamicsEvaluator
from halluworld.lm import OpenAILM
from halluworld.lm.baseten_lm import BasetenLM
from halluworld.multiturn import run_multiturn_benchmark
from halluworld.data import LEVELS_DIR

LEVEL = str(LEVELS_DIR / "windy_room.txt")

# Separate system prompt for dynamics eval:
# - explains absolute (row, col) coordinates so position diffs are meaningful
# - explicitly does NOT mention wind — model must infer dynamics from trajectory
OBSERVE_SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. "
    "The coordinate system has (row=0, col=0) at the top-left corner; "
    "row increases downward (south) and col increases rightward (east). "
    "Walls block movement — if you try to move into a wall you stay put. "
    "Each step of your trajectory shows your absolute grid position followed "
    "by your egocentric field-of-view observation. "
    "Study the trajectory carefully: if your position after a move differs "
    "from what pure movement geometry would predict, there may be environmental "
    "effects at work. "
    "Answer the final question about where you will land after an action. "
    "Reply with exactly: row=<integer>, col=<integer>"
)


class PositionAwareSerializer(Serializer):
    """Wraps any serializer and prepends the agent's absolute (row, col) position.

    This makes wind effects directly observable in the trajectory:
    the LM sees 'Position: row=7, col=4' before a move east, then
    'Position: row=6, col=5' after it — one row higher than expected —
    revealing that col 5 has a northward offset.
    """

    def __init__(self, inner: Serializer) -> None:
        self._inner = inner

    def serialize(self, env) -> str:
        ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
        return f"[Position: row={ay}, col={ax}]\n{self._inner.serialize(env)}"


def run_for_model(
    model: str,
    n_episodes: int,
    seed: int,
    verbose: bool,
    wind_hint_mode: str,
    max_tokens: int = 256,
) -> pd.DataFrame:
    env = make_env_from_ascii(LEVEL, render_mode=None)
    probe = DynamicsProbe(
        include_turn_variants=True,
        wind_hint_mode=wind_hint_mode,
    )
    evaluator = DynamicsEvaluator()
    is_baseten = (model.lower().startswith("glm") or model.startswith("zai-org/")
                  or model.startswith("openai/") or model.lower().startswith("qwen"))
    if is_baseten:
        lm = BasetenLM(model=model if "/" in model else f"zai-org/{model}",
                       base_url=os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
                       api_key=os.environ.get("BASETEN_API_KEY"), temperature=0.0, max_tokens=max_tokens)
    else:
        lm = OpenAILM(model=model, api_key=os.environ.get("OPENAI_API_KEY"), temperature=0.0, max_tokens=max_tokens)

    run = run_multiturn_benchmark(
        env=env,
        serializer=PositionAwareSerializer(SymbolicSerializer()),
        probes=[probe],
        lm=lm,
        evaluators=[evaluator],
        n_episodes=n_episodes,
        trajectory_length=30,   # long enough to wander through wind cols several times
        warmup_steps=0,
        policy="random",        # biased random walk so agent stays near wind zone
        hide_final_obs=False,
        seed=seed,
        verbose=verbose,
        system_prompt=OBSERVE_SYSTEM_PROMPT,
    )

    rows = []
    for r in run.results:
        rows.append({
            "model": model,
            "wind_hint_mode": wind_hint_mode,
            "correct": r.is_correct,
            "score": r.score,
            "has_wind": r.metadata.get("has_wind", False),
            "wind_offset": r.metadata.get("wind_offset", 0),
            "wind_naive_error": r.metadata.get("wind_naive_error", False),
            "parse_note": r.metadata.get("parse_note", "ok"),
            "action": r.metadata.get("action", ""),
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("DynamicsProbe Evaluation Results")
    print("=" * 72)

    for (model, hint_mode), mdf in df.groupby(["model", "wind_hint_mode"]):
        wind_df = mdf[mdf["has_wind"] & (mdf["wind_offset"] != 0)]
        nowind_df = mdf[~(mdf["has_wind"] & (mdf["wind_offset"] != 0))]
        acc = mdf["correct"].mean()
        hall = 1.0 - mdf["score"].mean()
        print(f"\nModel: {model}  hint_mode: {hint_mode}")
        print(f"  Overall        n={len(mdf):3d}  accuracy={acc:.1%}  hallucination={hall:.1%}")
        if len(wind_df):
            wnaive = wind_df["wind_naive_error"].mean()
            print(f"  Wind episodes  n={len(wind_df):3d}  accuracy={wind_df['correct'].mean():.1%}  wind_naive_error={wnaive:.1%}")
        if len(nowind_df):
            print(f"  No-wind eps    n={len(nowind_df):3d}  accuracy={nowind_df['correct'].mean():.1%}")
        parse_fail = (mdf["parse_note"] != "ok").sum()
        if parse_fail:
            print(f"  Parse failures: {parse_fail}")

    print("\n" + "=" * 72)
    print("Summary table:")

    summary_rows = []
    for (model, hint_mode), mdf in df.groupby(["model", "wind_hint_mode"]):
        wind_df = mdf[mdf["has_wind"] & (mdf["wind_offset"] != 0)]
        nowind_df = mdf[~(mdf["has_wind"] & (mdf["wind_offset"] != 0))]
        summary_rows.append({
            "model": model,
            "hint_mode": hint_mode,
            "n": len(mdf),
            "accuracy": f"{mdf['correct'].mean():.1%}",
            "hallucination": f"{1 - mdf['score'].mean():.1%}",
            "wind_acc": f"{wind_df['correct'].mean():.1%}" if len(wind_df) else "—",
            "wind_naive_err": f"{wind_df['wind_naive_error'].mean():.1%}" if len(wind_df) else "—",
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DynamicsProbe eval")
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt-4o-mini", "gpt-4o", "o3-mini"],
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--hint-modes", nargs="+", default=["none", "hint", "full"],
        choices=["none", "hint", "full"],
        help="Wind hint conditions to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="max_tokens for LM (reasoning models auto-scale to ≥4096)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip (model, hint_mode) pairs already present in --out CSV",
    )
    args = parser.parse_args()

    # Load existing results for resume / incremental save
    existing_df: pd.DataFrame | None = None
    if args.out and args.resume and Path(args.out).exists():
        existing_df = pd.read_csv(args.out)
        already = set(zip(existing_df["model"], existing_df["wind_hint_mode"]))
        print(f"Resuming from {args.out}: already have {already}")
    else:
        already = set()

    all_dfs = [] if existing_df is None else [existing_df]

    for model in args.models:
        for hint_mode in args.hint_modes:
            if (model, hint_mode) in already:
                print(f"\nSkipping {model}  hint_mode={hint_mode}  (already in CSV)")
                continue
            print(f"\nRunning {model}  hint_mode={hint_mode}  ({args.episodes} episodes)...")
            try:
                df = run_for_model(model, args.episodes, args.seed, args.verbose, hint_mode, args.max_tokens)
            except Exception as exc:
                print(f"\nERROR running {model}/{hint_mode}: {exc}")
                print("Saving partial results and continuing with next condition...")
                # Save whatever we have so far before propagating
                if args.out and all_dfs:
                    partial = pd.concat(all_dfs, ignore_index=True)
                    partial.to_csv(args.out, index=False)
                    print(f"Partial results saved to {args.out}")
                continue
            all_dfs.append(df)
            # Incremental save after every (model, hint_mode) completes
            if args.out:
                partial = pd.concat(all_dfs, ignore_index=True)
                partial.to_csv(args.out, index=False)
                print(f"  → saved to {args.out}")

    if not all_dfs:
        print("No results to report.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    report(combined)

    if args.out:
        combined.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
