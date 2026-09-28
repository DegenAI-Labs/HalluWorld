#!/usr/bin/env python3
"""Evaluate LMs on InventoryProbe (object-carrying tracking).

The agent uses the ``pickup_oracle`` policy, which routes it to the nearest
pickupable object, picks it up, and then navigates to the goal.  Once an object
is picked up it vanishes from the field-of-view — the LM must track what is
being *carried* purely from the action history.

Three question sources are compared:
  carrying_match  — agent IS carrying X; asks about X  →  GT=True
                    Tests: does the model remember the pickup?
  carrying_foil   — agent IS carrying X; asks about Y (seen but not held)  →  GT=False
                    Tests: does the model confuse "seen" with "held"?
  not_carrying    — agent is NOT carrying anything; asks about something seen  →  GT=False
                    Tests: does the model hallucinate possession of seen objects?

Usage:
    conda run -n halluworld python run_inventory_eval.py
    conda run -n halluworld python run_inventory_eval.py \\
        --models gpt-4o-mini gpt-4o o3-mini --episodes 40
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
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.tracks.grid import InventoryProbe
from halluworld.tracks.grid import PresenceEvaluator
from halluworld.lm import OpenAILM
from halluworld.lm.baseten_lm import BasetenLM
from halluworld.multiturn import run_multiturn_benchmark
from halluworld.data import LEVELS_DIR

LEVEL = str(LEVELS_DIR / "open_room_16.txt")

# System prompt for inventory eval.
# Explicitly notes that picked-up objects leave the FOV but stay in inventory,
# so the model must reason from action history — not just current observation.
INVENTORY_SYSTEM_PROMPT = (
    "You are an agent navigating a gridworld. "
    "You can pick up objects (keys, balls, boxes) as you move through the environment. "
    "When you pick up an object it is placed in your inventory and no longer appears in "
    "your field-of-view observations — the only record of the pickup is the action "
    "'pickup <color> <type>' in your trajectory. "
    "When you drop an object (action 'drop <color> <type>') it is placed back on the grid "
    "and you are no longer carrying it. "
    "You can carry at most one object at a time. "
    "Study your full trajectory carefully, including all pickup and drop actions, "
    "to determine what, if anything, you are currently carrying. "
    "Answer the question about your inventory with exactly 'yes' or 'no'."
)


def run_for_model(
    model: str,
    n_episodes: int,
    seed: int,
    verbose: bool,
    max_tokens: int = 256,
    level: str = LEVEL,
) -> pd.DataFrame:
    env = make_env_from_ascii(level, render_mode=None)
    probe = InventoryProbe(positive_rate=0.5, rng=None)
    evaluator = PresenceEvaluator()
    is_baseten = (model.lower().startswith("glm") or model.startswith("zai-org/")
                  or model.startswith("openai/") or model.lower().startswith("qwen"))
    if is_baseten:
        lm = BasetenLM(model=model if "/" in model else f"zai-org/{model}",
                       base_url=os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
                       api_key=os.environ.get("BASETEN_API_KEY"), temperature=0.0, max_tokens=max_tokens)
    else:
        lm = OpenAILM(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0.0,
            max_tokens=max_tokens,
        )

    run = run_multiturn_benchmark(
        env=env,
        serializer=SymbolicSerializer(hide_carrying=True),
        probes=[probe],
        lm=lm,
        evaluators=[evaluator],
        n_episodes=n_episodes,
        trajectory_length=100,
        warmup_steps=0,
        policy="pickup_wander",
        wander_steps=20,
        pickup_max=2,
        hide_final_obs=False,
        seed=seed,
        verbose=verbose,
        system_prompt=INVENTORY_SYSTEM_PROMPT,
    )

    rows = []
    for r in run.results:
        rows.append({
            "model": model,
            "correct": r.is_correct,
            "score": r.score,
            "ground_truth": r.ground_truth,
            "source": r.metadata.get("source", ""),
            "carrying": r.metadata.get("carrying", None),
            "asked_color": r.metadata.get("color", ""),
            "asked_object": r.metadata.get("object", ""),
            "n_trajectory_seen": r.metadata.get("n_trajectory_seen", 0),
            "parse_note": r.metadata.get("parse_note", "ok"),
            "lm_response": r.lm_response,
        })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("InventoryProbe Evaluation Results")
    print("=" * 72)

    models = df["model"].unique()
    sources = ["carrying_match", "carrying_foil", "not_carrying"]

    for model in models:
        mdf = df[df["model"] == model]
        print(f"\nModel: {model}  (n={len(mdf)})")

        overall_acc = mdf["correct"].mean()
        hall = 1.0 - mdf["score"].mean()
        parse_fail = (mdf["parse_note"] == "ambiguous_response").sum()
        print(f"  Overall  accuracy={overall_acc:.1%}  hallucination_rate={hall:.1%}"
              + (f"  parse_ambiguous={parse_fail}" if parse_fail else ""))

        for src in sources:
            sdf = mdf[mdf["source"] == src]
            if len(sdf) == 0:
                continue
            acc = sdf["correct"].mean()
            # For GT=False sources, false-positive rate = fraction the model wrongly said "yes"
            if sdf["ground_truth"].iloc[0] is False or sdf["ground_truth"].iloc[0] == False:
                false_positive = (sdf["lm_response"].str.strip().str.lower() == "yes").mean()
                print(f"  {src:<20} n={len(sdf):3d}  accuracy={acc:.1%}  "
                      f"false_positive_rate={false_positive:.1%}")
            else:
                false_negative = (sdf["lm_response"].str.strip().str.lower() == "no").mean()
                print(f"  {src:<20} n={len(sdf):3d}  accuracy={acc:.1%}  "
                      f"false_negative_rate={false_negative:.1%}")

    print("\n" + "=" * 72)
    print("Summary table (by model × source):")

    summary_rows = []
    for model in models:
        mdf = df[df["model"] == model]
        for src in sources:
            sdf = mdf[mdf["source"].str.startswith(src.split("_")[0] + "_" + src.split("_")[1]
                                                     if "_" in src else src)]
            # simpler: exact match on source
            sdf = mdf[mdf["source"] == src]
            if len(sdf) == 0:
                continue
            is_positive = bool(sdf["ground_truth"].iloc[0])
            if is_positive:
                extra_key = "false_neg"
                extra_val = f"{(sdf['lm_response'].str.strip().str.lower() == 'no').mean():.1%}"
            else:
                extra_key = "false_pos"
                extra_val = f"{(sdf['lm_response'].str.strip().str.lower() == 'yes').mean():.1%}"
            summary_rows.append({
                "model": model,
                "source": src,
                "n": len(sdf),
                "accuracy": f"{sdf['correct'].mean():.1%}",
                extra_key: extra_val,
            })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run InventoryProbe eval")
    parser.add_argument(
        "--models", nargs="+",
        default=["gpt-4o-mini", "gpt-4o", "o3-mini"],
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--level", type=str, default=LEVEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="max_tokens for LM (reasoning models auto-scale to ≥4096)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip models already present in --out CSV",
    )
    args = parser.parse_args()

    existing_df: pd.DataFrame | None = None
    if args.out and args.resume and Path(args.out).exists():
        existing_df = pd.read_csv(args.out)
        already = set(existing_df["model"].unique())
        print(f"Resuming from {args.out}: already have {already}")
    else:
        already = set()

    all_dfs = [] if existing_df is None else [existing_df]

    for model in args.models:
        if model in already:
            print(f"\nSkipping {model}  (already in CSV)")
            continue
        print(f"\nRunning {model}  ({args.episodes} episodes)...")
        try:
            df = run_for_model(model, args.episodes, args.seed, args.verbose,
                               args.max_tokens, args.level)
        except Exception as exc:
            print(f"\nERROR running {model}: {exc}")
            if args.out and all_dfs:
                partial = pd.concat(all_dfs, ignore_index=True)
                partial.to_csv(args.out, index=False)
                print(f"Partial results saved to {args.out}")
            continue
        all_dfs.append(df)
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
