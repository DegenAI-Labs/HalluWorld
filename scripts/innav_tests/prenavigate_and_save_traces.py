#!/usr/bin/env python3
"""Pre-navigate all levels and save navigation traces.

This script navigates through all innav levels once with gpt-5.4,
saving the navigation traces for reuse by probe models. This eliminates
redundant navigation when testing multiple probe models.

Usage:
    python prenavigate_and_save_traces.py --levels P2_corridor_gauntlet M1_river_field C1a_persistent_chain --episodes 5 --seed 42

    python prenavigate_and_save_traces.py --all-levels --episodes 5 --seed 42
"""

import argparse
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.tracks.grid.serializers.memory import MemorySerializer
from halluworld.lm.openai_lm import OpenAILM
from halluworld.tracks.innav.engine import InNavEpisode
from halluworld.tracks.grid.probes.visibility import PresenceProbe  # Dummy probe (not used)
from halluworld.data import LEVELS_DIR


# Level definitions
LEVELS = {
    # Perception levels
    "P1_dense_array": str(LEVELS_DIR / "dense_array.txt"),
    "P2_corridor_gauntlet": str(LEVELS_DIR / "corridor_gauntlet.txt"),
    "P3_rotation_challenge": str(LEVELS_DIR / "rotation_challenge.txt"),

    # Causal levels
    "C1a_persistent_chain": str(LEVELS_DIR / "c1a_persistent_chain.txt"),
    "C2_fire_crossing": str(LEVELS_DIR / "c2_fire_crossing.txt"),
    "C3_flood_room": str(LEVELS_DIR / "c3_flood_room.txt"),

    # Memory levels
    "M1_river_field": str(LEVELS_DIR / "river_field.txt"),
    "M4_narrator": str(LEVELS_DIR / "narrator_room.txt"),
}


def prenavigate_level(
    level_key: str,
    level_path: str,
    n_episodes: int,
    base_seed: int,
    trace_dir: str,
    navigation_model: str = "gpt-5.4",
    reasoning_effort: str = "medium",
):
    """Pre-navigate a level and save traces.

    Args:
        level_key: Level identifier (e.g., "P2_corridor_gauntlet")
        level_path: Path to level .txt file
        n_episodes: Number of episodes to navigate
        base_seed: Base random seed
        trace_dir: Directory to save traces
        navigation_model: Navigation model name
        reasoning_effort: Reasoning effort for navigation
    """
    print(f"\n{'='*70}")
    print(f"PRE-NAVIGATING: {level_key}")
    print(f"{'='*70}")

    # Create output directory
    Path(trace_dir).mkdir(parents=True, exist_ok=True)

    # Load environment
    env = AsciiEnv.from_file(level_path)

    # Use MemorySerializer for memory levels, SymbolicSerializer otherwise
    if level_key.startswith("M"):
        serializer = MemorySerializer()
        print(f"  Using MemorySerializer (memory level)")
    else:
        serializer = SymbolicSerializer()
        print(f"  Using SymbolicSerializer")

    # Navigation LM
    navigation_lm = OpenAILM(
        model=navigation_model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        max_tokens=256,
        reasoning_effort=reasoning_effort,
    )

    # Dummy probe LM (not used, but required for InNavEpisode)
    probe_lm = navigation_lm

    # Dummy probes (not used)
    probes = [PresenceProbe()]

    # Navigate each episode
    for ep in range(n_episodes):
        seed = base_seed + ep
        print(f"\n  Episode {ep+1}/{n_episodes} (seed={seed})...", end=" ", flush=True)

        # Create episode
        ego_episode = InNavEpisode(
            env=env,
            lm=probe_lm,
            serializer=serializer,
            probes=probes,
            probe_timesteps=0,  # No probing, just navigation
            max_steps=100,
            navigation_lm=navigation_lm,
        )

        # Navigate and extract trajectory
        try:
            ego_episode.env.reset(seed=seed)
            trajectory = ego_episode._run_navigation(seed)

            # Save trace
            # Check if goal reached (either done flag or positive reward)
            reached_goal = False
            if trajectory:
                last_step = trajectory[-1]
                reached_goal = last_step.get("done", False) or last_step.get("reward", 0) > 0

            trace_data = {
                "level": level_key,
                "seed": seed,
                "navigation_model": navigation_model,
                "reasoning_effort": reasoning_effort,
                "serializer": serializer.__class__.__name__,
                "reached_goal": reached_goal,
                "steps_taken": len(trajectory),
                "action_sequence": [t["action"] for t in trajectory],
            }

            trace_filename = f"nav_trace_seed{seed}.json"
            trace_path = Path(trace_dir) / trace_filename

            with open(trace_path, 'w') as f:
                json.dump(trace_data, f, indent=2)

            print(f"✅ Saved ({len(trajectory)} steps) → {trace_path}")

        except Exception as e:
            print(f"❌ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-navigate levels and save traces for reuse"
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=[],
        choices=list(LEVELS.keys()),
        help="Specific levels to pre-navigate"
    )
    parser.add_argument(
        "--all-levels",
        action="store_true",
        help="Pre-navigate all levels"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per level (default: 5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--navigation-model",
        default="gpt-5.4",
        help="Navigation model (default: gpt-5.4)"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default="medium",
        help="Reasoning effort (default: medium)"
    )
    parser.add_argument(
        "--trace-dir",
        default="navigation_traces_precomputed",
        help="Directory to save traces (default: navigation_traces_precomputed)"
    )

    args = parser.parse_args()

    # Determine which levels to run
    if args.all_levels:
        levels_to_run = list(LEVELS.keys())
    elif args.levels:
        levels_to_run = args.levels
    else:
        parser.error("Must specify either --levels or --all-levels")

    print(f"\n{'='*70}")
    print(f"PRE-NAVIGATION SETUP")
    print(f"{'='*70}")
    print(f"Levels: {', '.join(levels_to_run)}")
    print(f"Episodes per level: {args.episodes}")
    print(f"Seeds: {args.seed} to {args.seed + args.episodes - 1}")
    print(f"Navigation model: {args.navigation_model} ({args.reasoning_effort})")
    print(f"Output directory: {args.trace_dir}/")

    # Pre-navigate each level
    for level_key in levels_to_run:
        level_path = LEVELS[level_key]

        # Create level-specific subdirectory
        level_trace_dir = Path(args.trace_dir) / level_key

        prenavigate_level(
            level_key=level_key,
            level_path=level_path,
            n_episodes=args.episodes,
            base_seed=args.seed,
            trace_dir=str(level_trace_dir),
            navigation_model=args.navigation_model,
            reasoning_effort=args.reasoning_effort,
        )

    print(f"\n{'='*70}")
    print("✅ PRE-NAVIGATION COMPLETE")
    print(f"{'='*70}")
    print(f"Saved traces to: {args.trace_dir}/")
    print()
    print("Now run probing with:")
    print(f"  python run_innav_eval.py \\")
    print(f"    --models gpt-4o gpt-4o-mini gpt-5.4-mini \\")
    print(f"    --levels {' '.join(levels_to_run)} \\")
    print(f"    --episodes {args.episodes} --seed {args.seed} \\")
    print(f"    --navigation-model {args.navigation_model} \\")
    print(f"    --trace-dir {args.trace_dir}/{{level}} \\")
    print(f"    --output results_innav.csv")
    print()


if __name__ == "__main__":
    main()
