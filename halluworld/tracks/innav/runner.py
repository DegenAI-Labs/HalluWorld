#!/usr/bin/env python3
"""InNav evaluation runner.

Tests whether "acting while observing" degrades perception accuracy compared
to pure observation (static mode).

Usage:
    python run_innav_eval.py --models gpt-4o-mini --levels P2_corridor_gauntlet --episodes 10 --seed 42 --output innav_results.csv

    python run_innav_eval.py --models qwen-3-30b-thinking --levels C2_fire_crossing --episodes 5 --seed 42 --output ego_qwen_c2.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.tracks.grid.serializers.memory import MemorySerializer
from halluworld.tracks.grid.serializers.grid import GridSerializer
from halluworld.tracks.innav.canonical_probes import make_canonical_probes
from halluworld.tracks.grid.probes.visibility import (
    PresenceProbe,
    CountProbe,
    AttributeProbe,
    AllocentricLocationProbe,
)
from halluworld.lm.openai_lm import OpenAILM
from halluworld.lm.baseten_lm import BasetenLM
from halluworld.lm.anthropic_lm import AnthropicLM
from halluworld.tracks.innav.engine import run_innav_episodes
from halluworld.data import LEVELS_DIR


# Level definitions
LEVELS = {
    # Perception levels
    "P1_dense_array": str(LEVELS_DIR / "dense_array.txt"),
    "P2_corridor_gauntlet": str(LEVELS_DIR / "corridor_gauntlet.txt"),
    "P3_rotation_challenge": str(LEVELS_DIR / "rotation_challenge.txt"),
    "P4_harder_array": str(LEVELS_DIR / "harder_array.txt"),
    "P4_delta_perception": str(LEVELS_DIR / "delta_perception.txt"),
    "P5_object_permanence": str(LEVELS_DIR / "permanence_init.txt"),

    # Causal levels
    "C1a_persistent_chain": str(LEVELS_DIR / "c1a_persistent_chain.txt"),
    "C1a_noboard": str(LEVELS_DIR / "c1a_noboard.txt"),
    "C1b_continuous_chain": str(LEVELS_DIR / "c1b_continuous_chain.txt"),
    "C1b_noboard": str(LEVELS_DIR / "c1b_noboard.txt"),
    "C2_fire_crossing": str(LEVELS_DIR / "c2_fire_crossing.txt"),
    "C3_flood_room": str(LEVELS_DIR / "c3_flood_room.txt"),
    "C4_forking_paths": str(LEVELS_DIR / "c4_forking_paths.txt"),
    "C5a_adversarial_board": str(LEVELS_DIR / "c5a_adversarial_board.txt"),
    "C6_flood_fire_escape": str(LEVELS_DIR / "c6_flood_fire_escape.txt"),

    # Memory levels
    "M1_river_3": str(LEVELS_DIR / "river_field.txt"),
    "M1_river_6": str(LEVELS_DIR / "river_field.txt"),
    "M1_river_9": str(LEVELS_DIR / "river_field.txt"),
    "M2_witness_stand": str(LEVELS_DIR / "witness_chamber1.txt"),
    "M3_incident_report": str(LEVELS_DIR / "incident_t0.txt"),
    "M4_narrator": str(LEVELS_DIR / "narrator_room.txt"),

    # Uncertainty levels
    "U1_fog_of_war": str(LEVELS_DIR / "u1_fog_of_war.txt"),
    "U2_oracle_high": str(LEVELS_DIR / "u2_agent_zone_high.txt"),
    "U2_oracle_mid": str(LEVELS_DIR / "u2_agent_zone_mid.txt"),
    "U2_oracle_low": str(LEVELS_DIR / "u2_agent_zone_low.txt"),
    "U4_amnesiac": str(LEVELS_DIR / "u4_amnesiac.txt"),

    # Multi-zone compound levels (X-tier)
    "X1_facility_tour": str(LEVELS_DIR / "x1_zone_a.txt"),  # 3-zone baseline tour
    "X2_facility_tour": str(LEVELS_DIR / "x2_zone_d.txt"),  # 5-zone + lying signpost
    "X3_facility_tour": str(LEVELS_DIR / "x3_zone_a.txt"),  # 7-zone + recency bias
    "X4_facility_tour": str(LEVELS_DIR / "x4_zone_a.txt"),  # Witness Stand embedded
    "X5_facility_tour": str(LEVELS_DIR / "x5_zone_a.txt"),  # 7-zone cascading testimony
    "X6_return_visit": str(LEVELS_DIR / "x6_zone_a.txt"),  # 5-zone + revisit
    "X7_dragon_keep": str(LEVELS_DIR / "x7_zone_a.txt"),  # 8-zone + NPCs
}


def _make_lm(model: str, max_tokens: int, reasoning_effort: str | None = None):
    """Create LM instance based on model name.

    Supports:
    - OpenAI: gpt-*, o3*, o4* (uses OPENAI_API_KEY)
    - Anthropic: claude-* (uses ANTHROPIC_API_KEY)
    - Baseten: zai-org/*, deepseek-ai/*, moonshotai/*, qwen-*, glm* (uses BASETEN_API_KEY)
    """
    provider = os.environ.get("HALLUWORLD_PROVIDER", "").strip().lower()
    if provider not in ("", "openai", "anthropic", "baseten"):
        raise ValueError(f"unknown provider {provider!r}")
    # Baseten models: glm, zai-org/, deepseek-ai/, moonshotai/, qwen
    is_baseten = (
        model.lower().startswith("glm")
        or model.startswith("zai-org/")
        or model.startswith("deepseek-ai/")
        or model.startswith("moonshotai/")
        or model.lower().startswith("qwen")
    )

    # Anthropic models: claude-*
    is_anthropic = model.lower().startswith("claude")

    if provider == "baseten" or (not provider and is_baseten):
        return BasetenLM(
            model=model,
            base_url=os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
            api_key=os.environ.get("BASETEN_API_KEY"),
            max_tokens=max_tokens,
        )
    elif provider == "anthropic" or (not provider and is_anthropic):
        # Extract thinking_effort from model name if present (e.g., "claude-sonnet-4-6_thinking")
        thinking_effort_param = None
        if "_thinking" in model:
            thinking_effort_param = reasoning_effort or "medium"  # Default to medium thinking
            model = model.replace("_thinking", "")  # Remove suffix for API call

        return AnthropicLM(
            model=model,
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            max_tokens=max_tokens,
            thinking_effort=thinking_effort_param,
        )
    else:
        # OpenAI models (default)
        return OpenAILM(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )


def run_level_innav(
    level_key: str,
    level_path: str,
    model: str,
    n_episodes: int,
    base_seed: int,
    probe_timesteps: int = 5,
    max_steps: int = 100,
    max_tokens: int = 256,
    reasoning_effort: str | None = None,
    navigation_model: str | None = None,
    trace_dir: str | None = None,
    prenavigate: bool = False,  # NEW: Skip probing, only navigate and save traces
    serializer_override: str | None = None,  # NEW: Explicit serializer override
    use_canonical_probes: bool = True,  # NEW: Use Emmy's canonical probe sets
) -> pd.DataFrame:
    """Run innav evaluation on a single level.

    Args:
        level_key: Level identifier (e.g., "C2_fire_crossing")
        level_path: Path to level .txt file
        model: Model identifier (for probes)
        n_episodes: Number of episodes to run
        base_seed: Base random seed
        probe_timesteps: Number of timesteps to probe per episode
        max_steps: Maximum episode length
        max_tokens: Max completion tokens
        reasoning_effort: Reasoning effort (GPT-5 only)
        navigation_model: Optional separate model for navigation (e.g., gpt-5 for better rollouts)
        trace_dir: Directory with pre-saved traces (enables reuse)
        prenavigate: If True, only navigate and save traces (skip probing)

    Returns:
        DataFrame with innav vs static comparison results
    """
    if prenavigate:
        print(f"🚀 PRE-NAVIGATION MODE: {navigation_model or model} on {level_key}")
        print(f"  Navigating until sufficiency and saving traces (NO probing)")
    else:
        print(f"Running innav eval: {model} on {level_key}")
        if navigation_model:
            print(f"  Navigation model: {navigation_model} (probes use {model})")

    print(f"  Episodes: {n_episodes}, Max steps: {max_steps}, Probes/episode: {probe_timesteps}")

    # === STEP 1: Load .innav.json config ===
    import json
    from pathlib import Path

    config_path = Path(level_path).with_suffix('.innav.json')
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Extract navigation config
        nav_params = config.get('navigation_params', {})
        static_locations = [tuple(loc) for loc in config.get('navigation_target_locations', [])]

        max_steps = nav_params.get('max_steps', max_steps)  # Override CLI with config
        min_steps = nav_params.get('min_steps', 10)
        sufficiency_distance = nav_params.get('sufficiency_distance', 5.0)
        min_distance_from_start = nav_params.get('min_distance_from_start', 5.0)

        print(f"  ✅ Loaded .innav.json config:")
        print(f"     Static locations: {static_locations}")
        print(f"     Navigation params: max_steps={max_steps}, min_steps={min_steps}, "
              f"sufficiency_dist={sufficiency_distance}, min_dist_from_start={min_distance_from_start}")
    else:
        print(f"  ⚠️  No .innav.json config found at {config_path}")
        print(f"     Using default navigation parameters (no sufficiency checking)")
        static_locations = []
        min_steps = 10
        sufficiency_distance = 5.0
        min_distance_from_start = 5.0

    # Level-specific trace directory with metadata (if base trace_dir provided)
    level_trace_dir = None
    if trace_dir:
        nav_model_name = (navigation_model or model).replace("/", "-")  # Handle model names like "zai-org/..."
        # Use CLI serializer if provided, otherwise default logic
        if serializer_override:
            serializer_name = f"{serializer_override.capitalize()}Serializer"
        else:
            serializer_name = "MemorySerializer" if (level_key.startswith("M") or level_key.startswith("C")) else "SymbolicSerializer"

        # Format: {level}_{nav_model}_{serializer} (NO DATE - simpler and more reliable)
        level_subdir = f"{level_key}_{nav_model_name}_{serializer_name}"
        level_trace_dir = str(Path(trace_dir) / level_subdir)
        print(f"  Trace directory: {level_trace_dir}")

    # Initialize probe LM
    lm = _make_lm(model, max_tokens, reasoning_effort)

    # Initialize navigation LM (if different from probe LM)
    navigation_lm_obj = None
    if navigation_model and navigation_model != model:
        navigation_lm_obj = _make_lm(navigation_model, max_tokens, reasoning_effort)
        print(f"  Using separate navigation LM: {navigation_model}")
    elif prenavigate:
        # In prenavigate mode, use the main model for navigation
        navigation_lm_obj = lm

    env = AsciiEnv.from_file(level_path)

    # Serializer selection: CLI override > level-based default
    if serializer_override:
        # Explicit serializer specified via CLI
        if serializer_override == "memory":
            serializer = MemorySerializer()
            print(f"  Using MemorySerializer (CLI override)")
        elif serializer_override == "grid":
            serializer = GridSerializer()
            print(f"  Using GridSerializer (CLI override)")
        else:  # symbolic
            serializer = SymbolicSerializer()
            print(f"  Using SymbolicSerializer (CLI override)")
    else:
        # Default: M-levels use Memory, C-levels use Memory, others use Symbolic
        if level_key.startswith("M") or level_key.startswith("C"):
            serializer = MemorySerializer()
            print(f"  Using MemorySerializer (default for {level_key[0]}-tier)")
        else:
            serializer = SymbolicSerializer()
            print(f"  Using SymbolicSerializer (default)")

    # Select probe set based on flag
    # In prenavigate mode, we still need to pass probes but they won't be used
    if use_canonical_probes:
        probe_rng = random.Random(base_seed)
        probes = make_canonical_probes(level_key, probe_rng)
        print(f"  Using canonical probes: {[type(p).__name__ for p in probes]}")
    else:
        # Generic probe set (backward compatibility)
        probes = [
            PresenceProbe(),
            CountProbe(),
            AttributeProbe(),
            AllocentricLocationProbe(),
        ]
        print(f"  Using generic probes: {[type(p).__name__ for p in probes]}")

    # === STEP 2: Run episodes with navigation config ===
    results_df = run_innav_episodes(
        env=env,
        lm=lm,
        serializer=serializer,
        probes=probes,
        n_episodes=n_episodes,
        base_seed=base_seed,
        probe_timesteps=probe_timesteps if not prenavigate else 0,  # Skip probes in prenavigate mode
        max_steps=max_steps,
        navigation_lm=navigation_lm_obj,
        trace_dir=level_trace_dir,
        save_traces=True,  # Always save traces
        static_probe_locations=static_locations,
        min_steps=min_steps,
        sufficiency_distance=sufficiency_distance,
        min_distance_from_start=min_distance_from_start,
    )

    # Add metadata columns
    results_df.insert(0, "model", model)
    results_df.insert(1, "level", level_key)

    return results_df


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run innav evaluation (agent acts while being probed)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini"],
        help="Model(s) to evaluate"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "baseten"],
        default=None,
        help="Explicit provider routing (recommended)",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["P2_corridor_gauntlet"],
        choices=list(LEVELS.keys()),
        help="Level(s) to run"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes per level"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed"
    )
    parser.add_argument(
        "--probe-timesteps",
        type=int,
        default=5,
        help="Number of timesteps to probe per episode"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max completion tokens"
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Reasoning effort (GPT-5 models only)"
    )
    parser.add_argument(
        "--navigation-model",
        default=None,
        help="Optional separate model for navigation (e.g., gpt-5 for quality rollouts, while testing weaker models on perception)"
    )
    parser.add_argument(
        "--trace-dir",
        default=None,
        help="Directory containing pre-saved navigation traces (enables trace reuse, avoids re-navigation)"
    )
    parser.add_argument(
        "--prenavigate",
        action="store_true",
        help="Pre-navigation mode: Navigate and save traces only (skip probing). Use this to generate traces once, then reuse with --trace-dir"
    )
    parser.add_argument(
        "--output",
        default="innav_results.csv",
        help="Output CSV file"
    )
    parser.add_argument(
        "--serializer",
        choices=["symbolic", "memory", "grid"],
        default=None,
        help="Serializer to use (overrides default level-based selection)"
    )
    parser.add_argument(
        "--use-canonical-probes",
        action="store_true",
        default=True,
        help="Use Emmy's canonical probe sets (default: True). Use --no-use-canonical-probes for generic probes"
    )

    args = parser.parse_args(argv)
    if args.provider:
        os.environ["HALLUWORLD_PROVIDER"] = args.provider

    # Validation: prenavigate mode needs navigation model
    if args.prenavigate:
        if not args.navigation_model:
            print("⚠️  WARNING: --prenavigate mode should specify --navigation-model (e.g., gpt-5.4)")
            print("   Using probe model for navigation instead: {}".format(args.models[0]))

    # Run evaluations
    all_results = []

    for model in args.models:
        for level_key in args.levels:
            level_path = LEVELS[level_key]

            results_df = run_level_innav(
                level_key=level_key,
                level_path=level_path,
                model=model,
                n_episodes=args.episodes,
                base_seed=args.seed,
                probe_timesteps=args.probe_timesteps,
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                navigation_model=args.navigation_model,
                trace_dir=args.trace_dir,
                prenavigate=args.prenavigate,
                serializer_override=args.serializer,
                use_canonical_probes=args.use_canonical_probes,
            )

            all_results.append(results_df)

            # Save individual probe file per model-level combination (prevents overwriting)
            # Determine serializer name for filename
            if args.serializer:
                serializer_name = args.serializer
            else:
                serializer_name = "memory" if (level_key.startswith("M") or level_key.startswith("C") or level_key.startswith("U")) else "symbolic"

            # Format: probe_{level}_{serializer}_{model}.csv (match existing pattern)
            safe_model_name = model.replace("/", "-")  # Handle model names like "moonshotai/Kimi-K2.6"
            output_dir = Path(args.output).expanduser().resolve().parent
            output_dir.mkdir(parents=True, exist_ok=True)
            individual_filename = output_dir / f"probe_{level_key}_{serializer_name}_{safe_model_name}.csv"
            results_df.to_csv(individual_filename, index=False)
            print(f"  ✅ Saved individual probe file → {individual_filename}")

            # Also save combined file incrementally (for backup/reference)
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(args.output, index=False)
            print(f"  📦 Updated combined file → {args.output}")

    # Print summary (skip if prenavigate mode - no probe results)
    if not args.prenavigate and len(combined) > 0:
        print("\n=== InNav vs Static Summary ===")
        if "innav_accuracy" in combined.columns and "controlled_static_accuracy" in combined.columns:
            summary = combined.groupby(["model", "level"])[["innav_accuracy", "controlled_static_accuracy"]].mean()
            print(summary)
        else:
            print("No probe results to summarize")
    elif args.prenavigate:
        print("\n=== Pre-navigation Complete ===")
        # In prenavigate mode, count episodes not probe results (probe_timesteps=0 → no rows)
        n_traces = len(args.models) * len(args.levels) * args.episodes
        print(f"Saved {n_traces} navigation traces")
        print(f"Trace directory: {args.trace_dir}")

        # Show trace locations for each level
        for level_key in args.levels:
            nav_model_name = (args.navigation_model or args.models[0]).replace("/", "-")
            # Use CLI serializer if provided, otherwise default logic
            if args.serializer:
                serializer_name = f"{args.serializer.capitalize()}Serializer"
            else:
                serializer_name = "MemorySerializer" if (level_key.startswith("M") or level_key.startswith("C")) else "SymbolicSerializer"
            level_subdir = f"{level_key}_{nav_model_name}_{serializer_name}"
            trace_path = Path(args.trace_dir) / level_subdir
            if trace_path.exists():
                trace_files = list(trace_path.glob("nav_trace_seed*.json"))
                print(f"  {level_key}: {len(trace_files)} traces in {trace_path}")

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
