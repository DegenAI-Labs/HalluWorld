#!/usr/bin/env python3
"""Experiment status tracker - shows navigation and probe coverage.

Usage:
    python check_experiment_status.py
"""

import json
from pathlib import Path
from collections import defaultdict

# Import LEVELS dict
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_innav_eval import LEVELS
from halluworld.data import LEVELS_DIR


def check_status():
    """Check navigation and probe coverage."""

    print("=" * 80)
    print("INNAV EXPERIMENT STATUS")
    print("=" * 80)
    print()

    # === NAVIGATION COVERAGE ===
    print("=== NAVIGATION TRACES (World × Serializer) ===")
    print()

    nav_traces_dir = Path("navigation_traces")
    nav_status = defaultdict(lambda: defaultdict(int))

    if nav_traces_dir.exists():
        for trace_dir in sorted(nav_traces_dir.glob("*_gpt-5.4-mini_*Serializer")):
            # Parse: Level_navmodel_Serializer
            parts = trace_dir.name.split("_gpt-5.4-mini_")
            if len(parts) == 2:
                level = parts[0]
                serializer = parts[1].replace("Serializer", "")
                trace_count = len(list(trace_dir.glob("nav_trace_seed*.json")))
                nav_status[level][serializer] = trace_count

    # Print by tier
    tiers = {
        "P": "Perception",
        "M": "Memory",
        "C": "Causal",
        "U": "Uncertainty"
    }

    for tier_prefix, tier_name in tiers.items():
        print(f"--- {tier_name} (tier) ---")
        tier_levels = {k: v for k, v in nav_status.items() if k.startswith(tier_prefix)}

        if tier_levels:
            for level in sorted(tier_levels.keys()):
                serializers = tier_levels[level]
                serializer_str = ", ".join([f"{s}:{count}" for s, count in sorted(serializers.items())])
                print(f"  {level:30s} → {serializer_str}")
        else:
            print(f"  (no traces yet)")
        print()

    # Total counts
    total_traces = sum(sum(s.values()) for s in nav_status.values())
    total_combinations = len([(l, s) for l, sers in nav_status.items() for s in sers])
    print(f"TOTAL: {total_traces} traces across {total_combinations} (world, serializer) combinations")
    print()

    # === PROBE COVERAGE ===
    print("=== PROBE RESULTS (World × Serializer × Model) ===")
    print()

    probe_files = list(Path(".").glob("probe_*.csv"))
    if probe_files:
        print(f"Found {len(probe_files)} probe result files:")
        for pf in sorted(probe_files):
            size = pf.stat().st_size / 1024
            print(f"  {pf.name:40s} ({size:.1f} KB)")
    else:
        print("  (no probe results yet)")
    print()

    # === CONFIGURATION STATUS ===
    print("=== CONFIGURATION STATUS ===")
    print()

    # Check what's in LEVELS dict
    print(f"LEVELS dict: {len(LEVELS)} worlds defined")

    # Check .innav.json configs
    configs = list(LEVELS_DIR.glob("*.innav.json"))
    print(f"InNav configs: {len(configs)} files")

    # Find LEVELS entries without configs
    missing_configs = []
    for level_key, level_path in LEVELS.items():
        level_file = Path(level_path).stem
        config_path = LEVELS_DIR / f"{level_file}.innav.json"
        if not config_path.exists():
            missing_configs.append(level_key)

    if missing_configs:
        print(f"\n⚠️  LEVELS entries missing .innav.json configs: {len(missing_configs)}")
        for lk in missing_configs[:5]:
            print(f"  - {lk}")
        if len(missing_configs) > 5:
            print(f"  ... and {len(missing_configs) - 5} more")
    else:
        print("✅ All LEVELS entries have .innav.json configs")

    # Find configs not in LEVELS
    config_levels = {c.stem.replace(".innav", "") for c in configs}
    level_files = {Path(p).stem for p in LEVELS.values()}
    extra_configs = config_levels - level_files

    if extra_configs:
        print(f"\n⚠️  Configs not in LEVELS dict: {len(extra_configs)}")
        for ec in sorted(extra_configs)[:5]:
            print(f"  - {ec}")
        if len(extra_configs) > 5:
            print(f"  ... and {len(extra_configs) - 5} more")

    print()

    # === HALLUWORLD-HARD COVERAGE ===
    print("=== HALLUWORLD-HARD COVERAGE (12 pairs) ===")
    print()

    hard_pairs = [
        ("P1_dense_array", "Grid"),
        ("P2_corridor_gauntlet", "Grid"),
        ("P4_harder_array", "Grid"),
        ("M1_river_6", "Memory"),
        ("M1_river_9", "Memory"),
        ("C1a_persistent_chain", "Memory"),
        ("C1a_noboard", "Memory"),
        ("C1b_continuous_chain", "Memory"),
        ("C1b_noboard", "Memory"),
        ("C5a_adversarial_board", "Memory"),
        ("C6_flood_fire_escape", "Memory"),
        ("X5_facility_tour", "Grid"),
    ]

    covered = 0
    for level, serializer in hard_pairs:
        serializer_key = serializer
        has_traces = nav_status.get(level, {}).get(serializer_key, 0) > 0
        status = "✅" if has_traces else "❌"
        trace_count = nav_status.get(level, {}).get(serializer_key, 0)
        print(f"  {status} {level:30s} + {serializer:10s} ({trace_count} traces)")
        if has_traces:
            covered += 1

    print(f"\nCoverage: {covered}/12 HalluWorld-Hard pairs")
    print()

    # === RECOMMENDATIONS ===
    print("=== NEXT STEPS ===")
    print()

    # Missing HalluWorld-Hard
    missing_hard = [(l, s) for l, s in hard_pairs if nav_status.get(l, {}).get(s, 0) == 0]
    if missing_hard:
        print("Missing HalluWorld-Hard worlds (high priority):")
        for level, serializer in missing_hard[:5]:
            in_levels = level in LEVELS
            config_exists = (LEVELS_DIR / f"{level.replace('_', '')}.innav.json").exists()
            blockers = []
            if not in_levels:
                blockers.append("not in LEVELS")
            if not config_exists:
                blockers.append("no config")
            blocker_str = f" [{', '.join(blockers)}]" if blockers else ""
            print(f"  - {level} + {serializer}{blocker_str}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    check_status()
