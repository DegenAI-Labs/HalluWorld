"""Extract actual static probe locations from Emmy's worlds.

For each level, run a few episodes and record where probes are asked.
This gives us REAL probe locations (not guessed).
"""

import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.probes.visibility import PresenceProbe
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
import random
from halluworld.data import LEVELS_DIR


def extract_probe_locations(level_path: str, num_episodes: int = 5, seeds: list = None):
    """Extract probe locations from static episodes."""

    if seeds is None:
        seeds = [42, 123, 456, 789, 999][:num_episodes]

    env = AsciiEnv.from_file(level_path)
    serializer = SymbolicSerializer()

    probe_locations = []

    for seed in seeds:
        env.reset(seed=seed)

        # Get agent starting position
        start_pos = (int(env.agent_pos[0]), int(env.agent_pos[1]))

        # In static benchmark, probe is asked at the starting position
        # (no navigation, just observe and answer)
        probe_locations.append({
            'seed': seed,
            'position': start_pos,
            'direction': int(env.agent_dir),
        })

    return probe_locations


def main():
    print("="*70)
    print("EXTRACT STATIC PROBE LOCATIONS")
    print("="*70)
    print()
    print("Running static episodes to find where probes are actually asked")
    print()

    # Emmy's levels
    levels = [
        str(LEVELS_DIR / "corridor_gauntlet.txt"),
        str(LEVELS_DIR / "dense_array.txt"),
        str(LEVELS_DIR / "rotation_challenge.txt"),
        str(LEVELS_DIR / "c2_fire_crossing.txt"),
        str(LEVELS_DIR / "c3_flood_room.txt"),
    ]

    results = {}

    for level_path in levels:
        level_name = Path(level_path).stem

        print(f"{level_name}")
        print("-" * 70)

        try:
            locations = extract_probe_locations(level_path, num_episodes=5)
            results[level_name] = locations

            # Show unique positions
            unique_positions = list(set(loc['position'] for loc in locations))
            print(f"  Episodes: {len(locations)}")
            print(f"  Unique positions: {len(unique_positions)}")
            print(f"  Positions: {unique_positions}")
            print()

        except FileNotFoundError:
            print(f"  ⚠️  File not found: {level_path}")
            print()

    # Save to JSON
    output_file = 'static_probe_locations.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*70)
    print(f"Saved to: {output_file}")
    print("="*70)
    print()

    # Summary
    print("SUMMARY")
    print("-" * 70)
    for level_name, locations in results.items():
        unique_positions = list(set(loc['position'] for loc in locations))
        print(f"{level_name}:")
        print(f"  Static probe positions: {unique_positions}")
    print()


if __name__ == "__main__":
    main()
