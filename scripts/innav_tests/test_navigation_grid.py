"""Test navigation with GridSerializer on worlds that failed with Symbolic.

Worlds to test (failed with Symbolic):
- dense_array
- rotation_challenge
- u1_fog_of_war
- u4_amnesiac
- c1b_continuous_chain
- open_room
"""

import sys
sys.path.insert(0, '.')
import json
from pathlib import Path

from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.grid import GridSerializer  # Use Grid instead of Symbolic
from halluworld.lm.openai_lm import OpenAILM
from halluworld.tracks.innav.navigation import navigate_until_sufficient
from halluworld.data import LEVELS_DIR


def test_with_grid_serializer(level_name: str, seed: int = 42):
    """Test navigation with GridSerializer and save trace."""

    # Load config
    config_path = fstr(LEVELS_DIR / "{level_name}.innav.json")
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"  ❌ Config not found: {config_path}")
        return None

    # Load world
    level_path = fstr(LEVELS_DIR / "{level_name}.txt")
    env = AsciiEnv.from_file(level_path)

    # Use GridSerializer (KEY DIFFERENCE!)
    serializer = GridSerializer()

    # Navigation model
    lm = OpenAILM(model="gpt-5.4", max_tokens=256, reasoning_effort="medium")

    # Extract params
    targets = [tuple(t) for t in config['navigation_target_locations']]
    params = config['navigation_params']

    print(f"  Testing {level_name} with GridSerializer...")

    # Navigate
    result = navigate_until_sufficient(
        env=env,
        lm=lm,
        serializer=serializer,
        static_probe_locations=targets,
        seed=seed,
        max_steps=params['max_steps'],
        max_distance=params['sufficiency_distance'],
        min_steps=params['min_steps'],
        min_distance_from_start=params['min_distance_from_start']
    )

    # Save trace with GridSerializer in filename
    if result['sufficient']:
        trace_dir = Path("navigation_traces")
        trace_dir.mkdir(exist_ok=True)

        trace_filename = f"trace_GridSerializer_{level_name}_seed{seed}.json"
        trace_path = trace_dir / trace_filename

        trace_data = {
            "level_name": level_name,
            "seed": seed,
            "serializer": "GridSerializer",  # Explicitly mark
            "navigation_model": "gpt-5.4",
            "reasoning_effort": "medium",
            "sufficient": True,
            "sufficient_step": int(result['sufficient_step']),
            "sufficient_position": [int(x) for x in result['sufficient_position']],
            "sufficient_location": [int(x) for x in result['sufficient_location']],
            "sufficient_distance": float(result['sufficient_distance']),
            "messages": result['messages'],  # Full chat history
            "trajectory": [int(step['action']) if isinstance(step, dict) else int(step) for step in result['trajectory']],  # Extract actions
            "reached_goal": bool(result['reached_goal']),
        }

        with open(trace_path, 'w') as f:
            json.dump(trace_data, f, indent=2)

        print(f"  ✅ SUCCESS! Saved trace to: {trace_path}")
        print(f"     Step: {result['sufficient_step']}, Distance: {result['sufficient_distance']:.2f}")
        return trace_path
    else:
        print(f"  ❌ FAILED: {result['reason']}")
        return None


def main():
    print("="*70)
    print("TESTING GRID SERIALIZER ON FAILED WORLDS")
    print("="*70)
    print()
    print("Model: GPT-5.4 with reasoning_effort='medium'")
    print("Serializer: GridSerializer (ASCII grid representation)")
    print()

    # Worlds that failed with Symbolic
    test_worlds = [
        'dense_array',
        'rotation_challenge',
        'u1_fog_of_war',
        'u4_amnesiac',
        'open_room',
    ]

    results = {}

    for world in test_worlds:
        print(f"\n{world.upper()}")
        print("-" * 70)

        trace_path = test_with_grid_serializer(world, seed=42)
        results[world] = trace_path

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    successes = [w for w, path in results.items() if path is not None]

    print(f"Successes: {len(successes)}/{len(test_worlds)}")
    for world in successes:
        print(f"  ✅ {world}")

    failures = [w for w, path in results.items() if path is None]
    if failures:
        print()
        print(f"Failures: {len(failures)}/{len(test_worlds)}")
        for world in failures:
            print(f"  ❌ {world}")

    print()
    print("Saved traces can be used for innav probing with:")
    print("  python run_innav_eval.py --trace-file navigation_traces/trace_GridSerializer_<world>_seed42.json")
    print()


if __name__ == "__main__":
    main()
