"""Test GPT-5.4-mini navigation on all Emmy's worlds using innav configs.

For each world:
1. Load innav config
2. Test navigation with real targets
3. Check if sufficiency is reached
4. Report success/failure
"""

import sys
sys.path.insert(0, '.')
import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from halluworld.tracks.innav.navigation import navigate_until_sufficient
from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.lm.openai_lm import OpenAILM
from halluworld.data import LEVELS_DIR


def load_innav_config(level_name: str):
    """Load innav config for a level."""
    config_path = fstr(LEVELS_DIR / "{level_name}.innav.json")
    with open(config_path, 'r') as f:
        return json.load(f)


def test_world_navigation(level_name: str, seeds: list = None):
    """Test navigation on a single world."""

    if seeds is None:
        seeds = [42, 123]

    # Load config
    try:
        config = load_innav_config(level_name)
    except FileNotFoundError:
        print(f"  ❌ Config not found: {level_name}.innav.json")
        return None

    # Load world
    level_path = fstr(LEVELS_DIR / "{level_name}.txt")
    try:
        env = AsciiEnv.from_file(level_path)
    except FileNotFoundError:
        print(f"  ❌ Level not found: {level_path}")
        return None

    lm = OpenAILM(model="gpt-5.4-mini", max_tokens=256, reasoning_effort="high")
    serializer = SymbolicSerializer()

    # Extract params from config
    targets = [tuple(t) for t in config['navigation_target_locations']]
    params = config['navigation_params']

    results = []

    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)

        try:
            # Suppress parse warnings
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
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

            if result['sufficient']:
                print(f"✅ Step {result['sufficient_step']}, "
                      f"target={result['sufficient_location']}, "
                      f"dist={result['sufficient_distance']:.2f}")
            else:
                reason = result['reason']
                if 'Max steps' in reason:
                    print(f"❌ Timeout ({params['max_steps']} steps)")
                elif 'Goal reached' in reason:
                    steps = len(result['trajectory'])
                    print(f"⚠️  Goal reached at step {steps} (before sufficiency)")
                else:
                    print(f"❌ {reason[:50]}")

            results.append(result)

        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            results.append(None)

    return results


def main():
    print("="*70)
    print("TEST ALL WORLDS - GPT-5.4-MINI NAVIGATION")
    print("="*70)
    print()
    print("Model: GPT-5.4-mini with reasoning_effort='high'")
    print("Using innav configs with real static probe locations")
    print()

    worlds = [
        'corridor_gauntlet',
        'dense_array',
        'rotation_challenge',
        'c2_fire_crossing',
        'c3_flood_room',
    ]

    all_results = {}

    for world in worlds:
        print(f"\n{world.upper()}")
        print("-" * 70)

        results = test_world_navigation(world, seeds=[42, 123])
        all_results[world] = results

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()

    total_success = 0
    total_tests = 0

    for world, results in all_results.items():
        if results is None:
            print(f"⚠️  {world}: SKIPPED (config/level not found)")
            continue

        successes = [r for r in results if r and r['sufficient']]
        total_success += len(successes)
        total_tests += len(results)

        success_rate = len(successes) / len(results) if results else 0
        status = "✅" if success_rate == 1.0 else "⚠️" if success_rate >= 0.5 else "❌"

        print(f"{status} {world:25} {len(successes)}/{len(results)} ({success_rate*100:3.0f}%)")

    print()
    print(f"Overall: {total_success}/{total_tests} ({total_success/total_tests*100:.0f}%)")
    print()

    # Recommendations
    if total_success == total_tests:
        print("🎉 PERFECT! All worlds navigable with GPT-5.4-mini high")
        print("   → Ready for innav experiments")
    elif total_success >= total_tests * 0.8:
        print("✅ GOOD! Most worlds work, minor adjustments needed")
        print("   → Consider adjusting innav_start_position for failures")
    elif total_success >= total_tests * 0.5:
        print("⚠️  PARTIAL: Some worlds work, others need adjustment")
        print("   → Adjust start positions or targets for failing worlds")
    else:
        print("❌ ISSUES: Most worlds timing out or failing")
        print("   → May need different navigation model or longer max_steps")

    print()


if __name__ == "__main__":
    main()
