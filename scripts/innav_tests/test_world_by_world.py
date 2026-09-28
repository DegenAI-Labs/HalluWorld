"""Test each world individually with GPT-5.4-mini high.

Tests one world at a time for debugging and getting good navigation traces.
Essential for producing comprehensive innav results table.
"""

import sys
sys.path.insert(0, '.')
import json
import io
from contextlib import redirect_stdout, redirect_stderr

from halluworld.tracks.innav.navigation import navigate_until_sufficient
from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.lm.openai_lm import OpenAILM
from halluworld.data import LEVELS_DIR


def test_single_world(level_name: str, seed: int = 42, verbose: bool = True):
    """Test navigation on a single world."""

    # Load config
    config_path = fstr(LEVELS_DIR / "{level_name}.innav.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load world
    level_path = fstr(LEVELS_DIR / "{level_name}.txt")
    env = AsciiEnv.from_file(level_path)

    # Override start position if specified
    if config['innav_start_position']:
        # Note: This would require modifying the env after reset
        # For now, just note it in the output
        custom_start = config['innav_start_position']
        if verbose:
            print(f"  Note: Config specifies custom start {custom_start}")
            print(f"        (Current implementation uses world default)")

    lm = OpenAILM(model="gpt-5.4", max_tokens=256, reasoning_effort="medium")
    serializer = SymbolicSerializer()

    # Extract params
    targets = [tuple(t) for t in config['navigation_target_locations']]
    params = config['navigation_params']

    if verbose:
        print(f"  Targets: {targets}")
        print(f"  Max steps: {params['max_steps']}")
        print(f"  Sufficiency distance: ±{params['sufficiency_distance']}")
        print()

    # Run navigation (suppress parse warnings)
    if not verbose:
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
    else:
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

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test individual world navigation")
    parser.add_argument('world', type=str, help='World name (e.g., corridor_gauntlet)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

    args = parser.parse_args()

    print("="*70)
    print(f"TESTING: {args.world.upper()}")
    print("="*70)
    print(f"Seed: {args.seed}")
    print(f"Model: GPT-5.4-mini with reasoning_effort='high'")
    print()

    try:
        result = test_single_world(args.world, seed=args.seed, verbose=not args.quiet)

        print()
        print("="*70)
        print("RESULT")
        print("="*70)

        if result['sufficient']:
            print(f"✅ SUCCESS!")
            print(f"   Step: {result['sufficient_step']}")
            print(f"   Position: {result['sufficient_position']}")
            print(f"   Target: {result['sufficient_location']}")
            print(f"   Distance: {result['sufficient_distance']:.2f}")
            print(f"   Messages captured: {len(result['messages'])}")
            print()
            print("→ Ready for innav probing!")
        else:
            print(f"❌ FAILED")
            print(f"   Reason: {result['reason']}")
            print(f"   Steps taken: {len(result['trajectory'])}")
            print()
            if 'Max steps' in result['reason']:
                print("→ Consider: Increase max_steps or adjust targets")
            elif 'Goal reached' in result['reason']:
                print("→ Agent reached goal before sufficiency")
                print("→ Consider: Adjust targets to earlier positions")

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

    print()
