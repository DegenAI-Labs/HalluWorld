"""Comprehensive test of Anthropic Claude models.

Tests temperature and thinking_effort (for Claude 4.6 models).

Usage:
    python scripts/smoke/anthropic.py                           # Default: Sonnet 4.6, temp test, seed=42
    python scripts/smoke/anthropic.py --seed 123                # Custom seed
    python scripts/smoke/anthropic.py --test thinking           # Test thinking effort
    python scripts/smoke/anthropic.py --model claude-opus-4-6   # Different model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import argparse
import random
from halluworld.tracks.grid import make_env
from halluworld.tracks.grid import SymbolicSerializer
from halluworld.tracks.grid import PresenceProbe, LocationProbe
from halluworld.lm import AnthropicLM
from halluworld.tracks.grid import PresenceEvaluator, LocationEvaluator
from halluworld.benchmark import run_benchmark

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test Anthropic Claude models")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--model", type=str, default="claude-sonnet-4-6",
                    help="Model name (default: claude-sonnet-4-6)")
parser.add_argument("--test", type=str, default="temperature",
                    choices=["temperature", "thinking"],
                    help="Test type: temperature or thinking (default: temperature)")
args = parser.parse_args()

print(f"Testing {args.model} (seed={args.seed})")
print("=" * 70)

if args.test == "temperature":
    # Test different temperatures
    temperatures = [0.0, 0.5, 1.0]

    print("\nTesting temperature effects:")
    print("-" * 70)

    for temp in temperatures:
        print(f"\n{'='*70}")
        print(f"TEMPERATURE = {temp}")
        print(f"{'='*70}")

        env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

        lm = AnthropicLM(
            model=args.model,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=temp
        )

        print(f"LM configured: model={lm.model}, temperature={lm.temperature}")

        results = run_benchmark(
            env=env,
            serializer=SymbolicSerializer(),
            probes=[
                PresenceProbe(positive_rate=0.5, rng=random.Random(100)),
                LocationProbe(rng=random.Random(101))
            ],
            lm=lm,
            evaluators=[PresenceEvaluator(), LocationEvaluator()],
            n_episodes=10,
            steps_before_probe=3,
            seed=args.seed,
            verbose=False,
        )

        print(f"\nResults:")
        print(results.summary().to_string(index=False))
        print(f"\nOverall hallucination rate: {results.hallucination_rate():.1%}")

elif args.test == "thinking":
    # Test different thinking efforts (Claude 4.6 models only)
    if "4-6" not in args.model:
        print(f"\nWARNING: Model {args.model} may not support adaptive thinking.")
        print("Adaptive thinking is only available for Claude Opus 4.6 and Sonnet 4.6.")
        print("Continuing anyway...\n")

    efforts = ["low", "medium", "high"]

    print("\nTesting thinking effort effects:")
    print("-" * 70)

    for effort in efforts:
        print(f"\n{'='*70}")
        print(f"THINKING_EFFORT = {effort}")
        print(f"{'='*70}")

        env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

        lm = AnthropicLM(
            model=args.model,
            api_key=os.environ["ANTHROPIC_API_KEY"],
            temperature=1.0,  # MUST be 1.0 when using adaptive thinking
            thinking_effort=effort
        )

        print(f"LM configured: model={lm.model}, thinking_effort={lm.thinking_effort}")

        results = run_benchmark(
            env=env,
            serializer=SymbolicSerializer(),
            probes=[
                PresenceProbe(positive_rate=0.5, rng=random.Random(100)),
                LocationProbe(rng=random.Random(101))
            ],
            lm=lm,
            evaluators=[PresenceEvaluator(), LocationEvaluator()],
            n_episodes=10,
            steps_before_probe=3,
            seed=args.seed,
            verbose=False,
        )

        print(f"\nResults:")
        print(results.summary().to_string(index=False))
        print(f"\nOverall hallucination rate: {results.hallucination_rate():.1%}")

print("\n" + "="*70)
print("Anthropic test complete!")
print("="*70)
