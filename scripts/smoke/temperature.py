"""Test different temperature settings with gpt-4o-mini.

Compare hallucination rates at temperature 0.0, 0.5, and 1.0.
Probes are seeded to ensure SAME questions across all runs.

Usage:
    python scripts/smoke/temperature.py                # Default seed=42
    python scripts/smoke/temperature.py --seed 123     # Custom seed
    python scripts/smoke/temperature.py --seed 42 && python scripts/smoke/temperature.py --seed 123
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
from halluworld.lm import OpenAILM
from halluworld.tracks.grid import PresenceEvaluator, LocationEvaluator
from halluworld.benchmark import run_benchmark

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test temperature effects on hallucination")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment (default: 42)")
args = parser.parse_args()

# Test with different temperatures
temperatures = [0.0, 0.5, 1.0]

print(f"Testing gpt-4o-mini with different temperature settings (seed={args.seed})")
print("=" * 70)

for temp in temperatures:
    print(f"\n{'='*70}")
    print(f"TEMPERATURE = {temp}")
    print(f"{'='*70}")

    env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

    lm = OpenAILM(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=temp
    )

    print(f"LM configured with: model={lm.model}, temperature={lm.temperature}")

    results = run_benchmark(
        env=env,
        serializer=SymbolicSerializer(),
        probes=[
            PresenceProbe(positive_rate=0.5, rng=random.Random(100)),  # Seeded for fair comparison
            LocationProbe(rng=random.Random(101))                       # Seeded for fair comparison
        ],
        lm=lm,
        evaluators=[PresenceEvaluator(), LocationEvaluator()],
        n_episodes=10,
        steps_before_probe=3,
        seed=args.seed,  # Use specified seed
        verbose=False,
    )

    print(f"\nResults:")
    print(results.summary().to_string(index=False))
    print(f"\nOverall hallucination rate: {results.hallucination_rate():.1%}")

print("\n" + "="*70)
print("Temperature comparison complete!")
print("="*70)
