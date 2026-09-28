"""Test different reasoning_effort settings with GPT-5 models.

Compare hallucination rates at different reasoning depths:
low, medium, high (and xhigh if supported).

Usage:
    python scripts/smoke/reasoning_effort.py                # Default seed=42
    python scripts/smoke/reasoning_effort.py --seed 123     # Custom seed
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
parser = argparse.ArgumentParser(description="Test reasoning effort effects on hallucination")
parser.add_argument("--seed", type=int, default=42, help="Random seed for environment (default: 42)")
args = parser.parse_args()

# Test with different reasoning efforts
model = "gpt-5.4-mini"  # Update as needed
efforts = ["low", "medium", "high"]

print(f"Testing {model} with different reasoning_effort settings (seed={args.seed})")
print("=" * 70)

for effort in efforts:
    print(f"\n{'='*70}")
    print(f"REASONING_EFFORT = {effort}")
    print(f"{'='*70}")

    env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

    lm = OpenAILM(
        model=model,
        api_key=os.environ["OPENAI_API_KEY"],
        reasoning_effort=effort
    )

    print(f"LM configured: model={lm.model}, reasoning_effort={lm.reasoning_effort}")

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
        seed=args.seed,  # Use specified seed
        verbose=False,
    )

    print(f"\nResults:")
    print(results.summary().to_string(index=False))
    print(f"\nOverall hallucination rate: {results.hallucination_rate():.1%}")

print("\n" + "="*70)
print("Reasoning effort comparison complete!")
print("="*70)
