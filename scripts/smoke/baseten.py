"""Test Baseten deployed or serverless models.

Usage:
    # Deployed model
    python scripts/smoke/baseten.py \
        --model "google/gemma-4-26B-A4B-it" \
        --base-url "https://model-xxxx.api.baseten.co/environments/production/sync/v1" \
        --seed 42

    # Serverless model (e.g., GLM-5)
    python scripts/smoke/baseten.py \
        --model "zai-org/GLM-5" \
        --base-url "https://inference.baseten.co/v1" \
        --seed 42 --temperature 0.7
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
from halluworld.lm import BasetenLM
from halluworld.tracks.grid import PresenceEvaluator, LocationEvaluator
from halluworld.benchmark import run_benchmark

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Test Baseten models")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
parser.add_argument("--model", type=str, required=True,
                    help="Model name (e.g., 'google/gemma-4-26B-A4B-it')")
parser.add_argument("--base-url", type=str, required=True,
                    help="Baseten API base URL")
parser.add_argument("--temperature", type=float, default=0.0,
                    help="Temperature (default: 0.0)")
args = parser.parse_args()

print(f"Testing Baseten model: {args.model}")
print(f"Base URL: {args.base_url}")
print(f"Seed: {args.seed}, Temperature: {args.temperature}")
print("=" * 70)

env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

lm = BasetenLM(
    model=args.model,
    base_url=args.base_url,
    api_key=os.environ.get("BASETEN_API_KEY"),
    temperature=args.temperature
)

print(f"\nLM configured:")
print(f"  model: {lm.model}")
print(f"  temperature: {lm.temperature}")
print(f"  base_url: {lm.base_url}")

print(f"\nRunning benchmark...")

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
    verbose=True,
)

print()
print(results.summary().to_string(index=False))
print(f"\nHallucination rate (presence): {results.hallucination_rate('presence'):.1%}")
print(f"Hallucination rate (location): {results.hallucination_rate('location'):.1%}")

print("\n" + "="*70)
print("Baseten test complete!")
print("="*70)
