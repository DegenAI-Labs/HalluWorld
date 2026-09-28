"""Detailed temperature test - shows actual LM responses."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from halluworld.tracks.grid import make_env
from halluworld.tracks.grid import SymbolicSerializer
from halluworld.tracks.grid import PresenceProbe, LocationProbe
from halluworld.lm import OpenAILM
from halluworld.tracks.grid import PresenceEvaluator, LocationEvaluator
from halluworld.benchmark import run_benchmark
import random

# Test with temperature 1.0 to see actual responses
temp = 1.0

print(f"Testing temperature={temp} with VERBOSE output")
print("=" * 70)

env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

# IMPORTANT: Seed the probes' RNG for reproducibility
probe_rng = random.Random(42)

results = run_benchmark(
    env=env,
    serializer=SymbolicSerializer(),
    probes=[
        PresenceProbe(positive_rate=0.5, rng=random.Random(42)),
        LocationProbe(rng=random.Random(43))
    ],
    lm=OpenAILM(
        model="gpt-4o-mini",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=temp
    ),
    evaluators=[PresenceEvaluator(), LocationEvaluator()],
    n_episodes=5,  # Just 5 for detailed inspection
    steps_before_probe=3,
    seed=42,
    verbose=True,  # Shows each result
)

print("\n" + "=" * 70)
print("DETAILED ANALYSIS")
print("=" * 70)

# Show raw responses for first few results
print("\nFirst 10 raw LM responses:")
for i, r in enumerate(results.results[:10]):
    print(f"\n[{i}] {r.probe_name}:")
    print(f"  Question: {r.question[:80]}...")
    print(f"  LM Response: {r.lm_response!r}")
    print(f"  Ground Truth: {r.ground_truth}")
    print(f"  Correct: {r.is_correct}, Score: {r.score}")
    if 'parse_note' in r.metadata:
        print(f"  Parse Note: {r.metadata['parse_note']}")

print("\n" + "=" * 70)
print(results.summary().to_string(index=False))
