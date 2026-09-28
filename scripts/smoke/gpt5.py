"""Test GPT-5 reasoning model (should work without temperature parameter).

This tests that our implementation correctly handles GPT-5 models
by not passing the temperature parameter.
"""
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

# Test with GPT-5 model (update model name as needed)
model = "gpt-5.4-mini"  # Change to available GPT-5 model

print(f"Testing {model} (reasoning model - no temperature support)")
print("=" * 70)

env = make_env(render_mode="rgb_array", agent_view_size=7, num_objects=1)

# Try to set temperature - should be ignored with a warning
lm = OpenAILM(
    model=model,
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.5  # Will be ignored for GPT-5
)

print(f"\nLM configured:")
print(f"  model: {lm.model}")
print(f"  temperature: {lm.temperature} (None = not supported)")
print(f"  max_tokens: {lm.max_tokens}")

print(f"\nRunning benchmark with {model}...")

results = run_benchmark(
    env=env,
    serializer=SymbolicSerializer(),
    probes=[PresenceProbe(positive_rate=0.5), LocationProbe()],
    lm=lm,
    evaluators=[PresenceEvaluator(), LocationEvaluator()],
    n_episodes=10,
    steps_before_probe=3,
    seed=42,
    verbose=True,
)

print()
print(results.summary().to_string(index=False))
print(f"\nHallucination rate (presence): {results.hallucination_rate('presence'):.1%}")
print(f"Hallucination rate (location): {results.hallucination_rate('location'):.1%}")

print("\n" + "="*70)
print(f"{model} test complete!")
print("="*70)
