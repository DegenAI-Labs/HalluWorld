"""Quick stress test for Qwen deployments to check capacity."""
import os
import sys
import time
from halluworld.lm import BasetenLM

# Test both deployments
deployments = {
    "Qwen-Instruct": "https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1",
    "Qwen-Thinking": "https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1",
}

print("=" * 80)
print("Testing Qwen Deployment Capacity")
print("=" * 80)

for name, base_url in deployments.items():
    print(f"\n🧪 Testing {name}...")
    print(f"   URL: {base_url}")

    lm = BasetenLM(
        model=f"qwen-3-30b-{name.split('-')[1].lower()}",
        base_url=base_url,
        api_key=os.environ.get("BASETEN_API_KEY"),
        temperature=0.0,
    )

    # Try 3 quick requests
    success_count = 0
    total_time = 0

    for i in range(3):
        try:
            start = time.time()
            response = lm.query(
                system="You are a helpful assistant.",
                user=f"What is {i+1} + {i+1}? Answer with just the number."
            )
            elapsed = time.time() - start
            total_time += elapsed

            print(f"   ✅ Request {i+1}: {response.text[:50]} ({elapsed:.2f}s)")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Request {i+1} failed: {e}")

    avg_time = total_time / success_count if success_count > 0 else 0
    print(f"\n   Summary: {success_count}/3 successful, avg {avg_time:.2f}s per request")

    if success_count < 3:
        print(f"   ⚠️  WARNING: {name} might be overloaded or rate-limited!")

print("\n" + "=" * 80)
print("Deployment test complete!")
print("=" * 80)
