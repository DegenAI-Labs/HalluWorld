"""Test Qwen with 1024 max_tokens to see if content field gets populated."""
import os
from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("BASETEN_API_KEY", "vvfY1jrg.KzZyhNdgakDp1uIi4qBPheIsHr2CETHb"),
    base_url="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
)

print("=" * 80)
print("Testing Qwen with max_tokens=1024")
print("=" * 80)

response = client.chat.completions.create(
    model="",  # Empty for deployed models
    messages=[
        {"role": "system", "content": "You are an agent in a gridworld."},
        {"role": "user", "content": "Is there a red key visible? Answer yes or no."}
    ],
    max_tokens=1024,  # Much more space!
    temperature=0.0
)

choice = response.choices[0]

print(f"\nFinish reason: {choice.finish_reason}")
print(f"Content is None: {choice.message.content is None}")
print(f"Has reasoning field: {hasattr(choice.message, 'reasoning')}")

print("\n" + "=" * 80)
print("CONTENT FIELD:")
print("=" * 80)
print(choice.message.content)

if hasattr(choice.message, 'reasoning'):
    reasoning = choice.message.reasoning
    print("\n" + "=" * 80)
    print(f"REASONING FIELD (length={len(reasoning) if reasoning else 0}):")
    print("=" * 80)
    # Show last 500 chars to see the conclusion
    if reasoning and len(reasoning) > 500:
        print(f"...{reasoning[-500:]}")
    else:
        print(reasoning)

print("\n" + "=" * 80)
print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
print("=" * 80)
