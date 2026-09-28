"""Test Qwen with benchmark-style prompt to see if content field is populated."""
import os
from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("BASETEN_API_KEY", "vvfY1jrg.KzZyhNdgakDp1uIi4qBPheIsHr2CETHb"),
    base_url="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
)

# Simulate benchmark-style prompt with long object list
user_prompt = """You are an agent in a grid world. Here's what you currently see:

- red key, 3 steps ahead, 2 steps to your left
- blue ball, 5 steps ahead, 1 step to your right
- yellow door (locked), 7 steps ahead
- red key, 2 steps ahead, 3 steps to your left
- green ball, 4 steps ahead, 2 steps to your right
- blue ball, 6 steps ahead, 3 steps to your left
- red door (closed), 8 steps ahead

Question: Is there a blue ball visible?"""

print("=" * 80)
print(f"Testing with max_tokens=2048, prompt length={len(user_prompt)}")
print("=" * 80)

response = client.chat.completions.create(
    model="",  # Empty for deployed models
    messages=[
        {"role": "system", "content": "You are an agent in a gridworld. You will be shown what you currently see, then asked a question about it. Answer precisely and concisely."},
        {"role": "user", "content": user_prompt}
    ],
    max_tokens=2048,
    temperature=0.0
)

choice = response.choices[0]

print(f"\nFinish reason: {choice.finish_reason}")
print(f"Content is None: {choice.message.content is None}")
print(f"Has reasoning field: {hasattr(choice.message, 'reasoning')}")
print(f"Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")

print("\n" + "=" * 80)
print("CONTENT FIELD:")
print("=" * 80)
print(repr(choice.message.content))

if hasattr(choice.message, 'reasoning') and choice.message.reasoning:
    reasoning = choice.message.reasoning
    print("\n" + "=" * 80)
    print(f"REASONING FIELD (length={len(reasoning)}):")
    print("=" * 80)
    # Show last 300 chars
    if len(reasoning) > 300:
        print(f"...{reasoning[-300:]}")
    else:
        print(reasoning)
