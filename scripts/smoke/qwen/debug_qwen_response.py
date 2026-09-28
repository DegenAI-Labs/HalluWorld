import os
from openai import OpenAI
import json

client = OpenAI(
    api_key=os.environ.get("BASETEN_API_KEY", "vvfY1jrg.KzZyhNdgakDp1uIi4qBPheIsHr2CETHb"),
    base_url="https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1"
)

response = client.chat.completions.create(
    model="",
    messages=[
        {"role": "system", "content": "You are an agent in a gridworld."},
        {"role": "user", "content": "Is there a red key visible? Answer yes or no."}
    ],
    max_tokens=256,
    temperature=0.0
)

print("="*80)
print("FULL RESPONSE OBJECT:")
print("="*80)
print(response)
print("\n" + "="*80)
print("CHOICE[0]:")
print("="*80)
choice = response.choices[0]
print(f"Message: {choice.message}")
print(f"Content: {choice.message.content}")
print(f"Has 'reasoning' attr: {hasattr(choice.message, 'reasoning')}")
if hasattr(choice.message, 'reasoning'):
    print(f"Reasoning: {choice.message.reasoning}")
print("\n" + "="*80)
print("DICT REPRESENTATION:")
print("="*80)
print(json.dumps(response.model_dump(), indent=2))
