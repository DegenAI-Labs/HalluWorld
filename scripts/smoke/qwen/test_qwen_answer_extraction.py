"""Test Qwen answer extraction from reasoning field."""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from halluworld.lm.baseten_lm import BasetenLM

# Get base_url from environment or use default
base_url = os.environ.get("BASETEN_BASE_URL",
                          "https://model-<YOUR-DEPLOYMENT-ID>.api.baseten.co/environments/production/sync/v1")

# Initialize Qwen with new extraction feature
lm = BasetenLM(
    model="Qwen/Qwen3-30B-A3B-Thinking-2507",
    base_url=base_url,
    api_key=os.environ.get("BASETEN_API_KEY"),
    temperature=0.0,
)

print(f"Using base_url: {base_url}")

print("=" * 80)
print("Testing Qwen Answer Extraction")
print("=" * 80)

# Test simple yes/no question
response = lm.query(
    system="You are an agent in a gridworld.",
    user="Is there a red key visible in this list? [red key, blue ball, yellow door]. Answer yes or no."
)

print(f"\nExtracted answer: {response.text}")
print(f"Length: {len(response.text)} chars")
print(f"Tokens: prompt={response.prompt_tokens}, completion={response.completion_tokens}")
print()

# Test counting question
response2 = lm.query(
    system="You are an agent in a gridworld.",
    user="How many red objects are visible in this list? [red key, blue ball, red door, green ball]. Answer with just a number."
)

print(f"\nExtracted answer: {response2.text}")
print(f"Length: {len(response2.text)} chars")
print()

print("=" * 80)
print("Answer extraction test complete!")
print("If answers are concise (not verbose reasoning), extraction is working!")
print("=" * 80)
