#!/usr/bin/env python3
"""Test LLM fallback for action parsing."""

import sys
sys.path.insert(0, '.')

from halluworld.utils.action_parsing import parse_action

print("="*70)
print("Testing LLM Fallback for Action Parsing")
print("="*70)
print()

# Cases where regex should fail but LLM can understand
challenging_cases = [
    ("Let me turn to the left", 0),  # Natural language
    ("I think moving forward would be best", 2),  # Sentence
    ("pick up that object", 3),  # Natural command
    ("toggle the door", 5),  # Natural command
    ("I'll go right", 1),  # Informal
]

print("Testing cases where regex fails but LLM should succeed:")
print("-"*70)

for response, expected in challenging_cases:
    # First try without LLM fallback (should fail)
    result_no_llm = parse_action(response, use_llm_fallback=False)

    # Then try with LLM fallback (should succeed)
    print(f"\nResponse: {response!r}")
    print(f"  Without LLM: {result_no_llm}")

    result_with_llm = parse_action(response, use_llm_fallback=True)
    print(f"  With LLM:    {result_with_llm} (expected {expected})")

    if result_with_llm == expected:
        print(f"  ✅ LLM fallback worked!")
    else:
        print(f"  ⚠️  Unexpected result")

print()
print("="*70)
print("Test complete")
print("="*70)
