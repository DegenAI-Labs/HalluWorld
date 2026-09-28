#!/usr/bin/env python3
"""Test shared action parsing utility."""

import sys
sys.path.insert(0, '.')

from halluworld.utils.action_parsing import parse_action

# Test cases from actual LM responses
test_cases = [
    # Direct digit responses
    ("2", 2),
    ("0", 0),
    ("5", 5),

    # With whitespace
    ("  2  ", 2),
    ("2\n", 2),

    # In sentences
    ("I choose action 2", 2),
    ("Action 3 is best", 3),
    ("Move forward (2)", 2),

    # Parenthesized
    ("(2)", 2),
    ("(0)", 0),

    # Invalid
    ("invalid", None),
    ("", None),
    ("6", None),  # Out of range
    ("no digits here", None),
]

print("Testing shared action parsing utility")
print("="*60)

all_pass = True
for response, expected in test_cases:
    result = parse_action(response)
    status = "✓" if result == expected else "✗"

    if result != expected:
        all_pass = False

    result_str = str(result) if result is not None else "None"
    expected_str = str(expected) if expected is not None else "None"
    print(f"{status} parse_action({repr(response):30s}) = {result_str:5} (expected {expected_str})")

print("="*60)
if all_pass:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed")
    sys.exit(1)
