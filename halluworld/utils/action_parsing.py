"""Action parsing utilities for navigation.

Centralized action parsing logic extracted from halluworld/innav.py
to ensure consistency across all navigation components.
"""

import re
import os
from typing import Optional


def _llm_judge_action(response: str) -> Optional[int]:
    """Last resort: Use GPT-4o-mini to interpret action intent.

    This is called only when all regex/pattern matching fails.
    Uses a small, fast LLM to understand what action the model intended.

    Args:
        response: The unparseable LM response

    Returns:
        int | None: Action number (0-5) if LLM can interpret it, None otherwise
    """
    # Only import OpenAI if needed (avoid dependency for most cases)
    try:
        from halluworld.lm.openai_lm import OpenAILM
    except ImportError:
        print("⚠️  LLM judge unavailable (OpenAI import failed)")
        return None

    # Check if API key available
    if not os.environ.get("OPENAI_API_KEY"):
        return None

    print(f"🤖 LLM judge fallback: Interpreting response with GPT-4o-mini")
    print(f"   Original response: {response[:100]}")

    # Create judge LM (fast, cheap, good enough)
    judge = OpenAILM(model="gpt-4o-mini", max_tokens=10)

    # Rubric: Clear action vocabulary
    judge_prompt = f"""You are helping parse a navigation agent's response. The agent should choose ONE action from this list:

**Valid Actions:**
0 = Turn left (rotate 90° counterclockwise)
1 = Turn right (rotate 90° clockwise)
2 = Move forward (step in current direction)
3 = Pick up (grab object in front of you)
4 = Drop (release carried object)
5 = Toggle (open/close door in front of you)

**Agent's Response:**
"{response}"

**Task:** What action number (0-5) did the agent intend?

**Rules:**
- If the response clearly indicates one of the actions above, return ONLY that number (0-5)
- If the response is ambiguous or doesn't indicate any action, return: NONE
- Do NOT explain, just return the number or NONE

**Your answer:**"""

    try:
        judge_response = judge.query(system="You extract action numbers from text.", user=judge_prompt)
        judge_answer = judge_response.text.strip()

        # Parse judge's response
        if judge_answer in ['0', '1', '2', '3', '4', '5']:
            action = int(judge_answer)
            print(f"   ✅ LLM judge extracted: {action}")
            return action
        else:
            print(f"   ❌ LLM judge could not interpret (returned: {judge_answer})")
            return None

    except Exception as e:
        print(f"   ⚠️  LLM judge error: {e}")
        return None


def parse_action(response: str, use_llm_fallback: bool = True) -> int | None:
    """Parse action number from LM response.

    This function extracts a valid MiniGrid action (0-5) from various response formats:
    - Direct digit: "2"
    - With text: "I choose action 2"
    - Parenthesized: "(2)"
    - Full sentence: "The best action is 2."

    Args:
        response: Raw LM response string
        use_llm_fallback: If True, use GPT-4o-mini as last resort when regex fails (default: True)

    Returns:
        int | None: Action number (0-5) if parsing succeeds, None otherwise
            - 0: Turn left
            - 1: Turn right
            - 2: Move forward
            - 3: Pick up object
            - 4: Drop object
            - 5: Toggle door

    Examples:
        >>> parse_action("2")
        2
        >>> parse_action("I'll move forward (2)")
        2
        >>> parse_action("Action: 3")
        3
        >>> parse_action("invalid response")
        None

    Note:
        This is the canonical action parsing logic extracted from innav.py.
        Any changes to parsing behavior should be made here to maintain consistency.

        If all regex/pattern strategies fail and use_llm_fallback=True, will attempt
        to use GPT-4o-mini as a last resort to interpret the action intent.
    """
    response = response.strip()

    # Strategy 1: Direct match - single digit response
    # Handles: "0", "1", "2", "3", "4", "5"
    if response in ['0', '1', '2', '3', '4', '5']:
        return int(response)

    # Strategy 2: First digit in response
    # Handles: "I choose 2", "2 is best", "Action 2"
    for char in response:
        if char in '012345':
            return int(char)

    # Strategy 3: Regex word boundary match
    # Handles: "(2)", "action: 2", "2."
    # More strict - requires digit to be a separate token
    match = re.search(r'\b([0-5])\b', response)
    if match:
        return int(match.group(1))

    # Strategy 4 (LAST RESORT): LLM judge fallback
    # If all pattern matching failed, use GPT-4o-mini to interpret intent
    if use_llm_fallback:
        llm_result = _llm_judge_action(response)
        if llm_result is not None:
            return llm_result

    # All strategies failed - cannot parse
    return None
