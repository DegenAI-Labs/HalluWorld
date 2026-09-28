"""Navigate until sufficiency validator fires, then stop and probe.

Instead of navigating for full max_steps, we stop as soon as we reach a
"sufficient" point for probing:
- Substantiality: ≥10 steps, moved ≥5 units from start
- Sanity: Agent stable (not stuck/looping)
- Suitability: Within ±N diagonal of static probe location

This saves time and ensures we probe while agent is still stable.
"""

import json
import math
from typing import List, Tuple, Optional

from halluworld.utils.action_parsing import parse_action
from halluworld.tracks.innav.probe_windows import (
    is_agent_stable,
    has_made_progress,
    diagonal_distance,
)
from halluworld.data import LEVELS_DIR


def check_sufficiency(
    trajectory: List[dict],
    current_step_idx: int,
    static_probe_locations: List[Tuple[int, int]],
    min_steps: int = 10,
    max_distance: float = 6.0,
    min_distance_from_start: float = 5.0,
) -> Optional[Tuple[int, int]]:
    """Check if current position is sufficient for probing.

    Returns the nearest static location if sufficient, None otherwise.

    Args:
        trajectory: Navigation trajectory so far
        current_step_idx: Current step index
        static_probe_locations: Target static probe locations
        min_steps: Minimum steps before sufficiency
        max_distance: Max diagonal distance to static location
        min_distance_from_start: Minimum movement from start

    Returns:
        (x, y) of nearest static location if sufficient, None otherwise
    """
    # Check 1: Substantiality (enough steps)
    if current_step_idx < min_steps:
        return None

    # Check 2: Substantiality (moved from start)
    if not has_made_progress(trajectory, current_step_idx, min_distance_from_start):
        return None

    # Check 3: Sanity (stable)
    if not is_agent_stable(trajectory, current_step_idx, window=10):
        return None

    # Check 4: Long-term stability (if enough history)
    if current_step_idx >= 15:
        if not is_agent_stable(trajectory, current_step_idx, window=15):
            return None

    # Check 5: Suitability (near static location)
    current_pos = (
        int(trajectory[current_step_idx]['position'][0]),
        int(trajectory[current_step_idx]['position'][1])
    )

    for static_loc in static_probe_locations:
        dist = diagonal_distance(current_pos, static_loc)
        if dist <= max_distance:
            # SUFFICIENCY ACHIEVED!
            return static_loc

    return None


def navigate_until_sufficient(
    env,
    lm,
    serializer,
    static_probe_locations: List[Tuple[int, int]],
    seed: int = 42,
    max_steps: int = 50,
    min_steps: int = 10,
    max_distance: float = 6.0,
    min_distance_from_start: float = 5.0,
) -> dict:
    """Navigate until sufficiency validator fires, then stop.

    Returns:
        dict with:
        - trajectory: Navigation steps until sufficiency
        - sufficient: Boolean flag (did we reach sufficiency?)
        - sufficient_step: Step index where sufficiency was reached
        - sufficient_location: Which static location we're near
        - sufficient_distance: Distance to that location
        - reason: Why we stopped
    """
    # Helper to convert messages to LM query
    def generate_from_messages(messages_list):
        """Convert messages to system+user format for LM."""
        system_prompt = None
        user_parts = []
        for msg in messages_list:
            if msg["role"] == "system":
                if system_prompt is None:
                    system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                user_parts.append(f"Assistant: {msg['content']}")
        user_prompt = "\\n\\n".join(user_parts)
        response = lm.query(system=system_prompt or "", user=user_prompt)
        return response.text

    env.reset(seed=seed)
    trajectory = []

    # System prompt with efficiency + anti-stuck nudges
    # (From archive_scripts/navigate_with_efficiency_nudge.py - proven to help navigation)
    system_msg = {
        "role": "system",
        "content": (
            "You are an agent navigating a gridworld environment. "
            "Your objective is to find and reach the goal location efficiently. "
            "The goal is typically somewhere ahead of you (north, east, or both), "
            "reasonably far from your starting position, though the exact location is unknown.\\n\\n"
            "IMPORTANT - Movement Rules:\\n"
            "- Objects (keys, balls) BLOCK your movement - you cannot move through them\\n"
            "- To pass an object in your path, you must PICK IT UP first (action 3)\\n"
            "- After picking up, you can move forward, then drop it (action 4) if needed\\n"
            "- Walls BLOCK movement - if stuck, TURN (actions 0 or 1) to find a new path\\n"
            "- When you reach the goal, the episode will end automatically\\n\\n"
            "Object Behaviors (use this knowledge!):\\n"
            "- Keys can open doors - pick up a key if you see a locked door ahead\\n"
            "- Doors must be TOGGLED (action 5) to open (may need matching colored key first)\\n"
            "- Balls are movable objects - pick up and drop to clear paths\\n"
            "- Goal location is marked - when you see it, navigate directly toward it\\n\\n"
            "EFFICIENCY SCORING:\\n"
            "- You start with 100 points\\n"
            "- Each timestep costs you 0.5 points\\n"
            "- Navigate efficiently to maximize your score!\\n\\n"
            "ADAPTIVE STRATEGY:\\n"
            "- If you stay at the same position for 3+ steps, you are STUCK\\n"
            "- When stuck: STOP repeating the same action - try turning or a different approach\\n"
            "- Repeating failed actions wastes timesteps and loses points!"
        )
    }

    messages = [system_msg]
    done = False

    for step in range(max_steps):
        if done:
            break

        # Get current state
        state_desc = serializer.serialize(env)
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        carrying = env.carrying

        # Check if stuck (same position for 3+ steps)
        stuck_warning = ""
        if len(trajectory) >= 3:
            last_3_positions = [trajectory[-i]['position'] for i in range(1, 4)]
            if len(set(last_3_positions)) == 1:  # All same position
                stuck_warning = "⚠️  WARNING: You've been stuck at the same position for 3 steps! Try turning or a different action.\\n\\n"

        # Format action prompt
        action_prompt = f"""Step {step + 1}:
{state_desc}

{stuck_warning}Choose your next action:

(0) Turn left
(1) Turn right
(2) Move forward
(3) Pick up object in front of you
(4) Drop carried object
(5) Toggle door in front of you

Respond with the number of your chosen action (0-5)."""

        messages.append({"role": "user", "content": action_prompt})

        # Get LM response
        response = generate_from_messages(messages)

        # Parse action using shared utility (same logic as innav.py)
        action = parse_action(response)

        if action is None:
            # Failed to parse - cannot continue navigation
            print(f"⚠️  Could not parse action at step {step}: {response[:100]}")
            break

        messages.append({"role": "assistant", "content": response.strip()})

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos
        new_carrying = env.carrying
        done = terminated or truncated

        # Record step
        trajectory.append({
            "step": step,
            "position": tuple(agent_pos),
            "direction": ["right", "down", "left", "up"][agent_dir],
            "carrying": carrying.type if carrying else None,
            "action": action,
            "action_chosen": f"{action} ({['Turn left', 'Turn right', 'Move forward', 'Pick up', 'Drop', 'Toggle'][action]})",
            "new_position": tuple(new_pos),
            "new_carrying": new_carrying.type if new_carrying else None,
            "position_changed": not (agent_pos[0] == new_pos[0] and agent_pos[1] == new_pos[1]),
            "reward": reward,
        })

        # CHECK SUFFICIENCY after each step
        sufficient_location = check_sufficiency(
            trajectory,
            step,
            static_probe_locations,
            min_steps=min_steps,
            max_distance=max_distance,
            min_distance_from_start=min_distance_from_start
        )

        if sufficient_location is not None:
            # SUFFICIENCY ACHIEVED - STOP NAVIGATION
            current_pos = tuple(new_pos)
            dist = diagonal_distance(current_pos, sufficient_location)

            return {
                "trajectory": trajectory,
                "sufficient": True,
                "sufficient_step": step,
                "sufficient_position": current_pos,
                "sufficient_location": sufficient_location,
                "sufficient_distance": dist,
                "reason": f"Sufficiency reached at step {step}: near {sufficient_location} (dist={dist:.2f})",
                "reached_goal": done,
                "messages": messages,  # For innav probing
            }

        # Check if goal reached
        if done:
            return {
                "trajectory": trajectory,
                "sufficient": False,
                "sufficient_step": None,
                "sufficient_position": None,
                "sufficient_location": None,
                "sufficient_distance": None,
                "reason": f"Goal reached at step {step} before sufficiency",
                "reached_goal": True,
                "messages": messages,
            }

    # Max steps reached without sufficiency
    return {
        "trajectory": trajectory,
        "sufficient": False,
        "sufficient_step": None,
        "sufficient_position": None,
        "sufficient_location": None,
        "sufficient_distance": None,
        "reason": f"Max steps ({max_steps}) reached without sufficiency",
        "reached_goal": False,
        "messages": messages,
    }


if __name__ == "__main__":
    from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
    from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
    from halluworld.lm.openai_lm import OpenAILM

    # Example: Test on corridor with early stopping
    env = AsciiEnv.from_file(str(LEVELS_DIR / "corridor_gauntlet.txt"))
    lm = OpenAILM(model="gpt-5.4-mini", max_tokens=256, reasoning_effort="high")
    serializer = SymbolicSerializer()

    static_locations = [
        (5, 2),   # Near first key
        (10, 2),  # Midpoint
        (15, 2),  # Near goal
    ]

    print("Testing early-stop navigation...")
    print("Will stop as soon as we reach a sufficient point")
    print()

    result = navigate_until_sufficient(
        env=env,
        lm=lm,
        serializer=serializer,
        static_probe_locations=static_locations,
        seed=42,
        max_steps=50,
        max_distance=6.0  # Try 6, 5, or 4
    )

    print("="*70)
    print(f"Result: {result['reason']}")
    print("="*70)
    print()

    if result['sufficient']:
        print(f"✅ SUFFICIENCY REACHED!")
        print(f"  Step: {result['sufficient_step']}")
        print(f"  Position: {result['sufficient_position']}")
        print(f"  Nearest static: {result['sufficient_location']}")
        print(f"  Distance: {result['sufficient_distance']:.2f}")
        print()
        print("→ At this point, we would ask innav probe")
        print("→ Agent is stable, substantial, and suitable")
        print()
        print(f"Message history captured: {len(result['messages'])} messages")
        print(f"  System: {sum(1 for m in result['messages'] if m['role'] == 'system')}")
        print(f"  User: {sum(1 for m in result['messages'] if m['role'] == 'user')}")
        print(f"  Assistant: {sum(1 for m in result['messages'] if m['role'] == 'assistant')}")
        print()
        print("Last assistant message (action at sufficient step):")
        last_assistant = [m for m in result['messages'] if m['role'] == 'assistant'][-1]
        print(f"  '{last_assistant['content'][:100]}...'")
    else:
        print(f"❌ Sufficiency not reached")
        print(f"  Reason: {result['reason']}")
        print(f"  Steps taken: {len(result['trajectory'])}")
