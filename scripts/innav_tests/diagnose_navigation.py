"""Diagnose navigation quality - detailed step-by-step analysis."""

import os
import json
import random
from pathlib import Path

from minigrid.core.constants import IDX_TO_OBJECT
from halluworld.lm.openai_lm import OpenAILM
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer
from halluworld.tracks.grid.envs.ascii_env import AsciiEnv
from halluworld.data import LEVELS_DIR


def _generate_from_messages(lm, messages):
    """Convert messages to system+user format and query LM."""
    system_prompt = None
    user_parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            if system_prompt is None:
                system_prompt = content
        elif role == "user":
            user_parts.append(f"User: {content}")
        elif role == "assistant":
            user_parts.append(f"Assistant: {content}")

    user_prompt = "\n\n".join(user_parts)
    response = lm.query(system=system_prompt or "", user=user_prompt)
    return response.text


def diagnose_navigation(
    model_name: str,
    level_path: str,
    seed: int,
    max_steps: int = 50,
    reasoning_effort: str | None = None,
):
    """Run one episode with detailed step-by-step logging."""

    print(f"\n{'='*80}")
    print(f"NAVIGATION DIAGNOSTIC: {model_name}")
    print(f"Level: {level_path}")
    print(f"Seed: {seed}, Max steps: {max_steps}")
    if reasoning_effort:
        print(f"Reasoning effort: {reasoning_effort}")
    print(f"{'='*80}\n")

    # Setup
    lm = OpenAILM(model=model_name, max_tokens=256, reasoning_effort=reasoning_effort)
    env = AsciiEnv.from_file(level_path)
    serializer = SymbolicSerializer()

    # Reset
    obs, _ = env.reset(seed=seed)

    # System prompt with directional hint (non-leaky, just general guidance)
    system_msg = {
        "role": "system",
        "content": (
            "You are an agent navigating a gridworld environment. "
            "Your objective is to find and reach the goal location. "
            "The goal is typically somewhere ahead of you (north, east, or both), "
            "reasonably far from your starting position, though the exact location is unknown. "
            "You will be shown what you can currently observe, including objects, walls, and your position.\n\n"
            "IMPORTANT:\n"
            "- Objects (keys, balls) BLOCK your movement - you cannot move through them\n"
            "- To pass an object in your path, you must PICK IT UP first (action 3)\n"
            "- After picking up, you can move forward, then drop it (action 4) if needed\n"
            "- Doors must be TOGGLED (action 5) to open them\n"
            "- When you reach the goal, the episode will end automatically"
        )
    }

    messages = [system_msg]
    trajectory = []
    done = False

    for step in range(max_steps):
        # Check if already done
        if done:
            print(f"\n✅ REACHED GOAL at step {step}!")
            break

        # Get current state
        state_desc = serializer.serialize(env)
        agent_pos = env.agent_pos
        agent_dir = env.agent_dir
        carrying = env.carrying

        # Format action prompt
        action_prompt = f"""Step {step + 1}:
{state_desc}

Choose your next action:

(0) Turn left
(1) Turn right
(2) Move forward
(3) Pick up object in front of you
(4) Drop carried object
(5) Toggle door in front of you

Respond with the number of your chosen action (0-5)."""

        messages.append({"role": "user", "content": action_prompt})

        # Get LM response
        response = _generate_from_messages(lm, messages)

        # Parse action
        action = None
        text = response.strip()
        for attempt in range(3):
            match = re.search(r'\b([0-5])\b', text)
            if match:
                action = int(match.group(1))
                break
            if attempt < 2:
                # Retry
                retry_msg = "Please respond with just a number 0-5."
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": retry_msg})
                response = _generate_from_messages(lm, messages)
                text = response.strip()

        if action is None:
            print(f"\n❌ FAILED to parse action at step {step}: {text}")
            break

        # Record assistant message with chosen action
        messages.append({"role": "assistant", "content": text})

        # Get object in front (if any)
        front_cell = env.unwrapped.grid.get(*env.front_pos)
        front_obj = None
        if front_cell is not None:
            front_obj = front_cell.type

        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        new_pos = env.agent_pos
        new_carrying = env.carrying
        done = terminated or truncated

        # Log step
        action_names = ["Turn left", "Turn right", "Move forward", "Pick up", "Drop", "Toggle"]
        step_info = {
            "step": step + 1,
            "position": tuple(agent_pos),
            "direction": ["right", "down", "left", "up"][agent_dir],
            "carrying": carrying.type if carrying else None,
            "front_object": front_obj,
            "action_chosen": f"{action} ({action_names[action]})",
            "action_text": text[:100],  # First 100 chars
            "new_position": tuple(new_pos),
            "new_carrying": new_carrying.type if new_carrying else None,
            "position_changed": not (agent_pos[0] == new_pos[0] and agent_pos[1] == new_pos[1]),
            "reward": reward,
        }
        trajectory.append(step_info)

        # Print diagnostic
        pos_changed = not (agent_pos[0] == new_pos[0] and agent_pos[1] == new_pos[1])
        print(f"\nStep {step + 1}:")
        print(f"  Position: {tuple(agent_pos)} → {tuple(new_pos)} {'✓' if pos_changed else '✗ STUCK'}")
        print(f"  Carrying: {step_info['carrying']} → {step_info['new_carrying']}")
        print(f"  Front object: {front_obj}")
        print(f"  Action: {step_info['action_chosen']}")
        print(f"  LM response: {text[:80]}...")

        # Check if stuck
        if step > 5:
            recent_positions = [tuple(t["position"]) for t in trajectory[-5:]]
            if len(set(recent_positions)) == 1:
                print(f"\n⚠️  WARNING: Agent stuck at {agent_pos} for 5+ steps!")
                recent_actions = [t["action_chosen"] for t in trajectory[-5:]]
                print(f"  Recent actions: {recent_actions}")

    else:
        print(f"\n❌ Did NOT reach goal after {max_steps} steps")

    # Save trace
    trace_output = {
        "model": model_name,
        "level": level_path,
        "seed": seed,
        "max_steps": max_steps,
        "steps_taken": len(trajectory),
        "reached_goal": done,
        "trajectory": trajectory,
        "full_messages": messages,
    }

    os.makedirs("navigation_traces", exist_ok=True)
    # Include level name to avoid overwriting
    level_name = Path(level_path).stem
    trace_path = f"navigation_traces/{model_name.replace('/', '_')}_{level_name}_seed{seed}.json"
    with open(trace_path, 'w') as f:
        json.dump(trace_output, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print(f"Trace saved to: {trace_path}")
    print(f"{'='*80}\n")

    return trace_output


if __name__ == "__main__":
    import re
    import sys

    # Test both simple and complex levels
    if len(sys.argv) > 1:
        level = sys.argv[1]
    else:
        level = str(LEVELS_DIR / "test_simple_corridor.txt")

    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o"
    reasoning_effort = sys.argv[3] if len(sys.argv) > 3 else None

    diagnose_navigation(
        model_name=model,
        level_path=level,
        seed=42,
        max_steps=50,
        reasoning_effort=reasoning_effort,
    )
