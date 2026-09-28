"""Find valid probe windows in navigation trajectories.

A valid probe point must satisfy the "3 S's":
1. Substantiality: At least 10 steps into navigation (not trivially at start)
2. Sanity: Agent is stable (not confused, looping, or stuck)
3. Suitability: Within ±6 diagonal distance of static probe location

This ensures innav probes are asked when the agent is meaningfully
navigating but still behaving rationally.
"""

import json
import math
from typing import List, Tuple, Optional
from collections import deque


def diagonal_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def is_agent_stable(trajectory: List[dict], step_idx: int, window: int = 5) -> bool:
    """Check if agent is stable (not stuck or looping) around given step.

    Stability criteria:
    - Not stuck: position has changed in last `window` steps
    - Not looping: not repeating same position multiple times
    - Not confused: actions show some variety (not spamming same action)

    Args:
        trajectory: Full trajectory
        step_idx: Current step index to check
        window: Number of recent steps to examine (default 5)

    Returns:
        True if agent appears stable, False if stuck/confused
    """
    # Need enough history
    if step_idx < window:
        return True  # Assume stable early on

    # Get recent steps
    recent = trajectory[max(0, step_idx - window + 1):step_idx + 1]

    # Check 1: Position changes (not completely stuck)
    positions = [tuple(s['position']) for s in recent]
    unique_positions = len(set(positions))

    if unique_positions == 1:
        # Completely stuck at one position for `window` steps
        return False

    # Check 2: Not looping between 2 positions
    if unique_positions == 2 and len(positions) >= 4:
        # If alternating between just 2 positions, likely confused
        return False

    # Check 3: Action diversity (not spamming same action)
    actions = [s['action_chosen'] for s in recent]
    # Extract action number from strings like "2 (Move forward)"
    action_nums = []
    for a in actions:
        try:
            num = int(a.split()[0])
            action_nums.append(num)
        except:
            pass

    if len(action_nums) >= 4:
        # If same action repeated 80%+ of time, likely confused
        most_common = max(set(action_nums), key=action_nums.count)
        repeat_rate = action_nums.count(most_common) / len(action_nums)
        if repeat_rate > 0.8:
            return False

    # Check 4: Generally moving (not just turning)
    if len(positions) >= 3:
        # At least 30% of steps should result in position change
        position_changed = sum(1 for s in recent if s.get('position_changed', False))
        change_rate = position_changed / len(recent)
        if change_rate < 0.3:
            return False

    return True


def has_made_progress(
    trajectory: List[dict],
    step_idx: int,
    min_distance_from_start: float = 5.0
) -> bool:
    """Check if agent has moved meaningfully from start position.

    Prevents hack where agent hasn't explored but is at step 10+.

    Args:
        trajectory: Full trajectory
        step_idx: Current step
        min_distance_from_start: Minimum distance from start position

    Returns:
        True if agent has moved enough from start
    """
    if step_idx >= len(trajectory):
        return False

    start_pos = (int(trajectory[0]['position'][0]), int(trajectory[0]['position'][1]))
    current_pos = (int(trajectory[step_idx]['position'][0]), int(trajectory[step_idx]['position'][1]))

    distance = diagonal_distance(start_pos, current_pos)
    return distance >= min_distance_from_start


def find_valid_probe_windows(
    trajectory: List[dict],
    static_probe_locations: List[Tuple[int, int]],
    min_steps: int = 10,
    max_distance: float = 6.0,
    max_steps: int = 35,
    min_distance_from_start: float = 5.0,
) -> List[dict]:
    """Find all valid probe windows in a trajectory.

    Args:
        trajectory: Navigation trajectory
        static_probe_locations: Where static probes were asked
        min_steps: Minimum steps before probing (substantiality)
        max_distance: Max diagonal distance to static location (suitability)
        max_steps: Max steps to consider (avoid confused/anxious agent)
        min_distance_from_start: Minimum movement from start (prevents trivial exploration)

    Returns:
        List of valid probe windows, each with:
        - step: Step index in trajectory
        - position: Agent position at this step
        - static_location: Nearest static probe location
        - distance: Distance to that location
        - is_stable: Stability assessment
        - has_progressed: Moved far enough from start
        - reason: Why this is valid
    """
    valid_windows = []

    # Only consider steps in range [min_steps, max_steps]
    for step_idx in range(min_steps, min(len(trajectory), max_steps)):
        step = trajectory[step_idx]
        agent_pos = (int(step['position'][0]), int(step['position'][1]))

        # HARDENED CHECK 1: Stability (with longer 10-step window)
        stable = is_agent_stable(trajectory, step_idx, window=10)
        if not stable:
            continue  # Skip unstable points

        # HARDENED CHECK 2: Meaningful progress from start
        has_progressed = has_made_progress(trajectory, step_idx, min_distance_from_start)
        if not has_progressed:
            continue  # Skip if still near start position

        # HARDENED CHECK 3: Longer-term stability (check last 15 steps)
        if step_idx >= 15:
            long_term_stable = is_agent_stable(trajectory, step_idx, window=15)
            if not long_term_stable:
                continue  # Skip if recently confused

        # Find nearest static probe location
        for static_loc in static_probe_locations:
            dist = diagonal_distance(agent_pos, static_loc)

            if dist <= max_distance:
                # All hardened checks passed!
                valid_windows.append({
                    'step': step_idx,
                    'position': agent_pos,
                    'static_location': static_loc,
                    'distance': dist,
                    'is_stable': stable,
                    'has_progressed': has_progressed,
                    'reason': f"Step {step_idx}: Near {static_loc} (dist={dist:.1f}), stable, progressed"
                })

    return valid_windows


def analyze_probe_suitability(
    trace_path: str,
    static_probe_locations: List[Tuple[int, int]],
    min_steps: int = 10,
    max_steps: int = 35,
) -> dict:
    """Analyze a navigation trace for probe suitability.

    Returns summary with:
    - valid_windows: List of valid probe points
    - coverage: Which static locations we can probe
    - recommendation: Whether trajectory is suitable for innav probing
    """
    with open(trace_path) as f:
        trace = json.load(f)

    valid_windows = find_valid_probe_windows(
        trace['trajectory'],
        static_probe_locations,
        min_steps=min_steps,
        max_steps=max_steps
    )

    # Which static locations are covered?
    covered_locations = list(set(w['static_location'] for w in valid_windows))
    missed_locations = [loc for loc in static_probe_locations if loc not in covered_locations]

    # Overall suitability
    coverage_rate = len(covered_locations) / len(static_probe_locations) if static_probe_locations else 0

    suitable = coverage_rate >= 0.5  # At least 50% coverage

    return {
        'model': trace['model'],
        'level': trace['level'],
        'steps_taken': trace['steps_taken'],
        'reached_goal': trace['reached_goal'],
        'valid_windows': valid_windows,
        'num_valid_windows': len(valid_windows),
        'covered_locations': covered_locations,
        'missed_locations': missed_locations,
        'coverage_rate': coverage_rate,
        'suitable_for_innav': suitable,
        'recommendation': (
            f"✅ Suitable - {len(valid_windows)} valid probe points, {coverage_rate*100:.0f}% coverage"
            if suitable else
            f"⚠️ Limited - Only {len(valid_windows)} valid probe points, {coverage_rate*100:.0f}% coverage"
        )
    }


def print_suitability_report(analysis: dict):
    """Pretty-print probe suitability analysis."""
    print(f"\n{'='*70}")
    print(f"InNav Probe Suitability: {analysis['model']}")
    print(f"Level: {analysis['level']}")
    print(f"{'='*70}")
    print()
    print(f"Navigation: {analysis['steps_taken']} steps, "
          f"Goal: {'✅' if analysis['reached_goal'] else '❌'}")
    print()
    print(f"Valid Probe Windows (10-35 steps, stable, ±6 diagonal):")
    print(f"  Total: {analysis['num_valid_windows']}")
    print(f"  Coverage: {len(analysis['covered_locations'])}/{len(analysis['covered_locations']) + len(analysis['missed_locations'])} "
          f"static locations ({analysis['coverage_rate']*100:.0f}%)")
    print()
    print(analysis['recommendation'])
    print()

    if analysis['valid_windows']:
        print("Valid probe points:")
        for i, window in enumerate(analysis['valid_windows'][:10], 1):  # Show first 10
            print(f"  {i}. {window['reason']}")

        if len(analysis['valid_windows']) > 10:
            print(f"  ... and {len(analysis['valid_windows']) - 10} more")

    if analysis['missed_locations']:
        print()
        print("⚠️ Missed static locations (not covered in valid windows):")
        for loc in analysis['missed_locations']:
            print(f"  {loc}")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Example: Check GPT-5.4 corridor gauntlet
    static_locations = [
        (5, 2),   # Near first key
        (10, 2),  # Midpoint
        (15, 2),  # Near goal
    ]

    trace_file = 'navigation_traces/gpt-5.4-mini_seed42.json'

    analysis = analyze_probe_suitability(
        trace_file,
        static_locations,
        min_steps=10,
        max_steps=35
    )

    print_suitability_report(analysis)
