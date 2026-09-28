"""multiturn.py — Multi-turn trajectory evaluation for HalluWorld.

Two evaluation modes:

Oracle mode
-----------
A BFS shortest-path policy steers the agent toward the goal.  After the full
trajectory the probe is generated against the **current** (final) FOV.  The LM
receives the entire step-by-step history as context and must reason about what
is still visible *now* versus what it saw earlier.

Egocentric mode
---------------
The LM itself decides each action (``turn_left`` / ``turn_right`` /
``move_forward``), more closely mirroring real agent use.

Key probe: PersistenceProbe
---------------------------
Ask whether an object that appeared earlier in the trajectory is *currently*
visible.  Because the agent has since moved, the answer is often 'no' — but a
model biased by earlier observations will hallucinate 'yes'.

Typical usage::

    from halluworld.tracks.grid import make_env_from_ascii
    from halluworld.tracks.grid import SymbolicSerializer
    from halluworld.tracks.grid import PersistenceProbe
    from halluworld.lm import OpenAILM
    from halluworld.tracks.grid import PresenceEvaluator
    from halluworld.multiturn import run_multiturn_benchmark

    env = make_env_from_ascii(str(LEVELS_DIR / "open_room_12.txt"), render_mode="rgb_array")
    run = run_multiturn_benchmark(
        env=env,
        serializer=SymbolicSerializer(),
        probes=[PersistenceProbe(positive_rate=0.3)],
        lm=OpenAILM(model="gpt-4o-mini"),
        evaluators=[PresenceEvaluator()],
        n_episodes=20,
        trajectory_length=12,
        warmup_steps=5,
        policy="oracle",
        seed=42,
        verbose=True,
    )
    print(run.summary())
"""
from __future__ import annotations

import random
from collections import deque

from tqdm import tqdm
from dataclasses import dataclass, field

from minigrid.core.world_object import Ball, Box, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

from halluworld.benchmark import BenchmarkRun, EpisodeResult
from halluworld.evaluator import Evaluator
from halluworld.lm.base import LM
from halluworld.probe import Probe
from halluworld.serializer import Serializer
from halluworld.data import LEVELS_DIR

# ------------------------------------------------------------------ #
# Trajectory data structures                                           #
# ------------------------------------------------------------------ #

@dataclass
class TrajectoryStep:
    """One step in an agent trajectory."""
    step_num: int       # 1-indexed for human-readable prompts
    obs_text: str       # serialized observation
    action: str | None  # action taken *after* this obs; None = final step


# ------------------------------------------------------------------ #
# Oracle policy: BFS shortest path to goal                            #
# ------------------------------------------------------------------ #

# MiniGrid direction index → (dx, dy) in grid coordinates
_DIR_DELTA: dict[int, tuple[int, int]] = {
    0: (1, 0),   # right / east
    1: (0, 1),   # down  / south
    2: (-1, 0),  # left  / west
    3: (0, -1),  # up    / north
}

# MiniGrid action indices
_ACT_TURN_LEFT  = 0
_ACT_TURN_RIGHT = 1
_ACT_MOVE_FWD   = 2
_ACT_PICKUP     = 3
_ACT_DROP       = 4

_ACT_NAMES = {
    _ACT_TURN_LEFT:  "turn_left",
    _ACT_TURN_RIGHT: "turn_right",
    _ACT_MOVE_FWD:   "move_forward",
    _ACT_PICKUP:     "pickup",
    _ACT_DROP:       "drop",
}


def _find_goal(env: MiniGridEnv) -> tuple[int, int] | None:
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if isinstance(cell, Goal):
                return (x, y)
    return None


def _bfs(env: MiniGridEnv, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """BFS on the static grid; returns position path from start (exclusive) to goal."""
    queue: deque[tuple[tuple[int, int], list]] = deque([(start, [])])
    visited = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == goal:
            return path
        x, y = pos
        for dx, dy in _DIR_DELTA.values():
            nx, ny = x + dx, y + dy
            npos = (nx, ny)
            if npos in visited:
                continue
            if not (0 <= nx < env.width and 0 <= ny < env.height):
                continue
            cell = env.grid.get(nx, ny)
            if cell is None or cell.can_overlap() or npos == goal:
                visited.add(npos)
                queue.append((npos, path + [npos]))
    return None


def _path_to_actions(
    start_pos: tuple[int, int],
    start_dir: int,
    path: list[tuple[int, int]],
) -> list[tuple[int, str]]:
    """Convert a position path to a sequence of (action_idx, action_name)."""
    actions: list[tuple[int, str]] = []
    pos = list(start_pos)
    facing = start_dir
    delta_to_dir = {v: k for k, v in _DIR_DELTA.items()}

    for next_pos in path:
        dx, dy = next_pos[0] - pos[0], next_pos[1] - pos[1]
        required_dir = delta_to_dir[(dx, dy)]

        # Rotate to face required direction (choose shortest turn)
        while facing != required_dir:
            if (required_dir - facing) % 4 <= (facing - required_dir) % 4:
                actions.append((_ACT_TURN_RIGHT, _ACT_NAMES[_ACT_TURN_RIGHT]))
                facing = (facing + 1) % 4
            else:
                actions.append((_ACT_TURN_LEFT, _ACT_NAMES[_ACT_TURN_LEFT]))
                facing = (facing - 1) % 4

        actions.append((_ACT_MOVE_FWD, _ACT_NAMES[_ACT_MOVE_FWD]))
        pos = list(next_pos)

    return actions


def _oracle_next_action(env: MiniGridEnv) -> tuple[int, str] | None:
    """Return the first BFS-optimal action toward the goal, or None if unreachable."""
    goal_pos = _find_goal(env)
    if goal_pos is None:
        return None
    path = _bfs(env, tuple(env.agent_pos), goal_pos)
    if not path:
        return None
    actions = _path_to_actions(tuple(env.agent_pos), env.agent_dir, path)
    return actions[0] if actions else None


def _find_nearest_pickupable(env: MiniGridEnv) -> tuple[int, int] | None:
    """Return the grid position of the nearest Key, Ball, or Box, or None."""
    ax, ay = env.agent_pos
    best: tuple[int, int] | None = None
    best_dist = float("inf")
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if isinstance(cell, (Key, Ball, Box)):
                d = abs(x - ax) + abs(y - ay)
                if d < best_dist:
                    best_dist, best = d, (x, y)
    return best


def _pickup_oracle_next_action(env: MiniGridEnv) -> tuple[int, str] | None:
    """Navigate to the nearest pickupable object, pick it up, then go to goal.

    State machine:
      1. If already carrying something  →  route to goal (standard oracle).
      2. If adjacent and facing the target  →  execute pickup.
      3. If adjacent but not facing  →  turn toward the target.
      4. Otherwise  →  BFS toward the target cell (treating it as a goal so the
         BFS can enter it), then follow the first action of the trimmed path.
    """
    if env.carrying is not None:
        return _oracle_next_action(env)

    target = _find_nearest_pickupable(env)
    if target is None:
        return _oracle_next_action(env)

    ax, ay = env.agent_pos
    tx, ty = target
    dx, dy = tx - ax, ty - ay
    delta_to_dir = {v: k for k, v in _DIR_DELTA.items()}

    # Already adjacent to the object
    if abs(dx) + abs(dy) == 1 and (dx, dy) in delta_to_dir:
        required_dir = delta_to_dir[(dx, dy)]
        if env.agent_dir == required_dir:
            obj = env.grid.get(tx, ty)
            pickup_label = (
                f"pickup {obj.color} {type(obj).__name__.lower()}"
                if obj is not None else "pickup"
            )
            return (_ACT_PICKUP, pickup_label)
        # Turn toward the object (shortest rotation)
        diff = (required_dir - env.agent_dir) % 4
        if diff <= 2:
            return (_ACT_TURN_RIGHT, _ACT_NAMES[_ACT_TURN_RIGHT])
        else:
            return (_ACT_TURN_LEFT, _ACT_NAMES[_ACT_TURN_LEFT])

    # BFS toward target; the target cell is passed as 'goal' so the BFS enters it
    path = _bfs(env, (ax, ay), target)
    if path is None:
        return _oracle_next_action(env)

    # Drop the last element (the target cell itself) — we stop 1 step before
    if path and path[-1] == target:
        path = path[:-1]
    if not path:
        # Degenerate: already adjacent (handled above); shouldn't reach here
        return _oracle_next_action(env)

    actions = _path_to_actions((ax, ay), env.agent_dir, path)
    return actions[0] if actions else None


# ------------------------------------------------------------------ #
# Egocentric policy: LM decides each action                           #
# ------------------------------------------------------------------ #

_EGOCENTRIC_SYSTEM = (
    "You are an agent navigating a gridworld to reach a goal (*). "
    "At each step you receive a description of your current field of view. "
    "Choose one action to take. "
    "Respond with EXACTLY one of the following words (nothing else): "
    "turn_left, turn_right, move_forward"
)

# Normalise various LM phrasings → action index
_EGOCENTRIC_ACTION_MAP: dict[str, int] = {
    "turn_left":     _ACT_TURN_LEFT,
    "turn left":     _ACT_TURN_LEFT,
    "left":          _ACT_TURN_LEFT,
    "turn_right":    _ACT_TURN_RIGHT,
    "turn right":    _ACT_TURN_RIGHT,
    "right":         _ACT_TURN_RIGHT,
    "move_forward":  _ACT_MOVE_FWD,
    "move forward":  _ACT_MOVE_FWD,
    "forward":       _ACT_MOVE_FWD,
    "move":          _ACT_MOVE_FWD,
}


def _egocentric_action(
    lm: LM,
    obs_text: str,
    history: list[TrajectoryStep],
) -> tuple[int, str]:
    """Ask the LM for an action given the current obs and trajectory history."""
    parts: list[str] = []
    for step in history:
        parts.append(f"## Step {step.step_num}\n{step.obs_text}\nAction taken: {step.action}")
    parts.append(f"## Step {len(history) + 1} (current)\n{obs_text}")
    user_msg = "\n\n".join(parts) + "\n\nWhat action do you take?"

    response = lm.query(system=_EGOCENTRIC_SYSTEM, user=user_msg)
    raw = response.text.strip().lower()
    action_idx = _EGOCENTRIC_ACTION_MAP.get(raw, _ACT_MOVE_FWD)
    return action_idx, _ACT_NAMES[action_idx]


# ------------------------------------------------------------------ #
# Pickup-wander oracle: pick up objects with random-walk interludes   #
# ------------------------------------------------------------------ #

class _PickupWanderOracle:
    """Stateful per-episode policy: pick up objects with random-walk interludes.

    Phases cycle as: seek → pickup → wander → drop → seek → pickup → wander …
    After ``max_pickups`` pickups the agent wanders indefinitely.

    This ensures the trajectory contains multiple ``pickup <color> <type>``
    and ``drop <color> <type>`` events separated by wandering, making it
    non-trivial for an LM to track which object (if any) is currently held.
    """

    def __init__(self, rng: random.Random, wander_steps: int = 20, max_pickups: int = 2) -> None:
        self._rng = rng
        self._wander_steps = wander_steps
        self._max_pickups = max_pickups
        self._pickups_done = 0
        self._wander_remaining = 0

    def next_action(self, env: MiniGridEnv) -> tuple[int, str]:
        # Wander phase
        if self._wander_remaining > 0:
            self._wander_remaining -= 1
            return _random_walk_action(self._rng)

        # Seek phase — carrying something: drop it before picking up the next
        if env.carrying is not None:
            if self._pickups_done >= self._max_pickups:
                return _random_walk_action(self._rng)
            drop = self._drop_action(env)
            if drop[0] == _ACT_DROP:
                # Drop will succeed — start a wander so we move away from the
                # dropped item before seeking the next one
                self._wander_remaining = self._wander_steps
            return drop

        # Seek phase — not carrying: navigate to / pick up next object
        if self._pickups_done >= self._max_pickups:
            return _random_walk_action(self._rng)
        act = _pickup_oracle_next_action(env)
        if act is None:
            return _random_walk_action(self._rng)
        if act[0] == _ACT_PICKUP:
            self._pickups_done += 1
            self._wander_remaining = self._wander_steps
        return act

    def _drop_action(self, env: MiniGridEnv) -> tuple[int, str]:
        """Drop if the cell directly ahead is clear, else turn right."""
        ax, ay = env.agent_pos
        dx, dy = _DIR_DELTA[env.agent_dir]
        fx, fy = ax + dx, ay + dy
        if (0 <= fx < env.width and 0 <= fy < env.height
                and env.grid.get(fx, fy) is None):
            label = f"drop {env.carrying.color} {type(env.carrying).__name__.lower()}"
            return (_ACT_DROP, label)
        return (_ACT_TURN_RIGHT, _ACT_NAMES[_ACT_TURN_RIGHT])


# ------------------------------------------------------------------ #
# Random walk policy: movement-biased exploration                     #
# ------------------------------------------------------------------ #

# 60% move_forward, 20% turn_left, 20% turn_right
_RANDOM_WALK_WEIGHTS = [0.20, 0.20, 0.60]   # left, right, forward


def _random_walk_action(rng: random.Random) -> tuple[int, str]:
    """Biased random action: forward 60%, turn_left/right 20% each."""
    r = rng.random()
    if r < 0.20:
        return _ACT_TURN_LEFT, _ACT_NAMES[_ACT_TURN_LEFT]
    elif r < 0.40:
        return _ACT_TURN_RIGHT, _ACT_NAMES[_ACT_TURN_RIGHT]
    else:
        return _ACT_MOVE_FWD, _ACT_NAMES[_ACT_MOVE_FWD]


# ------------------------------------------------------------------ #
# Multi-turn benchmark runner                                         #
# ------------------------------------------------------------------ #

_MULTITURN_SYSTEM = (
    "You are an agent navigating a gridworld. "
    "At each step you receive an egocentric observation: objects are described "
    "by how many steps forward and how many steps left/right they are relative "
    "to you. Your field of view (FOV) is a square centred ahead of you; the "
    "depth and width are both equal to (agent_view_size - 1) steps. "
    "Possible actions include: turn_left, turn_right, move_forward, pickup, and drop. "
    "When you execute 'pickup <color> <type>', that object is removed from the grid "
    "and you begin carrying it — it no longer appears in your field of view. "
    "When you execute 'drop <color> <type>', the carried object is placed on the "
    "cell directly in front of you and you are no longer carrying anything. "
    "You can carry at most one object at a time. "
    "You will be shown a series of observations from consecutive steps. "
    "Answer questions about your current state. "
    "Base your answers only on what you can observe or reason from your history."
)


def run_multiturn_benchmark(
    env: MiniGridEnv,
    serializer: Serializer,
    probes: list[Probe],
    lm: LM,
    evaluators: list[Evaluator],
    n_episodes: int = 50,
    trajectory_length: int = 8,
    warmup_steps: int = 0,
    policy: str = "oracle",
    hide_final_obs: bool = False,
    seed: int = 42,
    verbose: bool = False,
    system_prompt: str | None = None,
    wander_steps: int = 20,
    pickup_max: int = 2,
) -> BenchmarkRun:
    """Run a multi-turn trajectory eval; probes are grounded in the **final** FOV.

    All probes are generated at the *end* of the trajectory against the agent's
    current field of view.  The full step-by-step history is given to the LM as
    context so it must reason about what is still visible now vs. what it saw
    earlier — not merely retrieve the answer from a named step.

    For ``PersistenceProbe`` specifically this creates the key hallucination
    test: the probe preferentially asks about objects that appeared in earlier
    steps but are no longer visible, so a model that over-relies on earlier
    observations will hallucinate.

    For ``PresenceProbe`` / ``LocationProbe`` the questions are about the
    current FOV with the trajectory as (potentially distracting) context.

    Args:
        env:               MiniGrid environment.
        serializer:        Serializes env → string.
        probes:            Probe instances.  ``PersistenceProbe`` instances are
                           updated with trajectory context automatically.
        lm:                Language model.
        evaluators:        Evaluators paired with probes by index.
        n_episodes:        Number of independent env resets.
        trajectory_length: Max steps to take per episode.
        warmup_steps:      Random actions before the main trajectory to vary the
                           agent's starting position (useful for fixed-layout
                           AsciiEnv levels).
        policy:            "oracle" (BFS toward goal) or "egocentric" (LM picks
                           actions).
        hide_final_obs:    If True, the final step's observation is withheld from
                           the LM context.  The model must infer the current state
                           from its movement history alone — a true spatial memory
                           test.  Ground truth is still derived from the real final
                           FOV.  Works best with ``PersistenceProbe``.
        seed:              Master RNG seed.
        verbose:           Print trial results.

    Returns:
        BenchmarkRun with all EpisodeResults.
    """
    assert len(probes) == len(evaluators), (
        f"probes and evaluators length mismatch: {len(probes)} vs {len(evaluators)}"
    )
    assert policy in ("oracle", "egocentric", "pickup_oracle", "pickup_wander", "random"), f"Unknown policy: {policy!r}"

    rng = random.Random(seed)
    episode_id = 0
    run = BenchmarkRun(run_metadata={
        "n_episodes": n_episodes,
        "trajectory_length": trajectory_length,
        "warmup_steps": warmup_steps,
        "policy": policy,
        "hide_final_obs": hide_final_obs,
        "seed": seed,
        "agent_view_size": getattr(env, "agent_view_size", None),
    })
    policy_rng = random.Random(seed + 1)  # separate rng for random-walk policy

    for env_reset_id in tqdm(range(n_episodes), desc="episodes", unit="ep"):
        env_seed = rng.randint(0, 2**31)
        env.reset(seed=env_seed)

        # Instantiate a fresh stateful oracle for pickup_wander policy
        wander_oracle = (
            _PickupWanderOracle(policy_rng, wander_steps=wander_steps, max_pickups=pickup_max)
            if policy == "pickup_wander" else None
        )

        # Reset trajectory-context for probes that track seen objects
        for probe in probes:
            if hasattr(probe, "reset_trajectory"):
                probe.reset_trajectory()

        # Random warm-up: vary agent position before the timed trajectory.
        # Uses the master rng so results are reproducible.
        for _ in range(warmup_steps):
            idx = rng.randint(0, 2)  # only turn_left/turn_right/move_forward
            _, _, term, trunc, _ = env.step(idx)
            if term or trunc:
                env.reset(seed=rng.randint(0, 2**31))

        trajectory: list[TrajectoryStep] = []
        terminated = truncated = False
        step = 0

        while step < trajectory_length and not (terminated or truncated):
            obs_text = serializer.serialize(env)

            # Feed current FOV into probes that track trajectory context
            for probe in probes:
                if hasattr(probe, "update_trajectory_seen"):
                    probe.update_trajectory_seen(env)

            # Get next action
            if policy in ("oracle", "pickup_oracle"):
                if policy == "pickup_oracle":
                    act = _pickup_oracle_next_action(env)
                else:
                    act = _oracle_next_action(env)
                if act is None:
                    # No path to goal — fall back to random
                    idx = env.action_space.sample()
                    act = (idx, _ACT_NAMES.get(idx, "unknown"))
                action_idx, action_name = act
            elif policy == "pickup_wander":
                action_idx, action_name = wander_oracle.next_action(env)
            elif policy == "random":
                action_idx, action_name = _random_walk_action(policy_rng)
            else:  # egocentric
                action_idx, action_name = _egocentric_action(lm, obs_text, trajectory)

            trajectory.append(TrajectoryStep(step + 1, obs_text, action_name))
            _, _, terminated, truncated, _ = env.step(action_idx)
            step += 1

        # Final observation (after last action); update trajectory context but
        # optionally withhold from LM to force memory-based inference.
        final_obs = serializer.serialize(env)
        for probe in probes:
            if hasattr(probe, "update_trajectory_seen"):
                probe.update_trajectory_seen(env)

        # The action that produced the final state (last entry in trajectory
        # before we append the final step).
        final_action = trajectory[-1].action if trajectory else None

        if hide_final_obs:
            final_step_num = len(trajectory) + 1
            view_size = getattr(env, "agent_view_size", None)
            fov_depth = (view_size - 1) if view_size else "unknown"
            action_reminder = (
                f" Your last action was: {final_action}." if final_action else ""
            )
            placeholder = (
                f"[observation withheld — reason from your history to infer "
                f"what is now in your FOV (depth {fov_depth} steps, "
                f"width {fov_depth} steps either side).{action_reminder}]"
            )
            trajectory.append(TrajectoryStep(final_step_num, placeholder, None))
        else:
            trajectory.append(TrajectoryStep(len(trajectory) + 1, final_obs, None))

        # Build trajectory context string for the LM
        parts: list[str] = []
        for ts in trajectory:
            header = f"## Step {ts.step_num}"
            body = ts.obs_text
            if ts.action:
                body += f"\n→ Action taken after this observation: {ts.action}"
            parts.append(f"{header}\n{body}")
        history_text = "\n\n".join(parts)

        view_size = getattr(env, "agent_view_size", None)
        fov_note = f" (FOV: {view_size - 1} steps forward, {view_size - 1} steps left/right)" if view_size else ""

        # Generate probes against the FINAL step's FOV
        for i, (probe, evaluator) in enumerate(zip(probes, evaluators)):
            probe_result = probe.generate(env)
            if probe_result is None or probe_result.metadata.get("skipped"):
                episode_id += 1
                continue

            user_prompt = (
                f"Here is your trajectory across {len(trajectory)} steps{fov_note}:\n\n"
                f"{history_text}\n\n"
                f"## Question (about your current observation at Step {len(trajectory)})\n"
                f"{probe_result.question}"
            )

            response = lm.query(system=system_prompt or _MULTITURN_SYSTEM, user=user_prompt)
            eval_result = evaluator.evaluate(response, probe_result)

            result = EpisodeResult(
                episode_id=episode_id,
                probe_name=probe_result.probe_type,
                question=probe_result.question,
                ground_truth=probe_result.ground_truth,
                lm_response=response.text,
                is_correct=eval_result.correct,
                score=eval_result.score,
                metadata={
                    "env_reset_id": env_reset_id,
                    "env_seed": env_seed,
                    "agent_view_size": getattr(env, "agent_view_size", None),
                    "policy": policy,
                    "hide_final_obs": hide_final_obs,
                    "trajectory_length": len(trajectory),
                    "warmup_steps": warmup_steps,
                    **probe_result.metadata,
                    **{k: v for k, v in eval_result.details.items()
                       if k not in probe_result.metadata},
                },
                messages=[
                    {"role": "system", "content": system_prompt or _MULTITURN_SYSTEM},
                    {"role": "user",   "content": user_prompt},
                    {"role": "assistant", "content": response.text},
                ],
            )
            run.results.append(result)

            if verbose:
                tag = "✓" if eval_result.correct else "✗"
                src = probe_result.metadata.get("source", "")
                print(
                    f"[ep {episode_id:4d}][{probe_result.probe_type:11s}] {tag} "
                    f"gt={str(probe_result.ground_truth):<6} "
                    f"pred={str(eval_result.predicted):<6} "
                    f"lm={response.text[:40]!r} "
                    f"src={src}"
                )

            episode_id += 1

    return run
