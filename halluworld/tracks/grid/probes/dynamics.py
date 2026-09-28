from __future__ import annotations

import random
from typing import Optional

from minigrid.minigrid_env import MiniGridEnv

from halluworld.probe import Probe, ProbeResult
from halluworld.tracks.grid.envs.ascii_env import AsciiEnv

# MiniGrid direction index → (dx, dy) forward step in grid coords
_DIR_DELTA: dict[int, tuple[int, int]] = {
    0: (1, 0),   # right / east
    1: (0, 1),   # down  / south
    2: (-1, 0),  # left  / west
    3: (0, -1),  # up    / north
}

_DIR_LABELS = {0: "right (east)", 1: "down (south)", 2: "left (west)", 3: "up (north)"}

# Action names probed
_PROBED_ACTIONS = ("move_forward", "turn_left_then_move", "turn_right_then_move")


class DynamicsProbe(Probe):
    """Tests whether the LM correctly predicts the agent's landing cell after an action.

    Core idea — Windy Gridworld (S&B Ch.6.5)
    -----------------------------------------
    When the agent moves into a *wind column*, an additional environmental
    displacement is applied after the step.  The LM must reason about BOTH
    the agent's volitional move AND the wind offset to predict the correct
    landing cell.

    Without wind, "move forward" is trivially predictable.  With wind, the
    model must track the column-specific offset — this exposes whether it has
    internalized the dynamics description given in the system prompt.

    Probe question format
    ---------------------
    "You are at grid position (row=R, col=C) facing <direction>.
     If you move forward now, what grid position will you land on?
     Answer in the format: row=<int>, col=<int>"

    Ground truth is computed by simulating the move + wind offset without
    mutating the env.

    The probe optionally asks about TURN + MOVE sequences to also probe
    direction-tracking (``include_turn_variants=True``).

    Wind hint modes
    ---------------
    ``wind_hint_mode`` controls how much wind information the probe leaks:

    * ``"full"``  — explicit column-by-column spec (easy mode; useful as an
      ablation baseline but defeats the purpose of the benchmark).
    * ``"hint"``  — tells the model wind exists but withholds columns and
      magnitude; the model must rely on the trajectory to learn the effect.
      This is the recommended default.
    * ``"none"``  — no wind mention at all; the model must fully induce the
      dynamics from a position-annotated trajectory.

    Args:
        include_turn_variants: Also sometimes ask about the result of
                               turn_left + move_forward, or turn_right +
                               move_forward (default False).
        wind_hint_mode:        How much wind information to include in the
                               question (default ``"hint"``).
        rng:                   Optional seeded Random instance.
    """

    def __init__(
        self,
        include_turn_variants: bool = False,
        wind_hint_mode: str = "hint",
        rng: Optional[random.Random] = None,
    ) -> None:
        assert wind_hint_mode in ("full", "hint", "none"), (
            f"wind_hint_mode must be 'full', 'hint', or 'none', got {wind_hint_mode!r}"
        )
        self.include_turn_variants = include_turn_variants
        self.wind_hint_mode = wind_hint_mode
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult | None:
        ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
        direction = int(env.agent_dir)

        # Choose which action to ask about
        if self.include_turn_variants:
            action_choice = self.rng.choice(["move_forward", "turn_left_then_move", "turn_right_then_move"])
        else:
            action_choice = "move_forward"

        # Derive the facing direction at the moment of the forward step
        if action_choice == "move_forward":
            effective_dir = direction
        elif action_choice == "turn_left_then_move":
            effective_dir = (direction - 1) % 4
        else:  # turn_right_then_move
            effective_dir = (direction + 1) % 4

        dx, dy = _DIR_DELTA[effective_dir]
        nx, ny = ax + dx, ay + dy

        # Check bounds and wall — if blocked the agent stays put
        can_move = (
            0 <= nx < env.width
            and 0 <= ny < env.height
            and _cell_passable(env, nx, ny)
        )

        if can_move:
            land_col, land_row = nx, ny
        else:
            land_col, land_row = ax, ay  # bumped into wall / boundary

        naive_col, naive_row = land_col, land_row  # pre-wind position

        # Apply wind offset if this env has wind.
        # Wind is applied to the landing cell (which equals the start cell if
        # the move was blocked), matching AsciiEnv.step() behaviour.
        wind_offset = 0
        has_wind = False
        if isinstance(env, AsciiEnv) and env._wind:
            has_wind = True
            offset = env._wind.get(land_col, 0)
            if offset != 0:
                wy = land_row + offset
                if 0 <= wy < env.height and _cell_passable(env, land_col, wy):
                    wind_offset = offset
                    land_row = wy

        ground_truth = {"row": land_row, "col": land_col}

        # Build question text
        facing_label = _DIR_LABELS[direction]
        action_desc = _action_description(action_choice, effective_dir)

        if self.wind_hint_mode == "full" and has_wind and isinstance(env, AsciiEnv) and env._wind:
            wind_parts = ", ".join(
                f"col {col}: {offset:+d} rows" for col, offset in sorted(env._wind.items())
            )
            wind_hint = (
                f" Note: this gridworld has wind. After each move the wind "
                f"may shift your row position. Wind specification: {wind_parts}. "
                f"Positive offset = south (row increases), negative = north (row decreases)."
            )
        elif self.wind_hint_mode == "hint" and has_wind:
            wind_hint = (
                " Note: this gridworld has wind that may shift your row "
                "position after some moves."
            )
        else:
            wind_hint = ""
        question = (
            f"You are currently at grid position (row={ay}, col={ax}) "
            f"facing {facing_label}.{wind_hint}\n"
            f"{action_desc}\n"
            "Answer with exactly: row=<integer>, col=<integer>"
        )

        return ProbeResult(
            probe_type="dynamics",
            question=question,
            ground_truth=ground_truth,
            metadata={
                "start_row": ay,
                "start_col": ax,
                "direction": direction,
                "action": action_choice,
                "effective_dir": effective_dir,
                "can_move": can_move,
                "has_wind": has_wind,
                "wind_offset": wind_offset,
                "naive_row": naive_row,
                "naive_col": naive_col,
                "wind_hint_mode": self.wind_hint_mode,
            },
        )


# ── helpers ────────────────────────────────────────────────────────────────────

def _cell_passable(env: MiniGridEnv, x: int, y: int) -> bool:
    """Return True if cell (x, y) can be walked into."""
    cell = env.grid.get(x, y)
    return cell is None or cell.can_overlap()


def _action_description(action: str, effective_dir: int) -> str:
    dir_label = _DIR_LABELS[effective_dir]
    if action == "move_forward":
        return f"If you move forward now (facing {dir_label}), what grid position will you land on?"
    elif action == "turn_left_then_move":
        return (
            f"If you turn left (now facing {dir_label}) and then move forward, "
            "what grid position will you land on?"
        )
    else:  # turn_right_then_move
        return (
            f"If you turn right (now facing {dir_label}) and then move forward, "
            "what grid position will you land on?"
        )
