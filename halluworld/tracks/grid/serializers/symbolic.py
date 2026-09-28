from __future__ import annotations

import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from minigrid.minigrid_env import MiniGridEnv

from halluworld.serializer import Serializer
from halluworld.tracks.grid.envs.tiles import CombatLogObject, NoticeBoardObject, SignpostObject

# Door state index → label
_IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

# Object types excluded from visible_objects() output
_INVISIBLE = {"unseen", "empty", "floor", "agent"}


# Agent direction index → compass label
_DIR_LABELS = {0: "right (east)", 1: "down (south)", 2: "left (west)", 3: "up (north)"}


class SymbolicSerializer(Serializer):
    """Serializes the agent's partial FOV as a natural-language bullet list.

    Uses visible_objects() internally so the format stays consistent with
    what probes use for ground-truth computation.

    Args:
        hide_carrying: If True, suppress the carried item from the text output.
                       MiniGrid encodes the carried object at the agent's own
                       cell (steps_ahead=0, lateral=0), so it would otherwise
                       appear as "red key, at your current position" in every
                       observation after pickup.  Setting this to True removes
                       that entry so the only evidence of carrying is the
                       ``pickup <color> <type>`` action in the trajectory.

    Example output (agent sees a red key and locked blue door):

        You are an agent in a gridworld. Here is what you currently see:
        - red key, 2 steps forward, directly in front
        - blue door (locked), 3 steps forward, 1 step to your right
        Nothing else is visible.

    If the FOV contains no notable objects:

        You are an agent in a gridworld. You see nothing around you.
    """

    def __init__(self, hide_carrying: bool = False) -> None:
        self.hide_carrying = hide_carrying

    def serialize(self, env: MiniGridEnv) -> str:
        objs = visible_objects(env)
        if self.hide_carrying and env.carrying is not None:
            objs = [o for o in objs if not (o["steps_ahead"] == 0 and o["lateral"] == 0)]
        facing = _DIR_LABELS.get(int(env.agent_dir), "unknown")
        facing_line = f"You are currently facing: {facing}."

        if not objs:
            return f"You are an agent in a gridworld. {facing_line} You see nothing around you."

        lines = [f"You are an agent in a gridworld. {facing_line} Here is what you currently see:"]
        for o in objs:
            label = f"{o['color']} {o['object']}" if o["color"] else o["object"]
            if o["object"] == "door" and o["state"]:
                label += f" ({o['state']})"

            ahead = o["steps_ahead"]
            lat = o["lateral"]
            ahead_str = f"{ahead} step{'s' if ahead != 1 else ''} forward"
            if lat == 0:
                lat_str = "directly in front"
            elif lat < 0:
                lat_str = f"{abs(lat)} step{'s' if abs(lat) != 1 else ''} to your left"
            else:
                lat_str = f"{lat} step{'s' if lat != 1 else ''} to your right"

            if ahead == 0 and lat == 0:
                pos_str = "at your current position"
            elif ahead == 0:
                pos_str = lat_str
            else:
                pos_str = f"{ahead_str}, {lat_str}"

            lines.append(f"- {label}, {pos_str}")

        lines.append("Nothing else is visible.")
        sign_lines = _visible_sign_text(env)
        if sign_lines:
            lines.extend(sign_lines)
        return "\n".join(lines)


def _visible_sign_text(env: MiniGridEnv) -> list[str]:
    """Return text lines for any signpost/notice board currently in the agent's FOV."""
    lines = []
    grid = env.grid
    for i, obj in enumerate(grid.grid):
        if obj is None:
            continue
        if not isinstance(obj, (NoticeBoardObject, SignpostObject, CombatLogObject)):
            continue
        x = i % grid.width
        y = i // grid.width
        if env.in_view(x, y):
            if isinstance(obj, CombatLogObject):
                kind = "Combat Log"
            elif isinstance(obj, NoticeBoardObject):
                kind = "Notice board"
            else:
                kind = "Signpost"
            lines.append(f'[{kind}]: "{obj.text}"')
    return lines


def visible_objects(env: MiniGridEnv) -> list[dict]:
    """Return a list of visible objects from the agent's current FOV.

    Each entry is a dict with keys:
      - object:      str  (e.g. 'key')
      - color:       str  (e.g. 'red')
      - state:       str  (e.g. 'locked', or '' for non-doors)
      - fov_x:       int  column in FOV (0=leftmost)
      - fov_y:       int  row in FOV (0=furthest ahead)
      - steps_ahead: int  0 = same row as agent
      - lateral:     int  negative=left, 0=center, positive=right

    This is used by probes to compute ground-truth answers without needing
    to re-parse the serialized string.
    """
    obs = env.gen_obs()
    image: np.ndarray = obs["image"]
    view_size = image.shape[0]
    agent_col = view_size // 2

    objects = []
    for x in range(view_size):
        for y in range(view_size):
            obj_idx, color_idx, state = image[x, y]
            obj = IDX_TO_OBJECT.get(int(obj_idx), "unknown")
            if obj in _INVISIBLE or obj == "wall" or obj == "unknown":
                continue
            objects.append({
                "object": obj,
                "color": IDX_TO_COLOR.get(int(color_idx), ""),
                "state": _IDX_TO_STATE.get(int(state), ""),
                "fov_x": x,
                "fov_y": y,
                "steps_ahead": (view_size - 1) - y,
                "lateral": x - agent_col,
            })
    return objects
