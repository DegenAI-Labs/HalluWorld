from __future__ import annotations

import numpy as np
from minigrid.core.constants import IDX_TO_OBJECT, IDX_TO_COLOR, STATE_TO_IDX
from minigrid.minigrid_env import MiniGridEnv

from halluworld.serializer import Serializer
from halluworld.tracks.grid.serializers.symbolic import _visible_sign_text

_IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}

# Single-char object codes
_OBJ_CHAR = {
    "key":   "K",
    "ball":  "B",
    "box":   "X",
    "goal":  "*",
    "wall":  "#",
    "floor": ".",
    "empty": ".",
}

# Door state → single char (state_idx: 0=open, 1=closed, 2=locked)
_DOOR_STATE_CHAR = {0: "O", 1: "D", 2: "L"}

# Single-char color codes
_COLOR_CHAR = {
    "red":    "r",
    "green":  "g",
    "blue":   "b",
    "yellow": "y",
    "purple": "p",
    "grey":   "e",
}

# Agent direction index → compass label
_DIR_LABELS = {0: "right (east)", 1: "down (south)", 2: "left (west)", 3: "up (north)"}

_HEADER = (
    "You are an agent in a gridworld. Below is your current field of view (FOV) "
    "rendered as an ASCII grid.\n"
    "  - Each cell is two characters: [color][type]  (e.g. 'rK' = red key).\n"
    "  - Columns are lateral offsets: negative = left of you, 0 = directly ahead, "
    "positive = right of you.\n"
    "  - Rows are 'steps ahead': 0 = your current row, larger = further away.\n"
    "  - '@' marks your position.  '??' = unseen.  '##' = wall.  '..' = empty floor.\n"
)

_LEGEND = (
    "Legend — types: K=key  B=ball  X=box  D=door(closed)  O=door(open)  L=door(locked)  *=goal  #=wall  .=floor\n"
    "         colors: r=red  g=green  b=blue  y=yellow  p=purple  e=grey"
)


class GridSerializer(Serializer):
    """Renders the agent's FOV as a 2-D ASCII grid.

    Unlike SymbolicSerializer (bullet list), this preserves the spatial layout
    of the field of view, giving the LM explicit row/column coordinates for
    every visible cell.

    Example output (7×7 FOV):

        You are an agent in a gridworld. Below is your current field of view ...
        (header explaining axes)

                L3   L2   L1    0   R1   R2   R3
        ahead 6  ??   ??   ??   ??   ??   ??   ??
        ahead 5  ??   ??   ??   ..   ??   ??   ??
        ...
        ahead 0  ..   ..   ..   @.   ..   ##   ..

        Legend ...
    """

    def serialize(self, env: MiniGridEnv) -> str:
        obs = env.gen_obs()
        image: np.ndarray = obs["image"]   # shape (view_size, view_size, 3), axes [x, y, channel]
        view_size = image.shape[0]
        agent_col = view_size // 2

        # Build lateral header labels
        col_labels = []
        for x in range(view_size):
            lat = x - agent_col
            if lat < 0:
                col_labels.append(f"L{abs(lat)}")
            elif lat == 0:
                col_labels.append(" 0")
            else:
                col_labels.append(f"R{lat}")

        # Column width: 4 chars each
        cw = 4
        row_label_width = 8  # "ahead N " prefix

        header_row = " " * row_label_width + "".join(f"{lbl:>{cw}}" for lbl in col_labels)
        facing = _DIR_LABELS.get(int(env.agent_dir), "unknown")
        lines = [_HEADER + f"You are currently facing: {facing}.\n", header_row]

        # Rows: y=0 is furthest ahead, y=view_size-1 is agent's row
        for y in range(view_size - 1, -1, -1):   # render furthest ahead last (top of display)
            steps_ahead = (view_size - 1) - y
            row_label = f"ahead {steps_ahead}"
            row_cells = []
            for x in range(view_size):
                obj_idx, color_idx, state_idx = int(image[x, y, 0]), int(image[x, y, 1]), int(image[x, y, 2])
                obj_name = IDX_TO_OBJECT.get(obj_idx, "unknown")

                # Mark agent position
                if x == agent_col and steps_ahead == 0:
                    cell = "@."
                elif obj_name == "unseen":
                    cell = "??"
                elif obj_name in ("empty", "floor", "agent"):
                    cell = ".."
                elif obj_name == "wall":
                    cell = "##"
                else:
                    color = IDX_TO_COLOR.get(color_idx, "")
                    color_ch = _COLOR_CHAR.get(color, "-")
                    if obj_name == "door":
                        obj_ch = _DOOR_STATE_CHAR.get(state_idx, "D")
                    else:
                        obj_ch = _OBJ_CHAR.get(obj_name, "?")
                    cell = f"{color_ch}{obj_ch}"

                row_cells.append(f"{cell:>{cw}}")

            lines.append(f"{row_label:<{row_label_width}}" + "".join(row_cells))

        lines.append("")
        lines.append(_LEGEND)
        sign_lines = _visible_sign_text(env)
        if sign_lines:
            lines.append("")
            lines.extend(sign_lines)
        return "\n".join(lines)
