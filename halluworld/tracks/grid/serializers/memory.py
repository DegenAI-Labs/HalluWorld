"""MemorySerializer — full-room serializer for multi-step memory levels.

Unlike SymbolicSerializer (which only shows the agent's FOV cone),
MemorySerializer iterates the entire grid and reports every object with its
absolute compass position relative to the agent.

Intended use in a multi-step runner::

    ser = MemorySerializer()
    history = []
    env.reset(seed=seed)
    history.append(ser.serialize(env, step=0, label="Step 0"))
    for t in range(1, N + 1):
        env.step(forward)
        history.append(ser.serialize(env, step=t))
    combined = "\\n\\n".join(history)

Output example::

    [Step 3]
    You are at row 13, col 7, facing north.

    - yellow key: 12 steps north, 3 steps west
    - red ball: 8 steps north, 3 steps west (in river, flowing east)
    - blue ball: 8 steps north (in river, flowing east) [WET, dries in 4 turns]
    - red key: 6 steps north, 1 step east (in river, flowing east) [WET, dries in 2 turns]
    - green ball: 5 steps north, 4 steps west
    - yellow door (closed): 2 steps south, 1 step east
    - [Notice board]: "blue ball: col 3, WET(4) | red key: col 7, WET(2)"
    Nothing else visible.
"""
from __future__ import annotations

from typing import Any

from minigrid.core.world_object import Door, Wall

from halluworld.serializer import Serializer
from halluworld.tracks.grid.envs.tiles import (
    Boulder,
    DarkZone,
    ElevatedTile,
    FireSource,
    FireTile,
    FloodTile,
    MudTile,
    CombatLogObject,
    NoticeBoardObject,
    PressurePlate,
    RiverTile,
    SignpostObject,
    Torch,
    TreeTile,
    WaterTile,
    condition_label,
    DIR_LABEL,
)

_DIR_COMPASS = {0: "east", 1: "south", 2: "west", 3: "north"}
_DIR_FACING = {0: "east (right)", 1: "south (down)", 2: "west (left)", 3: "north (up)"}

# Object types the serializer treats as invisible terrain / meta
_SKIP_TYPES = {"wall", "floor", "unseen", "empty", "agent"}


def _compass_offset(ax: int, ay: int, ox: int, oy: int) -> str:
    """Describe (ox,oy) relative to (ax,ay) in compass terms.

    MiniGrid coordinate axes: x = col (east+), y = row (south+).
    So north = decreasing y, east = increasing x.
    """
    dy = ay - oy  # positive = object is north of agent
    dx = ox - ax  # positive = object is east of agent

    parts: list[str] = []
    if dy > 0:
        parts.append(f"{dy} step{'s' if dy != 1 else ''} north")
    elif dy < 0:
        ady = abs(dy)
        parts.append(f"{ady} step{'s' if ady != 1 else ''} south")

    if dx > 0:
        parts.append(f"{dx} step{'s' if dx != 1 else ''} east")
    elif dx < 0:
        adx = abs(dx)
        parts.append(f"{adx} step{'s' if adx != 1 else ''} west")

    if not parts:
        return "at your position"
    return ", ".join(parts)


def _manhattan(ax: int, ay: int, ox: int, oy: int) -> int:
    return abs(ox - ax) + abs(oy - ay)


class MemorySerializer(Serializer):
    """Full-grid serializer for multi-step memory levels.

    Args:
        sort_by_distance:  List objects nearest-first (default True).
        show_terrain:      Include river / fire / dark-zone terrain notes
                           at the top of the output (default True).
    """

    def __init__(
        self,
        sort_by_distance: bool = True,
        show_terrain: bool = True,
    ) -> None:
        self.sort_by_distance = sort_by_distance
        self.show_terrain = show_terrain

    def serialize(
        self,
        env: Any,
        *,
        step: int | None = None,
        label: str | None = None,
    ) -> str:
        """Serialize the full env grid to a natural-language string.

        Args:
            env:   AsciiEnv (or any MiniGridEnv) instance after reset.
            step:  Current step number (prepended as ``[Step N]`` if given).
            label: Override the step label entirely (e.g. ``"Observation 2"``).
        """
        ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
        facing = _DIR_FACING.get(int(env.agent_dir), "unknown")

        lines: list[str] = []

        # Header
        header = f"You are at row {ay}, col {ax}, facing {facing}."
        if label is not None:
            lines.append(f"[{label}]")
        elif step is not None:
            lines.append(f"[Step {step}]")
        lines.append(header)

        river_map: dict = getattr(env, "_river_map", {})

        # ── Collect terrain features ──────────────────────────────────────────
        if self.show_terrain:
            terrain_notes: list[str] = []

            # Summarise river zones
            river_dirs: dict[int, list[tuple[int, int]]] = {}
            for (rx, ry), rt in river_map.items():
                river_dirs.setdefault(rt.direction, []).append((rx, ry))
            for rdir, cells in sorted(river_dirs.items()):
                rows_set = sorted({ry for _, ry in cells})
                cols_set = sorted({rx for rx, _ in cells})
                if len(rows_set) <= 3:
                    rows_str = "/".join(str(r) for r in rows_set)
                else:
                    rows_str = f"{rows_set[0]}–{rows_set[-1]}"
                cols_str = f"{cols_set[0]}–{cols_set[-1]}"
                terrain_notes.append(
                    f"River (rows {rows_str}, cols {cols_str}): flowing {DIR_LABEL[rdir]}"
                )

            # Fire tiles
            fire_cells = []
            for y in range(env.height):
                for x in range(env.width):
                    obj = env.grid.get(x, y)
                    if isinstance(obj, FireTile) and obj.active:
                        fire_cells.append((x, y))
            if fire_cells:
                unique_fire_rows = sorted({y for _, y in fire_cells})
                rows_str = "/".join(f"row {y}" for y in unique_fire_rows[:3])
                terrain_notes.append(f"Active fire: {rows_str} (burning — impassable)")

            if terrain_notes:
                lines.append("")
                lines.append("[Terrain]:")
                for note in terrain_notes:
                    lines.append(f"  {note}")

        # ── Flood progress note ───────────────────────────────────────────────
        flood_tiles: list = getattr(env, "_flood_tiles", [])
        if flood_tiles:
            flooded_rows = sorted(
                {fy for (_, fy), tile in flood_tiles if tile.flooded},
                reverse=True,
            )
            dry_flood_rows = sorted(
                {fy for (_, fy), tile in flood_tiles if not tile.flooded}
            )
            if flooded_rows:
                lines.append("")
                deepest = flooded_rows[0]  # highest row index = deepest flood
                lines.append(
                    f"[Flood]: Water has reached row {deepest}. "
                    f"Rows {deepest} and above (lower row numbers) "
                    "are flooded and impassable."
                )
            elif dry_flood_rows:
                lines.append("")
                lines.append("[Flood]: No flooding has occurred yet.")

        # ── Pressure plate terrain note ───────────────────────────────────────
        plate_positions = getattr(env, "_plate_positions", {})
        if plate_positions:
            lines.append("")
            for (px, py), plate in sorted(plate_positions.items()):
                boulder_positions = {pos for pos, _ in getattr(env, "_boulders", [])}
                # For trigger plates, triggered=permanent takes priority over weighted.
                # For open plates, only weighted/unweighted matters.
                if plate.effect == "trigger":
                    state = (
                        "triggered (permanent)" if plate.triggered else
                        "weighted (active)" if plate.weighted else
                        "unweighted"
                    )
                else:
                    state = "weighted (active)" if plate.weighted else "unweighted"
                pos_str = _compass_offset(ax, ay, px, py)
                lines.append(
                    f"[Pressure plate at {pos_str}]: effect={plate.effect}, "
                    f"target={plate.target_id!r}, state={state}"
                )

        # ── Continuous door terrain notes ─────────────────────────────────────
        continuous_doors = getattr(env, "_continuous_doors", {})
        if continuous_doors:
            lines.append("")
            for (cx, cy), key_id in sorted(continuous_doors.items()):
                pos_str = _compass_offset(ax, ay, cx, cy)
                lines.append(
                    f"[Door at {pos_str}]: continuous rule — open ONLY while "
                    f"{key_id} key remains in the lock; removing the key closes this door"
                )
        # ── Fire source terrain notes ───────────────────────────────────────────────
        fire_source_positions = getattr(env, "_fire_sources", [])
        if fire_source_positions:
            lines.append("")
            for (fsx, fsy) in fire_source_positions:
                pos_str = _compass_offset(ax, ay, fsx, fsy)
                lines.append(
                    f"[Fire source at {pos_str}]: interactive — stand here with an "
                    "unlit torch to light it; a lit torch can burn through wood doors"
                )

        # ── Extinguished fire (scorch marks) terrain notes ─────────────────────────
        scorch_tiles = [
            (x, y, env.grid.get(x, y))
            for y in range(env.height)
            for x in range(env.width)
            if isinstance(env.grid.get(x, y), FireTile)
            and not env.grid.get(x, y).active
        ]
        if scorch_tiles:
            lines.append("")
            for (ftx, fty, _ft) in scorch_tiles:
                pos_str = _compass_offset(ax, ay, ftx, fty)
                lines.append(
                    f"[Scorch marks at {pos_str}]: fire was extinguished here — now passable"
                )

        # ── Water tile terrain note ──────────────────────────────────────────────────
        water_positions = [
            (wx, wy)
            for wy in range(env.height)
            for wx in range(env.width)
            if isinstance(env.grid.get(wx, wy), WaterTile)
        ]
        if water_positions:
            lines.append("")
            rows_set = sorted({wy for _, wy in water_positions})
            row_str = "/".join(f"row {r}" for r in rows_set)
            lines.append(
                f"[Water]: Water tiles at {row_str} — impassable without a boat; "
                "no boat is present in this room."
            )
        # ── Collect all objects ───────────────────────────────────────────────
        object_entries: list[tuple[int, str]] = []  # (distance, text)
        noticeboards: list[str] = []

        for y in range(env.height):
            for x in range(env.width):
                obj = env.grid.get(x, y)
                if obj is None:
                    continue
                if isinstance(obj, (Wall, RiverTile, DarkZone, MudTile, FireTile, FloodTile, ElevatedTile, PressurePlate, WaterTile, FireSource)):
                    continue   # terrain, not objects
                if obj.type in _SKIP_TYPES:
                    continue

                # ── Notice / signpost / combat-log boards ────────────────────
                if isinstance(obj, (NoticeBoardObject, SignpostObject, CombatLogObject)):
                    if isinstance(obj, CombatLogObject):
                        kind = "Combat Log"
                    elif isinstance(obj, NoticeBoardObject):
                        kind = "Notice board"
                    else:
                        kind = "Signpost"
                    pos = _compass_offset(ax, ay, x, y)
                    noticeboards.append(f'[{kind} at {pos}]: "{obj.text}"')
                    continue

                # ── Regular objects ───────────────────────────────────────────
                label_parts: list[str] = []

                # Object description
                type_name = obj.type
                color = getattr(obj, "color", "")
                obj_label = f"{color} {type_name}".strip() if color else type_name

                if isinstance(obj, Door):
                    from minigrid.core.constants import STATE_TO_IDX
                    _IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
                    # Door encodes state in index 2 of encode()
                    _, _, state_idx = obj.encode()
                    state_str = _IDX_TO_STATE.get(int(state_idx), "")
                    if state_str:
                        obj_label += f" ({state_str})"

                if isinstance(obj, Boulder):
                    obj_label += " (pushable, cannot be picked up)"

                if isinstance(obj, Torch):
                    if obj.is_lit:
                        obj_label = f"{obj.color} torch (lit — can burn wood doors)"
                    else:
                        obj_label = f"{obj.color} torch (unlit — must be lit at a fire source before it can burn wood doors)"

                # Position
                pos_str = _compass_offset(ax, ay, x, y)
                label_parts.append(f"{obj_label}: {pos_str}")

                # River annotation
                if (x, y) in river_map:
                    rt = river_map[(x, y)]
                    label_parts.append(f"(in river, flowing {DIR_LABEL[rt.direction]})")

                # Elevated ground annotation (flood-immune)
                elevated_cells: set = getattr(env, "_elevated_cells", set())
                if elevated_cells and (x, y) in elevated_cells:
                    label_parts.append("(on elevated ground — flood-immune)")

                # Condition annotation
                cond = condition_label(obj)
                if cond:
                    label_parts.append(cond)

                dist = _manhattan(ax, ay, x, y)
                object_entries.append((dist, " ".join(label_parts)))

        if self.sort_by_distance:
            object_entries.sort(key=lambda e: e[0])

        # ── Assemble output ───────────────────────────────────────────────────
        lines.append("")
        if not object_entries and not noticeboards:
            lines.append("Nothing visible around you.")
        else:
            if object_entries:
                for _, text in object_entries:
                    lines.append(f"- {text}")
            if noticeboards:
                for nb in noticeboards:
                    lines.append(f"- {nb}")

        return "\n".join(lines)
