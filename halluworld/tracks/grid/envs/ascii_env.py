from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Wall
from minigrid.minigrid_env import MiniGridEnv

# Custom tile types — must be imported early to register in OBJECT_TO_IDX
from halluworld.tracks.grid.envs.tiles import (
    DarkZone,
    Boulder,
    CombatLogObject,
    ElevatedTile,
    FireSource,
    FireTile,
    FloodTile,
    MudTile,
    NoticeBoardObject,
    PressurePlate,
    RiverTile,
    SignpostObject,
    Torch,
    TreeTile,
    WaterTile,
    advance_flood_tiles,
    apply_river_physics,
    decay_wet_conditions,
    ensure_condition,
    DIR_FROM_CHAR,
    DIR_DELTA,
    WET_TURNS_DEFAULT,
)
from halluworld.data import LEVELS_DIR

# ── character ↔ color maps ────────────────────────────────────────────────────
CHAR_TO_COLOR: dict[str, str] = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "y": "yellow",
    "p": "purple",
    "e": "grey",
}
COLOR_TO_CHAR: dict[str, str] = {v: k for k, v in CHAR_TO_COLOR.items()}

# object type char → MiniGrid class
# D = door auto-locked when a matching key exists in the grid
# L = door explicitly locked regardless of key presence
# O = door explicitly open
_TYPE_TO_CLASS: dict[str, type] = {
    "K": Key,
    "B": Ball,
    "X": Box,
    "D": Door,
    "L": Door,  # explicitly locked
    "O": Door,  # explicitly open
}

# Agent-direction char → MiniGrid dir index (right=0, down=1, left=2, up=3)
# Use 'r' for a random direction chosen at reset time.
_DIR_CHAR_TO_IDX: dict[str, int] = {
    ">": 0,
    "v": 1,
    "<": 2,
    "^": 3,
    ".": 0,  # default: right
    "r": -1,  # sentinel: randomise at reset
}


# ── level spec ────────────────────────────────────────────────────────────────

class LevelSpec:
    """Parsed representation of an ASCII level.

    Cell format (each cell is exactly 2 characters):
        ##   wall
        ..   empty floor
        A.   agent start (facing right)
        A>   agent facing right  A<  left  Av  down  A^  up
        **   goal
        rK   red key      (color char = r/g/b/y/p/e, type = K/D/B/X)
        bD   blue door    (locked if a matching key exists in the grid)
        gB   green ball
        yX   yellow box

    File format (space-separated cells, one row per line):
        ;; any line starting with ; or // is a comment
        metadata lines: "key: value" before the first grid row
        ## ## ## ## ##
        ## A. rK .. ##
        ## .. .. ** ##
        ## ## ## ## ##

    Wind metadata (optional):
        wind: col:offset,col:offset,...
        e.g. wind: 3:-1,4:-1,5:-2
        A negative offset shifts the agent UP (lower y-index); positive shifts DOWN.
        Wind is applied after every step() if the agent's new column is in the wind map.
    """

    def __init__(self, cells: list[list[str]], metadata: dict[str, str]) -> None:
        self.cells = cells
        self.metadata = metadata

    @property
    def height(self) -> int:
        return len(self.cells)

    @property
    def width(self) -> int:
        return len(self.cells[0]) if self.cells else 0

    @property
    def agent_view_size(self) -> int:
        return int(self.metadata.get("agent_view_size", 7))

    @property
    def see_through_walls(self) -> bool:
        return str(self.metadata.get("see_through_walls", "false")).lower() in ("true", "1", "yes")

    @property
    def wind(self) -> dict[int, int]:
        """Column → row-offset wind map parsed from ``wind: col:offset,...`` metadata.

        A negative offset moves the agent UP (decreasing y); positive moves DOWN.
        Returns an empty dict when no wind is defined.

        Example metadata line::

            wind: 3:-1,4:-1,5:-2
        """
        raw = self.metadata.get("wind", "").strip()
        if not raw:
            return {}
        result: dict[int, int] = {}
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            col_str, _, off_str = token.partition(":")
            try:
                result[int(col_str.strip())] = int(off_str.strip())
            except ValueError:
                pass
        return result

    def to_text(self) -> str:
        """Serialise back to the space-separated ASCII format."""
        lines: list[str] = []
        for k, v in self.metadata.items():
            lines.append(f"{k}: {v}")
        if self.metadata:
            lines.append("")
        for row in self.cells:
            lines.append(" ".join(row))
        return "\n".join(lines)


# ── parser ────────────────────────────────────────────────────────────────────

def _looks_like_grid_row(line: str) -> bool:
    """Return True if the line looks like a row of 2-char grid cells."""
    parts = line.split()
    if parts and all(len(p) == 2 for p in parts):
        return True
    # packed format: "##..rK.." — even length, each pair is a valid cell
    if len(line) % 2 == 0 and len(line) >= 2 and " " not in line:
        return True
    return False


def _parse_row(line: str) -> list[str]:
    """Split a grid row string into 2-char cell tokens."""
    if " " in line:
        parts = line.split()
        if all(len(p) == 2 for p in parts):
            return parts
    if len(line) % 2 == 0:
        return [line[i : i + 2] for i in range(0, len(line), 2)]
    return []


def parse_level(text: str) -> LevelSpec:
    """Parse ASCII level text into a LevelSpec."""
    metadata: dict[str, str] = {}
    grid_lines: list[list[str]] = []
    in_grid = False

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            continue

        if not in_grid:
            if _looks_like_grid_row(line):
                in_grid = True
            elif ":" in line:
                key, _, val = line.partition(":")
                metadata[key.strip()] = val.strip()
                continue

        if in_grid:
            cells = _parse_row(line)
            if cells:
                grid_lines.append(cells)

    if not grid_lines:
        raise ValueError("No grid data found in level text.")

    widths = {len(row) for row in grid_lines}
    if len(widths) > 1:
        raise ValueError(
            f"Grid rows have inconsistent widths: {sorted(widths)}. "
            "All rows must have the same number of cells."
        )

    return LevelSpec(grid_lines, metadata)


def load_level(path: str | Path) -> LevelSpec:
    """Load a LevelSpec from a .txt file."""
    return parse_level(Path(path).read_text())


# ── environment ───────────────────────────────────────────────────────────────

class AsciiEnv(MiniGridEnv):
    """MiniGrid environment defined by a fixed ASCII layout.

    Unlike SimpleEnv (which randomises positions each episode), AsciiEnv
    reproduces the exact layout on every reset — making it suited for
    controlled, repeatable spatial experiments.

    Quick construction::

        env = AsciiEnv.from_text('''
            ## ## ## ## ##
            ## A. rK .. ##
            ## ## rD ## ##
            ## .. .. ** ##
            ## ## ## ## ##
        ''', render_mode="rgb_array")

        env = AsciiEnv.from_file(str(LEVELS_DIR / "two_room_small.txt"), render_mode="rgb_array")
    """

    def __init__(self, spec: LevelSpec, **kwargs: object) -> None:
        self._spec = spec
        self._wind: dict[int, int] = spec.wind  # column → row offset
        self._river_map: dict[tuple[int, int], RiverTile] = {}  # populated in _gen_grid
        self._flood_tiles: list[tuple[tuple[int, int], FloodTile]] = []  # populated in _gen_grid
        self._plate_targets: dict[str, object] = {}  # target_id → door WorldObj
        self._plate_positions: dict[tuple[int, int], PressurePlate] = {}  # pos → plate
        self._boulders: list[tuple[tuple[int, int], Boulder]] = []  # (pos, boulder)
        self._continuous_doors: dict[tuple[int, int], str] = {}  # (col, row) → key_id for continuous-rule doors
        self._torches: list[tuple[tuple[int, int], Torch]] = []  # (pos, torch) for all torches
        self._fire_sources: list[tuple[int, int]] = []  # positions of FireSource tiles

        agent_view_size = int(kwargs.pop("agent_view_size", spec.agent_view_size))
        see_through_walls = bool(kwargs.pop("see_through_walls", spec.see_through_walls))
        max_steps = int(kwargs.pop("max_steps", 4 * spec.width * spec.height))

        self._mission_str = spec.metadata.get("mission", "navigate the gridworld")
        super().__init__(
            mission_space=MissionSpace(mission_func=lambda: self._mission_str),
            width=spec.width,
            height=spec.height,
            agent_view_size=agent_view_size,
            see_through_walls=see_through_walls,
            max_steps=max_steps,
            **kwargs,
        )

    @classmethod
    def from_text(cls, text: str, **kwargs: object) -> "AsciiEnv":
        return cls(parse_level(text), **kwargs)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: object) -> "AsciiEnv":
        return cls(load_level(path), **kwargs)

    def wind_offset_for(self, col: int) -> int:
        """Return the row offset applied to column *col* each step, or 0."""
        return self._wind.get(col, 0)

    def step(self, action: int):  # type: ignore[override]
        """Execute action, then apply column wind to the agent's new position.

        Wind displaces the agent by ``wind_offset_for(agent_col)`` rows after
        every step.  If the target cell is a wall the displacement is skipped
        (the agent stays put).  The displacement does not consume an action
        slot — it is an *environmental effect* applied transparently after every
        move.
        """
        # Boulder push: if agent moves forward into a boulder, push it first
        MG_ACTION_FORWARD = 2
        if action == MG_ACTION_FORWARD and self._boulders:
            self._try_push_boulder()

        obs, reward, terminated, truncated, info = super().step(action)

        # Apply tile physics after every step
        if not terminated and not truncated:
            if self._river_map:
                apply_river_physics(self)
            # Decay wet conditions regardless of whether there's a river
            decay_wet_conditions(self)
            # Advance flood tiles (activates any whose rise_step <= step_count)
            if self._flood_tiles:
                advance_flood_tiles(self)
            # Update boulder positions and check pressure plates
            if self._plate_targets:
                self._check_pressure_plates()

        if self._wind and not terminated and not truncated:
            ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
            offset = self._wind.get(ax, 0)
            if offset != 0:
                ny = ay + offset
                # Clamp to grid and skip if the target cell is a wall
                if 0 <= ny < self.height:
                    target = self.grid.get(ax, ny)
                    if target is None or target.can_overlap():
                        self.agent_pos = np.array([ax, ny])
                        # Re-generate observation after the displacement
                        obs = self.gen_obs()

        return obs, reward, terminated, truncated, info

    def _try_push_boulder(self) -> None:
        """If agent faces a Boulder, push it one tile in the agent's direction.

        The push succeeds only if the destination cell is empty (None) or a
        PressurePlate.  On success the boulder's grid position is updated and
        the _boulders list is patched.  On failure nothing changes and the
        agent's forward move will simply be blocked by the boulder as usual.
        """
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        dx, dy = DIR_DELTA[self.agent_dir]
        bx, by = ax + dx, ay + dy

        boulder_obj = self.grid.get(bx, by)
        if not isinstance(boulder_obj, Boulder):
            return

        tx, ty = bx + dx, by + dy
        if tx < 0 or tx >= self.width or ty < 0 or ty >= self.height:
            return  # out of bounds — push blocked

        dest = self.grid.get(tx, ty)
        if dest is not None and not isinstance(dest, PressurePlate):
            return  # blocked by wall or object

        # Move boulder
        self.grid.set(bx, by, None)
        self.grid.set(tx, ty, boulder_obj)

        # Update _boulders list
        for i, (pos, b) in enumerate(self._boulders):
            if b is boulder_obj:
                self._boulders[i] = ((tx, ty), b)
                break

        # Update pressure plate weighted state
        self._check_pressure_plates()

    def _plate_cells(self):
        """Yield (pos, PressurePlate) for all plate objects tracked in _plate_positions."""
        for pos, plate in getattr(self, "_plate_positions", {}).items():
            yield pos, plate

    def _check_pressure_plates(self) -> None:
        """Update weighted state for all plates, then fire their effects."""
        plate_positions: dict[tuple[int, int], PressurePlate] = getattr(
            self, "_plate_positions", {}
        )
        boulder_positions: set[tuple[int, int]] = {pos for pos, _ in self._boulders}
        for pos, plate in plate_positions.items():
            plate.weighted = pos in boulder_positions
            plate.check(self)

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self._river_map = {}
        self._flood_tiles = []
        self._elevated_cells: set[tuple[int, int]] = set()
        self._continuous_doors = {}
        self._torches = []
        self._fire_sources = []
        self._notice_texts: dict[tuple[int, int], str] = {}  # (x,y) → board text

        # ── Parse river zone from metadata ───────────────────────────────────
        river_rows: set[int] = set()
        river_cols: set[int] = set()  # empty = all cols
        river_tile_proto: RiverTile | None = None
        raw_rows = self._spec.metadata.get("river_rows", "").strip()
        if raw_rows:
            if "-" in raw_rows:
                lo, _, hi = raw_rows.partition("-")
                river_rows = set(range(int(lo.strip()), int(hi.strip()) + 1))
            else:
                river_rows = {int(raw_rows)}
            rdir_str = self._spec.metadata.get("river_dir", "east").strip().lower()
            rdir = DIR_FROM_CHAR.get(rdir_str, 0)
            rspeed = int(self._spec.metadata.get("river_speed", "1").strip())
            river_tile_proto = RiverTile(direction=rdir, speed=rspeed)
            # Optional column range for the river (default: full width)
            raw_cols = self._spec.metadata.get("river_cols", "").strip()
            if raw_cols:
                if "-" in raw_cols:
                    clo, _, chi = raw_cols.partition("-")
                    river_cols = set(range(int(clo.strip()), int(chi.strip()) + 1))
                else:
                    river_cols = {int(raw_cols)}

        # ── Parse notice/sign/flood texts from metadata ──────────────────────
        notice_texts: dict[str, str] = {}  # "0".."9" → text
        sign_texts: dict[str, str] = {}
        combat_log_texts: dict[str, str] = {}
        flood_rise_steps: dict[str, int] = {}  # cell idx → rise_step
        # Parse elevated cell coordinates from metadata: elevated_cell_N: row=R,col=C
        elevated_meta_cells: set[tuple[int, int]] = set()
        plate_meta: dict[str, dict] = {}  # idx → {target_id, effect}
        plate_by_idx: dict[str, "PressurePlate"] = {}  # idx → PressurePlate (populated during grid scan)
        for key, val in self._spec.metadata.items():
            if key.startswith("notice_") and key[7:].isdigit():
                notice_texts[key[7:]] = val
            elif key.startswith("sign_") and key[5:].isdigit():
                sign_texts[key[5:]] = val
            elif key.startswith("combat_log_") and key[11:].isdigit():
                combat_log_texts[key[11:]] = val
            elif key.startswith("flood_") and (key[6:].isdigit() or key[6:] in "abcdef"):
                try:
                    flood_rise_steps[key[6:]] = int(val.strip())
                except ValueError:
                    pass
            elif key.startswith("elevated_cell_") and key[14:].isdigit():
                params: dict[str, int] = {}
                for token in val.split(","):
                    k2, _, v2 = token.strip().partition("=")
                    try:
                        params[k2.strip()] = int(v2.strip())
                    except ValueError:
                        pass
                ec_row = params.get("row")
                ec_col = params.get("col")
                if ec_row is not None and ec_col is not None:
                    elevated_meta_cells.add((ec_col, ec_row))
            elif key.startswith("plate_") and key[6:].isdigit():
                # Syntax: plate_N: target=<id>,effect=trigger|open
                p: dict[str, str] = {}
                for token in val.split(","):
                    k2, _, v2 = token.strip().partition("=")
                    p[k2.strip()] = v2.strip()
                plate_meta[key[6:]] = p

        # Track whether we found the agent position
        agent_pos: tuple[int, int] | None = None
        agent_dir: int = 0

        for row_idx, row in enumerate(self._spec.cells):
            for col_idx, cell in enumerate(row):
                x, y = col_idx, row_idx  # MiniGrid uses (col, row) = (x, y)

                # ── River zone background fill ─────────────────────────────────
                if (
                    river_tile_proto is not None
                    and row_idx in river_rows
                    and (not river_cols or col_idx in river_cols)
                ):
                    bg = RiverTile(river_tile_proto.direction, river_tile_proto.speed)
                    self._river_map[(x, y)] = bg
                    # Place background river tile; objects placed below will overwrite it
                    self.grid.set(x, y, bg)

                if cell == "##":
                    self.grid.set(x, y, Wall())

                elif cell in ("..", "  "):
                    pass  # empty floor (or river background already set above)

                elif cell[0] == "A":
                    agent_pos = (x, y)
                    dir_ch = cell[1]
                    if dir_ch == "r":
                        agent_dir = int(self.np_random.integers(0, 4))
                    else:
                        agent_dir = _DIR_CHAR_TO_IDX.get(dir_ch, 0)

                elif cell == "**":
                    self.grid.set(x, y, Goal())

                # ── New tile codes ─────────────────────────────────────────────
                elif cell == "~~":   # fire tile  (check before generic ~X)
                    self.grid.set(x, y, FireTile())

                elif cell[0] == "~":   # river tile (explicit, any direction)
                    dir_ch = cell[1].upper()
                    rdir = DIR_FROM_CHAR.get(dir_ch, 0)
                    rspeed = int(self._spec.metadata.get("river_speed", "1").strip())
                    rt = RiverTile(direction=rdir, speed=rspeed)
                    self.grid.set(x, y, rt)
                    self._river_map[(x, y)] = rt

                elif cell == "TT":
                    self.grid.set(x, y, TreeTile())

                elif cell == "ZZ":
                    self.grid.set(x, y, DarkZone())

                elif cell == "MM":
                    self.grid.set(x, y, MudTile())

                elif cell == "ET":
                    self.grid.set(x, y, ElevatedTile())
                    self._elevated_cells.add((x, y))

                elif cell == "BO":
                    bo = Boulder()
                    self.grid.set(x, y, bo)
                    self._boulders.append(((x, y), bo))

                elif cell[0] == "b" and cell[1].isdigit():
                    # Boulder sitting on pressure plate N at reset time.
                    # Creates the plate (tracked in _plate_positions) and places
                    # a boulder on top of it in the grid.
                    idx = cell[1]
                    meta = plate_meta.get(idx, {})
                    target_id = meta.get("target", f"plate_target_{idx}")
                    effect = meta.get("effect", "trigger")
                    plate = PressurePlate(target_id=target_id, effect=effect)
                    self._plate_positions[(x, y)] = plate
                    plate_by_idx[idx] = plate
                    bo = Boulder()
                    self.grid.set(x, y, bo)
                    self._boulders.append(((x, y), bo))

                elif cell == "WW":
                    self.grid.set(x, y, WaterTile())

                elif cell == "FS":
                    self.grid.set(x, y, FireSource())
                    self._fire_sources.append((x, y))

                elif len(cell) == 2 and cell[1] == "H":
                    color = CHAR_TO_COLOR.get(cell[0], "yellow")
                    is_lit = (cell[0] == "y")
                    th = Torch(color=color, is_lit=is_lit)
                    self.grid.set(x, y, th)
                    self._torches.append(((x, y), th))

                elif cell[0] == "p" and cell[1].isdigit():
                    idx = cell[1]
                    meta = plate_meta.get(idx, {})
                    target_id = meta.get("target", f"plate_target_{idx}")
                    effect = meta.get("effect", "trigger")
                    plate = PressurePlate(target_id=target_id, effect=effect)
                    self.grid.set(x, y, plate)
                    self._plate_positions[(x, y)] = plate
                    plate_by_idx[idx] = plate

                elif cell[0] == "f" and (cell[1].isdigit() or cell[1] in "abcdef"):
                    idx = cell[1]
                    rise_step = flood_rise_steps.get(idx, 999)
                    ft = FloodTile(rise_step=rise_step)
                    self.grid.set(x, y, ft)
                    self._flood_tiles.append(((x, y), ft))

                elif cell[0] == "n" and cell[1].isdigit():
                    idx = cell[1]
                    text = notice_texts.get(idx, f"Notice board {idx}")
                    nb = NoticeBoardObject(text)
                    self.grid.set(x, y, nb)
                    self._notice_texts[(x, y)] = text

                elif cell[0] == "s" and cell[1].isdigit():
                    idx = cell[1]
                    text = sign_texts.get(idx, f"Signpost {idx}")
                    sg = SignpostObject(text)
                    self.grid.set(x, y, sg)

                elif cell[0] == "c" and cell[1].isdigit():
                    idx = cell[1]
                    text = combat_log_texts.get(idx, f"Combat Log {idx}")
                    cl_obj = CombatLogObject(text)
                    self.grid.set(x, y, cl_obj)

                elif len(cell) == 2 and cell[1] in _TYPE_TO_CLASS:
                    color_ch, type_ch = cell[0], cell[1]
                    color = CHAR_TO_COLOR.get(color_ch, "red")
                    obj_cls = _TYPE_TO_CLASS[type_ch]
                    if obj_cls is Door:
                        if type_ch == "O":
                            obj = Door(color, is_open=True)
                        elif type_ch == "L":
                            obj = Door(color, is_locked=True)
                        else:
                            obj = Door(color, is_locked=self._has_key(color))
                    else:
                        obj = obj_cls(color)
                    self.grid.set(x, y, obj)

        # Fallback agent position: first empty interior cell
        if agent_pos is None:
            for row_idx, row in enumerate(self._spec.cells):
                for col_idx, cell in enumerate(row):
                    if cell == "..":
                        agent_pos = (col_idx, row_idx)
                        break
                if agent_pos is not None:
                    break

        if agent_pos is None:
            raise ValueError("Level has no agent position (A.) and no empty cell to fall back to.")

        self.agent_pos = np.array(agent_pos)
        self.agent_dir = agent_dir
        self.mission = self._mission_str

        # Merge metadata-declared elevated cells into _elevated_cells
        self._elevated_cells.update(elevated_meta_cells)

        # Wire up plate targets: door_id_N: row=R,col=C,id=<target_id>
        for key, val in self._spec.metadata.items():
            if key.startswith("door_id_") and key[8:].isdigit():
                params: dict[str, str] = {}
                for token in val.split(","):
                    k2, _, v2 = token.strip().partition("=")
                    params[k2.strip()] = v2.strip()
                try:
                    dc = int(params["col"])
                    dr = int(params["row"])
                    did = params["id"]
                    door_obj = self.grid.get(dc, dr)
                    if door_obj is not None:
                        self._plate_targets[did] = door_obj
                except (KeyError, ValueError):
                    pass

        # ── Pre-trigger / pre-weight demo plates ──────────────────────────────
        # plate_N_pretriggered: true  → one-shot trigger already fired; door stays open
        # plate_N_preweighted: true   → continuous plate with boulder already on it; door open
        for key, val in self._spec.metadata.items():
            if not val.strip().lower() == "true":
                continue
            parts = key.split("_")
            # Expect: plate_<idx>_pretriggered  or  plate_<idx>_preweighted
            if len(parts) == 3 and parts[0] == "plate" and parts[1].isdigit():
                pidx = parts[1]
                action = parts[2]
                plate = plate_by_idx.get(pidx)
                if plate is None:
                    continue
                if action == "pretriggered":
                    plate.triggered = True
                    # Directly open the target door (check() won't fire when already triggered)
                    target = self._plate_targets.get(plate.target_id)
                    if target is not None and hasattr(target, "is_open"):
                        target.is_open = True
                        target.is_locked = False
                elif action == "preweighted":
                    plate.weighted = True
                    plate.check(self)  # Opens door for effect=open while weighted

        # Wire up continuous_door_N: row=R,col=C,key=<key_id>
        for key, val in self._spec.metadata.items():
            if key.startswith("continuous_door_") and key[16:].isdigit():
                params: dict[str, str] = {}
                for token in val.split(","):
                    k2, _, v2 = token.strip().partition("=")
                    params[k2.strip()] = v2.strip()
                try:
                    cc = int(params["col"])
                    cr = int(params["row"])
                    ckey = params["key"]
                    self._continuous_doors[(cc, cr)] = ckey
                except (KeyError, ValueError):
                    pass

        # Activate any flood tiles with rise_step=0 immediately on reset
        if self._flood_tiles:
            advance_flood_tiles(self)

        # ── Apply prewet conditions from metadata ──────────────────────────────
        # Syntax: prewet_N: row=<r>,col=<c>,turns=<t>
        # Sets initial wet_turns on the object at (col, row), marking it as
        # having been in the river for some steps before the episode began.
        for key, val in self._spec.metadata.items():
            if not (key.startswith("prewet_") and key[7:].isdigit()):
                continue
            params: dict[str, int] = {}
            for token in val.split(","):
                k, _, v = token.strip().partition("=")
                try:
                    params[k.strip()] = int(v.strip())
                except ValueError:
                    pass
            pw_row = params.get("row")
            pw_col = params.get("col")
            pw_turns = params.get("turns", WET_TURNS_DEFAULT)
            if pw_row is None or pw_col is None:
                continue
            obj = self.grid.get(pw_col, pw_row)
            if obj is not None and not isinstance(obj, RiverTile):
                ensure_condition(obj)
                obj.wet_turns_remaining = pw_turns
                obj.condition = "wet"

        # ── Initialise plate states from physical boulder positions ──────────
        # Handles bN compound tokens (boulder on plate at reset) and ensures
        # effect=open plates are immediately active if their boulder is present.
        if self._plate_targets:
            self._check_pressure_plates()

    def _has_key(self, color: str) -> bool:
        """Return True if a key of this color exists anywhere in the spec."""
        for row in self._spec.cells:
            for cell in row:
                if len(cell) == 2 and cell[1] == "K" and CHAR_TO_COLOR.get(cell[0]) == color:
                    return True
        return False


# ── convenience factory ───────────────────────────────────────────────────────

def make_env_from_ascii(source: str | Path, **kwargs: object) -> AsciiEnv:
    """Create an AsciiEnv from a file path or raw ASCII text string.

    Args:
        source: Either a file path (str or Path) or multi-line ASCII level text.
        **kwargs: Forwarded to AsciiEnv (e.g. render_mode, agent_view_size).
    """
    p = Path(source) if not isinstance(source, Path) else source
    if p.exists() and p.is_file():
        return AsciiEnv.from_file(p, **kwargs)
    # treat as raw text
    return AsciiEnv.from_text(str(source), **kwargs)
