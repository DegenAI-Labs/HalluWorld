"""Custom tile and object types for HalluWorld memory levels.

Must be imported before any level file is loaded so that the MiniGrid type
registry is patched in time.

New cell codes (2-char, used in ASCII level files):
    ~E  ~S  ~W  ~N   RiverTile flowing east / south / west / north
    ~~               FireTile (impassable while active)
    TT               TreeTile (impassable, blocks FOV)
    ZZ               DarkZone (passable, blocks FOV from outside)
    MM               MudTile (passable, 2-step cost)
    ET               ElevatedTile (always passable, flood immune)
    f0..fa           FloodTile (rise_step encoded in ``flood_N`` / ``flood_a`` metadata key)
    n0..n9           NoticeBoardObject (text from ``notice_N`` metadata key)
    s0..s9           SignpostObject    (text from ``sign_N``   metadata key)
    p0..p9           PressurePlate     (target_id/effect from ``plate_N`` metadata key)
    BO               Boulder (pushable, cannot be picked up)
    WW               WaterTile (impassable static water, no boat mechanic)
    eH               Torch (unlit grey torch; yH = lit yellow torch)
    FS               FireSource (passable interactive tile; lights an unlit torch held by agent)

River zone metadata keys (parsed in AsciiEnv):
    river_rows: <start>-<end>   e.g. ``river_rows: 4-6``
    river_dir:  east|south|west|north  (default east)
    river_speed: <int>           (default 1)
    prewet_N:   row=<r>,col=<c>,turns=<t>   (N = 0,1,2,…)
"""
from __future__ import annotations

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, IDX_TO_OBJECT, OBJECT_TO_IDX
from minigrid.core.world_object import WorldObj
from minigrid.minigrid_env import MiniGridEnv

# ── Extend MiniGrid's type registry ──────────────────────────────────────────
# Must happen before any WorldObj subclass using these types is constructed.

_CUSTOM_TYPES: list[str] = [
    "river",
    "fire",
    "tree",
    "darkzone",
    "mud",
    "noticeboard",
    "signpost",
    "flood",
    "elevated",
    "pressureplate",
    "boulder",
    "water",
    "torch",
    "firesource",
]

_next_idx = max(OBJECT_TO_IDX.values()) + 1
for _t in _CUSTOM_TYPES:
    if _t not in OBJECT_TO_IDX:
        OBJECT_TO_IDX[_t] = _next_idx
        IDX_TO_OBJECT[_next_idx] = _t
        _next_idx += 1
del _t, _next_idx  # tidy namespace

# ── Direction helpers ─────────────────────────────────────────────────────────
# MiniGrid directions: 0=east, 1=south, 2=west, 3=north (matches agent_dir)

DIR_DELTA: dict[int, tuple[int, int]] = {
    0: (1, 0),   # east  → col+1
    1: (0, 1),   # south → row+1
    2: (-1, 0),  # west  → col-1
    3: (0, -1),  # north → row-1
}

DIR_FROM_CHAR: dict[str, int] = {
    "E": 0, "east": 0,
    "S": 1, "south": 1,
    "W": 2, "west": 2,
    "N": 3, "north": 3,
}

DIR_LABEL: dict[int, str] = {0: "east", 1: "south", 2: "west", 3: "north"}

# ── Condition tracking ────────────────────────────────────────────────────────
WET_TURNS_DEFAULT = 4   # W: how many off-river steps before an object dries


def ensure_condition(obj: WorldObj) -> None:
    """Add wet-condition tracking attributes to any WorldObj if not already present."""
    if not hasattr(obj, "condition"):
        obj.condition: str = "dry"
        obj.wet_turns_remaining: int = 0
        obj.consecutive_river_steps: int = 0


def condition_label(obj: WorldObj) -> str:
    """Return a serializer annotation like '[WET, dries in 3 turns]' or ''."""
    if not hasattr(obj, "condition") or obj.condition == "dry":
        return ""
    t = obj.wet_turns_remaining
    if obj.condition == "soaked":
        return f"[SOAKED, dries in {t} turn{'s' if t != 1 else ''}]"
    return f"[WET, dries in {t} turn{'s' if t != 1 else ''}]"


# ── Tile / Object classes ─────────────────────────────────────────────────────

class RiverTile(WorldObj):
    """Passable background tile.  Objects placed on river cells drift each step.

    ``direction``: 0=east, 1=south, 2=west, 3=north
    ``speed``:     cells displaced per env.step()
    """

    def __init__(self, direction: int = 0, speed: int = 1, color: str = "blue") -> None:
        super().__init__("river", color)
        self.direction = direction
        self.speed = speed

    def can_overlap(self) -> bool:
        return True

    def see_behind(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["river"], COLOR_TO_IDX[self.color], self.direction)


class FireTile(WorldObj):
    """Impassable terrain while active.  Extinguished by a wet object."""

    def __init__(self, color: str = "red") -> None:
        super().__init__("fire", color)
        self.active = True

    def can_overlap(self) -> bool:
        # Once deactivated it becomes passable floor
        return not self.active

    def see_behind(self) -> bool:
        return True

    def try_extinguish(self, obj: WorldObj) -> bool:
        """Attempt to extinguish using obj.  Returns True on success."""
        if getattr(obj, "wet_turns_remaining", 0) > 0:
            self.active = False
            return True
        return False

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["fire"], COLOR_TO_IDX[self.color], int(not self.active))


class TreeTile(WorldObj):
    """Impassable freestanding terrain that also blocks FOV behind it."""

    def __init__(self, color: str = "green") -> None:
        super().__init__("tree", color)

    def can_overlap(self) -> bool:
        return False

    def see_behind(self) -> bool:
        return False   # opaque column

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["tree"], COLOR_TO_IDX[self.color], 0)


class DarkZone(WorldObj):
    """Passable tile that blocks FOV for any agent not standing on it."""

    def __init__(self, color: str = "grey") -> None:
        super().__init__("darkzone", color)

    def can_overlap(self) -> bool:
        return True

    def see_behind(self) -> bool:
        return False   # opaque from the outside

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["darkzone"], COLOR_TO_IDX[self.color], 0)


class MudTile(WorldObj):
    """Passable terrain; traversing costs 2 movement steps.

    The extra step cost is enforced by the runner/env step hook, not by
    MiniGrid's built-in action system.
    """

    def __init__(self, color: str = "yellow") -> None:
        super().__init__("mud", color)
        self.move_cost = 2

    def can_overlap(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["mud"], COLOR_TO_IDX[self.color], 0)


class FloodTile(WorldObj):
    """A floor tile that becomes impassable when env.step_count >= rise_step.

    Each FloodTile has its own ``rise_step`` set at level-construction time to
    encode a flood front that advances one row per step.

    While not yet flooded (``flooded=False``) the tile behaves like plain floor
    (passable, transparent).  Once flooded it is treated as a wall — not
    literally a Wall object, but ``can_overlap()`` returns False so MiniGrid's
    movement check blocks the agent.
    """

    def __init__(self, rise_step: int = 0, color: str = "blue") -> None:
        super().__init__("flood", color)
        self.rise_step: int = rise_step
        self.flooded: bool = False

    def can_overlap(self) -> bool:
        return not self.flooded

    def see_behind(self) -> bool:
        return True

    def activate(self) -> None:
        """Mark this tile as flooded (impassable)."""
        self.flooded = True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["flood"], COLOR_TO_IDX[self.color], int(self.flooded))


class ElevatedTile(WorldObj):
    """High ground that water/flood flows around — always passable.

    Objects placed on elevated tiles are never submerged by flood physics.
    Serializer annotates it as "(on elevated ground)".
    """

    def __init__(self, color: str = "yellow") -> None:
        super().__init__("elevated", color)

    def can_overlap(self) -> bool:
        return True

    def see_behind(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["elevated"], COLOR_TO_IDX[self.color], 0)


class NoticeBoardObject(WorldObj):
    """Readable notice board.  Always rendered by MemorySerializer.

    Represents a *stale* t=0 snapshot — the text never changes during an
    episode even when the world state it describes has changed.
    """

    def __init__(self, text: str, color: str = "yellow") -> None:
        super().__init__("noticeboard", color)
        self.text = text

    def can_overlap(self) -> bool:
        return False

    def see_behind(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["noticeboard"], COLOR_TO_IDX[self.color], 0)


class SignpostObject(WorldObj):
    """Like NoticeBoardObject but may contain intentionally false information."""

    def __init__(self, text: str, color: str = "grey") -> None:
        super().__init__("signpost", color)
        self.text = text

    def can_overlap(self) -> bool:
        return False

    def see_behind(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["signpost"], COLOR_TO_IDX[self.color], 0)


class CombatLogObject(WorldObj):
    """A wall-mounted combat log — rendered as [Combat Log] by the serializer."""

    def __init__(self, text: str, color: str = "red") -> None:
        super().__init__("signpost", color)  # reuse signpost type idx
        self.text = text

    def can_overlap(self) -> bool:
        return False

    def see_behind(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["signpost"], COLOR_TO_IDX[self.color], 0)


# ── River physics ─────────────────────────────────────────────────────────────

def apply_river_physics(env: MiniGridEnv) -> None:
    """Move objects sitting on river cells one step downstream.

    Reads ``env._river_map: dict[tuple[int,int], RiverTile]``, which is
    populated by ``AsciiEnv._gen_grid()``.

    For each object found at a river-map position:
    1. Compute the target cell (pos + direction * speed).
    2. If the target is empty or another river cell, move the object there.
    3. Restore the background RiverTile at the vacated cell.
    4. Update condition tracking on the object.
    """
    river_map: dict[tuple[int, int], RiverTile] = getattr(env, "_river_map", {})
    if not river_map:
        return

    # Snapshot moveable objects so we don't double-move any of them
    to_move: list[tuple[tuple[int, int], WorldObj, RiverTile]] = []
    for (rx, ry), river_tile in river_map.items():
        obj = env.grid.get(rx, ry)
        if obj is not None and not isinstance(obj, RiverTile):
            to_move.append(((rx, ry), obj, river_tile))

    for (rx, ry), obj, river_tile in to_move:
        dx, dy = DIR_DELTA[river_tile.direction]
        nx = rx + dx * river_tile.speed
        ny = ry + dy * river_tile.speed

        # Restore the background river tile in the vacated cell
        env.grid.set(rx, ry, river_tile)

        if 0 <= nx < env.width and 0 <= ny < env.height:
            target = env.grid.get(nx, ny)
            if target is None or isinstance(target, RiverTile):
                # Place object in new cell
                env.grid.set(nx, ny, obj)
                if hasattr(obj, "cur_pos") and obj.cur_pos is not None:
                    obj.cur_pos = np.array([nx, ny])

                # Update condition
                ensure_condition(obj)
                obj.consecutive_river_steps += 1
                obj.wet_turns_remaining = WET_TURNS_DEFAULT
                obj.condition = "soaked" if obj.consecutive_river_steps >= 3 else "wet"
        # else: object hits wall or exits grid — leave it removed (river carried it away)


def decay_wet_conditions(env: MiniGridEnv) -> None:
    """Decrement wet_turns for objects that are no longer on river tiles.

    Objects still on a river cell are handled by ``apply_river_physics`` and
    are skipped here.  Objects that have left the river dry out over time.
    """
    river_map: dict[tuple[int, int], RiverTile] = getattr(env, "_river_map", {})

    for y in range(env.height):
        for x in range(env.width):
            obj = env.grid.get(x, y)
            if obj is None:
                continue
            if not hasattr(obj, "condition") or obj.condition == "dry":
                continue
            if (x, y) in river_map:
                continue  # handled by apply_river_physics

            obj.consecutive_river_steps = 0
            obj.wet_turns_remaining -= 1
            if obj.wet_turns_remaining <= 0:
                obj.wet_turns_remaining = 0
                obj.condition = "dry"


def advance_flood_tiles(env: MiniGridEnv) -> None:
    """Activate any FloodTile whose rise_step has been reached.

    Reads ``env._flood_tiles: list[tuple[tuple[int,int], FloodTile]]`` which
    is populated by ``AsciiEnv._gen_grid()``.  Each activated tile becomes
    impassable (``flooded=True``).  Already-flooded tiles are skipped cheaply.

    Also advances the ``env._flood_step_count`` counter used by the serializer
    to report current flood progress.
    """
    flood_tiles: list[tuple[tuple[int, int], "FloodTile"]] = getattr(env, "_flood_tiles", [])
    if not flood_tiles:
        return

    step = getattr(env, "step_count", 0)
    for (fx, fy), tile in flood_tiles:
        if not tile.flooded and step >= tile.rise_step:
            tile.activate()
            # When a FloodTile activates, extinguish any adjacent active FireTile.
            # Flood water spreading into a neighbouring cell puts out the fire there.
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = fx + dx, fy + dy
                if 0 <= nx < env.grid.width and 0 <= ny < env.grid.height:
                    neighbor = env.grid.get(nx, ny)
                    if isinstance(neighbor, FireTile) and neighbor.active:
                        neighbor.active = False


# ── Pressure Plate ────────────────────────────────────────────────────────────

class PressurePlate(WorldObj):
    """Floor-level pressure plate.

    effect="trigger"  — one-shot: fires once when first weighted, then stays
                        in triggered state regardless of weight.
    effect="open"     — continuous: opens target while weighted, closes when not.

    target_id matches an entry in env._plate_targets: dict[str, WorldObj].
    The env is responsible for calling plate.check(env) each step.
    """

    def __init__(self, target_id: str, effect: str = "trigger", color: str = "grey") -> None:
        super().__init__("pressureplate", color)
        self.target_id: str = target_id
        self.effect: str = effect          # "trigger" | "open"
        self.weighted: bool = False        # True when a boulder sits on it
        self.triggered: bool = False       # True after first trigger (one-shot only)

    def can_overlap(self) -> bool:
        return True                        # agent / objects can stand on it

    def see_behind(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        state = 1 if (self.weighted or self.triggered) else 0
        return (OBJECT_TO_IDX["pressureplate"], COLOR_TO_IDX[self.color], state)

    def check(self, env) -> None:
        """Evaluate plate state and apply effect to the target door."""
        target = env._plate_targets.get(self.target_id)
        if target is None:
            return

        if self.effect == "trigger":
            if self.weighted and not self.triggered:
                self.triggered = True
                # Open the target door permanently
                if hasattr(target, "is_open"):
                    target.is_open = True
                    target.is_locked = False
        elif self.effect == "open":
            if self.weighted:
                if hasattr(target, "is_open"):
                    target.is_open = True
                    target.is_locked = False
            else:
                if hasattr(target, "is_open") and target.is_open:
                    target.is_open = False


# ── Boulder ───────────────────────────────────────────────────────────────────

class Boulder(WorldObj):
    """Heavy boulder — cannot be picked up; can be pushed one tile per step.

    Push mechanic (handled in AsciiEnv.step):
      If the agent faces the boulder and steps forward, the boulder moves one
      tile in the same direction IF the destination is empty floor or a
      PressurePlate.  If the destination is blocked the push fails and the
      agent does not move.
    """

    def __init__(self, color: str = "grey") -> None:
        super().__init__("boulder", color)

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return False

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["boulder"], COLOR_TO_IDX[self.color], 0)


class WaterTile(WorldObj):
    """Impassable static water.  No boat mechanic — crossing is simply blocked."""

    def __init__(self, color: str = "blue") -> None:
        super().__init__("water", color)

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return False

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["water"], COLOR_TO_IDX[self.color], 0)


class Torch(WorldObj):
    """Pickup-able torch — starts unlit; must be lit at a fire source.

    Once ``is_lit=True``, the torch can burn through wood doors.
    The burn mechanic is enforced by notice-board rule; ``is_lit`` state
    is set by the env step hook when the agent uses the torch on a FireTile.
    """

    def __init__(self, color: str = "yellow", is_lit: bool = False) -> None:
        super().__init__("torch", color)
        self.is_lit: bool = is_lit

    def can_overlap(self) -> bool:
        return False

    def can_pickup(self) -> bool:
        return True

    def encode(self) -> tuple[int, int, int]:
        state = 1 if self.is_lit else 0
        return (OBJECT_TO_IDX["torch"], COLOR_TO_IDX[self.color], state)


class FireSource(WorldObj):
    """Passable interactive fire source.

    An agent holding an unlit torch can light it here.
    Unlike FireTile, this tile is passable and does NOT block movement.
    """

    def __init__(self, color: str = "red") -> None:
        super().__init__("firesource", color)

    def can_overlap(self) -> bool:
        return True

    def can_pickup(self) -> bool:
        return False

    def encode(self) -> tuple[int, int, int]:
        return (OBJECT_TO_IDX["firesource"], COLOR_TO_IDX[self.color], 0)

