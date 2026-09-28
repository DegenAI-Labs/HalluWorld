from __future__ import annotations

import random
from typing import Optional

from minigrid.minigrid_env import MiniGridEnv

from halluworld.probe import Probe, ProbeResult
from halluworld.tracks.grid.serializers.symbolic import visible_objects

# Plausible (color, object) pairs that could appear in our envs.
# Used to generate foil questions (asking about absent objects).
_FOIL_POOL: list[tuple[str, str]] = [
    ("red", "key"),
    ("blue", "key"),
    ("green", "key"),
    ("red", "ball"),
    ("blue", "ball"),
    ("green", "ball"),
    ("yellow", "ball"),
    ("red", "door"),
    ("blue", "door"),
    ("green", "box"),
    ("purple", "box"),
    ("yellow", "key"),
    ("grey", "ball"),
]


def _lateral_label(lateral: int) -> str:
    if lateral < 0:
        return "left"
    if lateral > 0:
        return "right"
    return "directly ahead"


class PresenceProbe(Probe):
    """Ask: 'Is there a [color] [object] in your current view?'

    With probability `positive_rate` the question is about an object that IS
    visible (ground_truth=True). Otherwise it is about a plausible foil that is
    NOT visible (ground_truth=False).

    This directly probes perceptual hallucination: does the LM claim to see
    something that isn't there, or deny something that is?
    """

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
    ):
        """
        Args:
            positive_rate: Fraction of calls that will ask about a present object.
            rng:           Optional seeded Random instance for reproducibility.
        """
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        ask_positive = (
            bool(visible) and self.rng.random() < self.positive_rate
        )

        if ask_positive:
            obj = self.rng.choice(visible)
            color, obj_type = obj["color"], obj["object"]
            ground_truth = True
            source = "visible"
        else:
            # Build foil: a (color, object) pair NOT currently visible
            visible_set = {(o["color"], o["object"]) for o in visible}
            foils = [f for f in _FOIL_POOL if f not in visible_set]
            if not foils:
                # Fallback: just ask about a visible object
                obj = self.rng.choice(visible) if visible else {"color": "red", "object": "key"}
                color, obj_type = obj["color"], obj["object"]
                ground_truth = True
                source = "visible_fallback"
            else:
                color, obj_type = self.rng.choice(foils)
                ground_truth = False
                source = "foil"

        question = (
            f"Is there a {color} {obj_type} in your current field of view?\n"
            f"Answer with exactly 'yes' or 'no'."
        )

        return ProbeResult(
            probe_type="presence",
            question=question,
            ground_truth=ground_truth,
            metadata={
                "color": color,
                "object": obj_type,
                "source": source,
                "n_visible_objects": len(visible),
            },
        )


class LocationProbe(Probe):
    """Ask: 'Where is the [color] [object] relative to you?'

    Only asks about objects that ARE currently visible, since asking for the
    location of an absent object is a different type of probe (see PresenceProbe).

    Ground truth is a dict:
        {
            "steps_ahead": int,   # 0 = same row as agent, 1 = one step ahead, …
            "lateral":     int,   # negative=left, 0=center, positive=right
            "direction":   str,   # "left" | "right" | "directly ahead"
            "description": str,   # human-readable, same as what we'd accept from LM
        }

    This probes spatial hallucination: does the LM correctly represent the
    position of a visible object relative to itself?
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        if not visible:
            # No non-trivial objects visible; return a sentinel result.
            return ProbeResult(
                probe_type="location",
                question="Are there any notable objects in your current field of view?",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_visible_objects"},
            )

        obj = self.rng.choice(visible)
        color, obj_type = obj["color"], obj["object"]
        steps = obj["steps_ahead"]
        lateral = obj["lateral"]
        direction = _lateral_label(lateral)

        # Canonical ground-truth description. The evaluator will parse LM output
        # against these fields rather than requiring a verbatim match.
        gt_description = (
            f"{steps} step{'s' if steps != 1 else ''} ahead"
            + (f", {abs(lateral)} step{'s' if abs(lateral) != 1 else ''} to the {direction}"
               if direction != "directly ahead" else "")
        )

        question = (
            f"Where is the {color} {obj_type} relative to your current position and facing direction?\n"
            f"Respond with valid JSON only, no explanation. Use exactly this format:\n"
            f'{{"steps_ahead": <integer>, "lateral": <integer>}}\n'
            f"where steps_ahead is the number of steps forward (0 = same row as you), "
            f"and lateral is positive for right, negative for left, 0 for directly ahead."
        )

        return ProbeResult(
            probe_type="location",
            question=question,
            ground_truth={
                "steps_ahead": steps,
                "lateral": lateral,
                "direction": direction,
                "description": gt_description,
            },
            metadata={
                "color": color,
                "object": obj_type,
                "fov_x": obj["fov_x"],
                "fov_y": obj["fov_y"],
            },
        )


# ── helpers for allocentric coordinate conversion ────────────────────────────

# Forward vectors (dx, dy) = (dcol, drow) for each agent direction
_FWD:   list[tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # E S W N
# Right-of-agent vectors (90° clockwise from forward)
_RIGHT: list[tuple[int, int]] = [(0, 1), (-1, 0), (0, -1), (1, 0)]   # S W N E

_COMPASS_LABELS = {( 0, -1): "north", (0,  1): "south",
                   ( 1,  0): "east",  (-1, 0): "west"}


def _to_world(env: MiniGridEnv, steps_ahead: int, lateral: int) -> tuple[int, int]:
    """Convert egocentric (steps_ahead, lateral) to world (col, row)."""
    ax, ay = int(env.agent_pos[0]), int(env.agent_pos[1])
    d = int(env.agent_dir)
    fx, fy = _FWD[d]
    rx, ry = _RIGHT[d]
    return (ax + steps_ahead * fx + lateral * rx,
            ay + steps_ahead * fy + lateral * ry)


def _compass(from_xy: tuple[int, int], to_xy: tuple[int, int]) -> str:
    """Return the dominant compass direction (N/S/E/W) from one cell to another."""
    dx = to_xy[0] - from_xy[0]
    dy = to_xy[1] - from_xy[1]
    if abs(dx) >= abs(dy):
        return "east" if dx >= 0 else "west"
    else:
        return "south" if dy >= 0 else "north"


def _egocentric_desc(steps_ahead: int, lateral: int) -> str:
    """Short human-readable egocentric position for use in probe questions."""
    fwd = f"{steps_ahead} step{'s' if steps_ahead != 1 else ''} ahead"
    if lateral == 0:
        return f"{fwd}, directly in front of you"
    side = "right" if lateral > 0 else "left"
    lat = abs(lateral)
    return f"{fwd}, {lat} step{'s' if lat != 1 else ''} to your {side}"


# ── CountProbe ────────────────────────────────────────────────────────────────

class CountProbe(Probe):
    """Ask: 'How many [color] [type] do you see?'

    Ground truth is the integer count of matching visible objects. With
    ``target_violation=True`` the probe preferentially picks a (color, type) pair
    whose count does NOT match the dominant pattern (i.e., one that has been
    altered), making it harder for a model to answer by pattern-completion.
    """

    def __init__(
        self,
        positive_rate: float = 0.7,
        rng: Optional[random.Random] = None,
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        if not visible:
            return ProbeResult(
                probe_type="count",
                question="How many objects do you see?",
                ground_truth=0,
                metadata={"skipped": True, "reason": "no_visible_objects"},
            )

        if self.rng.random() < self.positive_rate:
            obj = self.rng.choice(visible)
            color, obj_type = obj["color"], obj["object"]
            count = sum(
                1 for o in visible if o["color"] == color and o["object"] == obj_type
            )
            source = "visible"
        else:
            # Ask about a foil (color, type) not visible → count = 0
            visible_set = {(o["color"], o["object"]) for o in visible}
            foils = [f for f in _FOIL_POOL if f not in visible_set]
            if foils:
                color, obj_type = self.rng.choice(foils)
                count = 0
                source = "foil"
            else:
                obj = self.rng.choice(visible)
                color, obj_type = obj["color"], obj["object"]
                count = sum(
                    1 for o in visible if o["color"] == color and o["object"] == obj_type
                )
                source = "visible_fallback"

        question = (
            f"How many {color} {obj_type}s do you currently see in your field of view?\n"
            f"Answer with just the number."
        )

        return ProbeResult(
            probe_type="count",
            question=question,
            ground_truth=count,
            metadata={"color": color, "object": obj_type, "source": source},
        )


# ── AttributeProbe ────────────────────────────────────────────────────────────

class AttributeProbe(Probe):
    """Ask about a specific attribute (color or state) of a visible object.

    ``attribute="color"`` asks what color a specific object is — useful when
    a color violation (e.g., one blue key among red keys) might be missed by
    a model defaulting to the dominant pattern.

    ``attribute="state"`` asks about the door state (open / closed / locked)
    and only targets Door objects.  Ground truth comes directly from the
    observed state field.
    """

    def __init__(
        self,
        attribute: str = "color",
        rng: Optional[random.Random] = None,
    ):
        if attribute not in ("color", "state"):
            raise ValueError(f"attribute must be 'color' or 'state', got {attribute!r}")
        self.attribute = attribute
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        if self.attribute == "state":
            candidates = [o for o in visible if o["object"] == "door"]
        else:
            candidates = visible

        if not candidates:
            return ProbeResult(
                probe_type="attribute",
                question="",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_candidates",
                          "attribute": self.attribute},
            )

        obj = self.rng.choice(candidates)
        loc = _egocentric_desc(obj["steps_ahead"], obj["lateral"])

        if self.attribute == "color":
            question = (
                f"What color is the {obj['object']} that is {loc}?\n"
                f"Answer with just the color name."
            )
            ground_truth = obj["color"]
        else:
            state = obj["state"] or "closed"
            question = (
                f"What state is the {obj['color']} {obj['object']} that is {loc}? "
                f"(open, closed, or locked)\n"
                f"Answer with just the state word."
            )
            ground_truth = state

        return ProbeResult(
            probe_type="attribute",
            question=question,
            ground_truth=ground_truth,
            metadata={
                "attribute": self.attribute,
                "color": obj["color"],
                "object": obj["object"],
                "state": obj.get("state", ""),
                "steps_ahead": obj["steps_ahead"],
                "lateral": obj["lateral"],
            },
        )


# ── AllocentricLocationProbe ──────────────────────────────────────────────────

_INVISIBLE_TYPES = {"floor", "empty", "unseen", "agent", ""}


def _ray_first_object(
    env: MiniGridEnv,
    start_xy: tuple[int, int],
    dx: int,
    dy: int,
) -> tuple[tuple[int, int] | None, str]:
    """Walk a compass ray from *start_xy* and return (world_pos, label) of the
    first *named* object hit (Key, Ball, Door, Goal, custom tile, …), or
    (None, 'a wall') if a wall or grid boundary is reached first.

    Unlike the old approach this scans the *actual grid*, not just the FOV
    subset — so invisible objects are correctly identified.
    """
    for dist in range(1, max(env.width, env.height)):
        nx, ny = start_xy[0] + dist * dx, start_xy[1] + dist * dy
        if not (0 <= nx < env.width and 0 <= ny < env.height):
            return (None, "a wall")
        cell = env.grid.get(nx, ny)
        if cell is None:
            continue
        cell_type = getattr(cell, "type", "")
        if cell_type == "wall":
            return (None, "a wall")
        if cell_type not in _INVISIBLE_TYPES:
            color = getattr(cell, "color", "")
            label = f"{color} {cell_type}".strip() if color else cell_type
            return ((nx, ny), label)
    return (None, "a wall")


class AllocentricLocationProbe(Probe):
    """Ask: 'What is [north/south/east/west] of the [color] [type]?'

    The question is framed in absolute compass terms.  The model must:
      1. Note its current facing direction from the serializer output.
      2. Convert the egocentric position of an anchor object to world coords.
      3. Scan along the given compass ray to identify the first named object.

    Ground truth is determined by scanning the *actual grid* (not just the
    FOV) for the first named object along the ray.  The probe only emits a
    question when that first real-world object is also **visible** in the
    current serialised observation — guaranteeing that the question is
    answerable purely from what the model can see, without world-knowledge
    of the full level layout.

    If no such answerable pair exists (e.g. when facing west with few visible
    objects), the probe is skipped rather than producing an unanswerable or
    wrong question.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        if len(visible) < 2:
            return ProbeResult(
                probe_type="allocentric_location",
                question="",
                ground_truth=None,
                metadata={"skipped": True, "reason": "fewer_than_2_visible_objects"},
            )

        # Build world-position → visible-object lookup
        world_pos: dict[tuple[int, int], dict] = {}
        for obj in visible:
            wp = _to_world(env, obj["steps_ahead"], obj["lateral"])
            world_pos[wp] = obj

        shuffled = list(visible)
        self.rng.shuffle(shuffled)

        anchor = None
        direction = None
        target_desc = None

        # For each (anchor, compass-direction) pair:
        #   1. Ray-cast the *actual grid* to find the first named object.
        #   2. Accept only if that object is also visible (in world_pos).
        # This ensures (a) GT is correct and (b) the question is answerable.
        for obj_a in shuffled:
            world_a = _to_world(env, obj_a["steps_ahead"], obj_a["lateral"])
            compass_items = list(_COMPASS_LABELS.items())
            self.rng.shuffle(compass_items)
            for (dx, dy), label in compass_items:
                hit_pos, hit_label = _ray_first_object(env, world_a, dx, dy)
                if hit_pos is not None and hit_pos in world_pos:
                    # First real object along ray is visible → answerable question
                    anchor = obj_a
                    direction = label
                    target_desc = hit_label
                    break
            if anchor is not None:
                break

        if anchor is None:
            return ProbeResult(
                probe_type="allocentric_location",
                question="",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_answerable_compass_pair"},
            )

        anchor_loc = _egocentric_desc(anchor["steps_ahead"], anchor["lateral"])
        question = (
            f"What is the first object directly {direction} of the {anchor['color']} {anchor['object']} "
            f"(which is {anchor_loc}), moving in a straight {direction} line?\n"
            f"Answer with a short description (e.g. 'blue key', 'yellow door', 'wall')."
        )

        return ProbeResult(
            probe_type="allocentric_location",
            question=question,
            ground_truth=target_desc,
            metadata={
                "anchor_color": anchor["color"],
                "anchor_object": anchor["object"],
                "direction": direction,
                "anchor_world": _to_world(env, anchor["steps_ahead"], anchor["lateral"]),
            },
        )


# ── OrderProbe ────────────────────────────────────────────────────────────────

def _obj_label(o: dict) -> str:
    label = f"{o['color']} {o['object']}" if o["color"] else o["object"]
    if o["object"] == "door" and o["state"]:
        label += f" ({o['state']})"
    return label


class OrderProbe(Probe):
    """Ask: 'List all visible objects from nearest to furthest.'

    Ground truth is a list of object descriptors sorted by ``steps_ahead``
    ascending, with ``abs(lateral)`` as a tiebreaker (center objects first).
    The evaluator can check whether the LM's ordering matches this list by
    parsing the numbered response and comparing positions sequentially.

    Designed for dense, linear scenes (e.g. Corridor Gauntlet) where ordering
    errors are the primary failure mode.
    """

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        if not visible:
            return ProbeResult(
                probe_type="order",
                question="",
                ground_truth=None,
                metadata={"skipped": True, "reason": "no_visible_objects"},
            )

        sorted_objs = sorted(
            visible,
            key=lambda o: (o["steps_ahead"], abs(o["lateral"])),
        )

        gt_list = [
            {
                "label": _obj_label(o),
                "steps_ahead": o["steps_ahead"],
                "lateral": o["lateral"],
            }
            for o in sorted_objs
        ]

        question = (
            "List all visible objects from nearest to furthest "
            "(by number of steps ahead of you).\n"
            "Format your answer as a numbered list, one object per line:\n"
            "1. [color] [type], [N] steps ahead, [left/right/directly ahead]\n"
            "Include every visible object in the correct order."
        )

        return ProbeResult(
            probe_type="order",
            question=question,
            ground_truth=gt_list,
            metadata={"n_objects": len(gt_list)},
        )


# ── BetweenProbe ──────────────────────────────────────────────────────────────

class BetweenProbe(Probe):
    """Ask: 'What is between you and the [color] [type]?'

    Selects a distant target object and asks the model to list everything on
    the same lateral row that is closer than the target.  Prefers targets that
    have at least one intervening object on the exact same lateral so the ground
    truth is non-empty and verifiable.

    Ground truth is a list of matching objects sorted by ``steps_ahead``.
    An empty list indicates nothing is between the agent and the target on
    that row (expected answer: "nothing").
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()

    def generate(self, env: MiniGridEnv) -> ProbeResult:
        visible = visible_objects(env)

        if len(visible) < 2:
            return ProbeResult(
                probe_type="between",
                question="",
                ground_truth=None,
                metadata={"skipped": True, "reason": "fewer_than_2_visible_objects"},
            )

        # Sort candidates from farthest to nearest so we pick a distant target first.
        candidates = sorted(visible, key=lambda o: o["steps_ahead"], reverse=True)

        target = None
        between_objs: list[dict] = []

        # Prefer a target that has same-lateral objects between it and the agent.
        for obj in candidates:
            same_row = [
                o for o in visible
                if o is not obj
                and o["steps_ahead"] < obj["steps_ahead"]
                and o["lateral"] == obj["lateral"]
            ]
            if same_row:
                target = obj
                between_objs = sorted(same_row, key=lambda o: o["steps_ahead"])
                break

        if target is None:
            # Fallback: pick furthest object; "between" = any closer visible object
            target = candidates[0]
            between_objs = sorted(
                [o for o in visible if o["steps_ahead"] < target["steps_ahead"]],
                key=lambda o: o["steps_ahead"],
            )

        gt_list = [
            {
                "label": _obj_label(o),
                "steps_ahead": o["steps_ahead"],
                "lateral": o["lateral"],
            }
            for o in between_objs
        ]

        tgt_label = _obj_label(target)
        tgt_loc = _egocentric_desc(target["steps_ahead"], target["lateral"])
        lat_clause = (
            "directly in front of you (lateral 0)"
            if target["lateral"] == 0
            else (
                f"{abs(target['lateral'])} step{'s' if abs(target['lateral']) != 1 else ''} "
                f"to your {'left' if target['lateral'] < 0 else 'right'} "
                f"(lateral {target['lateral']})"
            )
        )

        question = (
            f"The {tgt_label} is {tgt_loc}. "
            f"Looking at objects that share the same lateral offset ({lat_clause}), "
            f"which ones are closer to you (fewer steps ahead) than the {tgt_label}?\n"
            f"List them in order from nearest to furthest, one per line. "
            f"Do not list the {tgt_label} itself. "
            f"If there are none, answer 'nothing'."
        )

        return ProbeResult(
            probe_type="between",
            question=question,
            ground_truth=gt_list,
            metadata={
                "target_label": tgt_label,
                "target_steps_ahead": target["steps_ahead"],
                "target_lateral": target["lateral"],
                "n_between": len(gt_list),
            },
        )


class FixedProbe(Probe):
    """A hardcoded probe with a pre-specified question and ground truth.

    Used to target known violation cells or cross-zone absence queries in
    specific hand-crafted levels where the correct answer is deterministic
    regardless of episode seed.

    Args:
        probe_type:   Identifier string (e.g. "attribute", "presence", "count").
        question:     The exact natural-language question to ask the LM.
        ground_truth: The correct answer (same type as the corresponding dynamic probe).
        metadata:     Optional extra context stored in the ProbeResult.
    """

    def __init__(
        self,
        probe_type: str,
        question: str,
        ground_truth,
        metadata: dict | None = None,
    ) -> None:
        self._probe_type = probe_type
        self._question = question
        self._ground_truth = ground_truth
        self._metadata = metadata or {}

    def generate(self, env) -> ProbeResult:
        return ProbeResult(
            probe_type=self._probe_type,
            question=self._question,
            ground_truth=self._ground_truth,
            metadata=self._metadata,
        )

