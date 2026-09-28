from __future__ import annotations

import random
from typing import Optional

from minigrid.minigrid_env import MiniGridEnv

from halluworld.probe import Probe, ProbeResult
from halluworld.tracks.grid.serializers.symbolic import visible_objects

# Generic foil pool: plausible (color, type) pairs that could appear in levels.
# Used when no trajectory-seen objects are available as foils.
_FOIL_POOL: list[tuple[str, str]] = [
    ("red", "key"),
    ("blue", "key"),
    ("green", "key"),
    ("yellow", "key"),
    ("red", "ball"),
    ("blue", "ball"),
    ("green", "ball"),
    ("yellow", "ball"),
    ("green", "box"),
    ("purple", "box"),
    ("grey", "ball"),
]


class InventoryProbe(Probe):
    """Tests whether the LM correctly tracks what the agent is currently carrying.

    Core idea
    ---------
    The agent navigates with the ``pickup_oracle`` policy, which routes it to a
    nearby pickupable object (key, ball, or box) and executes ``pickup`` before
    continuing to the goal.  Once picked up, the object is removed from the grid
    and exists only as ``env.carrying`` — it no longer appears in the FOV.

    A hallucinating model may:
      * **Forget**: claim it is carrying *nothing* even though it just picked up
        a key (false negative).
      * **Confuse**: claim it is carrying an object it only *saw* in an earlier
        step but never picked up (false positive).

    This probe creates both types of test:

    GT = True  (``source="carrying_match"``)
        Agent IS carrying an object; probe asks about that exact object.
        A correct model says "yes".

    GT = False  (``source="carrying_foil"``)
        Agent IS carrying object X; probe asks about object Y (seen in the
        trajectory but not the carried object).  A correct model says "no".

    GT = False  (``source="not_carrying"``)
        Agent is NOT carrying anything; probe asks about something seen earlier.
        A correct model says "no" — hallucination would be saying "yes".

    Usage in multi-turn eval
    ------------------------
    ``run_multiturn_benchmark`` (with ``policy="pickup_oracle"``) calls:
      1. ``probe.reset_trajectory()``   at the start of each episode.
      2. ``probe.update_trajectory_seen(env)``  after every trajectory step.
      3. ``probe.generate(env)``  at the **final** step.

    The probe is evaluated by ``PresenceEvaluator`` (yes/no format), same as
    ``PersistenceProbe``.

    Args:
        positive_rate: Fraction of calls where GT=True (asking about what IS
                       carried).  The remainder ask about objects not being
                       carried (GT=False, the hallucination-inducing cases).
                       Default 0.5 for balanced positive / negative split.
        rng:           Optional seeded Random instance.
    """

    def __init__(
        self,
        positive_rate: float = 0.5,
        rng: Optional[random.Random] = None,
    ):
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self._seen_pairs: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ #
    # Trajectory context management                                        #
    # ------------------------------------------------------------------ #

    def reset_trajectory(self) -> None:
        """Clear the seen-objects set.  Call at the start of each episode."""
        self._seen_pairs.clear()

    def update_trajectory_seen(self, env: MiniGridEnv) -> None:
        """Record all currently-visible objects into the trajectory set."""
        for o in visible_objects(env):
            self._seen_pairs.add((o["color"], o["object"]))

    # ------------------------------------------------------------------ #
    # Probe generation                                                     #
    # ------------------------------------------------------------------ #

    def generate(self, env: MiniGridEnv) -> ProbeResult | None:
        carrying = env.carrying

        if carrying is not None:
            c_color = carrying.color
            # type(carrying).__name__.lower() → "key", "ball", "box"
            c_type = type(carrying).__name__.lower()
            carrying_pair = (c_color, c_type)

            ask_positive = self.rng.random() < self.positive_rate

            if ask_positive:
                # Ask about what IS carried → GT=True
                color, obj_type = c_color, c_type
                ground_truth = True
                source = "carrying_match"
            else:
                # Ask about something seen but not being carried → GT=False
                foils = [p for p in self._seen_pairs if p != carrying_pair]
                if not foils:
                    foils = [p for p in _FOIL_POOL if p != carrying_pair]
                if not foils:
                    # Degenerate: only one possible object ever; flip to positive
                    color, obj_type = c_color, c_type
                    ground_truth = True
                    source = "carrying_match_fallback"
                else:
                    color, obj_type = self.rng.choice(foils)
                    ground_truth = False
                    source = "carrying_foil"
        else:
            # Not carrying: ask about something seen → GT=False
            seen = list(self._seen_pairs)
            if seen:
                color, obj_type = self.rng.choice(seen)
            else:
                color, obj_type = self.rng.choice(_FOIL_POOL)
            ground_truth = False
            source = "not_carrying"

        question = (
            f"Are you currently carrying a {color} {obj_type}?\n"
            "Answer with exactly 'yes' or 'no'."
        )

        carrying_label = (
            f"{carrying.color} {type(carrying).__name__.lower()}"
            if carrying is not None else None
        )

        return ProbeResult(
            probe_type="inventory",
            question=question,
            ground_truth=ground_truth,
            metadata={
                "color": color,
                "object": obj_type,
                "carrying": carrying_label,
                "source": source,
                "n_trajectory_seen": len(self._seen_pairs),
            },
        )
