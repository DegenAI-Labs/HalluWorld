from __future__ import annotations

import random
from typing import Optional

from minigrid.minigrid_env import MiniGridEnv

from halluworld.probe import Probe, ProbeResult
from halluworld.tracks.grid.serializers.symbolic import visible_objects


class PersistenceProbe(Probe):
    """Tests object-persistence tracking over a multi-turn trajectory.

    Core idea
    ---------
    As the agent navigates, objects enter and leave its field of view.
    A hallucinating model may claim an object is *currently* visible because
    it saw it in an earlier step — even though that object has since left the
    FOV.  This probe specifically constructs questions that expose this failure.

    Usage in multi-turn eval
    ------------------------
    ``run_multiturn_benchmark`` calls:
      1. ``probe.reset_trajectory()``  at the start of each episode.
      2. ``probe.update_trajectory_seen(env)``  after each trajectory step.
      3. ``probe.generate(env)``  at the **final** step.

    With probability ``(1 - positive_rate)`` the question asks about an object
    that WAS seen earlier in the trajectory but is NOT in the current FOV
    (ground_truth=False).  This is the hallucination-inducing case.

    With probability ``positive_rate`` it asks about a currently-visible object
    (ground_truth=True) as a positive control.

    Probe type: ``"persistence"``  — evaluated by ``PresenceEvaluator``.
    """

    def __init__(
        self,
        positive_rate: float = 0.3,
        rng: Optional[random.Random] = None,
    ):
        """
        Args:
            positive_rate: Fraction of calls that ask about a currently-visible
                           object (GT=True).  The remainder ask about objects
                           seen earlier but absent now (GT=False).
            rng:           Optional seeded Random instance.
        """
        self.positive_rate = positive_rate
        self.rng = rng or random.Random()
        self._trajectory_seen: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ #
    # Trajectory context management (called by run_multiturn_benchmark)   #
    # ------------------------------------------------------------------ #

    def reset_trajectory(self) -> None:
        """Clear the seen-objects set.  Call at the start of each episode."""
        self._trajectory_seen.clear()

    def update_trajectory_seen(self, env: MiniGridEnv) -> None:
        """Record all currently-visible objects into the trajectory set."""
        for o in visible_objects(env):
            self._trajectory_seen.add((o["color"], o["object"]))

    # ------------------------------------------------------------------ #
    # Probe generation                                                     #
    # ------------------------------------------------------------------ #

    def generate(self, env: MiniGridEnv) -> ProbeResult | None:
        visible = visible_objects(env)
        visible_set = {(o["color"], o["object"]) for o in visible}

        # Objects seen during the trajectory but NOT in current FOV
        absent_seen = [
            (c, t) for (c, t) in self._trajectory_seen
            if (c, t) not in visible_set
        ]

        ask_positive = self.rng.random() < self.positive_rate

        if ask_positive and visible:
            o = self.rng.choice(visible)
            color, obj_type = o["color"], o["object"]
            ground_truth = True
            source = "current_visible"
        elif absent_seen:
            color, obj_type = self.rng.choice(absent_seen)
            ground_truth = False
            source = "trajectory_seen_absent"
        elif visible:
            # Fallback when no absent-seen objects are available
            o = self.rng.choice(visible)
            color, obj_type = o["color"], o["object"]
            ground_truth = True
            source = "visible_fallback"
        else:
            return None  # nothing to probe at all

        question = (
            f"Is there a {color} {obj_type} currently in your field of view?\n"
            "Answer with exactly 'yes' or 'no'."
        )

        return ProbeResult(
            probe_type="persistence",
            question=question,
            ground_truth=ground_truth,
            metadata={
                "color": color,
                "object": obj_type,
                "source": source,
                "n_visible_objects": len(visible),
                "n_trajectory_seen": len(self._trajectory_seen),
                "n_absent_seen": len(absent_seen),
            },
        )
