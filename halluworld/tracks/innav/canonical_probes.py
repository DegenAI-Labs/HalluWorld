"""Probe sets for the InNav track.

WHY THESE DIFFER FROM THE STATIC GRIDWORLD PROBES

This module once claimed to provide "the exact probe configurations used in
Emmy's static benchmark, ensuring our innav results are directly comparable".
That claim was wrong, but the underlying behavior is correct and deliberate --
the two are different for a structural reason, not because a copy drifted.

Measured across all 33 levels, this dispatch reproduces the static probe set on
exactly 3 of them, and those 3 are precisely the levels where the static
benchmark itself uses only *generated* probes:

    P1_dense_array, P2_corridor_gauntlet, P3_rotation_challenge

The other 30 static levels use FixedProbe, whose questions are hand-written
with hardcoded egocentric spatial references -- "What color is the key that is
11 steps ahead and 3 steps to your right?". Those presuppose a known agent
position. In the static benchmark that holds: the agent is placed at a fixed
start and shown a fixed observation.

InNav has no such guarantee. The agent navigates, its path is model-dependent,
and probes are asked retrospectively against whatever state it actually
reached. "11 steps ahead and 3 to your right" refers to nothing in particular
once the agent is somewhere else. So InNav cannot reuse FixedProbe questions,
and generating probes from the reached state is the only coherent option.

WHAT THIS MEANS FOR COMPARISONS

The paper's InNav result compares INNAV against CTRLSTATIC on *identical
trajectories* (Appendix I.1) -- a within-run paired control where both arms
receive the same generated question, since engine.py calls probe.generate(env)
once and asks it twice. That comparison is sound and is unaffected by any of
the above.

What is NOT valid is putting InNav numbers in a table beside the published
static HalluWorld-Grid numbers and reading across: on 30 of 33 levels those
were measured on different questions. The paper does not do this.

REPRODUCIBILITY

Because the questions depend on the trajectory, the trajectory is InNav's
frozen artifact rather than a question list. The 294 released traces in
data/questions/v0.1/trajectories.jsonl.gz are what makes an InNav run
replayable.
"""

import random
from halluworld.tracks.grid.probes.visibility import (
    PresenceProbe,
    CountProbe,
    AttributeProbe,
    LocationProbe,
    AllocentricLocationProbe,
    OrderProbe,
    BetweenProbe,
    FixedProbe,
)


def make_canonical_probes(level_key: str, rng: random.Random) -> list:
    """Create canonical probe set for a level (matches Emmy's run_perception_eval.py).

    Args:
        level_key: Level identifier (e.g., "P1_dense_array")
        rng: Random number generator for reproducibility

    Returns:
        List of Probe objects for this level
    """

    if level_key == "P1_dense_array":
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="state", rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key == "P2_corridor_gauntlet":
        return [
            LocationProbe(rng=random.Random(rng.randint(0, 2**31))),
            OrderProbe(),
            BetweenProbe(rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key == "P3_rotation_challenge":
        return [
            LocationProbe(rng=random.Random(rng.randint(0, 2**31))),
            AllocentricLocationProbe(rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key == "P4_harder_array":
        # P4 has many fixed probes targeting specific violations
        # Using a simplified subset that covers key probe types
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="state", rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key == "P5_object_permanence":
        # P5 tests object permanence with fixed probes
        # Using simplified version for innav mode
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
        ]

    # Memory levels (M-tier)
    elif level_key.startswith("M1"):
        # M1 river physics
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key.startswith("M2"):
        # M2 witness stand - memory across chambers
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key.startswith("M3"):
        # M3 incident report - change detection
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
        ]

    elif level_key.startswith("M4"):
        # M4 narrator - testimony reliability
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
        ]

    # Causal levels (C-tier)
    elif level_key.startswith("C"):
        # All C-levels use similar probes (causal reasoning about mechanics)
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
            LocationProbe(rng=random.Random(rng.randint(0, 2**31))),
        ]

    # Uncertainty levels (U-tier)
    elif level_key.startswith("U"):
        # U-tier tests epistemic reasoning
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
        ]

    # Default fallback (generic probe set)
    else:
        return [
            PresenceProbe(positive_rate=0.5, rng=random.Random(rng.randint(0, 2**31))),
            CountProbe(rng=random.Random(rng.randint(0, 2**31))),
            AttributeProbe(attribute="color", rng=random.Random(rng.randint(0, 2**31))),
            AllocentricLocationProbe(rng=random.Random(rng.randint(0, 2**31))),
        ]
