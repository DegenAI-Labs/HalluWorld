"""Gridworld track: MiniGrid-based environments, probes, serializers, evaluators.

Everything specific to the gridworld track lives under this package:

    envs/           ascii_env (the level loader), simple_env, tiles
    probes/         visibility, inventory, dynamics, persistence
    serializers/    symbolic (bullet list), grid (ASCII FOV), memory (compass-absolute)
    evaluators.py   presence, location, dynamics graders
    perception.py   the P/M/C/U/X level battery
    inventory.py  dynamics.py  trajectory.py  record.py

Depends only on the core install -- no optional extras.
"""

from halluworld.tracks.grid.envs.ascii_env import (
    AsciiEnv,
    LevelSpec,
    load_level,
    make_env_from_ascii,
    parse_level,
)
from halluworld.tracks.grid.envs.simple_env import SimpleEnv, make_env
from halluworld.tracks.grid.envs import tiles  # registers custom types in OBJECT_TO_IDX
from halluworld.tracks.grid.evaluators import (
    DynamicsEvaluator,
    LocationEvaluator,
    PresenceEvaluator,
)
from halluworld.tracks.grid.probes.dynamics import DynamicsProbe
from halluworld.tracks.grid.probes.inventory import InventoryProbe
from halluworld.tracks.grid.probes.persistence import PersistenceProbe
from halluworld.tracks.grid.probes.visibility import (
    AllocentricLocationProbe,
    AttributeProbe,
    BetweenProbe,
    CountProbe,
    FixedProbe,
    LocationProbe,
    OrderProbe,
    PresenceProbe,
)
from halluworld.tracks.grid.serializers.grid import GridSerializer
from halluworld.tracks.grid.serializers.memory import MemorySerializer
from halluworld.tracks.grid.serializers.symbolic import SymbolicSerializer

__all__ = [
    # environments
    "AsciiEnv",
    "LevelSpec",
    "load_level",
    "make_env_from_ascii",
    "parse_level",
    "SimpleEnv",
    "make_env",
    "tiles",
    # serializers
    "GridSerializer",
    "MemorySerializer",
    "SymbolicSerializer",
    # probes
    "AllocentricLocationProbe",
    "AttributeProbe",
    "BetweenProbe",
    "CountProbe",
    "DynamicsProbe",
    "FixedProbe",
    "InventoryProbe",
    "LocationProbe",
    "OrderProbe",
    "PersistenceProbe",
    "PresenceProbe",
    # evaluators
    "DynamicsEvaluator",
    "LocationEvaluator",
    "PresenceEvaluator",
]
