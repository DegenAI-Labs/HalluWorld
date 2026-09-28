"""Verify DynamicsProbe ground truth when wind displaces the agent.

Previously this file ran its assertions at module scope, so pytest executed it
during *collection* rather than as a test, and a failure surfaced as a
collection error rather than a test failure.
"""

import numpy as np

from halluworld.data import LEVELS_DIR
from halluworld.tracks.grid.envs.ascii_env import make_env_from_ascii
from halluworld.tracks.grid import DynamicsProbe


def test_wind_offset_is_applied_to_ground_truth():
    env = make_env_from_ascii(str(LEVELS_DIR / "windy_room.txt"), render_mode=None)
    env.reset(seed=0)

    # Place the agent at col=6, row=9 facing north (dir=3).
    # Moving north takes it to col=6, row=8 before wind is applied;
    # wind at col=6 is -1, carrying it to col=6, row=7, which is unobstructed.
    env.agent_pos = np.array([6, 9])
    env.agent_dir = 3

    result = DynamicsProbe().generate(env)

    assert result.ground_truth == {"row": 7, "col": 6}
    assert result.metadata["naive_row"] == 8, "pre-wind position should be row 8"
    assert result.metadata["wind_offset"] == -1
    assert result.metadata["has_wind"] is True
