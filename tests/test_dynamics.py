"""Integration tests for DynamicsProbe + DynamicsEvaluator on a windy level.

Previously this file ran its assertions at module scope, so pytest executed it
during *collection* rather than as a test. Split into functions, with the
diagnostic prints dropped -- pytest shows the failing expression on its own.

The last check in the original file printed the agent's landing position but
never asserted on it; that assertion is now made explicit.
"""

import numpy as np
import pytest

from halluworld.data import LEVELS_DIR
from halluworld.tracks.grid.envs.ascii_env import make_env_from_ascii
from halluworld.tracks.grid import DynamicsEvaluator
from halluworld.evaluator import LMResponse
from halluworld.tracks.grid import DynamicsProbe


def _windy_env():
    env = make_env_from_ascii(str(LEVELS_DIR / "windy_room.txt"), render_mode=None)
    env.reset(seed=42)
    return env


@pytest.fixture
def probe_result():
    return DynamicsProbe().generate(_windy_env())


def test_probe_reports_wind_metadata(probe_result):
    assert probe_result.metadata["has_wind"] is True
    assert "naive_row" in probe_result.metadata
    assert "naive_col" in probe_result.metadata
    assert "wind_offset" in probe_result.metadata


def test_correct_answer_scores_one(probe_result):
    gt = probe_result.ground_truth
    resp = LMResponse(text=f"row={gt['row']}, col={gt['col']}")
    result = DynamicsEvaluator().evaluate(resp, probe_result)
    assert result.correct
    assert result.score == 1.0


def test_naive_no_wind_answer_is_flagged(probe_result):
    """A model that ignores wind should be detectable as such, not merely wrong."""
    md = probe_result.metadata
    resp = LMResponse(text=f"row={md['naive_row']}, col={md['naive_col']}")
    result = DynamicsEvaluator().evaluate(resp, probe_result)
    assert result.details.get("wind_naive_error") is not None


def test_wrong_answer_scores_zero(probe_result):
    result = DynamicsEvaluator().evaluate(LMResponse(text="row=99, col=99"), probe_result)
    assert not result.correct
    assert result.score == 0.0


def test_wind_pushes_agent_when_move_is_blocked():
    """Blocked move, then wind still applies.

    The level places a blue ball at [5, 6], so moving south from [5, 5] is
    blocked. The agent stays at row 5, then wind at col 5 (offset -1) pushes
    it north to row 4.
    """
    env = make_env_from_ascii(str(LEVELS_DIR / "windy_room.txt"), render_mode=None)
    env.reset(seed=0)
    env.agent_pos = np.array([5, 5])
    env.agent_dir = 1  # south
    env.step(2)
    assert list(env.agent_pos) == [5, 4]


def test_wind_compounds_with_an_unblocked_move():
    """Unblocked move north from [6, 8] lands at row 7, then wind carries it to row 6."""
    env = make_env_from_ascii(str(LEVELS_DIR / "windy_room.txt"), render_mode=None)
    env.reset(seed=0)
    env.agent_pos = np.array([6, 8])
    env.agent_dir = 3  # north
    env.step(2)
    assert list(env.agent_pos) == [6, 6]
