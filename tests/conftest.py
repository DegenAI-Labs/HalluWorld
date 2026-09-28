"""Shared pytest configuration.

The autouse fixture below is the important part: it removes every provider API
key from the environment unless a test is explicitly marked `live_api`. That
turns "this test accidentally spent money" into "this test failed with a
missing-key error", which is a much better failure to have.
"""

import pytest

PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BASETEN_API_KEY")


@pytest.fixture(autouse=True)
def _offline_by_default(request, monkeypatch):
    if request.node.get_closest_marker("live_api"):
        return
    for key in PROVIDER_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def stub_lm():
    from halluworld.lm.stub import StubLM

    return StubLM(mode="random", seed=0)


@pytest.fixture(scope="session")
def bank_dir():
    """The v0.1 question bank, or a skip when this checkout cannot reach it.

    The bank is a gated Hugging Face dataset, not a tracked file, so a checkout
    without access (or a token) skips the bank-dependent tests rather than
    erroring. CI sets HF_TOKEN or HALLUWORLD_QUESTIONS_DIR to run them.
    """
    from halluworld.data import QuestionBankUnavailable, questions_dir

    try:
        return questions_dir("v0.1")
    except QuestionBankUnavailable as exc:
        pytest.skip(str(exc).splitlines()[0])


@pytest.fixture
def levels_dir():
    from halluworld.data import LEVELS_DIR

    return LEVELS_DIR
