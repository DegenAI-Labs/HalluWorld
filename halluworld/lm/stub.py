from __future__ import annotations

import random
from typing import Callable

from halluworld.lm.base import LM, LMResponse


class StubLM(LM):
    """Deterministic/configurable LM for unit tests and dry runs.

    Modes:
      - ``"yes"``      — always returns "yes"
      - ``"no"``       — always returns "no"
      - ``"random"``   — randomly returns "yes" or "no" each call
      - ``"fixed"``    — always returns the ``response`` string
      - ``"callable"`` — calls ``response_fn(system, user) -> str``

    Example::

        lm = StubLM(mode="yes")
        lm = StubLM(mode="random", seed=42)
        lm = StubLM(mode="fixed", response="3 steps ahead, 1 step to the right")
        lm = StubLM(mode="callable", response_fn=lambda s, u: u.split()[-1])
    """

    _MODES = ("yes", "no", "random", "fixed", "callable")

    def __init__(
        self,
        mode: str = "random",
        response: str = "yes",
        response_fn: Callable[[str, str], str] | None = None,
        seed: int | None = None,
    ):
        assert mode in self._MODES, f"Unknown mode {mode!r}. Choose from {self._MODES}"
        if mode == "callable":
            assert response_fn is not None, "response_fn required for callable mode"
        self.mode = mode
        self.response = response
        self.response_fn = response_fn
        self._rng = random.Random(seed)

    def query(self, system: str, user: str, max_tokens: int | None = None) -> LMResponse:
        # max_tokens is accepted for interface compatibility and ignored: the
        # stub never calls an API, so there is no output budget to respect.
        if self.mode == "yes":
            text = "yes"
        elif self.mode == "no":
            text = "no"
        elif self.mode == "random":
            text = self._rng.choice(["yes", "no"])
        elif self.mode == "fixed":
            text = self.response
        else:  # callable
            text = self.response_fn(system, user)  # type: ignore[misc]

        return LMResponse(text=text, model="stub")
