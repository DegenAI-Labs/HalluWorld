from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProbeResult:
    """Everything produced by a single probe invocation.

    Attributes:
        probe_type:   Identifier for the probe class (e.g. "presence", "location").
        question:     The natural-language question passed to the LM.
        ground_truth: The correct answer. Type depends on probe_type:
                        - presence:  bool
                        - location:  dict with keys steps_ahead, lateral, direction
        metadata:     Arbitrary extra context (e.g. which object was asked about,
                      whether it was a foil, etc.). Useful for analysis.
        observation_override: When set, the benchmark loop uses this string in
                      place of the serializer's default observation. Memory and
                      uncertainty probes use this to hide or rewrite parts of V.
        prepend_env_observation: When True (and ``observation_override`` is set),
                      the benchmark prepends the serializer's normal observation
                      (e.g. ASCII board + FEN for chess) before the override text so
                      the model still sees **V** alongside an appendix (e.g. moves-only
                      under ``render_history(..., companion_board=True)``), not a second
                      full history block that repeats FEN or says no grid is shown.
    """
    probe_type: str
    question: str
    ground_truth: Any
    metadata: dict = field(default_factory=dict)
    observation_override: Optional[str] = None
    prepend_env_observation: bool = False


class Probe(ABC):
    """Generates a (question, ground_truth) pair grounded in the live env state.

    Probes are the second major axis of variation in HalluWorld: by changing
    the probe type you change *what aspect of world modeling* is being tested
    (perceptual, spatial, causal, memory, …).

    The key invariant: question and ground_truth are always derived from the
    same env state snapshot, so they are guaranteed to be consistent.
    """

    @abstractmethod
    def generate(self, env: Any) -> ProbeResult:
        """Inspect the current env state and return a grounded ProbeResult."""
        ...
