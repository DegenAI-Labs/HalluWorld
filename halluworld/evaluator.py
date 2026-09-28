from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from halluworld.lm.base import LMResponse
from halluworld.probe import ProbeResult


@dataclass
class EvalResult:
    """Outcome of evaluating one LM response against a ProbeResult.

    Attributes:
        correct:      Whether the response is fully correct.
        score:        Numeric score in [0, 1]. 1.0 = perfect, 0.0 = wrong.
                      Allows partial credit for structured probes (e.g. location).
        predicted:    The parsed answer extracted from the LM response.
        ground_truth: The expected answer (copied from ProbeResult).
        probe_type:   Probe type identifier (copied from ProbeResult).
        lm_text:      Raw response text for inspection.
        details:      Probe metadata + any extra evaluator notes.
    """
    correct: bool
    score: float
    predicted: Any
    ground_truth: Any
    probe_type: str
    lm_text: str
    details: dict = field(default_factory=dict)


class Evaluator(ABC):
    """Compares an LM response to the ground truth in a ProbeResult.

    Each Probe type has a companion Evaluator that knows how to parse and
    score that probe's response format. Swapping the evaluator is how you
    vary the *conflict policy* axis of the HalluWorld framework (e.g.
    exact match vs. embedding similarity vs. LM judge).
    """

    @abstractmethod
    def evaluate(self, response: LMResponse, probe_result: ProbeResult) -> EvalResult:
        """Score the LM response against the ground truth."""
        ...
