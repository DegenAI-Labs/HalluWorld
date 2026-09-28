from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LMResponse:
    """Raw response from an LM plus lightweight provenance metadata."""
    text: str             # raw response text
    model: str = ""       # model identifier used for this call
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Set when `text` is blank because the call failed rather than because the
    # model had nothing to say -- e.g. a reasoning model's token budget was too
    # small to satisfy the API (400 BadRequestError). run_benchmark() checks
    # this before handing the response to an evaluator: a caller-side failure
    # is not a wrong answer, and scoring it as one would misrepresent an
    # infrastructure problem as a measured hallucination.
    error: str | None = None
    # Why generation stopped, when the provider says. Recorded so an empty
    # answer can be told apart from a refusal, a hit token cap, or a stop
    # sequence -- all of which otherwise land as the same blank string.
    stop_reason: str | None = None
    # How many API calls produced this response. 1 unless a client re-asked
    # (e.g. after a refusal); recorded so a re-asked answer is distinguishable
    # from a first-draw one in the results.
    attempts: int = 1


class LM(ABC):
    """Thin interface for querying a language model.

    All benchmark components receive an LM instance rather than calling a
    specific API directly, so swapping models (GPT-4, Llama, stub for tests)
    requires only changing the LM instantiation in benchmark.py.
    """

    @abstractmethod
    def query(self, system: str, user: str, max_tokens: int | None = None) -> LMResponse:
        """Send a system + user prompt and return the model's response.

        Args:
            system: System prompt (context, instructions to the LM).
            user:   User turn containing the observation + question.
            max_tokens: Per-call output cap. ``None`` means use the value the
                instance was constructed with. Implementations that support
                reasoning or thinking modes may raise this value but must not
                lower it below their configured minimum, since a cap that is
                too small starves such models of output and yields an empty
                response that would otherwise be scored as a hallucination.
        """
        ...
