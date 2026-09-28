from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Serializer(ABC):
    """Converts a MiniGrid environment state into a string suitable for LM input.

    Subclasses control *what* the LM sees (full grid, partial FOV, natural language
    description, image, etc.). Swapping the serializer is one of the two main axes
    for varying the reference world model in the HalluWorld framework.
    """

    @abstractmethod
    def serialize(self, env: Any) -> str:
        """Return a string representation of the current env observation."""
        ...
