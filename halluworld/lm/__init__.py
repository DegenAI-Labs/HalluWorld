from halluworld.lm.base import LM
from halluworld.lm.stub import StubLM

__all__ = ["LM", "StubLM", "OpenAILM", "AnthropicLM", "BasetenLM"]

try:
    from halluworld.lm.openai_lm import OpenAILM
except ImportError:
    OpenAILM = None  # type: ignore[misc, assignment]
    __all__ = [x for x in __all__ if x != "OpenAILM"]

try:
    from halluworld.lm.anthropic_lm import AnthropicLM
except ImportError:
    pass

try:
    from halluworld.lm.baseten_lm import BasetenLM
except ImportError:
    pass
