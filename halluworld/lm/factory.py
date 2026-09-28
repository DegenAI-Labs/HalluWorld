"""Build a provider client from parsed CLI arguments.

Every replay path needs the same thing -- turn --provider/--model/--max-tokens
into a client -- and each one growing its own copy is how the Baseten base_url
bug got written once already. One factory, so a provider fix lands everywhere.
"""

from __future__ import annotations

import os

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "baseten": "BASETEN_API_KEY",
    "xai": "XAI_API_KEY",
}

#: xAI serves an OpenAI-compatible API, so it reuses OpenAILM with a base_url
#: rather than needing a client of its own. https://docs.x.ai/overview
XAI_BASE_URL = "https://api.x.ai/v1"


class LMConfigError(RuntimeError):
    """A client cannot be built from these arguments."""


def make_lm(args):
    """Return a provider client for `args`, or raise LMConfigError."""
    provider = args.provider
    if provider not in PROVIDER_KEYS:
        raise LMConfigError(f"unknown provider {provider!r}")
    key_name = PROVIDER_KEYS[provider]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise LMConfigError(f"{key_name} must be set for provider={provider}")

    max_tokens = getattr(args, "max_tokens", 256)
    retries = int(getattr(args, "retries", None) or 3)
    temperature = float(getattr(args, "temperature", 0.0) or 0.0)

    if provider in ("openai", "xai"):
        from halluworld.lm.openai_lm import OpenAILM

        base_url = getattr(args, "api_base", None)
        if provider == "xai":
            base_url = base_url or XAI_BASE_URL
        return OpenAILM(
            model=args.model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=retries,
            reasoning_effort=getattr(args, "reasoning_effort", None),
            base_url=base_url,
        )

    if provider == "anthropic":
        from halluworld.lm.anthropic_lm import AnthropicLM

        thinking = getattr(args, "thinking_effort", None)
        return AnthropicLM(
            model=args.model,
            api_key=api_key,
            temperature=1.0 if thinking else temperature,
            max_tokens=max_tokens,
            max_retries=retries,
            thinking_effort=thinking,
            refusal_retries=int(getattr(args, "refusal_retries", 0) or 0),
        )

    from halluworld.lm.baseten_lm import BasetenLM

    # base_url has no default on BasetenLM, so omitting it is a TypeError rather
    # than a fallback. Same resolution order battery.py and perception.py use;
    # --api-base additionally points this at any OpenAI-compatible endpoint.
    base_url = (
        getattr(args, "api_base", None)
        or os.environ.get("BASETEN_BASE_URL")
        or "https://inference.baseten.co/v1"
    )
    return BasetenLM(
        model=args.model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=retries,
    )
