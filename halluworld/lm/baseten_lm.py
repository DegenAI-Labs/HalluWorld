from __future__ import annotations

import logging
import time
import os

from halluworld.lm.base import LM, LMResponse

try:
    from openai import OpenAI as _OpenAIClient, RateLimitError as _RateLimitError
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install openai: pip install openai") from exc

log = logging.getLogger(__name__)


class BasetenLM(LM):
    """LM implementation for Baseten deployed models and serverless API.

    Baseten uses OpenAI-compatible API format with custom base URLs.

    Example (Deployed Model)::

        import os
        lm = BasetenLM(
            model="google/gemma-4-26B-A4B-it",
            base_url="https://model-xxxx.api.baseten.co/environments/production/sync/v1",
            api_key=os.environ["BASETEN_API_KEY"]
        )
        response = lm.query(system="You are an agent...", user="Is there a red key?")
        print(response.text)

    Example (Serverless API)::

        lm = BasetenLM(
            model="zai-org/GLM-5",
            base_url="https://inference.baseten.co/v1",
            api_key=os.environ["BASETEN_API_KEY"]
        )

    Note:
        - Deployed models require a custom base_url specific to your deployment
        - Serverless API uses base_url="https://bridge.baseten.co/v1/direct"
        - Both support temperature and standard OpenAI parameters
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
    ):
        """Initialize Baseten LM.

        Args:
            model: Model name (e.g., "google/gemma-4-26B-A4B-it", "zai-org/GLM-5")
            base_url: Baseten API endpoint URL
                - Deployed: "https://model-xxxx.api.baseten.co/environments/production/sync/v1"
                - Serverless: "https://inference.baseten.co/v1"
            api_key: Baseten API key (from environment if not provided)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            max_retries: Number of retries on rate limit
        """
        self.model = model
        self.base_url = base_url
        self.temperature = temperature

        # Increase max_tokens for thinking models (Qwen) to avoid cutoffs
        # Qwen generates verbose reasoning in separate 'reasoning' field
        # but needs enough tokens to complete and populate 'content' field with final answer
        if "qwen" in model.lower():
            self.max_tokens = max(max_tokens, 4096)
            log.info(f"Increased max_tokens to {self.max_tokens} for Qwen model {model}")
        else:
            self.max_tokens = max_tokens

        self.max_retries = max_retries

        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.environ.get("BASETEN_API_KEY")
            if not api_key:
                raise ValueError(
                    "BASETEN_API_KEY must be provided or set in environment"
                )

        self._client = _OpenAIClient(api_key=api_key, base_url=base_url)

    def query(self, system: str, user: str, max_tokens: int | None = None) -> LMResponse:
        """Query the Baseten API with exponential backoff on rate-limit errors.

        A caller-supplied ``max_tokens`` acts as a floor, never a ceiling: the
        constructor already raises the cap for verbose thinking models (Qwen
        emits its reasoning into a separate field and needs headroom to also
        populate ``content``), and honoring a smaller per-call value would
        reintroduce the empty-response failure that raise exists to prevent.
        """
        effective_max_tokens = (
            self.max_tokens if max_tokens is None else max(max_tokens, self.max_tokens)
        )
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                # Deployed models (custom base_url) need model=""
                # Serverless models (inference.baseten.co) need actual model name
                model_param = "" if "model-" in self.base_url else self.model

                completion = self._client.chat.completions.create(
                    model=model_param,
                    temperature=self.temperature,
                    max_tokens=effective_max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                )
                choice = completion.choices[0]
                usage = completion.usage
                log.debug(
                    "tokens used: prompt=%d completion=%d",
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )

                # Thinking models (Qwen) separate verbose reasoning from final answer:
                # - 'reasoning' field: full thinking process
                # - 'content' field: clean final answer (populated when enough tokens)
                # With 2048 max_tokens, content should always be populated
                text = choice.message.content
                if text is None and hasattr(choice.message, 'reasoning'):
                    # Fallback to reasoning if content not populated (shouldn't happen with 2048 tokens)
                    text = choice.message.reasoning
                    log.warning("Content field was None, falling back to reasoning field")

                # Strip whitespace from response
                text = (text or "").strip()

                return LMResponse(
                    text=text or "",
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens if usage else 0,
                    completion_tokens=usage.completion_tokens if usage else 0,
                )
            except _RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                log.warning("Rate limit hit, retrying in %.1fs (attempt %d/%d)",
                            delay, attempt + 1, self.max_retries)
                time.sleep(delay)
                delay *= 2
