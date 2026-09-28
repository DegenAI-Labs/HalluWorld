from __future__ import annotations

import logging
import time

from halluworld.lm.base import LM, LMResponse

try:
    from anthropic import Anthropic as _AnthropicClient, RateLimitError as _RateLimitError
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install anthropic: pip install anthropic") from exc

log = logging.getLogger(__name__)


class AnthropicLM(LM):
    """LM implementation backed by the Anthropic Messages API.

    Example::

        import os
        lm = AnthropicLM(api_key=os.environ["ANTHROPIC_API_KEY"], model="claude-3-5-sonnet-20241022")
        response = lm.query(system="You are an agent...", user="Is there a red key?")
        print(response.text)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        max_retries: int = 3,
        thinking_effort: str | None = None,
        refusal_retries: int = 0,
    ):
        """Initialize Anthropic LM.

        Args:
            model: Model name (e.g., "claude-sonnet-4-6", "claude-opus-4-6")
            api_key: Anthropic API key
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            max_retries: Number of retries on rate limit
            thinking_effort: Thinking depth for models with adaptive thinking.
                For Claude Opus 4.6 and Sonnet 4.6 with adaptive thinking support.
                Values: "low", "medium" (recommended), "high" (default), "max"
                Controls when and how deeply Claude thinks before responding.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.thinking_effort = thinking_effort
        # Re-ask a refused prompt up to this many more times. The API samples
        # at temperature 1.0 (we no longer send temperature), so a refusal that
        # is a sampling artifact can clear on a second draw, while one the
        # safety layer applies consistently will not. Bounded and opt-in: this
        # biases the sample toward content the model will engage with, and it
        # belongs in the methods note of anything that uses it.
        self.refusal_retries = max(0, int(refusal_retries))
        self._client = _AnthropicClient(api_key=api_key)

        # The SDK dropped `temperature` from Messages.create (absent in 1.1.0),
        # so passing it unconditionally raises TypeError before any request is
        # sent -- every Anthropic call fails client-side, which reads as an
        # api_error rather than a version mismatch. Ask the installed SDK once
        # instead of assuming, so older versions that do accept it still get it.
        try:
            import inspect

            self._sdk_accepts_temperature = "temperature" in inspect.signature(
                self._client.messages.create
            ).parameters
        except (TypeError, ValueError):  # pragma: no cover -- unintrospectable stub
            self._sdk_accepts_temperature = False
        if not self._sdk_accepts_temperature and temperature not in (None, 0.0):
            log.warning(
                "Installed anthropic SDK does not accept temperature; "
                "ignoring temperature=%s for model '%s'.", temperature, model
            )

        # Check if model supports adaptive thinking (Claude 4.6 models)
        self.supports_adaptive_thinking = (
            "opus-4-6" in model or "sonnet-4-6" in model
        )

    def query(self, system: str, user: str, max_tokens: int | None = None) -> LMResponse:
        """Query once, re-asking a refusal up to refusal_retries more times."""
        for refusal_attempt in range(1 + self.refusal_retries):
            response = self._query_once(system, user, max_tokens)
            response.attempts = refusal_attempt + 1
            if response.stop_reason != "refusal" or refusal_attempt == self.refusal_retries:
                return response
            log.info(
                "Model '%s' refused; re-asking (%d/%d)",
                self.model, refusal_attempt + 1, self.refusal_retries,
            )
        return response  # pragma: no cover -- loop always returns

    def _query_once(self, system: str, user: str, max_tokens: int | None = None) -> LMResponse:
        """One API call with exponential backoff on rate-limit errors."""
        # When thinking is enabled, max_tokens covers both thinking + response tokens.
        # Never let a caller override below self.max_tokens for thinking models.
        if max_tokens is not None and self.thinking_effort is not None and self.supports_adaptive_thinking:
            _max = max(max_tokens, self.max_tokens)
        else:
            _max = max_tokens if max_tokens is not None else self.max_tokens
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                # Build request params
                params = {
                    "model": self.model,
                    "max_tokens": _max,
                    "system": system,
                    "messages": [
                        {"role": "user", "content": user},
                    ],
                    # Explicit, not the client default. The SDK refuses a
                    # non-streaming request whose max_tokens implies it could
                    # run past 10 minutes (above ~21333 tokens), raising a
                    # client-side ValueError before anything is sent -- which
                    # is how every 32000-token request to fable-5.1 died as an
                    # "api_error" that never reached the API. The guard is
                    # skipped when a timeout is given, so give one sized to the
                    # request: the SDK's own expected time with 2x headroom,
                    # never under its 10-minute default.
                    "timeout": float(max(600, 2 * 3600 * _max / 128_000)),
                }
                if self._sdk_accepts_temperature and self.temperature is not None:
                    params["temperature"] = self.temperature

                # Add adaptive thinking for supported models
                # Correct format: thinking at top level, effort in output_config
                if self.thinking_effort is not None and self.supports_adaptive_thinking:
                    params["thinking"] = {"type": "adaptive"}
                    params["output_config"] = {"effort": self.thinking_effort}
                elif self.thinking_effort is not None and not self.supports_adaptive_thinking:
                    log.warning(
                        f"Model '{self.model}' does not support adaptive thinking. "
                        f"Ignoring thinking_effort={self.thinking_effort}."
                    )

                message = self._client.messages.create(**params)

                # Extract text from content blocks
                text = ""
                if message.content:
                    for block in message.content:
                        if hasattr(block, 'text'):
                            text += block.text

                stop_reason = getattr(message, "stop_reason", None)
                if not text:
                    # An empty answer is not self-explanatory, and silently
                    # returning "" turns every cause into the same
                    # empty_response. A refusal in particular is the model
                    # declining, not the harness failing, and OpenAILM already
                    # surfaces refusals as text -- match that so the two
                    # providers' records are comparable.
                    refusal = getattr(message, "refusal", None)
                    if refusal:
                        text = f"[refusal] {refusal}"
                        stop_reason = "refusal"
                    elif stop_reason == "refusal":
                        text = "[refusal]"
                    else:
                        log.warning(
                            "Anthropic returned empty text: model=%s stop_reason=%s "
                            "blocks=%r output_tokens=%s",
                            self.model,
                            stop_reason,
                            [type(b).__name__ for b in (message.content or [])],
                            getattr(getattr(message, "usage", None), "output_tokens", None),
                        )

                usage = message.usage
                log.debug(
                    "tokens used: input=%d output=%d",
                    usage.input_tokens if usage else 0,
                    usage.output_tokens if usage else 0,
                )
                return LMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=usage.input_tokens if usage else 0,
                    completion_tokens=usage.output_tokens if usage else 0,
                    stop_reason=stop_reason,
                )
            except _RateLimitError:
                if attempt == self.max_retries - 1:
                    raise
                log.warning("Rate limit hit, retrying in %.1fs (attempt %d/%d)",
                            delay, attempt + 1, self.max_retries)
                time.sleep(delay)
                delay *= 2
