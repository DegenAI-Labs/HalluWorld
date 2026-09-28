from __future__ import annotations

import logging
import time

from halluworld.lm.base import LM, LMResponse

try:
    from openai import OpenAI as _OpenAIClient, RateLimitError as _RateLimitError, BadRequestError as _BadRequestError
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install openai: pip install openai") from exc

log = logging.getLogger(__name__)


def _assistant_text_from_choice(choice) -> str:
    """Normalize Chat Completions assistant content across SDK / model variants."""
    msg = getattr(choice, "message", None)
    if msg is None:
        return ""

    refusal = getattr(msg, "refusal", None)
    if refusal:
        return f"[refusal] {refusal}"

    content = getattr(msg, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype in ("text", "output_text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            else:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks)
    return str(content)


class OpenAILM(LM):
    """LM implementation backed by the OpenAI Chat Completions API.

    Example::

        import os
        lm = OpenAILM(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o")
        response = lm.query(system="You are an agent...", user="Is there a red key?")
        print(response.text)

    Note:
        Reasoning models (GPT-5 series, o-series) have different parameter requirements:
        - temperature parameter is not supported (automatically disabled)
        - max_completion_tokens is used instead of max_tokens (handled automatically)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_retries: int = 3,
        reasoning_effort: str | None = None,
        base_url: str | None = None,
    ):
        """Initialize OpenAI LM.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-5.4-mini", "o3")
            api_key: OpenAI API key
            temperature: Sampling temperature (0.0-2.0, only for non-reasoning models)
            max_tokens: Maximum tokens to generate (``None`` → 8192 for reasoning models,
                256 for standard models; reasoning models map to ``max_completion_tokens``).
            max_retries: Number of retries on rate limit
            base_url: Override the API endpoint. Any OpenAI-compatible provider
                (xAI, for one) can then reuse this client rather than a thinner
                one -- notably its BadRequestError handling, which turns a
                too-small token budget into error="bad_request" (an exclusion)
                instead of a raised exception or a blank answer scored wrong.
            reasoning_effort: Reasoning depth for reasoning models (``None`` omits the
                parameter and uses the API default). Typical values:
                - "minimal": Fastest, least thorough
                - "low": Quick responses
                - "medium": Balanced
                - "high": Deep reasoning
                - "xhigh": Maximum depth (model-dependent)
        """
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self._client = (
            _OpenAIClient(api_key=api_key, base_url=base_url) if base_url
            else _OpenAIClient(api_key=api_key)
        )

        # Detect reasoning models that don't support temperature
        # GPT-6 / GPT-5 series: gpt-6-astra, gpt-5.4, gpt-5.6-sol, etc.
        # o-series: o1, o1-mini, o1-preview, o3, o3-mini, o3-pro, etc.
        #
        # This list is a guess about names, and it has been wrong: gpt-6-astra
        # matched nothing, so every request sent `max_tokens` and the API
        # rejected all of them. query() therefore also recovers at run time
        # when the API says so, making an unrecognised name cost one wasted
        # request rather than every request in the run.
        self.is_reasoning_model = (
            model.startswith("gpt-6") or
            model.startswith("gpt-5") or
            model.startswith("o1") or
            model.startswith("o3") or
            model.startswith("o4")
        )

        # Reasoning models can spend the entire completion budget on internal
        # reasoning unless the cap is high enough, which surfaces as empty
        # `message.content` with short `max_completion_tokens`.
        if max_tokens is None:
            max_tokens = 8192 if self.is_reasoning_model else 256
        self.max_tokens = int(max_tokens)

        if self.is_reasoning_model:
            if temperature != 0.0:
                log.warning(
                    f"Model '{model}' is a reasoning model that does not support "
                    f"temperature parameter. Ignoring temperature={temperature}."
                )
            self.temperature = None
            # Reasoning models consume thinking tokens inside max_completion_tokens.
            # The default 256 leaves no budget for the actual response after thinking tokens.
            # 4096 is also insufficient for complex chain-of-thought (o3-mini exhausts it on
            # dependency-chain count probes). Use 16000 as the minimum for reasoning models.
            if self.max_tokens < 16000:
                log.debug(
                    f"Reasoning model '{model}': raising max_completion_tokens from "
                    f"{self.max_tokens} to 16000 to accommodate thinking tokens."
                )
                self.max_tokens = 16000
        else:
            self.temperature = temperature
            if reasoning_effort is not None:
                log.warning(
                    f"Model '{model}' is not a reasoning model. "
                    f"Ignoring reasoning_effort={reasoning_effort}."
                )

    def query(self, system: str, user: str, max_tokens: int | None = None) -> LMResponse:
        """Query the API with exponential backoff on rate-limit errors."""
        # For reasoning models self.max_tokens was already raised to 16000 at init.
        # Never let a caller override below that floor (e.g. hardcoded 1024 for count probes).
        if max_tokens is not None and self.is_reasoning_model:
            _max = max(max_tokens, self.max_tokens)
        else:
            _max = max_tokens if max_tokens is not None else self.max_tokens
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                # Build request params - reasoning models need different parameters
                params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }

                # Reasoning models (GPT-5, o-series) use max_completion_tokens
                # Standard models use max_tokens
                if self.is_reasoning_model:
                    params["max_completion_tokens"] = _max
                    # Add reasoning_effort for reasoning models
                    if self.reasoning_effort is not None:
                        params["reasoning_effort"] = self.reasoning_effort
                else:
                    params["max_tokens"] = _max

                # Only add temperature for non-reasoning models
                if self.temperature is not None:
                    params["temperature"] = self.temperature

                try:
                    completion = self._client.chat.completions.create(**params)
                except _BadRequestError as exc:
                    # "Unsupported parameter: 'max_tokens' ... use
                    # 'max_completion_tokens'" means this is a reasoning model
                    # the name check above did not recognise. Switch the
                    # instance over and retry once, so one model naming scheme
                    # we have not seen costs a single request instead of every
                    # request in the run.
                    if self.is_reasoning_model or "max_completion_tokens" not in str(exc):
                        raise
                    log.warning(
                        "Model '%s' rejected max_tokens; treating it as a reasoning "
                        "model and retrying with max_completion_tokens.", self.model
                    )
                    self.is_reasoning_model = True
                    self.temperature = None
                    # Apply the same floor __init__ uses for reasoning models:
                    # thinking tokens come out of this budget, so retrying with
                    # a standard model's 256 returns empty text that would be
                    # scored as a wrong answer rather than recognised as a
                    # starved request.
                    if self.max_tokens < 16000:
                        self.max_tokens = 16000
                    _max = max(_max, self.max_tokens)
                    params.pop("max_tokens", None)
                    params.pop("temperature", None)
                    params["max_completion_tokens"] = _max
                    if self.reasoning_effort is not None:
                        params["reasoning_effort"] = self.reasoning_effort
                    completion = self._client.chat.completions.create(**params)
                choice = completion.choices[0]
                usage = completion.usage
                text = _assistant_text_from_choice(choice).strip()
                # Structural, from the API's own refusal field -- not inferred
                # from the "[refusal]" prefix in the text, which a model could
                # in principle emit itself and which would then be
                # indistinguishable from a real refusal.
                message = getattr(choice, "message", None)
                stop_reason = (
                    "refusal" if getattr(message, "refusal", None)
                    else getattr(choice, "finish_reason", None)
                )
                if not text:
                    log.warning(
                        "OpenAI returned empty assistant text: model=%s finish_reason=%s "
                        "prompt_tokens=%s completion_tokens=%s completion_tokens_details=%r",
                        self.model,
                        getattr(choice, "finish_reason", None),
                        getattr(usage, "prompt_tokens", None) if usage else None,
                        getattr(usage, "completion_tokens", None) if usage else None,
                        getattr(usage, "completion_tokens_details", None) if usage else None,
                    )
                log.debug(
                    "tokens used: prompt=%d completion=%d",
                    usage.prompt_tokens if usage else 0,
                    usage.completion_tokens if usage else 0,
                )
                return LMResponse(
                    stop_reason=stop_reason,
                    text=text,
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
            except _BadRequestError as e:
                # Reasoning models return 400 when max_completion_tokens is too
                # small for the thinking budget. Rather than crashing or retrying
                # with a larger limit (which would give these models special
                # treatment), record the episode as EXCLUDED rather than scored.
                # error="bad_request" tells run_benchmark() to skip the evaluator
                # entirely: an infrastructure failure is not a wrong answer, and
                # scoring it as one (this used to return a blank response that
                # scored 0) silently turned a token-budget problem into a
                # measured hallucination. We can revisit the limit for affected
                # probe types / levels later.
                if "max_tokens" in str(e).lower() and self.is_reasoning_model:
                    log.warning(
                        "max_completion_tokens=%d too small for reasoning model %s "
                        "(probe will be excluded, not scored 0); consider raising "
                        "the limit for this probe type if all reasoning models hit this.",
                        _max, self.model,
                    )
                    return LMResponse(
                        text="",
                        model=self.model,
                        prompt_tokens=0,
                        completion_tokens=0,
                        error="bad_request",
                    )
                raise
