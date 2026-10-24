"""Tiered LLM client with retries and strict JSON validation.

This module is import-safe: importing it never requires OpenAI/OpenRouter
configuration. The network client (``openai.AsyncOpenAI``) is constructed
lazily on first use, and ``LLMClient._chat`` is the ONLY network seam so
tests can monkeypatch it (along with ``_sleep``) to exercise retries,
extraction, and validation paths without any real network access.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

# Settings access is deferred to call time (via _default_model) so that a
# missing/misconfigured settings module can never break importing this file.
try:  # pragma: no cover - import-time tolerance for optional config layout
    from src.config.settings import settings  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    settings = None  # type: ignore[assignment]


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "z-ai/glm-5.3-flash"
DEFAULT_TIER = "fast"
MAX_ATTEMPTS = 3


class LLMError(Exception):
    """Raised on retry exhaustion, unparseable output, or validation failure."""


TModel = TypeVar("TModel", bound=BaseModel)


def _default_model() -> str:
    """Resolve the default model from settings tier 'fast', with a hard fallback."""
    tier_model: Any = None
    try:
        if settings is not None:
            tiers = getattr(settings, "llm_tiers", None)
            if isinstance(tiers, dict):
                tier_model = tiers.get(DEFAULT_TIER)
            if tier_model is None:
                tier = getattr(settings, DEFAULT_TIER, None)
                if tier is not None:
                    tier_model = (
                        tier.get("model")
                        if isinstance(tier, dict)
                        else getattr(tier, "model", None)
                    )
    except Exception:
        tier_model = None
    if isinstance(tier_model, str) and tier_model:
        return tier_model
    return DEFAULT_MODEL


class LLMClient:
    """Thin OpenRouter chat client with bounded retries and strict JSON parsing."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or _default_model()
        self.base_url = base_url
        # Lazy handle: openai.AsyncOpenAI is built on first use only.
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            import openai  # imported lazily so module import needs no openai config

            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url or DEFAULT_BASE_URL,
            )
        return self._client

    # ------------------------------------------------------------------
    # Network seam (the ONLY place that touches the network)
    # ------------------------------------------------------------------
    async def _chat(self, system: str, user: str, effort: str) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            extra_body={"reasoning_effort": effort},
        )
        return response.choices[0].message.content or ""

    async def _sleep(self, seconds: float) -> None:
        """Overridable sleep seam used between retry attempts."""
        await asyncio.sleep(seconds)

    # ------------------------------------------------------------------
    # Retrying completion
    # ------------------------------------------------------------------
    async def complete(self, system: str, user: str, effort: str = "low") -> str:
        """Call ``_chat`` up to MAX_ATTEMPTS times with capped backoff.

        Backoff between attempts is ``min(2 ** attempt, 8)`` seconds, routed
        through ``await self._sleep(...)``. Exactly MAX_ATTEMPTS total calls
        are made; on the final failure an ``LLMError`` is raised.
        """
        last_error: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return await self._chat(system, user, effort)
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last_error = exc
                if attempt < MAX_ATTEMPTS - 1:
                    await self._sleep(min(2 ** attempt, 8))
        raise LLMError(f"LLM completion failed after {MAX_ATTEMPTS} attempts") from last_error

    # ------------------------------------------------------------------
    # Strict JSON completion
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> str:
        """Return the first balanced JSON object embedded in ``text``.

        Scans from the first '{', tracking brace depth while respecting
        string literals and escape characters. Raises ValueError when no
        balanced object exists.
        """
        start = text.find("{")
        if start == -1:
            raise ValueError("no JSON object found in LLM output")

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        raise ValueError("unbalanced JSON object in LLM output")

    async def complete_json(
        self,
        system: str,
        user: str,
        model_cls: Type[TModel],
        effort: str = "low",
    ) -> TModel:
        """Complete, extract the first balanced JSON object, and validate it.

        Any unparseable output or pydantic validation failure is converted
        into ``LLMError``; raw ``JSONDecodeError``/``ValidationError`` never
        escape this method.
        """
        text = await self.complete(system, user, effort)
        try:
            payload = json.loads(self._extract_json(text))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(f"could not parse JSON from LLM output: {exc}") from exc
        try:
            return model_cls.model_validate(payload)
        except ValidationError as exc:
            raise LLMError(f"LLM output failed schema validation: {exc}") from exc

    # ------------------------------------------------------------------
    # Event-loop-safe shutdown
    # ------------------------------------------------------------------
    async def aclose(self) -> None:
        """Close the cached HTTP client and clear the handle.

        Call this on the SAME event loop that ran the completions --
        i.e. inside the ``asyncio.run`` coroutine -- so connection-close
        tasks never outlive their loop (a cached client garbage-collected
        during a later cycle surfaces as ``RuntimeError: Event loop is
        closed`` noise).  Idempotent; a never-used client is a no-op.
        """
        client, self._client = self._client, None
        if client is None:
            return
        close = getattr(client, "close", None)
        if close is None:
            return
        try:
            outcome = close()
            if asyncio.iscoroutine(outcome):
                await outcome
        except Exception:
            pass
