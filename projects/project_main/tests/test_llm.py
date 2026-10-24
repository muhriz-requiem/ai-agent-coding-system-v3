"""Tests for the tiered LLM client (src/agents/llm.py).

All network behaviour sits behind the ``LLMClient._chat`` method (which
tests monkeypatch) and backoff sleeps behind ``LLMClient._sleep`` -- the
suite never touches the network.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.agents.llm", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import asyncio
import json

import pytest
from pydantic import BaseModel

from src.agents.llm import LLMClient, LLMError
from src.agents.models import AnalystReport


class _FakeClient(LLMClient):
    """LLMClient whose transport and sleeps are recording fakes."""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self.responses = list(responses)
        self.chat_calls = 0
        self.sleeps = []

    async def _chat(self, system, user, effort):
        self.chat_calls += 1
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def _sleep(self, seconds):
        self.sleeps.append(seconds)


def test_complete_returns_text_without_retry():
    client = _FakeClient(["hello"])
    assert asyncio.run(client.complete("sys", "user")) == "hello"
    assert client.chat_calls == 1
    assert client.sleeps == []


def test_complete_retries_then_succeeds():
    client = _FakeClient([Exception("HTTP 429"), Exception("HTTP 503"), "ok"])
    assert asyncio.run(client.complete("sys", "user")) == "ok"
    assert client.chat_calls == 3
    assert len(client.sleeps) == 2  # slept between attempts


def test_complete_raises_llm_error_after_exhaustion():
    client = _FakeClient([Exception("HTTP 500")] * 5)
    with pytest.raises(LLMError):
        asyncio.run(client.complete("sys", "user"))
    assert client.chat_calls == 3  # exactly 3 total attempts


def test_complete_json_parses_wrapped_json():
    payload = json.dumps({
        "agent": "fundamental", "ticker": "AAPL", "stance": "bearish",
        "confidence": 0.6, "thesis": "Leverage rising",
        "citations": ["debt_to_equity=187"],
    })
    client = _FakeClient([f"Here is my analysis:\n{payload}\nDone."])
    report = asyncio.run(
        client.complete_json("sys", "user", AnalystReport))
    assert isinstance(report, AnalystReport)
    assert report.stance == "bearish"


def test_complete_json_invalid_json_raises():
    client = _FakeClient(["not json at all"])
    with pytest.raises(LLMError):
        asyncio.run(client.complete_json("sys", "user", AnalystReport))


def test_complete_json_schema_violation_raises():
    bad = json.dumps({
        "agent": "fundamental", "ticker": "AAPL", "stance": "bullish",
        "confidence": 0.6, "thesis": "no citations here", "citations": [],
    })
    client = _FakeClient([bad])
    with pytest.raises(LLMError):
        asyncio.run(client.complete_json("sys", "user", AnalystReport))


def test_default_model_is_configured():
    client = LLMClient()
    assert client.model  # a model name must be configured by default


class _FakeHttp:
    """Minimal stand-in for the cached openai.AsyncOpenAI handle."""

    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


def test_aclose_closes_and_clears_handle():
    client = LLMClient()
    fake = _FakeHttp()
    client._client = fake
    asyncio.run(client.aclose())
    assert fake.closed == 1
    assert client._client is None
    # Idempotent: the second call is a no-op.
    asyncio.run(client.aclose())
    assert fake.closed == 1


def test_aclose_tolerates_missing_close():
    client = LLMClient()
    client._client = object()  # no .close attribute
    asyncio.run(client.aclose())  # must not raise
    assert client._client is None


def test_aclose_noop_when_never_used():
    client = LLMClient()
    asyncio.run(client.aclose())
    assert client._client is None


def test_aclose_swallows_close_errors():
    class _Broken:
        async def close(self):
            raise RuntimeError("boom")

    client = LLMClient()
    client._client = _Broken()
    asyncio.run(client.aclose())  # must not raise
    assert client._client is None
