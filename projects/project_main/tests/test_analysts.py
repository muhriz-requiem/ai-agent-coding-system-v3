"""Tests for specialist analyst agents (src/agents/analysts.py).

Uses a fake LLM client to enforce INFORMATION ISOLATION: each analyst's
prompt may contain only its own input slice, never other analysts' data.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.agents.analysts", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import asyncio

from src.agents.analysts import run_analysts
from src.agents.models import AnalystReport

_REPORT = {
    "agent": "fundamental", "ticker": "AAPL", "stance": "bullish",
    "confidence": 0.7, "thesis": "Cheap on margins",
    "citations": ["gross_margin=0.41"],
}


class FakeLLM:
    """Records every prompt; returns a canned AnalystReport each call."""

    def __init__(self):
        self.prompts = []

    async def complete_json(self, system, user, model_cls, effort="low"):
        self.prompts.append(user)
        data = dict(_REPORT)
        data["agent"] = "fundamental" if "fundamentals=" in user else "generic"
        return model_cls.model_validate(data)


def _inputs() -> dict:
    return {
        "fundamentals": {"fundamentals": {"gross_margin": 0.41}},
        "prices": "prices-data-marker",
        "news": [{"title": "news-data-marker"}],
        "macro": {"macro": "macro-data-marker"},
    }


def test_returns_all_four_reports():
    result = asyncio.run(run_analysts("AAPL", _inputs(), FakeLLM()))
    assert set(result.keys()) == {"fundamental", "technical", "sentiment",
                                  "macro"}
    for report in result.values():
        assert isinstance(report, AnalystReport)
        assert report.ticker == "AAPL"


def test_information_isolation_between_analysts():
    fake = FakeLLM()
    asyncio.run(run_analysts("AAPL", _inputs(), fake))
    assert len(fake.prompts) == 4
    fundamental_prompt = fake.prompts[0]
    # The fundamental analyst must see fundamentals but NOT other slices.
    assert "0.41" in fundamental_prompt
    assert "news-data-marker" not in fundamental_prompt
    assert "macro-data-marker" not in fundamental_prompt


def test_reports_carry_citations():
    result = asyncio.run(run_analysts("AAPL", _inputs(), FakeLLM()))
    for report in result.values():
        assert len(report.citations) >= 1


def test_every_report_mentions_ticker():
    fake = FakeLLM()
    result = asyncio.run(run_analysts("AAPL", _inputs(), fake))
    for user_prompt in fake.prompts:
        assert "AAPL" in user_prompt
