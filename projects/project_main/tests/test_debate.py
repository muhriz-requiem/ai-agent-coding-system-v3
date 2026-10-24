"""Tests for the bull/bear research debate (src/agents/debate.py)."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.agents.debate", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import asyncio

from src.agents.debate import run_debate
from src.agents.models import AnalystReport, DebateRecord


class FakeLLM:
    def __init__(self, response: dict):
        self.response = response
        self.prompts = []

    async def complete_json(self, system, user, model_cls, effort="low"):
        self.prompts.append(user)
        return model_cls.model_validate(self.response)


def _reports() -> dict:
    def report(agent, stance):
        return AnalystReport.model_validate({
            "agent": agent, "ticker": "AAPL", "stance": stance,
            "confidence": 0.6, "thesis": f"{agent} thesis",
            "citations": [f"{agent}=1.0"],
        })

    return {
        "fundamental": report("fundamental", "bullish"),
        "technical": report("technical", "bearish"),
        "sentiment": report("sentiment", "neutral"),
        "macro": report("macro", "bullish"),
    }


def test_returns_debate_record():
    fake = FakeLLM({
        "ticker": "AAPL", "bull_points": ["margins"], "bear_points": ["debt"],
        "winner": "bull", "citations": ["gross_margin=0.41"],
    })
    record = asyncio.run(run_debate("AAPL", _reports(), fake))
    assert isinstance(record, DebateRecord)
    assert record.winner == "bull"
    assert record.ticker == "AAPL"


def test_prompt_contains_both_sides_and_all_theses():
    fake = FakeLLM({
        "ticker": "AAPL", "bull_points": [], "bear_points": [],
        "winner": "undecided", "citations": ["x=1"],
    })
    asyncio.run(run_debate("AAPL", _reports(), fake))
    prompt = fake.prompts[0]
    assert "fundamental thesis" in prompt
    assert "technical thesis" in prompt
    assert "bull" in prompt.lower()
    assert "bear" in prompt.lower()


def test_prompt_carries_ticker():
    fake = FakeLLM({
        "ticker": "AAPL", "bull_points": [], "bear_points": [],
        "winner": "bear", "citations": ["x=1"],
    })
    asyncio.run(run_debate("AAPL", _reports(), fake))
    assert "AAPL" in fake.prompts[0]
