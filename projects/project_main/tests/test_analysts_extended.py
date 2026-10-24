"""Extended analyst-crew tests: whale/flow dispatch gating, sentinel
rendering, and per-analyst failure isolation (src/agents/analysts.py).
"""


def _impl_ready() -> bool:
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

from src.agents.analysts import ANALYSTS, OPTIONAL_ANALYSTS, _DATA_LABELS, run_analysts
from src.agents.models import AnalystReport

_LABEL_TO_NAME = {f"{label} (JSON)": name for name, label in _DATA_LABELS.items()}


class FakeLLM:
    """Records every prompt; returns a per-analyst report (or raises)."""

    def __init__(self, fail_for=()):
        self.prompts = []
        self.systems = []
        self.fail_for = set(fail_for)

    async def complete_json(self, system, user, model_cls, effort="low"):
        self.prompts.append(user)
        self.systems.append(system)
        name = "generic"
        for label, candidate in _LABEL_TO_NAME.items():
            if label in user:
                name = candidate
                break
        if name in self.fail_for:
            raise RuntimeError("LLM exploded")
        return model_cls.model_validate({
            "agent": name, "ticker": "AAPL", "stance": "neutral",
            "confidence": 0.5, "thesis": "flat",
            "citations": ["evidence-marker"],
        })


def _core_inputs() -> dict:
    return {
        "fundamentals": {"gross_margin": 0.41},
        "prices": "prices-marker",
        "news": [{"title": "news-marker"}],
        "macro": {"macro-marker": True},
    }


def _insider_envelope():
    return {"status": "no_data", "ticker": "AAPL", "window_days": 90,
            "form4": [], "form144": [], "cluster_buys": 0,
            "cluster_sells": 0, "net_insider_flow": 0.0, "citations": []}


def _institutional_envelope():
    return {"status": "ok", "ticker": "AAPL",
            "top_holders": [{"manager": "Citadel", "cik": "1", "shares": 10,
                             "value": 20.0, "pct_outstanding": 0.1,
                             "change_qoq": 0.5}],
            "total_holders": 1, "total_institutional_pct": 1.0,
            "new_positions": 0, "closed_positions": 0,
            "increased_positions": 1, "decreased_positions": 0,
            "net_flow": None, "whale_alerts": [],
            "citations": ["13F: Citadel holds 10 shares ($20)"]}


def _options_envelope():
    return {"status": "ok", "ticker": "AAPL",
            "unusual_contracts": [{"type": "call", "strike": 180.0,
                                   "expiry": "2026-09-20", "dte": 13,
                                   "volume": 45000.0, "oi": 8000.0,
                                   "vol_oi_ratio": 5.6, "premium": 2.1e6,
                                   "iv": 0.35, "delta": None, "side": "buy"}],
            "call_put_premium_ratio": None, "total_unusual_premium": 2.1e6,
            "top_signal": None, "scanned_expiries": 1,
            "citations": ["UOA: call 180 2026-09-20 vol 45,000/oi 8,000"]}


def _all_inputs() -> dict:
    inputs = _core_inputs()
    inputs.update({
        "insider": _insider_envelope(),
        "institutional": _institutional_envelope(),
        "options_flow": _options_envelope(),
    })
    return inputs


def _prompt_for(fake, name):
    label = f"{_DATA_LABELS[name]} (JSON):"
    matches = [p for p in fake.prompts if label in p]
    assert len(matches) == 1
    return matches[0]


def test_seven_reports_when_all_slices_present():
    fake = FakeLLM()
    result = asyncio.run(run_analysts("AAPL", _all_inputs(), fake))
    assert set(result.keys()) == set(ANALYSTS)
    assert len(fake.prompts) == len(ANALYSTS) == 7
    for report in result.values():
        assert isinstance(report, AnalystReport)
        assert len(report.citations) >= 1


def test_four_reports_when_flow_slices_absent():
    fake = FakeLLM()
    result = asyncio.run(run_analysts("AAPL", _core_inputs(), fake))
    assert set(result.keys()) == {"fundamental", "technical", "sentiment", "macro"}
    assert len(fake.prompts) == 4


def test_dispatch_gating_is_per_slice():
    fake = FakeLLM()
    inputs = _core_inputs()
    inputs["insider"] = _insider_envelope()
    result = asyncio.run(run_analysts("AAPL", inputs, fake))
    assert set(result.keys()) == {"fundamental", "technical", "sentiment",
                                  "macro", "insider"}
    assert len(fake.prompts) == 5


def test_sentinel_line_rendered_for_no_data():
    fake = FakeLLM()
    asyncio.run(run_analysts("AAPL", _all_inputs(), fake))
    insider_prompt = _prompt_for(fake, "insider")
    assert "NO INSIDER FILINGS: form4=[], form144=[] (window: 90d)" in insider_prompt


def test_optional_analyst_prompts_stay_isolated():
    fake = FakeLLM()
    asyncio.run(run_analysts("AAPL", _all_inputs(), fake))
    insider_prompt = _prompt_for(fake, "insider")
    assert "Citadel" not in insider_prompt        # institutional data
    assert "gross_margin" not in insider_prompt   # fundamentals data
    assert "prices-marker" not in insider_prompt  # technical data
    options_prompt = _prompt_for(fake, "options_flow")
    assert "Citadel" not in options_prompt
    assert "NO INSIDER FILINGS" not in options_prompt
    institutional_prompt = _prompt_for(fake, "institutional")
    assert "Citadel" in institutional_prompt
    assert "NO INSIDER FILINGS" not in institutional_prompt


def test_size_capped_serializers():
    fake = FakeLLM()
    inputs = _all_inputs()
    inputs["institutional"]["top_holders"] = [
        {"manager": f"M{i}", "shares": i, "value": float(i)} for i in range(30)
    ]
    asyncio.run(run_analysts("AAPL", inputs, fake))
    institutional_prompt = _prompt_for(fake, "institutional")
    assert "M29" in institutional_prompt  # largest value kept
    assert "M0" not in institutional_prompt  # smallest value dropped
    assert "M10" not in institutional_prompt  # capped at 10 rows


def test_system_prompt_has_sentinel_rule_and_no_empty_citation_instruction():
    fake = FakeLLM()
    asyncio.run(run_analysts("AAPL", _all_inputs(), fake))
    system = fake.systems[0]
    assert "NO-DATA sentinel" in system
    assert "never return an empty `citations` list" in system
    assert "and return an empty `citations` list" not in system


def test_failure_isolation_produces_neutral_placeholder():
    fake = FakeLLM(fail_for={"technical"})
    result = asyncio.run(run_analysts("AAPL", _all_inputs(), fake))
    assert set(result.keys()) == set(ANALYSTS)  # batch survived
    placeholder = result["technical"]
    assert placeholder.stance == "neutral"
    assert placeholder.confidence == 0.0
    assert placeholder.citations == ["PRICE SERIES: unavailable this run"]
    # The other analysts completed normally.
    assert result["fundamental"].citations == ["evidence-marker"]


def test_all_analysts_failing_still_returns_reports():
    fake = FakeLLM(fail_for=set(ANALYSTS))
    result = asyncio.run(run_analysts("AAPL", _core_inputs(), fake))
    assert set(result.keys()) == {"fundamental", "technical", "sentiment", "macro"}
    for report in result.values():
        assert report.stance == "neutral"
        assert report.confidence == 0.0
        assert len(report.citations) == 1
        assert report.citations[0].endswith("unavailable this run")


def test_optional_analyst_failure_placeholder():
    fake = FakeLLM(fail_for={"options_flow"})
    result = asyncio.run(run_analysts("AAPL", _all_inputs(), fake))
    placeholder = result["options_flow"]
    assert placeholder.citations == ["OPTIONS FLOW: unavailable this run"]
    assert result["insider"].citations == ["evidence-marker"]
