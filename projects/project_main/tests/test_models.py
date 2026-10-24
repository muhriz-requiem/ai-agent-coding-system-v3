"""Tests for the strict-JSON LLM message models (src/agents/models.py)."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.agents.models",)
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import pytest
from pydantic import ValidationError

from src.agents.models import (AnalystReport, DebateRecord, OrderPlan,
                               RiskDecision)


def _report(**overrides) -> dict:
    base = {
        "agent": "fundamental",
        "ticker": "AAPL",
        "stance": "bullish",
        "confidence": 0.7,
        "thesis": "Undervalued on margins",
        "citations": ["gross_margin=0.41 (10-K FY25)"],
    }
    base.update(overrides)
    return base


def test_analyst_report_valid():
    report = AnalystReport.model_validate(_report())
    assert report.stance == "bullish"
    assert report.citations == ["gross_margin=0.41 (10-K FY25)"]


def test_analyst_report_accepts_common_llm_aliases():
    report = AnalystReport.model_validate({
        "analyst": "technical", "ticker": "AAPL", "stance": "neutral",
        "confidence": 0.5, "summary": "flat", "citations": ["close=12.0"],
    })
    assert report.agent == "technical"  # canonicalized on validation
    assert report.thesis == "flat"
    assert report.citations == ["close=12.0"]  # citation gate unchanged


def test_analyst_report_alias_still_requires_citations():
    with pytest.raises(ValidationError):
        AnalystReport.model_validate({
            "analyst": "technical", "ticker": "AAPL", "stance": "neutral",
            "confidence": 0.5, "summary": "flat", "citations": [],
        })


def test_analyst_report_requires_citations():
    with pytest.raises(ValidationError):
        AnalystReport.model_validate(_report(citations=[]))


def test_analyst_report_coerces_dict_citations():
    report = AnalystReport.model_validate(_report(citations=[
        {"claim": "gross_margin=0.41 (10-K FY25)", "source": "10-K FY25"},
    ]))
    assert report.citations == ["gross_margin=0.41 (10-K FY25) [10-K FY25]"]


def test_analyst_report_coerces_claim_with_observed_at_keeps_sentinel_text():
    sentinel = "NO INSIDER FILINGS: form4=[], form144=[] (window: 90d)"
    report = AnalystReport.model_validate(_report(citations=[
        {"claim": sentinel, "observed_at": "2026-09-07T22:48:59.373998+00:00"},
    ]))
    assert report.citations == [sentinel]


def test_analyst_report_coerces_unknown_dict_citation_to_json():
    report = AnalystReport.model_validate(_report(citations=[{"foo": "bar"}]))
    assert report.citations == ['{"foo": "bar"}']


def test_analyst_report_wraps_single_string_citation():
    report = AnalystReport.model_validate(_report(citations="close=12.0"))
    assert report.citations == ["close=12.0"]


def test_analyst_report_coerces_nonstring_citation():
    report = AnalystReport.model_validate(_report(citations=[41]))
    assert report.citations == ["41"]


def test_analyst_report_still_rejects_empty_or_blank_citations():
    with pytest.raises(ValidationError):
        AnalystReport.model_validate(_report(citations=[]))
    with pytest.raises(ValidationError):
        AnalystReport.model_validate(_report(citations="   "))


def test_analyst_report_stance_is_limited():
    with pytest.raises(ValidationError):
        AnalystReport.model_validate(_report(stance="bullish-ish"))


def test_analyst_report_confidence_bounded():
    with pytest.raises(ValidationError):
        AnalystReport.model_validate(_report(confidence=1.5))
    with pytest.raises(ValidationError):
        AnalystReport.model_validate(_report(confidence=-0.1))


def test_debate_record_valid():
    record = DebateRecord.model_validate({
        "ticker": "AAPL",
        "bull_points": ["margin expansion"],
        "bear_points": ["multiple compression"],
        "winner": "bull",
        "citations": ["operating_margin=0.29"],
    })
    assert record.winner == "bull"
    with pytest.raises(ValidationError):
        DebateRecord.model_validate({
            "ticker": "AAPL", "bull_points": [], "bear_points": [],
            "winner": "tie", "citations": ["x"],
        })


def test_debate_record_coerces_dict_lists():
    record = DebateRecord.model_validate({
        "ticker": "AAPL",
        "bull_points": [{"claim": "margin expansion", "source": "10-K FY25"}],
        "bear_points": [{"claim": "multiple compression"}],
        "winner": "bear",
        "citations": [{"claim": "operating_margin=0.29"}],
    })
    assert record.bull_points == ["margin expansion [10-K FY25]"]
    assert record.bear_points == ["multiple compression"]
    assert record.citations == ["operating_margin=0.29"]


def test_debate_record_empty_points_accepted_but_empty_citations_rejected():
    record = DebateRecord.model_validate({
        "ticker": "AAPL", "bull_points": [], "bear_points": [],
        "winner": "bull", "citations": ["x"],
    })
    assert record.bull_points == []
    with pytest.raises(ValidationError):
        DebateRecord.model_validate({
            "ticker": "AAPL", "bull_points": [], "bear_points": [],
            "winner": "bull", "citations": [],
        })


def test_risk_decision_defaults():
    decision = RiskDecision.model_validate({"ticker": "AAPL",
                                            "approved": False})
    assert decision.veto_reasons == []
    assert decision.quant_override is False
    assert decision.narrative == ""


def test_order_plan_valid_and_sides():
    plan = OrderPlan.model_validate({
        "ticker": "AAPL", "side": "buy", "quantity": 10,
        "limit_price": 227.5, "rationale": "z-score mean reversion",
    })
    assert plan.quantity == 10
    with pytest.raises(ValidationError):
        OrderPlan.model_validate({
            "ticker": "AAPL", "side": "market", "quantity": 10,
            "limit_price": 227.5, "rationale": "not allowed",
        })
    with pytest.raises(ValidationError):
        OrderPlan.model_validate({
            "ticker": "AAPL", "side": "buy", "quantity": 0,
            "limit_price": 227.5, "rationale": "qty must be positive",
        })


def test_order_plan_fractional_quantity():
    plan = OrderPlan.model_validate({
        "ticker": "AAPL", "side": "buy", "quantity": 0.4,
        "limit_price": 250.0, "rationale": "fractional test",
    })
    assert plan.quantity == 0.4
    # zero still invalid
    with pytest.raises(ValidationError):
        OrderPlan.model_validate({
            "ticker": "AAPL", "side": "buy", "quantity": 0.0,
            "limit_price": 250.0, "rationale": "zero not allowed",
        })
    # negative invalid
    with pytest.raises(ValidationError):
        OrderPlan.model_validate({
            "ticker": "AAPL", "side": "buy", "quantity": -1.0,
            "limit_price": 250.0, "rationale": "negative not allowed",
        })
