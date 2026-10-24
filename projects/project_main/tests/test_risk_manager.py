"""Tests for the deterministic Risk Manager veto gate.

These rules are HARD-CODED logic (not prompt-based): an LLM narrative may
accompany the decision but can never change its outcome.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.agents.risk_manager", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.agents.models import AnalystReport, DebateRecord
from src.agents.risk_manager import evaluate_risk


def _report(citations=None, stance="bullish") -> AnalystReport:
    data = {
        "agent": "fundamental", "ticker": "AAPL", "stance": stance,
        "confidence": 0.7, "thesis": "t",
    }
    # `citations or default` would swallow the explicit empty list, so the
    # default is applied with an explicit None check instead.
    data["citations"] = citations if citations is not None else ["m=0.41"]
    if citations is not None and not citations:
        # Empty citations are invalid by model contract (min_length=1);
        # build the object without validation so evaluate_risk's veto
        # path can be exercised.
        return AnalystReport.model_construct(**data)
    return AnalystReport.model_validate(data)


def _debate(citations=None) -> DebateRecord:
    data = {
        "ticker": "AAPL", "bull_points": ["b"], "bear_points": ["r"],
        "winner": "bull",
        "citations": citations if citations is not None else ["m=0.41"],
    }
    if citations is not None and not citations:
        return DebateRecord.model_construct(**data)
    return DebateRecord.model_validate(data)


def _account(**overrides) -> dict:
    base = {"equity": 100_000.0, "max_position_pct": 0.10,
            "max_daily_loss_pct": 0.05}
    base.update(overrides)
    return base


def test_approves_clean_thesis_within_limits():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report()},
        debate=_debate(),
        quant={"signal": "HOLD", "confidence": 0.5, "z_score": 0.2},
        account=_account(),
        proposed_position_pct=0.08,
    )
    assert decision.approved is True
    assert decision.veto_reasons == []
    assert decision.quant_override is False


def test_vetoes_missing_report_citations():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report(citations=[])},
        debate=_debate(),
        quant={"signal": "HOLD", "confidence": 0.5},
        account=_account(),
        proposed_position_pct=0.05,
    )
    assert decision.approved is False
    assert any("citation" in reason.lower() for reason in decision.veto_reasons)


def test_vetoes_missing_debate_citations():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report()},
        debate=_debate(citations=[]),
        quant={"signal": "HOLD", "confidence": 0.5},
        account=_account(),
        proposed_position_pct=0.05,
    )
    assert decision.approved is False


def test_quant_sell_override_downgrades_to_block():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report()},
        debate=_debate(),
        quant={"signal": "SELL", "confidence": 0.90, "z_score": 2.4},
        account=_account(),
        proposed_position_pct=0.05,
    )
    assert decision.quant_override is True
    assert decision.approved is False
    assert any("quant" in reason.lower() for reason in decision.veto_reasons)


def test_quant_sell_below_threshold_does_not_override():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report()},
        debate=_debate(),
        quant={"signal": "SELL", "confidence": 0.60, "z_score": 1.6},
        account=_account(),
        proposed_position_pct=0.05,
    )
    assert decision.quant_override is False
    assert decision.approved is True


def test_vetoes_position_over_limit():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report()},
        debate=_debate(),
        quant={"signal": "BUY", "confidence": 0.7},
        account=_account(),
        proposed_position_pct=0.25,
    )
    assert decision.approved is False
    assert any("position" in reason.lower() for reason in decision.veto_reasons)


def test_missing_citation_check_beats_approval_even_bullish():
    decision = evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report(citations=[]),
                 "macro": _report(citations=[])},
        debate=_debate(),
        quant={"signal": "BUY", "confidence": 0.95},
        account=_account(),
        proposed_position_pct=0.01,
    )
    assert decision.approved is False
    assert len(decision.veto_reasons) >= 2
