"""Tests for the stance-aware flow-conviction sizing factor
(src/agents/portfolio_manager.py).

Pins: directional blend (a bearish high-confidence report must not
inflate a BUY), scale-down-only clamping, weight renormalization over
present reports, duck-typed report objects/dicts, and the rationale
parity rule (no ``flow x`` citation when no flow reports contributed).
"""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.agents.portfolio_manager", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import pytest

from src.agents.models import AnalystReport, RiskDecision
from src.agents.portfolio_manager import (
    FLOW_WEIGHTS,
    _flow_conviction_multiplier,
    build_order_plan,
)

_ACCOUNT = {"equity": 100_000.0, "max_position_pct": 0.10}


def _report(stance, confidence) -> AnalystReport:
    return AnalystReport.model_validate({
        "agent": "x", "ticker": "AAPL", "stance": stance,
        "confidence": confidence, "thesis": "t", "citations": ["e"],
    })


def _reports(stance, confidence) -> dict:
    return {name: _report(stance, confidence) for name in FLOW_WEIGHTS}


def _decision(ticker="AAPL") -> RiskDecision:
    return RiskDecision.model_validate({"ticker": ticker, "approved": True})


def _signals(price=250.0, signal="BUY") -> dict:
    return {"AAPL": {"signal": signal, "confidence": 0.95,
                     "reference_price": price}}


# ---------------------------------------------------------------------------
# _flow_conviction_multiplier
# ---------------------------------------------------------------------------


def test_agreeing_reports_scale_down_proportionally():
    # All bullish 0.9 vs BUY: blended 0.9 -> 0.9/0.95.
    assert _flow_conviction_multiplier(_reports("bullish", 0.9), "buy") == \
        pytest.approx(0.9 / 0.95)


def test_opposed_reports_hit_the_scale_down_floor():
    # All bearish 0.9 vs BUY: 0.1 blended -> 0.105, clamped to 0.25.
    assert _flow_conviction_multiplier(_reports("bearish", 0.9), "buy") == 0.25


def test_neutral_reports_score_half():
    assert _flow_conviction_multiplier(_reports("neutral", 0.9), "buy") == \
        pytest.approx(0.5 / 0.95)


def test_direction_is_signal_relative():
    # Bearish reports AGREE with a SELL signal -> full conviction weight.
    assert _flow_conviction_multiplier(_reports("bearish", 0.9), "sell") == \
        pytest.approx(0.9 / 0.95)
    # Bullish reports OPPOSE a SELL signal -> floor.
    assert _flow_conviction_multiplier(_reports("bullish", 0.9), "sell") == 0.25


def test_never_scales_up():
    # Perfect bullish agreement yields exactly 1.0, never more.
    assert _flow_conviction_multiplier(_reports("bullish", 1.0), "buy") == 1.0


def test_weights_renormalize_over_present_reports():
    # Only options_flow (0.4) present: blended == its own score.
    reports = {"options_flow": _report("bullish", 0.95)}
    assert _flow_conviction_multiplier(reports, "buy") == pytest.approx(1.0)
    reports = {"options_flow": _report("bearish", 0.9)}
    assert _flow_conviction_multiplier(reports, "buy") == 0.25


def test_mixed_stances_blend_by_weight():
    reports = {
        "insider": _report("bullish", 1.0),       # 0.3 * 1.0
        "institutional": _report("bearish", 1.0),  # 0.3 * 0.0
        "options_flow": _report("bullish", 1.0),   # 0.4 * 1.0
    }
    assert _flow_conviction_multiplier(reports, "buy") == \
        pytest.approx(0.7 / 0.95)


def test_no_reports_is_noop():
    assert _flow_conviction_multiplier({}, "buy") == 1.0
    assert _flow_conviction_multiplier(None, "buy") == 1.0
    assert _flow_conviction_multiplier({"insider": None}, "buy") == 1.0


def test_dict_shaped_reports_are_accepted():
    reports = {
        name: {"stance": "bullish", "confidence": 0.9}
        for name in FLOW_WEIGHTS
    }
    assert _flow_conviction_multiplier(reports, "buy") == \
        pytest.approx(0.9 / 0.95)


def test_invalid_confidence_fails_open_to_neutral_like_scoring():
    reports = {name: {"stance": "bullish", "confidence": "high"}
               for name in FLOW_WEIGHTS}
    # confidence -> 0.0 (agreeing) -> blended 0.0 -> floor 0.25.
    assert _flow_conviction_multiplier(reports, "buy") == 0.25


# ---------------------------------------------------------------------------
# build_order_plan integration
# ---------------------------------------------------------------------------


def test_flow_reports_shrink_buy_and_cite_factor():
    orders = build_order_plan(
        decisions=[_decision()],
        signals=_signals(),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        analyst_reports={"AAPL": _reports("bearish", 0.9)},
    )
    assert len(orders) == 1
    # budget = 100k * 0.10 * 0.25 = 2500 -> 10 shares @ 250.
    assert orders[0].quantity == pytest.approx(10.0)
    assert "flow x0.25" in orders[0].rationale


def test_no_flow_reports_keeps_sizing_and_rationale_unchanged():
    orders = build_order_plan(
        decisions=[_decision()],
        signals=_signals(),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1
    assert orders[0].quantity == 40  # 100k * 0.10 / 250
    assert "flow x" not in orders[0].rationale  # flags-off parity


def test_flow_never_inflates_a_buy():
    orders = build_order_plan(
        decisions=[_decision()],
        signals=_signals(),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        analyst_reports={"AAPL": _reports("bullish", 1.0)},
    )
    assert orders[0].quantity == 40  # capped at 1.0


def test_flow_applies_to_sell_budget_before_held_cap():
    orders = build_order_plan(
        decisions=[_decision()],
        signals=_signals(signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 40},
        analyst_reports={"AAPL": _reports("bearish", 0.9)},
    )
    assert len(orders) == 1
    # budget = 100k * 0.10 * (0.9/0.95) -> qty = floor(.../250, 6dp) < 40 held.
    expected = int((100_000.0 * 0.10 * (0.9 / 0.95) / 250.0) * 1e6) / 1e6
    assert orders[0].quantity == pytest.approx(expected)
    assert orders[0].side == "sell"


def test_ok_status_only_reports_feed_sizing():
    # graph.py filters to status=="ok" envelopes before calling; simulate a
    # single usable report set here to keep the contract explicit.
    orders = build_order_plan(
        decisions=[_decision()],
        signals=_signals(),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        analyst_reports={"AAPL": {"options_flow": _report("bearish", 0.9)}},
    )
    assert orders[0].quantity == pytest.approx(10.0)  # renormalized floor
