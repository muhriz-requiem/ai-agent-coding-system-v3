"""Extended risk-manager tests: whale/flow veto rules 4-6.

Pins the BUY-gating invariant (no new rule ever vetoes a SELL), the
sample-size guard, threshold overrides via ``account``, and fail-open on
absent data (src/agents/risk_manager.py).
"""


def _impl_ready() -> bool:
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


def _report() -> AnalystReport:
    return AnalystReport.model_validate({
        "agent": "fundamental", "ticker": "AAPL", "stance": "bullish",
        "confidence": 0.7, "thesis": "t", "citations": ["m=0.41"],
    })


def _debate() -> DebateRecord:
    return DebateRecord.model_validate({
        "ticker": "AAPL", "bull_points": ["b"], "bear_points": ["r"],
        "winner": "bull", "citations": ["m=0.41"],
    })


def _account(**overrides) -> dict:
    base = {"equity": 100_000.0, "max_position_pct": 0.10}
    base.update(overrides)
    return base


_FLOW = {
    "insider": {"status": "ok", "cluster_sells": 3, "cluster_buys": 0,
                "net_insider_flow": -1_000.0},
    "institutional": {"status": "ok", "total_holders": 12,
                      "closed_positions": 5, "new_positions": 1},
    "options_flow": {"status": "ok", "call_put_premium_ratio": 0.20},
}


def _evaluate(quant_signal, flow_data=_FLOW, account=None):
    return evaluate_risk(
        ticker="AAPL",
        reports={"fundamental": _report()},
        debate=_debate(),
        quant={"signal": quant_signal, "confidence": 0.5, "z_score": 0.2},
        account=account or _account(),
        proposed_position_pct=None,
        flow_data=flow_data,
    )


def test_all_rules_fire_on_buy():
    decision = _evaluate("BUY")
    assert decision.approved is False
    reasons = decision.veto_reasons
    assert any("insider cluster sell: 3 insiders selling" in r for r in reasons)
    assert any("institutional exodus" in r for r in reasons)
    assert any("extreme put skew: call/put premium ratio 0.20" in r for r in reasons)


def test_no_rule_ever_vetoes_a_sell():
    decision = _evaluate("SELL")
    assert decision.approved is True
    assert decision.veto_reasons == []


def test_no_rule_fires_on_hold():
    decision = _evaluate("HOLD")
    assert decision.approved is True


def test_lowercase_buy_signal_is_gated():
    decision = _evaluate("buy")
    assert decision.approved is False


def test_thresholds_coming_from_account():
    # Loosen every threshold: none of the flow data should veto now.
    account = _account(cluster_insider_threshold=10,
                       flow_veto_min_holders=100,
                       uoa_put_skew_veto=0.10)
    decision = _evaluate("BUY", account=account)
    assert decision.approved is True


def test_cluster_threshold_boundary():
    # Default threshold 3: exactly 3 cluster sells vetoes (>=).
    decision = _evaluate("BUY", flow_data={
        "insider": {"cluster_sells": 3}})
    assert decision.approved is False
    decision = _evaluate("BUY", flow_data={
        "insider": {"cluster_sells": 2}})
    assert decision.approved is True


def test_exodus_requires_min_holders():
    # 9 holders < default guard 10: closed 5 > new 0 * 2 but no veto.
    decision = _evaluate("BUY", flow_data={
        "institutional": {"total_holders": 9, "closed_positions": 5,
                          "new_positions": 0}})
    assert decision.approved is True
    # At the guard boundary (10 holders) the same shape vetoes.
    decision = _evaluate("BUY", flow_data={
        "institutional": {"total_holders": 10, "closed_positions": 5,
                          "new_positions": 0}})
    assert decision.approved is False
    assert any("institutional exodus" in r for r in decision.veto_reasons)


def test_exodus_needs_closed_greater_than_twice_new():
    decision = _evaluate("BUY", flow_data={
        "institutional": {"total_holders": 10, "closed_positions": 4,
                          "new_positions": 2}})
    assert decision.approved is True  # 4 is not > 2*2
    decision = _evaluate("BUY", flow_data={
        "institutional": {"total_holders": 10, "closed_positions": 5,
                          "new_positions": 2}})
    assert decision.approved is False  # 5 > 2*2


def test_put_skew_strictly_below_threshold():
    decision = _evaluate("BUY", flow_data={
        "options_flow": {"call_put_premium_ratio": 0.30}})
    assert decision.approved is True  # equal is not strictly below
    decision = _evaluate("BUY", flow_data={
        "options_flow": {"call_put_premium_ratio": 0.29}})
    assert decision.approved is False


def test_put_skew_skipped_when_ratio_missing():
    decision = _evaluate("BUY", flow_data={
        "options_flow": {"call_put_premium_ratio": None}})
    assert decision.approved is True


def test_flow_data_none_fails_open():
    decision = _evaluate("BUY", flow_data=None)
    assert decision.approved is True


def test_empty_flow_data_fails_open():
    decision = _evaluate("BUY", flow_data={})
    assert decision.approved is True


def test_partial_flow_data_fails_open_per_slice():
    decision = _evaluate("BUY", flow_data={"insider": {"status": "ok"}})
    assert decision.approved is True  # missing aggregates -> no vetoes


def test_garbage_values_fail_open():
    decision = _evaluate("BUY", flow_data={
        "insider": {"cluster_sells": "lots"},
        "institutional": {"total_holders": "many", "closed_positions": "?",
                          "new_positions": None},
        "options_flow": {"call_put_premium_ratio": "bearish"},
    })
    assert decision.approved is True
