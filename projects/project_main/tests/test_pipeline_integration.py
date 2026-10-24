"""End-to-end pipeline integration tests for the whale/flow wiring
(src/pipeline/graph.py).

Pins:
  * flow envelopes thread through process_ticker into ``evaluate_risk``;
  * only reports backed by ``status == "ok"`` fetches feed the sizing
    ``analyst_reports`` mapping;
  * flags-off parity: absent fetch components leave the slices at None,
    ``flow_data`` at None and sizing reports empty;
  * a successful dark-pool fetch merges into the macro slice without
    touching the regime snapshot.
"""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.pipeline.graph", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import asyncio

import pandas as pd

from src.pipeline.graph import run_pipeline
from src.agents.models import AnalystReport, DebateRecord, RiskDecision

_FLOW_ANALYSTS = ("insider", "institutional", "options_flow")


def _report(name, stance="bullish", confidence=0.9) -> AnalystReport:
    return AnalystReport.model_validate({
        "agent": name, "ticker": "FAKE1", "stance": stance,
        "confidence": confidence, "thesis": "t", "citations": ["ev"],
    })


def _components(captured, flow_envelopes=None, dark_pool=None):
    """Full fake component graph with capture dicts for the batch calls."""

    async def fetch_insider(ticker):
        return (flow_envelopes or {}).get("insider")

    async def fetch_institutional(ticker):
        return (flow_envelopes or {}).get("institutional")

    async def fetch_options_flow(ticker):
        return (flow_envelopes or {}).get("options_flow")

    async def fetch_dark_pool(ticker):
        return dark_pool

    def run_analysts(ticker, inputs, client=None):
        captured.setdefault("inputs", {})[ticker] = inputs
        names = ["fundamental", "technical", "sentiment", "macro"]
        for flow_name in _FLOW_ANALYSTS:
            if inputs.get(flow_name) is not None:  # dispatch gating
                names.append(flow_name)
        return {name: _report(name) for name in names}

    def run_debate(ticker, reports, client=None):
        return DebateRecord.model_validate({
            "ticker": ticker, "bull_points": [], "bear_points": [],
            "winner": "undecided", "citations": ["ev"]})

    def evaluate_risk(ticker, reports, debate, quant, account,
                      proposed_position_pct=None, flow_data=None):
        captured["risk_kwargs"] = {
            "flow_data": flow_data, "proposed_position_pct": proposed_position_pct}
        return RiskDecision.model_validate({"ticker": ticker, "approved": True})

    def build_order_plan(decisions, signals, account, day_trade_dates,
                         as_of=None, positions=None, sector_map=None,
                         regime=None, analyst_reports=None, debates=None,
                         cluster_map=None):
        captured["plan_kwargs"] = {
            "analyst_reports": analyst_reports, "positions": positions,
            "sector_map": sector_map, "regime": regime, "debates": debates,
            "cluster_map": cluster_map}
        return []

    def broker_submit(orders):
        return []

    components = {
        "fetch_ohlcv": lambda ticker, period="6mo": pd.Series(
            [100.0, 101.0, 99.0] * 10),
        "fetch_fundamentals": lambda ticker: {"sector": "Technology"},
        "fetch_news": lambda ticker: [],
        "generate_signal": lambda prices, regime=None: {
            "signal": "BUY", "confidence": 0.9, "z_score": 1.0,
            "momentum": 0.0, "reference_price": 100.0},
        "run_analysts": run_analysts,
        "run_debate": run_debate,
        "evaluate_risk": evaluate_risk,
        "build_order_plan": build_order_plan,
        "broker_submit": broker_submit,
        "account": {"equity": 100_000.0, "max_position_pct": 0.10},
        "day_trade_dates": [],
        "client": None,
    }
    if flow_envelopes is not None:
        # Register only the components whose envelope is configured, the
        # same way cli._components registers flag-gated closures.
        if "insider" in flow_envelopes:
            components["fetch_insider"] = fetch_insider
        if "institutional" in flow_envelopes:
            components["fetch_institutional"] = fetch_institutional
        if "options_flow" in flow_envelopes:
            components["fetch_options_flow"] = fetch_options_flow
    if dark_pool is not None:
        components["fetch_dark_pool"] = fetch_dark_pool
    return components


_OK_ENVELOPES = {
    "insider": {"status": "ok", "cluster_sells": 0},
    "institutional": {"status": "ok", "total_holders": 10},
    "options_flow": {"status": "ok", "call_put_premium_ratio": 1.2},
}


def test_flow_envelopes_thread_to_risk_and_sizing():
    captured = {}
    envelopes = dict(_OK_ENVELOPES)
    envelopes["options_flow"] = {"status": "error", "reason": "boom"}
    decisions = asyncio.run(run_pipeline(["FAKE1"], _components(captured, envelopes)))
    assert len(decisions) == 1

    # Risk saw every fetched envelope (even the failed one -- the veto
    # rules fail open per slice).
    assert captured["risk_kwargs"]["flow_data"] == envelopes

    # Sizing only saw reports backed by a successful fetch.
    sizing_reports = captured["plan_kwargs"]["analyst_reports"]["FAKE1"]
    assert set(sizing_reports) == {"insider", "institutional"}

    # The analysts received their isolated slices.
    inputs = captured["inputs"]["FAKE1"]
    assert inputs["insider"] == envelopes["insider"]
    assert inputs["institutional"] == envelopes["institutional"]
    assert inputs["options_flow"] == envelopes["options_flow"]


def test_flags_off_parity_slices_absent():
    captured = {}
    decisions = asyncio.run(run_pipeline(["FAKE1"], _components(captured)))
    assert len(decisions) == 1

    inputs = captured["inputs"]["FAKE1"]
    for name in _FLOW_ANALYSTS:
        assert inputs.get(name) is None  # slice present-but-None -> gated
    assert captured["risk_kwargs"]["flow_data"] is None
    assert captured["plan_kwargs"]["analyst_reports"] == {"FAKE1": {}}
    # Macro slice unchanged (raw regime, no dark-pool merge).
    assert inputs["macro"] is None


def test_partial_flags_dispatch_only_present_slices():
    captured = {}
    envelopes = {"insider": _OK_ENVELOPES["insider"]}
    asyncio.run(run_pipeline(["FAKE1"], _components(captured, envelopes)))
    inputs = captured["inputs"]["FAKE1"]
    assert inputs["insider"]["status"] == "ok"
    assert inputs.get("institutional") is None
    assert inputs.get("options_flow") is None
    assert captured["risk_kwargs"]["flow_data"] == envelopes
    assert set(captured["plan_kwargs"]["analyst_reports"]["FAKE1"]) == {"insider"}


def test_dark_pool_ok_merges_into_macro_slice():
    captured = {}
    dp = {"status": "ok", "dp_pct_latest": 0.42, "citations": ["FINRA ATS: ..."]}
    asyncio.run(run_pipeline(["FAKE1"], _components(captured, dark_pool=dp)))
    macro = captured["inputs"]["FAKE1"]["macro"]
    assert macro == {"regime": None, "dark_pool": dp}


def test_dark_pool_error_leaves_macro_slice_unchanged():
    captured = {}
    dp = {"status": "error", "reason": "no key", "citations": []}
    asyncio.run(run_pipeline(["FAKE1"], _components(captured, dark_pool=dp)))
    assert captured["inputs"]["FAKE1"]["macro"] is None  # raw regime (None)
