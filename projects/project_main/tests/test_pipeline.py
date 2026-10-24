"""End-to-end offline tests for the pipeline DAG (src/pipeline/graph.py).

Every stage is injected via ``components`` -- the DAG itself must contain
no I/O and no network.  Call-order recording proves the decoupling.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.pipeline.graph", "src.pipeline.state",
               "src.store.decisions", "src.agents.models")
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
from src.agents.models import (AnalystReport, DebateRecord, OrderPlan,
                               RiskDecision)

_CALLS = []


def _components(tickers):
    def fetch_ohlcv(ticker, period="6mo"):
        _CALLS.append(f"ohlcv:{ticker}")
        return pd.Series([100.0, 101.0, 99.0] * 10)

    def fetch_fundamentals(ticker):
        _CALLS.append(f"fundamentals:{ticker}")
        return {"gross_margin": 0.4}

    def fetch_news(ticker):
        _CALLS.append(f"news:{ticker}")
        return [{"title": "headline", "publisher": "p", "url": "u"}]

    def generate_signal(prices, regime=None):
        _CALLS.append("quant")
        return {"signal": "HOLD", "confidence": 0.5, "z_score": 0.1,
                "momentum": 0.0, "reference_price": 100.0}

    def run_analysts(ticker, inputs, client=None):
        _CALLS.append(f"analysts:{ticker}")
        report = AnalystReport.model_validate({
            "agent": "fundamental", "ticker": ticker, "stance": "neutral",
            "confidence": 0.5, "thesis": "flat",
            "citations": ["gross_margin=0.4"]})
        return {"fundamental": report, "technical": report,
                "sentiment": report, "macro": report}

    def run_debate(ticker, reports, client=None):
        _CALLS.append(f"debate:{ticker}")
        return DebateRecord.model_validate({
            "ticker": ticker, "bull_points": [], "bear_points": [],
            "winner": "undecided", "citations": ["gross_margin=0.4"]})

    def evaluate_risk(ticker, reports, debate, quant, account,
                      proposed_position_pct=None, flow_data=None):
        _CALLS.append(f"risk:{ticker}")
        return RiskDecision.model_validate({"ticker": ticker,
                                            "approved": False})

    def build_order_plan(decisions, signals, account, day_trade_dates,
                         as_of=None, positions=None, sector_map=None,
                         regime=None, analyst_reports=None, debates=None,
                         cluster_map=None):
        _CALLS.append("portfolio")
        return []

    def broker_submit(orders):
        _CALLS.append("broker")
        return []

    return {
        "fetch_ohlcv": fetch_ohlcv,
        "fetch_fundamentals": fetch_fundamentals,
        "fetch_news": fetch_news,
        "generate_signal": generate_signal,
        "run_analysts": run_analysts,
        "run_debate": run_debate,
        "evaluate_risk": evaluate_risk,
        "build_order_plan": build_order_plan,
        "broker_submit": broker_submit,
        "account": {"equity": 100_000.0, "max_position_pct": 0.10},
        "day_trade_dates": [],
        "client": None,
    }


def test_run_pipeline_two_tickers_offline():
    global _CALLS
    _CALLS = []
    decisions = asyncio.run(run_pipeline(["FAKE1", "FAKE2"],
                                         _components(["FAKE1", "FAKE2"])))
    assert len(decisions) == 2
    assert {d["ticker"] for d in decisions} == {"FAKE1", "FAKE2"}


def test_ingestion_precedes_analysis():
    global _CALLS
    _CALLS = []
    asyncio.run(run_pipeline(["FAKE1"], _components(["FAKE1"])))
    first_analyst = _CALLS.index("analysts:FAKE1")
    assert _CALLS.index("ohlcv:FAKE1") < first_analyst
    assert _CALLS.index("fundamentals:FAKE1") < first_analyst
    assert _CALLS.index("news:FAKE1") < first_analyst


def test_quant_runs_per_ticker_before_analysts():
    global _CALLS
    _CALLS = []
    asyncio.run(run_pipeline(["FAKE1"], _components(["FAKE1"])))
    assert _CALLS.index("quant") < _CALLS.index("analysts:FAKE1")


def test_broker_submitted_once_as_batch():
    global _CALLS
    _CALLS = []
    asyncio.run(run_pipeline(["FAKE1", "FAKE2"],
                             _components(["FAKE1", "FAKE2"])))
    assert _CALLS.count("broker") == 1
    assert _CALLS.index("portfolio") < _CALLS.index("broker")


def test_decision_contains_veto_and_quant():
    global _CALLS
    _CALLS = []
    decisions = asyncio.run(run_pipeline(["FAKE1"], _components(["FAKE1"])))
    decision = decisions[0]
    assert decision["quant"]["signal"] == "HOLD"
    assert decision["veto"]["approved"] is False
    assert decision["reports"]["fundamental"].ticker == "FAKE1"


def test_orders_routed_to_own_ticker_and_audited():
    global _CALLS
    _CALLS = []
    comps = _components(["FAKE1", "FAKE2"])

    def build_two(decisions, signals, account, day_trade_dates,
                  as_of=None, positions=None, sector_map=None,
                  regime=None, analyst_reports=None, debates=None,
                  cluster_map=None):
        _CALLS.append("portfolio")
        # graph threads broker positions, fundamentals-derived sector
        # labels and the regime snapshot into the batch sizing call.
        assert positions == {}
        assert set(sector_map) == {"FAKE1", "FAKE2"}
        assert sector_map["FAKE1"] is None  # stub fundamentals carry no sector
        assert regime is None  # stub components carry no regime key
        assert set(debates) == {"FAKE1", "FAKE2"}  # item 10: debate records
        assert debates["FAKE1"].winner == "undecided"
        # Stub closes are far too short for clustering (29 returns < 60
        # min observations) -> no clusters (item 11 fail-open).
        assert cluster_map == {}
        return [OrderPlan.model_validate({
            "ticker": "FAKE1", "side": "buy", "quantity": 1.0,
            "limit_price": 100.0, "rationale": "r"})]

    comps["build_order_plan"] = build_two
    decisions = asyncio.run(run_pipeline(["FAKE1", "FAKE2"], comps))

    # Each decision carries only its own orders (audit-log ready).
    assert len(decisions[0]["orders"]) == 1
    assert decisions[0]["orders"][0]["ticker"] == "FAKE1"
    assert isinstance(decisions[0]["orders"][0], dict)
    assert decisions[1]["orders"] == []


def test_regime_snapshot_stamped_into_decisions():
    global _CALLS
    _CALLS = []
    comps = _components(["FAKE1"])
    comps["regime"] = {"regime": "risk_off", "fear_z": 1.4}

    def build_none(*args, **kwargs):
        _CALLS.append("portfolio")
        return []

    comps["build_order_plan"] = build_none
    decisions = asyncio.run(run_pipeline(["FAKE1"], comps))
    assert decisions[0]["regime"] == {"regime": "risk_off", "fear_z": 1.4}


def test_regime_absent_yields_none_in_audit():
    global _CALLS
    _CALLS = []
    decisions = asyncio.run(run_pipeline(["FAKE1"], _components(["FAKE1"])))
    assert decisions[0]["regime"] is None


def test_macro_analyst_receives_regime_snapshot():
    global _CALLS
    _CALLS = []
    comps = _components(["FAKE1"])
    comps["regime"] = {"regime": "neutral"}

    seen = {}

    def run_analysts_spy(ticker, inputs, client=None):
        _CALLS.append(f"analysts:{ticker}")
        seen["macro"] = inputs.get("macro")
        return {}

    comps["run_analysts"] = run_analysts_spy
    asyncio.run(run_pipeline(["FAKE1"], comps))
    assert seen["macro"] == {"regime": "neutral"}


def test_graph_threads_correlation_clusters():
    # Two tickers with identical 130-point closes are perfectly
    # correlated -> graph clusters them and passes the map to sizing.
    global _CALLS
    _CALLS = []
    comps = _components(["FAKE1", "FAKE2"])
    long_series = pd.Series([100.0 + (0.5 if i % 2 else -0.5)
                             for i in range(130)])

    def fetch_long(ticker, period="6mo"):
        _CALLS.append(f"ohlcv:{ticker}")
        return long_series

    comps["fetch_ohlcv"] = fetch_long
    captured = {}

    def build_spy(decisions, signals, account, day_trade_dates,
                  as_of=None, positions=None, sector_map=None,
                  regime=None, analyst_reports=None, debates=None,
                  cluster_map=None):
        _CALLS.append("portfolio")
        captured["cluster_map"] = cluster_map
        return []

    comps["build_order_plan"] = build_spy
    asyncio.run(run_pipeline(["FAKE1", "FAKE2"], comps))
    cluster_map = captured["cluster_map"]
    assert set(cluster_map) == {"FAKE1", "FAKE2"}
    assert cluster_map["FAKE1"] == cluster_map["FAKE2"]
    assert str(cluster_map["FAKE1"]).startswith("corr:")


def test_graph_correlation_disabled_passes_no_map():
    global _CALLS
    _CALLS = []
    comps = _components(["FAKE1", "FAKE2"])
    long_series = pd.Series([100.0 + (0.5 if i % 2 else -0.5)
                             for i in range(130)])
    comps["fetch_ohlcv"] = lambda ticker, period="6mo": long_series
    comps["correlation"] = {"enabled": False}
    captured = {}

    def build_spy(*args, **kwargs):
        _CALLS.append("portfolio")
        captured["cluster_map"] = kwargs.get("cluster_map")
        return []

    comps["build_order_plan"] = build_spy
    asyncio.run(run_pipeline(["FAKE1", "FAKE2"], comps))
    assert captured["cluster_map"] is None
