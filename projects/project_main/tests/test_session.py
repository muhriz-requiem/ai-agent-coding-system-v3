"""Tests for the session loop discipline (src/pipeline/session.py,
src/store/session.py) and the cheap-cycle injection in graph.py.

All pieces are offline: SQLite via tmp_path, fake components, no
network, no LLM, no broker I/O.
"""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.pipeline.session", "src.store.session",
               "src.pipeline.graph", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.pipeline.session import (SubmissionGate, order_notional,
                                  order_side, order_ticker, seconds_until)
from src.store.session import SessionStore


# ---------------------------------------------------------------------------
# order helpers
# ---------------------------------------------------------------------------

def test_order_notional_from_dict_and_object():
    as_dict = {"ticker": "AAPL", "side": "buy", "quantity": 10,
               "limit_price": 227.5}
    class _Order:
        ticker = "AAPL"
        side = "sell"
        quantity = 2.5
        limit_price = 100.0
    assert order_notional(as_dict) == pytest.approx(2275.0)
    assert order_notional(_Order()) == pytest.approx(250.0)


def test_order_notional_junk_is_zero():
    assert order_notional({"quantity": "junk", "limit_price": 10}) == 0.0
    assert order_notional({"quantity": 10}) == 0.0
    assert order_notional(None) == 0.0
    assert order_notional({"quantity": -1, "limit_price": 10}) == 0.0


def test_order_ticker_side():
    assert order_ticker({"ticker": "aapl"}) == "AAPL"
    assert order_side({"side": "BUY"}) == "buy"
    assert order_ticker(None) == ""


# ---------------------------------------------------------------------------
# seconds_until (Alpaca clock parsing)
# ---------------------------------------------------------------------------

def test_seconds_until_future_timestamps():
    now = datetime(2026, 9, 8, 13, 0, tzinfo=timezone.utc)
    zulu = "2026-09-08T14:00:00Z"
    offset = "2026-09-08T10:00:00-04:00"
    assert seconds_until(zulu, now=now) == pytest.approx(3600.0)
    assert seconds_until(offset, now=now) == pytest.approx(3600.0)


def test_seconds_until_past_or_junk_is_none():
    now = datetime(2026, 9, 8, 13, 0, tzinfo=timezone.utc)
    assert seconds_until("2026-09-08T12:00:00Z", now=now) is None
    assert seconds_until("junk", now=now) is None
    assert seconds_until(None, now=now) is None
    assert seconds_until("", now=now) is None


def test_seconds_until_naive_treated_as_utc():
    now = datetime(2026, 9, 8, 13, 0, tzinfo=timezone.utc)
    assert seconds_until("2026-09-08T13:30:00", now=now) == pytest.approx(1800.0)


# ---------------------------------------------------------------------------
# SubmissionGate
# ---------------------------------------------------------------------------

def _order(ticker="AAPL", side="buy", qty=10, price=100.0):
    return {"ticker": ticker, "side": side, "quantity": qty,
            "limit_price": price, "rationale": "session"}


def test_gate_daily_budget_one_buy_per_ticker():
    gate = SubmissionGate(max_buys_per_ticker=1, max_sells_per_ticker=1)
    allowed, skipped = gate.gate(
        [_order(), _order()],
        counts={}, open_pairs=set(),
        buy_notional_used=0.0, buy_budget=10_000.0,
    )
    assert len(allowed) == 1
    assert [reason for _, reason in skipped] == ["daily buy budget used"]


def test_gate_session_buy_budget_exhausted():
    gate = SubmissionGate(max_buys_per_ticker=10, max_sells_per_ticker=1)
    allowed, skipped = gate.gate(
        [_order("AAPL", qty=1), _order("MSFT", qty=5)],
        counts={}, open_pairs=set(),
        buy_notional_used=900.0, buy_budget=1000.0,
    )
    assert [order_ticker(o) for o in allowed] == ["AAPL"]
    assert [reason for _, reason in skipped] == ["session buy budget exhausted"]


def test_gate_open_order_pair_skipped():
    gate = SubmissionGate()
    allowed, skipped = gate.gate(
        [_order("AAPL", side="buy"), _order("AAPL", side="sell")],
        counts={}, open_pairs={("AAPL", "buy")},
        buy_notional_used=0.0, buy_budget=10_000.0,
    )
    assert [order_side(o) for o in allowed] == ["sell"]
    assert [reason for _, reason in skipped] == ["open order exists"]


def test_gate_counts_from_store_limit_repeats():
    gate = SubmissionGate()
    counts = {"AAPL": {"buy": 1, "sell": 1}}
    allowed, skipped = gate.gate(
        [_order("AAPL"), _order("AAPL", side="sell")],
        counts=counts, open_pairs=set(),
        buy_notional_used=0.0, buy_budget=10_000.0,
    )
    assert allowed == []
    reasons = {reason for _, reason in skipped}
    assert reasons == {"daily buy budget used", "daily sell budget used"}


def test_gate_malformed_orders_skipped():
    gate = SubmissionGate()
    allowed, skipped = gate.gate(
        [{"ticker": "AAPL", "side": "sideways", "quantity": 1,
          "limit_price": 1.0}, None],
        counts={}, open_pairs=set(), buy_notional_used=0.0, buy_budget=10_000.0,
    )
    assert allowed == []
    assert len(skipped) == 2


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

def test_session_store_log_and_query(tmp_path):
    store = SessionStore(db_path=tmp_path / "db.sqlite3")
    store.log_action("2026-09-08", "aapl", "buy", notional=250.0, ts="t1")
    store.log_action("2026-09-08", "AAPL", "sell", notional=0.0, ts="t2")
    store.log_action("2026-09-09", "MSFT", "buy", notional=100.0, ts="t3")
    assert len(store.actions_for_date("2026-09-08")) == 2
    assert store.buy_notional_for_date("2026-09-08") == pytest.approx(250.0)
    assert store.buy_notional_for_date("2026-09-09") == pytest.approx(100.0)
    counts = store.action_counts_for_date("2026-09-08")
    assert counts == {"AAPL": {"buy": 1, "sell": 1}}
    assert store.action_counts_for_date("2026-09-09") == {"MSFT": {"buy": 1, "sell": 0}}


def test_session_store_schema_is_idempotent(tmp_path):
    path = tmp_path / "db.sqlite3"
    SessionStore(db_path=path)
    SessionStore(db_path=path)  # repeated instantiation must not fail
    store = SessionStore(db_path=path)
    store.log_action("2026-09-08", "AAPL", "buy", notional=10.0)
    assert store.buy_notional_for_date("2026-09-08") == pytest.approx(10.0)


def test_session_store_unknown_sides_ignored_in_counts(tmp_path):
    store = SessionStore(db_path=tmp_path / "db.sqlite3")
    store.log_action("2026-09-08", "AAPL", "sideways", notional=5.0)
    assert store.action_counts_for_date("2026-09-08") == {"AAPL": {"buy": 0, "sell": 0}}


# ---------------------------------------------------------------------------
# cheap-cycle injection (graph.py precomputed reuse)
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls = []


def _session_components(rec: _Recorder):
    from src.agents.models import (AnalystReport, DebateRecord, RiskDecision)

    def fetch_ohlcv(ticker, period="6mo"):
        rec.calls.append(f"ohlcv:{ticker}")
        return pd.Series([100.0, 101.0, 99.0] * 10)

    def fetch_fundamentals(ticker):
        rec.calls.append(f"fundamentals:{ticker}")
        return {"gross_margin": 0.4, "sector": "Technology"}

    def fetch_news(ticker):
        rec.calls.append(f"news:{ticker}")
        return [{"title": "h", "publisher": "p", "url": "u"}]

    def generate_signal(prices, regime=None):
        rec.calls.append("quant")
        return {"signal": "HOLD", "confidence": 0.5, "z_score": 0.1,
                "momentum": 0.0, "reference_price": 100.0}

    report = AnalystReport.model_validate({
        "agent": "fundamental", "ticker": "FAKE1", "stance": "neutral",
        "confidence": 0.5, "thesis": "flat", "citations": ["gross_margin=0.4"]})
    debate = DebateRecord.model_validate({
        "ticker": "FAKE1", "bull_points": [], "bear_points": [],
        "winner": "undecided", "citations": ["gross_margin=0.4"]})

    def run_analysts(ticker, inputs, client=None):
        rec.calls.append(f"analysts:{ticker}")
        return {"fundamental": report}

    def run_debate(ticker, reports, client=None):
        rec.calls.append(f"debate:{ticker}")
        return debate

    def evaluate_risk(ticker, reports, debate, quant, account,
                      proposed_position_pct=None, flow_data=None):
        rec.calls.append(f"risk:{ticker}")
        return RiskDecision.model_validate({"ticker": ticker,
                                            "approved": False})

    def build_order_plan(*args, **kwargs):
        rec.calls.append("portfolio")
        return []

    def broker_submit(orders):
        rec.calls.append("broker")
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
    }, report, debate


def test_precomputed_cycle_skips_analysis_and_slow_fetches():
    rec = _Recorder()
    components, report, debate = _session_components(rec)
    precomputed = {
        "FAKE1": {
            "reports": {"fundamental": report},
            "debate": debate,
            "sector": "Technology",
            "flow_data": {},
        }
    }
    decisions = run_pipeline_session(["FAKE1"], components, precomputed)
    assert len(decisions) == 1
    call_text = rec.calls
    assert "ohlcv:FAKE1" in call_text
    assert "quant" in call_text
    assert "risk:FAKE1" in call_text
    # Slow-moving ingestion + LLM stages are skipped entirely.
    assert "fundamentals:FAKE1" not in call_text
    assert "news:FAKE1" not in call_text
    assert "analysts:FAKE1" not in call_text
    assert "debate:FAKE1" not in call_text
    # Sector label carried through for the batch sector cap.
    assert decisions[0]["sector"] == "Technology"
    assert decisions[0]["debate"] is not None


def test_full_cycle_still_runs_analysis():
    rec = _Recorder()
    components, _, _ = _session_components(rec)
    decisions = run_pipeline_session(["FAKE1"], components, None)
    call_text = rec.calls
    assert "analysts:FAKE1" in call_text
    assert "debate:FAKE1" in call_text
    assert "fundamentals:FAKE1" in call_text
    assert decisions[0]["sector"] == "Technology"


def run_pipeline_session(tickers, components, precomputed):
    from src.pipeline.graph import run_pipeline

    if precomputed is not None:
        components = dict(components)
        components["precomputed"] = precomputed
    return asyncio.run(run_pipeline(tickers, components))
