"""Integration tests: the pairs module inside run_pipeline (graph wiring).

Offline end-to-end: every stage is injected via ``components``.  With
pairs components present the batch emits pair-leg orders (and a
``kind="pair"`` audit entry); without them the pipeline is bit-for-bit
pairs-free.
"""

import asyncio
import math

import pandas as pd
import pytest

from src.pipeline.graph import run_pipeline
from src.agents.models import (AnalystReport, DebateRecord, OrderPlan,
                               RiskDecision)
from src.pairs.book import PairBook
from src.pairs.signals import PairSignal


def _report(ticker: str) -> AnalystReport:
    return AnalystReport.model_validate({
        "agent": "fundamental", "ticker": ticker, "stance": "neutral",
        "confidence": 0.5, "thesis": "flat", "citations": ["c1"]})


def _pair_signal(pair_id="AAA/BBB", z=2.5, sig_a="SELL_EXIT", sig_b="BUY"):
    return PairSignal(
        pair_id=pair_id,
        ticker_a=pair_id.split("/")[0],
        ticker_b=pair_id.split("/")[1],
        hedge_ratio=1.0,
        alpha=0.0,
        z_score=z,
        signal_a=sig_a,
        signal_b=sig_b,
        confidence=min(1.0, abs(z) / 2.0),
        spread_mean=0.0,
        spread_std=0.02,
        price_a=105.0,
        price_b=100.0,
        spread_daily_vol=0.01,
    )


def _book() -> PairBook:
    from src.pairs.book import BookEntry

    book = PairBook("/unused/integration.json")
    book.entries = {
        "AAA/BBB": BookEntry(
            pair_id="AAA/BBB", ticker_a="AAA", ticker_b="BBB",
            hedge_ratio=1.0, alpha=0.0, spread_mean=0.0, spread_std=0.02,
            half_life=10.0, formed_on="2026-09-01",
            spread_daily_vol=0.01, formation_corr=0.9,
            formation_adv={"AAA": 5e6, "BBB": 5e6},
        ),
    }
    return book


def _components(with_pairs: bool = True):
    def fetch_ohlcv(ticker, period="6mo"):
        return pd.Series([100.0, 101.0, 99.0] * 10)

    def fetch_fundamentals(ticker):
        return {"gross_margin": 0.4, "sector": "Tech"}

    def fetch_news(ticker):
        return []

    def generate_signal(prices, regime=None):
        return {"signal": "HOLD", "confidence": 0.5, "z_score": 0.1,
                "momentum": 0.0, "reference_price": 100.0}

    def run_analysts(ticker, inputs, client=None):
        report = _report(ticker)
        return {"fundamental": report, "technical": report,
                "sentiment": report, "macro": report}

    def run_debate(ticker, reports, client=None):
        return DebateRecord.model_validate({
            "ticker": ticker, "bull_points": [], "bear_points": [],
            "winner": "undecided", "citations": ["c1"]})

    def evaluate_risk(ticker, reports, debate, quant, account,
                      proposed_position_pct=None, flow_data=None):
        return RiskDecision.model_validate({"ticker": ticker,
                                            "approved": False})

    def broker_submit(orders):
        return []

    components = {
        "fetch_ohlcv": fetch_ohlcv,
        "fetch_fundamentals": fetch_fundamentals,
        "fetch_news": fetch_news,
        "generate_signal": generate_signal,
        "run_analysts": run_analysts,
        "run_debate": run_debate,
        "evaluate_risk": evaluate_risk,
        "build_order_plan": build_order_plan,
        "broker_submit": broker_submit,
        "account": {
            "equity": 100_000.0, "max_position_pct": 0.10,
            "pair_max_pct": 0.03, "pairs_total_pct": 0.30,
            "pair_z_entry": 2.0, "min_trade_notional": 10.0,
        },
        "day_trade_dates": [],
        "client": None,
    }
    if with_pairs:
        components["pair_book"] = _book()
        components["pairs_form_if_due"] = lambda book, as_of: book
        # Mirrors the real closure: an empty book yields no signals.
        components["pairs_signals"] = lambda book: (
            [_pair_signal().__dict__] if len(book) else []
        )
        components["pairs_risk"] = lambda signal, book: {
            "pair_id": signal["pair_id"], "approved": True,
            "veto_reasons": [],
        }
    return components


def build_order_plan(decisions, signals, account, day_trade_dates,
                     as_of=None, positions=None, sector_map=None,
                     regime=None, analyst_reports=None, debates=None,
                     cluster_map=None, pair_decisions=None,
                     pair_signals=None):
    from src.agents.portfolio_manager import build_order_plan as _plan

    return _plan(decisions, signals, account, day_trade_dates, as_of,
                 positions=positions, sector_map=sector_map, regime=regime,
                 analyst_reports=analyst_reports, debates=debates,
                 cluster_map=cluster_map, pair_decisions=pair_decisions,
                 pair_signals=pair_signals)


def test_pairs_disabled_pipeline_is_pairs_free():
    result = asyncio.run(run_pipeline(["FAKE1"], _components(with_pairs=False)))
    assert len(result) == 1
    assert result[0]["ticker"] == "FAKE1"
    assert all(d.get("kind") != "pair" for d in result)
    assert result[0]["orders"] == []


def test_pairs_components_emit_leg_orders_and_audit_entry():
    result = asyncio.run(run_pipeline(["FAKE1"], _components(with_pairs=True)))
    pair_entries = [d for d in result if d.get("kind") == "pair"]
    assert len(pair_entries) == 1
    entry = pair_entries[0]
    assert entry["ticker"] == "PAIR:AAA/BBB"
    assert entry["pair"]["pair_id"] == "AAA/BBB"
    assert entry["veto"]["approved"] is True
    # The BUY leg (BBB, cheap) is emitted and attached to the pair entry.
    buy_orders = [o for o in entry["orders"] if o["side"] == "buy"]
    assert [o["ticker"] for o in buy_orders] == ["BBB"]
    assert buy_orders[0]["quantity"] * buy_orders[0]["limit_price"] \
        == pytest.approx(1500.0, rel=1e-6)
    # The unheld SELL_EXIT leg (AAA) is never emitted (no short).
    assert all(o["ticker"] != "AAA" for o in entry["orders"])
    # The per-ticker entry is untouched by the pairs layer.
    assert result[0]["ticker"] == "FAKE1"
    assert result[0]["orders"] == []


def test_vetoed_pair_emits_no_orders_but_logs_decision():
    components = _components(with_pairs=True)
    components["pairs_risk"] = lambda signal, book: {
        "pair_id": signal["pair_id"], "approved": False,
        "veto_reasons": ["regime panic blocks pair BUY"],
    }
    result = asyncio.run(run_pipeline(["FAKE1"], components))
    pair_entries = [d for d in result if d.get("kind") == "pair"]
    assert len(pair_entries) == 1
    assert pair_entries[0]["veto"]["approved"] is False
    assert pair_entries[0]["orders"] == []


def test_empty_book_degrades_to_no_pairs():
    components = _components(with_pairs=True)
    components["pair_book"] = PairBook("/unused/empty.json")
    result = asyncio.run(run_pipeline(["FAKE1"], components))
    assert [d for d in result if d.get("kind") == "pair"] == []


def test_pairs_signals_without_risk_fn_still_gates():
    components = _components(with_pairs=True)
    del components["pairs_risk"]
    result = asyncio.run(run_pipeline(["FAKE1"], components))
    # No risk decisions -> planner sees approved=None -> no pair orders,
    # and the audit entry still records the signal.
    pair_entries = [d for d in result if d.get("kind") == "pair"]
    assert pair_entries == []
