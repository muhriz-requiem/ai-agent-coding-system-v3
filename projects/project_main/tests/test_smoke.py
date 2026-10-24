"""Tests for the end-to-end paper smoke runner (src/pipeline/smoke.py).

Only the OFFLINE dry-run path is exercised in the suite; the manual live
paper run is documented in README and requires Alpaca keys.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.pipeline.smoke", "src.pipeline.graph",
               "src.pipeline.state")
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
import pytest

from src.pipeline import smoke


def _stub_components(submitted):
    def fetch_ohlcv(ticker, period="6mo"):
        return pd.Series([100.0, 101.0, 99.0] * 10)

    def broker_submit(orders):
        submitted.extend(orders)
        return []

    return {
        "fetch_ohlcv": fetch_ohlcv,
        "fetch_fundamentals": lambda t: {"gross_margin": 0.4},
        "fetch_news": lambda t: [],
        "generate_signal": lambda prices, regime=None: {
            "signal": "HOLD", "confidence": 0.5, "z_score": 0.0,
            "momentum": 0.0, "reference_price": 100.0},
        "run_analysts": lambda ticker, inputs, client=None: {},
        "run_debate": lambda ticker, reports, client=None: None,
        "evaluate_risk": lambda ticker, reports, debate, quant, account,
                         proposed_position_pct=None, flow_data=None: None,
        "build_order_plan": lambda decisions, signals, account,
                            day_trade_dates, as_of=None, positions=None,
                            sector_map=None, regime=None,
                            analyst_reports=None, debates=None,
                            cluster_map=None: [],
        "broker_submit": broker_submit,
        "account": {"equity": 100_000.0, "max_position_pct": 0.10},
        "day_trade_dates": [],
        "client": None,
    }


def test_dry_run_never_submits_orders(monkeypatch, tmp_path):
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "d.sqlite3"))
    submitted = []
    summary = asyncio.run(smoke.run_smoke(
        ["FAKE1"], dry_run=True, components=_stub_components(submitted)))
    assert submitted == []
    assert summary["dry_run"] is True
    assert summary["orders_submitted"] == 0


def test_dry_run_writes_decision_log(monkeypatch, tmp_path):
    db_path = tmp_path / "d.sqlite3"
    monkeypatch.setenv("DECISIONS_DB", str(db_path))
    asyncio.run(smoke.run_smoke(["FAKE1"], dry_run=True,
                                components=_stub_components([])))
    assert db_path.exists()


def test_live_run_without_keys_raises_typed_error(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "d.sqlite3"))
    with pytest.raises(smoke.MissingCredentialsError):
        asyncio.run(smoke.run_smoke(["FAKE1"], dry_run=False,
                                    components=_stub_components([])))


def test_missing_credentials_error_is_exception():
    assert issubclass(smoke.MissingCredentialsError, Exception)
