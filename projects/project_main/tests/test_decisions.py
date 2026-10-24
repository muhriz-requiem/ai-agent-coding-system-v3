"""Tests for the append-only SQLite decision store + pipeline state."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.pipeline.state", "src.store.decisions")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.pipeline.state import new_state
from src.store.decisions import DecisionStore


def test_state_factory_defaults():
    state = new_state(["AAPL", "MSFT"])
    assert state["tickers"] == ["AAPL", "MSFT"]
    for key in ("reports", "debates", "decisions", "orders"):
        assert state[key] == []


def test_log_and_tail_roundtrip(tmp_path):
    store = DecisionStore(db_path=tmp_path / "decisions.sqlite3")
    store.log_decision({
        "ts": "2026-08-31T12:00:00Z",
        "ticker": "AAPL",
        "quant": {"signal": "BUY", "confidence": 0.65, "z_score": -1.63},
        "reports": [{"agent": "fundamental", "stance": "bullish"}],
        "veto": {"approved": True, "veto_reasons": []},
        "orders": [{"ticker": "AAPL", "side": "buy", "quantity": 40}],
    })
    rows = store.tail(10)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert rows[0]["quant"]["signal"] == "BUY"
    assert rows[0]["orders"][0]["quantity"] == 40


def test_tail_returns_last_n_in_order(tmp_path):
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    for i in range(5):
        store.log_decision({"ts": f"2026-08-31T12:0{i}:00Z",
                            "ticker": f"T{i}", "quant": {}, "reports": [],
                            "veto": {}, "orders": []})
    tail = store.tail(2)
    assert [row["ticker"] for row in tail] == ["T3", "T4"]


def test_store_is_append_only(tmp_path):
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    public_api = [name for name in dir(store) if not name.startswith("_")]
    for forbidden in ("update", "delete", "drop", "truncate", "clear"):
        assert forbidden not in public_api


def test_count_grows_monotonically(tmp_path):
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    assert store.count() == 0
    store.log_decision({"ts": "t", "ticker": "X", "quant": {}, "reports": [],
                        "veto": {}, "orders": []})
    assert store.count() == 1
