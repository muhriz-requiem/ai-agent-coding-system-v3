"""Tests for the overnight (halo/drift) cycle helpers
(src/pipeline/overnight.py).  All offline: tmp_path stats files, dict
payloads, no network.
"""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.pipeline.overnight", "src.store.session")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import json

import pytest

from src.pipeline.overnight import (exclude_session_bought, exit_orders,
                                    load_overnight_kelly, pending_exits,
                                    select_entries)


# ---------------------------------------------------------------------------
# load_overnight_kelly
# ---------------------------------------------------------------------------

def _write_stats(tmp_path, buy_fraction, buy_n, name="overnight.json"):
    payload = {
        "mode": "overnight",
        "per_side": {
            "BUY": {"n_trades": buy_n, "kelly_fraction": buy_fraction,
                    "usable": buy_fraction is not None and buy_n >= 30},
            "SELL": {"n_trades": 809, "kelly_fraction": None, "usable": False},
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_overnight_kelly_usable(tmp_path):
    path = _write_stats(tmp_path, 0.0961, 112)
    kelly = load_overnight_kelly(path, min_trades=30)
    assert kelly == {"fraction": pytest.approx(0.0961), "n": 112}


def test_load_overnight_kelly_unusable(tmp_path):
    # Negative fraction, too-small sample, and None fraction all degrade.
    assert load_overnight_kelly(_write_stats(tmp_path, -0.1, 100, "a.json"),
                                min_trades=30) is None
    assert load_overnight_kelly(_write_stats(tmp_path, 0.2, 10, "b.json"),
                                min_trades=30) is None
    assert load_overnight_kelly(_write_stats(tmp_path, None, 100, "c.json"),
                                min_trades=30) is None


def test_load_overnight_kelly_missing_or_junk_file(tmp_path):
    assert load_overnight_kelly(tmp_path / "missing.json") is None
    junk = tmp_path / "junk.json"
    junk.write_text("not json", encoding="utf-8")
    assert load_overnight_kelly(junk) is None
    assert load_overnight_kelly(None) is None


# ---------------------------------------------------------------------------
# select_entries
# ---------------------------------------------------------------------------

def _decision(ticker, signal, approved, confidence):
    return {
        "ticker": ticker,
        "quant": {"signal": signal, "confidence": confidence},
        "veto": {"approved": approved},
    }


def test_select_entries_ranks_and_caps():
    decisions = [
        _decision("AAPL", "BUY", True, 0.6),
        _decision("MSFT", "BUY", True, 0.9),
        _decision("GOOGL", "BUY", True, 0.75),
        _decision("NVDA", "SELL", True, 0.99),
        _decision("META", "BUY", False, 0.99),
    ]
    selected = select_entries(decisions, max_names=2)
    assert selected == {"MSFT", "GOOGL"}


def test_select_entries_zero_cap_and_junk():
    decisions = [_decision("AAPL", "BUY", True, 0.6)]
    assert select_entries(decisions, max_names=0) == set()
    assert select_entries([], max_names=5) == set()
    assert select_entries([None, "junk", _decision("AAPL", "BUY", True, 0.6)],
                          max_names=5) == {"AAPL"}


# ---------------------------------------------------------------------------
# exclude_session_bought (cross-mode double-buy guard)
# ---------------------------------------------------------------------------

def test_exclude_session_bought_drops_exhausted_tickers():
    counts = {"AAPL": {"buy": 1, "sell": 0}, "MSFT": {"buy": 0, "sell": 1}}
    assert exclude_session_bought({"AAPL", "MSFT", "GOOGL"}, counts) == \
        {"MSFT", "GOOGL"}


def test_exclude_session_bought_respects_custom_budget():
    counts = {"AAPL": {"buy": 2, "sell": 0}, "MSFT": {"buy": 1, "sell": 0}}
    assert exclude_session_bought({"AAPL", "MSFT"}, counts, max_buys=2) == \
        {"MSFT"}


def test_exclude_session_bought_empty_counts_passthrough():
    assert exclude_session_bought({"AAPL", "MSFT"}, {}) == {"AAPL", "MSFT"}
    assert exclude_session_bought(set(), {"AAPL": {"buy": 0, "sell": 0}}) == set()


# ---------------------------------------------------------------------------
# pending_exits
# ---------------------------------------------------------------------------

def _action(id, ticker, kind):
    return {"id": id, "ticker": ticker, "kind": kind, "trade_date": "2026-09-08"}


def test_pending_exits_open_and_closed():
    actions = [
        _action(1, "AAPL", "overnight_entry"),
        _action(2, "MSFT", "overnight_entry"),
        _action(3, "AAPL", "overnight_exit"),
    ]
    assert pending_exits(actions, "2026-09-09") == {"MSFT"}


def test_pending_exits_idempotent_after_exit():
    actions = [
        _action(1, "AAPL", "overnight_entry"),
        _action(2, "AAPL", "overnight_exit"),
    ]
    assert pending_exits(actions, "2026-09-09") == set()


def test_pending_exits_cycle_action_kinds_ignored():
    actions = [
        {"id": 1, "ticker": "AAPL", "kind": "cycle"},
        {"id": 2, "ticker": "AAPL", "kind": "overnight_entry"},
    ]
    assert pending_exits(actions, "2026-09-09") == {"AAPL"}


# ---------------------------------------------------------------------------
# exit_orders
# ---------------------------------------------------------------------------

def test_exit_orders_sell_held_quantity():
    orders = exit_orders(
        {"AAPL", "MSFT"},
        positions={"AAPL": 12.5},
        price_map={"AAPL": 230.0, "MSFT": 500.0},
    )
    assert len(orders) == 1
    assert orders[0]["ticker"] == "AAPL"
    assert orders[0]["side"] == "sell"
    assert orders[0]["quantity"] == pytest.approx(12.5)
    assert orders[0]["limit_price"] == pytest.approx(230.0)


def test_exit_orders_skip_unheld_and_junk_prices():
    orders = exit_orders(
        {"AAPL", "MSFT", "GOOGL"},
        positions={"AAPL": 5.0, "MSFT": 3.0},
        price_map={"AAPL": 230.0, "GOOGL": 100.0},  # MSFT price missing
    )
    assert [o["ticker"] for o in orders] == ["AAPL"]


def test_exit_orders_empty_when_nothing_held():
    assert exit_orders({"AAPL"}, positions={}, price_map={"AAPL": 100.0}) == []
