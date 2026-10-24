"""Tests for the insider (SEC Form 4/144) fetch module (src/data/insider.py).

All seams are monkeypatched -- no network, no edgartools import.
"""


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.data.insider") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import src.data.insider as insider_mod
from src.data.cache import Cache


def _tx(insider, date_, type_, shares, price=None):
    return {
        "insider": insider,
        "title": "CEO",
        "transaction_date": date_,
        "type": type_,
        "shares": shares,
        "price": price,
        "value": (shares * price) if price else None,
        "ownership": "D",
    }


def _ok_seams(monkeypatch, form4, form144=None):
    monkeypatch.setattr(insider_mod, "_form4_transactions", lambda t, d: form4)
    monkeypatch.setattr(
        insider_mod, "_form144_proposals", lambda t, d: (form144 or [])
    )


def _identity(monkeypatch):
    monkeypatch.setenv("EDGAR_IDENTITY", "Tester tester@example.com")


def test_parses_clusters_and_net_flow(monkeypatch):
    _identity(monkeypatch)
    form4 = [
        _tx("Alice", "2026-08-15", "buy", 5000, 150.0),
        _tx("Bob", "2026-08-20", "buy", 2000, 151.0),
        _tx("Alice", "2026-01-10", "sell", 1000, 140.0),
    ]
    _ok_seams(monkeypatch, form4)
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["status"] == "ok"
    assert env["cluster_buys"] == 2  # two insiders buying within 30 days
    assert env["cluster_sells"] == 1
    assert env["net_insider_flow"] == (5000 * 150.0 + 2000 * 151.0) - (1000 * 140.0)
    assert env["citations"]
    assert all(isinstance(line, str) for line in env["citations"])


def test_cluster_window_boundary(monkeypatch):
    _identity(monkeypatch)
    # 31 days apart: no single 30-day window contains both insiders.
    _ok_seams(monkeypatch, [
        _tx("Alice", "2026-08-01", "buy", 1000, 10.0),
        _tx("Bob", "2026-09-01", "buy", 1000, 10.0),
    ])
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["cluster_buys"] == 1


def test_empty_window_is_no_data_never_raises(monkeypatch):
    _identity(monkeypatch)
    _ok_seams(monkeypatch, [])
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["status"] == "no_data"
    assert env["form4"] == []
    assert env["form144"] == []
    assert env["cluster_buys"] == 0
    assert env["cluster_sells"] == 0
    assert env["net_insider_flow"] == 0.0
    assert env["citations"] == []


def test_seam_failure_is_error_envelope(monkeypatch):
    _identity(monkeypatch)

    def boom(ticker, days):
        raise RuntimeError("network down")

    monkeypatch.setattr(insider_mod, "_form4_transactions", boom)
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["status"] == "error"
    assert "network down" in env["reason"]


def test_missing_identity_fails_open(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["status"] == "error"
    assert "EDGAR_IDENTITY" in env["reason"]


def test_cache_roundtrip(tmp_path, monkeypatch):
    _identity(monkeypatch)
    calls = []

    def seam(ticker, days):
        calls.append(ticker)
        return [_tx("Alice", "2026-08-15", "buy", 100, 10.0)]

    monkeypatch.setattr(insider_mod, "_form4_transactions", seam)
    monkeypatch.setattr(insider_mod, "_form144_proposals", lambda t, d: [])
    cache = Cache(root=tmp_path, ttl_seconds=60)

    first = insider_mod.fetch_insider_activity("AAPL", cache=cache)
    second = insider_mod.fetch_insider_activity("AAPL", cache=cache)

    assert calls == ["AAPL"]  # second call served from cache
    assert first == second


def test_error_envelope_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    cache = Cache(root=tmp_path, ttl_seconds=60)

    env = insider_mod.fetch_insider_activity("AAPL", cache=cache)

    assert env["status"] == "error"
    assert cache.get("insider:AAPL:90d") is None


def test_form144_only_window_is_ok(monkeypatch):
    _identity(monkeypatch)
    proposals = [{
        "insider": "Carol", "proposed_sale_date": "2026-08-30",
        "shares": 12000, "held": 50000,
    }]
    _ok_seams(monkeypatch, [], form144=proposals)
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["status"] == "ok"
    assert env["form144"] == proposals
    assert env["cluster_buys"] == 0  # proposals are not executed buys


def test_cache_key_includes_window(monkeypatch, tmp_path):
    _identity(monkeypatch)
    _ok_seams(monkeypatch, [])
    cache = Cache(root=tmp_path, ttl_seconds=60)
    insider_mod.fetch_insider_activity("AAPL", cache=cache, days=90)
    insider_mod.fetch_insider_activity("AAPL", cache=cache, days=30)
    # Different windows are distinct cache entries (both fetch, both stored).
    assert cache.get("insider:AAPL:90d") is not None
    assert cache.get("insider:AAPL:30d") is not None


def test_classify_type_maps_sec_codes_and_descriptions():
    # SEC Form 4 codes and the edgartools to_dataframe() descriptions.
    assert insider_mod._classify_type("Sale") == "sell"
    assert insider_mod._classify_type("S - Open Market Sale") == "sell"
    assert insider_mod._classify_type("S") == "sell"
    assert insider_mod._classify_type("Purchase") == "buy"
    assert insider_mod._classify_type("P - Purchase") == "buy"
    assert insider_mod._classify_type("P") == "buy"
    assert insider_mod._classify_type("A - Award") == "other"
    assert insider_mod._classify_type("M - Option Exercise") == "other"
    assert insider_mod._classify_type("") == "other"
    assert insider_mod._classify_type(None) == "other"


def test_live_rows_classified_before_aggregation(monkeypatch):
    # Regression: raw to_dataframe rows ("Sale"/"S") must be classified
    # before clusters and net flow are computed from them.
    _identity(monkeypatch)
    form4 = [
        {"insider": "A", "title": "SVP", "transaction_date": "2026-08-15",
         "type": insider_mod._classify_type("Sale"), "shares": 1439.0,
         "price": 317.01, "value": 456177.39, "ownership": None},
        {"insider": "B", "title": "CFO", "transaction_date": "2026-08-20",
         "type": insider_mod._classify_type("P - Purchase"), "shares": 5000.0,
         "price": 200.0, "value": 1_000_000.0, "ownership": None},
    ]
    _ok_seams(monkeypatch, form4)
    env = insider_mod.fetch_insider_activity("AAPL")
    assert env["status"] == "ok"
    assert env["cluster_buys"] == 1
    assert env["cluster_sells"] == 1
    # 5000*200 (buy) - 1439*317.01 (sell) = 543,948.61
    assert env["net_insider_flow"] == (5000 * 200.0) - (1439 * 317.01)
