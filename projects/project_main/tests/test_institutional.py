"""Tests for the institutional (13F) fetch module (src/data/institutional.py).

The SecuritiesDB HTTP seam is monkeypatched with the documented payload
shape -- no network.
"""


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.data.institutional") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import src.data.institutional as inst_mod
from src.data.cache import Cache

PAYLOAD = {
    "status": "ok",
    "holders": [
        {"manager": "Citadel", "cik": "123", "shares": 2_100_000,
         "value": 420_000_000.0, "pct_outstanding": 1.2, "change_qoq": 0.25},
        {"manager": "Bridgewater", "cik": "456", "shares": 900_000,
         "value": 180_000_000.0, "pct_outstanding": 0.5, "change_qoq": -0.4},
        {"manager": "Two Sigma", "cik": "789", "shares": 100_000,
         "value": 20_000_000.0, "pct_outstanding": 0.05, "change_qoq": 0.0},
    ],
}


def _key(monkeypatch):
    monkeypatch.setenv("SECURITIESDB_API_KEY", "test-key")


def test_parses_counts_whale_alerts_and_citations(monkeypatch):
    _key(monkeypatch)
    monkeypatch.setattr(inst_mod, "_fetch_raw", lambda ticker: PAYLOAD)
    env = inst_mod.fetch_institutional_holdings("AAPL")
    assert env["status"] == "ok"
    assert env["total_holders"] == 3
    assert env["increased_positions"] == 1
    assert env["decreased_positions"] == 1
    assert env["new_positions"] == 0
    assert env["closed_positions"] == 0
    assert env["top_holders"][0]["manager"] == "Citadel"  # largest value first
    # Whale alerts are magnitude-ordered: Bridgewater's -40% on $180M
    # (~$120M) outweighs Citadel's +25% on $420M (~$84M).
    assert env["whale_alerts"][0]["manager"] == "Bridgewater"
    assert env["whale_alerts"][0]["action"] == "trimmed"
    assert env["citations"]
    assert all(isinstance(line, str) for line in env["citations"])


def test_summary_block_overrides_derived_counts(monkeypatch):
    _key(monkeypatch)
    payload = {
        "summary": {"total_holders": 42, "new_positions": 3, "closed_positions": 1,
                    "increased_positions": 20, "decreased_positions": 18,
                    "net_flow": 1_500_000.0, "total_institutional_pct": 61.5},
        "holders": PAYLOAD["holders"],
    }
    monkeypatch.setattr(inst_mod, "_fetch_raw", lambda ticker: payload)
    env = inst_mod.fetch_institutional_holdings("AAPL")
    assert env["total_holders"] == 42
    assert env["new_positions"] == 3
    assert env["closed_positions"] == 1
    assert env["increased_positions"] == 20
    assert env["decreased_positions"] == 18
    assert env["net_flow"] == 1_500_000.0
    assert env["total_institutional_pct"] == 61.5


def test_missing_key_is_error_envelope(monkeypatch):
    monkeypatch.delenv("SECURITIESDB_API_KEY", raising=False)
    env = inst_mod.fetch_institutional_holdings("AAPL")
    assert env["status"] == "error"
    assert "SECURITIESDB_API_KEY" in env["reason"]


def test_api_failure_is_error_envelope(monkeypatch):
    _key(monkeypatch)

    def boom(ticker):
        raise RuntimeError("http 500")

    monkeypatch.setattr(inst_mod, "_fetch_raw", boom)
    env = inst_mod.fetch_institutional_holdings("AAPL")
    assert env["status"] == "error"
    assert "http 500" in env["reason"]


def test_no_coverage_is_no_data(monkeypatch):
    _key(monkeypatch)
    monkeypatch.setattr(inst_mod, "_fetch_raw", lambda ticker: {"holders": []})
    env = inst_mod.fetch_institutional_holdings("AAPL")
    assert env["status"] == "no_data"
    assert env["total_holders"] == 0
    assert env["citations"] == []


def test_unparseable_payload_degrades_to_no_data(monkeypatch):
    _key(monkeypatch)
    monkeypatch.setattr(inst_mod, "_fetch_raw", lambda ticker: "unexpected-shape")
    env = inst_mod.fetch_institutional_holdings("AAPL")
    assert env["status"] == "no_data"


def test_cache_roundtrip(tmp_path, monkeypatch):
    _key(monkeypatch)
    calls = []

    def raw(ticker):
        calls.append(ticker)
        return PAYLOAD

    monkeypatch.setattr(inst_mod, "_fetch_raw", raw)
    cache = Cache(root=tmp_path, ttl_seconds=60)

    first = inst_mod.fetch_institutional_holdings("AAPL", cache=cache)
    second = inst_mod.fetch_institutional_holdings("AAPL", cache=cache)

    assert calls == ["AAPL"]
    assert first == second
