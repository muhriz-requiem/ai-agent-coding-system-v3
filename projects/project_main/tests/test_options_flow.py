"""Tests for the unusual-options (UOA) fetch module (src/data/options_flow.py).

The yfinance seam is monkeypatched with fake chain objects; the rate-limit
sleep is zeroed so tests stay fast.  Async entry point is driven via
``asyncio.run``.
"""


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.data.options_flow") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import asyncio
from datetime import date, timedelta

import pytest

import src.data.options_flow as opts_mod
from src.data.cache import Cache

EXPIRY = (date.today() + timedelta(days=10)).isoformat()
EXPIRY2 = (date.today() + timedelta(days=24)).isoformat()
FAR = (date.today() + timedelta(days=120)).isoformat()


class _Chain:
    def __init__(self, calls=None, puts=None):
        self.calls = calls or []
        self.puts = puts or []


class _Ticker:
    def __init__(self, expiries, chains):
        self.options = expiries
        self._chains = chains

    def option_chain(self, expiry):
        return self._chains[expiry]


def _contract(strike=180.0, volume=45_000, oi=8_000, price=2.1, kind="call"):
    return {
        "contractSymbol": f"AAPL{kind}{strike}",
        "strike": strike,
        "lastPrice": price,
        "volume": volume,
        "openInterest": oi,
        "impliedVolatility": 0.35,
    }


def _wire(monkeypatch, expiries, chains):
    monkeypatch.setattr(opts_mod, "_yf_ticker", lambda t: _Ticker(expiries, chains))
    monkeypatch.setattr(opts_mod, "_RATE_LIMIT_SECONDS", 0.0)


def _fetch(monkeypatch, expiries, chains, **kwargs):
    _wire(monkeypatch, expiries, chains)
    return asyncio.run(opts_mod.fetch_unusual_options("AAPL", **kwargs))


def test_detects_unusual_contracts_and_citations(monkeypatch):
    env = _fetch(monkeypatch, (EXPIRY, EXPIRY2), {
        EXPIRY: _Chain(calls=[_contract()]),
        EXPIRY2: _Chain(),
    })
    assert env["status"] == "ok"
    assert len(env["unusual_contracts"]) == 1
    contract = env["unusual_contracts"][0]
    assert contract["type"] == "call"
    assert contract["vol_oi_ratio"] == pytest.approx(45_000 / 8_000)
    assert contract["premium"] == pytest.approx(45_000 * 2.1 * 100)
    assert contract["side"] == "buy"  # volume far exceeds open interest
    assert env["call_put_premium_ratio"] is None  # no put-side premium
    assert env["total_unusual_premium"] == pytest.approx(45_000 * 2.1 * 100)
    assert env["top_signal"]["type"] == "call"
    assert env["citations"][0].startswith("UOA: call 180 ")
    assert env["scanned_expiries"] == 2


def test_call_put_premium_ratio(monkeypatch):
    env = _fetch(monkeypatch, (EXPIRY,), {
        EXPIRY: _Chain(
            calls=[_contract(price=2.0, volume=3_000, oi=1_000)],
            puts=[_contract(price=1.0, volume=3_000, oi=1_000, kind="put")],
        ),
    })
    assert env["status"] == "ok"
    assert env["call_put_premium_ratio"] == pytest.approx(2.0)


def test_filters_vol_oi_and_premium(monkeypatch):
    env = _fetch(monkeypatch, (EXPIRY,), {
        EXPIRY: _Chain(calls=[
            _contract(volume=1_000, oi=8_000),   # ratio 0.125x < 3.0
            _contract(volume=500, oi=100, price=0.4),  # premium 20k < 25k
        ]),
    })
    assert env["status"] == "no_data"
    assert env["unusual_contracts"] == []


def test_no_listed_chain_is_no_data(monkeypatch):
    env = _fetch(monkeypatch, (), {})
    assert env["status"] == "no_data"
    assert "no listed options chain" in env["reason"]


def test_expiries_beyond_max_dte_skipped(monkeypatch):
    env = _fetch(monkeypatch, (FAR,), {FAR: _Chain(calls=[_contract()])})
    assert env["status"] == "no_data"
    assert env["scanned_expiries"] == 0


def test_dte_limit_configurable(monkeypatch):
    env = _fetch(monkeypatch, (FAR,), {FAR: _Chain(calls=[_contract()])},
                 max_dte=180)
    assert env["status"] == "ok"
    assert env["unusual_contracts"][0]["dte"] == (date.fromisoformat(FAR) - date.today()).days


def test_chain_failure_is_error_envelope(monkeypatch):
    class _BoomTicker:
        options = (EXPIRY,)

        def option_chain(self, expiry):
            raise RuntimeError("yfinance exploded")

    monkeypatch.setattr(opts_mod, "_yf_ticker", lambda t: _BoomTicker())
    monkeypatch.setattr(opts_mod, "_RATE_LIMIT_SECONDS", 0.0)
    env = asyncio.run(opts_mod.fetch_unusual_options("AAPL"))
    assert env["status"] == "error"
    assert "yfinance exploded" in env["reason"]


def test_contracts_capped_and_sorted_by_premium(monkeypatch):
    big = _contract(strike=200.0, volume=90_000, oi=1_000, price=5.0)
    small = _contract(strike=100.0, volume=5_000, oi=1_000, price=1.0)
    env = _fetch(monkeypatch, (EXPIRY,), {EXPIRY: _Chain(calls=[small, big])})
    assert env["status"] == "ok"
    assert [c["strike"] for c in env["unusual_contracts"]] == [200.0, 100.0]


def test_cache_roundtrip(tmp_path, monkeypatch):
    calls = []

    class _CountingTicker(_Ticker):
        def option_chain(self, expiry):
            calls.append(expiry)
            return super().option_chain(expiry)

    _wire(monkeypatch, (EXPIRY,), {EXPIRY: _Chain(calls=[_contract()])})
    monkeypatch.setattr(opts_mod, "_yf_ticker",
                        lambda t: _CountingTicker((EXPIRY,), {EXPIRY: _Chain(calls=[_contract()])}))
    cache = Cache(root=tmp_path, ttl_seconds=60)

    first = asyncio.run(opts_mod.fetch_unusual_options("AAPL", cache=cache))
    second = asyncio.run(opts_mod.fetch_unusual_options("AAPL", cache=cache))

    assert calls == [EXPIRY]  # second run served from cache
    assert first == second
