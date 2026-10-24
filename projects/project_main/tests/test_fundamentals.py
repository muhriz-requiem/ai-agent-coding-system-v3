"""Tests for fundamentals ingestion (src/data/fundamentals.py).

``_raw_info`` is the single yfinance seam; tests monkeypatch it.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.data.fundamentals", "src.data.cache")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.data import fundamentals as fd
from src.data.cache import Cache


def _info() -> dict:
    return {
        "grossMargins": 0.41,
        "operatingMargins": 0.29,
        "debtToEquity": 187.4,
        "trailingPE": 31.2,
        "marketCap": 3_400_000_000_000,
        "sector": "Technology",
    }


def test_fields_are_mapped(monkeypatch):
    monkeypatch.setattr(fd, "_raw_info", lambda t: _info())
    data = fd.fetch_fundamentals("AAPL")
    assert data == {
        "gross_margin": 0.41,
        "operating_margin": 0.29,
        "debt_to_equity": 187.4,
        "trailing_pe": 31.2,
        "market_cap": 3_400_000_000_000,
        "sector": "Technology",
    }


def test_missing_fields_default_to_none(monkeypatch):
    monkeypatch.setattr(fd, "_raw_info", lambda t: {"trailingPE": 15.0})
    data = fd.fetch_fundamentals("XYZ")
    assert data["trailing_pe"] == 15.0
    assert data["gross_margin"] is None
    assert data["operating_margin"] is None
    assert data["debt_to_equity"] is None
    assert data["market_cap"] is None
    # ETFs and unknown symbols carry no sector label -> exempt from caps.
    assert data["sector"] is None


def test_raw_info_exception_propagates_typed(monkeypatch):
    def boom(ticker):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(fd, "_raw_info", boom)
    try:
        fd.fetch_fundamentals("AAPL")
    except fd.FundamentalsError:
        assert True
    except RuntimeError:
        raise AssertionError("expected typed FundamentalsError, got RuntimeError")
    else:
        raise AssertionError("expected FundamentalsError")


def test_cache_hit_skips_second_call(monkeypatch, tmp_path):
    calls = []

    def fake_raw(ticker):
        calls.append(ticker)
        return _info()

    monkeypatch.setattr(fd, "_raw_info", fake_raw)
    cache = Cache(root=tmp_path / "c", ttl_seconds=300.0)
    fd.fetch_fundamentals("AAPL", cache=cache)
    fd.fetch_fundamentals("AAPL", cache=cache)
    assert calls == ["AAPL"]
