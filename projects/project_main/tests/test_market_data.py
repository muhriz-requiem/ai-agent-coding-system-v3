"""Tests for market data ingestion (src/data/market_data.py).

The yfinance network call must live in the module-level ``_download``
function so tests can monkeypatch it -- nothing here touches the network.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.data.market_data", "src.data.cache")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import pandas as pd
import pytest

from src.data import market_data as md
from src.data.cache import Cache


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=5, freq="D")
    return pd.DataFrame({"Close": [100.0, 101.0, 102.0, 101.5, 103.0]},
                        index=index)


def test_fetch_returns_download_result(monkeypatch):
    calls = []

    def fake_download(ticker, period):
        calls.append((ticker, period))
        return _frame()

    monkeypatch.setattr(md, "_download", fake_download)
    frame = md.fetch_ohlcv("AAPL", period="6mo")
    assert list(frame["Close"]) == [100.0, 101.0, 102.0, 101.5, 103.0]
    assert calls == [("AAPL", "6mo")]


def test_empty_download_raises_data_error(monkeypatch):
    monkeypatch.setattr(md, "_download", lambda t, p: pd.DataFrame())
    with pytest.raises(md.DataError):
        md.fetch_ohlcv("AAPL")


def test_none_download_raises_data_error(monkeypatch):
    monkeypatch.setattr(md, "_download", lambda t, p: None)
    with pytest.raises(md.DataError):
        md.fetch_ohlcv("AAPL")


def test_cache_hit_skips_second_download(monkeypatch, tmp_path):
    calls = []

    def fake_download(ticker, period):
        calls.append(ticker)
        return _frame()

    monkeypatch.setattr(md, "_download", fake_download)
    cache = Cache(root=tmp_path / "c", ttl_seconds=300.0)
    md.fetch_ohlcv("MSFT", cache=cache)
    md.fetch_ohlcv("MSFT", cache=cache)
    assert calls == ["MSFT"]  # second call served from cache


def test_cache_miss_for_different_ticker(monkeypatch, tmp_path):
    calls = []

    def fake_download(ticker, period):
        calls.append(ticker)
        return _frame()

    monkeypatch.setattr(md, "_download", fake_download)
    cache = Cache(root=tmp_path / "c", ttl_seconds=300.0)
    md.fetch_ohlcv("MSFT", cache=cache)
    md.fetch_ohlcv("GOOGL", cache=cache)
    assert calls == ["MSFT", "GOOGL"]


def _multiindex_frame() -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=3, freq="D")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY"]],
        names=["Price", "Ticker"],
    )
    data = [
        [1.0, 2.0, 0.5, 1.5, 1000],
        [1.5, 2.5, 1.0, 2.0, 2000],
        [2.0, 3.0, 1.5, 2.5, 3000],
    ]
    return pd.DataFrame(data, index=index, columns=columns)


def test_multiindex_columns_flattened(monkeypatch):
    monkeypatch.setattr(md, "_download", lambda t, p: _multiindex_frame())
    frame = md.fetch_ohlcv("SPY")
    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(frame["Close"]) == [1.5, 2.0, 2.5]


def test_multiindex_frame_cache_roundtrip(monkeypatch, tmp_path):
    calls = []

    def fake_download(ticker, period):
        calls.append(ticker)
        return _multiindex_frame()

    monkeypatch.setattr(md, "_download", fake_download)
    cache = Cache(root=tmp_path / "c", ttl_seconds=300.0)
    md.fetch_ohlcv("SPY", period="1y", cache=cache)
    rebuilt = md.fetch_ohlcv("SPY", period="1y", cache=cache)
    assert calls == ["SPY"]
    assert list(rebuilt.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert list(rebuilt["Close"]) == [1.5, 2.0, 2.5]
    assert "Volume" in rebuilt.columns


def test_payload_coerces_numpy_scalars():
    import json

    index = pd.date_range("2026-08-01", periods=2, freq="D")
    frame = pd.DataFrame({"Close": [1.0, 2.0], "Volume": [10, 20]}, index=index)
    payload = md._json_safe_payload(frame)
    json.dumps(payload)
    assert payload["Volume"] == [10, 20]


def test_data_error_is_exception_subclass():
    assert issubclass(md.DataError, Exception)
