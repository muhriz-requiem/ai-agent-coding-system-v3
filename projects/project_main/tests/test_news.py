"""Tests for news ingestion (src/data/news.py).

``_raw_news`` is the single yfinance seam; tests monkeypatch it.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.data.news", "src.data.cache")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.data import news as ns
from src.data.cache import Cache


def _raw() -> list[dict]:
    return [
        {"title": "Apple beats earnings", "publisher": "Reuters",
         "url": "https://example.com/1"},
        {"title": "Apple beats earnings", "publisher": "Reuters",
         "url": "https://example.com/1"},  # exact duplicate
        {"title": "iPhone sales surge", "publisher": "Bloomberg",
         "url": "https://example.com/2"},
        {"title": "Analysts raise target", "publisher": "CNBC",
         "url": "https://example.com/3"},
        {"title": "Guidance trimmed", "publisher": "FT",
         "url": "https://example.com/4"},
    ]


def test_deduplicates_and_preserves_order(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: _raw())
    headlines = ns.fetch_headlines("AAPL")
    titles = [h["title"] for h in headlines]
    assert titles == ["Apple beats earnings", "iPhone sales surge",
                      "Analysts raise target", "Guidance trimmed"]


def test_respects_limit(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: _raw())
    headlines = ns.fetch_headlines("AAPL", limit=2)
    assert len(headlines) == 2


def test_empty_news_returns_empty_list(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: [])
    assert ns.fetch_headlines("AAPL") == []


def test_headline_shape(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: _raw())
    headline = ns.fetch_headlines("AAPL")[0]
    assert set(headline.keys()) == {"title", "publisher", "url"}


def test_raw_news_exception_raises_typed(monkeypatch):
    def boom(ticker):
        raise OSError("network down")

    monkeypatch.setattr(ns, "_raw_news", boom)
    try:
        ns.fetch_headlines("AAPL")
    except ns.NewsError:
        assert True
    except OSError:
        raise AssertionError("expected typed NewsError, got OSError")
    else:
        raise AssertionError("expected NewsError")


def _nested_raw() -> list[dict]:
    return [
        {
            "id": "n1",
            "content": {
                "title": "Palantir leads rally",
                "provider": {"displayName": "GuruFocus.com"},
                "canonicalUrl": {"url": "https://example.com/n1"},
                "clickThroughUrl": {"url": "https://example.com/n1-click"},
                "description": "big movers",
            },
        },
        {"id": "n2", "content": {"title": "Palantir leads rally"}},
        {
            "id": "n3",
            "content": {
                "title": "Fed signals hold",
                "provider": {"displayName": "Reuters"},
                "clickThroughUrl": {"url": "https://example.com/n3"},
            },
        },
        "not-a-dict",
    ]


def test_nested_yfinance_content_normalized(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: _nested_raw())
    headlines = ns.fetch_headlines("AAPL")
    assert [h["title"] for h in headlines] == [
        "Palantir leads rally",
        "Fed signals hold",
    ]
    assert headlines[0]["publisher"] == "GuruFocus.com"
    assert headlines[0]["url"] == "https://example.com/n1"
    assert headlines[1]["publisher"] == "Reuters"
    assert headlines[1]["url"] == "https://example.com/n3"


def test_nested_items_shape(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: _nested_raw())
    headline = ns.fetch_headlines("AAPL")[0]
    assert set(headline.keys()) == {"title", "publisher", "url"}


def test_flat_item_urls_flow_through(monkeypatch):
    monkeypatch.setattr(ns, "_raw_news", lambda t: _raw())
    headlines = ns.fetch_headlines("AAPL")
    assert headlines[0]["url"] == "https://example.com/1"


def test_cache_roundtrip(monkeypatch, tmp_path):
    calls = []

    def fake_raw(ticker):
        calls.append(ticker)
        return _raw()

    monkeypatch.setattr(ns, "_raw_news", fake_raw)
    cache = Cache(root=tmp_path / "c", ttl_seconds=300.0)
    ns.fetch_headlines("AAPL", cache=cache)
    ns.fetch_headlines("AAPL", cache=cache)
    assert calls == ["AAPL"]
