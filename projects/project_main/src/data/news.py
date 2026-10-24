"""News headlines ingestion.

Fetches headlines for a ticker through a thin, lazy-imported yfinance
adapter, normalizes them to ``{'title', 'publisher', 'url'}`` items,
deduplicates by title (first-seen order), caps the result at ``limit``,
and optionally caches the normalized output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import used only for static typing
    from src.data.cache import Cache

__all__ = ["NewsError", "_raw_news", "fetch_headlines"]


class NewsError(Exception):
    """Raised when the underlying news adapter fails."""


def _publisher(item: dict) -> Any:
    """Return the publisher name from flat or nested (yfinance ``content``) items."""
    provider = item.get("provider")
    if isinstance(provider, dict):
        name = provider.get("displayName")
        if name:
            return name
    return item.get("publisher")


def _url(item: dict) -> Any:
    """Return the article URL from flat or nested (yfinance ``content``) items."""
    for key in ("canonicalUrl", "clickThroughUrl"):
        nested = item.get(key)
        if isinstance(nested, dict) and nested.get("url"):
            return nested["url"]
    return item.get("link") or item.get("url")


def _raw_news(ticker: str) -> list[dict]:
    """Return raw yfinance news items for *ticker*.

    yfinance is imported lazily inside this function so importing this
    module never requires yfinance to be installed; failures surface
    only when the adapter is actually invoked.
    """
    import yfinance as yf

    return list(yf.Ticker(ticker).news or [])


def fetch_headlines(
    ticker: str,
    limit: int = 10,
    cache: Cache | None = None,
) -> list[dict]:
    """Fetch normalized, deduplicated headlines for *ticker*.

    Items are shaped exactly ``{'title', 'publisher', 'url'}``, mapped
    from the raw yfinance fields ``title`` / ``publisher`` / ``link``.
    Duplicates (by title) keep only the first occurrence, order is
    preserved, and the result is capped at ``limit``. An empty source
    yields ``[]``. Any adapter failure is re-raised as :class:`NewsError`.

    When *cache* is provided, results are stored under ``f'news:{ticker}'``
    and a cached hit is returned without invoking the adapter.
    """
    cache_key = f"news:{ticker}"

    if cache is not None:
        cached: Any = cache.get(cache_key)
        if cached is not None:
            return cached

    if limit <= 0:
        return []

    try:
        raw = _raw_news(ticker)
    except NewsError:
        raise
    except Exception as exc:
        raise NewsError(f"failed to fetch news for {ticker!r}: {exc}") from exc

    headlines: list[dict] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        source = content if isinstance(content, dict) else item
        title = source.get("title")
        if not title or title in seen:
            continue
        seen.add(title)
        headlines.append(
            {
                "title": title,
                "publisher": _publisher(source),
                "url": _url(source),
            }
        )
        if len(headlines) >= limit:
            break

    if cache is not None:
        cache.set(cache_key, headlines)

    return headlines
