"""Fundamentals ingestion from yfinance.

Exposes :func:`fetch_fundamentals`, which maps selected yfinance ``.info``
fields into a normalized fundamentals dict, with optional read-through
caching via the :class:`~src.data.cache.Cache` helper and a typed
:class:`FundamentalsError` raised when the yfinance adapter fails.

yfinance is imported lazily inside :func:`_raw_info`, so importing this
module never imports (or requires) yfinance at import time.
"""

from __future__ import annotations

from typing import Any

from src.data.cache import Cache

__all__ = ["FundamentalsError", "fetch_fundamentals"]

#: Maps normalized output keys to the underlying yfinance ``.info`` keys.
_FIELD_MAP: dict[str, str] = {
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "debt_to_equity": "debtToEquity",
    "trailing_pe": "trailingPE",
    "market_cap": "marketCap",
    "sector": "sector",
}


class FundamentalsError(Exception):
    """Raised when fundamentals data cannot be fetched from yfinance."""


def _raw_info(ticker: str) -> dict:
    """Return the raw yfinance ``.info`` payload for *ticker*.

    yfinance is imported lazily inside the function body so that this
    module can be imported (and unit-tested with monkeypatched seams)
    without yfinance ever appearing in ``sys.modules``.

    Raises:
        Exception: Whatever yfinance raises (network errors, HTTP errors,
            delisted or invalid tickers, ...). Callers are responsible for
            normalizing failures.
    """
    import yfinance as yf  # deliberate lazy import; keep module import side-effect free

    return yf.Ticker(ticker).info


def fetch_fundamentals(ticker: str, cache: Cache | None = None) -> dict:
    """Fetch normalized fundamentals metrics for *ticker*.

    The returned dict contains exactly the keys ``gross_margin``,
    ``operating_margin``, ``debt_to_equity``, ``trailing_pe``,
    ``market_cap`` and ``sector``. yfinance fields that are missing (or
    ``None``) degrade to ``None`` instead of raising.

    Args:
        ticker: Ticker symbol, passed through unmodified to both yfinance
            and the cache key suffix.
        cache: Optional cache. On a hit (key ``fundamentals:<ticker>``)
            the stored mapping is returned without contacting yfinance;
            on a miss the freshly fetched mapping is written to the cache.

    Returns:
        The normalized fundamentals mapping with exactly the five keys
        described above, each a numeric value or ``None``.

    Raises:
        FundamentalsError: If the underlying yfinance call fails; the
            original exception is chained as ``__cause__``.
    """
    cache_key = f"fundamentals:{ticker}"

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        info: dict[Any, Any] = _raw_info(ticker)
    except Exception as err:  # adapter boundary: normalize any upstream failure
        raise FundamentalsError(
            f"failed to fetch fundamentals for {ticker!r}"
        ) from err

    result: dict[str, Any] = {
        out_key: info.get(yf_key) for out_key, yf_key in _FIELD_MAP.items()
    }

    if cache is not None:
        cache.set(cache_key, result)

    return result
