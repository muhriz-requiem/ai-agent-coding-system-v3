"""Market data adapter: cache-first OHLCV fetching on top of yfinance.

Public API
----------
``DataError``            -- raised when OHLCV data cannot be retrieved.
``_download``            -- the ONLY yfinance seam (lazy import inside).
``fetch_ohlcv``          -- cache-first fetch with a JSON-safe cache payload.

Cache contract (see ``src/data/cache.py``):
    Cache(root: pathlib.Path, ttl_seconds: float = 300.0)
    cache.get(key)            -> Optional[Any]   (exactly one positional arg)
    cache.set(key, value)     -> None           (TypeError if not JSON-serializable)

Payload recipe (critical): the fetched frame is stored as
``frame.reset_index().to_dict(orient='list')`` with the datetime index column
converted to ISO-8601 strings via ``.isoformat()``.  Raw ``pd.Timestamp``
values are NOT JSON-serializable and would make ``Cache.set`` raise
``TypeError('Value is not JSON-serializable')``.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.data.cache import Cache

__all__ = ["DataError", "fetch_ohlcv", "_download"]

# Column names a reset DatetimeIndex can take, in lookup priority order.
_INDEX_COLUMN_CANDIDATES = ("index", "Date", "date", "Datetime", "datetime")


class DataError(Exception):
    """Raised when OHLCV data is missing or empty for the requested ticker."""


def _download(ticker: str, period: str) -> pd.DataFrame:
    """Download OHLCV history via yfinance.

    This is the single yfinance seam in the module: yfinance is imported
    lazily inside this function so importing ``src.data.market_data`` never
    imports yfinance at module import time.
    """
    import yfinance  # lazy: keeps module import free of the yfinance dep

    return yfinance.download(ticker, period=period, progress=False)


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Flatten a MultiIndex column header to its first level.

    yfinance returns columns like ``('Close', 'SPY')``; the tuple names are
    not JSON-serializable cache keys and downstream consumers index by the
    plain price-field name (``frame['Close']``).
    """
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame
    flattened = frame.copy()
    flattened.columns = flattened.columns.get_level_values(0)
    return flattened


def _json_safe_payload(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """Build the JSON-serializable cache payload for a fetched frame.

    The frame is stored as ``frame.reset_index().to_dict(orient='list')`` and
    every value exposing ``.isoformat()`` (i.e. the datetime index column and
    any other datetime column) is converted to an ISO-8601 string, because
    ``pd.Timestamp`` values are not JSON-serializable.  Numpy scalars
    (``np.int64`` volume, etc.) are coerced via ``.item()`` for the same
    reason.
    """
    reset = frame.reset_index()
    payload: dict[str, list[Any]] = reset.to_dict(orient="list")
    for column, values in payload.items():
        payload[column] = [
            value.isoformat()
            if hasattr(value, "isoformat")
            else value.item()
            if hasattr(value, "item")
            else value
            for value in values
        ]
    return payload


def _index_column_of(frame: pd.DataFrame) -> Optional[str]:
    """Return the column holding the stored (reset) index, if any.

    ``reset_index`` places the old index as the first column; when it was
    unnamed the column is literally called ``'index'``.
    """
    for candidate in _INDEX_COLUMN_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    if len(frame.columns):
        return str(frame.columns[0])
    return None


def _frame_from_payload(stored: Any) -> pd.DataFrame:
    """Rebuild a DataFrame from a cached ``reset_index`` dict payload."""
    frame = pd.DataFrame(stored)
    index_col = _index_column_of(frame)
    if index_col is None:
        return frame
    frame[index_col] = pd.to_datetime(frame[index_col])
    frame = frame.set_index(index_col)
    if index_col == "index":
        # A fresh download's DatetimeIndex is unnamed; keep the rebuild equal.
        frame.index.name = None
    return frame


def fetch_ohlcv(
    ticker: str,
    period: str = "6mo",
    cache: Cache | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV rows for ``ticker`` over ``period``, cache-first.

    Cache key: ``f'ohlcv:{ticker}:{period}'``.  On a hit the stored payload is
    rebuilt into a DataFrame (DatetimeIndex restored via ``pd.to_datetime`` +
    ``set_index``) without touching the network.  On a miss the data is
    downloaded through ``_download``; ``None``/empty results raise
    ``DataError`` before anything is written to the cache, and a successful
    frame is stored via the JSON-safe payload recipe.
    """
    cache_key = f"ohlcv:{ticker}:{period}"

    if cache is not None:
        stored: Any | None = cache.get(cache_key)
        if stored is not None:
            # Cache hit: no _download call.
            return _frame_from_payload(stored)

    frame = _download(ticker, period)
    if frame is None or frame.empty:
        raise DataError(
            f"No OHLCV data returned for ticker={ticker!r} period={period!r}"
        )

    frame = _flatten_columns(frame)

    if cache is not None:
        # Never store a raw DataFrame: store the ISO-8601 stringified
        # reset_index dict so Cache.set does not raise TypeError.
        cache.set(cache_key, _json_safe_payload(frame))

    return frame
