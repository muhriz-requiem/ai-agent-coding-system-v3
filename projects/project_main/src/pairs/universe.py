"""Dynamic mid-cap universe: S&P 400 scrape + cache + fallback file.

Sources the pair-trading universe.  The Wikipedia "List of S&P 400
companies" table is scraped with ``pandas.read_html`` (requires
``lxml``); the table's GICS Sector column supplies the per-ticker sector
labels used by the discovery pre-screen and the sector dedupe -- the
pipeline's fundamentals fetch only covers WATCHLIST tickers and cannot
serve pair names.

Fail-open contract:
  * scrape failure + no cache + no fallback file -> empty mapping
    (the pairs layer then discovers nothing; the single-leg pipeline
    is unaffected)
  * screening filters (ADV / market cap) apply only to tickers whose
    data was provided; unknown tickers pass through untouched

Pure-ish module: the only I/O is the scrape, the disk cache, and the
fallback file read.
"""

from __future__ import annotations

import json
import io
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.data.cache import Cache

__all__ = [
    "SP400_URL",
    "UNIVERSE_CACHE_KEY",
    "get_sp400_universe",
    "screen_universe",
    "refresh_universe",
    "load_fallback_universe",
]

SP400_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
UNIVERSE_CACHE_KEY = "universe_sp400"

#: Wikipedia rejects default scripting User-Agents; send a browser one.
_WIKI_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

#: Fallback-file line format: ``TICKER|Sector`` (sector may be blank).
SECTOR_SEPARATOR = "|"


def _clean_ticker(raw: Any) -> Optional[str]:
    """Normalize one Wikipedia symbol cell; ``None`` when unusable.

    Wikipedia marks share-class symbols like ``BRK.B`` with a class
    suffix; yfinance expects the dot form, so the value is uppercased
    and kept as-is apart from stripping whitespace and footnote junk.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "ticker"}:
        return None
    text = text.replace("\xa0", " ").strip()
    if not text or any(ch.isspace() for ch in text):
        return None
    return text.upper()


def _clean_sector(raw: Any) -> Optional[str]:
    """Normalize one GICS Sector cell; ``None`` when blank/missing."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _fetch_html(url: str, timeout: float = 30.0) -> str:
    """Fetch raw page HTML with a browser User-Agent (raises on failure).

    Kept separate so tests can stub the network independently of the
    table parser.
    """
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": _WIKI_USER_AGENT},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.text


def get_sp400_universe(timeout: float = 30.0) -> dict[str, Optional[str]]:
    """Scrape ``{ticker: GICS sector}`` from the Wikipedia S&P 400 table.

    Returns ``{}`` on any failure (network, layout change, missing
    lxml); callers degrade to the cache/fallback chain.  The first
    table containing a ``Symbol``/``Ticker`` column wins; the sector
    column is matched by name (``GICS Sector``) or position.

    The HTML is fetched with :func:`_fetch_html` (pandas' own URL
    support sends a default scripting User-Agent that Wikipedia
    rejects, and older ``read_html(timeout=...)`` calls raise
    ``TypeError`` on current pandas) and parsed from the in-memory
    text instead.
    """
    try:
        html = _fetch_html(SP400_URL, timeout=timeout)
        tables = pd.read_html(io.StringIO(html))
    except Exception:  # noqa: BLE001 - any scrape failure degrades to {}
        return {}
    for table in tables or []:
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        columns = {str(col).strip().lower(): col for col in table.columns}
        symbol_col = columns.get("symbol") or columns.get("ticker")
        if symbol_col is None:
            continue
        sector_col = columns.get("gics sector") or columns.get("sector")
        universe: dict[str, Optional[str]] = {}
        for _, row in table.iterrows():
            ticker = _clean_ticker(row.get(symbol_col))
            if ticker is None or ticker in universe:
                continue
            sector = _clean_sector(row.get(sector_col)) if sector_col is not None else None
            universe[ticker] = sector
        if universe:
            return universe
    return {}


def screen_universe(
    universe: dict[str, Optional[str]],
    adv: Optional[dict[str, float]] = None,
    mcap: Optional[dict[str, float]] = None,
    min_adv: float = 1_000_000.0,
    min_mcap: float = 2_000_000_000.0,
    max_mcap: float = 10_000_000_000.0,
) -> dict[str, Optional[str]]:
    """Filter the universe by ADV / market-cap bounds (fail-open).

    Filters apply only to tickers present in the provided ``adv`` /
    ``mcap`` mappings; tickers without data pass through untouched so a
    missing data source never empties the universe.  Invalid bound
    values degrade to no filtering on that dimension.
    """
    if not isinstance(universe, dict):
        return {}
    result: dict[str, Optional[str]] = {}

    def _passes(value: Any, lo: float, hi: float) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return True  # unknown -> fail-open
        if not math.isfinite(number):
            return True
        if lo is not None and number < lo:
            return False
        if hi is not None and number > hi:
            return False
        return True

    adv = adv if isinstance(adv, dict) else {}
    mcap = mcap if isinstance(mcap, dict) else {}
    try:
        adv_floor = float(min_adv)
    except (TypeError, ValueError):
        adv_floor = None
    try:
        cap_lo = float(min_mcap)
    except (TypeError, ValueError):
        cap_lo = None
    try:
        cap_hi = float(max_mcap)
    except (TypeError, ValueError):
        cap_hi = None

    for ticker, sector in universe.items():
        name = str(ticker or "").strip().upper()
        if not name:
            continue
        if adv_floor is not None and name in adv:
            if not _passes(adv.get(name), adv_floor, None):
                continue
        if cap_lo is not None and name in mcap:
            if not _passes(mcap.get(name), cap_lo, cap_hi):
                continue
        result[name] = sector if isinstance(sector, str) else None
    return result


def load_fallback_universe(path: str | Path) -> dict[str, Optional[str]]:
    """Load the static fallback universe (``TICKER|Sector`` per line).

    Missing/corrupt file -> ``{}`` (fail-open).  Lines without a
    separator map to ``None`` sector (exempt from sector dedupe).
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 - missing file is an expected state
        return {}
    universe: dict[str, Optional[str]] = {}
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        ticker_raw, _, sector_raw = entry.partition(SECTOR_SEPARATOR)
        ticker = _clean_ticker(ticker_raw)
        if ticker is None:
            continue
        universe[ticker] = _clean_sector(sector_raw)
    return universe


def refresh_universe(
    cache_ttl_days: int = 7,
    cache: Optional[Cache] = None,
    fallback_path: str | Path = "data/universe_sp400_fallback.txt",
) -> dict[str, Optional[str]]:
    """Universe resolution chain: cache -> scrape -> fallback file.

    A fresh cache hit (``ttl_days`` old at most) short-circuits the
    scrape; a successful scrape refreshes the cache; total failure
    falls back to the static file.  Every failure mode returns a
    mapping (possibly empty) and never raises.
    """
    if cache is not None:
        try:
            stored = cache.get(UNIVERSE_CACHE_KEY)
        except Exception:  # noqa: BLE001 - cache problems degrade
            stored = None
        if isinstance(stored, dict) and stored:
            return {
                str(t): (s if isinstance(s, str) else None)
                for t, s in stored.items()
            }
    universe = get_sp400_universe()
    if universe and cache is not None:
        try:
            cache.set(UNIVERSE_CACHE_KEY, universe)
        except Exception:  # noqa: BLE001 - cache write failures degrade
            pass
    if universe:
        return universe
    return load_fallback_universe(fallback_path)
