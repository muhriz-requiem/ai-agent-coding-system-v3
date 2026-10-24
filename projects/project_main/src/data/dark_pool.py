"""FINRA ATS weekly summary -- dark-pool (TRF) volume proxy.

Public API: :func:`fetch_dark_pool_proxy`; same envelope contract as
:mod:`src.data.insider` (never raises; ``ok``/``no_data``/``error``).

The FINRA data API (``api.finra.org``) requires a free registered key
(``FINRA_API_KEY``); without one the fetcher fails open to an ``error``
envelope.  FINRA periodically renames dataset fields, so parsing is
defensive (candidate key lists; unknown shapes degrade to ``no_data``).
The baseline for the 4-week average and z-score is the ``weeks`` window
*excluding* the latest week, so the z-score measures how unusual the
latest week is against recent history.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

from src.data.cache import Cache

__all__ = ["TTL_SECONDS", "fetch_dark_pool_proxy"]

#: Weekly data; a 24 hour TTL keeps repeats cheap.
TTL_SECONDS: float = 24 * 3600.0

_DATASET = "weeklySummary"
_BASE_URL = "https://api.finra.org/data/group/otcMarket/name"

_VOLUME_KEYS = ("totalWeeklyShareQuantity", "weeklyShareQuantity", "shareQuantity")
_DATE_KEYS = ("weekStartDate", "calendarWeekStartDate", "weekDate", "summaryDate")
_MARKET_KEYS = ("market", "marketType", "summaryTypeCode")


def _api_key() -> str:
    return os.environ.get("FINRA_API_KEY", "").strip()


def _fetch_weekly(ticker: str) -> list[dict]:
    """HTTP seam: raw weekly-summary rows (raises on failure)."""
    import httpx

    body = {
        "limit": 5000,
        "compareFilters": [
            {
                "fieldName": "issueSymbolIdentifier",
                "fieldValue": ticker.upper(),
                "compareType": "EQUAL",
            }
        ],
    }
    response = httpx.post(
        f"{_BASE_URL}/{_DATASET}",
        json=body,
        headers={"X-API-KEY": _api_key(), "accept": "application/json"},
        timeout=30.0,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _row_get(row: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        try:
            value = row.get(name)
        except AttributeError:
            value = getattr(row, name, None)
        if value is not None:
            return value
    return None


def _weekly_buckets(rows: list[Any]) -> dict[str, dict[str, float]]:
    """Group rows into ``week -> {'trf': v, 'total': v}`` (ISO-date keys)."""
    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        volume = _finite(_row_get(row, _VOLUME_KEYS))
        week = str(_row_get(row, _DATE_KEYS) or "").strip()[:10]
        if volume is None or not week:
            continue
        market = str(_row_get(row, _MARKET_KEYS) or "").upper()
        slot = buckets.setdefault(week, {"trf": 0.0, "total": 0.0})
        if "TRF" in market:
            slot["trf"] += volume
        slot["total"] += volume
    return buckets


def _summarize(buckets: dict[str, dict[str, float]], weeks: int) -> Optional[dict[str, Any]]:
    """Reduce weekly buckets into the envelope aggregates (None = no data)."""
    ordered = sorted(buckets.items(), reverse=True)  # newest first (ISO dates)
    if not ordered:
        return None
    latest_week, latest_slot = ordered[0]
    if latest_slot["trf"] <= 0.0:
        return None

    baseline = [slot for _, slot in ordered[1 : weeks + 1] if slot["trf"] > 0.0]
    trf_volumes = [slot["trf"] for slot in baseline]
    avg_volume: Optional[float] = None
    zscore: Optional[float] = None
    if trf_volumes:
        avg_volume = math.fsum(trf_volumes) / len(trf_volumes)
        if len(trf_volumes) >= 2:
            mean = avg_volume
            variance = math.fsum((v - mean) ** 2 for v in trf_volumes) / len(trf_volumes)
            std = math.sqrt(variance)
            if std > 0.0:
                zscore = (latest_slot["trf"] - mean) / std

    def pct(slot: dict[str, float]) -> Optional[float]:
        return slot["trf"] / slot["total"] if slot["total"] > 0.0 else None

    pct_latest = pct(latest_slot)
    baseline_pcts = [p for p in (pct(slot) for slot in baseline) if p is not None]
    pct_avg: Optional[float] = None
    if baseline_pcts:
        pct_avg = math.fsum(baseline_pcts) / len(baseline_pcts)

    return {
        "dp_volume_latest": latest_slot["trf"],
        "dp_volume_4wk_avg": avg_volume,
        "dp_pct_latest": pct_latest,
        "dp_pct_4wk_avg": pct_avg,
        "dp_volume_zscore": zscore,
    }


def _citation_line(summary: dict[str, Any]) -> str:
    if summary["dp_pct_latest"] is not None and summary["dp_pct_4wk_avg"] is not None:
        return (
            f"FINRA ATS: TRF volume {summary['dp_pct_latest']:.0%} of total "
            f"(4wk avg {summary['dp_pct_4wk_avg']:.0%})"
        )
    if summary["dp_volume_latest"] is not None:
        avg_text = (
            f" (4wk avg {summary['dp_volume_4wk_avg']:,.0f})"
            if summary["dp_volume_4wk_avg"] is not None
            else ""
        )
        return f"FINRA ATS: TRF volume {summary['dp_volume_latest']:,.0f} shares{avg_text}"
    return "FINRA ATS: TRF data unavailable"


def fetch_dark_pool_proxy(
    ticker: str, cache: Cache | None = None, weeks: int = 4
) -> dict:
    """Fetch the dark-pool (TRF) volume proxy for ``ticker`` (never raises).

    Cache key: ``dark_pool:{ticker}:{weeks}w`` (TTL 24h).  ``block_trades``
    stays 0 until the block-summary dataset is integrated.
    """
    weeks = max(int(weeks), 2)
    cache_key = f"dark_pool:{ticker}:{weeks}w"

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    envelope: dict[str, Any] = {
        "status": "no_data",
        "ticker": ticker,
        "weeks": weeks,
        "dp_volume_latest": None,
        "dp_volume_4wk_avg": None,
        "dp_pct_latest": None,
        "dp_pct_4wk_avg": None,
        "dp_volume_zscore": None,
        "block_trades": 0,
        "citations": [],
        "reason": None,
    }

    if not _api_key():
        envelope["status"] = "error"
        envelope["reason"] = "FINRA_API_KEY not configured"
        return envelope

    try:
        rows = _fetch_weekly(ticker)
    except Exception as err:
        envelope["status"] = "error"
        envelope["reason"] = f"failed to fetch FINRA ATS data for {ticker!r}: {err}"
        return envelope

    summary = _summarize(_weekly_buckets(rows), weeks)
    if summary is None:
        envelope["status"] = "no_data"
        envelope["reason"] = "no TRF weekly volume in the FINRA response"
        if cache is not None:
            cache.set(cache_key, envelope)
        return envelope

    envelope["status"] = "ok"
    envelope.update(summary)
    envelope["citations"] = [_citation_line(summary)]

    if cache is not None:
        cache.set(cache_key, envelope)
    return envelope
