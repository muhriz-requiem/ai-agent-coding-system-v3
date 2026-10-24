"""Institutional 13F holder data via the SecuritiesDB API.

Public API: :func:`fetch_institutional_holdings`; same envelope contract
as :mod:`src.data.insider` (never raises; ``ok``/``no_data``/``error``).

Backend note: edgartools parses 13F filings *filed by* an institution,
but "who holds ticker X" needs a reverse lookup that is impractical
offline, so this module uses a free unified institutional-data API
(SecuritiesDB; see .env.example for sourcing the key) behind a thin HTTP
seam.  The endpoint and field names must be verified with a smoke script
before relying on live data; the parsing below is defensive -- unknown
shapes degrade to ``no_data``.  Field-name candidate lists tolerate
common spellings, and the base URL can be overridden via
``SECURITIESDB_BASE_URL``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

from src.data.cache import Cache

__all__ = ["TTL_SECONDS", "fetch_institutional_holdings"]

#: Quarterly filings; a 24 hour TTL keeps repeats cheap.
TTL_SECONDS: float = 24 * 3600.0

#: Default base URL (verify with a smoke script; override via env).
DEFAULT_BASE_URL = "https://api.securitiesdb.com/v1"

#: Size caps (prompt boundedness).
_MAX_HOLDERS = 10
_MAX_WHALE_ALERTS = 3
_MAX_CITATIONS = 10


def _api_key() -> str:
    return os.environ.get("SECURITIESDB_API_KEY", "").strip()


def _base_url() -> str:
    return os.environ.get("SECURITIESDB_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _fetch_raw(ticker: str) -> Any:
    """HTTP seam: raw institutional-holdings payload (raises on failure)."""
    import httpx

    response = httpx.get(
        f"{_base_url()}/institutional/{ticker}",
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def _clean(value: Any) -> Any:
    """Scrub NaN/None-style placeholders to ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in ("", "nan", "nat", "none", "<na>", "null"):
        return None
    return value


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _holders_of(payload: Any) -> list[Any]:
    """Best-effort extraction of the holder list from a raw payload."""
    if not isinstance(payload, dict):
        return []
    for key in ("holders", "top_holders", "institutional_holders", "positions", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _summary_of(payload: Any) -> dict:
    """Best-effort extraction of the aggregate-summary mapping."""
    if isinstance(payload, dict):
        summary = payload.get("summary")
        if isinstance(summary, dict):
            return summary
        return payload  # tolerate flat payloads
    return {}


def _summary_number(summary: dict, *names: str) -> Optional[float]:
    for name in names:
        value = _finite(_clean(summary.get(name)))
        if value is not None:
            return value
    return None


def _summary_int(summary: dict, *names: str) -> Optional[int]:
    value = _summary_number(summary, *names)
    return int(value) if value is not None else None


def _holder(row: Any) -> Optional[dict[str, Any]]:
    """Normalize one holder row (defensive; ``None`` for unusable rows)."""
    if not isinstance(row, dict):
        return None

    def get(*names: str) -> Any:
        for name in names:
            value = _clean(row.get(name))
            if value is not None:
                return value
        return None

    manager = get("manager", "managerName", "holder", "name")
    if manager is None:
        return None
    cik = get("cik", "managerCik")
    return {
        "manager": str(manager),
        "cik": str(cik) if cik is not None else None,
        "shares": _finite(get("shares", "sharesHeld", "position")),
        "value": _finite(get("value", "marketValue", "positionValue")),
        "pct_outstanding": _finite(
            get("pct_outstanding", "pctOutstanding", "pctOfOutstanding")
        ),
        "change_qoq": _finite(get("change_qoq", "changeQoq", "change")),
    }


def _whale_alerts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Largest QoQ moves, magnitude-ordered (``change_qoq`` is fractional)."""
    alerts: list[dict[str, Any]] = []
    for row in rows:
        change = row.get("change_qoq")
        if change is None or change == 0.0 or change <= -1.0:
            continue
        shares = row.get("shares")
        value = row.get("value")
        if shares is None and value is None:
            continue
        factor = change / (1.0 + change)
        alerts.append(
            {
                "manager": row["manager"],
                "action": "added" if change > 0 else "trimmed",
                "shares_change": shares * factor if shares is not None else None,
                "value_change": value * factor if value is not None else None,
            }
        )
    alerts.sort(
        key=lambda a: abs(a["value_change"] if a["value_change"] is not None
                          else (a["shares_change"] or 0.0)),
        reverse=True,
    )
    return alerts[:_MAX_WHALE_ALERTS]


def _citation_lines(
    top_holders: list[dict[str, Any]], whale_alerts: list[dict[str, Any]]
) -> list[str]:
    lines: list[str] = []
    for alert in whale_alerts:
        if len(lines) >= _MAX_CITATIONS:
            break
        if alert["shares_change"] is not None:
            shares_text = f"{abs(alert['shares_change']):,.0f} shares"
        else:
            shares_text = "shares"
        value_text = (
            f" (${abs(alert['value_change']):,.0f})"
            if alert["value_change"] is not None
            else ""
        )
        lines.append(f"13F: {alert['manager']} {alert['action']} {shares_text}{value_text}")
    for row in top_holders:
        if len(lines) >= _MAX_CITATIONS:
            break
        shares_text = f"{row['shares']:,.0f}" if row["shares"] is not None else "?"
        value_text = f" (${row['value']:,.0f})" if row["value"] is not None else ""
        lines.append(f"13F: {row['manager']} holds {shares_text} shares{value_text}")
    return lines


def fetch_institutional_holdings(ticker: str, cache: Cache | None = None) -> dict:
    """Fetch 13F institutional holdings for ``ticker`` (never raises).

    Cache key: ``institutional:{ticker}`` (TTL 24h).  ``total_holders`` is
    the sample-size guard consumed by the risk manager's exodus veto
    (Rule 5).  ``change_qoq`` is interpreted as a fractional QoQ change
    (0.25 == +25%); whale alerts derive the approximate absolute change
    from it.
    """
    cache_key = f"institutional:{ticker}"

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    envelope: dict[str, Any] = {
        "status": "no_data",
        "ticker": ticker,
        "top_holders": [],
        "total_holders": 0,
        "total_institutional_pct": None,
        "new_positions": 0,
        "closed_positions": 0,
        "increased_positions": 0,
        "decreased_positions": 0,
        "net_flow": None,
        "whale_alerts": [],
        "citations": [],
        "reason": None,
    }

    if not _api_key():
        envelope["status"] = "error"
        envelope["reason"] = "SECURITIESDB_API_KEY not configured"
        return envelope

    try:
        payload = _fetch_raw(ticker)
    except Exception as err:
        envelope["status"] = "error"
        envelope["reason"] = f"failed to fetch 13F holdings for {ticker!r}: {err}"
        return envelope

    rows = [holder for holder in (_holder(row) for row in _holders_of(payload))
            if holder is not None]
    if not rows:
        envelope["status"] = "no_data"
        envelope["reason"] = "no 13F coverage"
        if cache is not None:
            cache.set(cache_key, envelope)
        return envelope

    summary = _summary_of(payload)
    rows.sort(
        key=lambda r: r["value"] if r["value"] is not None else -1.0, reverse=True
    )
    envelope["status"] = "ok"
    envelope["top_holders"] = rows[:_MAX_HOLDERS]
    total_holders = _summary_int(summary, "total_holders", "totalHolders")
    envelope["total_holders"] = (
        total_holders if total_holders is not None else len(rows)
    )
    envelope["total_institutional_pct"] = _summary_number(
        summary, "total_institutional_pct", "totalInstitutionalPct"
    )

    increased = _summary_int(summary, "increased_positions", "increasedPositions")
    decreased = _summary_int(summary, "decreased_positions", "decreasedPositions")
    if increased is None:
        increased = sum(1 for r in rows if (r["change_qoq"] or 0.0) > 0.0)
    if decreased is None:
        decreased = sum(1 for r in rows if (r["change_qoq"] or 0.0) < 0.0)
    envelope["increased_positions"] = increased
    envelope["decreased_positions"] = decreased

    envelope["new_positions"] = _summary_int(summary, "new_positions", "newPositions") or 0
    envelope["closed_positions"] = _summary_int(summary, "closed_positions", "closedPositions") or 0
    envelope["net_flow"] = _summary_number(summary, "net_flow", "netFlow")

    envelope["whale_alerts"] = _whale_alerts(rows)
    envelope["citations"] = _citation_lines(envelope["top_holders"], envelope["whale_alerts"])

    if cache is not None:
        cache.set(cache_key, envelope)
    return envelope
