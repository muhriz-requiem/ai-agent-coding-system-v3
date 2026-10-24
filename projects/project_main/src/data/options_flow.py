"""Unusual options activity (UOA) scanner on yfinance chains.

Public API: :func:`fetch_unusual_options`; same envelope contract as
:mod:`src.data.insider` (never raises; ``ok``/``no_data``/``error``).

Scans the first ``_MAX_EXPIRIES`` expiries within ``max_dte`` days for
contracts whose volume/open-interest ratio and dollar premium clear the
configured thresholds.  yfinance is rate-limited (~1 req/s), so the scan
is async and sleeps ``_RATE_LIMIT_SECONDS`` between chain fetches (never
blocking the event loop -- graph.py awaits this coroutine directly).
Premium is approximated as ``volume * lastPrice * 100`` (contract
multiplier).  ``call_put_premium_ratio`` is computed over ALL unusual
contracts (not just the top-10 shown to the analyst) and is ``None`` when
there is no put-side premium (Rule 6 then skips the veto).
"""

from __future__ import annotations

import asyncio
import math
from datetime import date
from typing import Any, Optional

from src.data.cache import Cache

__all__ = ["TTL_SECONDS", "fetch_unusual_options"]

#: Intraday chain updates; a 2 hour TTL keeps repeats cheap.
TTL_SECONDS: float = 2 * 3600.0

#: yfinance rate limit: pause between chain fetches (overridable in tests).
_RATE_LIMIT_SECONDS = 1.0

#: Max expiries scanned per ticker (prompt boundedness + rate limits).
_MAX_EXPIRIES = 6

_CONTRACT_MULTIPLIER = 100.0

#: Size caps (prompt boundedness).
_MAX_CONTRACTS = 10
_MAX_CITATIONS = 10


def _yf_ticker(ticker: str) -> Any:
    """Lazy yfinance seam (keeps the module importable offline)."""
    import yfinance as yf

    return yf.Ticker(ticker)


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in ("", "nan", "nat", "none", "<na>"):
        return None
    return value


def _dte_of(expiry: Any, today: date) -> Optional[int]:
    """Days to expiry (``None`` for unparseable dates)."""
    expiry = _clean(expiry)
    if expiry is None:
        return None
    try:
        expiry_date = date.fromisoformat(str(expiry)[:10])
    except ValueError:
        return None
    return (expiry_date - today).days


def _rows(frame: Any) -> list[Any]:
    if frame is None:
        return []
    if hasattr(frame, "iterrows"):
        return [row for _, row in frame.iterrows()]
    return list(frame)


def _scan_frame(
    frame: Any,
    kind: str,
    expiry: Any,
    dte: int,
    min_vol_oi: float,
    min_premium: float,
) -> list[dict[str, Any]]:
    """Extract unusual contracts from one calls/puts frame."""
    out: list[dict[str, Any]] = []
    for row in _rows(frame):
        if hasattr(row, "get"):
            def get(*names: str) -> Any:
                for name in names:
                    value = _clean(row.get(name))
                    if value is not None:
                        return value
                return None
        else:
            def get(*names: str) -> Any:  # type: ignore[misc]
                for name in names:
                    value = _clean(getattr(row, name, None))
                    if value is not None:
                        return value
                return None

        volume = _finite(get("volume", "Volume"))
        oi = _finite(get("openInterest", "open_interest", "Open Interest"))
        if volume is None or oi is None or volume <= 0.0 or oi <= 0.0:
            continue
        ratio = volume / oi
        if ratio < min_vol_oi:
            continue
        price = _finite(get("lastPrice", "last_price", "lastTradePrice")) or 0.0
        premium = volume * price * _CONTRACT_MULTIPLIER
        if premium < min_premium:
            continue
        out.append(
            {
                "type": kind,
                "strike": _finite(get("strike", "Strike")),
                "expiry": str(expiry),
                "dte": dte,
                "volume": volume,
                "oi": oi,
                "vol_oi_ratio": ratio,
                "premium": premium,
                "iv": _finite(get("impliedVolatility", "iv")),
                "delta": None,
                "side": "buy" if volume > oi else "unknown",
            }
        )
    return out


def _citation_lines(contracts: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for contract in contracts[:_MAX_CITATIONS]:
        strike = f"{contract['strike']:g}" if contract["strike"] is not None else "?"
        lines.append(
            f"UOA: {contract['type']} {strike} {contract['expiry']} "
            f"vol {contract['volume']:,.0f}/oi {contract['oi']:,.0f} "
            f"({contract['vol_oi_ratio']:.1f}x) "
            f"${contract['premium'] / 1e6:.1f}M premium"
        )
    return lines


async def fetch_unusual_options(
    ticker: str,
    cache: Cache | None = None,
    min_vol_oi: float = 3.0,
    min_premium: float = 25000.0,
    max_dte: int = 45,
) -> dict:
    """Scan options chains for unusual activity (never raises).

    Cache key: ``options_flow:{ticker}:{min_vol_oi}:{min_premium}`` (TTL
    2h).  Returns the envelope described in the module docstring; the
    ``unusual_contracts`` list is size-capped to the top contracts by
    premium.
    """
    min_vol_oi = float(min_vol_oi)
    min_premium = float(min_premium)
    max_dte = int(max_dte)
    cache_key = f"options_flow:{ticker}:{min_vol_oi}:{min_premium}"

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    envelope: dict[str, Any] = {
        "status": "no_data",
        "ticker": ticker,
        "unusual_contracts": [],
        "call_put_premium_ratio": None,
        "total_unusual_premium": 0.0,
        "top_signal": None,
        "scanned_expiries": 0,
        "citations": [],
        "reason": None,
    }

    try:
        tk = _yf_ticker(ticker)
        expiries = list(getattr(tk, "options", None) or [])
    except Exception as err:
        envelope["status"] = "error"
        envelope["reason"] = f"failed to list options chain for {ticker!r}: {err}"
        return envelope

    if not expiries:
        envelope["status"] = "no_data"
        envelope["reason"] = "no listed options chain"
        if cache is not None:
            cache.set(cache_key, envelope)
        return envelope

    today = date.today()
    contracts: list[dict[str, Any]] = []
    call_premium = 0.0
    put_premium = 0.0
    scanned = 0
    try:
        for expiry in expiries:
            dte = _dte_of(expiry, today)
            if dte is None or dte < 0 or dte > max_dte:
                continue
            if scanned >= _MAX_EXPIRIES:
                break
            if scanned > 0 and _RATE_LIMIT_SECONDS > 0:
                await asyncio.sleep(_RATE_LIMIT_SECONDS)
            chain = tk.option_chain(expiry)
            scanned += 1
            calls = _scan_frame(
                getattr(chain, "calls", None), "call", expiry, dte,
                min_vol_oi, min_premium,
            )
            puts = _scan_frame(
                getattr(chain, "puts", None), "put", expiry, dte,
                min_vol_oi, min_premium,
            )
            contracts.extend(calls)
            contracts.extend(puts)
    except Exception as err:
        envelope["status"] = "error"
        envelope["reason"] = f"failed to fetch options chain for {ticker!r}: {err}"
        return envelope

    envelope["scanned_expiries"] = scanned
    if scanned == 0:
        envelope["status"] = "no_data"
        envelope["reason"] = f"no expiries within {max_dte}d"
        if cache is not None:
            cache.set(cache_key, envelope)
        return envelope
    if not contracts:
        envelope["status"] = "no_data"
        envelope["reason"] = (
            f"no contracts above vol/oi {min_vol_oi:g}x and ${min_premium:,.0f} premium"
        )
        if cache is not None:
            cache.set(cache_key, envelope)
        return envelope

    contracts.sort(key=lambda c: c["premium"], reverse=True)
    for contract in contracts:
        if contract["type"] == "call":
            call_premium += contract["premium"]
        else:
            put_premium += contract["premium"]

    envelope["status"] = "ok"
    envelope["unusual_contracts"] = contracts[:_MAX_CONTRACTS]
    envelope["total_unusual_premium"] = float(call_premium + put_premium)
    if put_premium > 0.0:
        envelope["call_put_premium_ratio"] = float(call_premium / put_premium)
    top = contracts[0]
    envelope["top_signal"] = {
        "type": top["type"],
        "strike": top["strike"],
        "expiry": top["expiry"],
        "premium": top["premium"],
        "vol_oi_ratio": top["vol_oi_ratio"],
    }
    envelope["citations"] = _citation_lines(contracts)

    if cache is not None:
        cache.set(cache_key, envelope)
    return envelope
