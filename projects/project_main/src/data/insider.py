"""Insider activity (SEC Form 4 / Form 144) via edgartools.

Public API: :func:`fetch_insider_activity`, returning a deterministic
status envelope and never raising (whale/flow data contract):

* ``status == "ok"``      -- filings found in the lookback window.
* ``status == "no_data"`` -- zero filings in the window (never an error).
* ``status == "error"``   -- infrastructure/config failure (missing
  ``EDGAR_IDENTITY``, network failure, whole-fetch breakdown).

Only ``ok``/``no_data`` envelopes are cached; error envelopes are retried
on the next run.  edgartools is imported lazily inside the fetch seams so
importing this module (and the whole offline test suite) never needs it.
"""

from __future__ import annotations

import math
import os
from datetime import date, datetime, timedelta
from typing import Any, Optional

from src.data.cache import Cache

__all__ = ["TTL_SECONDS", "fetch_insider_activity"]

#: Form 4 filings update daily; a 4 hour TTL keeps repeats cheap.
TTL_SECONDS: float = 4 * 3600.0

#: Rolling window for insider cluster detection (calendar days).
CLUSTER_WINDOW_DAYS = 30

#: Cap on rendered citation lines (prompt boundedness).
_MAX_CITATIONS = 10


def _edgar_identity() -> str:
    """SEC-required declared identity (``EDGAR_IDENTITY`` env var)."""
    return os.environ.get("EDGAR_IDENTITY", "").strip()


def _finite(value: Any) -> Optional[float]:
    """Coerce to a finite float (``None`` when missing/invalid)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _clean(value: Any) -> Any:
    """Scrub pandas-style NaN/NaT/None placeholders to ``None``."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if text.lower() in ("", "nan", "nat", "none", "<na>"):
        return None
    return value


def _iso_date(value: Any) -> Optional[str]:
    """Normalize a date-ish value to an ISO ``YYYY-MM-DD`` string.

    Handles ``date``/``datetime`` objects, ISO strings, and the
    ``MM/DD/YYYY`` strings Form 144 filings use.
    """
    value = _clean(value)
    if value is None:
        return None
    if hasattr(value, "date") and callable(value.date):
        try:
            value = value.date()
        except Exception:
            pass
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    for candidate in (text, text[:10]):
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError:
            continue
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _row_get(row: Any, *names: str) -> Any:
    """Read the first present, non-placeholder field off a row/dict."""
    for name in names:
        try:
            value = row.get(name)
        except AttributeError:
            value = getattr(row, name, None)
        value = _clean(value)
        if value is not None:
            return value
    return None


def _pick(obj: Any, *names: str) -> Any:
    """Read the first present, non-placeholder attribute off an object."""
    for name in names:
        value = _clean(getattr(obj, name, None))
        if value is not None:
            return value
    return None


# --------------------------------------------------------------------------- #
# Fetch seams (lazily import edgartools; raise on infrastructure failure)     #
# --------------------------------------------------------------------------- #


def _form4_transactions(ticker: str, days: int) -> list[dict[str, Any]]:
    """Normalized Form 4 transactions inside the lookback window (seam).

    Raises on infrastructure failure; the caller converts that into an
    ``error`` envelope.  A single unreadable filing is skipped.  Uses
    ``Form4.to_dataframe()`` (edgartools >= 5.31 normalizes each filing's
    transactions into columns ``Transaction Type, Code, Shares, Price,
    Value, Date, Insider, Position, Remaining Shares``).
    """
    from edgar import Company, set_identity

    identity = _edgar_identity()
    if identity:
        set_identity(identity)
    company = Company(ticker)
    filings = company.get_filings(form="4") or []
    cutoff = date.today() - timedelta(days=days)
    out: list[dict[str, Any]] = []
    for filing in filings:
        try:
            filing_date = _iso_date(getattr(filing, "filing_date", None))
            if filing_date is None:
                continue
            if date.fromisoformat(filing_date) < cutoff:
                continue
            form4 = filing.obj()
        except Exception:
            continue
        frame = None
        try:
            frame = form4.to_dataframe()
        except Exception:
            frame = None
        for tx in _rows_of(frame):
            row = _normalize_form4_row(tx)
            if row is not None:
                out.append(row)
    return out


def _form144_proposals(ticker: str, days: int) -> list[dict[str, Any]]:
    """Normalized Form 144 sale proposals inside the lookback window (seam).

    Uses the ``Form144`` object attributes (edgartools >= 5.31):
    ``person_selling``, ``approx_sale_date`` (``MM/DD/YYYY``),
    ``total_units_to_be_sold``.
    """
    from edgar import Company, set_identity

    identity = _edgar_identity()
    if identity:
        set_identity(identity)
    company = Company(ticker)
    filings = company.get_filings(form="144") or []
    cutoff = date.today() - timedelta(days=days)
    out: list[dict[str, Any]] = []
    for filing in filings:
        try:
            filing_date = _iso_date(getattr(filing, "filing_date", None))
            if filing_date is None:
                continue
            if date.fromisoformat(filing_date) < cutoff:
                continue
            obj = filing.obj()
        except Exception:
            continue
        shares = _finite(
            _pick(obj, "total_units_to_be_sold", "units_to_be_sold", "shares")
        )
        if shares is None or shares <= 0:
            continue  # a 144 without share counts is not actionable
        insider = _pick(obj, "person_selling", "reporting_owner", "name", "insider")
        held = _finite(_pick(obj, "shares_owned", "num_shares_held", "held"))
        out.append(
            {
                "insider": str(insider) if insider is not None else None,
                "proposed_sale_date": _iso_date(
                    _pick(obj, "approx_sale_date", "proposed_sale_date",
                          "proposedSaleDate")
                )
                or filing_date,
                "shares": shares,
                "held": held,
            }
        )
    return out


def _rows_of(transactions: Any) -> list[Any]:
    """Best-effort row extraction from a DataFrame or list of mappings."""
    if transactions is None:
        return []
    if hasattr(transactions, "iterrows"):  # pandas DataFrame
        return [row for _, row in transactions.iterrows()]
    if isinstance(transactions, dict):
        return [transactions]
    return list(transactions)


_TYPE_CODES = {"p": "buy", "s": "sell"}


def _classify_type(raw: Any) -> str:
    """Map an SEC transaction type/code onto buy/sell/other."""
    text = str(raw or "").strip().lower()
    if not text:
        return "other"
    code = text[0]
    if code == "p" or _TYPE_CODES.get(code) == "buy" or "purchase" in text or "buy" in text:
        return "buy"
    if code == "s" or _TYPE_CODES.get(code) == "sell" or "sale" in text or "sell" in text or "sold" in text:
        return "sell"
    return "other"


def _normalize_form4_row(row: Any) -> Optional[dict[str, Any]]:
    """Normalize one Form 4 transaction row into the envelope shape.

    Column candidates cover both the edgartools ``to_dataframe()`` output
    (``Insider, Position, Transaction Type, Code, Shares, Price, Value,
    Date``) and rawer per-filing row spellings.
    """
    shares = _finite(
        _row_get(row, "Shares", "transaction_shares", "shares", "transactionShares")
    )
    if shares is None or shares <= 0:
        return None
    price = _finite(
        _row_get(row, "Price", "transaction_price_per_share", "price",
                 "transactionPricePerShare")
    )
    value = _finite(_row_get(row, "Value", "transaction_value", "value", "transactionValue"))
    if value is None and price is not None:
        value = shares * price
    insider = _row_get(row, "Insider", "reporting_owner", "insider", "ownerName")
    title = _row_get(row, "Position", "reporting_owner_title", "officer_title", "title")
    ownership = _row_get(row, "ownership", "ownership_nature", "ownershipNature")
    return {
        "insider": str(insider) if insider is not None else None,
        "title": str(title) if title is not None else None,
        "transaction_date": _iso_date(
            _row_get(row, "Date", "transaction_date", "transactionDate", "filing_date")
        ),
        "type": _classify_type(
            _row_get(row, "Transaction Type", "Code", "transaction_type",
                     "transactionCode", "type")
        ),
        "shares": shares,
        "price": price,
        "value": value,
        "ownership": str(ownership) if ownership is not None else None,
    }


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def _cluster_count(events: list[tuple[str, str]], window_days: int = CLUSTER_WINDOW_DAYS) -> int:
    """Max distinct insiders transacting within any rolling ``window_days``.

    ``events`` are ``(iso_date, insider)`` pairs.  Windows are anchored at
    each event date (sorted), which covers the maximum count.
    """
    ordered = sorted((d, who) for d, who in events if d and who)
    if not ordered:
        return 0
    best = 0
    count = len(ordered)
    for i in range(count):
        start = date.fromisoformat(ordered[i][0])
        insiders: set[str] = set()
        for j in range(i, count):
            if (date.fromisoformat(ordered[j][0]) - start).days > window_days:
                break
            insiders.add(ordered[j][1])
        best = max(best, len(insiders))
    return best


def _citation_lines(form4: list[dict], form144: list[dict]) -> list[str]:
    """Human-readable evidence lines (capped, most significant first)."""
    lines: list[str] = []
    for tx in form4:
        if len(lines) >= _MAX_CITATIONS:
            break
        verb = {"buy": "bought", "sell": "sold"}.get(tx["type"], "transacted")
        line = f"Form 4: {tx['insider'] or 'insider'} {verb} {tx['shares']:,.0f} shares"
        if tx["price"]:
            line += f" @ ${tx['price']:g}"
        if tx["transaction_date"]:
            line += f" on {tx['transaction_date']}"
        lines.append(line)
    for proposal in form144:
        if len(lines) >= _MAX_CITATIONS:
            break
        line = f"Form 144: {proposal['insider'] or 'insider'} proposed sale of {proposal['shares']:,.0f} shares"
        if proposal["proposed_sale_date"]:
            line += f" on {proposal['proposed_sale_date']}"
        lines.append(line)
    return lines


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #


def fetch_insider_activity(
    ticker: str, cache: Cache | None = None, days: int = 90
) -> dict:
    """Fetch insider Form 4/144 activity for ``ticker`` (never raises).

    Cache key: ``insider:{ticker}:{days}d`` (TTL 4h).  Returns the status
    envelope described in the module docstring; ``cluster_buys`` /
    ``cluster_sells`` count the max distinct insiders transacting (buy /
    sell respectively) inside any rolling 30-day window, and
    ``net_insider_flow`` is buy value minus sell value in dollars.
    """
    days = max(int(days), 1)
    cache_key = f"insider:{ticker}:{days}d"

    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    envelope: dict[str, Any] = {
        "status": "no_data",
        "ticker": ticker,
        "window_days": days,
        "form4": [],
        "form144": [],
        "cluster_buys": 0,
        "cluster_sells": 0,
        "net_insider_flow": 0.0,
        "citations": [],
        "reason": None,
    }

    if not _edgar_identity():
        envelope["status"] = "error"
        envelope["reason"] = "EDGAR_IDENTITY not configured (required by the SEC)"
        return envelope

    try:
        form4 = _form4_transactions(ticker, days)
        form144 = _form144_proposals(ticker, days)
    except Exception as err:
        envelope["status"] = "error"
        envelope["reason"] = f"failed to fetch insider filings for {ticker!r}: {err}"
        return envelope

    envelope["form4"] = form4
    envelope["form144"] = form144

    if not form4 and not form144:
        envelope["status"] = "no_data"
    else:
        envelope["status"] = "ok"
        buys = [
            tx
            for tx in form4
            if tx["type"] == "buy" and tx["transaction_date"] and tx["insider"]
        ]
        sells = [
            tx
            for tx in form4
            if tx["type"] == "sell" and tx["transaction_date"] and tx["insider"]
        ]
        envelope["cluster_buys"] = _cluster_count(
            [(tx["transaction_date"], tx["insider"]) for tx in buys]
        )
        envelope["cluster_sells"] = _cluster_count(
            [(tx["transaction_date"], tx["insider"]) for tx in sells]
        )
        buy_value = math.fsum(tx["value"] or 0.0 for tx in buys)
        sell_value = math.fsum(tx["value"] or 0.0 for tx in sells)
        envelope["net_insider_flow"] = float(buy_value - sell_value)
        envelope["citations"] = _citation_lines(form4, form144)

    if cache is not None and envelope["status"] != "error":
        cache.set(cache_key, envelope)
    return envelope
