"""Overnight (halo / overnight-drift) cycle: enter at the close, exit at
the next open.

Empirical gate (src/backtest/replay.py ``--mode overnight``): the BUY
side of the quant signal carries a positive overnight Kelly fraction
(2y replay, whole watchlist: n=112, win ~0.64, fraction ~+0.10,
usable=True); the SELL side does not.  This module implements the
operational cycle around that finding:

* :func:`load_overnight_kelly` loads the BUY-side stats so sizing uses
  the overnight edge (never the intraday stats file, and never the
  pooled aggregate).
* :func:`select_entries` ranks approved BUY decisions by quant
  confidence and caps the number of concurrent overnight names.
* :func:`pending_exits` / :func:`exit_orders` make the next-morning
  exit mechanical and idempotent: every entry is exited at held
  quantity regardless of the fresh signal, and re-running the command
  never double-exits.

Entries and exits are recorded through :class:`src.store.session.SessionStore`
(kind ``overnight_entry`` / ``overnight_exit``), so a crash or PC
restart mid-overnight still yields a complete exit on the next run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

__all__ = [
    "load_overnight_kelly",
    "select_entries",
    "pending_exits",
    "exit_orders",
    "exclude_session_bought",
]


def load_overnight_kelly(stats_path: Any, min_trades: int = 30) -> Optional[dict]:
    """BUY-side overnight Kelly mapping for the sizer (``None`` when unusable).

    Reads ``per_side.BUY`` from the overnight replay stats file — the
    strategy this cycle actually trades.  Same usability rules as the
    intraday loader: positive finite fraction and ``n >= min_trades``.
    """
    try:
        raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    per_side = raw.get("per_side") if isinstance(raw, dict) else None
    buy = per_side.get("BUY") if isinstance(per_side, dict) else None
    if not isinstance(buy, dict):
        return None
    try:
        fraction = float(buy.get("kelly_fraction"))
        n_trades = int(buy.get("n_trades"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fraction) or fraction <= 0.0 or n_trades < min_trades:
        return None
    return {"fraction": fraction, "n": n_trades}


def select_entries(decisions: List[Mapping[str, Any]], max_names: int) -> Set[str]:
    """Tickers allowed to enter tonight: approved BUYs, best confidence first.

    ``max_names`` caps concurrent overnight exposure (concentration
    guard).  Non-approvals, non-BUY signals and junk payloads never
    qualify.
    """
    try:
        cap = max(0, int(max_names))
    except (TypeError, ValueError):
        cap = 0
    ranked: List[tuple] = []
    for decision in decisions or []:
        if not isinstance(decision, Mapping):
            continue
        quant = decision.get("quant") or {}
        veto = decision.get("veto") or {}
        ticker = str(decision.get("ticker") or "").upper()
        if not ticker:
            continue
        if str(quant.get("signal") or "").upper() != "BUY":
            continue
        if not veto.get("approved"):
            continue
        try:
            confidence = float(quant.get("confidence"))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence):
            continue
        ranked.append((confidence, ticker))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return {ticker for _, ticker in ranked[:cap]}


def exclude_session_bought(
    selected: Set[str],
    counts: Mapping[str, Mapping[str, int]],
    max_buys: int = 1,
) -> Set[str]:
    """Drop candidates that already consumed their session BUY budget today.

    Closes the cross-mode guard in both directions: the session loop's
    gate counts overnight entries (shared session store), and this
    filter makes overnight entries respect the session loop's per-day
    BUY budget — so running both modes the same day can never buy the
    same ticker twice.
    """
    try:
        cap = max(0, int(max_buys))
    except (TypeError, ValueError):
        cap = 1
    return {
        ticker for ticker in selected
        if (counts.get(ticker) or {}).get("buy", 0) < cap
    }


def pending_exits(
    actions: List[Mapping[str, Any]],
    trade_date: str,
) -> Set[str]:
    """Tickers holding overnight entries that still need today's exit.

    A scan over chronological actions: an entry opens a position and a
    later exit for the same ticker closes it.  Idempotent — entries
    already closed (today or later) never appear again.
    """
    open_positions: Set[str] = set()
    for action in sorted(actions or [], key=lambda a: a.get("id") or 0):
        ticker = str(action.get("ticker") or "").upper()
        kind = str(action.get("kind") or "")
        if kind == "overnight_entry" and ticker:
            open_positions.add(ticker)
        elif kind == "overnight_exit" and ticker:
            open_positions.discard(ticker)
    return {ticker for ticker in open_positions}


def exit_orders(
    tickers: Set[str],
    positions: Mapping[str, float],
    price_map: Mapping[str, float],
) -> List[Dict[str, Any]]:
    """Market-sell order dicts at held quantity for the pending tickers.

    A ticker with no live position (already flat, or the paper account
    was reset) is skipped — never re-sold.  ``limit_price`` carries the
    reference price for the audit trail only; the broker submits market
    orders.
    """
    orders: List[Dict[str, Any]] = []
    for ticker in sorted(tickers):
        try:
            held = float(positions.get(ticker, 0.0))
        except (TypeError, ValueError):
            held = 0.0
        if held <= 0.0:
            continue
        try:
            reference = float(price_map.get(ticker, 0.0))
        except (TypeError, ValueError):
            reference = 0.0
        if not math.isfinite(reference) or reference <= 0.0:
            continue
        orders.append({
            "ticker": ticker,
            "side": "sell",
            "quantity": held,
            "limit_price": reference,
            "rationale": "overnight exit (mechanical next-open sale)",
        })
    return orders
