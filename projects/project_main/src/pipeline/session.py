"""Cross-cycle discipline for the intraday session loop.

The one-shot pipeline enforces every sizing cap *per invocation*; a
session loop that simply re-ran it would re-submit the same orders each
cycle and let every cycle independently consume the full equity budget.
This module adds the cross-cycle layer:

* :class:`SubmissionGate` decides, per cycle, which planner-emitted
  orders to actually submit given today's logged actions, currently
  open broker orders, per-ticker daily budgets, and the remaining
  session-wide BUY budget.
* :func:`seconds_until` parses Alpaca clock timestamps so the loop can
  sleep until open in Ctrl-C-responsive chunks.
* :func:`order_notional` computes the budget-consumed notional of a
  planner order (dict or OrderPlan).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = ["order_notional", "seconds_until", "SubmissionGate"]


def order_notional(order: Any) -> float:
    """``quantity x limit_price`` for a planner order (dict or object).

    Junk-safe: anything non-numeric or non-positive is 0.0, which never
    consumes budget.
    """
    if isinstance(order, dict):
        quantity = order.get("quantity")
        price = order.get("limit_price")
    else:
        quantity = getattr(order, "quantity", None)
        price = getattr(order, "limit_price", None)
    try:
        value = float(quantity) * float(price)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0.0 else 0.0


def order_ticker(order: Any) -> str:
    if isinstance(order, dict):
        return str(order.get("ticker") or "").upper()
    return str(getattr(order, "ticker", "") or "").upper()


def order_side(order: Any) -> str:
    if isinstance(order, dict):
        return str(order.get("side") or "").lower()
    return str(getattr(order, "side", "") or "").lower()


def seconds_until(iso_ts: Any, now: Optional[datetime] = None) -> Optional[float]:
    """Seconds from ``now`` until an ISO-8601 timestamp (Alpaca clock).

    Returns ``None`` for junk input or a timestamp already in the past.
    Naive timestamps are interpreted in UTC.
    """
    if not isinstance(iso_ts, str) or not iso_ts:
        return None
    try:
        target = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    delta = (target - current).total_seconds()
    return delta if delta > 0.0 else None


class SubmissionGate:
    """Per-cycle submission filter enforcing session-wide budgets.

    The planner already caps sizing within one invocation; the gate adds
    the cross-cycle rules on top of its output:

    * at most ``max_buys_per_ticker`` BUYs and ``max_sells_per_ticker``
      SELLs per ticker per trade date (counts come from SessionStore),
    * session-wide cumulative BUY notional stays within ``buy_budget``,
    * no duplicate submission while an open order exists for the same
      (symbol, side).
    """

    def __init__(
        self,
        max_buys_per_ticker: int = 1,
        max_sells_per_ticker: int = 1,
    ) -> None:
        self.max_buys = max(0, int(max_buys_per_ticker))
        self.max_sells = max(0, int(max_sells_per_ticker))

    def gate(
        self,
        orders: Iterable[Any],
        counts: Mapping[str, Mapping[str, int]],
        open_pairs: Set[Tuple[str, str]],
        buy_notional_used: float,
        buy_budget: float,
    ) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        """Split planner orders into ``(allowed, skipped)``.

        ``skipped`` entries are ``(order, reason)`` pairs; reasons are
        stable strings suitable for the session log.  BUY notional is
        consumed in order, so a partial budget admits the earliest
        orders and skips the rest.
        """
        allowed: List[Any] = []
        skipped: List[Tuple[Any, str]] = []
        used = float(buy_notional_used)
        budget = float(buy_budget)
        # Local per-ticker counts seeded from today's store counts so the
        # gate also de-duplicates repeated orders inside one planner batch.
        local_counts: Dict[str, Dict[str, int]] = {
            ticker: dict(buckets) for ticker, buckets in counts.items()
        }
        for order in orders or []:
            ticker = order_ticker(order)
            side = order_side(order)
            if side not in ("buy", "sell") or not ticker:
                skipped.append((order, "malformed order"))
                continue
            bucket = local_counts.setdefault(ticker, {"buy": 0, "sell": 0})
            if side == "buy":
                if bucket.get("buy", 0) >= self.max_buys:
                    skipped.append((order, "daily buy budget used"))
                    continue
                notional = order_notional(order)
                if used + notional > budget:
                    skipped.append((order, "session buy budget exhausted"))
                    continue
                used += notional
            else:
                if bucket.get("sell", 0) >= self.max_sells:
                    skipped.append((order, "daily sell budget used"))
                    continue
            bucket[side] = bucket.get(side, 0) + 1
            if (ticker, side) in open_pairs:
                bucket[side] -= 1
                skipped.append((order, "open order exists"))
                continue
            allowed.append(order)
        return allowed, skipped
