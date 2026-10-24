"""Execution cost model for realistic backtesting.

Models bid-ask spread and slippage for market orders (Alpaca paper: no commission).
All costs in basis points (1 bp = 0.01%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExecutionCosts:
    """Execution cost parameters.

    Attributes:
        spread_bps: Bid-ask spread in basis points (default 10 bps for liquid names).
        slippage_bps: Market order slippage in basis points (default 3 bps).
        scale_by_vol: If True, scale costs by daily_vol relative to 2% baseline.
    """
    spread_bps: float = 10.0
    slippage_bps: float = 3.0
    scale_by_vol: bool = True

    def round_trip_bps(self, daily_vol: Optional[float] = None) -> float:
        """Total round-trip cost in basis points.

        Round-trip = spread (cross once) + 2 * slippage (entry + exit).
        """
        base = self.spread_bps + 2.0 * self.slippage_bps
        if self.scale_by_vol and daily_vol is not None and daily_vol > 0:
            base *= (daily_vol / 0.02)
        return base

    def one_way_bps(self, daily_vol: Optional[float] = None) -> float:
        """One-way cost (entry OR exit) in basis points."""
        return self.round_trip_bps(daily_vol) / 2.0

    def adjust_entry(self, price: float, side: str, daily_vol: Optional[float] = None) -> float:
        """Adjust entry price for costs.

        BUY: pay more (price * (1 + cost))
        SELL: receive less (price * (1 - cost))
        """
        if price <= 0:
            return price
        cost = self.one_way_bps(daily_vol) / 1e4
        if side == "BUY":
            return price * (1.0 + cost)
        elif side == "SELL":
            return price * (1.0 - cost)
        return price

    def adjust_exit(self, price: float, side: str, daily_vol: Optional[float] = None) -> float:
        """Adjust exit price for costs.

        BUY exit (sell): receive less (price * (1 - cost))
        SELL exit (buy back): pay more (price * (1 + cost))
        """
        if price <= 0:
            return price
        cost = self.one_way_bps(daily_vol) / 1e4
        if side == "BUY":
            return price * (1.0 - cost)
        elif side == "SELL":
            return price * (1.0 + cost)
        return price


def costs_from_env() -> ExecutionCosts:
    """Build ExecutionCosts from environment variables."""
    import os
    try:
        spread = float(os.environ.get("SPREAD_BPS", "10.0"))
    except (TypeError, ValueError):
        spread = 10.0
    try:
        slippage = float(os.environ.get("SLIPPAGE_BPS", "3.0"))
    except (TypeError, ValueError):
        slippage = 3.0
    scale_str = os.environ.get("COST_SCALE_BY_VOL", "1").strip().lower()
    scale_by_vol = scale_str not in ("0", "false", "no")
    return ExecutionCosts(
        spread_bps=max(0.0, spread),
        slippage_bps=max(0.0, slippage),
        scale_by_vol=scale_by_vol,
    )