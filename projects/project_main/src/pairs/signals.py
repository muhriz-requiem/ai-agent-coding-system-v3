"""Spread signal generation -- long-only conditioned mean reversion.

The z-score comes from the PERSISTED pair book (``BookEntry``):
``z = (log(P_A) - alpha - beta*log(P_B) - spread_mean) / spread_std``.
Signal generation never re-fits; the book is the single source of
truth so signals are stable across cycles and restarts.

Long-only logic (no short leg -- this is a directional bet on the
cheap leg reverting upward relative to the expensive one):

  * ``z >= Z_ENTRY``: spread wide, A expensive / B cheap ->
    **BUY B**, ``SELL_EXIT`` A (skipped at sizing when unheld)
  * ``z <= -Z_ENTRY``: B expensive / A cheap -> **BUY A**, ``SELL_EXIT`` B
  * ``|z| < Z_EXIT``: ``SELL_EXIT`` both legs (take the reversion)
  * ``|z| > Z_STOP``: ``SELL_EXIT`` both legs (stop loss)

``SELL_EXIT`` is emitted regardless of holdings; the sizing layer caps
it at held quantity and skips it when nothing is held (the existing
SELL-as-exit semantics), so the no-short contract holds end to end.
Missing/invalid prices -> ``None`` (fail-open: no signal).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd

from src.pairs.book import BookEntry

__all__ = ["PairSignal", "generate_pair_signal", "generate_pair_signals"]


@dataclass
class PairSignal:
    """One pair's spread signal for the current cycle."""

    pair_id: str
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    alpha: float
    z_score: float
    signal_a: Literal["BUY", "HOLD", "SELL_EXIT"]
    signal_b: Literal["BUY", "HOLD", "SELL_EXIT"]
    confidence: float
    spread_mean: float
    spread_std: float
    price_a: float
    price_b: float
    spread_daily_vol: Optional[float] = None

    def leg_signal(self, ticker: str) -> Literal["BUY", "HOLD", "SELL_EXIT"]:
        """Signal for one leg (``HOLD`` when the ticker is not in the pair)."""
        if ticker == self.ticker_a:
            return self.signal_a
        if ticker == self.ticker_b:
            return self.signal_b
        return "HOLD"


def _price_of(current_prices: dict[str, Any], ticker: str) -> Optional[float]:
    """Reference price for one ticker (float / Series / OHLCV frame)."""
    if not isinstance(current_prices, dict):
        return None
    raw = current_prices.get(ticker)
    if raw is None:
        raw = current_prices.get(ticker.upper())
    series: Any = raw
    if isinstance(series, pd.DataFrame):
        if "Close" in series.columns:
            series = series["Close"]
        elif len(series.columns):
            series = series.iloc[:, 0]
        if isinstance(series, pd.DataFrame):
            if series.empty:
                return None
            series = series.iloc[:, 0]
    if isinstance(series, pd.Series):
        if series.empty:
            return None
        try:
            series = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:  # noqa: BLE001 - junk -> no price
            return None
        if series.empty:
            return None
        raw = series.iloc[-1]
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def generate_pair_signal(
    entry: BookEntry,
    current_prices: dict[str, Any],
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
) -> Optional[PairSignal]:
    """One pair's signal from the persisted book entry; ``None`` fail-open."""
    if entry is None:
        return None
    price_a = _price_of(current_prices, entry.ticker_a)
    price_b = _price_of(current_prices, entry.ticker_b)
    if price_a is None or price_b is None:
        return None
    try:
        beta = float(entry.hedge_ratio)
        alpha = float(entry.alpha)
        mean = float(entry.spread_mean)
        std = float(entry.spread_std)
        z_entry = float(z_entry)
        z_exit = float(z_exit)
        z_stop = float(z_stop)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (beta, alpha, mean, std,
                                          z_entry, z_exit, z_stop)):
        return None
    if std <= 0.0 or beta <= 0.0 or price_a <= 0.0 or price_b <= 0.0:
        return None

    spread = math.log(price_a) - alpha - beta * math.log(price_b)
    z = (spread - mean) / std
    if not math.isfinite(z):
        return None

    if abs(z) > z_stop:
        signal_a: Literal["BUY", "HOLD", "SELL_EXIT"] = "SELL_EXIT"
        signal_b: Literal["BUY", "HOLD", "SELL_EXIT"] = "SELL_EXIT"
    elif z >= z_entry:
        # Spread wide: A expensive, B cheap -> long B.
        signal_a, signal_b = "SELL_EXIT", "BUY"
    elif z <= -z_entry:
        # Spread wide the other way: B expensive, A cheap -> long A.
        signal_a, signal_b = "BUY", "SELL_EXIT"
    elif abs(z) < z_exit:
        signal_a, signal_b = "SELL_EXIT", "SELL_EXIT"
    else:
        signal_a, signal_b = "HOLD", "HOLD"

    reference = z_entry if z_entry > 0 else 1.0
    confidence = max(0.0, min(1.0, abs(z) / reference))

    return PairSignal(
        pair_id=entry.pair_id,
        ticker_a=entry.ticker_a,
        ticker_b=entry.ticker_b,
        hedge_ratio=beta,
        alpha=alpha,
        z_score=z,
        signal_a=signal_a,
        signal_b=signal_b,
        confidence=confidence,
        spread_mean=mean,
        spread_std=std,
        price_a=price_a,
        price_b=price_b,
        spread_daily_vol=(
            float(entry.spread_daily_vol)
            if entry.spread_daily_vol is not None
            else None
        ),
    )


def generate_pair_signals(
    book: Any,
    current_prices: dict[str, Any],
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
) -> list[PairSignal]:
    """Signals for every entry in the pair book (deterministic order)."""
    entries = list(getattr(book, "entries", {}).values()) if book is not None else []
    signals: list[PairSignal] = []
    for entry in sorted(entries, key=lambda e: e.pair_id):
        signal = generate_pair_signal(
            entry, current_prices, z_entry=z_entry, z_exit=z_exit, z_stop=z_stop
        )
        if signal is not None:
            signals.append(signal)
    return signals
