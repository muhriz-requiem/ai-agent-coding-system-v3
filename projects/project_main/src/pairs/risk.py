"""Pairs risk gates -- BUY-side vetoes, fail-open on absent data.

Gate set (plan's risk section; caps 5-7 live in the sizing layer's
shared accumulators, not here):

  1. Regime panic  -> block any signal carrying a BUY leg
     (SELL_EXIT legs always pass -- risk-reducing exits).
  2. Correlation breakdown: current return corr < 0.5 x formation corr
     -> veto (the pair's economic premise is gone).
  3. Half-life drift: current half-life > 2 x formation half-life
     -> veto (the spread no longer reverts on the tradeable horizon).
  4. Liquidity: either leg's current ADV < 50% of its formation ADV
     -> veto (slippage risk).

Gates 2-4 evaluate only when the data needed for the check is
provided; absent/corrupt data leaves the gate open (the same
fail-open shape as every deterministic veto in this pipeline).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from src.pairs.book import BookEntry
from src.pairs.cointegration import _ar1_slope, _close_series, spread_residuals
from src.pairs.signals import PairSignal

__all__ = ["PairRiskDecision", "evaluate_pair_risk"]

#: Live-return window for the correlation-breakdown gate.
_CORR_WINDOW = 60
#: Current corr below this fraction of formation corr -> veto.
DEFAULT_CORR_FLOOR = 0.5
#: Current half-life above this multiple of formation half-life -> veto.
DEFAULT_HALF_LIFE_MULT = 2.0
#: Current ADV below this fraction of formation ADV -> veto.
DEFAULT_ADV_FLOOR = 0.5


@dataclass
class PairRiskDecision:
    """Veto result for one pair signal (mirrors RiskDecision's shape)."""

    pair_id: str
    approved: bool
    veto_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe payload for the audit log / planner."""
        return {
            "pair_id": self.pair_id,
            "approved": self.approved,
            "veto_reasons": list(self.veto_reasons),
        }


def _sf(obj: Any, name: str, default: Any = None) -> Any:
    """Duck-typed field read: dicts and attribute objects both work."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _live_correlation(
    current_closes: dict[str, Any],
    ticker_a: str,
    ticker_b: str,
    window: int = _CORR_WINDOW,
) -> Optional[float]:
    """Recent return correlation for one pair; ``None`` fail-open."""
    if not isinstance(current_closes, dict):
        return None
    close_a = _close_series(current_closes.get(ticker_a))
    close_b = _close_series(current_closes.get(ticker_b))
    if close_a is None or close_b is None:
        return None
    frame = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).dropna()
    if window and window > 0:
        frame = frame.tail(int(window))
    if len(frame) < 10:
        return None
    returns = frame.pct_change().dropna()
    if len(returns) < 5:
        return None
    corr = float(returns["a"].corr(returns["b"]))
    return corr if math.isfinite(corr) else None


def _live_half_life(entry: Any, current_closes: dict[str, Any]) -> Optional[float]:
    """Current AR(1) half-life of the pair's spread; ``None`` fail-open.

    Rebuilds the residual series from the persisted alpha/beta (no
    re-fit of the hedge ratio) over the recent window and re-estimates
    only the AR(1) slope -- drift shows up there first.
    """
    if entry is None or not isinstance(current_closes, dict):
        return None
    close_a = current_closes.get(_sf(entry, "ticker_a", ""))
    close_b = current_closes.get(_sf(entry, "ticker_b", ""))
    if close_a is None or close_b is None:
        return None
    residuals = spread_residuals(entry, close_a, close_b, min_observations=30)
    if residuals is None or len(residuals) < 20:
        return None
    phi = _ar1_slope(residuals.to_numpy(dtype=float))
    if not math.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
        return None
    half_life = -math.log(2.0) / math.log(phi)
    return half_life if math.isfinite(half_life) else None


def _leg_adv(
    current_adv: Optional[dict[str, float]],
    ticker: str,
) -> Optional[float]:
    """Current dollar ADV for one leg; ``None`` when absent (fail-open)."""
    if not isinstance(current_adv, dict):
        return None
    value = current_adv.get(ticker)
    if value is None:
        value = current_adv.get(ticker.upper())
    try:
        adv = float(value)
    except (TypeError, ValueError):
        return None
    return adv if math.isfinite(adv) and adv > 0.0 else None


def evaluate_pair_risk(
    signal: PairSignal,
    entry: BookEntry,
    regime: Any = None,
    positions: Any = None,
    current_closes: Optional[dict[str, Any]] = None,
    current_adv: Optional[dict[str, float]] = None,
    corr_floor: float = DEFAULT_CORR_FLOOR,
    half_life_mult: float = DEFAULT_HALF_LIFE_MULT,
    adv_floor: float = DEFAULT_ADV_FLOOR,
) -> PairRiskDecision:
    """Evaluate the BUY-side veto gates for one pair signal.

    Args:
        signal: The pair's spread signal for this cycle.
        entry: The persisted book entry (formation stats).
        regime: Batch regime snapshot (``{"regime": "panic"|"risk_off"|...}``).
        positions: Unused at the veto layer (kept for signature symmetry
            with the single-leg risk seam); held-quantity capping happens
            at sizing.
        current_closes: Recent close Series / OHLCV frames per ticker for
            the drift gates (absent -> gates 2-3 fail open).
        current_adv: Current dollar ADV per ticker (absent -> gate 4
            fails open).
        corr_floor / half_life_mult / adv_floor: Gate thresholds.

    Returns:
        A :class:`PairRiskDecision`; ``approved=False`` with the fired
        veto reasons listed (never raises on data problems).
    """
    del positions  # exits are capped at held qty at sizing, not vetoed here
    reasons: list[str] = []
    pair_id = _sf(signal, "pair_id", "") or ""
    has_buy_leg = (
        _sf(signal, "signal_a", None) == "BUY"
        or _sf(signal, "signal_b", None) == "BUY"
    )

    # Gate 1: regime panic blocks risk-adding BUY legs (SELL_EXIT passes).
    if has_buy_leg and isinstance(regime, dict):
        if regime.get("regime") == "panic":
            reasons.append("regime panic blocks pair BUY")

    # Gate 2: correlation breakdown.
    formation_corr = _sf(entry, "formation_corr", None)
    if has_buy_leg and isinstance(formation_corr, (int, float)) \
            and math.isfinite(formation_corr) and formation_corr > 0.0:
        live = _live_correlation(
            current_closes,
            _sf(signal, "ticker_a", ""),
            _sf(signal, "ticker_b", ""),
        )
        try:
            floor = float(corr_floor)
        except (TypeError, ValueError):
            floor = DEFAULT_CORR_FLOOR
        if live is not None and live < floor * float(formation_corr):
            reasons.append(
                f"correlation breakdown: live {live:.2f} < "
                f"{floor:.2f} x formation {formation_corr:.2f}"
            )

    # Gate 3: half-life drift.
    formation_hl = _sf(entry, "half_life", None)
    if has_buy_leg and isinstance(formation_hl, (int, float)) \
            and math.isfinite(formation_hl) and formation_hl > 0.0:
        live_hl = _live_half_life(entry, current_closes or {})
        try:
            mult = float(half_life_mult)
        except (TypeError, ValueError):
            mult = DEFAULT_HALF_LIFE_MULT
        if live_hl is not None and live_hl > mult * float(formation_hl):
            reasons.append(
                f"half-life drift: live {live_hl:.1f}d > "
                f"{mult:.1f} x formation {formation_hl:.1f}d"
            )

    # Gate 4: liquidity deterioration on either leg.
    if has_buy_leg and entry is not None:
        formation_adv = _sf(entry, "formation_adv", None) or {}
        try:
            adv_ratio = float(adv_floor)
        except (TypeError, ValueError):
            adv_ratio = DEFAULT_ADV_FLOOR
        for leg in (_sf(signal, "ticker_a", ""), _sf(signal, "ticker_b", "")):
            formed = formation_adv.get(leg) if isinstance(formation_adv, dict) else None
            if not isinstance(formed, (int, float)) or not math.isfinite(formed) \
                    or formed <= 0.0:
                continue  # formation ADV unknown -> fail open
            live_adv = _leg_adv(current_adv, leg)
            if live_adv is not None and live_adv < adv_ratio * float(formed):
                reasons.append(
                    f"liquidity: {leg} ADV {live_adv:,.0f} < "
                    f"{adv_ratio:.0%} of formation {float(formed):,.0f}"
                )
                break  # one leg is enough to veto

    return PairRiskDecision(
        pair_id=pair_id,
        approved=not reasons,
        veto_reasons=reasons,
    )
