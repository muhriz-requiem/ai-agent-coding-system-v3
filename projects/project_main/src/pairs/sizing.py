"""Pairs position sizing -- dataclass adapter over the pure planner.

The single source of truth for pair-leg order math lives in
``src.agents.portfolio_manager`` (its pinned import contract forbids
importing ``src.pairs``, so the pure, duck-typed implementation lives
there and this module converts the pairs dataclasses into the plain
dict shape the planner consumes).

``size_pair_legs`` plans ONE pair:

  * base budget = ``equity * pair_max_pct`` (3%), split 50/50 across
    the two legs (1.5% each),
  * BUY-leg factors: spread-vol scale-down, conviction
    ``clamp(|z| / z_entry, 0.25, 1.0)``, regime multiplier, cost
    buffer, pairs Kelly (the only scale-up),
  * SELL_EXIT legs are mechanical full exits of held quantity.

Account dict keys consumed (all with planner defaults):
    ``equity``, ``pair_max_pct``, ``pairs_total_pct``, ``pair_z_entry``,
    ``target_daily_vol``, ``cost_buffer_pct``, ``regime_budget_mult``,
    ``min_trade_notional``, ``min_trade_notional_pct``,
    ``pair_kelly``, ``pair_kelly_fraction_cap``, ``pair_kelly_max_mult``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from src.agents.models import OrderPlan
from src.agents.portfolio_manager import (
    DEFAULT_PAIR_KELLY_FRACTION_CAP,
    DEFAULT_PAIR_KELLY_MAX_MULT,
    _build_pair_order_plans,
    _kelly_multiplier,
)

__all__ = ["size_pair_legs", "pair_kelly_from_stats"]


def _signal_dict(signal: Any) -> dict:
    """PairSignal (or dict) -> plain dict for the planner."""
    if isinstance(signal, dict):
        return signal
    return {
        "pair_id": getattr(signal, "pair_id", ""),
        "ticker_a": getattr(signal, "ticker_a", ""),
        "ticker_b": getattr(signal, "ticker_b", ""),
        "hedge_ratio": getattr(signal, "hedge_ratio", None),
        "alpha": getattr(signal, "alpha", None),
        "z_score": getattr(signal, "z_score", None),
        "signal_a": getattr(signal, "signal_a", "HOLD"),
        "signal_b": getattr(signal, "signal_b", "HOLD"),
        "price_a": getattr(signal, "price_a", None),
        "price_b": getattr(signal, "price_b", None),
        "spread_daily_vol": getattr(signal, "spread_daily_vol", None),
    }


def _decision_dict(decision: Any) -> dict:
    """PairRiskDecision (or dict) -> plain dict for the planner."""
    if isinstance(decision, dict):
        return decision
    to_dict = getattr(decision, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return {
        "pair_id": getattr(decision, "pair_id", ""),
        "approved": getattr(decision, "approved", False),
        "veto_reasons": list(getattr(decision, "veto_reasons", []) or []),
    }


def size_pair_legs(
    signal: Any,
    risk_decision: Any,
    account: dict,
    positions: Optional[dict[str, float]] = None,
    kelly: Optional[dict] = None,
    regime: Optional[dict] = None,
) -> list[OrderPlan]:
    """Plan one pair's legs via the pure planner (shared cap logic).

    Args:
        signal: :class:`src.pairs.signals.PairSignal` (or dict).
        risk_decision: :class:`src.pairs.risk.PairRiskDecision` (or dict);
            unapproved pairs emit nothing.
        account: Account mapping (see module docstring for keys).
        positions: ``{ticker: qty}`` held quantities (SELL_EXIT capping).
        kelly: ``{"fraction": float, "n": int}`` from the pairs backtest
            stats file (absent -> no scale-up).
        regime: Batch regime snapshot (panic blocks BUY legs).

    Returns:
        The pair's :class:`OrderPlan` list (empty when vetoed/untradable).
    """
    pair_account = dict(account or {})
    if kelly is not None:
        pair_account.setdefault("pair_kelly", kelly)

    # Validate the sizing inputs the planner assumes (equity / dust
    # floor); junk account data plans nothing (fail-open no-op).
    try:
        equity_value = float(pair_account.get("equity"))
    except (TypeError, ValueError):
        return []
    if not math.isfinite(equity_value) or equity_value <= 0.0:
        return []
    try:
        min_notional = float(pair_account.get("min_trade_notional", 10.0))
        if not math.isfinite(min_notional) or min_notional < 0.0:
            min_notional = 10.0
    except (TypeError, ValueError):
        min_notional = 10.0

    pair_kelly_info = _kelly_multiplier(
        pair_account.get("pair_kelly"),
        pair_account.get("pair_kelly_fraction_cap",
                         DEFAULT_PAIR_KELLY_FRACTION_CAP),
        pair_account.get("pair_kelly_max_mult", DEFAULT_PAIR_KELLY_MAX_MULT),
    )
    state = {
        "account": pair_account,
        "equity": equity_value,
        "positions": positions or {},
        "sector_map": pair_account.get("sector_map") or {},
        "cluster_map": pair_account.get("cluster_map") or {},
        "sector_budget": None,
        "corr_budget": None,
        "restricted": False,
        "regime_block_buys": bool(
            isinstance(regime, dict) and regime.get("regime") == "panic"
        ),
        "regime_mult": 1.0,
        "regime_label": "",
        "cost_mult": 1.0,
        "target_daily_vol": pair_account.get("target_daily_vol"),
        "min_notional_floor": min_notional,
        "cumulative_notional": pair_account.get("cumulative_buy_notional", 0.0),
        "sector_notional": {},
        "corr_notional": {},
        "pair_notional": {},
        "pairs_total_notional": 0.0,
        "pair_kelly_mult": (
            pair_kelly_info["mult"] if pair_kelly_info else 1.0
        ),
    }
    try:
        state["cumulative_notional"] = max(0.0, float(
            pair_account.get("cumulative_buy_notional", 0.0)))
    except (TypeError, ValueError):
        state["cumulative_notional"] = 0.0
    if isinstance(regime, dict) and regime.get("regime") == "risk_off":
        try:
            state["regime_mult"] = max(0.0, min(
                1.0, float(pair_account.get("regime_budget_mult", 0.5))))
        except (TypeError, ValueError):
            state["regime_mult"] = 0.5
        state["regime_label"] = "risk_off"
    try:
        state["cost_mult"] = 1.0 - max(0.0, min(
            1.0, float(pair_account.get("cost_buffer_pct", 0.0))))
    except (TypeError, ValueError):
        state["cost_mult"] = 1.0

    return _build_pair_order_plans(
        [_decision_dict(risk_decision)],
        [_signal_dict(signal)],
        state,
    )


def pair_kelly_from_stats(stats_path: Any, min_trades: int = 30) -> Optional[dict]:
    """Load the pairs Kelly payload from ``pairs_backtest_stats.json``.

    Returns ``{"fraction": float, "n": int}`` when the aggregate sample
    is usable (>= ``min_trades`` trades, positive Kelly fraction);
    ``None`` otherwise (absent/stale/junk stats -> no scale-up).
    """
    import json
    import math
    from pathlib import Path

    try:
        raw = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent file is an expected state
        return None
    aggregate = raw.get("aggregate") if isinstance(raw, dict) else None
    if not isinstance(aggregate, dict):
        return None
    try:
        fraction = float(aggregate.get("kelly_fraction"))
        n_trades = int(aggregate.get("n_trades"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fraction) or fraction <= 0 or n_trades < min_trades:
        return None
    return {"fraction": fraction, "n": n_trades}
