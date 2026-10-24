"""Portfolio manager: pure order-plan construction.

Sizing, batching, and pattern-day-trader (PDT) gating.

Import contract (pinned by tests/test_portfolio_manager.py): this module may
import ONLY ``from src.agents.models import OrderPlan, RiskDecision`` plus the
stdlib (math, datetime, typing).  No risk_manager, llm, report, or broker
imports.  All functions are pure -- no I/O, no mutation of inputs.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any, Iterable, Optional

from src.agents.models import OrderPlan, RiskDecision

__all__ = ["build_order_plan"]


# ---------------------------------------------------------------------------
# Date helpers (pure)
# ---------------------------------------------------------------------------


def _parse_iso_date(value: Any) -> Optional[date]:
    """Parse an ISO 'YYYY-MM-DD' string into a ``datetime.date``.

    Returns ``None`` for anything malformed/non-string so callers can skip
    bad entries instead of raising.  Values that are already ``date`` objects
    (but not ``datetime``) are passed through; ``datetime`` is reduced to its
    date part.
    """
    if isinstance(value, date) and not isinstance(value, datetime):  # type: ignore[name-defined]
        return value
    try:
        # ``datetime`` is referenced lazily to keep the module import surface
        # limited to the allowed stdlib names; it is part of ``datetime``.
        from datetime import datetime as _datetime

        if isinstance(value, _datetime):
            return value.date()
    except Exception:  # pragma: no cover - defensive only
        return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except (ValueError, TypeError):
        return None


def _business_days_ending_as_of(as_of: date, count: int = 5) -> set[date]:
    """Return the set of the ``count`` business days (Mon-Fri only) ending
    with ``as_of`` (inclusive), walking back over weekends."""
    window: set[date] = set()
    cursor = as_of
    while len(window) < count:
        if cursor.weekday() < 5:  # Monday=0 .. Friday=4
            window.add(cursor)
        cursor -= timedelta(days=1)
    return window


def _pdt_restricted(day_trade_dates: Iterable[Any], as_of: str) -> bool:
    """PDT rule: count day-trade dates within the 5 business days ending
    ``as_of`` (inclusive, Mon-Fri only).  Restricted when count >= 3."""
    anchor = _parse_iso_date(as_of)
    if anchor is None:
        return False
    window = _business_days_ending_as_of(anchor, 5)
    count = 0
    for entry in day_trade_dates or []:
        parsed = _parse_iso_date(entry)
        if parsed is not None and parsed in window:
            count += 1
            if count >= 3:
                return True
    return False


# ---------------------------------------------------------------------------
# Signal payload helpers (pure)
# ---------------------------------------------------------------------------

CONVICTION_REFERENCE = 0.95
MIN_CONVICTION_WEIGHT = 0.25
DEFAULT_TARGET_DAILY_VOL = 0.02
DEFAULT_MAX_SECTOR_PCT = 0.30
DEFAULT_REGIME_BUDGET_MULT = 0.50
#: Debate-winner opposition multiplier (scale-down-only; item 10).
DEFAULT_DEBATE_OPPOSE_MULT = 0.75
#: Round-trip transaction-cost buffer fraction (item 9).  Budgets are
#: multiplied by (1 - buffer) so sizing nets out estimated spread/costs.
#: Defaults to 0.0 (no-op) so enabling it is an explicit operator choice.
DEFAULT_COST_BUFFER_PCT = 0.0
#: Dust filter: orders below this notional (qty * price) are skipped.
DEFAULT_MIN_TRADE_NOTIONAL = 10.0
#: Equity-relative floor component (small-account operation): the
#: effective dust floor is the SMALLER of the absolute floor and
#: ``max(ALPACA_MIN_NOTIONAL, equity * pct)`` -- accounts at or above
#: $1,000 equity behave exactly as before, smaller accounts scale the
#: floor down (never below Alpaca's $1 minimum) so tiny budgets and
#: dust exits still trade.  An absolute floor ABOVE the default acts as
#: an explicit stricter override and wins over the relative scaling.
DEFAULT_MIN_TRADE_NOTIONAL_PCT = 0.01
#: Alpaca's minimum notional for fractional market orders.
ALPACA_MIN_NOTIONAL = 1.0
#: Correlated-basket cap as a fraction of equity (item 11): BUY notional
#: per correlation cluster is capped like a pseudo-sector.
DEFAULT_MAX_CORR_PCT = 0.30
#: Kelly sizing (item 7): the fraction from the backtest stats file maps
#: linearly onto a multiplier in [1.0, max_mult]; ``fraction_cap`` is the
#: edge strength considered "full Kelly" for that mapping.
DEFAULT_KELLY_FRACTION_CAP = 0.25
DEFAULT_KELLY_MAX_MULT = 1.5
# Pairs sizing (stat-arb module): per-pair notional cap counts BOTH legs
# combined (2 x 1.5% legs); the total-pairs cap is an additional rail on
# top of the shared equity / sector / cluster accumulators.
DEFAULT_PAIR_MAX_PCT = 0.03
DEFAULT_PAIRS_TOTAL_PCT = 0.30
DEFAULT_PAIR_Z_ENTRY = 2.0
DEFAULT_PAIR_KELLY_FRACTION_CAP = 0.5
DEFAULT_PAIR_KELLY_MAX_MULT = 1.5
#: Regime labels that act on order flow (others are treated as neutral).
REGIME_PANIC = "panic"
REGIME_RISK_OFF = "risk_off"


def _reference_price(payload: dict) -> Optional[float]:
    """Extract a usable, positive, finite reference price from a signal
    payload.  ``None`` means the ticker must be skipped."""
    raw = payload.get("reference_price", payload.get("price"))
    if raw is None:
        return None
    try:
        price = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price) or price <= 0.0:
        return None
    return price


def _normalized_signal(payload: dict) -> Optional[str]:
    """Normalize the signal to lowercase 'buy'/'sell'/'hold' (or None)."""
    raw = payload.get("signal")
    if not isinstance(raw, str):
        return None
    text = raw.strip().lower()
    if text in {"buy", "sell", "hold"}:
        return text
    return None


def _daily_vol(payload: dict) -> Optional[float]:
    """Extract a usable, positive, finite daily volatility from a signal
    payload.  ``None`` means the volatility factor must be left at 1.0."""
    raw = payload.get("daily_vol")
    if raw is None:
        return None
    try:
        vol = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(vol) or vol <= 0.0:
        return None
    return vol


def _vol_multiplier(payload: dict, target_daily_vol: Any) -> float:
    """Scale-DOWN-only volatility multiplier.

    ``min(1.0, target_daily_vol / daily_vol)``: high-volatility names get a
    smaller budget, low-volatility names never exceed the base budget.
    Missing/invalid inputs leave the factor at 1.0 (no scaling).
    """
    vol = _daily_vol(payload)
    if vol is None:
        return 1.0
    try:
        target = float(target_daily_vol)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(target) or target <= 0.0:
        return 1.0
    return min(1.0, target / vol)


def _conviction_multiplier(payload: dict) -> float:
    """Conviction multiplier from the quant confidence.

    ``clamp(confidence / CONVICTION_REFERENCE, MIN_CONVICTION_WEIGHT, 1.0)``
    where CONVICTION_REFERENCE is the maximum confidence the quant can emit
    (0.95).  Missing/invalid confidence leaves the factor at 1.0.
    """
    raw = payload.get("confidence")
    if raw is None or isinstance(raw, bool):
        return 1.0
    try:
        confidence = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(confidence):
        return 1.0
    weight = confidence / CONVICTION_REFERENCE
    return max(MIN_CONVICTION_WEIGHT, min(1.0, weight))


def _held_quantity(ticker: str, positions: Any) -> float:
    """Held quantity for ``ticker`` from a ``{ticker: qty}`` mapping.

    Falls back to the upper-cased ticker (Alpaca symbols are uppercase).
    Anything missing/invalid/non-positive counts as 0.0 (nothing held).
    """
    if not isinstance(positions, dict) or not positions:
        return 0.0
    raw = positions.get(ticker)
    if raw is None:
        raw = positions.get(ticker.upper())
    try:
        qty = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(qty) or qty <= 0.0:
        return 0.0
    return qty


def _z_score_text(payload: dict) -> str:
    """Best-effort extraction of the quant z-score for the rationale.

    Accepts common key spellings ('z_score', 'zscore', 'z', 'zscore_value')
    and also looks one level into a nested 'confidence' payload.  Falls back
    to 'unavailable' so the rationale always cites the quant z-score.
    """
    candidates = ("z_score", "zscore", "z", "z_score_value", "zscore_value")
    value: Any = None
    for key in candidates:
        if key in payload:
            value = payload[key]
            break
    if value is None:
        nested = payload.get("quant") or payload.get("confidence")
        if isinstance(nested, dict):
            for key in candidates:
                if key in nested:
                    value = nested[key]
                    break
        elif isinstance(nested, (int, float)) and not isinstance(nested, bool):
            # Some payloads encode the z-score directly as 'confidence'.
            value = nested
    if value is None:
        return "quant z-score unavailable"
    try:
        return f"quant z-score {float(value):.4f}"
    except (TypeError, ValueError):
        return f"quant z-score {value}"


def _regime_context(
    regime: Any, account: dict
) -> tuple[bool, float, str]:
    """Derive ``(block_buys, budget_mult, label)`` from a regime snapshot.

    * ``panic``    -> block buys, multiplier 1.0 (sizing untouched).
    * ``risk_off`` -> buys allowed, multiplier ``account['regime_budget_mult']``
      (default 0.5; clamped to [0.0, 1.0] -- never scales up).
    * anything else (including missing/invalid snapshots) -> neutral no-op:
      buys allowed, multiplier 1.0 (fail-open).
    """
    if not isinstance(regime, dict):
        return False, 1.0, ""
    label = regime.get("regime")
    if label == REGIME_PANIC:
        return True, 1.0, "panic"
    if label == REGIME_RISK_OFF:
        raw = account.get("regime_budget_mult", DEFAULT_REGIME_BUDGET_MULT) \
            if isinstance(account, dict) else DEFAULT_REGIME_BUDGET_MULT
        try:
            mult = float(raw)
        except (TypeError, ValueError):
            mult = DEFAULT_REGIME_BUDGET_MULT
        if not math.isfinite(mult) or mult < 0.0:
            mult = 0.0
        mult = min(mult, 1.0)
        return False, mult, "risk_off"
    return False, 1.0, ""


# ---------------------------------------------------------------------------
# Flow-conviction sizing (pure)
# ---------------------------------------------------------------------------

#: Directional blend weights for the whale/flow analyst reports
#: (renormalized over the reports actually present).
FLOW_WEIGHTS = {"insider": 0.3, "institutional": 0.3, "options_flow": 0.4}


def _report_field(report: Any, name: str) -> Any:
    """Read a field off a report object or dict (duck-typed)."""
    if isinstance(report, dict):
        return report.get(name)
    return getattr(report, name, None)


def _flow_conviction_multiplier(reports: Any, signal: str) -> float:
    """Directional blend of whale/flow analyst stances; scale-down-only.

    ``AnalystReport.confidence`` is NOT directional -- a bearish
    0.9-confidence report must not inflate a BUY.  Per report: agreement =
    confidence when the stance agrees with the signal (bullish for buy,
    bearish for sell), ``(1 - confidence)`` when opposed, ``0.5`` when
    neutral.  Weights renormalize over the reports actually present.
    All reports missing -> 1.0 (no-op).  The result is clamped to
    ``[MIN_CONVICTION_WEIGHT, 1.0]`` so the factor never scales a
    position up and never zeroes it out.
    """
    if not isinstance(reports, dict) or not reports:
        return 1.0
    bullish = str(signal or "").strip().lower() == "buy"
    weighted = 0.0
    total_weight = 0.0
    for name, weight in FLOW_WEIGHTS.items():
        report = reports.get(name)
        if report is None:
            continue
        stance = _report_field(report, "stance")
        try:
            confidence = float(_report_field(report, "confidence"))
        except (TypeError, ValueError):
            confidence = 0.0
        if not math.isfinite(confidence):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        if stance == "neutral":
            score = 0.5
        elif (stance == "bullish") == bullish:
            score = confidence
        else:
            score = 1.0 - confidence
        weighted += score * weight
        total_weight += weight
    if total_weight <= 0.0:
        return 1.0
    blended = weighted / total_weight
    return max(MIN_CONVICTION_WEIGHT, min(1.0, blended / CONVICTION_REFERENCE))


# ---------------------------------------------------------------------------
# Debate-winner sizing (pure)
# ---------------------------------------------------------------------------


def _debate_multiplier(debate: Any, signal: str, oppose_mult: Any) -> float:
    """Scale-down-only debate-winner multiplier (item 10).

    A ``bear`` debate winner shrinks BUYs (the debate argued against the
    risk-adding side); a ``bull`` winner shrinks SELLs symmetrically.
    ``undecided``/missing/invalid records leave the factor at 1.0.  The
    multiplier itself is clamped to ``[MIN_CONVICTION_WEIGHT, 1.0]`` so the
    factor never flips a side and never zeroes the budget out.
    """
    winner = _report_field(debate, "winner")
    if not isinstance(winner, str):
        return 1.0
    winner = winner.strip().lower()
    if winner not in {"bull", "bear"}:
        return 1.0
    try:
        mult = float(oppose_mult)
    except (TypeError, ValueError):
        mult = DEFAULT_DEBATE_OPPOSE_MULT
    if not math.isfinite(mult):
        mult = DEFAULT_DEBATE_OPPOSE_MULT
    mult = max(MIN_CONVICTION_WEIGHT, min(1.0, mult))
    adverse = (signal == "sell") if winner == "bull" else (signal == "buy")
    return mult if adverse else 1.0


def _cost_multiplier(cost_buffer_pct: Any) -> float:
    """Transaction-cost multiplier ``(1 - buffer)`` (item 9).

    Missing/invalid input leaves the factor at 1.0 (fail-open); finite
    numeric values are clamped to ``[0.0, 1.0]`` (same shape as the
    regime multiplier) so explicit over-range configs clamp rather than
    error.  The result scales budgets down so sizing nets out estimated
    round-trip costs.
    """
    if cost_buffer_pct is None or isinstance(cost_buffer_pct, bool):
        return 1.0
    try:
        buffer = float(cost_buffer_pct)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(buffer):
        return 1.0
    buffer = max(0.0, min(1.0, buffer))
    return 1.0 - buffer


def _kelly_multiplier(
    kelly: Any, fraction_cap: Any, max_mult: Any
) -> Optional[dict]:
    """Kelly stats -> sizing multiplier (item 7); ``None`` when inactive.

    ``kelly`` is the ``account['kelly']`` mapping ``{"fraction": float,
    "n": int}`` loaded from the backtest stats file (absent stats ->
    inactive -> factor 1.0, exactly the pre-feature behavior).  A
    non-positive fraction carries no usable edge and is also inactive.
    A positive fraction maps linearly: ``fraction == fraction_cap`` (or
    beyond) reaches ``max_mult``; the result never exceeds
    ``max(1.0, max_mult)``.  Junk config falls back to the defaults.
    """
    if not isinstance(kelly, dict):
        return None
    raw = kelly.get("fraction")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        fraction = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fraction) or fraction <= 0.0:
        return None
    try:
        cap = float(fraction_cap)
    except (TypeError, ValueError):
        cap = DEFAULT_KELLY_FRACTION_CAP
    if not math.isfinite(cap) or cap <= 0.0:
        cap = DEFAULT_KELLY_FRACTION_CAP
    try:
        ceiling = float(max_mult)
    except (TypeError, ValueError):
        ceiling = DEFAULT_KELLY_MAX_MULT
    if not math.isfinite(ceiling) or ceiling < 1.0:
        ceiling = 1.0
    ratio = min(fraction / cap, 1.0)
    mult = 1.0 + (ceiling - 1.0) * ratio
    mult = min(mult, ceiling)
    n = kelly.get("n")
    try:
        n = int(n) if n is not None else None
    except (TypeError, ValueError):
        n = None
    return {"mult": mult, "fraction": fraction, "n": n}


# ---------------------------------------------------------------------------
# Pairs order planning (pure, duck-typed; stat-arb module)
# ---------------------------------------------------------------------------
#
# Pair signals/decisions arrive as plain dicts (or same-named attributes)
# so this module stays inside its pinned import contract (models + stdlib
# only -- src.pairs converts its dataclasses to dicts before calling).
#
# Pair signal dict shape:
#   {"pair_id", "ticker_a", "ticker_b", "hedge_ratio", "z_score",
#    "signal_a": "BUY"|"HOLD"|"SELL_EXIT", "signal_b": ...,
#    "price_a", "price_b", "spread_daily_vol" (optional)}
# Pair decision dict shape: {"pair_id", "approved", "veto_reasons"}.
#
# Long-only contract: legs normalize BUY->buy, SELL_EXIT->sell (capped at
# held qty, skipped when unheld); nothing else is emitted.  BUY legs are
# blocked by PDT/panic exactly like single-leg BUYs; SELL_EXIT passes.
# Accumulators are SHARED with the single-leg plan: cumulative BUY <=
# equity, plus the sector / cluster caps when the leg appears in the
# maps.  Pairs additionally enforce per-pair (equity * pair_max_pct) and
# total-pairs (equity * pairs_total_pct) caps.  Pair plans are appended
# AFTER single-leg plans: when the shared equity cap binds, pair legs
# are skipped first (documented cap priority -- the single-leg edge
# carries the validated Kelly stats).

PAIR_LEG_ACTIONS = {"BUY": "buy", "SELL_EXIT": "sell"}


def _pair_field(obj: Any, name: str) -> Any:
    """Duck-typed field read off a pair signal/decision (dict or attr)."""
    return _report_field(obj, name)


def _pair_decision_map(pair_decisions: Any) -> dict[str, Any]:
    """Index pair decisions by ``pair_id`` (first occurrence wins)."""
    mapping: dict[str, Any] = {}
    if not pair_decisions:
        return mapping
    for decision in pair_decisions:
        if decision is None:
            continue
        pair_id = _pair_field(decision, "pair_id")
        if isinstance(pair_id, str) and pair_id and pair_id not in mapping:
            mapping[pair_id] = decision
    return mapping


def _pair_leg_budget(
    leg_base: float,
    signal: Any,
    account: dict,
    regime_mult: float,
    cost_mult: float,
    pair_kelly_mult: float,
    target_daily_vol: Any,
) -> Optional[float]:
    """Factor budget for one BUY leg (scale-down-only except Kelly)."""
    try:
        pair_z_entry = float(account.get("pair_z_entry", DEFAULT_PAIR_Z_ENTRY))
    except (TypeError, ValueError):
        pair_z_entry = DEFAULT_PAIR_Z_ENTRY
    if not math.isfinite(pair_z_entry) or pair_z_entry <= 0.0:
        pair_z_entry = DEFAULT_PAIR_Z_ENTRY

    # Conviction: clamp(|z| / z_entry, 0.25, 1.0) -- the z-score IS the
    # edge strength; missing/invalid z invalidates the leg entirely.
    z_raw = _pair_field(signal, "z_score")
    try:
        z = float(z_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(z):
        return None
    conviction = max(0.25, min(1.0, abs(z) / pair_z_entry))

    # Volatility scale: spread daily vol vs the account target (missing
    # -> 1.0, same fail-open shape as the single-leg vol factor).
    vol_mult = 1.0
    spread_vol_raw = _pair_field(signal, "spread_daily_vol")
    if spread_vol_raw is not None:
        try:
            spread_vol = float(spread_vol_raw)
            target = float(target_daily_vol)
            if math.isfinite(spread_vol) and spread_vol > 0.0 \
                    and math.isfinite(target) and target > 0.0:
                vol_mult = min(1.0, target / spread_vol)
        except (TypeError, ValueError):
            vol_mult = 1.0

    return leg_base * vol_mult * conviction * regime_mult * cost_mult * pair_kelly_mult


def _build_pair_order_plans(
    pair_decisions: Any,
    pair_signals: Any,
    state: dict,
) -> list[OrderPlan]:
    """Plan pair legs and update the shared accumulators in ``state``.

    ``state`` (built by :func:`build_order_plan`) carries the account
    config plus the shared cap accumulators; BUY-leg notions mutate
    ``cumulative_notional`` / ``sector_notional`` / ``corr_notional`` /
    ``pair_notional`` / ``pairs_total_notional`` in place.
    """
    plans: list[OrderPlan] = []
    if not pair_signals or not pair_decisions:
        return plans

    account = state["account"]
    equity = state["equity"]
    try:
        pair_max_pct = float(account.get("pair_max_pct", DEFAULT_PAIR_MAX_PCT))
    except (TypeError, ValueError):
        pair_max_pct = DEFAULT_PAIR_MAX_PCT
    if not math.isfinite(pair_max_pct) or pair_max_pct <= 0.0:
        pair_max_pct = DEFAULT_PAIR_MAX_PCT
    try:
        pairs_total_pct = float(account.get("pairs_total_pct", DEFAULT_PAIRS_TOTAL_PCT))
    except (TypeError, ValueError):
        pairs_total_pct = DEFAULT_PAIRS_TOTAL_PCT
    if not math.isfinite(pairs_total_pct) or pairs_total_pct <= 0.0:
        pairs_total_pct = DEFAULT_PAIRS_TOTAL_PCT

    pair_budget = equity * pair_max_pct          # per pair, BOTH legs
    leg_base = pair_budget * 0.5                 # 1.5% per leg at defaults
    pairs_budget = equity * pairs_total_pct      # 30% at defaults

    decisions_by_id = _pair_decision_map(pair_decisions)

    for signal in pair_signals:
        if signal is None:
            continue
        pair_id = _pair_field(signal, "pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            continue
        decision = decisions_by_id.get(pair_id)
        if decision is None or not bool(_pair_field(decision, "approved")):
            continue

        ticker_a = _pair_field(signal, "ticker_a")
        ticker_b = _pair_field(signal, "ticker_b")
        if not isinstance(ticker_a, str) or not isinstance(ticker_b, str):
            continue

        for ticker, leg_action, price_key in (
            (ticker_a, _pair_field(signal, "signal_a"), "price_a"),
            (ticker_b, _pair_field(signal, "signal_b"), "price_b"),
        ):
            action = PAIR_LEG_ACTIONS.get(
                str(leg_action or "").strip().upper()
            )
            if action is None:
                continue  # HOLD / unknown label -> no order
            price_raw = _pair_field(signal, price_key)
            try:
                price = float(price_raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price) or price <= 0.0:
                continue

            if action == "buy":
                # Risk-adding: PDT and panic block BUY legs outright.
                if state["restricted"] or state["regime_block_buys"]:
                    continue
                budget = _pair_leg_budget(
                    leg_base, signal, account,
                    state["regime_mult"], state["cost_mult"],
                    state["pair_kelly_mult"], state["target_daily_vol"],
                )
                if budget is None or budget <= 0.0:
                    continue
                raw_qty = budget / price
            else:
                # SELL_EXIT: mechanical full exit of held quantity --
                # the exit fires because the spread reverted, so no
                # conviction/vol/regime scaling applies (scaling it
                # would strand partial residues forever).  Skipped
                # when unheld (never a short); no cap consumption.
                held_qty = _held_quantity(ticker, state["positions"])
                if held_qty <= 0.0:
                    continue
                budget = held_qty * price
                raw_qty = held_qty

            quantity = math.floor(raw_qty * 1e6) / 1e6
            if quantity < 0.000001:
                continue
            notional = quantity * price
            if notional < state["min_notional_floor"]:
                continue

            z_raw = _pair_field(signal, "z_score")
            z_text = f"{float(z_raw):.2f}" if isinstance(z_raw, (int, float)) \
                and math.isfinite(float(z_raw)) else "n/a"

            if action == "buy":
                # Per-pair cap (both legs combined).
                pair_used = state["pair_notional"].get(pair_id, 0.0)
                if pair_used + notional > pair_budget:
                    continue
                # Total-pairs cap.
                if state["pairs_total_notional"] + notional > pairs_budget:
                    continue
                # Shared equity cap.
                if state["cumulative_notional"] + notional > equity:
                    continue
                # Shared sector / cluster caps (same accumulators as the
                # single-leg plan).
                sector = state["sector_map"].get(ticker) \
                    if isinstance(state["sector_map"], dict) else None
                if state["sector_budget"] is not None \
                        and isinstance(sector, str) and sector.strip():
                    used = state["sector_notional"].get(sector, 0.0)
                    if used + notional > state["sector_budget"]:
                        continue
                    state["sector_notional"][sector] = used + notional
                cluster = state["cluster_map"].get(ticker) \
                    if isinstance(state["cluster_map"], dict) else None
                if state["corr_budget"] is not None \
                        and isinstance(cluster, str) and cluster.strip():
                    used = state["corr_notional"].get(cluster, 0.0)
                    if used + notional > state["corr_budget"]:
                        continue
                    state["corr_notional"][cluster] = used + notional
                state["pair_notional"][pair_id] = pair_used + notional
                state["pairs_total_notional"] += notional
                state["cumulative_notional"] += notional

            plans.append(
                OrderPlan(
                    ticker=ticker,
                    side=action,
                    quantity=quantity,
                    limit_price=price,
                    rationale=(
                        f"{action.upper()} {ticker}: pair {pair_id} leg "
                        f"(spread z {z_text}"
                        + (f"; budget {budget:.2f}"
                           + (f", regime {state['regime_label']} "
                              f"x{state['regime_mult']:.2f}"
                              if state["regime_label"] else "")
                           + (f", cost x{state['cost_mult']:.3f}"
                              if state["cost_mult"] < 1.0 else "")
                           + (f", kelly x{state['pair_kelly_mult']:.2f}"
                              if state["pair_kelly_mult"] > 1.0 else "")
                           if action == "buy" else "; mechanical exit")
                        + f", pairs notional "
                          f"{state['pairs_total_notional']:.2f} / "
                          f"{pairs_budget:.2f})"
                    ),
                )
            )

    return plans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_order_plan(
    decisions: list[RiskDecision],
    signals: dict[str, dict],
    account: dict,
    day_trade_dates: list[str],
    as_of: str,
    positions: Optional[dict[str, float]] = None,
    sector_map: Optional[dict[str, Optional[str]]] = None,
    regime: Optional[dict] = None,
    analyst_reports: Optional[dict[str, dict]] = None,
    debates: Optional[dict[str, Any]] = None,
    cluster_map: Optional[dict[str, str]] = None,
    pair_decisions: Optional[list[Any]] = None,
    pair_signals: Optional[list[Any]] = None,
) -> list[OrderPlan]:
    """Build the executable order plan from approved risk decisions.

    Rules:
      * Only decisions with ``approved=True`` whose signal (from ``signals``)
        is BUY or SELL are considered; ``signals`` is the source of truth for
        the signal.
      * Sizing is sourced from ACCOUNT.  Base budget =
        ``account['equity'] * account['max_position_pct']``, then scaled:
        volatility factor ``min(1.0, account['target_daily_vol'] / payload
        'daily_vol')`` (scale-down only, 1.0 when absent) x conviction factor
        ``clamp(payload 'confidence' / 0.95, 0.25, 1.0)`` (1.0 when absent)
        x directional flow-conviction factor from ``analyst_reports[ticker]``
        (scale-down only, 1.0 when absent).  ``quantity = floor(budget /
        reference_price, 6dp)``; skip when quantity < 0.000001 (Alpaca min
        fractional qty).
      * ``analyst_reports`` maps ticker -> ``{analyst_name: AnalystReport}``
        for the whale/flow analysts.  Reports are only passed for analysts
        backed by a successful fetch, so absent/failed data never shrinks
        budgets; only reports backed by real data can scale down.
      * ``debates`` maps ticker -> debate record (object or dict with a
        ``winner`` field, duck-typed).  A ``bear`` winner shrinks BUYs and
        a ``bull`` winner shrinks SELLs by
        ``account['debate_oppose_mult']`` (default 0.75, clamped to
        [0.25, 1.0] -- scale-down only); ``undecided``/absent debates are
        a no-op.  Cited in the rationale as ``debate <winner> x<mult>``
        only when the factor actually applied.
      * Transaction costs (item 9): budgets are multiplied by
        ``(1 - account['cost_buffer_pct'])`` (default 0.0 = no-op,
        clamped to [0, 1]) so sizing nets out estimated round-trip costs
        when the operator enables a buffer, and orders below the
        effective minimum notional (qty * price) are skipped as dust --
        on both sides.  The effective floor is the smaller of the
        absolute ``account['min_trade_notional']`` (default 10.0) and
        ``max(1.0, equity * account['min_trade_notional_pct'])``
        (default pct 0.01): accounts >= $1,000 equity keep the absolute
        floor, smaller accounts scale down to Alpaca's $1 fractional
        minimum so tiny budgets (and dust exits) still trade.  An
        absolute floor above the default is an explicit stricter
        override and wins over the relative scaling.
      * Kelly sizing (item 7): when ``account['kelly']`` carries usable
        backtest stats (``{"fraction": f > 0, "n": n}``), budgets are
        multiplied by ``1 + (max_mult - 1) * min(f / fraction_cap, 1)``
        (defaults: fraction_cap 0.25, max_mult 1.5).  This is the ONLY
        factor that may scale a budget above the base; it is bounded by
        ``account['kelly_max_mult']`` and by the hard per-name cap
        ``equity * max_position_pct * max(1.0, kelly_max_mult)``.  The
        cumulative BUY cap remains ``equity``.  Absent/junk stats leave
        sizing untouched (factor 1.0); a non-positive fraction means no
        usable edge and is likewise inactive.  Cited in the rationale as
        ``kelly x<m> (n=<n>)`` only when active.
      * Inverse-vol risk parity (item 8): when ``account['invol_enabled']``
        (default True) and at least two same-day BUY candidates carry a
        valid ``daily_vol``, each candidate's budget is additionally
        capped at its equal-risk-contribution share of the batch's total
        BUY budget (weights proportional to ``1 / daily_vol``).  The
        parity number is a CAP only: low-vol names never exceed their
        factor budget, so overall sizing stays scale-down-only.
        Candidates without a valid vol are exempt and excluded from the
        weights.  Cited in the rationale as ``invol x<f>`` only when the
        cap actually bound.
      * SELL orders are capped at the held quantity from ``positions``
        (``{ticker: qty}``) and skipped entirely when nothing is held.  SELL
        notional does not consume the buy-side cumulative cap.
      * ``limit_price = reference_price`` (used for sizing/audit; market orders
        are submitted so this is not a limit price); ``side`` is lowercase buy/sell.
      * Rationale cites the quant z-score (and the flow factor when flow
        reports contributed).
      * Cumulative BUY notional (quantity * limit_price) must never exceed
        ``account['equity']``; breaching orders are skipped and iteration
        continues with later tickers.  Exactly-equal spend is allowed.
      * Sector cap: when ``sector_map`` maps tickers to sector labels, BUY
        notional per sector must never exceed
        ``account['max_sector_pct'] * equity`` (default 0.30); breaching
        orders are skipped.  Tickers without a sector label are exempt.
      * Correlation-cluster cap (item 11): when ``cluster_map`` maps
        tickers to cluster ids (from src/quant/correlation.py), BUY
        notional per cluster must never exceed
        ``account['max_corr_pct'] * equity`` (default 0.30) -- the same
        mechanics as the sector cap, with its own accumulator (a name can
        hit both caps).  Tickers absent from the map are exempt.
      * PDT: when >= 3 day trades fall within the 5 business days ending
        ``as_of`` (inclusive, Mon-Fri only), NO buy orders are emitted
        (sells -- risk-reducing exits -- are still allowed).
      * Regime: when ``regime['regime'] == 'panic'`` (VIX backwardation +
        elevated spot VIX), NO buy orders are emitted (sells pass).
        When ``'risk_off'``, the budget is multiplied by
        ``account['regime_budget_mult']`` (default 0.5, clamped to
        [0.0, 1.0] -- never scales up).  Missing/invalid snapshots are
        neutral (fail-open).  The active regime is cited in the rationale.
      * Pairs (stat-arb module): ``pair_decisions``/``pair_signals`` are
        plain dicts (see the pairs section above for the shape).  Pair
        legs obey the long-only contract (BUY legs / SELL_EXIT legs
        capped at held qty), the same PDT/panic BUY gates, the shared
        equity/sector/cluster accumulators, plus per-pair
        (``account['pair_max_pct']``, default 0.03, both legs) and
        total-pairs (``account['pairs_total_pct']``, default 0.30) caps.
        Pairs Kelly reads ``account['pair_kelly']`` with caps
        ``pair_kelly_fraction_cap`` (0.5) / ``pair_kelly_max_mult``
        (1.5) -- a separate stats file from the single-leg edge.  Pair
        plans are appended AFTER single-leg plans (cap priority: the
        shared equity cap skips pair legs first).

    Pure function: no I/O, inputs are never mutated.
    """
    plans: list[OrderPlan] = []
    if not decisions or not isinstance(signals, dict):
        return plans

    equity_raw = account.get("equity") if isinstance(account, dict) else None
    pct_raw = account.get("max_position_pct") if isinstance(account, dict) else None
    try:
        equity = float(equity_raw)
        max_position_pct = float(pct_raw)
    except (TypeError, ValueError):
        return plans
    if not math.isfinite(equity) or not math.isfinite(max_position_pct):
        return plans

    target_daily_vol = account.get("target_daily_vol", DEFAULT_TARGET_DAILY_VOL) \
        if isinstance(account, dict) else DEFAULT_TARGET_DAILY_VOL
    max_sector_pct = account.get("max_sector_pct", DEFAULT_MAX_SECTOR_PCT) \
        if isinstance(account, dict) else DEFAULT_MAX_SECTOR_PCT
    max_corr_pct = account.get("max_corr_pct", DEFAULT_MAX_CORR_PCT) \
        if isinstance(account, dict) else DEFAULT_MAX_CORR_PCT
    debate_oppose_mult = account.get(
        "debate_oppose_mult", DEFAULT_DEBATE_OPPOSE_MULT
    ) if isinstance(account, dict) else DEFAULT_DEBATE_OPPOSE_MULT
    cost_buffer_pct = account.get(
        "cost_buffer_pct", DEFAULT_COST_BUFFER_PCT
    ) if isinstance(account, dict) else DEFAULT_COST_BUFFER_PCT
    min_trade_notional = account.get(
        "min_trade_notional", DEFAULT_MIN_TRADE_NOTIONAL
    ) if isinstance(account, dict) else DEFAULT_MIN_TRADE_NOTIONAL
    try:
        min_notional_floor = float(min_trade_notional)
    except (TypeError, ValueError):
        min_notional_floor = DEFAULT_MIN_TRADE_NOTIONAL
    if not math.isfinite(min_notional_floor) or min_notional_floor < 0.0:
        min_notional_floor = DEFAULT_MIN_TRADE_NOTIONAL
    # Equity-relative floor component: the effective floor is the smaller
    # of the absolute floor and max($1, equity * pct), so large accounts
    # keep the absolute floor and small accounts can still trade (and
    # still exit dust positions).  An absolute floor above the default
    # is an explicit stricter override (e.g. the 1000000 kill switch)
    # and wins over the relative scaling.
    if min_notional_floor > DEFAULT_MIN_TRADE_NOTIONAL:
        pass  # explicit stricter absolute floor: no relative scaling
    else:
        min_trade_notional_pct = account.get(
            "min_trade_notional_pct", DEFAULT_MIN_TRADE_NOTIONAL_PCT
        ) if isinstance(account, dict) else DEFAULT_MIN_TRADE_NOTIONAL_PCT
        try:
            floor_pct = float(min_trade_notional_pct)
        except (TypeError, ValueError):
            floor_pct = DEFAULT_MIN_TRADE_NOTIONAL_PCT
        if not math.isfinite(floor_pct) or floor_pct < 0.0:
            floor_pct = DEFAULT_MIN_TRADE_NOTIONAL_PCT
        relative_floor = max(ALPACA_MIN_NOTIONAL, equity * floor_pct)
        min_notional_floor = min(min_notional_floor, relative_floor)

    base_budget = equity * max_position_pct
    sector_budget: Optional[float] = None
    if isinstance(sector_map, dict) and sector_map:
        try:
            cap = float(max_sector_pct)
        except (TypeError, ValueError):
            cap = float("nan")
        if math.isfinite(cap) and cap > 0.0 and math.isfinite(equity):
            sector_budget = equity * cap
    corr_budget: Optional[float] = None
    if isinstance(cluster_map, dict) and cluster_map:
        try:
            cap = float(max_corr_pct)
        except (TypeError, ValueError):
            cap = float("nan")
        if math.isfinite(cap) and cap > 0.0 and math.isfinite(equity):
            corr_budget = equity * cap
    restricted = _pdt_restricted(day_trade_dates or [], as_of)
    regime_block_buys, regime_mult, regime_label = _regime_context(regime, account)
    # Transaction-cost buffer (item 9): constant across the batch, so it
    # is resolved once here (also used by the pairs planner below).
    cost_mult = _cost_multiplier(cost_buffer_pct)
    cumulative_notional = 0.0
    sector_notional: dict[str, float] = {}
    corr_notional: dict[str, float] = {}
    invol_enabled = True
    if isinstance(account, dict) and "invol_enabled" in account:
        invol_enabled = bool(account.get("invol_enabled"))
    kelly_raw = account.get("kelly") if isinstance(account, dict) else None
    kelly_fraction_cap = account.get(
        "kelly_fraction_cap", DEFAULT_KELLY_FRACTION_CAP
    ) if isinstance(account, dict) else DEFAULT_KELLY_FRACTION_CAP
    kelly_max_mult = account.get(
        "kelly_max_mult", DEFAULT_KELLY_MAX_MULT
    ) if isinstance(account, dict) else DEFAULT_KELLY_MAX_MULT
    kelly_info = _kelly_multiplier(kelly_raw, kelly_fraction_cap, kelly_max_mult)
    kelly_mult = kelly_info["mult"] if kelly_info else 1.0
    # Hard per-name ceiling: Kelly is the only scale-up factor, and even
    # with it a name can never exceed the base budget times the ceiling.
    kelly_ceiling = max(1.0, kelly_max_mult) \
        if isinstance(kelly_max_mult, (int, float)) \
        and math.isfinite(kelly_max_mult) else DEFAULT_KELLY_MAX_MULT
    per_name_cap = equity * max_position_pct * max(1.0, kelly_ceiling)

    # Pairs Kelly (stat-arb): a separate stats file / cap mapping from
    # the single-leg edge; absent/junk stats are a no-op (factor 1.0).
    pair_kelly_raw = account.get("pair_kelly") \
        if isinstance(account, dict) else None
    pair_kelly_fraction_cap = account.get(
        "pair_kelly_fraction_cap", DEFAULT_PAIR_KELLY_FRACTION_CAP
    ) if isinstance(account, dict) else DEFAULT_PAIR_KELLY_FRACTION_CAP
    pair_kelly_max_mult = account.get(
        "pair_kelly_max_mult", DEFAULT_PAIR_KELLY_MAX_MULT
    ) if isinstance(account, dict) else DEFAULT_PAIR_KELLY_MAX_MULT
    pair_kelly_info = _kelly_multiplier(
        pair_kelly_raw, pair_kelly_fraction_cap, pair_kelly_max_mult
    )
    pair_kelly_mult = pair_kelly_info["mult"] if pair_kelly_info else 1.0

    # -- pass 1: eligibility + per-name factor budgets -----------------------
    # Every gate (approval, signal, PDT, panic, price) and multiplier is
    # resolved here; order emission waits for the batch-level parity caps.
    contexts: list[dict] = []
    for decision in decisions:
        if decision is None or not getattr(decision, "approved", False):
            continue

        ticker = getattr(decision, "ticker", None)
        if not isinstance(ticker, str) or not ticker:
            continue
        payload = signals.get(ticker)
        if not isinstance(payload, dict):
            continue

        signal = _normalized_signal(payload)
        if signal not in {"buy", "sell"}:
            continue

        # PDT: never emit buy orders when restricted; sells (risk-reducing
        # exits) are still allowed.
        if restricted and signal == "buy":
            continue

        # Regime panic: same shape as the PDT gate -- block risk-adding
        # buys, let risk-reducing sells pass.
        if regime_block_buys and signal == "buy":
            continue

        price = _reference_price(payload)
        if price is None:
            continue

        vol_mult = _vol_multiplier(payload, target_daily_vol)
        conviction_mult = _conviction_multiplier(payload)
        # Directional flow conviction: only reports backed by a successful
        # fetch are passed in, so absent/failed data leaves the factor at
        # 1.0 (no-op) and genuine reports can only scale DOWN.
        ticker_reports = (
            analyst_reports.get(ticker) if isinstance(analyst_reports, dict) else None
        )
        flow_active = isinstance(ticker_reports, dict) and bool(ticker_reports)
        flow_mult = (
            _flow_conviction_multiplier(ticker_reports, signal) if flow_active else 1.0
        )
        # Debate-winner conviction (item 10): a bear winner shrinks BUYs,
        # a bull winner shrinks SELLs; undecided/absent debates no-op.
        debate = debates.get(ticker) if isinstance(debates, dict) else None
        debate_mult = _debate_multiplier(debate, signal, debate_oppose_mult)
        # The regime multiplier applies to both sides (a risk_off world
        # also de-scales exits to keep sizing symmetric), while the
        # panic gate only blocks buys and leaves sizing untouched.  The
        # flow multiplier is directional but scale-down-only on both
        # sides.  Kelly (item 7) is the only factor allowed above 1.0,
        # and even then the per-name hard cap bounds the result.
        budget = (
            base_budget * vol_mult * conviction_mult * regime_mult
            * flow_mult * debate_mult * cost_mult * kelly_mult
        )
        budget = min(budget, per_name_cap)
        if budget <= 0.0:
            continue

        sector = sector_map.get(ticker) if isinstance(sector_map, dict) else None
        sector_gated = sector_budget is not None and isinstance(sector, str) and bool(sector.strip())
        cluster = cluster_map.get(ticker) if isinstance(cluster_map, dict) else None
        cluster_gated = (
            corr_budget is not None
            and isinstance(cluster, str)
            and bool(cluster.strip())
        )

        contexts.append({
            "ticker": ticker,
            "payload": payload,
            "signal": signal,
            "price": price,
            "vol": _daily_vol(payload),
            "budget": budget,
            "vol_mult": vol_mult,
            "conviction_mult": conviction_mult,
            "flow_active": flow_active,
            "flow_mult": flow_mult,
            "debate": debate,
            "debate_mult": debate_mult,
            "cost_mult": cost_mult,
            "sector": sector,
            "sector_gated": sector_gated,
            "cluster": cluster,
            "cluster_gated": cluster_gated,
            "parity_cap": None,
        })

    # -- batch-level inverse-vol risk parity (item 8) ------------------------
    # Equal risk contribution across same-day BUY candidates: each name's
    # notional * daily_vol share is equalized over the batch's total BUY
    # budget.  Purely a cap -- applied as min() in pass 2, so it can only
    # shrink and never inflates a low-vol name past its factor budget.
    if invol_enabled:
        buy_contexts = [
            c for c in contexts
            if c["signal"] == "buy" and c["vol"] is not None
        ]
        if len(buy_contexts) >= 2:
            total_budget = sum(c["budget"] for c in buy_contexts)
            inv_vol_sum = sum(1.0 / c["vol"] for c in buy_contexts)
            if total_budget > 0.0 and inv_vol_sum > 0.0:
                for c in buy_contexts:
                    c["parity_cap"] = (
                        total_budget * (1.0 / c["vol"]) / inv_vol_sum
                    )

    # -- pass 2: emit orders --------------------------------------------------
    for c in contexts:
        ticker = c["ticker"]
        payload = c["payload"]
        signal = c["signal"]
        price = c["price"]
        factor_budget = c["budget"]
        budget = factor_budget
        parity_cap = c["parity_cap"]
        invol_active = parity_cap is not None and parity_cap < budget
        if invol_active:
            budget = parity_cap

        if signal == "sell":
            held_qty = _held_quantity(ticker, positions)
            if held_qty <= 0.0:
                continue  # never emit a sell without a held position
            raw_qty = min(budget / price, held_qty)
        else:
            raw_qty = budget / price

        quantity = math.floor(raw_qty * 1e6) / 1e6
        if quantity < 0.000001:
            continue

        notional = quantity * price

        # Dust filter (item 9): orders below the minimum trade notional
        # are skipped -- churning dust costs more than it risks.
        if notional < min_notional_floor:
            continue

        if signal == "buy":
            # Exactly-equal spend is allowed; only strictly-greater breaches.
            # SELL notional never consumes the buy-side cumulative cap.
            if cumulative_notional + notional > equity:
                continue  # skip and keep going with later tickers
            # Gate checks run BEFORE any accumulator mutates: an order
            # skipped by a later gate must not consume an earlier budget.
            new_sector_used: Optional[float] = None
            if c["sector_gated"]:
                used = sector_notional.get(c["sector"], 0.0)
                if used + notional > sector_budget:
                    continue  # skip and keep going with later tickers
                new_sector_used = used + notional
            if c["cluster_gated"]:
                used = corr_notional.get(c["cluster"], 0.0)
                if used + notional > corr_budget:
                    continue  # skip and keep going with later tickers
                corr_notional[c["cluster"]] = used + notional
            if new_sector_used is not None:
                sector_notional[c["sector"]] = new_sector_used
            cumulative_notional += notional

        plans.append(
            OrderPlan(
                ticker=ticker,
                side=signal,
                quantity=quantity,
                limit_price=price,
                rationale=(
                    f"{signal.upper()} {ticker}: sized {quantity:.6f} @ {price:.4f} "
                    f"({_z_score_text(payload)}; budget {budget:.2f} "
                    f"(vol x{c['vol_mult']:.2f}, conviction x{c['conviction_mult']:.2f}"
                    + (f", regime {regime_label} x{regime_mult:.2f}" if regime_label else "")
                    + (f", flow x{c['flow_mult']:.2f}" if c["flow_active"] else "")
                    + (f", debate {_report_field(c['debate'], 'winner')} x{c['debate_mult']:.2f}"
                       if c["debate_mult"] < 1.0 else "")
                    + (f", cost x{c['cost_mult']:.3f}" if c["cost_mult"] < 1.0 else "")
                    + (f", kelly x{kelly_info['mult']:.2f} (n={kelly_info['n']})"
                       if kelly_info and kelly_info["mult"] > 1.0 else "")
                    + (f", invol x{budget / factor_budget:.2f}" if invol_active else "")
                    + f"), buy notional {cumulative_notional:.2f} / equity {equity:.2f})"
                ),
            )
        )

    # -- pass 3: pairs (stat-arb) ---------------------------------------------
    # Shared accumulators flow into the pair planner, so pair BUY legs
    # consume the same equity / sector / cluster budgets the single-leg
    # plan already spent (and add their own per-pair / total rails).
    state = {
        "account": account if isinstance(account, dict) else {},
        "equity": equity,
        "positions": positions,
        "sector_map": sector_map,
        "cluster_map": cluster_map,
        "sector_budget": sector_budget,
        "corr_budget": corr_budget,
        "restricted": restricted,
        "regime_block_buys": regime_block_buys,
        "regime_mult": regime_mult,
        "regime_label": regime_label,
        "cost_mult": cost_mult,
        "target_daily_vol": target_daily_vol,
        "min_notional_floor": min_notional_floor,
        "cumulative_notional": cumulative_notional,
        "sector_notional": sector_notional,
        "corr_notional": corr_notional,
        "pair_notional": {},
        "pairs_total_notional": 0.0,
        "pair_kelly_mult": pair_kelly_mult,
    }
    plans.extend(_build_pair_order_plans(pair_decisions, pair_signals, state))

    return plans


from datetime import datetime as _dt  # noqa: E402  (kept near use for clarity)

# ``isinstance`` check above needs the datetime class; alias imported after
# the allowed stdlib import block to keep the import contract surface flat.
datetime = _dt  # type: ignore[name-defined]  # noqa: F841
