"""Deterministic signal-replay backtest for Kelly sizing stats (item 7).

Replays the deterministic quant signal generator
(:func:`src.quant.signals.generate_signal`) over historical closes and
scores every BUY/SELL signal by its forward return over a fixed horizon.
No LLM calls, no broker, no network beyond the market-data fetch seam --
the replay is fully deterministic given the price data, so two runs over
the same input produce identical stats.

Outputs feed Kelly-fraction sizing: the aggregate trade population gives
win rate / payoff, from which the Kelly fraction is computed as

    f* = W - (1 - W) / R        (R = avg_win / avg_loss)

Rough by design (quant signals only, no execution costs, single horizon)
-- consumers are expected to apply their own fraction caps (see
``KELLY_FRACTION_CAP`` in the sizing layer) and treat the number as a
de-risking signal, not an optimization oracle.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.costs import ExecutionCosts
from src.quant.signals import generate_signal

__all__ = [
    "replay_signals",
    "replay_overnight",
    "replay_random",
    "summarize_trades",
    "run_backtest",
]

#: generate_signal needs ``window + step`` observations before it can
#: emit non-HOLD signals (see src/quant/signals.py).
_MIN_WINDOW = 25


def _closes_from_frame(frame: Any) -> Optional[pd.Series]:
    """Extract a close-price Series from an OHLCV frame (or passthrough)."""
    if frame is None:
        return None
    if isinstance(frame, pd.Series):
        series = frame
    elif isinstance(frame, pd.DataFrame):
        if "Close" in frame.columns:
            series = frame["Close"]
        elif len(frame.columns):
            series = frame.iloc[:, 0]
        else:
            return None
        if isinstance(series, pd.DataFrame):
            if series.empty:
                return None
            series = series.iloc[:, 0]
    else:
        return None
    try:
        series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    except Exception:  # noqa: BLE001 - degrade per ticker
        return None
    if series.empty:
        return None
    return series


def replay_signals(
    closes: Any,
    horizon: int = 5,
    costs: Optional[ExecutionCosts] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> list[dict]:
    """Replay the quant signal over ``closes`` and score every trade.

    At each bar ``t`` (from ``_MIN_WINDOW`` to ``len - horizon``) the
    signal generator sees prices up to and including bar ``t-1``; the
    trade's outcome is the return from the last observed close over the
    next ``horizon`` bars.  SELL outcomes are direction-adjusted (a win
    means the price fell).

    If ``costs`` is provided, entry/exit prices are adjusted for spread
    and slippage using the signal's ``daily_vol`` for volatility scaling.

    ``start_index``/``end_index`` bound the collected ENTRY bars to
    ``[start_index, end_index)`` (walk-forward OOS folds collect only
    their test-window trades while the signal generator still sees the
    full history up to each bar; outcomes may extend past ``end_index``
    when the caller slices the frame with look-ahead room).  Defaults
    collect every trade (unchanged behavior).

    Returns a list of ``{"signal", "entry_price", "exit_price",
    "return"}`` dicts (``return`` is direction-adjusted and positive when
    the signal won).  Deterministic: same input -> same output.
    """
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        return []
    if horizon < 1:
        return []
    series = _closes_from_frame(closes)
    if series is None or len(series) < _MIN_WINDOW + horizon:
        return []
    try:
        start_index = max(0, int(start_index))
    except (TypeError, ValueError):
        start_index = 0

    # The signal generator receives the raw input (Series or full OHLCV
    # frame) so the volume-confirmation gates see the Volume column when
    # one exists; outcome prices always come from the close series.
    frame = closes if isinstance(closes, pd.DataFrame) else None
    values = series.to_numpy()
    t_hi = len(values) - horizon + 1
    if end_index is not None:
        try:
            t_hi = min(t_hi, max(0, int(end_index)))
        except (TypeError, ValueError):
            pass
    trades: list[dict] = []
    for t in range(max(_MIN_WINDOW, start_index), t_hi):
        look = frame.iloc[:t] if frame is not None else series.iloc[:t]
        sig = generate_signal(look)
        signal = str(sig.get("signal", "")).upper()
        if signal not in {"BUY", "SELL"}:
            continue
        entry = float(values[t - 1])
        exit_price = float(values[t - 1 + horizon])
        if not math.isfinite(entry) or entry <= 0.0:
            continue
        daily_vol = float(sig.get("daily_vol", 0.0))
        if costs is not None:
            entry = costs.adjust_entry(entry, signal, daily_vol)
            exit_price = costs.adjust_exit(exit_price, signal, daily_vol)
        fwd = exit_price / entry - 1.0
        outcome = fwd if signal == "BUY" else -fwd
        trades.append({
            "signal": signal,
            "entry_price": entry,
            "exit_price": exit_price,
            "return": outcome,
        })
    return trades


def _column_series(frame: Any, column: str) -> Optional[pd.Series]:
    """Extract a numeric float Series for ``column`` from an OHLCV frame."""
    if not isinstance(frame, pd.DataFrame) or column not in frame.columns:
        return None
    series = frame[column]
    if isinstance(series, pd.DataFrame):
        if series.empty:
            return None
        series = series.iloc[:, 0]
    try:
        series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    except Exception:  # noqa: BLE001 - degrade per ticker
        return None
    if series.empty:
        return None
    return series


def replay_overnight(
    frame: Any,
    costs: Optional[ExecutionCosts] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> list[dict]:
    """Replay the quant signal as an overnight (close -> next open) strategy.

    At each bar ``t`` the signal generator sees closes up to and including
    bar ``t``; a BUY/SELL signal enters at ``Close[t]`` (the live daily
    close) and exits at ``Open[t+1]`` the next morning.  Outcomes are
    direction-adjusted: a SELL wins when the next open gaps down.  Needs
    both ``Close`` and ``Open`` columns; anything else degrades to ``[]``.

    If ``costs`` is provided, entry/exit prices are adjusted for spread
    and slippage using the signal's ``daily_vol`` for volatility scaling.

    ``start_index``/``end_index`` bound the collected ENTRY bars to
    ``[start_index, end_index)`` (walk-forward OOS folds collect only
    their test-window trades while the signal generator still sees the
    full history up to each bar).  Defaults collect every trade
    (unchanged behavior).
    """
    closes = _column_series(frame, "Close")
    opens = _column_series(frame, "Open")
    if closes is None or opens is None:
        return []
    aligned = pd.concat([closes.rename("close"), opens.rename("open")], axis=1)
    # The volume column (when present) keeps its capital "Volume" name so
    # generate_signal's _volume_series sees it and the volume-confirmation
    # gate behaves exactly as it does live (a lowercase column would hide
    # the feature and pin volume_z to 0.0 -- the pre-2026-09-14 divergence).
    volume = _column_series(frame, "Volume")
    if volume is not None:
        aligned = pd.concat([aligned, volume.rename("Volume")], axis=1)
    aligned = aligned.dropna(subset=["close", "open"])
    if len(aligned) < _MIN_WINDOW + 1:
        return []
    try:
        start_index = max(0, int(start_index))
    except (TypeError, ValueError):
        start_index = 0

    close_values = aligned["close"].to_numpy()
    open_values = aligned["open"].to_numpy()
    t_hi = len(close_values) - 1
    if end_index is not None:
        try:
            t_hi = min(t_hi, max(0, int(end_index)))
        except (TypeError, ValueError):
            pass
    trades: list[dict] = []
    for t in range(max(_MIN_WINDOW - 1, start_index), t_hi):
        sig = generate_signal(aligned.iloc[: t + 1])
        signal = str(sig.get("signal", "")).upper()
        if signal not in {"BUY", "SELL"}:
            continue
        entry = float(close_values[t])
        exit_price = float(open_values[t + 1])
        if (
            not math.isfinite(entry)
            or entry <= 0.0
            or not math.isfinite(exit_price)
            or exit_price <= 0.0
        ):
            continue
        daily_vol = float(sig.get("daily_vol", 0.0))
        if costs is not None:
            entry = costs.adjust_entry(entry, signal, daily_vol)
            exit_price = costs.adjust_exit(exit_price, signal, daily_vol)
        fwd = exit_price / entry - 1.0
        outcome = fwd if signal == "BUY" else -fwd
        trades.append({
            "signal": signal,
            "entry_price": entry,
            "exit_price": exit_price,
            "return": outcome,
        })
    return trades


def replay_random(
    closes: Any,
    horizon: int = 5,
    signal_rate: float = 0.1,
    seed: int = 42,
    costs: Optional[ExecutionCosts] = None,
    mode: str = "close",
) -> list[dict]:
    """Generate random BUY/SELL signals matching a target signal rate.

    This provides a baseline to validate that the quant strategy has genuine
    edge. Random signals should yield ~0 Kelly, ~0.5 win rate, negative
    Sharpe after costs.

    Args:
        closes: Close price Series (mode="close") or OHLCV DataFrame with
            ``Close`` and ``Open`` columns (mode="overnight").
        horizon: Forward horizon for scoring trades (mode="close" only;
            overnight always scores close -> next open).
        signal_rate: Fraction of bars that generate a signal (default 0.1).
        seed: RNG seed for reproducibility.
        costs: Optional ExecutionCosts to apply.
        mode: ``"close"`` scores close->close over ``horizon`` (same
            contract as :func:`replay_signals`); ``"overnight"`` enters at
            ``Close[t]`` and exits at ``Open[t+1]`` (same contract as
            :func:`replay_overnight`) so the null distribution matches
            what the overnight strategy validates.

    Returns:
        List of trade dicts with same schema as replay_signals.
    """
    try:
        horizon = int(horizon)
    except (TypeError, ValueError):
        return []
    if horizon < 1:
        return []

    if mode == "overnight":
        closes_s = _column_series(closes, "Close")
        opens_s = _column_series(closes, "Open")
        if closes_s is None or opens_s is None:
            return []
        aligned = pd.concat([
            closes_s.rename("close"), opens_s.rename("open"),
        ], axis=1).dropna(subset=["close", "open"])
        if len(aligned) < _MIN_WINDOW + 1:
            return []
        values = aligned["close"].to_numpy()
        open_values = aligned["open"].to_numpy()
        rng = np.random.default_rng(seed)
        trades: list[dict] = []
        for t in range(_MIN_WINDOW - 1, len(values) - 1):
            if rng.random() >= signal_rate:
                continue
            signal = "BUY" if rng.random() < 0.5 else "SELL"
            entry = float(values[t])
            exit_price = float(open_values[t + 1])
            if (
                not math.isfinite(entry)
                or entry <= 0.0
                or not math.isfinite(exit_price)
                or exit_price <= 0.0
            ):
                continue
            if costs is not None:
                entry = costs.adjust_entry(entry, signal, None)
                exit_price = costs.adjust_exit(exit_price, signal, None)
            fwd = exit_price / entry - 1.0
            outcome = fwd if signal == "BUY" else -fwd
            trades.append({
                "signal": signal,
                "entry_price": entry,
                "exit_price": exit_price,
                "return": outcome,
            })
        return trades
    if mode != "close":
        raise ValueError(f"unknown random-baseline mode: {mode!r}")

    series = _closes_from_frame(closes)
    if series is None or len(series) < _MIN_WINDOW + horizon:
        return []

    rng = np.random.default_rng(seed)
    values = series.to_numpy()
    n = len(values)
    trades = []

    for t in range(_MIN_WINDOW, n - horizon + 1):
        # Randomly decide whether to emit a signal at this bar
        if rng.random() >= signal_rate:
            continue
        # Randomly choose BUY or SELL with equal probability
        signal = "BUY" if rng.random() < 0.5 else "SELL"
        entry = float(values[t - 1])
        exit_price = float(values[t - 1 + horizon])
        if not math.isfinite(entry) or entry <= 0.0:
            continue
        # No daily_vol available for random signals; use None for costs
        if costs is not None:
            entry = costs.adjust_entry(entry, signal, None)
            exit_price = costs.adjust_exit(exit_price, signal, None)
        fwd = exit_price / entry - 1.0
        outcome = fwd if signal == "BUY" else -fwd
        trades.append({
            "signal": signal,
            "entry_price": entry,
            "exit_price": exit_price,
            "return": outcome,
        })
    return trades


def _annualize_factor(horizon: int) -> float:
    """Annualization factor for trade-horizon returns (sqrt(252/horizon))."""
    if horizon < 1:
        return 1.0
    return math.sqrt(252.0 / horizon)


def _equity_curve(returns: list[float]) -> list[float]:
    """Cumulative equity curve starting at 1.0 from trade returns."""
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _max_drawdown(equity: list[float]) -> float:
    """Maximum peak-to-trough drawdown from equity curve (fraction 0..1)."""
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def summarize_trades(
    trades: list[dict],
    min_trades: int = 30,
    horizon: int = 5,
) -> dict:
    """Aggregate replayed trades into Kelly-ready summary stats.

    ``kelly_fraction`` is ``None`` when the sample is too small, the
    payoff ratio is unusable, or the edge is non-positive -- consumers
    should treat ``None``/non-positive exactly like absent stats
    (sizing factor 1.0).

    Also computes risk-adjusted metrics (Sharpe, Sortino, Calmar, Omega)
    annualized from the trade-horizon return series.
    """
    n = len(trades)
    summary: dict = {"n_trades": n}
    if n == 0:
        summary["kelly_fraction"] = None
        summary["usable"] = False
        summary.update({
            "sharpe": None,
            "sortino": None,
            "calmar": None,
            "omega": None,
        })
        return summary
    returns = [t["return"] for t in trades if isinstance(t.get("return"), (int, float))]
    if not returns:
        summary["kelly_fraction"] = None
        summary["usable"] = False
        summary.update({
            "sharpe": None,
            "sortino": None,
            "calmar": None,
            "omega": None,
        })
        return summary
    n = len(returns)
    wins = [r for r in returns if r > 0.0]
    losses = [abs(r) for r in returns if r <= 0.0]
    win_rate = len(wins) / n
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0.0 else None
    kelly: Optional[float] = None
    if payoff is not None and payoff > 0.0:
        kelly = win_rate - (1.0 - win_rate) / payoff
        if not math.isfinite(kelly) or kelly <= 0.0:
            kelly = None
    usable = n >= min_trades and kelly is not None

    # --- Risk-adjusted metrics (annualized from trade-horizon returns) ---
    ann = _annualize_factor(horizon)
    mean_r = sum(returns) / n
    # Population std for Sharpe; downside std for Sortino
    var = sum((r - mean_r) ** 2 for r in returns) / n
    std_r = math.sqrt(var) if var > 0 else 0.0
    downside = [r for r in returns if r < 0.0]
    if downside:
        var_down = sum((r - mean_r) ** 2 for r in downside) / len(downside)
        std_down = math.sqrt(var_down)
    else:
        std_down = 0.0

    sharpe = (mean_r / std_r * ann) if std_r > 0 else None
    sortino = (mean_r / std_down * ann) if std_down > 0 else None

    # Calmar: annualized return / max drawdown (equity curve)
    equity = _equity_curve(returns)
    max_dd = _max_drawdown(equity)
    total_return = equity[-1] - 1.0
    ann_return = total_return * ann  # crude annualization of total return
    calmar = (ann_return / max_dd) if max_dd > 0 else None

    # Omega: sum of positive returns / |sum of negative returns| (threshold 0)
    pos_sum = sum(r for r in returns if r > 0)
    neg_sum = abs(sum(r for r in returns if r < 0))
    omega = (pos_sum / neg_sum) if neg_sum > 0 else None

    summary.update({
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "kelly_fraction": kelly if usable else None,
        "usable": usable,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "omega": omega,
    })
    return summary


def run_backtest(
    tickers: list[str],
    period: str = "2y",
    horizon: int = 5,
    min_trades: int = 30,
    fetch: Optional[Callable[[str, str], Any]] = None,
    mode: str = "close",
    costs: Optional[ExecutionCosts] = None,
) -> dict:
    """Run the replay backtest across ``tickers`` and aggregate stats.

    ``mode="close"`` (default) replays the classic close-to-close
    strategy over ``horizon`` days; ``mode="overnight"`` scores the
    halo/overnight-drift strategy (enter at the signal-day close, exit
    the next open) and ignores ``horizon``.  ``fetch`` defaults to the
    project's cached ``fetch_ohlcv`` seam and is injectable for offline
    tests.  Every per-ticker failure degrades to ``{"n_trades": 0}`` for
    that ticker (fail-open); the function never raises on data problems.

    If ``costs`` is provided, execution costs (spread + slippage) are
    applied to each trade using the signal's daily_vol for scaling.
    """
    if mode not in ("close", "overnight"):
        raise ValueError(f"unknown backtest mode: {mode!r}")

    if fetch is None:
        from src.data.market_data import fetch_ohlcv

        def fetch(ticker: str, period: str) -> Any:  # noqa: F811
            return fetch_ohlcv(ticker, period=period)

    per_ticker: dict[str, dict] = {}
    all_trades: list[dict] = []
    horizon_eff = 1 if mode == "overnight" else horizon
    for ticker in tickers:
        name = str(ticker).strip().upper()
        if not name:
            continue
        try:
            data = fetch(name, period)
            if mode == "overnight":
                trades = replay_overnight(data, costs=costs)
            else:
                trades = replay_signals(data, horizon=horizon, costs=costs)
        except Exception:  # noqa: BLE001 - degrade per ticker
            trades = []
        per_ticker[name] = summarize_trades(trades, min_trades=min_trades, horizon=horizon_eff)
        all_trades.extend(trades)

    summary: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "period": period,
        "horizon_days": horizon_eff,
        "min_trades": min_trades,
        "per_ticker": per_ticker,
        "aggregate": summarize_trades(all_trades, min_trades=min_trades, horizon=horizon_eff),
    }
    if mode == "overnight":
        # The overnight strategy trades the BUY side only (entries at the
        # close, mechanical exits next open); the SELL side is reported for
        # completeness.  Phase 3 sizing consumes per_side BUY stats.
        summary["per_side"] = {
            side: summarize_trades(
                [t for t in all_trades if t.get("signal") == side],
                min_trades=min_trades,
                horizon=horizon_eff,
            )
            for side in ("BUY", "SELL")
        }
    return summary


def primary_payload(stats: dict) -> Optional[dict]:
    """Primary Kelly-stats payload from an overnight replay (``None`` n/a).

    The overnight BUY side is the production edge (positive Kelly in the
    2y replay while the pooled aggregate is not usable), so ``--primary``
    promotes ``per_side.BUY`` into the aggregate-shaped payload the
    intraday sizer reads from BACKTEST_STATS_PATH.  Returns ``None``
    when ``stats`` is not an overnight replay with a BUY side.
    """
    if not isinstance(stats, dict) or stats.get("mode") != "overnight":
        return None
    per_side = stats.get("per_side")
    buy = per_side.get("BUY") if isinstance(per_side, dict) else None
    if not isinstance(buy, dict):
        return None
    return {
        "generated_at": stats.get("generated_at"),
        "mode": "overnight_buy",
        "period": stats.get("period"),
        "horizon_days": 1,
        "min_trades": stats.get("min_trades"),
        "per_ticker": stats.get("per_ticker", {}),
        "aggregate": buy,
    }
