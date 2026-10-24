"""Walk-forward pairs backtest (Kelly stats for the pairs sizer).

Re-discovery + trading loop over one history window, mirroring the
replay backtest's contract (``src/backtest/replay.py``):

  * At each rebalance point ``t`` (every ``rebalance_days`` bars from
    ``formation_window`` to the end) pairs are formed on the trailing
    ``formation_window`` bars and traded on the NEXT ``rebalance_days``
    bars -- no lookahead.
  * Trading replays the long-only conditioned mean reversion exactly
    as live would: buy the cheap leg at ``|z| >= z_entry`` (z from the
    frozen formation stats), exit at ``|z| < z_exit`` or ``|z| >
    z_stop``; any position still open at the rebalance boundary is
    force-closed at that close (re-formation is a hard boundary).
  * Trades are leg-level returns (long entry close -> exit close).

Output stats file (``PAIR_BACKTEST_STATS_PATH``) carries the same
aggregate shape the single-leg sizer reads, so ``pair_kelly_from_stats``
and ``PAIR_KELLY_*`` sizing plug straight in.  Rough by design (no
execution costs, trade-level returns); consumers cap the fraction.

No LLM, no broker; the only I/O is the market-data fetch seam
(injectable for offline tests).  Deterministic given the prices.
"""

from __future__ import annotations

import itertools
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from src.backtest.costs import ExecutionCosts
from src.backtest.replay import summarize_trades
from src.pairs.book import BookEntry, entry_from_candidate
from src.pairs.cointegration import _close_series
from src.pairs.screening import form_pairs

__all__ = [
    "run_pairs_backtest",
    "run_pairs_param_sweep",
    "parse_pairs_param_option",
    "PAIRS_PARAM_SWEEP_ALLOWED",
]


#: Parameters that may be swept in pairs backtest.  Maps param name to
#: a converter callable (str -> typed value).  Only these may be swept.
PAIRS_PARAM_SWEEP_ALLOWED = {
    "z_entry": float,
    "z_exit": float,
    "z_stop": float,
    "max_pvalue": float,
    "min_half_life": float,
    "max_half_life": float,
    "prescreen_corr": float,
    "rebalance_days": int,
    "formation_window": int,
    "max_pairs": int,
    "max_per_sector": int,
    "min_sharpe": float,
}


def parse_pairs_param_option(raw: str) -> tuple[str, list[str]]:
    """Parse a single ``--param NAME=v1,v2`` option for pairs sweep.

    Returns ``(name, [v1, v2, ...])``.  Raises ``ValueError`` if the
    format is invalid or the name is not in :data:`PAIRS_PARAM_SWEEP_ALLOWED`.
    """
    if "=" not in raw:
        raise ValueError(f"invalid param format (expected NAME=v1,v2): {raw!r}")
    name, vals = raw.split("=", 1)
    name = name.strip()
    if name not in PAIRS_PARAM_SWEEP_ALLOWED:
        raise ValueError(
            f"param {name!r} not allowed; allowed: {', '.join(sorted(PAIRS_PARAM_SWEEP_ALLOWED))}"
        )
    values = [v.strip() for v in vals.split(",") if v.strip() != ""]
    if not values:
        raise ValueError(f"param {name!r} requires at least one value")
    return name, values


def _prefetch_frames(
    tickers: list[str],
    period: str,
    fetch: Optional[Callable[[str, str], Any]],
) -> dict[str, Any]:
    """Fetch each ticker once for the given period and return a lookup dict."""
    frames: dict[str, Any] = {}
    if fetch is None:
        from src.data.market_data import fetch_ohlcv
        fetch = fetch_ohlcv
    for ticker in tickers:
        name = str(ticker or "").strip().upper()
        if not name:
            continue
        try:
            frames[name] = fetch(name, period)
        except Exception:  # noqa: BLE001 - degrade per ticker
            pass
    return frames


def _trade_z(entry: BookEntry, price_a: float, price_b: float) -> Optional[float]:
    """Spread z-score from frozen formation stats; ``None`` when junk."""
    import math
    try:
        if price_a <= 0.0 or price_b <= 0.0:
            return None
        spread = (math.log(price_a) - entry.alpha
                  - entry.hedge_ratio * math.log(price_b))
        z = (spread - entry.spread_mean) / entry.spread_std
    except (ZeroDivisionError, TypeError, ValueError):
        return None
    return z if math.isfinite(z) else None


def _simulate_trade_window(
    entries: list[BookEntry],
    prices: dict[str, list[float]],
    start: int,
    end: int,
    z_entry: float,
    z_exit: float,
    z_stop: float,
    costs: Optional[ExecutionCosts] = None,
) -> list[dict]:
    """Replay one trade window; returns leg-level trade records.

    If ``costs`` is provided, entry/exit prices are adjusted for spread
    and slippage. No daily_vol available for pairs; costs applied without
    vol scaling.
    """
    trades: list[dict] = []
    open_positions: dict[str, dict] = {}  # pair_id -> {"leg", "entry"}

    def _close(pair_id: str, exit_price: float, leg: str) -> None:
        position = open_positions.pop(pair_id, None)
        if position is None:
            return
        entry_price = position["entry"]
        if not entry_price or entry_price <= 0.0 or exit_price <= 0.0:
            return
        if costs is not None:
            entry_price = costs.adjust_entry(entry_price, "BUY", None)
            exit_price = costs.adjust_exit(exit_price, "BUY", None)
        trades.append({
            "pair": pair_id,
            "leg": leg,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": exit_price / entry_price - 1.0,
        })

    window_end = min(end, min(len(v) for v in prices.values()))
    for day in range(start, window_end):
        for entry in entries:
            series_a = prices.get(entry.ticker_a)
            series_b = prices.get(entry.ticker_b)
            if not series_a or not series_b or day >= len(series_a) \
                    or day >= len(series_b):
                continue
            price_a = float(series_a[day])
            price_b = float(series_b[day])
            position = open_positions.get(entry.pair_id)
            if position is not None:
                # Exits first: reversion take-profit or stop loss.
                z = _trade_z(entry, price_a, price_b)
                if z is not None and (abs(z) < z_exit or abs(z) > z_stop):
                    leg = position["leg"]
                    leg_prices = series_b if leg == entry.ticker_b else series_a
                    _close(entry.pair_id, float(leg_prices[day]), leg)
                continue
            # No new entries on the window's final bar: a position opened
            # there would be force-closed the same day at the same price
            # (a guaranteed 0% trade -- a loss once costs apply).
            if day >= window_end - 1:
                continue
            z = _trade_z(entry, price_a, price_b)
            if z is None:
                continue
            if z >= z_entry:
                open_positions[entry.pair_id] = {
                    "leg": entry.ticker_b, "entry": price_b,
                }
            elif z <= -z_entry:
                open_positions[entry.pair_id] = {
                    "leg": entry.ticker_a, "entry": price_a,
                }

    # Force-close anything still open at the window boundary.
    for pair_id in list(open_positions):
        entry = next((e for e in entries if e.pair_id == pair_id), None)
        if entry is None:
            open_positions.pop(pair_id, None)
            continue
        leg = open_positions[pair_id]["leg"]
        series = prices.get(leg) or []
        if series:
            _close(pair_id, float(series[min(end, len(series)) - 1]), leg)
        else:
            open_positions.pop(pair_id, None)
    return trades


def run_pairs_backtest(
    tickers: list[str],
    period: str = "2y",
    formation_window: int = 252,
    rebalance_days: int = 21,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
    max_pvalue: float = 0.05,
    max_lag: int = 1,
    prescreen_corr: float = 0.70,
    max_pairs: int = 10,
    max_per_sector: int = 3,
    min_sharpe: Optional[float] = None,
    sectors: Optional[dict[str, Optional[str]]] = None,
    min_trades: int = 30,
    fetch: Optional[Callable[[str, str], Any]] = None,
    costs: Optional[ExecutionCosts] = None,
) -> dict:
    """Run the walk-forward pairs backtest and aggregate Kelly stats.

    ``fetch`` defaults to the project's cached ``fetch_ohlcv`` seam and
    is injectable for offline tests.  Every per-ticker fetch failure
    drops that ticker (fail-open); the function never raises on data
    problems, it just returns fewer/empty trades.

    If ``costs`` is provided, execution costs (spread + slippage) are
    applied to each leg trade. No daily_vol scaling (pairs lack signal
    daily_vol).
    """
    if fetch is None:
        from src.data.market_data import fetch_ohlcv

        def fetch(ticker: str, period: str) -> Any:  # noqa: F811
            return fetch_ohlcv(ticker, period=period)

    try:
        formation_window = max(40, int(formation_window))
    except (TypeError, ValueError):
        formation_window = 252
    try:
        rebalance_days = max(1, int(rebalance_days))
    except (TypeError, ValueError):
        rebalance_days = 21

    # -- one fetch per ticker; failures drop the ticker --------------------
    series_map: dict[str, Any] = {}
    for ticker in tickers or []:
        name = str(ticker or "").strip().upper()
        if not name:
            continue
        try:
            series_map[name] = fetch(name, period)
        except Exception:  # noqa: BLE001 - degrade per ticker
            continue
    closes = {
        ticker: series
        for ticker, series in (
            (name, _close_series(frame)) for name, frame in series_map.items()
        )
        if series is not None
    }
    if len(closes) < 2:
        aggregate = summarize_trades([], min_trades=min_trades, horizon=1)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "pairs",
            "period": period,
            "formation_window": formation_window,
            "rebalance_days": rebalance_days,
            "universe_size": len(closes),
            "dropped_short_history": [],
            "per_pair": {},
            "aggregate": aggregate,
        }

    # A common bar count keeps every rebalance slice aligned (the
    # frames come from the same yfinance period).  Tickers with less
    # history than one formation window + one rebalance can never form
    # a pair; keeping them would collapse n_bars to that minimum and
    # silently disable the whole walk-forward (zero formations), so
    # they are dropped like failed fetches.
    min_bars = formation_window + rebalance_days
    dropped_short_history = sorted(
        ticker for ticker, series in closes.items()
        if len(series) < min_bars
    )
    for ticker in dropped_short_history:
        closes.pop(ticker)
    if len(closes) < 2:
        aggregate = summarize_trades([], min_trades=min_trades, horizon=1)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mode": "pairs",
            "period": period,
            "formation_window": formation_window,
            "rebalance_days": rebalance_days,
            "universe_size": len(closes),
            "dropped_short_history": dropped_short_history,
            "per_pair": {},
            "aggregate": aggregate,
        }
    n_bars = min(len(series) for series in closes.values())
    prices: dict[str, list[float]] = {
        ticker: [float(v) for v in series.iloc[-n_bars:].tolist()]
        for ticker, series in closes.items()
    }

    all_trades: list[dict] = []
    formations = 0
    for start in range(formation_window, n_bars - 1, rebalance_days):
        window_closes = {
            ticker: series.iloc[start - formation_window:start]
            for ticker, series in closes.items()
        }
        ranked = form_pairs(
            window_closes,
            sectors=sectors,
            lookback_days=formation_window,
            min_half_life=min_half_life,
            max_half_life=max_half_life,
            max_pvalue=max_pvalue,
            max_lag=max_lag,
            prescreen_corr=prescreen_corr,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            max_pairs=max_pairs,
            max_per_sector=max_per_sector,
            min_sharpe=min_sharpe,
        )
        if not ranked:
            continue
        formations += 1
        entries = [
            entry_from_candidate(item.candidate, closes=window_closes)
            for item in ranked
        ]
        all_trades.extend(
            _simulate_trade_window(
                entries, prices, start,
                min(start + rebalance_days, n_bars),
                z_entry, z_exit, z_stop,
                costs=costs,
            )
        )

    per_pair = {
        pair_id: summarize_trades(
            [t for t in all_trades if t["pair"] == pair_id],
            min_trades=min_trades,
            horizon=1,
        )
        for pair_id in sorted({t["pair"] for t in all_trades})
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "pairs",
        "period": period,
        "formation_window": formation_window,
        "rebalance_days": rebalance_days,
        "universe_size": len(closes),
        "dropped_short_history": dropped_short_history,
        "formations": formations,
        "z_entry": z_entry,
        "z_exit": z_exit,
        "z_stop": z_stop,
        "spread_bps": costs.spread_bps if costs else None,
        "slippage_bps": costs.slippage_bps if costs else None,
        "cost_scale_by_vol": costs.scale_by_vol if costs else None,
        "per_pair": per_pair,
        "aggregate": summarize_trades(all_trades, min_trades=min_trades, horizon=1),
    }


def run_pairs_param_sweep(
    tickers: list[str],
    period: str = "2y",
    base_params: Optional[dict] = None,
    param_grid: Optional[dict[str, list[str]]] = None,
    min_trades: int = 30,
    sectors: Optional[dict[str, Optional[str]]] = None,
    fetch: Optional[Callable[[str, str], Any]] = None,
    costs: Optional[ExecutionCosts] = None,
) -> dict:
    """Run a sweep over pairs backtest parameter combinations.

    For each parameter combination from ``param_grid`` (merged with
    ``base_params``), runs :func:`run_pairs_backtest` with an injected
    fetch that reuses pre-fetched frames for the given period.

    ``param_grid`` maps parameter name to a list of string values.  The
    values are converted to the appropriate type (float/int) using the
    converter in :data:`PAIRS_PARAM_SWEEP_ALLOWED`.

    Returns a sweep document with an additional ``params`` field on each
    successful result row.

    If ``costs`` is provided, execution costs are applied to all backtest
    runs in the sweep.
    """
    base = dict(base_params) if base_params else {}
    # Reconcile the period: callers may pass it via base_params (the CLI
    # does); prefetching and the injected fetch must agree with the
    # period the backtest will actually run, or every fetch is rejected.
    if base.get("period") is not None:
        period = str(base["period"])
    base.setdefault("period", period)
    base.setdefault("min_trades", min_trades)
    base.setdefault("sectors", sectors)

    # Prefetch all frames once for this period
    frames = _prefetch_frames(tickers, period, fetch)

    def _injected_fetch(name: str, p: str) -> Any:
        if p != period:
            raise ValueError(f"sweep only supports single period; got {p!r} vs {period!r}")
        frame = frames.get(name)
        if frame is None:
            raise RuntimeError(f"no prefetched frame for {name}:{period}")
        return frame

    if not param_grid:
        # No param grid -> single run with base params
        stats = run_pairs_backtest(tickers, fetch=_injected_fetch, costs=costs, **base)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "tickers": list(tickers),
            "results": [{
                "params": base,
                "period": stats.get("period"),
                "formation_window": stats.get("formation_window"),
                "rebalance_days": stats.get("rebalance_days"),
                "universe_size": stats.get("universe_size"),
                "formations": stats.get("formations"),
                "per_pair": stats.get("per_pair"),
                "aggregate": stats.get("aggregate"),
            }],
        }

    # Build param grid combinations (values are strings, convert per allowed types)
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))

    results: list[dict] = []
    for combo_vals in combos:
        param_dict = dict(zip(param_names, combo_vals))
        # Convert string values to appropriate types
        converted: dict = {}
        for key, val in param_dict.items():
            converter = PAIRS_PARAM_SWEEP_ALLOWED[key]
            try:
                converted[key] = converter(val)
            except (TypeError, ValueError) as exc:
                results.append({
                    "params": param_dict,
                    "error": f"invalid value for {key}: {val!r} ({exc})",
                })
                break
        else:
            # Merge with base params
            kwargs = {**base, **converted}
            try:
                stats = run_pairs_backtest(tickers, fetch=_injected_fetch, costs=costs, **kwargs)
                results.append({
                    "params": converted,
                    "period": stats.get("period"),
                    "formation_window": stats.get("formation_window"),
                    "rebalance_days": stats.get("rebalance_days"),
                    "universe_size": stats.get("universe_size"),
                    "formations": stats.get("formations"),
                    "per_pair": stats.get("per_pair"),
                    "aggregate": stats.get("aggregate"),
                })
            except Exception as exc:  # noqa: BLE001 - fail-open per combo
                results.append({
                    "params": converted,
                    "error": str(exc),
                })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": list(tickers),
        "results": results,
    }