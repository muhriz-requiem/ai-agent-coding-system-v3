"""Portfolio-level walk-forward simulation of the overnight BUY edge.

The replay backtest (``src/backtest/replay.py``) scores every signal
independently: there is no capital, no selection among concurrent
candidates, no costs and no compounding.  This module answers the
question the replay cannot -- *does the strategy beat the index?* -- by
simulating the production overnight-BUY strategy as an actual portfolio:

* at each bar ``t`` the quant signal generator sees data up to and
  including bar ``t`` (identical no-lookahead contract to
  :func:`src.backtest.replay.replay_overnight`);
* every BUY candidate enters at ``Close[t]`` and exits at ``Open[t+1]``
  (mechanical overnight exit, exactly the live overnight cycle);
* at most ``top_n`` candidates are taken per day, ranked by signal
  confidence (descending) with the ticker symbol as the deterministic
  tie-break;
* each position is sized ``stake_pct`` of *current* equity (total
  deployed fraction capped at 100%); a round-trip cost in basis points
  is deducted from every trade's return;
* idle cash earns nothing (conservative), so the daily equity curve
  steps only on overnight returns -- exposure and cash drag are honest.

The buy rule is evaluated by a vectorized mirror of the
``generate_signal`` rule path as invoked by the overnight replay.  The
overnight replay feeds ``generate_signal`` a lowercase-columned aligned
frame (``close``/``open``), so the volume features are always 0.0 there
and the volume gate passes iff ``0.0 >= VOLUME_Z_MIN_BUY`` -- the
mirror reproduces exactly that behavior, including env-tuned thresholds
(``Z_BUY_ENTRY``, ``MOMENTUM_FLOOR_PCT``, ``VOLUME_Z_MIN_BUY``).  The
equivalence is property-tested against a naive ``generate_signal``
replay in ``tests/test_backtest_portfolio.py``.

The benchmark is a buy-and-hold on the same daily grid built from the
benchmark ticker's adjusted closes (yfinance ``auto_adjust`` defaults
on, so close-to-close is a fair total-return comparison).  Everything
is deterministic given the price data; per-ticker failures degrade to
"no entries" (fail-open), never an exception.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.costs import ExecutionCosts
from src.quant.signals import (
    DEFAULT_MOMENTUM_FLOOR_PCT,
    DEFAULT_Z_BUY_ENTRY,
    _threshold,
)

# Alignment contract shared with the replay backtest: same minimum
# window and the same close/open column extraction helpers, so the
# portfolio sim and the Kelly replay see identical bars.
from src.backtest.replay import _column_series, _MIN_WINDOW

__all__ = [
    "run_portfolio_backtest",
    "simulate_portfolio_overnight",
    "overnight_buy_entries",
    "benchmark_metrics",
]

_TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Price-frame alignment (mirrors replay_overnight's contract)
# ---------------------------------------------------------------------------


def _norm_dates(index: Any) -> pd.DatetimeIndex:
    """Normalize a frame index to tz-naive, midnight-normalized dates."""
    dates = pd.to_datetime(list(index), errors="coerce")
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _dedupe(series: pd.Series) -> pd.Series:
    """Re-index a Series on normalized dates, dropping NaT/dupe rows."""
    dates = _norm_dates(series.index)
    out = pd.Series(pd.to_numeric(series.to_numpy(), errors="coerce").astype(float),
                    index=dates)
    out = out[out.index.notna()]
    return out[~out.index.duplicated(keep="last")].sort_index()


def _aligned_close_open(
    frame: Any,
) -> Optional[tuple[pd.Series, pd.Series, Optional[pd.Series]]]:
    """Extract the close/open (and volume) bars the overnight replay sees.

    Same extraction helpers and the same dropna contract as
    :func:`src.backtest.replay.replay_overnight`; the result is indexed
    on normalized, de-duplicated dates so cross-ticker grids align.
    Volume (when the frame carries a ``Volume`` column) is reindexed to
    the common grid -- rows without volume are NaN, exactly like the
    concat alignment inside the replay.  Returns ``None`` when either
    price column is missing or too short.
    """
    close = _column_series(frame, "Close")
    open_ = _column_series(frame, "Open")
    if close is None or open_ is None:
        return None
    volume = _column_series(frame, "Volume")
    aligned = pd.concat([close.rename("close"), open_.rename("open")], axis=1)
    aligned = aligned.dropna(subset=["close", "open"])
    if aligned.empty:
        return None
    close_s = _dedupe(aligned["close"])
    open_s = _dedupe(aligned["open"])
    common = close_s.index.intersection(open_s.index)
    if len(common) < _MIN_WINDOW + 1:
        return None
    volume_s = None
    if volume is not None:
        volume_s = _dedupe(volume).reindex(common)
    return close_s.loc[common], open_s.loc[common], volume_s


# ---------------------------------------------------------------------------
# Vectorized BUY rule (exact mirror of the overnight replay's rule path)
# ---------------------------------------------------------------------------


def _overnight_buy_table(
    close: pd.Series, volume: Optional[pd.Series] = None
) -> pd.DataFrame:
    """Per-bar BUY decision + confidence, vectorized.

    Reproduces the ``generate_signal`` rule exactly as the overnight
    replay invokes it: no regime snapshot (fear/breadth gates pass) and
    the volume-confirmation gate fed by the frame's ``Volume`` column
    (real volume z-scores when present, 0.0 otherwise -- matching
    ``compute_volume_z``'s fail-open guards).  Guards mirror the naive
    feature functions: non-positive/NaN fast-MA -> momentum 0.0, NaN z
    never buys, BUY confidence = min(0.95, |z| * 0.4).
    """
    index = close.index
    empty = pd.DataFrame({"buy": pd.Series(False, index=index),
                          "confidence": pd.Series(0.0, index=index)})
    n = len(close)
    if n < _MIN_WINDOW:
        return empty

    window, step = 20, 5
    roll_mean = close.rolling(window, min_periods=window).mean()
    roll_std = close.rolling(window, min_periods=window).std(ddof=1)
    ma = close.rolling(10, min_periods=10).mean()
    # compute_momentum_slope's reference bar is ``ma_fast.iloc[-step]``:
    # iloc[-1] is the current bar, so the lookback is step-1 rows.
    ma_prev = ma.shift(step - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (close - roll_mean) / roll_std
        mom_raw = ((ma - ma_prev) / step / ma * 100.0).to_numpy()
    # compute_momentum_slope guards: NaN slope or non-positive MA -> 0.0
    mom = pd.Series(
        np.where(((ma > 0) & ma_prev.notna()).to_numpy(), mom_raw, 0.0),
        index=index,
    )

    # Volume gate: replicate compute_volume_z (rolling z with fail-open
    # guards -> 0.0) on the aligned volume column; without a Volume
    # column the feature is pinned to 0.0 exactly like the replay.
    if volume is not None:
        vol_mean = volume.rolling(window, min_periods=window).mean()
        vol_std = volume.rolling(window, min_periods=window).std(ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            volume_z = (volume - vol_mean) / vol_std
        ok = (vol_mean.notna() & vol_std.notna() & (vol_std != 0)
              & volume.notna())
        volume_z = volume_z.where(ok, 0.0).fillna(0.0)
        volume_gate_ok = (volume_z >= _threshold(
            "VOLUME_Z_MIN_BUY", float("-inf")))
    else:
        volume_gate_ok = pd.Series(
            0.0 >= _threshold("VOLUME_Z_MIN_BUY", float("-inf")),
            index=index,
        )

    z_buy = _threshold("Z_BUY_ENTRY", DEFAULT_Z_BUY_ENTRY)
    momentum_floor = _threshold("MOMENTUM_FLOOR_PCT", DEFAULT_MOMENTUM_FLOOR_PCT)

    mask = ((z < z_buy) & (mom > momentum_floor)
            & volume_gate_ok.reindex(index).fillna(False)).to_numpy()
    conf = np.where(mask, np.minimum(0.95, np.abs(z.to_numpy()) * 0.4), 0.0)
    return pd.DataFrame({
        "buy": pd.Series(mask.astype(bool), index=index),
        "confidence": pd.Series(conf, index=index),
    })


def overnight_buy_entries(frame: Any) -> dict[Any, dict]:
    """Collect the overnight BUY candidates a frame produces.

    Returns ``{normalized_date: {"entry": Close[t], "exit": Open[t+1],
    "confidence": float}}`` for every bar whose signal is a BUY.  The
    signal at bar ``t`` is computed from bars up to and including ``t``
    (no lookahead); the exit is the next bar's open.  Anything other
    than a usable OHLC frame degrades to ``{}``.
    """
    aligned = _aligned_close_open(frame)
    if aligned is None:
        return {}
    close, open_, volume = aligned
    table = _overnight_buy_table(close, volume)
    buys = table["buy"].to_numpy()
    confs = table["confidence"].to_numpy()
    close_v = close.to_numpy()
    open_v = open_.to_numpy()
    dates = list(close.index)
    n = len(close)
    entries: dict[Any, dict] = {}
    for t in range(_MIN_WINDOW - 1, n - 1):
        if not buys[t]:
            continue
        entry = float(close_v[t])
        exit_ = float(open_v[t + 1])
        if not (math.isfinite(entry) and entry > 0.0
                and math.isfinite(exit_) and exit_ > 0.0):
            continue
        entries[dates[t]] = {
            "entry": entry,
            "exit": exit_,
            "confidence": float(confs[t]),
        }
    return entries


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------


def _returns_metrics(returns: list[float]) -> dict:
    """Total return / CAGR / Sharpe / Sortino / max-DD from daily returns."""
    out: dict = {
        "n_days": len(returns),
        "total_return": None,
        "cagr": None,
        "sharpe": None,
        "sortino": None,
        "max_drawdown": None,
    }
    if not returns:
        return out
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1.0 + r)
        if equity > peak:
            peak = equity
        if peak > 0.0:
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
    n = len(returns)
    mean_r = sum(returns) / n
    var = sum((r - mean_r) ** 2 for r in returns) / n
    std_r = math.sqrt(var) if var > 0 else 0.0
    downside = [r for r in returns if r < 0.0]
    if downside:
        dvar = sum((r - mean_r) ** 2 for r in downside) / len(downside)
        std_down = math.sqrt(dvar)
    else:
        std_down = 0.0
    out["total_return"] = equity - 1.0
    years = n / _TRADING_DAYS
    if years > 0 and equity > 0:
        out["cagr"] = equity ** (1.0 / years) - 1.0
    out["sharpe"] = (mean_r / std_r * math.sqrt(_TRADING_DAYS)) if std_r > 0 else None
    out["sortino"] = (mean_r / std_down * math.sqrt(_TRADING_DAYS)) if std_down > 0 else None
    out["max_drawdown"] = max_dd
    return out


def simulate_portfolio_overnight(
    entries: dict[str, dict],
    top_n: int = 5,
    stake_pct: float = 0.20,
    cost_bps: float = 10.0,
    costs: Optional[ExecutionCosts] = None,
) -> dict:
    """Walk-forward overnight-BUY portfolio simulation.

    ``entries`` maps ticker -> {date -> candidate info} as produced by
    :func:`overnight_buy_entries`.  Each day the top ``top_n``
    candidates by confidence (ticker symbol breaks ties) are taken with
    ``stake_pct`` of current equity each (deployed fraction capped at
    100%); every trade pays ``cost_bps`` round-trip (or uses ``costs``
    if provided for spread+slippage modeling).  Deterministic.
    """
    try:
        top = max(1, int(top_n))
    except (TypeError, ValueError):
        top = 5
    try:
        stake = float(stake_pct)
    except (TypeError, ValueError):
        stake = 0.20
    stake = min(max(stake, 0.0), 1.0)

    # Use costs model if provided, otherwise fall back to simple cost_bps
    if costs is not None:
        use_cost_model = True
    else:
        use_cost_model = False
        try:
            cost = float(cost_bps)
        except (TypeError, ValueError):
            cost = 10.0
        cost = max(cost, 0.0) / 1e4

    by_date: dict[Any, list[tuple[str, float, float, float]]] = {}
    for ticker, per_date in (entries or {}).items():
        if not isinstance(per_date, dict):
            continue
        name = str(ticker).strip().upper()
        if not name:
            continue
        for date, info in per_date.items():
            if not isinstance(info, dict):
                continue
            try:
                entry = float(info.get("entry"))
                exit_ = float(info.get("exit"))
                conf = float(info.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(entry) and entry > 0.0
                    and math.isfinite(exit_) and exit_ > 0.0
                    and math.isfinite(conf)):
                continue
            by_date.setdefault(date, []).append((name, entry, exit_, conf))

    equity = 1.0
    dates_out: list[Any] = []
    daily_returns: list[float] = []
    n_positions: list[int] = []
    deployed_fracs: list[float] = []
    n_trades = 0
    selections: dict[str, int] = {}
    for date in sorted(by_date):
        candidates = sorted(by_date[date], key=lambda c: (-c[3], c[0]))[:top]
        day_ret = 0.0
        deployed = 0.0
        npos = 0
        for name, entry, exit_, _conf in candidates:
            frac = min(stake, 1.0 - deployed)
            if frac <= 0.0:
                break
            if use_cost_model:
                # No daily_vol available in portfolio sim entries; use None
                entry_adj = costs.adjust_entry(entry, "BUY", None)
                exit_adj = costs.adjust_exit(exit_, "BUY", None)
                trade_ret = exit_adj / entry_adj - 1.0
            else:
                trade_ret = exit_ / entry - 1.0 - cost
            day_ret += frac * trade_ret
            deployed += frac
            npos += 1
            n_trades += 1
            selections[name] = selections.get(name, 0) + 1
        equity *= (1.0 + day_ret)
        dates_out.append(date)
        daily_returns.append(day_ret)
        n_positions.append(npos)
        deployed_fracs.append(deployed)

    metrics = _returns_metrics(daily_returns)
    metrics["exposure"] = (
        sum(1 for p in n_positions if p > 0) / len(n_positions)
        if n_positions else None
    )
    metrics["avg_deployed"] = (
        sum(deployed_fracs) / len(deployed_fracs) if deployed_fracs else None
    )
    metrics["n_trades"] = n_trades
    metrics["selections"] = selections
    metrics["daily_returns"] = {
        d.isoformat()[:10]: r for d, r in zip(dates_out, daily_returns)
    }
    return {"metrics": metrics, "final_equity": equity}


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark_metrics(frame: Any, grid_dates: list) -> dict:
    """Buy-and-hold metrics on the portfolio's daily grid (fail-open)."""
    out: dict = {
        "n_days": 0,
        "total_return": None,
        "cagr": None,
        "sharpe": None,
        "sortino": None,
        "max_drawdown": None,
        "daily_returns": {},
    }
    close = _column_series(frame, "Close")
    if close is None or not len(close):
        return out
    series = _dedupe(close)
    values: dict[Any, float] = {}
    for date, price in series.items():
        if math.isfinite(price) and price > 0.0:
            values[date] = price
    path = [values[d] for d in grid_dates if d in values]
    if len(path) < 2:
        return out
    rets = [path[i] / path[i - 1] - 1.0
            for i in range(1, len(path)) if path[i - 1] > 0.0]
    out.update(_returns_metrics(rets))
    dates_path = [d for d in grid_dates if d in values]
    out["daily_returns"] = {
        dates_path[i + 1].isoformat()[:10]: rets[i] for i in range(len(rets))
    }
    return out


def _daily_return_corr(port_daily: dict, bench_daily: dict) -> Optional[float]:
    """Pearson correlation of overlapping daily returns (None if degenerate)."""
    common = sorted(set(port_daily) & set(bench_daily))
    if len(common) < 3:
        return None
    a = [port_daily[d] for d in common]
    b = [bench_daily[d] for d in common]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    if var_a <= 0 or var_b <= 0:
        return None
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    corr = cov / math.sqrt(var_a * var_b)
    return corr if math.isfinite(corr) else None


def _comparison(port: dict, bench: dict) -> dict:
    """Excess CAGR / Sharpe delta / drawdown delta / correlation."""
    port_cagr = port.get("cagr")
    bench_cagr = bench.get("cagr")
    port_sharpe = port.get("sharpe")
    bench_sharpe = bench.get("sharpe")
    port_dd = port.get("max_drawdown")
    bench_dd = bench.get("max_drawdown")
    return {
        "excess_cagr": (
            port_cagr - bench_cagr
            if port_cagr is not None and bench_cagr is not None else None
        ),
        "sharpe_delta": (
            port_sharpe - bench_sharpe
            if port_sharpe is not None and bench_sharpe is not None else None
        ),
        # Positive when the portfolio's worst peak-to-trough is shallower.
        "maxdd_delta": (
            bench_dd - port_dd
            if port_dd is not None and bench_dd is not None else None
        ),
        "daily_return_corr": _daily_return_corr(
            port.get("daily_returns") or {}, bench.get("daily_returns") or {}
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_portfolio_backtest(
    tickers: list[str],
    period: str = "2y",
    top_n: int = 5,
    stakes: tuple[float, ...] = (0.20, 0.10),
    cost_bps: float = 10.0,
    benchmark: str = "SPY",
    fetch: Optional[Callable[[str, str], Any]] = None,
    costs: Optional[ExecutionCosts] = None,
) -> dict:
    """Run the portfolio-level overnight-BUY backtest vs a benchmark.

    ``fetch`` defaults to the project's cached ``fetch_ohlcv`` seam and
    is injectable for offline tests.  Per-ticker fetch failures degrade
    to "no entries" (fail-open).  Returns a JSON-safe payload with the
    portfolio metrics per stake, the benchmark metrics on the same
    daily grid, and the head-to-head comparison.

    If ``costs`` is provided, execution costs (spread + slippage) are
    applied using the cost model; otherwise ``cost_bps`` is used as a
    simple round-trip cost.
    """
    if fetch is None:
        from src.data.market_data import fetch_ohlcv

        def fetch(ticker: str, period: str) -> Any:  # noqa: F811
            return fetch_ohlcv(ticker, period=period)

    stake_list: list[float] = []
    for s in stakes:
        try:
            value = float(s)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            stake_list.append(value)
    if not stake_list:
        stake_list = [0.20]

    entries: dict[str, dict] = {}
    requested = 0
    for ticker in tickers:
        name = str(ticker).strip().upper()
        if not name:
            continue
        requested += 1
        try:
            frame = fetch(name, period)
            per_ticker = overnight_buy_entries(frame)
        except Exception:  # noqa: BLE001 - degrade per ticker
            continue
        if per_ticker:
            entries[name] = per_ticker

    grid = sorted({d for per in entries.values() for d in per})
    bench_name = str(benchmark or "").strip().upper()
    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "overnight_buy",
        "period": period,
        "top_n": top_n,
        "cost_bps": cost_bps,
        "spread_bps": costs.spread_bps if costs else None,
        "slippage_bps": costs.slippage_bps if costs else None,
        "cost_scale_by_vol": costs.scale_by_vol if costs else None,
        "benchmark_ticker": bench_name or None,
        "n_tickers_requested": requested,
        "n_tickers_used": len(entries),
        "grid_days": len(grid),
        "first_date": grid[0].isoformat()[:10] if grid else None,
        "last_date": grid[-1].isoformat()[:10] if grid else None,
        "stakes": {},
    }
    if not entries or not grid:
        payload["error"] = "no tradable BUY entries produced"
        return payload

    bench_frame = None
    if bench_name:
        try:
            bench_frame = fetch(bench_name, period)
        except Exception:  # noqa: BLE001 - degrade
            bench_frame = None
    bench = benchmark_metrics(bench_frame, grid)
    payload["benchmark"] = bench

    for stake in stake_list:
        sim = simulate_portfolio_overnight(
            entries, top_n=top_n, stake_pct=stake, cost_bps=cost_bps, costs=costs,
        )
        payload["stakes"][f"{stake:.2f}"] = {
            "portfolio": sim["metrics"],
            "comparison": _comparison(sim["metrics"], bench),
        }
    return payload
