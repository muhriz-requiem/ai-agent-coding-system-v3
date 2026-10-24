"""Multi-period / multi-horizon / multi-mode backtest sweep.

Runs :func:`src.backtest.replay.run_backtest` over a grid of (mode,
period, horizon) combos and returns a JSON-safe matrix of aggregate +
per-ticker stats for offline analysis.  The overnight mode ignores the
horizon (it is always close -> next open), so each overnight period is
evaluated once regardless of the horizon list.

Deterministic given the fetched data; every per-combo failure degrades
to an ``error`` entry (fail-open) instead of aborting the sweep.
"""

from __future__ import annotations

import itertools
import os
from datetime import datetime, timezone
from typing import Any, Callable, Optional

__all__ = [
    "run_sweep",
    "run_walkforward",
    "parse_list",
    "parse_param_option",
    "run_param_sweep",
    "PARAM_SWEEP_ALLOWED",
]


#: Environment variables that the signal generator reads per-call
#: (``src/quant.signals.generate_signal``).  Only these may be swept.
PARAM_SWEEP_ALLOWED = (
    "Z_BUY_ENTRY",
    "Z_SELL_ENTRY",
    "MOMENTUM_FLOOR_PCT",
    "VOLUME_Z_MIN_BUY",
)


def parse_list(raw: Optional[str], default: list[str]) -> list[str]:
    """Comma-separated CLI list -> stripped non-empty items (default fallback)."""
    if not raw or not raw.strip():
        return list(default)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or list(default)


def parse_param_option(raw: str) -> tuple[str, list[str]]:
    """Parse a single ``--param NAME=v1,v2`` option.

    Returns ``(name, [v1, v2, ...])``.  Empty value string means "unset"
    (the env var is removed for that combo).  Raises ``ValueError`` if the
    format is invalid or the name is not in :data:`PARAM_SWEEP_ALLOWED`.
    """
    if "=" not in raw:
        raise ValueError(f"invalid param format (expected NAME=v1,v2): {raw!r}")
    name, vals = raw.split("=", 1)
    name = name.strip()
    if name not in PARAM_SWEEP_ALLOWED:
        raise ValueError(
            f"param {name!r} not allowed; allowed: {', '.join(PARAM_SWEEP_ALLOWED)}"
        )
    values = [v.strip() for v in vals.split(",") if v.strip() != ""]
    # Empty list means "unset" is one of the grid points
    if not values:
        values = [""]
    return name, values


def _prefetch_frames(
    tickers: list[str],
    periods: list[str],
    fetch: Optional[Callable[[str, str], Any]],
) -> dict[tuple[str, str], Any]:
    """Fetch each (ticker, period) once and return a lookup dict."""
    frames: dict[tuple[str, str], Any] = {}
    if fetch is None:
        from src.data.market_data import fetch_ohlcv
        fetch = fetch_ohlcv
    for ticker in tickers:
        for period in periods:
            try:
                frames[(ticker, period)] = fetch(ticker, period)
            except Exception:  # noqa: BLE001 - degrade per ticker/period
                pass
    return frames


def run_sweep(
    tickers: list[str],
    periods: list[str],
    horizons: list[int],
    modes: list[str],
    min_trades: int = 30,
    fetch: Optional[Callable[[str, str], Any]] = None,
    costs: Optional["ExecutionCosts"] = None,
) -> dict:
    """Run the replay grid and return the JSON-safe sweep document.

    ``modes`` accepts ``close`` and ``overnight``; the horizon axis only
    applies to ``close`` replays.  Results carry per-combo
    ``aggregate`` and ``per_ticker`` summaries produced by
    :func:`run_backtest` (Sharpe/Sortino/Calmar/Omega included).

    If ``costs`` is provided, execution costs are applied to all combos.
    """
    from src.backtest.replay import run_backtest

    results: list[dict] = []
    for mode in modes:
        if mode not in ("close", "overnight"):
            results.append({"mode": mode, "error": f"unknown mode: {mode!r}"})
            continue
        combo_horizons = [1] if mode == "overnight" else horizons
        for period in periods:
            for horizon in combo_horizons:
                try:
                    stats = run_backtest(
                        tickers, period=period, horizon=horizon,
                        min_trades=min_trades, fetch=fetch, mode=mode,
                        costs=costs,
                    )
                    results.append({
                        "mode": mode,
                        "period": period,
                        "horizon_days": stats.get("horizon_days"),
                        "aggregate": stats.get("aggregate"),
                        "per_ticker": stats.get("per_ticker"),
                        "per_side": stats.get("per_side"),
                    })
                except Exception as exc:  # noqa: BLE001 - fail-open per combo
                    results.append({
                        "mode": mode,
                        "period": period,
                        "horizon_days": horizon,
                        "error": str(exc),
                    })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": list(tickers),
        "results": results,
    }


def run_param_sweep(
    tickers: list[str],
    periods: list[str],
    horizons: list[int],
    modes: list[str],
    param_grid: dict[str, list[str]],
    min_trades: int = 30,
    fetch: Optional[Callable[[str, str], Any]] = None,
    costs: Optional["ExecutionCosts"] = None,
) -> dict:
    """Run a sweep over mode/period/horizon **and** env-var param combos.

    For each base combo (mode, period, horizon) and each param combination
    from ``param_grid``, temporarily sets the whitelisted env vars, runs
    :func:`run_backtest` with an injected fetch that reuses pre-fetched
    frames, then restores the original environment exactly.

    ``param_grid`` maps env var name to a list of string values.  An empty
    string value means "unset the variable for this combo".

    Returns a sweep document with an additional ``params`` field on each
    successful result row.

    If ``costs`` is provided, execution costs are applied to all combos.
    """
    from src.backtest.replay import run_backtest

    # Prefetch all frames once
    frames = _prefetch_frames(tickers, periods, fetch)

    def _injected_fetch(name: str, period: str) -> Any:
        frame = frames.get((name, period))
        if frame is None:
            raise RuntimeError(f"no prefetched frame for {name}:{period}")
        return frame

    # Build param grid combinations
    if not param_grid:
        # No param grid -> fall back to base sweep behavior
        return run_sweep(tickers, periods, horizons, modes, min_trades, _injected_fetch, costs)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))

    results: list[dict] = []
    for mode in modes:
        if mode not in ("close", "overnight"):
            # Still emit an error row for each param combo for consistency
            for combo_vals in combos:
                params = dict(zip(param_names, combo_vals))
                results.append({"mode": mode, "params": params, "error": f"unknown mode: {mode!r}"})
            continue
        combo_horizons = [1] if mode == "overnight" else horizons
        for period in periods:
            for horizon in combo_horizons:
                for combo_vals in combos:
                    params = dict(zip(param_names, combo_vals))
                    # Save original env values (including absent -> None)
                    saved = {}
                    for key in params:
                        if key in os.environ:
                            saved[key] = os.environ[key]
                        else:
                            saved[key] = None
                    # Apply param values
                    try:
                        for key, val in params.items():
                            if val == "":
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = val
                        stats = run_backtest(
                            tickers, period=period, horizon=horizon,
                            min_trades=min_trades, fetch=_injected_fetch, mode=mode,
                            costs=costs,
                        )
                        results.append({
                            "mode": mode,
                            "period": period,
                            "horizon_days": stats.get("horizon_days"),
                            "params": params,
                            "aggregate": stats.get("aggregate"),
                            "per_ticker": stats.get("per_ticker"),
                            "per_side": stats.get("per_side"),
                        })
                    except Exception as exc:  # noqa: BLE001 - fail-open per combo
                        results.append({
                            "mode": mode,
                            "period": period,
                            "horizon_days": horizon,
                            "params": params,
                            "error": str(exc),
                        })
                    finally:
                        # Restore original env exactly
                        for key, val in saved.items():
                            if val is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = val

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickers": list(tickers),
        "results": results,
    }


def run_walkforward(
    tickers: list[str],
    period: str = "5y",
    train_ratio: float = 0.7,
    step_days: int = 63,
    min_train_days: int = 252,
    horizon: int = 5,
    min_trades: int = 30,
    mode: str = "close",
    fetch: Optional[Callable[[str, str], Any]] = None,
    costs: Optional["ExecutionCosts"] = None,
) -> dict:
    """Walk-forward / out-of-sample backtest with expanding windows.

    For each ticker, splits the price history into sequential folds:
    - Train window: first ``train_ratio`` of data (min ``min_train_days``)
    - Test window: next ``step_days`` bars
    - Then expand train by ``step_days`` and repeat until end of data

    Each fold replays the frame sliced at ``test_end + horizon`` (the
    signal generator sees the full history up to each bar -- identical to
    the live no-lookahead contract; the extra look-ahead room lets every
    in-window entry's outcome be scored) but only collects trades whose
    entry bar falls inside ``[train_end, test_end)`` via the replay's
    ``start_index``/``end_index`` bounds.  Fold trade populations are
    therefore disjoint (they exactly tile the post-train region) and
    genuinely out-of-sample; because the signal features use rolling
    windows of at most 20 bars, the slice changes nothing about the
    signals inside the test window.

    If ``mode="overnight"``, horizon is ignored (always 1 day).

    This evaluates fixed parameters on unseen data (no re-optimization).
    For param re-optimization per fold, use ``run_param_sweep`` on each
    train window manually.

    Returns a document with per-fold summaries and the pooled
    ``aggregate_oos`` across every fold and ticker.
    """
    from src.backtest.replay import (  # noqa: F401 - re-exported helpers
        _closes_from_frame,
        replay_overnight,
        replay_signals,
        summarize_trades,
    )

    if mode not in ("close", "overnight"):
        raise ValueError(f"unknown mode: {mode!r}")

    if fetch is None:
        from src.data.market_data import fetch_ohlcv

        def fetch(ticker: str, period: str) -> Any:  # noqa: F811
            return fetch_ohlcv(ticker, period=period)

    horizon_eff = 1 if mode == "overnight" else horizon

    folds_results: list[dict] = []
    all_oos_trades: list[dict] = []

    for ticker in tickers:
        name = str(ticker).strip().upper()
        if not name:
            continue
        try:
            data = fetch(name, period)
        except Exception:  # noqa: BLE001 - degrade per ticker
            folds_results.append({
                "ticker": name,
                "error": "fetch failed",
            })
            continue

        series = _closes_from_frame(data)
        if series is None or len(series) < min_train_days + step_days:
            folds_results.append({
                "ticker": name,
                "error": "insufficient data",
            })
            continue

        n = len(series)
        train_end = int(n * train_ratio)
        if train_end < min_train_days:
            train_end = min_train_days
        if train_end >= n - step_days:
            folds_results.append({
                "ticker": name,
                "error": "not enough data for walk-forward",
            })
            continue

        fold_num = 0
        while train_end < n:
            test_end = min(train_end + step_days, n)
            # Signal generation sees the full history up to each bar (no
            # lookahead: bars after test_end + horizon are excluded); the
            # look-ahead room lets every in-window entry's outcome be
            # scored, and end_index keeps entry bars inside the fold so
            # folds tile the post-train region without overlap.
            slice_end = min(test_end + horizon_eff, n)
            test_frame = data
            if hasattr(data, "iloc"):
                test_frame = data.iloc[:slice_end]
            if mode == "overnight":
                trades = replay_overnight(
                    test_frame, costs=costs,
                    start_index=train_end, end_index=test_end,
                )
            else:
                trades = replay_signals(
                    test_frame, horizon=horizon_eff, costs=costs,
                    start_index=train_end, end_index=test_end,
                )

            all_oos_trades.extend(trades)
            folds_results.append({
                "ticker": name,
                "fold": fold_num,
                "train_bars": train_end,
                "test_bars": test_end - train_end,
                "aggregate": summarize_trades(
                    trades, min_trades=min_trades, horizon=horizon_eff,
                ),
            })

            train_end += step_days
            fold_num += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "period": period,
        "train_ratio": train_ratio,
        "step_days": step_days,
        "min_train_days": min_train_days,
        "horizon_days": horizon_eff,
        "min_trades": min_trades,
        "folds": folds_results,
        "aggregate_oos": summarize_trades(
            all_oos_trades, min_trades=min_trades, horizon=horizon_eff,
        ),
    }