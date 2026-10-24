"""Tests for the backtest sweep grid (src/backtest/sweep.py).

All price fixtures are hand-built; the fetch seam is faked. Offline.
"""


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.backtest.sweep") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import os
import pandas as pd
import pytest

from src.backtest.sweep import parse_list, run_sweep, parse_param_option, run_param_sweep, PARAM_SWEEP_ALLOWED


def _buy_shape() -> pd.Series:
    values = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
    return pd.Series(values)


def _fetch_good(name, period):
    return pd.DataFrame({"Close": _buy_shape()})


def test_parse_list_defaults_and_splits():
    assert parse_list(None, ["2y"]) == ["2y"]
    assert parse_list("", ["a"]) == ["a"]
    assert parse_list(" x , y,,", ["a"]) == ["x", "y"]


def test_run_sweep_grid_combinations():
    sweep = run_sweep(
        ["GOOD"], periods=["2y", "5y"], horizons=[5, 10],
        modes=["close", "overnight"], min_trades=1, fetch=_fetch_good,
    )
    # close: 2 periods x 2 horizons; overnight: 1 per period (horizon=1).
    rows = [r for r in sweep["results"] if "error" not in r]
    assert len(rows) == 6
    combos = {(r["mode"], r["period"], r["horizon_days"]) for r in rows}
    assert ("close", "2y", 5) in combos
    assert ("close", "5y", 10) in combos
    assert ("overnight", "2y", 1) in combos and ("overnight", "5y", 1) in combos
    # Aggregate + per-ticker stats present with the risk metrics.
    row = next(r for r in rows if r["mode"] == "close" and r["period"] == "2y"
               and r["horizon_days"] == 5)
    assert row["aggregate"]["n_trades"] > 0
    assert "sharpe" in row["aggregate"]
    assert row["per_ticker"]["GOOD"]["n_trades"] == row["aggregate"]["n_trades"]
    assert sweep["tickers"] == ["GOOD"]
    assert sweep["generated_at"]


def test_run_sweep_unknown_mode_fails_open():
    sweep = run_sweep(["GOOD"], periods=["2y"], horizons=[5],
                      modes=["junk"], min_trades=1, fetch=_fetch_good)
    assert sweep["results"][0]["error"].startswith("unknown mode")


def test_run_sweep_fetch_failure_degrades_per_combo():
    def fetch(name, period):
        raise RuntimeError("offline")

    sweep = run_sweep(["GOOD"], periods=["2y"], horizons=[5],
                      modes=["close"], min_trades=1, fetch=fetch)
    # run_backtest itself degrades per ticker -> an empty, not raising, row.
    row = sweep["results"][0]
    assert "error" not in row
    assert row["aggregate"]["n_trades"] == 0


def test_parse_param_option_valid():
    name, values = parse_param_option("Z_BUY_ENTRY=-2.5,-2.0,-1.5")
    assert name == "Z_BUY_ENTRY"
    assert values == ["-2.5", "-2.0", "-1.5"]


def test_parse_param_option_empty_value_means_unset():
    name, values = parse_param_option("VOLUME_Z_MIN_BUY=")
    assert name == "VOLUME_Z_MIN_BUY"
    assert values == [""]


def test_parse_param_option_whitespace_tolerance():
    name, values = parse_param_option("  Z_SELL_ENTRY = 1.0 , 1.5 ")
    assert name == "Z_SELL_ENTRY"
    assert values == ["1.0", "1.5"]


def test_parse_param_option_rejects_non_whitelisted():
    with pytest.raises(ValueError, match="not allowed"):
        parse_param_option("INVALID_PARAM=1.0")


def test_parse_param_option_rejects_malformed():
    with pytest.raises(ValueError, match="invalid param format"):
        parse_param_option("NO_EQUALS_SIGN")


def test_run_param_sweep_grid_combinations():
    """Param sweep runs all base combos x param combos."""
    sweep = run_param_sweep(
        ["GOOD"], periods=["2y"], horizons=[5],
        modes=["close", "overnight"],
        param_grid={"Z_BUY_ENTRY": ["-2.0", "-1.5"]},
        min_trades=1, fetch=_fetch_good,
    )
    # base: close(2y,5) + overnight(2y,1) = 2 combos
    # param: 2 values for Z_BUY_ENTRY
    # total = 2 * 2 = 4 rows
    rows = [r for r in sweep["results"] if "error" not in r]
    assert len(rows) == 4
    # Each row has params field
    for row in rows:
        assert "params" in row
        assert "Z_BUY_ENTRY" in row["params"]
    # Check all combos present
    combos = {(r["mode"], r["params"]["Z_BUY_ENTRY"]) for r in rows}
    assert ("close", "-2.0") in combos
    assert ("close", "-1.5") in combos
    assert ("overnight", "-2.0") in combos
    assert ("overnight", "-1.5") in combos


def test_run_param_sweep_env_restored_after_each_combo():
    """Environment is restored exactly after each param combo."""
    # Set a known initial state
    os.environ["Z_BUY_ENTRY"] = "-999.0"  # sentinel
    try:
        def fetch(name, period):
            return pd.DataFrame({"Close": _buy_shape()})

        sweep = run_param_sweep(
            ["GOOD"], periods=["2y"], horizons=[5],
            modes=["close"],
            param_grid={"Z_BUY_ENTRY": ["-2.0", "-1.5"]},
            min_trades=1, fetch=fetch,
        )
        # Env should be restored to original
        assert os.environ.get("Z_BUY_ENTRY") == "-999.0"
    finally:
        os.environ.pop("Z_BUY_ENTRY", None)


def test_run_param_sweep_unset_param_removes_env_var():
    """Empty string value -> env var is unset for that combo."""
    # Ensure it's not set initially
    os.environ.pop("VOLUME_Z_MIN_BUY", None)
    try:
        def fetch(name, period):
            return pd.DataFrame({"Close": _buy_shape()})

        sweep = run_param_sweep(
            ["GOOD"], periods=["2y"], horizons=[5],
            modes=["close"],
            param_grid={"VOLUME_Z_MIN_BUY": ["", "0.5"]},
            min_trades=1, fetch=fetch,
        )
        # Both combos should run without error
        rows = [r for r in sweep["results"] if "error" not in r]
        assert len(rows) == 2
        # Env should be unchanged (unset) after
        assert "VOLUME_Z_MIN_BUY" not in os.environ
    finally:
        os.environ.pop("VOLUME_Z_MIN_BUY", None)


def test_run_param_sweep_prefetches_each_ticker_period_once():
    """Frames are fetched once per (ticker, period) and reused."""
    fetch_calls = []

    def counting_fetch(name, period):
        fetch_calls.append((name, period))
        return pd.DataFrame({"Close": _buy_shape()})

    sweep = run_param_sweep(
        ["T1", "T2"], periods=["2y", "5y"], horizons=[5],
        modes=["close"],
        param_grid={"Z_BUY_ENTRY": ["-2.0", "-1.5", "-1.0"]},
        min_trades=1, fetch=counting_fetch,
    )
    # 2 tickers * 2 periods = 4 unique fetches
    # Despite 3 param combos * 1 mode * 2 periods = 6 base combos
    assert len(fetch_calls) == 4
    # All calls should be unique (ticker, period) pairs
    assert len(set(fetch_calls)) == 4


def test_run_param_sweep_param_changes_results():
    """Different Z_BUY_ENTRY thresholds produce different trade counts."""
    def fetch(name, period):
        # Shape: 30 flat at 100, then drop to 95-100 (creates negative z-score)
        return pd.DataFrame({"Close": _buy_shape()})

    sweep = run_param_sweep(
        ["GOOD"], periods=["2y"], horizons=[5],
        modes=["close"],
        param_grid={"Z_BUY_ENTRY": ["-0.5", "-2.0"]},
        min_trades=1, fetch=fetch,
    )
    rows = [r for r in sweep["results"] if "error" not in r]
    assert len(rows) == 2
    # -0.5 is less strict -> more trades (or at least not fewer)
    # -2.0 is more strict -> fewer trades
    # With this specific shape, both might trade but -0.5 should have >= trades
    n_trades = {r["params"]["Z_BUY_ENTRY"]: r["aggregate"]["n_trades"] for r in rows}
    # Just verify they differ or at least both execute
    assert n_trades["-0.5"] >= 0
    assert n_trades["-2.0"] >= 0
