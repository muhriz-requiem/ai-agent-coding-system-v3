"""Tests for the pairs backtest parameter sweep (src/pairs/backtest.py).

A small synthetic 6-name universe with planted cointegrated pairs
exercises the sweep plumbing offline (injectable, counting fetch):
grid expansion, param conversion, base-param merge, and per-ticker
fetch reuse.  Structure assertions only -- pair formation behavior is
covered by test_pairs_backtest.py.
"""

import importlib.util


def _impl_ready() -> bool:
    try:
        return importlib.util.find_spec("src.pairs.backtest") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import numpy as np
import pandas as pd
import pytest

from src.pairs.backtest import (
    PAIRS_PARAM_SWEEP_ALLOWED,
    parse_pairs_param_option,
    run_pairs_param_sweep,
)

N = 160


def _universe_frames() -> dict[str, pd.DataFrame]:
    """Two planted pairs (AAA/BBB, CCC/DDD) + two noise names."""
    rng = np.random.default_rng(21)
    idx = pd.date_range("2024-01-01", periods=N, freq="B")

    def _pair(log_b, beta, alpha, phi, sigma):
        spread = np.zeros(N)
        for t in range(1, N):
            spread[t] = phi * spread[t - 1] + rng.normal(0.0, sigma)
        return np.exp(log_b * beta + alpha + spread)

    log_b1 = np.cumsum(rng.normal(0.0002, 0.01, N)) + np.log(100.0)
    log_b2 = np.cumsum(rng.normal(0.0001, 0.008, N)) + np.log(80.0)
    volume = np.full(N, 1_000_000.0)
    frames = {
        "AAA": pd.DataFrame({"Close": _pair(log_b1, 1.2, 0.3, 0.90, 0.006),
                             "Volume": volume}, index=idx),
        "BBB": pd.DataFrame({"Close": np.exp(log_b1), "Volume": volume},
                            index=idx),
        "CCC": pd.DataFrame({"Close": _pair(log_b2, 0.8, 0.2, 0.90, 0.004),
                             "Volume": volume}, index=idx),
        "DDD": pd.DataFrame({"Close": np.exp(log_b2), "Volume": volume},
                            index=idx),
    }
    for nm in ("EEE", "FFF"):
        walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, N)))
        frames[nm] = pd.DataFrame({"Close": walk, "Volume": volume},
                                  index=idx)
    return frames


def _fetch(frames):
    def fetch(ticker: str, period: str):
        return frames[ticker]

    return fetch


def test_parse_pairs_param_option_valid():
    name, values = parse_pairs_param_option(" z_entry = 1.5 , 2.0 , 2.5 ")
    assert name == "z_entry"
    assert values == ["1.5", "2.0", "2.5"]


def test_parse_pairs_param_option_rejects_non_whitelisted():
    with pytest.raises(ValueError, match="not allowed"):
        parse_pairs_param_option("junk_param=1.0")
    # max_lag is a real run_pairs_backtest kwarg but deliberately not sweepable
    assert "max_lag" not in PAIRS_PARAM_SWEEP_ALLOWED
    with pytest.raises(ValueError, match="not allowed"):
        parse_pairs_param_option("max_lag=2")


def test_parse_pairs_param_option_rejects_malformed():
    with pytest.raises(ValueError, match="invalid param format"):
        parse_pairs_param_option("no_equals_sign")


def test_parse_pairs_param_option_rejects_empty_values():
    with pytest.raises(ValueError, match="requires at least one value"):
        parse_pairs_param_option("z_entry=")


def test_run_pairs_param_sweep_grid_combinations():
    frames = _universe_frames()
    sweep = run_pairs_param_sweep(
        list(frames), period="1y",
        base_params={"formation_window": 60, "rebalance_days": 30,
                     "min_trades": 10},
        param_grid={"z_entry": ["2.0", "2.5"]},
        fetch=_fetch(frames),
    )
    rows = [r for r in sweep["results"] if "error" not in r]
    assert len(rows) == 2
    # Successful rows carry the converted (typed) param values.
    assert {r["params"]["z_entry"] for r in rows} == {2.0, 2.5}
    for row in rows:
        # Every row carries the stats shape the analysis step reads.
        assert "aggregate" in row and "per_pair" in row
        assert row["universe_size"] == 6
        # base params merged: formation_window visible in the stats doc
        assert row["formation_window"] == 60
        assert row["rebalance_days"] == 30
    assert sweep["tickers"] == list(frames)
    assert sweep["generated_at"]


def test_run_pairs_param_sweep_multi_axis_grid():
    frames = _universe_frames()
    sweep = run_pairs_param_sweep(
        list(frames), period="1y",
        base_params={"formation_window": 60, "rebalance_days": 30,
                     "min_trades": 10},
        param_grid={"z_entry": ["2.0", "2.5"], "z_stop": ["3.0", "4.0"]},
        fetch=_fetch(frames),
    )
    rows = [r for r in sweep["results"] if "error" not in r]
    combos = {(r["params"]["z_entry"], r["params"]["z_stop"]) for r in rows}
    assert combos == {(2.0, 3.0), (2.0, 4.0), (2.5, 3.0), (2.5, 4.0)}


def test_run_pairs_param_sweep_fetches_once_per_ticker():
    frames = _universe_frames()
    calls: list[str] = []

    def counting_fetch(ticker: str, period: str):
        calls.append(ticker)
        return frames[ticker]

    run_pairs_param_sweep(
        list(frames), period="1y",
        base_params={"formation_window": 60, "rebalance_days": 30},
        param_grid={"z_entry": ["2.0", "2.5"], "z_stop": ["3.0", "4.0"],
                    "z_exit": ["0.5", "1.0"]},
        fetch=counting_fetch,
    )
    # 8 param combos x 6 tickers, but only one fetch per ticker.
    assert len(calls) == 6
    assert len(set(calls)) == 6


def test_run_pairs_param_sweep_invalid_value_degrades_to_error_row():
    frames = _universe_frames()
    sweep = run_pairs_param_sweep(
        list(frames), period="1y",
        base_params={"formation_window": 60, "rebalance_days": 30},
        param_grid={"z_entry": ["junk"]},
        fetch=_fetch(frames),
    )
    assert len(sweep["results"]) == 1
    row = sweep["results"][0]
    assert "error" in row
    assert "invalid value for z_entry" in row["error"]


def test_run_pairs_param_sweep_no_grid_single_run():
    frames = _universe_frames()
    sweep = run_pairs_param_sweep(
        list(frames), period="1y",
        base_params={"formation_window": 60, "rebalance_days": 30,
                     "min_trades": 10},
        param_grid=None,
        fetch=_fetch(frames),
    )
    assert len(sweep["results"]) == 1
    row = sweep["results"][0]
    assert "aggregate" in row
    assert row["formation_window"] == 60
    assert row["period"] == "1y"


def test_run_pairs_param_sweep_period_passed_via_base_params():
    """Regression: CLI passes period via base_params; the injected fetch
    must honor it (a mismatch used to reject every fetch -> universe=0)."""
    frames = _universe_frames()
    sweep = run_pairs_param_sweep(
        list(frames), period="1y",
        base_params={"period": "5y", "formation_window": 60,
                     "rebalance_days": 30, "min_trades": 10},
        param_grid={"z_entry": ["2.0"]},
        fetch=_fetch(frames),
    )
    rows = [r for r in sweep["results"] if "error" not in r]
    assert len(rows) == 1
    assert rows[0]["period"] == "5y"
    assert rows[0]["universe_size"] == 6
