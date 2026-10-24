"""Tests for the portfolio-level overnight BUY backtest.

All price fixtures are hand-built (no network); the fetch seam is faked.
The centerpiece is the vectorized-rule fidelity test: the portfolio
sim's BUY table must exactly match a naive generate_signal replay (the
same contract the Kelly replay uses) across several shapes.
"""

from __future__ import annotations


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.backtest.portfolio", "src.backtest.replay",
               "src.quant.signals")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import portfolio as P
from src.backtest.portfolio import (_aligned_close_open, _comparison,
                                    benchmark_metrics,
                                    overnight_buy_entries,
                                    run_portfolio_backtest,
                                    simulate_portfolio_overnight)
from src.quant.signals import generate_signal


# ---------------------------------------------------------------------------
# Fixture builders (OHLCV frames with Open + Close; Open gaps up 1% each
# morning so every overnight trade wins exactly +1% gross)
# ---------------------------------------------------------------------------

def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _frame(closes: list[float], volume: bool = False) -> pd.DataFrame:
    n = len(closes)
    opens = [closes[0]] + [closes[i - 1] * 1.01 for i in range(1, n)]
    data = {"Open": opens, "Close": closes}
    if volume:
        data["Volume"] = [1_000_000.0] * n
    return pd.DataFrame(data, index=_dates(n))


def _buy_frame(volume: bool = False) -> pd.DataFrame:
    """Flat then a dip that rebounds: dip bars fire BUYs."""
    closes = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
    return _frame(closes, volume=volume)


def _sell_frame() -> pd.DataFrame:
    """Flat then a spike that collapses: SELLs only, never a BUY."""
    closes = [100.0] * 30 + [105.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
    return _frame(closes)


def _noise_frame(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.0005, scale=0.015, size=n)
    closes = list(100.0 * np.cumprod(1.0 + steps))
    return _frame(closes)


def _spy_frame(n: int = 320) -> pd.DataFrame:
    """Smooth uptrend covering every fixture date (benchmark)."""
    closes = [100.0 * (1.001 ** i) for i in range(n)]
    return _frame(closes)


# ---------------------------------------------------------------------------
# Naive reference: the generate_signal rule path the replay uses
# ---------------------------------------------------------------------------

def _naive_buy_entries(frame: pd.DataFrame) -> dict:
    aligned = _aligned_close_open(frame)
    if aligned is None:
        return {}
    close, open_, volume = aligned
    out: dict = {}
    for t in range(P._MIN_WINDOW - 1, len(close) - 1):
        cols = [close.iloc[:t + 1].rename("close"),
                open_.iloc[:t + 1].rename("open")]
        # Capital "Volume" rides along exactly as the overnight replay
        # builds its frame, so generate_signal computes real volume z.
        if volume is not None:
            cols.append(volume.iloc[:t + 1].rename("Volume"))
        sig = generate_signal(pd.concat(cols, axis=1))
        if str(sig.get("signal", "")).upper() != "BUY":
            continue
        out[close.index[t]] = {
            "entry": float(close.iloc[t]),
            "exit": float(open_.iloc[t + 1]),
            "confidence": float(sig.get("confidence", 0.0)),
        }
    return out


# ---------------------------------------------------------------------------
# Vectorized rule fidelity vs generate_signal
# ---------------------------------------------------------------------------

def test_buy_frame_fires_buys():
    entries = overnight_buy_entries(_buy_frame())
    assert entries, "fixture must produce at least one BUY entry"
    for info in entries.values():
        # Open gaps up 1% each morning: exit = entry * 1.01 exactly.
        assert info["exit"] == pytest.approx(info["entry"] * 1.01)
        assert 0.0 < info["confidence"] <= 0.95


def test_vectorized_rule_matches_generate_signal_buy_frame():
    frame = _buy_frame(volume=True)
    assert overnight_buy_entries(frame) == _naive_buy_entries(frame)


def test_vectorized_rule_matches_generate_signal_noise():
    frame = _noise_frame()
    naive = _naive_buy_entries(frame)
    fast = overnight_buy_entries(frame)
    assert len(naive) > 10, "noise fixture must fire many BUYs"
    assert fast == naive


def test_sell_shape_never_buys():
    assert overnight_buy_entries(_sell_frame()) == {}
    assert overnight_buy_entries(_sell_frame()) == _naive_buy_entries(_sell_frame())


def test_no_lookahead_future_bars_do_not_change_entries():
    frame = _buy_frame()
    before = overnight_buy_entries(frame)
    # Perturb the closes of the last 5 bars only (opens untouched so
    # legitimate exit prices stay identical for earlier entries).
    k = len(frame) - 5
    mutated = frame.copy()
    mutated.iloc[k:, mutated.columns.get_loc("Close")] *= 2.0
    after = overnight_buy_entries(mutated)
    cutoff = frame.index[k - 1]  # entries strictly before the perturbation
    kept = {d for d in before if d < cutoff}
    assert kept, "fixture must have entries before the perturbation window"
    # No new entries appear before the cutoff either.
    assert {d for d in after if d < cutoff} == kept
    for d in kept:
        assert after.get(d) == before[d]


def test_junk_frames_degrade_to_empty():
    short = pd.DataFrame({"Open": [1.0] * 10, "Close": [1.0] * 10},
                         index=_dates(10))
    for junk in (None, "junk", 42, pd.DataFrame(), short, _sell_frame()):
        assert overnight_buy_entries(junk) == {}, junk


def test_missing_open_column_degrades():
    frame = pd.DataFrame({"Close": [100.0] * 40}, index=_dates(40))
    assert overnight_buy_entries(frame) == {}


def test_duplicate_and_string_dates_are_normalized():
    closes = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
    n = len(closes)
    opens = [closes[0]] + [closes[i - 1] * 1.01 for i in range(1, n)]
    # String dates plus a duplicated row: must parse, dedupe, still fire.
    idx = [d.strftime("%Y-%m-%d") for d in _dates(n)]
    frame = pd.DataFrame({"Open": opens, "Close": closes}, index=idx)
    frame = pd.concat([frame, frame.iloc[[-1]]])
    entries = overnight_buy_entries(frame)
    naive = _naive_buy_entries(frame)
    assert entries == naive
    assert entries


# ---------------------------------------------------------------------------
# simulate_portfolio_overnight: exact equity math
# ---------------------------------------------------------------------------

def _d(i: int) -> pd.Timestamp:
    return _dates(10)[i]


def _entries_fixture() -> dict:
    return {
        "AAA": {
            _d(0): {"entry": 100.0, "exit": 101.0, "confidence": 0.9},
            _d(1): {"entry": 100.0, "exit": 102.0, "confidence": 0.5},
        },
        "BBB": {
            _d(0): {"entry": 200.0, "exit": 204.0, "confidence": 0.7},
        },
    }


def test_portfolio_equity_math_exact():
    sim = simulate_portfolio_overnight(_entries_fixture(), top_n=2,
                                       stake_pct=0.20, cost_bps=0.0)
    m = sim["metrics"]
    # Day 0: 0.2*1% (AAA) + 0.2*2% (BBB) = 0.6%; day 1: 0.2*2% (AAA).
    assert m["daily_returns"][_d(0).isoformat()[:10]] == pytest.approx(0.006)
    assert m["daily_returns"][_d(1).isoformat()[:10]] == pytest.approx(0.004)
    assert sim["final_equity"] == pytest.approx(1.006 * 1.004)
    assert m["total_return"] == pytest.approx(1.006 * 1.004 - 1.0)
    assert m["n_trades"] == 3
    assert m["selections"] == {"AAA": 2, "BBB": 1}
    assert m["exposure"] == pytest.approx(1.0)
    assert m["n_days"] == 2


def test_costs_are_deducted():
    sim = simulate_portfolio_overnight(_entries_fixture(), top_n=2,
                                       stake_pct=0.20, cost_bps=100.0)
    m = sim["metrics"]
    # 100 bps = 1% round trip: day 0 -> 0.2*0% + 0.2*1% = 0.2%.
    assert m["daily_returns"][_d(0).isoformat()[:10]] == pytest.approx(0.002)


def test_top_n_ranks_by_confidence_with_ticker_tiebreak():
    entries = {
        "AAA": {_d(0): {"entry": 100.0, "exit": 103.0, "confidence": 0.5}},
        "BBB": {_d(0): {"entry": 100.0, "exit": 102.0, "confidence": 0.9}},
        "CCC": {_d(0): {"entry": 100.0, "exit": 101.0, "confidence": 0.7}},
    }
    sim = simulate_portfolio_overnight(entries, top_n=2, stake_pct=0.5,
                                       cost_bps=0.0)
    m = sim["metrics"]
    assert m["n_trades"] == 2
    assert m["selections"] == {"BBB": 1, "CCC": 1}
    assert m["daily_returns"][_d(0).isoformat()[:10]] == pytest.approx(0.5 * 0.02 + 0.5 * 0.01)
    # Equal confidence: the alphabetically-first ticker wins.
    tie = {
        "ZZZ": {_d(0): {"entry": 100.0, "exit": 101.0, "confidence": 0.5}},
        "AAA": {_d(0): {"entry": 100.0, "exit": 103.0, "confidence": 0.5}},
    }
    sim = simulate_portfolio_overnight(tie, top_n=1, stake_pct=0.5,
                                       cost_bps=0.0)
    assert sim["metrics"]["selections"] == {"AAA": 1}


def test_stake_cap_limits_deployed_fraction():
    entries = {
        "AAA": {_d(0): {"entry": 100.0, "exit": 110.0, "confidence": 0.9}},
        "BBB": {_d(0): {"entry": 100.0, "exit": 120.0, "confidence": 0.8}},
    }
    sim = simulate_portfolio_overnight(entries, top_n=2, stake_pct=0.90,
                                       cost_bps=0.0)
    m = sim["metrics"]
    # First takes 90%, the second is squeezed to the remaining 10%.
    assert m["daily_returns"][_d(0).isoformat()[:10]] == pytest.approx(0.9 * 0.10 + 0.1 * 0.20)
    assert m["avg_deployed"] == pytest.approx(1.0)


def test_portfolio_metrics_risk_numbers():
    rets = [0.01, -0.01, 0.02, -0.005, 0.015]
    m = P._returns_metrics(rets)
    assert m["n_days"] == 5
    equity = 1.0
    for r in rets:
        equity *= (1.0 + r)
    assert m["total_return"] == pytest.approx(equity - 1.0)
    assert m["sharpe"] > 0
    assert 0.0 < m["max_drawdown"] < 0.02
    empty = P._returns_metrics([])
    assert empty["total_return"] is None
    assert empty["n_days"] == 0


def test_simulation_is_deterministic_and_json_safe():
    sim_a = simulate_portfolio_overnight(_entries_fixture())
    sim_b = simulate_portfolio_overnight(_entries_fixture())
    assert sim_a == sim_b
    json.dumps(sim_a)


# ---------------------------------------------------------------------------
# Benchmark + comparison
# ---------------------------------------------------------------------------

def test_benchmark_metrics_on_grid():
    spy = _spy_frame(n=300)
    grid = list(spy.index[:280])
    bench = benchmark_metrics(spy, grid)
    assert bench["n_days"] == 279
    closes = [100.0 * (1.001 ** i) for i in range(280)]
    expected_total = math.prod(c / p for c, p in zip(closes[1:], closes[:-1])) - 1.0
    assert bench["total_return"] == pytest.approx(expected_total)
    assert bench["sharpe"] > 0
    assert bench["max_drawdown"] == pytest.approx(0.0)
    assert len(bench["daily_returns"]) == 279


def test_benchmark_degrades_on_junk():
    empty = benchmark_metrics(None, [_d(0), _d(1)])
    assert empty["n_days"] == 0
    assert empty["total_return"] is None
    no_close = pd.DataFrame({"Px": [1.0, 2.0]})
    assert benchmark_metrics(no_close, [_d(0), _d(1)])["n_days"] == 0


def test_benchmark_survives_missing_grid_dates():
    spy = _spy_frame(n=300)
    grid = list(spy.index[:280:2])  # every other date
    bench = benchmark_metrics(spy, grid)
    assert bench["n_days"] == len(grid) - 1


def test_comparison_math():
    port = {"cagr": 0.10, "sharpe": 1.0, "max_drawdown": 0.10,
            "daily_returns": {"2024-01-02": 0.01, "2024-01-03": -0.005,
                              "2024-01-04": 0.02}}
    bench = {"cagr": 0.06, "sharpe": 0.6, "max_drawdown": 0.20,
             "daily_returns": {"2024-01-02": 0.005, "2024-01-03": -0.004,
                               "2024-01-04": 0.015}}
    comp = _comparison(port, bench)
    assert comp["excess_cagr"] == pytest.approx(0.04)
    assert comp["sharpe_delta"] == pytest.approx(0.4)
    assert comp["maxdd_delta"] == pytest.approx(0.10)
    assert -1.0 <= comp["daily_return_corr"] <= 1.0
    none_comp = _comparison({"cagr": None}, {"cagr": 0.05})
    assert none_comp["excess_cagr"] is None


# ---------------------------------------------------------------------------
# run_portfolio_backtest with a faked fetch seam
# ---------------------------------------------------------------------------

def _fake_fetch_factory():
    frames = {"AAA": _buy_frame(), "BBB": _noise_frame(), "SPY": _spy_frame()}

    def fake_fetch(ticker: str, period: str) -> pd.DataFrame:
        if ticker == "BAD":
            raise RuntimeError("boom")
        return frames[ticker]

    return fake_fetch


def test_run_portfolio_backtest_payload():
    fetch = _fake_fetch_factory()
    payload = run_portfolio_backtest(
        ["AAA", "BBB", "BAD"], period="2y", top_n=2,
        stakes=(0.20, 0.10), cost_bps=10.0, benchmark="SPY", fetch=fetch,
    )
    assert payload["mode"] == "overnight_buy"
    assert payload["period"] == "2y"
    assert payload["n_tickers_requested"] == 3
    assert payload["n_tickers_used"] == 2  # BAD degraded silently
    assert payload["grid_days"] > 0
    assert set(payload["stakes"]) == {"0.20", "0.10"}
    for result in payload["stakes"].values():
        port = result["portfolio"]
        assert port["n_trades"] > 0
        for key in ("total_return", "cagr", "sharpe", "max_drawdown",
                    "exposure", "n_trades", "daily_returns"):
            assert key in port
        comp = result["comparison"]
        for key in ("excess_cagr", "sharpe_delta", "maxdd_delta",
                    "daily_return_corr"):
            assert key in comp
    bench = payload["benchmark"]
    assert bench["n_days"] == payload["grid_days"] - 1
    assert bench["total_return"] is not None
    json.dumps(payload)  # JSON-safe


def test_run_portfolio_backtest_is_deterministic():
    fetch = _fake_fetch_factory()
    a = run_portfolio_backtest(["AAA", "BBB"], benchmark="SPY", fetch=fetch)
    b = run_portfolio_backtest(["AAA", "BBB"], benchmark="SPY", fetch=fetch)
    a.pop("generated_at")
    b.pop("generated_at")
    assert a == b


def test_run_portfolio_backtest_no_entries_is_an_error_payload():
    def failing_fetch(ticker: str, period: str) -> pd.DataFrame:
        raise RuntimeError("offline")

    payload = run_portfolio_backtest(["AAA"], fetch=failing_fetch)
    assert payload.get("error") == "no tradable BUY entries produced"
    assert payload["stakes"] == {}


def test_run_portfolio_backtest_missing_benchmark_degrades():
    fetch = _fake_fetch_factory()
    payload = run_portfolio_backtest(["AAA"], benchmark="NOSUCH",
                                     fetch=fetch)
    assert payload["benchmark"]["n_days"] == 0
    assert payload["stakes"]["0.20"]["portfolio"]["n_trades"] > 0
