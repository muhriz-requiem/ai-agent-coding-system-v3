"""Tests for the deterministic replay backtest (src/backtest/replay.py).

All price fixtures are hand-built (no network); the fetch seam is faked.
"""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.backtest.replay", "src.quant.signals")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import math

import pandas as pd
import pytest

from src.backtest import replay as replay_mod
from src.backtest.replay import (replay_overnight, replay_signals,
                                 run_backtest, summarize_trades)


# ---------------------------------------------------------------------------
# Fixture builders: shapes that force the quant rules to fire
# ---------------------------------------------------------------------------

def _buy_shape() -> pd.Series:
    """Flat then a sharp dip that rebounds: the dip bars are BUYs (z < -1.5
    with momentum holding up) and the rebound makes them winners."""
    values = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
    return pd.Series(values)


def _sell_shape() -> pd.Series:
    """Flat then a sharp spike that collapses: spike bars are SELLs
    (z > 1.5) and the collapse makes them winners."""
    values = [100.0] * 30 + [105.0, 110.0, 108.0, 106.0, 104.0, 102.0, 100.0]
    return pd.Series(values)


# ---------------------------------------------------------------------------
# replay_signals
# ---------------------------------------------------------------------------


def test_replay_buy_signals_win_on_rebound():
    trades = replay_signals(_buy_shape(), horizon=5)
    assert trades, "fixture must produce at least one trade"
    assert all(t["signal"] == "BUY" for t in trades)
    assert all(t["return"] > 0.0 for t in trades)


def test_replay_sell_signals_win_on_collapse():
    trades = replay_signals(_sell_shape(), horizon=5)
    assert trades, "fixture must produce at least one trade"
    assert all(t["signal"] == "SELL" for t in trades)
    # Direction-adjusted: a SELL wins when the price falls.
    assert all(t["return"] > 0.0 for t in trades)


def test_replay_is_deterministic():
    assert replay_signals(_buy_shape(), horizon=5) == \
        replay_signals(_buy_shape(), horizon=5)


def test_replay_insufficient_data_is_empty():
    assert replay_signals(pd.Series([100.0] * 20), horizon=5) == []
    assert replay_signals(pd.Series([100.0] * 40), horizon=100) == []


def test_replay_junk_input_is_empty():
    for junk in (None, "junk", 42, pd.DataFrame(), pd.Series([1.0, 2.0])):
        assert replay_signals(junk, horizon=5) == [], junk


def test_replay_junk_horizon_is_empty():
    for junk in (0, -3, "junk", None):
        assert replay_signals(_buy_shape(), horizon=junk) == [], junk


def test_replay_ohlcv_frame_input():
    frame = pd.DataFrame({"Close": _buy_shape(),
                          "Volume": range(len(_buy_shape()))})
    trades = replay_signals(frame, horizon=5)
    assert trades == replay_signals(_buy_shape(), horizon=5)


# ---------------------------------------------------------------------------
# summarize_trades
# ---------------------------------------------------------------------------


def _trades(wins: int, losses: int, win_ret=0.03, loss_ret=-0.02) -> list:
    out = [{"signal": "BUY", "return": win_ret} for _ in range(wins)]
    out += [{"signal": "SELL", "return": loss_ret} for _ in range(losses)]
    return out


def test_summarize_kelly_math():
    # W = 0.6, payoff = 0.03 / 0.02 = 1.5 -> kelly = 0.6 - 0.4/1.5 = 1/3.
    summary = summarize_trades(_trades(18, 12), min_trades=30)
    assert summary["n_trades"] == 30
    assert summary["win_rate"] == pytest.approx(0.6)
    assert summary["payoff_ratio"] == pytest.approx(1.5)
    assert summary["kelly_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["usable"] is True


def test_summarize_small_sample_not_usable():
    summary = summarize_trades(_trades(3, 2), min_trades=30)
    assert summary["n_trades"] == 5
    assert summary["kelly_fraction"] is None
    assert summary["usable"] is False


def test_summarize_no_wins_has_no_payoff():
    summary = summarize_trades(_trades(0, 30), min_trades=30)
    assert summary["avg_win"] == 0.0
    assert summary["payoff_ratio"] == 0.0   # avg_win 0 / avg_loss > 0
    assert summary["kelly_fraction"] is None
    assert summary["usable"] is False


def test_summarize_negative_edge_is_not_usable():
    # W = 0.3, R = 1.0 -> kelly = 0.3 - 0.7 = -0.4 -> None (no usable edge).
    summary = summarize_trades(
        _trades(9, 21, win_ret=0.02, loss_ret=-0.02), min_trades=30
    )
    assert summary["payoff_ratio"] == pytest.approx(1.0)
    assert summary["kelly_fraction"] is None
    assert summary["usable"] is False


def test_summarize_empty_trades():
    summary = summarize_trades([], min_trades=30)
    assert summary["n_trades"] == 0
    assert summary["kelly_fraction"] is None
    assert summary["usable"] is False
    assert summary["sharpe"] is None
    assert summary["sortino"] is None
    assert summary["calmar"] is None
    assert summary["omega"] is None


# ---------------------------------------------------------------------------
# run_backtest (fetch seam faked)
# ---------------------------------------------------------------------------


def test_run_backtest_aggregates_and_degrades_per_ticker():
    good = {"GOOD": pd.DataFrame({"Close": _buy_shape()}),
            "SELLY": pd.DataFrame({"Close": _sell_shape()})}

    def fetch(name, period):
        if name == "BAD":
            raise RuntimeError("network down")
        return good[name]

    stats = run_backtest(["GOOD", "SELLY", "BAD"], period="2y",
                         horizon=5, min_trades=1, fetch=fetch)
    assert stats["period"] == "2y"
    assert stats["horizon_days"] == 5
    assert stats["generated_at"]
    assert set(stats["per_ticker"]) == {"GOOD", "SELLY", "BAD"}
    assert stats["per_ticker"]["BAD"]["n_trades"] == 0
    assert stats["per_ticker"]["GOOD"]["n_trades"] > 0
    # Aggregate pools both tickers' trades.
    total = sum(s["n_trades"] for s in stats["per_ticker"].values())
    assert stats["aggregate"]["n_trades"] == total
    # JSON-safe: every numeric leaf is finite.
    def _finite(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(_finite(v) for v in value.values())
        if isinstance(value, list):
            return all(_finite(v) for v in value)
        return True

    assert _finite(stats)


def test_run_backtest_is_deterministic_given_same_data():
    good = {"GOOD": pd.DataFrame({"Close": _buy_shape()})}

    def fetch(name, period):
        return good[name]

    stats_a = run_backtest(["GOOD"], period="2y", horizon=5,
                           min_trades=1, fetch=fetch)
    stats_b = run_backtest(["GOOD"], period="2y", horizon=5,
                           min_trades=1, fetch=fetch)
    assert stats_a["per_ticker"] == stats_b["per_ticker"]
    assert stats_a["aggregate"]["n_trades"] == stats_b["aggregate"]["n_trades"]


def test_run_backtest_default_fetch_failure_degrades(monkeypatch):
    from src.data import market_data

    def boom(ticker, period="6mo", cache=None):
        raise RuntimeError("offline")

    monkeypatch.setattr(market_data, "fetch_ohlcv", boom)
    stats = run_backtest(["GOOD"], period="2y", horizon=5, min_trades=1)
    assert stats["per_ticker"]["GOOD"]["n_trades"] == 0
    assert stats["aggregate"]["usable"] is False


# ---------------------------------------------------------------------------
# replay_overnight (halo / overnight drift: enter close, exit next open)
# ---------------------------------------------------------------------------

def _overnight_frame(gap: float) -> pd.DataFrame:
    """Close series with a quant BUY/SELL shape; Open[t+1] = Close[t]*(1+gap)."""
    closes = _buy_shape()
    opens = closes.shift(1) * (1.0 + gap)
    opens.iloc[0] = closes.iloc[0]
    return pd.DataFrame({"Close": closes, "Open": opens})


def test_replay_overnight_gap_up_is_direction_adjusted():
    trades = replay_overnight(_overnight_frame(+0.01))
    assert trades, "fixture must produce at least one trade"
    assert all(t["signal"] == "BUY" for t in trades)
    for trade in trades:
        assert trade["return"] == pytest.approx(0.01)
        # entry is the signal-day close; exit the NEXT bar's open
        assert trade["exit_price"] == pytest.approx(trade["entry_price"] * 1.01)


def test_replay_overnight_gap_down_sells_win():
    trades = replay_overnight(_overnight_frame(-0.01))
    assert trades
    assert all(t["signal"] == "BUY" for t in trades)
    assert all(t["return"] < 0.0 for t in trades)


def test_replay_overnight_sell_signal_direction(monkeypatch):
    frame = _overnight_frame(+0.02)
    monkeypatch.setattr(replay_mod, "generate_signal",
                        lambda closes: {"signal": "SELL", "confidence": 0.9})
    trades = replay_overnight(frame)
    assert trades
    for trade in trades:
        assert trade["signal"] == "SELL"
        # SELL enters at the close, exits next open +2% -> direction-adjusted -2%.
        assert trade["return"] == pytest.approx(-0.02)
        assert trade["exit_price"] == pytest.approx(trade["entry_price"] * 1.02)


def test_replay_overnight_junk_input_is_empty():
    assert replay_overnight(None) == []
    assert replay_overnight("junk") == []
    assert replay_overnight(pd.DataFrame()) == []
    # Close-only frame (no Open column) cannot score overnight outcomes.
    assert replay_overnight(pd.DataFrame({"Close": _buy_shape()})) == []


def test_replay_overnight_insufficient_data_is_empty():
    short = pd.DataFrame({
        "Close": [100.0] * 20,
        "Open": [101.0] * 20,
    })
    assert replay_overnight(short) == []


def test_replay_overnight_is_deterministic():
    frame = _overnight_frame(+0.01)
    assert replay_overnight(frame) == replay_overnight(frame)


def test_run_backtest_overnight_mode(monkeypatch):
    frames = {
        "GOOD": _overnight_frame(+0.01),
        "BAD": None,
    }

    def fetch(name, period):
        if frames[name] is None:
            raise RuntimeError("no data")
        return frames[name]

    stats = run_backtest(["GOOD", "BAD"], period="2y", horizon=5,
                         min_trades=1, fetch=fetch, mode="overnight")
    assert stats["mode"] == "overnight"
    assert stats["horizon_days"] == 1
    assert stats["per_ticker"]["BAD"]["n_trades"] == 0
    assert stats["per_ticker"]["GOOD"]["n_trades"] > 0
    assert stats["per_ticker"]["GOOD"]["win_rate"] == pytest.approx(1.0)
    total = sum(s["n_trades"] for s in stats["per_ticker"].values())
    assert stats["aggregate"]["n_trades"] == total
    # Side split: the fixture only fires BUY signals.
    assert stats["per_side"]["BUY"]["n_trades"] == total
    assert stats["per_side"]["SELL"]["n_trades"] == 0


def test_run_backtest_close_mode_has_no_per_side():
    good = {"GOOD": pd.DataFrame({"Close": _buy_shape()})}

    def fetch(name, period):
        return good[name]

    stats = run_backtest(["GOOD"], period="2y", horizon=5,
                         min_trades=1, fetch=fetch)
    assert "per_side" not in stats


def test_run_backtest_unknown_mode_raises():
    with pytest.raises(ValueError):
        run_backtest(["GOOD"], mode="junk")
# ---------------------------------------------------------------------------
# Overnight BUY promotion to the primary Kelly-stats file
# ---------------------------------------------------------------------------


def test_primary_payload_extracts_buy_side(monkeypatch):
    from src.backtest.replay import primary_payload

    frames = {"GOOD": _overnight_frame(+0.01)}

    def fetch(name, period):
        return frames[name]

    stats = run_backtest(["GOOD"], period="2y", horizon=5,
                         min_trades=1, fetch=fetch, mode="overnight")
    payload = primary_payload(stats)
    assert payload is not None
    assert payload["mode"] == "overnight_buy"
    assert payload["horizon_days"] == 1
    assert payload["aggregate"] == stats["per_side"]["BUY"]
    assert payload["aggregate"]["n_trades"] > 0


def test_primary_payload_rejects_close_mode():
    from src.backtest.replay import primary_payload

    stats = run_backtest(["GOOD"], period="2y", horizon=5,
                         min_trades=1,
                         fetch=lambda n, p: pd.DataFrame({"Close": _buy_shape()}),
                         mode="close")
    assert primary_payload(stats) is None
    assert primary_payload("junk") is None


def test_replay_signals_uses_full_frame_volume():
    # The volume column must ride along to generate_signal so the
    # volume-confirmation gates can fire during replays.
    prices = _buy_shape()
    frame = pd.DataFrame({"Close": prices,
                          "Volume": [1.0] * len(prices)})
    seen = []

    def fake_generate_signal(prices_arg, *args, **kwargs):
        seen.append(getattr(prices_arg, "columns", None))
        return {"signal": "BUY", "confidence": 0.9}

    original = replay_mod.generate_signal
    replay_mod.generate_signal = fake_generate_signal
    try:
        trades = replay_signals(frame, horizon=5)
    finally:
        replay_mod.generate_signal = original
    assert trades, "fixture must produce trades"
    assert all("Volume" in cols for cols in seen if cols is not None)


def test_replay_overnight_passes_capital_volume_column(monkeypatch):
    # The overnight replay builds a lowercase close/open frame; the
    # Volume column must keep its capital name or generate_signal cannot
    # see it (pre-2026-09-14 divergence vs live: volume_z was pinned 0.0).
    closes = _buy_shape()
    frame = pd.DataFrame({"Close": closes, "Open": closes.shift(1),
                          "Volume": [1_000_000.0] * len(closes)})
    seen = []

    def fake_generate_signal(prices_arg, *args, **kwargs):
        seen.append(getattr(prices_arg, "columns", None))
        return {"signal": "BUY", "confidence": 0.9}

    original = replay_mod.generate_signal
    replay_mod.generate_signal = fake_generate_signal
    try:
        trades = replay_overnight(frame)
    finally:
        replay_mod.generate_signal = original
    assert trades, "fixture must produce trades"
    assert seen, "generate_signal must be called"
    assert all("Volume" in cols for cols in seen if cols is not None)


def test_replay_overnight_volume_gate_matches_live(monkeypatch):
    # With VOLUME_Z_MIN_BUY=0.0 (the shipped .env), a BUY on a
    # below-average-volume dip bar must be blocked -- exactly like live.
    closes = _buy_shape()
    low_volume = [1_000_000.0] * 30 + [100.0] * 7  # dip bars: extreme low vol
    frame = pd.DataFrame({"Close": closes,
                          "Open": closes.shift(1).fillna(closes.iloc[0]),
                          "Volume": low_volume})
    monkeypatch.setenv("VOLUME_Z_MIN_BUY", "0.0")
    gated = replay_overnight(frame)
    assert gated == [], "low-volume dip BUYs must be blocked by the gate"

    monkeypatch.delenv("VOLUME_Z_MIN_BUY", raising=False)
    ungated = replay_overnight(frame)
    assert ungated, "without the gate the dip BUYs must fire"

    # High-volume dip bars pass the gate even with VOLUME_Z_MIN_BUY=0.0.
    high_volume = [100.0] * 30 + [1_000_000.0] * 7
    frame_hi = pd.DataFrame({"Close": closes,
                             "Open": closes.shift(1).fillna(closes.iloc[0]),
                             "Volume": high_volume})
    monkeypatch.setenv("VOLUME_Z_MIN_BUY", "0.0")
    gated_hi = replay_overnight(frame_hi)
    assert gated_hi, "high-volume dip BUYs must pass the gate"


def test_replay_start_index_restricts_trade_collection(monkeypatch):
    # start_index limits collected trades to entry bars at/after the
    # index without changing the signals (full history still feeds them).
    # Unique strictly-increasing prices let us identify each trade's
    # entry bar by its entry price; every bar fires via the stub.
    closes = pd.Series([100.0 + float(i) for i in range(60)])
    monkeypatch.setattr(replay_mod, "generate_signal",
                        lambda prices: {"signal": "BUY", "confidence": 0.9})
    full = replay_signals(closes, horizon=5)
    late = replay_signals(closes, horizon=5, start_index=40)
    assert full and late and len(late) < len(full)
    tail = [t for t in full if t["entry_price"] >= closes.iloc[39]]
    assert late == tail


def test_replay_overnight_start_index_restricts_trade_collection(monkeypatch):
    closes = pd.Series([100.0 + float(i) for i in range(60)])
    frame = pd.DataFrame({"Close": closes,
                          "Open": closes.shift(1).fillna(closes.iloc[0])})
    monkeypatch.setattr(replay_mod, "generate_signal",
                        lambda prices: {"signal": "BUY", "confidence": 0.9})
    full = replay_overnight(frame)
    late = replay_overnight(frame, start_index=40)
    assert full and late and len(late) < len(full)
    # Overnight enters at Close[t], so the entry bar IS the entry price.
    tail = [t for t in full if t["entry_price"] >= closes.iloc[40]]
    assert late == tail
