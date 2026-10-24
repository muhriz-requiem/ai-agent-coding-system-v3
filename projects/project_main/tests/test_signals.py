"""Tests for the deterministic quant node (src/quant/signals.py).

Fixture values were computed independently with the blueprint formulas
(pandas rolling mean/std, ddof=1) and are asserted with approx.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.quant.signals",)
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import pandas as pd
import pytest

from src.quant.signals import (attach_regime_features, compute_daily_vol,
                               compute_momentum_slope, compute_volume_trend,
                               compute_volume_z, compute_z_score,
                               generate_signal)


def _frame_with_volume(prices: pd.Series, volumes) -> pd.DataFrame:
    """OHLCV-shaped frame wrapping a close series with a Volume column."""
    return pd.DataFrame({"Close": prices, "Volume": volumes})


def _rising() -> pd.Series:
    """Rising ramp 1..40: overbought extreme (SELL zone)."""
    return pd.Series([float(i) for i in range(1, 41)])


def _falling() -> pd.Series:
    """Falling ramp 40..1: oversold but momentum still falling (HOLD)."""
    return pd.Series([float(i) for i in range(40, 0, -1)])


def _dip_recover() -> pd.Series:
    """Flat with noise, dip, slight recovery: BUY zone fixture."""
    values = [100, 100.1] * 10 + [99.9, 99.8, 99.7, 99.6, 99.5,
                                  99.4, 99.3, 99.2, 99.1, 99.0, 99.05]
    return pd.Series([float(v) for v in values])


def test_z_score_rising_fixture():
    # mean(21..40)=30.5, sample std=5.91608, z=(40-30.5)/5.91608
    assert compute_z_score(_rising()) == pytest.approx(1.605793, abs=1e-4)


def test_z_score_falling_fixture():
    assert compute_z_score(_falling()) == pytest.approx(-1.605793, abs=1e-4)


def test_z_score_zero_std_is_zero():
    flat = pd.Series([100.0] * 25)
    assert compute_z_score(flat) == 0.0


def test_momentum_slope_rising_fixture():
    # ma10[-1]=mean(31..40)=35.5, ma10[-5]=31.5 -> dollar slope 0.8/day,
    # normalized to %/day of the MA level: 0.8 / 35.5 * 100.
    assert compute_momentum_slope(_rising()) == pytest.approx(
        0.8 / 35.5 * 100.0, abs=1e-6)


def test_signal_sell_on_overbought():
    signal = generate_signal(_rising())
    assert signal["signal"] == "SELL"
    assert signal["z_score"] == pytest.approx(1.605793, abs=1e-4)
    assert signal["momentum"] == pytest.approx(0.8 / 35.5 * 100.0, abs=1e-4)
    assert signal["confidence"] == pytest.approx(0.642317, abs=1e-4)


def test_signal_hold_on_falling_knife():
    # Oversold (z < -1.5) but momentum <= -0.5: NOT a buy.
    signal = generate_signal(_falling())
    assert signal["signal"] == "HOLD"
    assert signal["confidence"] == pytest.approx(0.5, abs=1e-9)


def test_signal_buy_on_oversold_with_recovery():
    signal = generate_signal(_dip_recover())
    assert signal["signal"] == "BUY"
    assert signal["z_score"] == pytest.approx(-1.633300, abs=1e-4)
    # Dollar slope -0.075/day on a 99.365 MA level -> -0.0755 %/day.
    assert signal["momentum"] == pytest.approx(-0.075 / 99.365 * 100.0, abs=1e-4)
    assert signal["confidence"] == pytest.approx(0.653320, abs=1e-4)


def test_signal_buy_on_high_priced_name(monkeypatch):
    # Regression (momentum-dollar bug): the SAME dip shape at 10x the
    # price level has a -$0.75/day MA slope in dollars, which the old
    # absolute -0.5-dollar gate structurally blocked; the %-normalized
    # gate (-0.0755 %/day) correctly allows the BUY.
    for var in ("Z_BUY_ENTRY", "Z_SELL_ENTRY", "MOMENTUM_FLOOR_PCT"):
        monkeypatch.delenv(var, raising=False)
    signal = generate_signal(_dip_recover() * 10.0)
    assert signal["signal"] == "BUY"
    assert signal["momentum"] == pytest.approx(-0.075 / 99.365 * 100.0, abs=1e-4)


def test_momentum_floor_env_tunable(monkeypatch):
    # A floor above the fixture's -0.0755 %/day slope vetoes the BUY.
    monkeypatch.setenv("MOMENTUM_FLOOR_PCT", "0.0")
    signal = generate_signal(_dip_recover())
    assert signal["signal"] == "HOLD"
    # A stricter floor still admits it.
    monkeypatch.setenv("MOMENTUM_FLOOR_PCT", "-1.0")
    assert generate_signal(_dip_recover())["signal"] == "BUY"


def test_z_entry_thresholds_env_tunable(monkeypatch):
    flat_with_blip = pd.Series([100.0] * 20 + [100.5] + [100.0] * 19)
    # Defaults: z -0.2236 sits in the neutral zone.
    monkeypatch.delenv("Z_BUY_ENTRY", raising=False)
    monkeypatch.delenv("Z_SELL_ENTRY", raising=False)
    assert generate_signal(flat_with_blip)["signal"] == "HOLD"
    # Looser BUY entry fires on the mild dip (momentum 0 > -0.5 default).
    monkeypatch.setenv("Z_BUY_ENTRY", "-0.2")
    assert generate_signal(flat_with_blip)["signal"] == "BUY"
    # Lowered SELL entry flips the same fixture to a SELL.
    monkeypatch.delenv("Z_BUY_ENTRY", raising=False)
    monkeypatch.setenv("Z_SELL_ENTRY", "-0.5")
    assert generate_signal(flat_with_blip)["signal"] == "SELL"


def test_threshold_junk_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("Z_BUY_ENTRY", "not-a-number")
    monkeypatch.setenv("Z_SELL_ENTRY", "")
    monkeypatch.setenv("MOMENTUM_FLOOR_PCT", "nan")
    # Defaults restored: falling knife stays HOLD, dip-recovery BUYs.
    assert generate_signal(_falling())["signal"] == "HOLD"
    assert generate_signal(_dip_recover())["signal"] == "BUY"


def test_signal_hold_in_neutral_zone():
    flat_with_blip = pd.Series([100.0] * 20 + [100.5] + [100.0] * 19)
    signal = generate_signal(flat_with_blip)
    assert signal["signal"] == "HOLD"
    assert signal["z_score"] == pytest.approx(-0.223607, abs=1e-4)


def test_signal_insufficient_data_is_hold():
    short = pd.Series([100.0, 101.0, 99.0])
    signal = generate_signal(short)
    assert signal["signal"] == "HOLD"
    assert signal["confidence"] == 0.5
    assert signal["z_score"] == 0.0
    assert signal["momentum"] == 0.0


def test_confidence_never_exceeds_0_95():
    extreme = pd.Series([100.0] * 20 + [100.0] * 19 + [80.0])
    signal = generate_signal(extreme)
    assert signal["confidence"] <= 0.95


def test_daily_vol_matches_trailing_return_std():
    prices = _rising()
    signal = generate_signal(prices)
    expected = prices.pct_change().dropna().iloc[-20:].std(ddof=1)
    assert signal["daily_vol"] == pytest.approx(float(expected), abs=1e-12)
    assert compute_daily_vol(prices) == pytest.approx(float(expected), abs=1e-12)


def test_daily_vol_zero_when_insufficient():
    short = pd.Series([100.0, 101.0, 99.0])
    signal = generate_signal(short)
    assert signal["daily_vol"] == 0.0


def test_daily_vol_zero_std_series():
    flat = pd.Series([100.0] * 30)
    assert generate_signal(flat)["daily_vol"] == 0.0


# ---------------------------------------------------------------------------
# Advisory volume features (volume_z / volume_trend)
# ---------------------------------------------------------------------------


def test_volume_z_matches_trailing_window_stats():
    prices = _dip_recover()
    volumes = pd.Series(
        [1_000_000.0] * 24 + [3_000_000.0, 3_500_000.0] + [1_100_000.0] * 5)
    assert len(volumes) == len(prices)
    frame = _frame_with_volume(prices, volumes)
    window = 20
    last = volumes.iloc[-1]
    mean = volumes.rolling(window=window, min_periods=window).mean().iloc[-1]
    std = volumes.rolling(window=window, min_periods=window).std(ddof=1).iloc[-1]
    expected = (last - mean) / std
    assert compute_volume_z(frame) == pytest.approx(float(expected), abs=1e-12)
    assert generate_signal(frame)["volume_z"] == pytest.approx(
        float(expected), abs=1e-12)


def test_volume_z_zero_without_volume_column():
    assert compute_volume_z(_dip_recover()) == 0.0
    assert generate_signal(_dip_recover())["volume_z"] == 0.0


def test_volume_z_zero_when_insufficient():
    frame = _frame_with_volume(pd.Series([100.0, 101.0, 99.0]),
                               pd.Series([1.0, 2.0, 3.0]))
    assert compute_volume_z(frame) == 0.0


def test_volume_z_zero_std_volume():
    frame = _frame_with_volume(_dip_recover(), pd.Series([1_000.0] * 31))
    assert compute_volume_z(frame) == 0.0


def test_volume_trend_matches_manual_ma_slope():
    prices = _dip_recover()
    ramp = [1_000_000.0 * (1.0 + 0.01 * i) for i in range(31)]
    frame = _frame_with_volume(prices, pd.Series(ramp))
    window, step = 10, 5
    volumes = pd.Series(ramp)
    ma = volumes.rolling(window=window, min_periods=window).mean()
    expected = (ma.iloc[-1] - ma.iloc[-step]) / step / ma.iloc[-1] * 100.0
    assert compute_volume_trend(frame) == pytest.approx(float(expected), abs=1e-12)
    assert generate_signal(frame)["volume_trend"] == pytest.approx(
        float(expected), abs=1e-12)


def test_volume_trend_zero_without_volume_column():
    assert compute_volume_trend(_dip_recover()) == 0.0
    assert generate_signal(_dip_recover())["volume_trend"] == 0.0


def test_volume_features_advisory_signal_unchanged():
    # The volume spike must never flip the trading signal: the same close
    # series emits the same signal/confidence with and without volume.
    prices = _dip_recover()
    calm = generate_signal(prices)
    spiked = generate_signal(_frame_with_volume(
        prices, pd.Series([1_000_000.0] * 24 + [9_000_000.0, 9_500_000.0])))
    for key in ("signal", "confidence", "z_score", "momentum", "daily_vol"):
        assert spiked[key] == calm[key]


def test_ohlcv_dataframe_end_to_end():
    # Production shape: fetch_ohlcv returns a full frame with Close+Volume;
    # the signal keys must all be present and finite.
    frame = _frame_with_volume(
        _dip_recover(), pd.Series([2_000_000.0] * 31))
    signal = generate_signal(frame)
    for key in ("signal", "confidence", "z_score", "momentum", "daily_vol",
                "volume_z", "volume_trend"):
        assert key in signal
    assert signal["volume_z"] == 0.0  # constant volume -> zero std -> 0.0
    assert signal["volume_trend"] == 0.0


# ---------------------------------------------------------------------------
# Advisory regime breadth features (attach_regime_features)
# ---------------------------------------------------------------------------


def test_attach_regime_features_copies_composites_and_breadth():
    regime = {
        "fear_z": 1.2,
        "liquidity_z": -1.4,
        "components": {"gold_eq": 0.8, "rsp_spy": None, "iwm_spy": -0.5,
                       "vix": 2.0},
    }
    quant = {"signal": "BUY", "confidence": 0.7}
    enriched = attach_regime_features(quant, regime)
    assert enriched["fear_z"] == 1.2
    assert enriched["liquidity_z"] == -1.4
    assert enriched["breadth_gold_eq"] == 0.8
    assert enriched["breadth_iwm_spy"] == -0.5
    # Missing/irrelevant components stay absent.
    assert "breadth_rsp_spy" not in enriched
    assert "breadth_vix" not in enriched


def test_attach_regime_features_does_not_mutate_input():
    quant = {"signal": "HOLD"}
    regime = {"fear_z": 1.0, "components": {"gold_eq": 0.5}}
    attach_regime_features(quant, regime)
    assert "fear_z" not in quant
    assert "breadth_gold_eq" not in quant


def test_attach_regime_features_fail_open():
    quant = {"signal": "HOLD"}
    assert attach_regime_features(quant, None) == quant
    assert attach_regime_features(quant, "junk") == quant
    assert attach_regime_features(quant, {}) == quant
    assert attach_regime_features(quant, {"fear_z": None}) == quant
    assert attach_regime_features(quant, {"fear_z": float("nan")}) == quant

# ---------------------------------------------------------------------------
# BUY-side confirmation gates (volume / fear / breadth; fail-open)
# ---------------------------------------------------------------------------

REGIME_CALM = {"fear_z": 0.2, "components": {"gold_eq": 0.3}}


def test_gates_disabled_by_default_buy_still_fires():
    signal = generate_signal(_dip_recover())
    assert signal["signal"] == "BUY"
    assert signal["gates"] == {"volume_ok": True, "fear_ok": True,
                               "breadth_ok": True}


def test_volume_gate_blocks_low_volume_buy(monkeypatch):
    monkeypatch.setenv("VOLUME_Z_MIN_BUY", "5.0")
    frame = _frame_with_volume(_dip_recover(), range(len(_dip_recover())))
    signal = generate_signal(frame)
    assert signal["signal"] == "HOLD"
    assert signal["gates"]["volume_ok"] is False
    assert signal["confidence"] == 0.5


def test_volume_gate_passes_at_threshold(monkeypatch):
    # Gate 0.0 with a Series (volume absent -> volume_z 0.0): 0.0 >= 0.0.
    monkeypatch.setenv("VOLUME_Z_MIN_BUY", "0.0")
    signal = generate_signal(_dip_recover(), REGIME_CALM)
    assert signal["signal"] == "BUY"
    assert signal["gates"]["volume_ok"] is True


def test_fear_gate_blocks_extreme_fear_buy(monkeypatch):
    monkeypatch.setenv("FEAR_Z_MAX_BUY", "1.5")
    regime = {"fear_z": 2.5, "components": {"gold_eq": 0.0}}
    signal = generate_signal(_dip_recover(), regime)
    assert signal["signal"] == "HOLD"
    assert signal["gates"]["fear_ok"] is False


def test_breadth_gate_blocks_risk_off_buy(monkeypatch):
    monkeypatch.setenv("BREADTH_GOLD_EQ_MIN_BUY", "-1.0")
    regime = {"fear_z": 0.0, "components": {"gold_eq": -2.0}}
    signal = generate_signal(_dip_recover(), regime)
    assert signal["signal"] == "HOLD"
    assert signal["gates"]["breadth_ok"] is False


def test_regime_gates_fail_open_without_snapshot(monkeypatch):
    monkeypatch.setenv("FEAR_Z_MAX_BUY", "0.0")
    monkeypatch.setenv("BREADTH_GOLD_EQ_MIN_BUY", "1.0")
    signal = generate_signal(_dip_recover(), None)
    assert signal["signal"] == "BUY"  # no regime data -> gates pass


def test_regime_gates_fail_open_on_junk_snapshot(monkeypatch):
    monkeypatch.setenv("FEAR_Z_MAX_BUY", "0.0")
    monkeypatch.setenv("BREADTH_GOLD_EQ_MIN_BUY", "1.0")
    signal = generate_signal(_dip_recover(), {"fear_z": "junk"})
    assert signal["signal"] == "BUY"


def test_volume_gate_does_not_block_sells(monkeypatch):
    monkeypatch.setenv("VOLUME_Z_MIN_BUY", "9.0")
    signal = generate_signal(_rising())
    assert signal["signal"] == "SELL"
