import math
import os
from typing import Any, Optional

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Env-tunable entry thresholds (defaults match the original fixed rules).
# Junk/absent values fall back to the defaults, so behavior is unchanged
# unless explicitly configured; pytest stays hermetic (no .env load).
# ---------------------------------------------------------------------------

DEFAULT_Z_BUY_ENTRY = -1.5
DEFAULT_Z_SELL_ENTRY = 1.5
DEFAULT_MOMENTUM_FLOOR_PCT = -0.5


def _threshold(name: str, default: float) -> float:
    """Read a numeric threshold from the environment (junk -> default)."""
    try:
        value = float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def compute_z_score(prices: pd.Series, window: int = 20) -> float:
    """
    Compute the z-score for the last price relative to the rolling mean and standard deviation.
    
    Args:
        prices: Series of price data
        window: Rolling window size (default: 20)
    
    Returns:
        z-score as float. Returns 0.0 if insufficient data, zero std, or NaN std.
    """
    if len(prices) < window:
        return 0.0
    
    rolling_mean = prices.rolling(window=window, min_periods=window).mean()
    rolling_std = prices.rolling(window=window, min_periods=window).std(ddof=1)
    
    if len(rolling_mean) < 1 or len(rolling_std) < 1:
        return 0.0
    
    last_mean = rolling_mean.iloc[-1]
    last_std = rolling_std.iloc[-1]
    
    if pd.isna(last_std) or last_std == 0:
        return 0.0
    
    z_score = (prices.iloc[-1] - last_mean) / last_std
    return float(z_score)


def compute_momentum_slope(prices: pd.Series, fast: int = 10, step: int = 5) -> float:
    """
    Compute the momentum slope as the average daily change of the fast
    moving average over the last ``step`` periods, normalized to percent
    of the fast-MA level (percent per day).

    Normalizing makes the BUY gate price-scale-invariant: a $230 stock
    with a -$3/day MA slope (-1.3%/day) is no longer structurally blocked
    by a dollar-denominated threshold, while a $10 stock collapsing
    5%/day still is.

    Args:
        prices: Series of price data
        fast: Fast moving average window (default: 10)
        step: Number of periods to look back for slope calculation (default: 5)

    Returns:
        Momentum slope in percent per day as float. Returns 0.0 if
        insufficient data or the fast-MA level is NaN/non-positive.
    """
    if len(prices) < fast + step:
        return 0.0

    ma_fast = prices.rolling(window=fast, min_periods=fast).mean()

    if len(ma_fast) < step:
        return 0.0

    last = ma_fast.iloc[-1]
    slope = (last - ma_fast.iloc[-step]) / step

    if pd.isna(slope) or pd.isna(last) or last <= 0:
        return 0.0

    return float(slope / last * 100.0)


def compute_daily_vol(prices: pd.Series, window: int = 20) -> float:
    """
    Compute the (annualization-free) standard deviation of daily returns
    over the last ``window`` return observations.

    Args:
        prices: Series of price data
        window: Number of trailing daily returns to use (default: 20)

    Returns:
        Sample std (ddof=1) of daily pct returns as float. Returns 0.0
        when there is insufficient data or the std is NaN.
    """
    returns = prices.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    trailing = returns.iloc[-window:]
    std = trailing.std(ddof=1)
    if pd.isna(std):
        return 0.0
    return float(std)


def _close_series(prices: Any) -> Optional[pd.Series]:
    """Extract the close-price Series from a Series or OHLCV DataFrame."""
    if isinstance(prices, pd.Series):
        return prices
    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            close = prices["Close"]
        elif len(prices.columns):
            close = prices.iloc[:, 0]
        else:
            return None
        if isinstance(close, pd.DataFrame):
            if close.empty:
                return None
            close = close.iloc[:, 0]
        return close
    return None


def _volume_series(prices: Any) -> Optional[pd.Series]:
    """Extract the volume Series from a Series (absent) or OHLCV DataFrame."""
    if not isinstance(prices, pd.DataFrame) or "Volume" not in prices.columns:
        return None
    volume = prices["Volume"]
    if isinstance(volume, pd.DataFrame):
        if volume.empty:
            return None
        volume = volume.iloc[:, 0]
    try:
        volume = pd.to_numeric(volume, errors="coerce")
    except Exception:  # noqa: BLE001 - degrade per ticker
        return None
    if volume.empty:
        return None
    return volume.astype(float)


def compute_volume_z(prices: Any, window: int = 20) -> float:
    """
    Compute the z-score of the latest volume against its trailing window.

    Advisory feature (never gates the signal): a high positive value marks
    a volume spike that may corroborate a price move.  Fails open to 0.0
    when no volume column exists, data is insufficient, or std is zero.

    Args:
        prices: Close Series or OHLCV DataFrame (``Volume`` column used).
        window: Rolling window size (default: 20).

    Returns:
        Volume z-score as float, 0.0 when unavailable.
    """
    volume = _volume_series(prices)
    if volume is None or len(volume) < window:
        return 0.0
    rolling_mean = volume.rolling(window=window, min_periods=window).mean()
    rolling_std = volume.rolling(window=window, min_periods=window).std(ddof=1)
    if len(rolling_mean) < 1 or len(rolling_std) < 1:
        return 0.0
    last_mean = rolling_mean.iloc[-1]
    last_std = rolling_std.iloc[-1]
    last = volume.iloc[-1]
    if pd.isna(last_mean) or pd.isna(last_std) or last_std == 0 or pd.isna(last):
        return 0.0
    return float((last - last_mean) / last_std)


def compute_volume_trend(prices: Any, window: int = 10, step: int = 5) -> float:
    """
    Volume-MA slope in percent per day (accumulation/distribution proxy).

    Advisory feature: rising volume with rising prices corroborates
    conviction; rising volume on falling prices marks distribution.
    Mirrors :func:`compute_momentum_slope` (percent-of-MA normalization,
    price-scale invariant).  Fails open to 0.0 when unavailable.

    Args:
        prices: Close Series or OHLCV DataFrame (``Volume`` column used).
        window: Volume moving-average window (default: 10).
        step: Periods looked back for the slope (default: 5).

    Returns:
        Volume-MA slope in percent per day as float, 0.0 when unavailable.
    """
    volume = _volume_series(prices)
    if volume is None or len(volume) < window + step:
        return 0.0
    ma = volume.rolling(window=window, min_periods=window).mean()
    if len(ma) < step:
        return 0.0
    last = ma.iloc[-1]
    slope = (last - ma.iloc[-step]) / step
    if pd.isna(slope) or pd.isna(last) or last <= 0:
        return 0.0
    return float(slope / last * 100.0)


def attach_regime_features(quant: dict, regime: Any) -> dict:
    """Attach advisory breadth features from the batch regime snapshot.

    Pure helper (no I/O): copies ``fear_z``/``liquidity_z`` composites and
    the gold/equity, equal-weight and small-cap breadth component z-scores
    from the regime snapshot onto the quant dict (``breadth_*`` prefixed).
    Absent regime data or missing components leave the keys out entirely
    (fail-open); the quant dict is never mutated in place.  Advisory only:
    nothing downstream gates on these keys yet.
    """
    out = dict(quant)
    if not isinstance(regime, dict):
        return out
    for key in ("fear_z", "liquidity_z"):
        value = regime.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[key] = float(value)
    components = regime.get("components")
    if isinstance(components, dict):
        for key in ("gold_eq", "rsp_spy", "iwm_spy"):
            value = components.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                out[f"breadth_{key}"] = float(value)
    return out


def _regime_value(regime: Any, key: str, component: Optional[str] = None) -> Optional[float]:
    """Extract a finite float from a regime snapshot (``None`` when absent)."""
    if not isinstance(regime, dict):
        return None
    if component is None:
        value = regime.get(key)
    else:
        components = regime.get("components")
        if not isinstance(components, dict):
            return None
        value = components.get(component)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def generate_signal(prices: Any, regime: Optional[dict] = None) -> dict:
    """
    Generate trading signal based on z-score and momentum slope.

    Entry thresholds are env-tunable: ``Z_BUY_ENTRY`` (default -1.5),
    ``Z_SELL_ENTRY`` (default 1.5) and ``MOMENTUM_FLOOR_PCT`` (default
    -0.5, the falling-knife gate on the fast-MA slope in percent per
    day).  Junk/absent values fall back to the defaults.

    BUY-side confirmation gates (env-tunable, fail-open):

    * ``VOLUME_Z_MIN_BUY`` (default -inf = disabled): BUY requires the
      bar's volume z-score at or above the threshold (volume
      confirmation).
    * ``FEAR_Z_MAX_BUY`` (default +inf = disabled): BUY blocked when the
      regime ``fear_z`` composite exceeds the threshold.  Needs the
      ``regime`` snapshot argument; without one the gate passes.
    * ``BREADTH_GOLD_EQ_MIN_BUY`` (default -inf = disabled): BUY blocked
      when the regime gold/equity breadth z-score falls below the
      threshold (extreme risk-off).  Same ``regime`` dependency.

    Every gate evaluates fail-open: missing data (no volume column,
    no regime snapshot) never blocks a BUY.  SELL signals are ungated.

    Args:
        prices: Close-price Series or a full OHLCV DataFrame.  With a
            DataFrame, ``Volume`` feeds the ``volume_z`` /
            ``volume_trend`` features (0.0 without a volume column).
        regime: Optional batch regime snapshot (``fear_z`` composite and
            ``components.gold_eq`` breadth z-score) for the regime gates.

    Returns:
        Dictionary with keys: signal, confidence, z_score, momentum,
        daily_vol, volume_z, volume_trend, gates (the per-gate pass map
        recorded for the audit trail; ``momentum`` is the fast-MA slope
        in percent per day).
    """
    window = 20
    step = 5

    close = _close_series(prices)
    if close is None or len(close) < window + step:
        return {
            "signal": "HOLD",
            "confidence": 0.5,
            "z_score": 0.0,
            "momentum": 0.0,
            "daily_vol": 0.0,
            "volume_z": 0.0,
            "volume_trend": 0.0,
            "gates": {"volume_ok": True, "fear_ok": True, "breadth_ok": True},
        }

    z_score = compute_z_score(close, window)
    momentum = compute_momentum_slope(close, fast=10, step=step)
    daily_vol = compute_daily_vol(close, window)
    volume_z = compute_volume_z(prices, window)
    volume_trend = compute_volume_trend(prices, window=10, step=step)

    z_buy = _threshold("Z_BUY_ENTRY", DEFAULT_Z_BUY_ENTRY)
    z_sell = _threshold("Z_SELL_ENTRY", DEFAULT_Z_SELL_ENTRY)
    momentum_floor = _threshold("MOMENTUM_FLOOR_PCT", DEFAULT_MOMENTUM_FLOOR_PCT)
    # Confirmation gates (fail-open defaults = disabled).
    vol_gate_min = _threshold("VOLUME_Z_MIN_BUY", float("-inf"))
    fear_gate_max = _threshold("FEAR_Z_MAX_BUY", float("inf"))
    breadth_gate_min = _threshold("BREADTH_GOLD_EQ_MIN_BUY", float("-inf"))

    # Gate evaluations (BUY-side only; fail-open on missing data).
    fear_z = _regime_value(regime, "fear_z")
    breadth_gold_eq = _regime_value(regime, "components", "gold_eq")
    gates = {
        "volume_ok": bool(volume_z >= vol_gate_min),
        "fear_ok": bool(fear_z is None or fear_z <= fear_gate_max),
        "breadth_ok": bool(breadth_gold_eq is None or breadth_gold_eq >= breadth_gate_min),
    }

    # Apply trading rules
    if z_score < z_buy and momentum > momentum_floor:
        if all(gates.values()):
            confidence = min(0.95, abs(z_score) * 0.4)
            signal = "BUY"
        else:
            signal = "HOLD"
            confidence = 0.5
    elif z_score > z_sell:
        confidence = min(0.95, z_score * 0.4)
        signal = "SELL"
    else:
        signal = "HOLD"
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": float(confidence),
        "z_score": float(z_score),
        "momentum": float(momentum),
        "daily_vol": float(daily_vol),
        "volume_z": float(volume_z),
        "volume_trend": float(volume_trend),
        "gates": gates,
    }
