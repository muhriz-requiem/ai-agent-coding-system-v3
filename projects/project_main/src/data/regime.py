"""Global liquidity & fear regime layer.

Fetches a fixed set of free yfinance series (bonds, USD/JPY, GLD, VIX term
structure, credit, dollar, copper/gold, breadth), computes deterministic
z-score composites, and maps them to a regime used by the order planner:

* ``panic``    -- VIX term-structure backwardation + high spot VIX:
                  blocks BUY orders (SELLs always pass).
* ``risk_off`` -- elevated fear or impaired liquidity: scales the position
                  budget down by ``REGIME_BUDGET_MULT``.
* ``risk_on``  -- strong liquidity + low fear: treated as neutral (never
                  scales the budget up).
* ``neutral``  -- otherwise, and whenever data is missing (fail-open: the
                  gates never fire on absent data).

yfinance is imported lazily in the fetch seam; all computations are pure
and offline-testable.  Every fetch failure degrades that component to
"missing" and the composite falls back toward the neutral regime -- a
regime gate must never crash the pipeline or fire on bad data.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import pandas as pd

from src.data.cache import Cache
from src.data.market_data import fetch_ohlcv

__all__ = [
    "REGIME_NEUTRAL",
    "REGIMES",
    "REGIME_SERIES",
    "compute_regime",
    "fetch_regime",
]

#: Regime labels emitted by :func:`compute_regime`.
PANIC = "panic"
RISK_OFF = "risk_off"
RISK_ON = "risk_on"
NEUTRAL = "neutral"

REGIMES: tuple[str, ...] = (PANIC, RISK_OFF, RISK_ON, NEUTRAL)

#: Snapshot emitted when no regime data is available (fail-open default).
REGIME_NEUTRAL: dict[str, Any] = {
    "regime": NEUTRAL,
    "fear_z": None,
    "liquidity_z": None,
    "vix": None,
    "vix3m": None,
    "vix_term": None,
    "curve_inverted": False,
    "components": {},
    "as_of": None,
}

#: yfinance series the regime layer consumes, grouped by composite.
#: Values are the keys of the ``series`` mapping accepted by
#: :func:`compute_regime`.
REGIME_SERIES: tuple[str, ...] = (
    "^VIX",       # spot VIX level
    "^VIX3M",     # 3-month VIX (term-structure denominator)
    "SPY",        # equity benchmark (fear + breadth denominators)
    "GLD",        # gold (fear bid)
    "CPER",       # copper (growth vs fear)
    "HYG",        # high-yield credit (spread numerator)
    "IEF",        # 7-10y treasuries (spread denominator)
    "JPY=X",      # USD/JPY carry-trade proxy
    "DX-Y.NYB",   # broad dollar index
    "TLT",        # 20y+ treasuries (long-rate slope)
    "^IRX",       # 13-week T-bill yield (short end)
    "^TNX",       # 10y yield (long end)
    "RSP",        # equal-weight S&P (breadth proxy)
    "IWM",        # small caps (risk appetite)
)

#: Lookback window (calendar days) requested for each series.
_REGIME_PERIOD = "1y"

#: Rolling window (observations) for slope/ratio baselines.
_Z_WINDOW = 20

#: Thresholds (deterministic, no tuning knobs in v1).
VIX_BACKWARDATION = 1.00   # VIX / VIX3M >= 1.0 -> backwardation
VIX_PANIC_LEVEL = 30.0     # spot VIX >= 30 while backwardated -> panic
FEAR_Z_THRESHOLD = 1.00    # fear_z > 1.0 -> risk_off
LIQUIDITY_Z_FLOOR = -1.00  # liquidity_z < -1.0 -> risk_off
#: Component z-scores are clamped to this band: a near-constant ratio with
#: one tail blip would otherwise produce astronomical, meaningless scores.
Z_SCORE_CLAMP = 5.0

#: Default gate thresholds as a mapping (item 12a): operators may override
#: these via ``compute_regime``/``fetch_regime`` ``thresholds`` args (wired
#: to env vars in cli._components).  Keys mirror the constants above;
#: unknown keys and junk values are ignored (fail-open to the defaults).
DEFAULT_THRESHOLDS: dict[str, float] = {
    "vix_backwardation": VIX_BACKWARDATION,
    "vix_panic_level": VIX_PANIC_LEVEL,
    "fear_z_threshold": FEAR_Z_THRESHOLD,
    "liquidity_z_floor": LIQUIDITY_Z_FLOOR,
}


def _resolve_thresholds(thresholds: Any) -> dict[str, float]:
    """Merge operator-supplied thresholds over :data:`DEFAULT_THRESHOLDS`.

    Non-dict inputs, unknown keys, junk values and non-finite floats are
    ignored; every other value must parse as a finite float.
    """
    resolved = dict(DEFAULT_THRESHOLDS)
    if not isinstance(thresholds, dict):
        return resolved
    for key in resolved:
        if key not in thresholds:
            continue
        try:
            value = float(thresholds[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            resolved[key] = value
    return resolved


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #


def _series_from_frame(frame: Any) -> Optional[pd.Series]:
    """Extract a close-price Series from an OHLCV frame (or passthrough)."""
    if frame is None:
        return None
    if isinstance(frame, pd.Series):
        series = frame
    elif isinstance(frame, pd.DataFrame):
        if frame.empty:
            return None
        if "Close" in frame.columns:
            close = frame["Close"]
        else:
            close = frame.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            # yfinance MultiIndex columns -> take the first column.
            if close.empty:
                return None
            close = close.iloc[:, 0]
        series = close
    else:
        return None
    series = series.dropna()
    if series.empty:
        return None
    return series


def _zscore(value: float, history: Iterable[float]) -> Optional[float]:
    """Sample z-score of ``value`` against ``history`` (ddof=1)."""
    values = [float(v) for v in history if v is not None and math.isfinite(float(v))]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(var)
    if std <= 0.0:
        return None
    return (float(value) - mean) / std


def _last(series: pd.Series) -> Optional[float]:
    """Last finite value of a Series as float, or None."""
    if series is None or series.empty:
        return None
    try:
        value = float(series.iloc[-1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _slope(series: Optional[pd.Series], window: int = _Z_WINDOW) -> Optional[pd.Series]:
    """Rolling ``window``-day slope (mean daily change) of ``series``."""
    if series is None or len(series) < window + 1:
        return None
    ma = series.rolling(window=window, min_periods=window).mean()
    slope = (ma - ma.shift(window)) / window
    slope = slope.dropna()
    if slope.empty:
        return None
    return slope


def _ratio(a: Optional[pd.Series], b: Optional[pd.Series]) -> Optional[pd.Series]:
    """Elementwise a / b, aligned on the intersection of their indexes."""
    if a is None or b is None:
        return None
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        return None
    left, right = joined.iloc[:, 0], joined.iloc[:, 1]
    if (right == 0).any():
        return None
    return left / right


def _component(
    level: Optional[pd.Series],
    invert: bool = False,
) -> Optional[float]:
    """Z-score of the last value of ``level`` against its prior history.

    The score is clamped to ``[-Z_SCORE_CLAMP, Z_SCORE_CLAMP]`` for
    robustness against near-degenerate history (tiny std).
    """
    if level is None or len(level) < 3:
        return None
    value = _last(level)
    if value is None:
        return None
    score = _zscore(value, level.iloc[:-1])
    if score is None:
        return None
    score = max(-Z_SCORE_CLAMP, min(Z_SCORE_CLAMP, score))
    return -score if invert else score


def _mean_or_none(scores: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-None scores, requiring CORROBORATION.

    A composite only scores when at least two of its components are
    present: a single uncorroborated z-score must never fire a gate
    (fail-open).  Fewer than two components -> ``None``.
    """
    present = [s for s in scores if s is not None and math.isfinite(s)]
    if len(present) < 2:
        return None
    return sum(present) / len(present)


# --------------------------------------------------------------------------- #
# Pure regime computation                                                      #
# --------------------------------------------------------------------------- #


def compute_regime(series: dict[str, Any], thresholds: Any = None) -> dict[str, Any]:
    """Compute the regime snapshot from raw close-price series.

    Args:
        series: Mapping of series name (see :data:`REGIME_SERIES`) to an
            OHLCV DataFrame, a close-price ``pd.Series``, or ``None``.
            Missing/invalid entries degrade that component; if no component
            of a composite is available the composite is ``None``.
        thresholds: Optional mapping of gate-threshold overrides over
            :data:`DEFAULT_THRESHOLDS` (keys: ``vix_backwardation``,
            ``vix_panic_level``, ``fear_z_threshold``,
            ``liquidity_z_floor``).  ``None``/junk entries keep the
            defaults, so the snapshot is unchanged unless configured.

    Returns:
        Snapshot dict: ``regime``, ``fear_z``, ``liquidity_z``, ``vix``,
        ``vix3m``, ``vix_term``, ``curve_inverted``, ``components``
        (per-component z-scores) and ``as_of``.  With no usable data the
        snapshot equals :data:`REGIME_NEUTRAL`.
    """
    if not isinstance(series, dict):
        return dict(REGIME_NEUTRAL)
    th = _resolve_thresholds(thresholds)

    closes: dict[str, Optional[pd.Series]] = {
        name: _series_from_frame(series.get(name)) for name in REGIME_SERIES
    }
    if all(value is None for value in closes.values()):
        return dict(REGIME_NEUTRAL)

    vix, vix3m = closes.get("^VIX"), closes.get("^VIX3M")
    spy, gld = closes.get("SPY"), closes.get("GLD")
    cper = closes.get("CPER")
    hyg, ief = closes.get("HYG"), closes.get("IEF")
    jpy = closes.get("JPY=X")
    dxy = closes.get("DX-Y.NYB")
    tlt = closes.get("TLT")
    irx, tnx = closes.get("^IRX"), closes.get("^TNX")
    rsp, iwm = closes.get("RSP"), closes.get("IWM")

    vix_level = _last(vix) if vix is not None else None
    vix3m_level = _last(vix3m) if vix3m is not None else None
    vix_term: Optional[float] = None
    if vix_level is not None and vix3m_level not in (None, 0.0):
        vix_term = vix_level / vix3m_level

    # -- fear composite -----------------------------------------------------
    comp_vix = _component(vix)
    fear_parts: list[Optional[float]] = [comp_vix]
    vix_term_series = _ratio(vix, vix3m)
    comp_vix_term = _component(vix_term_series) if vix_term_series is not None else None
    if comp_vix_term is not None:
        fear_parts.append(comp_vix_term)
    gld_spy = _ratio(gld, spy)
    comp_gold_eq = _component(_slope(gld_spy)) if gld_spy is not None else None
    if comp_gold_eq is not None:
        fear_parts.append(comp_gold_eq)
    # Copper/gold: rising copper = growth = fear DOWN (inverted).
    copper_gold = _ratio(cper, gld)
    comp_copper_gold = (
        _component(_slope(copper_gold), invert=True) if copper_gold is not None else None
    )
    if comp_copper_gold is not None:
        fear_parts.append(comp_copper_gold)
    fear_z = _mean_or_none(fear_parts)

    # -- liquidity composite -------------------------------------------------
    # USD/JPY NOT inverted: a falling pair = yen strength = carry unwind =
    # liquidity drain (the dominant risk-off signature since 2008).
    comp_usdjpy = _component(_slope(jpy)) if jpy is not None else None
    comp_dxy = _component(_slope(dxy), invert=True) if dxy is not None else None
    comp_tlt = _component(_slope(tlt)) if tlt is not None else None
    hyg_ief = _ratio(hyg, ief)
    comp_hyg_ief = _component(_slope(hyg_ief), invert=True) if hyg_ief is not None else None
    liquidity_parts: list[Optional[float]] = [
        comp_usdjpy, comp_dxy, comp_tlt, comp_hyg_ief,
    ]
    # Breadth / risk-appetite components fold into liquidity.
    comp_rsp = None
    rsp_spy = _ratio(rsp, spy)
    if rsp_spy is not None:
        comp_rsp = _component(_slope(rsp_spy))
        if comp_rsp is not None:
            liquidity_parts.append(comp_rsp)
    comp_iwm = None
    iwm_spy = _ratio(iwm, spy)
    if iwm_spy is not None:
        comp_iwm = _component(_slope(iwm_spy))
        if comp_iwm is not None:
            liquidity_parts.append(comp_iwm)
    liquidity_z = _mean_or_none(liquidity_parts)

    irx_level = _last(irx) if irx is not None else None
    tnx_level = _last(tnx) if tnx is not None else None
    curve_inverted = bool(
        irx_level is not None and tnx_level is not None and (tnx_level - irx_level) < 0
    )

    components = {
        "vix": comp_vix,
        "vix_term": comp_vix_term,
        "gold_eq": comp_gold_eq,
        "copper_gold": comp_copper_gold,
        "usdjpy": comp_usdjpy,
        "dxy": comp_dxy,
        "tlt_slope": comp_tlt,
        "hyg_ief": comp_hyg_ief,
        "rsp_spy": comp_rsp,
        "iwm_spy": comp_iwm,
    }

    # -- regime mapping (deterministic, fail-open) ---------------------------
    backwardated = (
        vix_term is not None
        and vix_term >= th["vix_backwardation"]
    )
    panic = bool(
        backwardated
        and vix_level is not None
        and vix_level >= th["vix_panic_level"]
    )

    risk_off = False
    if fear_z is not None and fear_z > th["fear_z_threshold"]:
        risk_off = True
    if liquidity_z is not None and liquidity_z < th["liquidity_z_floor"]:
        risk_off = True
    if curve_inverted:
        risk_off = True

    if panic:
        regime = PANIC
    elif risk_off:
        regime = RISK_OFF
    else:
        regime = NEUTRAL

    return {
        "regime": regime,
        "fear_z": fear_z,
        "liquidity_z": liquidity_z,
        "vix": vix_level,
        "vix3m": vix3m_level,
        "vix_term": vix_term,
        "curve_inverted": curve_inverted,
        "components": components,
        "as_of": None,
    }


# --------------------------------------------------------------------------- #
# Fetch seam (batch level; called at most once per run)                        #
# --------------------------------------------------------------------------- #


def fetch_regime(
    cache: Cache | None = None,
    period: str = _REGIME_PERIOD,
    thresholds: Any = None,
) -> dict[str, Any]:
    """Fetch every regime series and compute the snapshot.

    Each series is fetched independently through ``fetch_ohlcv`` (module
    attribute, monkeypatchable in tests) with an optional read-through
    cache (key ``ohlcv:<name>:<period>`` inside that helper).  A failure
    in any single series degrades that component only; the function never
    raises -- on total failure it returns :data:`REGIME_NEUTRAL` with an
    ``as_of`` stamp.  ``thresholds`` overrides the gate thresholds (see
    :func:`compute_regime`).
    """
    raw: dict[str, Any] = {}
    for name in REGIME_SERIES:
        try:
            frame = fetch_ohlcv(name, period=period, cache=cache)
        except Exception:  # noqa: BLE001 - degrade per instrument
            frame = None
        raw[name] = frame

    snapshot = compute_regime(raw, thresholds=thresholds)
    snapshot["as_of"] = datetime.now(timezone.utc).isoformat()
    return snapshot
