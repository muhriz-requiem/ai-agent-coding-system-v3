"""Tests for the global liquidity & fear regime layer (src/data/regime.py).

All fixtures are hand-built pandas Series; no network, no yfinance.
"""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.data.regime",)
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

from src.data.regime import (NEUTRAL, PANIC, REGIME_NEUTRAL, REGIME_SERIES,
                             RISK_OFF, compute_regime, fetch_regime)


# --------------------------------------------------------------------------- #
# Fixture builders                                                             #
# --------------------------------------------------------------------------- #


def _ramp(level: float, drift: float, n: int = 40) -> pd.Series:
    """Series ending at ``level`` with daily drift ``drift`` (linear)."""
    start = level - drift * (n - 1)
    return pd.Series([start + drift * i for i in range(n)])


def _flat(level: float, n: int = 40) -> pd.Series:
    return pd.Series([level] * n)


def _wobble(level: float, amp: float, n: int = 60, tail: float | None = None) -> pd.Series:
    """Deterministic non-linear series around ``level`` with varying slopes.

    Consecutive increments differ (alternating, amplitude-modulated), so
    rolling slopes have non-zero variance -- required for slope z-scores.
    ``tail`` overrides the final observation (e.g. a spike).  n >= 42 keeps
    the 20d-MA/20d-shift slope series at >= 3 usable points.
    """
    values = [
        level + amp * ((-1) ** i) * (0.5 + 0.25 * (i % 3)) for i in range(n)
    ]
    if tail is not None:
        values[-1] = tail
    return pd.Series(values)


def _panic_inputs() -> dict:
    """Backwardated term structure + spot VIX above the panic level, plus
    corroborating fear (gold decoupling from equities)."""
    vix = _wobble(14.0, 0.4, tail=38.0)
    spy = _wobble(500.0, 2.0)
    gld = spy * 0.36
    gld.iloc[-10:] = gld.iloc[-10:] + [1.0 * (i + 1) for i in range(10)]
    return {
        "^VIX": vix,
        "^VIX3M": _wobble(17.0, 0.4),  # ratio 38/17 >= 1 -> backwardation
        "SPY": spy,
        "GLD": gld,
        # 13w bill 5.2% vs 10y 4.3%: inverted -> also a risk_off condition.
        "^IRX": _flat(52.0),
        "^TNX": _flat(43.0),
    }


def _calm_inputs() -> dict:
    """Contango, low VIX, constant-ratio pairs (ratio slope z-scores are
    undefined for constant ratios -> components excluded), non-inverted
    curve.  Nothing may gate."""
    vix = _wobble(13.0, 0.3, tail=13.0)
    spy = _wobble(500.0, 1.5)
    return {
        "^VIX": vix,
        "^VIX3M": vix * (17.0 / 13.0),  # ratio exactly 17/13 < 1
        "SPY": spy,
        "GLD": spy * 0.36,               # ratio exactly 0.36
        "JPY=X": _wobble(150.0, 0.3, tail=150.0),
        # ^IRX/^TNX are yield*10: 13w bill 4.0%, 10y 4.2% -> NOT inverted.
        "^IRX": _flat(40.0),
        "^TNX": _flat(42.0),
    }


# --------------------------------------------------------------------------- #
# Term structure & panic detection                                             #
# --------------------------------------------------------------------------- #


def test_vix_term_ratio_contango_vs_backwardation():
    calm = compute_regime(_calm_inputs())
    assert calm["vix_term"] == pytest.approx(13.0 / 17.0)
    assert calm["vix_term"] < 1.0  # contango

    panic = compute_regime(_panic_inputs())
    assert panic["vix_term"] >= 1.0  # backwardated


def test_panic_requires_backwardation_and_high_vix():
    snapshot = compute_regime(_panic_inputs())
    assert snapshot["regime"] == "panic"
    assert snapshot["vix"] >= 30.0


def test_backwardation_alone_is_not_panic():
    inputs = _panic_inputs()
    inputs["^VIX"] = _flat(20.0)  # backwardated but calm spot
    # Keep ratio >= 1: VIX3M below spot.
    inputs["^VIX3M"] = _flat(18.0)
    snapshot = compute_regime(inputs)
    assert snapshot["regime"] != "panic"


def test_high_vix_alone_is_not_panic():
    inputs = _calm_inputs()
    inputs["^VIX"] = _ramp(32.0, 0.3)   # high spot...
    inputs["^VIX3M"] = _ramp(38.0, 0.3)  # ...but still contango
    snapshot = compute_regime(inputs)
    assert snapshot["regime"] != "panic"


# --------------------------------------------------------------------------- #
# Composite thresholds                                                         #
# --------------------------------------------------------------------------- #


def test_risk_off_on_elevated_fear():
    inputs = _calm_inputs()
    inputs["^VIX"] = _wobble(13.0, 0.3, tail=40.0)      # fear spike
    inputs["^VIX3M"] = _wobble(45.0, 0.4)               # ratio ~0.9 < 1 -> not panic
    snapshot = compute_regime(inputs)
    assert snapshot["fear_z"] is not None and snapshot["fear_z"] > 1.0
    assert snapshot["regime"] == "risk_off"


def test_risk_off_on_impaired_liquidity():
    inputs = _calm_inputs()
    # USD/JPY breaking down = yen strength = carry unwind = liquidity drain
    # (NOT inverted); HYG/IEF rising = credit widening (inverted component).
    inputs["JPY=X"] = pd.Series(
        [150.0 + 0.3 * ((-1) ** i) for i in range(40)]
        + [148.0 - 2.0 * i for i in range(20)]
    )
    inputs["HYG"] = _wobble(80.0, 0.3, tail=85.0)       # HY stress vs flat IEF
    inputs["IEF"] = _wobble(95.0, 0.3)
    snapshot = compute_regime(inputs)
    assert snapshot["liquidity_z"] is not None and snapshot["liquidity_z"] < -1.0
    assert snapshot["regime"] == "risk_off"


def test_curve_inversion_forces_risk_off():
    inputs = _calm_inputs()
    inputs["^IRX"] = _flat(45.0)
    inputs["^TNX"] = _flat(42.0)  # 10y (4.2) below bill (4.5) -> inverted
    snapshot = compute_regime(inputs)
    assert snapshot["curve_inverted"] is True
    assert snapshot["regime"] == "risk_off"


def test_calm_inputs_are_neutral():
    snapshot = compute_regime(_calm_inputs())
    assert snapshot["regime"] == "neutral"
    assert snapshot["curve_inverted"] is False


def test_risk_on_maps_to_neutral_label():
    # Strong risk-on conditions never upscale: label stays neutral.  Only
    # corroborated composites score; USD/JPY carry strength alone cannot
    # gate (single component), and a calm VIX cannot either.
    inputs = _calm_inputs()
    inputs["^VIX"] = _wobble(13.0, 0.3, tail=8.0)        # very calm
    inputs["JPY=X"] = pd.Series(
        [150.0 + 0.3 * ((-1) ** i) for i in range(40)]
        + [148.0 + 2.0 * i for i in range(20)]           # dollar/carry strength
    )
    snapshot = compute_regime(inputs)
    # VIX level + VIX/VIX3M ratio both corroborate calm (negative); the
    # liquidity composite has a single component and stays None.
    assert snapshot["fear_z"] == -5.0
    assert snapshot["liquidity_z"] is None
    assert snapshot["regime"] == "neutral"


def test_component_z_scores_are_clamped():
    # A tail blip on a near-constant series must not manufacture a huge
    # score: components clamp to +/- Z_SCORE_CLAMP.
    inputs = _calm_inputs()
    inputs["^VIX"] = _wobble(13.0, 0.3, tail=8.0)  # z strongly negative raw
    snapshot = compute_regime(inputs)
    assert snapshot["components"]["vix"] == -5.0


def test_panic_overrides_risk_off():
    inputs = _panic_inputs()
    inputs["^IRX"] = _flat(55.0)
    inputs["^TNX"] = _flat(40.0)  # also inverted -> risk_off condition too
    snapshot = compute_regime(inputs)
    assert snapshot["regime"] == "panic"


# --------------------------------------------------------------------------- #
# Degradation                                                                  #
# --------------------------------------------------------------------------- #


def test_missing_all_series_is_neutral():
    snapshot = compute_regime({})
    assert snapshot == REGIME_NEUTRAL
    assert snapshot["regime"] == NEUTRAL


def test_none_series_degrade_gracefully():
    inputs = {name: None for name in REGIME_SERIES}
    snapshot = compute_regime(inputs)
    assert snapshot["regime"] == NEUTRAL
    assert snapshot["fear_z"] is None
    assert snapshot["liquidity_z"] is None


def test_partial_data_still_scores():
    # VIX + gold/equity decoupling: two corroborating fear components
    # gate risk_off even with every liquidity series missing.
    spy = _wobble(500.0, 2.0)
    gld = spy * 0.36
    gld.iloc[-10:] = gld.iloc[-10:] + [1.0 * (i + 1) for i in range(10)]
    inputs = {
        "^VIX": _wobble(13.0, 0.3, tail=40.0),
        "^VIX3M": _wobble(45.0, 0.4),  # ratio ~0.9 < 1 -> not panic
        "SPY": spy,
        "GLD": gld,
    }
    snapshot = compute_regime(inputs)
    assert snapshot["fear_z"] is not None
    assert snapshot["liquidity_z"] is None
    assert snapshot["regime"] == "risk_off"


def test_single_component_cannot_gate():
    # A lone spiking series is uncorroborated: the composite stays None
    # (fail-open) and no gate fires.
    inputs = {"^VIX": _wobble(13.0, 0.3, tail=40.0)}
    snapshot = compute_regime(inputs)
    assert snapshot["fear_z"] is None
    assert snapshot["regime"] == NEUTRAL


def test_non_dict_input_is_neutral():
    assert compute_regime(None)["regime"] == NEUTRAL
    assert compute_regime("junk")["regime"] == NEUTRAL


def test_dataframe_input_with_multiindex_close():
    # yfinance multi-ticker frames carry MultiIndex columns; the seam picks
    # the Close column's first level.
    index = pd.date_range("2026-01-01", periods=40, freq="D")
    frame = pd.DataFrame(
        {("Close", "^VIX"): [13.0 + 0.1 * i for i in range(40)]}, index=index
    )
    snapshot = compute_regime({"^VIX": frame})
    assert snapshot["vix"] == pytest.approx(13.0 + 0.1 * 39)


def test_components_block_reports_scores():
    snapshot = compute_regime(_calm_inputs())
    comps = snapshot["components"]
    assert comps["vix"] is not None            # wobble -> defined z-score
    assert comps["usdjpy"] is not None         # wobble slope -> defined
    assert comps["gold_eq"] is None            # constant ratio -> std 0 -> None
    assert comps["dxy"] is None                # series absent -> None


# --------------------------------------------------------------------------- #
# Fetch seam                                                                   #
# --------------------------------------------------------------------------- #


def test_fetch_regime_degrades_per_instrument(monkeypatch):
    from src.data import regime

    index = pd.date_range("2026-01-01", periods=40, freq="D")
    good = pd.DataFrame({"Close": [13.0 + 0.05 * i for i in range(40)]},
                        index=index)

    calls = []

    def fake_fetch(name, period="1y", cache=None):
        calls.append(name)
        if name == "^VIX":
            return good
        raise RuntimeError("network down")

    monkeypatch.setattr(regime, "fetch_ohlcv", fake_fetch)

    snapshot = regime.fetch_regime(cache=None)
    assert len(calls) == len(REGIME_SERIES)
    assert snapshot["vix"] == pytest.approx(13.0 + 0.05 * 39)
    assert snapshot["vix3m"] is None
    # One lone series is uncorroborated -> composite None -> never gates.
    assert snapshot["fear_z"] is None
    assert snapshot["regime"] == NEUTRAL
    assert snapshot["as_of"] is not None


def test_fetch_regime_total_failure_is_neutral(monkeypatch):
    from src.data import regime

    def boom(name, period="1y", cache=None):
        raise RuntimeError("network down")

    monkeypatch.setattr(regime, "fetch_ohlcv", boom)
    snapshot = regime.fetch_regime(cache=None)
    assert snapshot["regime"] == NEUTRAL
    assert snapshot["fear_z"] is None
    assert snapshot["as_of"] is not None


# --------------------------------------------------------------------------- #
# Threshold overrides (item 12a)                                               #
# --------------------------------------------------------------------------- #


def test_threshold_overrides_change_gates():
    # Panic inputs: VIX tail 38, VIX3M 17 (ratio ~2.24), inverted curve.
    # Raising the panic level declassifies panic -> falls back to risk_off
    # (the inverted curve still gates).
    snapshot = compute_regime(_panic_inputs(),
                              thresholds={"vix_panic_level": 40.0})
    assert snapshot["regime"] == RISK_OFF
    # Lowering the fear threshold cannot fire on calm inputs either way;
    # raising it de-gates an elevated-fear run.
    fear_inputs = _calm_inputs()
    fear_inputs["^VIX"] = _wobble(13.0, 0.3, tail=40.0)
    fear_inputs["^VIX3M"] = _wobble(45.0, 0.4)
    assert compute_regime(fear_inputs)["regime"] == RISK_OFF
    assert compute_regime(
        fear_inputs, thresholds={"fear_z_threshold": 5.0}
    )["regime"] == NEUTRAL
    # Raising the backwardation ratio above the actual ratio (38/17 ~ 2.24)
    # kills the panic classification (curve inversion still -> risk_off).
    snapshot = compute_regime(_panic_inputs(),
                              thresholds={"vix_backwardation": 2.5})
    assert snapshot["regime"] == RISK_OFF


def test_liquidity_floor_override_degates():
    inputs = _calm_inputs()
    inputs["JPY=X"] = pd.Series(
        [150.0 + 0.3 * ((-1) ** i) for i in range(40)]
        + [148.0 - 2.0 * i for i in range(20)]
    )
    inputs["HYG"] = _wobble(80.0, 0.3, tail=85.0)
    inputs["IEF"] = _wobble(95.0, 0.3)
    assert compute_regime(inputs)["regime"] == RISK_OFF
    snapshot = compute_regime(inputs, thresholds={"liquidity_z_floor": -5.0})
    assert snapshot["regime"] == NEUTRAL


def test_junk_thresholds_fail_open_to_defaults():
    # Default: panic fires.  Junk overrides (junk value, unknown key,
    # non-dict) must not change that.
    for junk in ({"vix_panic_level": "junk"}, {"unknown_key": 5.0}, None,
                 "junk", 42):
        snapshot = compute_regime(_panic_inputs(), thresholds=junk)
        assert snapshot["regime"] == PANIC, junk


def test_partial_override_keeps_other_defaults():
    # Only the fear threshold is overridden; the panic gate still uses
    # its default level.
    snapshot = compute_regime(_panic_inputs(),
                              thresholds={"fear_z_threshold": 99.0})
    assert snapshot["regime"] == PANIC


def test_fetch_regime_threads_thresholds(monkeypatch):
    from src.data import regime

    def fake_fetch(name, period="1y", cache=None):
        if name == "^VIX":
            return _wobble(14.0, 0.4, tail=38.0)
        if name == "^VIX3M":
            return _wobble(17.0, 0.4)
        raise RuntimeError("network down")

    monkeypatch.setattr(regime, "fetch_ohlcv", fake_fetch)
    # Default thresholds: backwardation + VIX 38 >= 30 -> panic (the
    # corroborated fear composite also gates risk_off, but panic wins).
    assert regime.fetch_regime(cache=None)["regime"] == PANIC
    # Overridden panic level: no panic; the fear composite still gates.
    snapshot = regime.fetch_regime(
        cache=None, thresholds={"vix_panic_level": 99.0}
    )
    assert snapshot["regime"] == RISK_OFF
