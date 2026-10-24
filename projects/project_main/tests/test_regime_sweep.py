"""Tests for the regime-threshold sweep (src/backtest/regime_sweep.py)."""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.backtest.regime_sweep", "src.data.regime")
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

from src.backtest.regime_sweep import (
    CANDIDATES,
    collect_regime_history,
    collect_series,
    sweep_thresholds,
)


def _wobble(level: float, amp: float, n: int = 60, tail: float | None = None) -> pd.Series:
    values = [level + amp * ((-1) ** i) * (0.5 + 0.25 * (i % 3)) for i in range(n)]
    if tail is not None:
        values[-1] = tail
    return pd.Series(values)


def _raw() -> dict:
    """Panic-flavoured series on a business-day index (no network)."""
    index = pd.bdate_range("2026-01-01", periods=60)
    vix = _wobble(14.0, 0.4, tail=38.0)
    vix.index = index
    vix3m = _wobble(17.0, 0.4)
    vix3m.index = index
    return {"^VIX": vix, "^VIX3M": vix3m}


# ---------------------------------------------------------------------------
# collect_series seam
# ---------------------------------------------------------------------------


def test_collect_series_uses_fetch_seam_and_degrades(monkeypatch):
    from src.backtest import regime_sweep as rs

    calls = []

    def fake_fetch(name, period="1y", cache=None):
        calls.append(name)
        if name == "^VIX":
            return _raw()["^VIX"]
        raise RuntimeError("network down")

    monkeypatch.setattr(rs, "fetch_ohlcv", fake_fetch)
    raw = collect_series(period="1y")
    assert calls == list(rs.REGIME_SERIES)
    assert raw["^VIX"] is not None
    assert all(raw[name] is None for name in rs.REGIME_SERIES if name != "^VIX")


# ---------------------------------------------------------------------------
# collect_regime_history
# ---------------------------------------------------------------------------


def test_history_includes_threshold_independent_quantities():
    history = collect_regime_history(_raw())
    assert history, "fixture must yield usable dates"
    first = history[0]
    for field in ("date", "vix", "vix_term", "fear_z", "liquidity_z",
                  "curve_inverted"):
        assert field in first
    # The tail spike must show up in the VIX levels.
    assert max(row["vix"] for row in history if row["vix"] is not None) >= 30.0


def test_history_skips_dates_without_any_data():
    assert collect_regime_history({}) == []
    assert collect_regime_history(None) == []
    assert collect_regime_history({"^VIX": None, "^VIX3M": None}) == []


def test_history_is_deterministic():
    assert collect_regime_history(_raw()) == collect_regime_history(_raw())


# ---------------------------------------------------------------------------
# sweep_thresholds
# ---------------------------------------------------------------------------


def test_sweep_reports_all_keys_and_candidates():
    report = sweep_thresholds(_raw())
    assert report["days_evaluated"] > 0
    assert set(report["keys"]) == set(CANDIDATES)
    for key, rows in report["keys"].items():
        assert [row["threshold"] for row in rows] == list(CANDIDATES[key])
        for row in rows:
            assert row["evaluated"] <= report["days_evaluated"]
            assert 0 <= row["fired"] <= row["evaluated"]
            rate = row["fire_rate"]
            if rate is not None:
                assert 0.0 <= rate <= 1.0


def _rates(report, key):
    return [row["fire_rate"] for row in report["keys"][key]
            if row["fire_rate"] is not None]


def test_sweep_fire_rates_are_monotonic():
    report = sweep_thresholds(_raw())
    # A higher (or more negative for floors) threshold can never fire
    # more often than an easier one.
    for key in ("vix_panic_level", "vix_backwardation", "fear_z_threshold"):
        rates = _rates(report, key)
        assert rates == sorted(rates, reverse=True), key
    rates = _rates(report, "liquidity_z_floor")
    assert rates == sorted(rates, reverse=True), "liquidity_z_floor"


def test_sweep_panic_level_discriminates_fixture():
    report = sweep_thresholds(_raw())
    rows = report["keys"]["vix_panic_level"]
    by_level = {row["threshold"]: row for row in rows}
    # The fixture spikes to VIX 38: level 30 must fire, 40 must not.
    assert by_level[30.0]["fired"] > 0
    assert by_level[40.0]["fired"] == 0


def test_sweep_junk_input_yields_empty_report():
    for junk in (None, {}, {"^VIX": None}, "junk"):
        report = sweep_thresholds(junk)
        assert report["days_evaluated"] == 0
        for rows in report["keys"].values():
            assert all(row["evaluated"] == 0 and row["fire_rate"] is None
                       for row in rows)


def test_sweep_baseline_override_affects_companion_gate():
    # Raising the backwardation requirement above the fixture's ratio
    # (~2.24) makes even level-20 panic candidates unable to fire.
    report = sweep_thresholds(_raw(), baseline={"vix_backwardation": 2.5})
    rows = report["keys"]["vix_panic_level"]
    assert all(row["fired"] == 0 for row in rows)
    # With the default baseline, easy panic levels do fire.
    report_default = sweep_thresholds(_raw())
    by_level = {row["threshold"]: row
                for row in report_default["keys"]["vix_panic_level"]}
    assert by_level[20.0]["fired"] > 0