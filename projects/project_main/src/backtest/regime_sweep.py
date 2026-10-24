"""Regime-threshold sweep (item 12b): gate fire rates over history.

Answers the tuning question "what would each threshold have done?" by
replaying the regime composites over historical data and reporting how
often each candidate threshold would have fired its gate.

Method:
  1. Fetch every :data:`src.data.regime.REGIME_SERIES` series once.
  2. For each historical date, evaluate :func:`compute_regime` on the
     series truncated to that date -- the composite z-scores, VIX level,
     term-structure ratio and curve flag are threshold-independent, so
     they are computed exactly once per date.
  3. For each tunable key, evaluate candidate thresholds against those
     quantities and report the fire rate over the days where the
     quantity was usable.

Pure analysis: no LLM, no broker, deterministic given the data.  Higher
thresholds must never fire more often than lower ones (monotonicity is
asserted in tests).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.data.regime import REGIME_SERIES, compute_regime, fetch_ohlcv

__all__ = ["collect_series", "collect_regime_history", "sweep_thresholds"]

#: Candidate thresholds per tunable key (item 12).  The first entry of
#: each list mirrors the v1 default.
CANDIDATES: dict[str, tuple[float, ...]] = {
    "vix_backwardation": (0.90, 1.00, 1.10, 1.20),
    "vix_panic_level": (20.0, 25.0, 30.0, 35.0, 40.0),
    "fear_z_threshold": (0.5, 1.0, 1.5, 2.0),
    "liquidity_z_floor": (-0.5, -1.0, -1.5, -2.0),
}


def collect_series(
    period: str = "1y",
    cache: Any = None,
) -> dict[str, Any]:
    """Fetch every regime series through the module fetch seam.

    Same degradation contract as ``regime.fetch_regime``: a failure in
    any single series degrades that component only.
    """
    raw: dict[str, Any] = {}
    for name in REGIME_SERIES:
        try:
            raw[name] = fetch_ohlcv(name, period=period, cache=cache)
        except Exception:  # noqa: BLE001 - degrade per instrument
            raw[name] = None
    return raw


def _truncated(series: Any, date: Any) -> Any:
    if series is None:
        return None
    try:
        return series[series.index <= date]
    except Exception:  # noqa: BLE001 - degrade per series
        return None


def collect_regime_history(
    raw: dict[str, Any],
) -> list[dict]:
    """Evaluate threshold-independent regime quantities per history date.

    Returns one dict per date with ``date``, ``vix``, ``vix_term``,
    ``fear_z``, ``liquidity_z`` and ``curve_inverted``.  Dates before
    any usable data are skipped; per-quantity gaps stay ``None`` and are
    excluded from that quantity's evaluated count later.
    """
    if not isinstance(raw, dict):
        return []
    calendar = sorted({
        date
        for frame in raw.values()
        if isinstance(frame, (pd.Series, pd.DataFrame)) and not frame.empty
        for date in frame.index
    })
    history: list[dict] = []
    for date in calendar:
        truncated = {name: _truncated(frame, date) for name, frame in raw.items()}
        # Cheap pre-filter: once the VIX series alone covers a date the
        # panic quantities are evaluable; composites need their own
        # history, which compute_regime's fail-open handles per slice.
        snapshot = compute_regime(truncated)
        if (snapshot.get("vix") is None
                and snapshot.get("fear_z") is None
                and snapshot.get("liquidity_z") is None):
            continue
        history.append({
            "date": date,
            "vix": snapshot.get("vix"),
            "vix_term": snapshot.get("vix_term"),
            "fear_z": snapshot.get("fear_z"),
            "liquidity_z": snapshot.get("liquidity_z"),
            "curve_inverted": snapshot.get("curve_inverted"),
        })
    return history


def _fire_rate(history: list[dict], key: str, predicate) -> dict:
    values = [row for row in history if row.get(key) is not None]
    evaluated = len(values)
    fired = sum(1 for row in values if predicate(row))
    rate = (fired / evaluated) if evaluated else None
    return {"evaluated": evaluated, "fired": fired, "fire_rate": rate}


def sweep_thresholds(
    raw: dict[str, Any],
    baseline: Optional[dict[str, float]] = None,
) -> dict:
    """Sweep candidate thresholds over the collected history.

    ``baseline`` (defaults to ``regime.DEFAULT_THRESHOLDS``) supplies the
    companion gate values where a candidate interacts with another key
    (e.g. the panic level needs the backwardation ratio to clear
    ``vix_backwardation`` first).  Returns a JSON-safe report mapping
    each tunable key to its candidates' fire-rate rows.
    """
    from src.data.regime import DEFAULT_THRESHOLDS, _resolve_thresholds

    base = _resolve_thresholds(baseline)
    history = collect_regime_history(raw)

    report: dict[str, Any] = {
        "days_evaluated": len(history),
        "keys": {},
    }

    for key, candidates in CANDIDATES.items():
        rows = []
        for candidate in candidates:
            if key == "vix_panic_level":
                pred = (
                    lambda row, c=candidate: (
                        row.get("vix_term") is not None
                        and row["vix_term"] >= base["vix_backwardation"]
                        and row.get("vix") is not None
                        and row["vix"] >= c
                    )
                )
                row = _fire_rate(history, "vix", pred)
            elif key == "vix_backwardation":
                pred = (
                    lambda row, c=candidate: (
                        row.get("vix_term") is not None
                        and row["vix_term"] >= c
                    )
                )
                row = _fire_rate(history, "vix_term", pred)
            elif key == "fear_z_threshold":
                pred = lambda row, c=candidate: row["fear_z"] > c  # noqa: E731
                row = _fire_rate(history, "fear_z", pred)
            else:  # liquidity_z_floor
                pred = lambda row, c=candidate: row["liquidity_z"] < c  # noqa: E731
                row = _fire_rate(history, "liquidity_z", pred)
            row["threshold"] = candidate
            rows.append(row)
        report["keys"][key] = rows
    return report
