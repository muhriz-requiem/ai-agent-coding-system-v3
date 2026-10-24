"""Return-correlation clustering for concentration control (item 11).

Groups same-day BUY candidates whose daily returns are highly correlated
into clusters, so the order plan can cap the aggregate notional of a
correlated basket exactly like a sector (a pseudo-sector cap).

Design notes:
  * Inputs are per-ticker close-price Series (whatever the pipeline
    already fetched -- typically 6mo daily closes); returns are computed
    internally on the aligned intersection of indexes.
  * Clustering is single-linkage union-find over pairs with Pearson
    correlation >= ``threshold``: A~B and B~C merges A,B,C into one
    cluster even when corr(A, C) is below the threshold.  Chains are
    deliberate -- concentration risk transitive through a shared driver.
  * Only POSITIVE correlation clusters.  Strongly negative pairs hedge
    each other and must not consume a concentration budget.
  * Fail-open everywhere: junk inputs, insufficient overlap (fewer than
    ``min_observations`` aligned returns) or fewer than two usable
    tickers yield no clusters, and the pipeline behaves exactly as
    before the feature existed.

Pure computation only: no I/O, no network, inputs are never mutated.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

__all__ = ["correlation_clusters", "returns_frame"]

#: Prefix used for cluster ids appearing in ``cluster_map`` values so
#: rationales/audit entries can distinguish them from sector labels.
CLUSTER_ID_PREFIX = "corr:"


def returns_frame(closes: dict[str, Any]) -> pd.DataFrame:
    """Build the aligned daily-returns frame from per-ticker closes.

    Values may be ``pd.Series``, OHLCV ``pd.DataFrame`` (the ``Close``
    column is used), or ``None``/junk -- invalid entries are dropped.
    Returns are simple pct-change on the aligned intersection; rows with
    any missing value are dropped so every pairwise correlation uses the
    same observation set.
    """
    series_map: dict[str, pd.Series] = {}
    if not isinstance(closes, dict):
        return pd.DataFrame()
    for ticker, frame in closes.items():
        if not isinstance(ticker, str) or not ticker.strip():
            continue
        series: Any = None
        if isinstance(frame, pd.Series):
            series = frame
        elif isinstance(frame, pd.DataFrame):
            if "Close" in frame.columns:
                series = frame["Close"]
            elif len(frame.columns):
                series = frame.iloc[:, 0]
            if isinstance(series, pd.DataFrame):
                if series.empty:
                    continue
                series = series.iloc[:, 0]
        if not isinstance(series, pd.Series) or series.empty:
            continue
        try:
            series = pd.to_numeric(series, errors="coerce").dropna()
        except Exception:  # noqa: BLE001 - degrade per ticker
            continue
        if len(series) < 3:
            continue
        series_map[ticker] = series.astype(float)
    if len(series_map) < 2:
        return pd.DataFrame()
    aligned = pd.DataFrame(series_map).dropna()
    if aligned.empty:
        return pd.DataFrame()
    returns = aligned.pct_change().dropna()
    return returns


def correlation_clusters(
    closes: dict[str, Any],
    threshold: float = 0.80,
    min_observations: int = 60,
) -> dict[str, str]:
    """Map ticker -> cluster id for correlated groups (item 11).

    Args:
        closes: Mapping ticker -> close-price Series / OHLCV frame.
        threshold: Pearson correlation at or above which two tickers
            link (positive correlations only; clamped to [0, 1]).
        min_observations: Minimum aligned daily-return observations
            required before any clustering is attempted (fail-open below
            this; the default suits the pipeline's 6mo fetch window).

    Returns:
        Mapping ticker -> cluster id (``"corr:A, B"``) for every member
        of a cluster with two or more tickers.  Uncorrelated tickers are
        absent from the mapping (exempt from any cluster cap).  With
        insufficient data the mapping is empty.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return {}
    if not math.isfinite(threshold):
        return {}
    threshold = max(0.0, min(1.0, threshold))
    try:
        min_obs = int(min_observations)
    except (TypeError, ValueError):
        return {}
    if min_obs < 2:
        min_obs = 2

    returns = returns_frame(closes)
    if returns.shape[0] < min_obs or returns.shape[1] < 2:
        return {}

    corr = returns.corr()
    tickers = list(corr.columns)

    # Union-find over linked pairs (single-linkage, positive corr only).
    parent = {t: t for t in tickers}

    def find(t: str) -> str:
        while parent[t] != t:
            parent[t] = parent[parent[t]]
            t = parent[t]
        return t

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic merge direction: lexicographically smaller
            # root always wins, so cluster ids are stable across runs.
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, left in enumerate(tickers):
        for right in tickers[i + 1:]:
            value = corr.at[left, right]
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= threshold:
                union(left, right)

    groups: dict[str, list[str]] = {}
    for ticker in tickers:
        groups.setdefault(find(ticker), []).append(ticker)

    mapping: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        cluster_id = CLUSTER_ID_PREFIX + ", ".join(members)
        for ticker in members:
            mapping[ticker] = cluster_id
    return mapping
