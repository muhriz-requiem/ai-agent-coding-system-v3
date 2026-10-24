"""Pair discovery ranking + dedupe (walk-forward Sharpe).

Ranks cointegrated candidates by an out-of-sample (walk-forward)
simulation of the long-only conditioned mean-reversion trade, then
greedily dedupes so no ticker appears in more than one pair and no
sector exceeds ``max_per_sector`` pairs.

Walk-forward design (deterministic, no lookahead):
  * the formation window is split in halves; spread parameters are
    estimated on the FIRST half only,
  * the long-only strategy (buy the cheap leg at |z| >= z_entry, exit
    at |z| < z_exit or |z| > z_stop) trades the SECOND half using those
    frozen parameters,
  * ``sharpe`` is the annualized Sharpe of the resulting daily strategy
    returns (0.0 when the half never trades),
  * ``stability_score`` measures hedge-ratio agreement across the two
    halves (1.0 = identical betas),
  * ``liquidity_score`` is the raw dollar ADV of the thinner leg.

Ranking is a tuple sort (sharpe, stability, liquidity, lower p-value,
pair id) so results are fully deterministic.  Pure computation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.pairs.cointegration import (
    MIN_OBSERVATIONS,
    PairCandidate,
    _close_series,
    find_cointegrated_pairs,
)

__all__ = [
    "RankedPair",
    "rank_pairs",
    "dedupe_pairs",
    "form_pairs",
    "adv_map",
]

#: Sentinel for pairs whose sector label is absent (dedupe-exempt).
_UNKNOWN_SECTOR = None


@dataclass
class RankedPair:
    """A candidate with its ranking statistics attached."""

    candidate: PairCandidate
    sharpe: float
    stability_score: float
    liquidity_score: float


def _aligned_halves(
    close_a: pd.Series,
    close_b: pd.Series,
    lookback_days: int,
) -> Optional[tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    """Aligned price series trimmed to ``lookback_days`` and split in halves."""
    frame = pd.concat([close_a.rename("a"), close_b.rename("b")], axis=1).dropna()
    if lookback_days and lookback_days > 0:
        frame = frame.tail(int(lookback_days))
    if len(frame) < 20:
        return None
    mid = len(frame) // 2
    if mid < 10:
        return None
    return (
        frame["a"].iloc[:mid],
        frame["b"].iloc[:mid],
        frame["a"].iloc[mid:],
        frame["b"].iloc[mid:],
    )


def _fit_spread(close_a: pd.Series, close_b: pd.Series):
    """OLS log-price fit (beta, alpha) on one window; ``None`` degenerate."""
    log_a = np.log(close_a.to_numpy(dtype=float))
    log_b = np.log(close_b.to_numpy(dtype=float))
    if not (np.all(np.isfinite(log_a)) and np.all(np.isfinite(log_b))):
        return None
    design = np.column_stack([log_b, np.ones_like(log_b)])
    try:
        coef, *_ = np.linalg.lstsq(design, log_a, rcond=None)
    except Exception:  # noqa: BLE001 - degenerate window -> skip
        return None
    beta, alpha = float(coef[0]), float(coef[1])
    if not math.isfinite(beta) or not math.isfinite(alpha) or beta <= 0.0:
        return None
    return beta, alpha


def _walk_forward_sharpe(
    candidate: PairCandidate,
    close_a: pd.Series,
    close_b: pd.Series,
    lookback_days: int,
    z_entry: float,
    z_exit: float,
    z_stop: float,
) -> tuple[float, float]:
    """OOS long-only Sharpe + hedge-ratio stability for one candidate."""
    halves = _aligned_halves(close_a, close_b, lookback_days)
    if halves is None:
        return 0.0, 0.0
    a1, b1, a2, b2 = halves
    fit1 = _fit_spread(a1, b1)
    fit2 = _fit_spread(a2, b2)
    if fit1 is None or fit2 is None:
        return 0.0, 0.0
    beta1, alpha1 = fit1
    beta2, alpha2 = fit2

    # Stability: hedge-ratio agreement across halves, in [0, 1].
    denom = max(abs(beta1), abs(beta2), 1e-9)
    stability = 1.0 - min(1.0, abs(beta1 - beta2) / denom)

    # OOS spread z-score on the second half with frozen first-half stats.
    log_a2 = np.log(a2.to_numpy(dtype=float))
    log_b2 = np.log(b2.to_numpy(dtype=float))
    spread = log_a2 - alpha1 - beta1 * log_b2
    std = float(np.std(spread))
    if not math.isfinite(std) or std <= 0.0:
        return 0.0, stability
    z = (spread - float(np.mean(spread))) / std

    # Long-only conditioned mean reversion on the second half.
    position = 0
    position_series = np.zeros(len(z))
    for t in range(len(z)):
        if position == 0:
            if z[t] >= z_entry:
                position = +1  # B cheap -> long B
            elif z[t] <= -z_entry:
                position = -1  # A cheap -> long A
        elif abs(z[t]) < z_exit or abs(z[t]) > z_stop:
            position = 0
        position_series[t] = position

    prices_a = a2.to_numpy(dtype=float)
    prices_b = b2.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_a = prices_a[1:] / prices_a[:-1] - 1.0
        ret_b = prices_b[1:] / prices_b[:-1] - 1.0
    # Daily strategy return: position held over day t->t+1 earns the
    # leg's return.  Position at t applies to return t (conservative:
    # entry at close t, first P&L day t->t+1).
    pos = position_series[:-1]
    strat = np.where(pos > 0, ret_b, np.where(pos < 0, ret_a, 0.0))
    strat = np.nan_to_num(strat, nan=0.0, posinf=0.0, neginf=0.0)
    if len(strat) < 2 or not np.any(position_series):
        return 0.0, stability
    mean = float(np.mean(strat))
    var = float(np.var(strat))
    if var <= 0.0:
        return 0.0, stability
    sharpe = mean / math.sqrt(var) * math.sqrt(252.0)
    if not math.isfinite(sharpe):
        sharpe = 0.0
    return sharpe, stability


def adv_map(
    closes: dict[str, Any],
    lookback_days: int = 252,
) -> dict[str, float]:
    """Dollar ADV (mean Volume * Close) per ticker over ``lookback_days``.

    Tickers without a usable Volume column are absent from the mapping
    (the liquidity gate fails open on them).  Accepts OHLCV frames or
    close Series (Series -> absent: no volume data).
    """
    result: dict[str, float] = {}
    for ticker, frame in (closes or {}).items():
        name = str(ticker or "").strip().upper()
        if not name or not isinstance(frame, pd.DataFrame):
            continue
        volume = frame.get("Volume") if hasattr(frame, "get") else None
        close = _close_series(frame)
        if volume is None or close is None:
            continue
        if isinstance(volume, pd.DataFrame):
            if volume.empty:
                continue
            volume = volume.iloc[:, 0]
        try:
            volume = pd.to_numeric(volume, errors="coerce").astype(float)
        except Exception:  # noqa: BLE001 - degrade per ticker
            continue
        aligned = pd.concat([close.rename("c"), volume.rename("v")], axis=1).dropna()
        if lookback_days and lookback_days > 0:
            aligned = aligned.tail(int(lookback_days))
        if aligned.empty:
            continue
        adv = float((aligned["c"] * aligned["v"]).mean())
        if math.isfinite(adv) and adv > 0.0:
            result[name] = adv
    return result


def rank_pairs(
    candidates: list[PairCandidate],
    closes: dict[str, Any],
    lookback_days: int = 252,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    min_observations: int = MIN_OBSERVATIONS,
) -> list[RankedPair]:
    """Rank candidates by walk-forward OOS Sharpe (deterministic order).

    ``liquidity_score`` is the thinner leg's dollar ADV (0.0 when no
    volume data exists -- ranking still works offline).  The output is
    sorted best-first by ``(sharpe, stability, liquidity, -pvalue,
    pair_id)``.
    """
    advs = adv_map(closes, lookback_days=lookback_days)
    ranked: list[RankedPair] = []
    for candidate in candidates or []:
        if candidate is None:
            continue
        close_a = _close_series(closes.get(candidate.ticker_a))
        close_b = _close_series(closes.get(candidate.ticker_b))
        if close_a is None or close_b is None:
            continue
        sharpe, stability = _walk_forward_sharpe(
            candidate, close_a, close_b, lookback_days, z_entry, z_exit, z_stop
        )
        leg_adv = min(
            advs.get(candidate.ticker_a, 0.0),
            advs.get(candidate.ticker_b, 0.0),
        )
        ranked.append(
            RankedPair(
                candidate=candidate,
                sharpe=sharpe,
                stability_score=stability,
                liquidity_score=leg_adv,
            )
        )
    ranked.sort(
        key=lambda r: (
            -r.sharpe,
            -r.stability_score,
            -r.liquidity_score,
            r.candidate.adf_pvalue,
            r.candidate.pair_id,
        )
    )
    return ranked


def dedupe_pairs(
    ranked: list[RankedPair],
    max_pairs: int = 10,
    max_per_sector: int = 3,
) -> list[RankedPair]:
    """Greedy dedupe: no reused ticker, sector-concentration bound.

    Sector labels come from the candidate (pairs are same-sector after
    the pre-screen); pairs with unknown sectors are exempt from the
    per-sector bound (fail-open) but still consume a ``max_pairs`` slot.
    """
    kept: list[RankedPair] = []
    used: set[str] = set()
    sector_counts: dict[str, int] = {}
    try:
        max_pairs = int(max_pairs)
    except (TypeError, ValueError):
        max_pairs = 10
    try:
        max_per_sector = int(max_per_sector)
    except (TypeError, ValueError):
        max_per_sector = 3
    for item in ranked or []:
        if item is None or len(kept) >= max(0, max_pairs):
            break
        candidate = item.candidate
        if candidate.ticker_a in used or candidate.ticker_b in used:
            continue
        sector = candidate.sector_a or _UNKNOWN_SECTOR
        if sector is not None:
            count = sector_counts.get(sector, 0)
            if count >= max(0, max_per_sector):
                continue
            sector_counts[sector] = count + 1
        used.add(candidate.ticker_a)
        used.add(candidate.ticker_b)
        kept.append(item)
    return kept


def form_pairs(
    closes: dict[str, Any],
    sectors: Optional[dict[str, Optional[str]]] = None,
    lookback_days: int = 252,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
    max_pvalue: float = 0.05,
    max_lag: int = 1,
    prescreen_corr: float = 0.70,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float = 3.5,
    max_pairs: int = 10,
    max_per_sector: int = 3,
    min_sharpe: Optional[float] = None,
) -> list[RankedPair]:
    """Full discovery pipeline: cointegration -> rank -> dedupe -> gate.

    ``min_sharpe`` (the ``MIN_PAIR_SHARPE`` config) filters the final
    list; ``None`` disables the gate (used by tests / diagnostics).
    """
    candidates = find_cointegrated_pairs(
        closes,
        sectors=sectors,
        min_half_life=min_half_life,
        max_half_life=max_half_life,
        max_pvalue=max_pvalue,
        max_lag=max_lag,
        prescreen_corr=prescreen_corr,
        lookback_days=lookback_days,
    )
    ranked = rank_pairs(
        candidates,
        closes,
        lookback_days=lookback_days,
        z_entry=z_entry,
        z_exit=z_exit,
        z_stop=z_stop,
    )
    kept = dedupe_pairs(ranked, max_pairs=max_pairs, max_per_sector=max_per_sector)
    if min_sharpe is not None:
        try:
            floor = float(min_sharpe)
            kept = [item for item in kept if item.sharpe >= floor]
        except (TypeError, ValueError):
            pass
    return kept
