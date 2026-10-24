"""Engle-Granger cointegration engine (log prices, ADF, half-life).

Discovers cointegrated pairs from a mid-cap universe.  Per the
integration plan:

  * Log-price regression with intercept (standard Engle-Granger):
    ``log(P_A) = alpha + beta * log(P_B) + eps``; ``hedge_ratio = beta``.
  * ADF test on the OLS residuals (statsmodels, hard dependency).
  * Half-life from the AR(1) slope ``phi`` on the residuals:
    ``half_life = -log(2) / log(phi)``, valid only for ``0 < phi < 1``
    (``phi >= 1`` is non-stationary -> reject; ``phi <= 0`` oscillates
    -> reject).
  * Cheap pre-screen before the expensive ADF: same-sector pairs (when
    sector labels are supplied) AND |Pearson corr| >= ``prescreen_corr``
    on aligned daily returns.  With ~400 S&P 400 names this cuts the
    ~79,800 brute-force combinations to a few thousand ADF tests --
    both a runtime control and a multiplicity guard.

Every reject path returns ``None`` / ``[]`` (fail-open); the function
never raises on data problems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

__all__ = [
    "PairCandidate",
    "test_cointegration",
    "find_cointegrated_pairs",
    "spread_residuals",
]

#: Minimum aligned daily observations before any pair is tested.
MIN_OBSERVATIONS = 60


@dataclass
class PairCandidate:
    """A cointegrated pair with its formation-window statistics.

    ``hedge_ratio``/``alpha`` parameterize the log-price spread
    ``log(P_A) - alpha - beta * log(P_B)``; ``spread_mean``/``spread_std``
    are the residual mean/std over the formation window (the z-score
    denominator); ``corr`` is the formation-window daily-return Pearson
    correlation (the risk gate compares the live correlation against
    it).  ``sector_a``/``sector_b`` ride along for the dedupe step.
    """

    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    half_life: float
    adf_pvalue: float
    lookback_days: int
    alpha: float = 0.0
    spread_mean: float = 0.0
    spread_std: float = 0.0
    corr: float = 0.0
    sector_a: Optional[str] = None
    sector_b: Optional[str] = None

    @property
    def pair_id(self) -> str:
        """Stable pair identifier ``A/B`` (tickers sorted at creation)."""
        return f"{self.ticker_a}/{self.ticker_b}"


def _close_series(frame: Any) -> Optional[pd.Series]:
    """Extract a numeric close-price Series (OHLCV frame or passthrough)."""
    if frame is None:
        return None
    series: Any = frame
    if isinstance(frame, pd.DataFrame):
        if "Close" in frame.columns:
            series = frame["Close"]
        elif len(frame.columns):
            series = frame.iloc[:, 0]
        else:
            return None
        if isinstance(series, pd.DataFrame):
            if series.empty:
                return None
            series = series.iloc[:, 0]
    if not isinstance(series, pd.Series) or series.empty:
        return None
    try:
        series = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    except Exception:  # noqa: BLE001 - degrade per ticker
        return None
    if series.empty:
        return None
    return series


def _ar1_slope(residuals: np.ndarray) -> float:
    """AR(1) slope ``phi`` of ``residuals`` (OLS with intercept)."""
    if len(residuals) < 3:
        return float("nan")
    lagged = residuals[:-1]
    nxt = residuals[1:]
    design = np.column_stack([lagged, np.ones_like(lagged)])
    try:
        coef, *_ = np.linalg.lstsq(design, nxt, rcond=None)
    except Exception:  # noqa: BLE001 - degenerate design -> reject
        return float("nan")
    return float(coef[0])


def _formation_corr(series_a: pd.Series, series_b: pd.Series) -> float:
    """Daily-return Pearson correlation on the aligned intersection."""
    frame = pd.concat([series_a.rename("a"), series_b.rename("b")], axis=1).dropna()
    if len(frame) < 3:
        return float("nan")
    returns = frame.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    corr = float(returns["a"].corr(returns["b"]))
    return corr


def test_cointegration(
    series_a: Any,
    series_b: Any,
    max_lag: int = 1,
    max_pvalue: float = 0.05,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
    min_observations: int = MIN_OBSERVATIONS,
    lookback_days: int = 252,
    sector_a: Optional[str] = None,
    sector_b: Optional[str] = None,
) -> Optional[PairCandidate]:
    """Test one ticker pair for cointegration; ``None`` when rejected.

    Args:
        series_a: Close-price Series / OHLCV frame for the first leg.
        series_b: Close-price Series / OHLCV frame for the second leg.
        max_lag: Fixed ADF lag order (deterministic; no autolag).
        max_pvalue: ADF p-value at or below which the pair passes.
        min_half_life / max_half_life: Acceptable half-life window.
        min_observations: Minimum aligned closes before testing.
        lookback_days: Formation-window label recorded on the candidate.
        sector_a / sector_b: Optional sector labels riding along.

    Returns:
        A :class:`PairCandidate` when the pair is cointegrated with a
        usable half-life, else ``None``.
    """
    clean_a = _close_series(series_a)
    clean_b = _close_series(series_b)
    if clean_a is None or clean_b is None:
        return None
    aligned = pd.concat(
        [clean_a.rename("a"), clean_b.rename("b")], axis=1
    ).dropna()
    try:
        min_obs = int(min_observations)
    except (TypeError, ValueError):
        min_obs = MIN_OBSERVATIONS
    if len(aligned) < max(3, min_obs):
        return None

    # Log prices (zero/negative prices yield non-finite logs -> reject).
    with np.errstate(divide="ignore", invalid="ignore"):
        log_a = np.log(aligned["a"].to_numpy(dtype=float))
        log_b = np.log(aligned["b"].to_numpy(dtype=float))
    if not (np.all(np.isfinite(log_a)) and np.all(np.isfinite(log_b))):
        return None

    # OLS with intercept: log(P_A) = alpha + beta * log(P_B) + eps.
    design = np.column_stack([log_b, np.ones_like(log_b)])
    try:
        coef, *_ = np.linalg.lstsq(design, log_a, rcond=None)
    except Exception:  # noqa: BLE001 - degenerate design -> reject
        return None
    beta, alpha = float(coef[0]), float(coef[1])
    if not (math.isfinite(beta) and math.isfinite(alpha)) or beta <= 0.0:
        return None
    residuals = log_a - alpha - beta * log_b

    # ADF on the residuals (fixed lag, no trend: the intercept already
    # demeans the spread).  Any statsmodels failure rejects the pair.
    try:
        import warnings

        with warnings.catch_warnings():
            # statsmodels >= 0.15 warns about its future result-type
            # switch; the tuple contract is pinned here deliberately.
            warnings.simplefilter("ignore", FutureWarning)
            adf_result = adfuller(residuals, maxlag=max(0, int(max_lag)),
                                  regression="n", autolag=None)
        pvalue = float(adf_result[1])
    except Exception:  # noqa: BLE001 - ADF failure -> reject
        return None
    if not math.isfinite(pvalue) or pvalue > float(max_pvalue):
        return None

    # Half-life from the AR(1) slope of the residuals.
    phi = _ar1_slope(residuals)
    if not math.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
        return None
    half_life = -math.log(2.0) / math.log(phi)
    if not math.isfinite(half_life):
        return None
    if half_life < float(min_half_life) or half_life > float(max_half_life):
        return None

    corr = _formation_corr(aligned["a"], aligned["b"])
    if not math.isfinite(corr):
        return None

    return PairCandidate(
        ticker_a=str(series_a.name) if getattr(series_a, "name", None) else "",
        ticker_b=str(series_b.name) if getattr(series_b, "name", None) else "",
        hedge_ratio=beta,
        half_life=half_life,
        adf_pvalue=pvalue,
        lookback_days=int(lookback_days) if lookback_days else 252,
        alpha=alpha,
        spread_mean=float(np.mean(residuals)),
        spread_std=float(np.std(residuals)),
        corr=corr,
        sector_a=sector_a,
        sector_b=sector_b,
    )


def spread_residuals(
    candidate: PairCandidate,
    series_a: Any,
    series_b: Any,
    min_observations: int = MIN_OBSERVATIONS,
) -> Optional[pd.Series]:
    """Residual spread series ``log(P_A) - alpha - beta*log(P_B)``.

    Recomputes the spread (without re-fitting) from a candidate's
    persisted parameters over the aligned intersection of the two price
    series -- used for the live half-life/correlation drift gates and
    for the formation ``spread_daily_vol`` statistic.  ``None`` when
    either series is unusable or the aligned window is too short.
    """
    clean_a = _close_series(series_a)
    clean_b = _close_series(series_b)
    if clean_a is None or clean_b is None:
        return None
    aligned = pd.concat(
        [clean_a.rename("a"), clean_b.rename("b")], axis=1
    ).dropna()
    try:
        min_obs = int(min_observations)
    except (TypeError, ValueError):
        min_obs = MIN_OBSERVATIONS
    if len(aligned) < max(3, min_obs // 2):
        return None
    log_a = np.log(aligned["a"].to_numpy(dtype=float))
    log_b = np.log(aligned["b"].to_numpy(dtype=float))
    if not (np.all(np.isfinite(log_a)) and np.all(np.isfinite(log_b))):
        return None
    residuals = log_a - candidate.alpha - candidate.hedge_ratio * log_b
    return pd.Series(residuals, index=aligned.index)


def _sector_gate(
    sectors: Optional[dict[str, Optional[str]]],
    ticker_a: str,
    ticker_b: str,
) -> bool:
    """Same-sector pre-screen.

    When a ``sectors`` mapping is supplied, both legs must carry the
    same non-empty sector label (unknown labels -> skip: the universe
    scrape is the sector source of truth).  ``sectors=None`` disables
    the gate entirely (offline/test runs).
    """
    if sectors is None:
        return True
    sector_a = sectors.get(ticker_a)
    sector_b = sectors.get(ticker_b)
    if not isinstance(sector_a, str) or not sector_a.strip():
        return False
    if not isinstance(sector_b, str) or not sector_b.strip():
        return False
    return sector_a == sector_b


def find_cointegrated_pairs(
    closes: dict[str, Any],
    sectors: Optional[dict[str, Optional[str]]] = None,
    min_half_life: float = 5.0,
    max_half_life: float = 60.0,
    max_pvalue: float = 0.05,
    max_lag: int = 1,
    prescreen_corr: float = 0.70,
    min_observations: int = MIN_OBSERVATIONS,
    lookback_days: int = 252,
) -> list[PairCandidate]:
    """Discover cointegrated pairs across ``closes`` (deterministic).

    Pipeline per unordered pair (tickers iterated in sorted order):
    1. sector gate (see :func:`_sector_gate`),
    2. |Pearson corr| >= ``prescreen_corr`` on aligned daily log returns,
    3. full Engle-Granger test (:func:`test_cointegration`).

    Returns candidates sorted by ``(adf_pvalue, pair_id)``; an empty or
    insufficient universe yields ``[]``.  With fewer than
    ``min_observations`` aligned rows no pair is tested at all (the
    correlation screen cannot be evaluated, so discovery fails closed
    to "no pairs" rather than gambling on short windows).
    """
    series_map: dict[str, pd.Series] = {}
    for ticker, frame in (closes or {}).items():
        name = str(ticker or "").strip().upper()
        if not name:
            continue
        series = _close_series(frame)
        if series is None:
            continue
        series_map[name] = series
    if len(series_map) < 2:
        return []

    prices = pd.DataFrame(series_map)
    try:
        min_obs = int(min_observations)
    except (TypeError, ValueError):
        min_obs = MIN_OBSERVATIONS
    log_returns = np.log(prices).diff()
    # Pairwise-complete correlations (pandas .corr() default), so a
    # single gappy ticker does not shrink every pair's window.
    corr_matrix = log_returns.corr(min_periods=min_obs)
    if len(prices) < max(3, min_obs):
        return []

    tickers = sorted(series_map)
    candidates: list[PairCandidate] = []
    try:
        prescreen = float(prescreen_corr)
    except (TypeError, ValueError):
        prescreen = 0.70
    prescreen = max(0.0, min(1.0, prescreen))

    for i, ticker_a in enumerate(tickers):
        for ticker_b in tickers[i + 1:]:
            if not _sector_gate(sectors, ticker_a, ticker_b):
                continue
            try:
                corr = float(corr_matrix.at[ticker_a, ticker_b])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(corr) or abs(corr) < prescreen:
                continue
            candidate = test_cointegration(
                series_map[ticker_a],
                series_map[ticker_b],
                max_lag=max_lag,
                max_pvalue=max_pvalue,
                min_half_life=min_half_life,
                max_half_life=max_half_life,
                min_observations=min_obs,
                lookback_days=lookback_days,
                sector_a=(sectors or {}).get(ticker_a) if sectors else None,
                sector_b=(sectors or {}).get(ticker_b) if sectors else None,
            )
            if candidate is not None:
                # The candidate carries no ticker names when constructed
                # from unnamed Series; stamp the universe names.
                candidate.ticker_a = ticker_a
                candidate.ticker_b = ticker_b
                candidates.append(candidate)

    candidates.sort(key=lambda c: (c.adf_pvalue, c.pair_id))
    return candidates
