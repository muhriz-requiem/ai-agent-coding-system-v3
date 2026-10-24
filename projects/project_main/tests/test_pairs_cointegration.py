"""Tests for the Engle-Granger engine (src/pairs/cointegration.py).

Synthetic data: an exact cointegrated pair built from a random-walk
log price plus a known-parameter OU spread, and independent random
walks that must be rejected.  Seeds are pinned so every assertion is
deterministic (verified empirically: the planted pair is recovered
with beta/half-life near truth and the noise pairs are rejected).
"""

import numpy as np
import pandas as pd
import pytest

from src.pairs.cointegration import (
    MIN_OBSERVATIONS,
    PairCandidate,
    find_cointegrated_pairs,
    spread_residuals,
    test_cointegration as run_coint_test,
)

N = 600
TRUE_BETA = 1.2
TRUE_ALPHA = 0.5
TRUE_HALF_LIFE = np.log(2.0) / -np.log(0.90)  # ~6.58 days


def _planted_pair(seed: int = 42) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0.0002, 0.01, N)) + np.log(100.0)
    phi, sigma = 0.90, 0.005
    spread = np.zeros(N)
    for t in range(1, N):
        spread[t] = phi * spread[t - 1] + rng.normal(0.0, sigma)
    log_a = log_b * TRUE_BETA + TRUE_ALPHA + spread
    return (
        pd.Series(np.exp(log_a), name="AAA"),
        pd.Series(np.exp(log_b), name="BBB"),
    )


def _independent_walks(seed: int = 1) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    w1 = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, N))),
                   name="XXX")
    w2 = pd.Series(50.0 * np.exp(np.cumsum(rng.normal(-0.0001, 0.015, N))),
                   name="YYY")
    return w1, w2


# ---------------------------------------------------------------------------
# test_cointegration -- accept path
# ---------------------------------------------------------------------------


def test_planted_pair_is_cointegrated():
    a, b = _planted_pair()
    candidate = run_coint_test(a, b)
    assert candidate is not None
    assert isinstance(candidate, PairCandidate)
    assert candidate.pair_id == "AAA/BBB"
    assert candidate.adf_pvalue < 0.01
    # Finite-sample OLS bias on a persistent OU error is real but small
    # at these parameters; the assertions pin the verified realization.
    assert candidate.hedge_ratio == pytest.approx(TRUE_BETA, abs=0.10)
    assert candidate.alpha == pytest.approx(TRUE_ALPHA, abs=0.25)
    assert candidate.half_life == pytest.approx(TRUE_HALF_LIFE, rel=0.5)
    assert 5.0 <= candidate.half_life <= 60.0
    assert candidate.corr > 0.8
    assert candidate.spread_std > 0.0
    assert candidate.lookback_days == 252


def test_half_life_window_rejects_outliers():
    a, b = _planted_pair()
    # The planted half-life (~6.6d) sits just above the floor: raising
    # the floor above it must reject.
    assert run_coint_test(a, b, min_half_life=TRUE_HALF_LIFE + 5.0) is None
    # And a ceiling below it likewise.
    assert run_coint_test(a, b, max_half_life=3.0) is None


# ---------------------------------------------------------------------------
# test_cointegration -- reject paths (fail-open -> None)
# ---------------------------------------------------------------------------


def test_independent_walks_are_rejected():
    w1, w2 = _independent_walks(seed=1)  # verified rejecting seed
    assert run_coint_test(w1, w2) is None


def test_short_windows_are_rejected():
    a, b = _planted_pair()
    assert run_coint_test(a, b, min_observations=10_000) is None
    assert run_coint_test(a.iloc[:20], b.iloc[:20]) is None


def test_junk_inputs_are_rejected():
    a, b = _planted_pair()
    assert run_coint_test(None, b) is None
    assert run_coint_test(a, None) is None
    assert run_coint_test(pd.Series([1.0, 2.0]), b) is None
    assert run_coint_test(pd.Series(["x"] * 100), b) is None
    # Zero price -> non-finite log -> reject.
    zeros = a.copy()
    zeros.iloc[-1] = 0.0
    assert run_coint_test(zeros, b) is None


def test_rejects_when_beta_nonpositive():
    # log(P_A) decreasing in log(P_B): beta <= 0 -> reject.
    rng = np.random.default_rng(5)
    log_b = np.cumsum(rng.normal(0.0, 0.01, N)) + np.log(100.0)
    log_a = -1.0 * log_b + 3.0
    assert run_coint_test(
        pd.Series(np.exp(log_a), name="A"),
        pd.Series(np.exp(log_b), name="B"),
        min_observations=100,
    ) is None


# ---------------------------------------------------------------------------
# spread_residuals
# ---------------------------------------------------------------------------


def test_spread_residuals_rebuilds_without_refit():
    a, b = _planted_pair()
    candidate = run_coint_test(a, b)
    assert candidate is not None
    residuals = spread_residuals(candidate, a, b)
    assert residuals is not None
    assert len(residuals) == len(a)  # full aligned window
    # Residuals re-computed from the persisted parameters reconstruct
    # the planted spread up to the OLS parameter error.
    log_a = np.log(a.to_numpy())
    log_b = np.log(b.to_numpy())
    planted = log_a - TRUE_ALPHA - TRUE_BETA * log_b
    assert float(np.corrcoef(residuals.to_numpy(), planted)[0, 1]) > 0.95


def test_spread_residuals_junk_inputs():
    a, b = _planted_pair()
    candidate = run_coint_test(a, b)
    assert candidate is not None
    assert spread_residuals(candidate, None, b) is None
    assert spread_residuals(candidate, a, pd.Series([1.0])) is None


# ---------------------------------------------------------------------------
# find_cointegrated_pairs
# ---------------------------------------------------------------------------


def _universe() -> tuple[dict[str, pd.Series], dict[str, str]]:
    a, b = _planted_pair()
    rng = np.random.default_rng(11)
    log_d = np.cumsum(rng.normal(0.0001, 0.008, N)) + np.log(80.0)
    sp2 = np.zeros(N)
    for t in range(1, N):
        sp2[t] = 0.92 * sp2[t - 1] + rng.normal(0.0, 0.005)
    log_c = log_d * 0.8 + 0.3 + sp2
    names = {
        "AAA": a,
        "BBB": b,
        "CCC": pd.Series(np.exp(log_c), name="CCC"),
        "DDD": pd.Series(np.exp(log_d), name="DDD"),
    }
    for i, nm in enumerate(("EEE", "FFF", "GGG")):
        names[nm] = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, N))),
            name=nm,
        )
    sectors = {
        "AAA": "Tech", "BBB": "Tech", "CCC": "Tech", "DDD": "Tech",
        "EEE": "Fin", "FFF": "Fin", "GGG": "Fin",
    }
    return names, sectors


def test_finds_planted_pairs_and_rejects_noise():
    closes, sectors = _universe()
    found = find_cointegrated_pairs(closes, sectors=sectors)
    ids = [c.pair_id for c in found]
    assert "AAA/BBB" in ids
    assert "CCC/DDD" in ids
    # Noise walks never make it into the result.
    assert len(found) == 2
    # Deterministic ordering: ascending ADF p-value, then pair id.
    pvalues = [c.adf_pvalue for c in found]
    assert pvalues == sorted(pvalues)


def test_sector_gate_skips_cross_sector_pairs():
    closes, _ = _universe()
    # AAA/BBB planted as Tech; claiming CCC is Fin removes CCC/DDD.
    sectors = {
        "AAA": "Tech", "BBB": "Tech", "CCC": "Fin", "DDD": "Fin2",
        "EEE": "Fin", "FFF": "Fin", "GGG": "Fin",
    }
    found = find_cointegrated_pairs(closes, sectors=sectors)
    ids = [c.pair_id for c in found]
    assert "CCC/DDD" not in ids
    assert "AAA/BBB" in ids


def test_sector_gate_without_labels_skips_pairs():
    closes, _ = _universe()
    sectors = {"AAA": "Tech", "BBB": "Tech", "EEE": None}
    found = find_cointegrated_pairs(closes, sectors=sectors)
    assert [c.pair_id for c in found] == ["AAA/BBB"]


def test_sectors_none_disables_gate_but_keeps_prescreen():
    closes, sectors = _universe()
    found_all = find_cointegrated_pairs(closes, sectors=None)
    # Without the sector gate more pairs are tested, but the corr
    # pre-screen still governs: planted pairs remain the top hits.
    ids = {c.pair_id for c in found_all}
    assert {"AAA/BBB", "CCC/DDD"} <= ids


def test_prescreen_corr_excludes_uncorrelated_pairs():
    closes, _ = _universe()
    found = find_cointegrated_pairs(closes, sectors=None, prescreen_corr=0.999)
    # At a 0.999 corr floor only near-identical series survive; the
    # planted pairs (corr ~0.9) must be screened out.
    assert all(abs(c.corr) >= 0.998 for c in found)


def test_insufficient_observations_fail_closed_to_no_pairs():
    closes, sectors = _universe()
    assert find_cointegrated_pairs(
        closes, sectors=sectors, min_observations=10_000
    ) == []


def test_junk_universe_yields_empty():
    assert find_cointegrated_pairs({}) == []
    assert find_cointegrated_pairs({"AAA": None}) == []
    assert find_cointegrated_pairs(
        {"AAA": pd.Series([1.0, 2.0]), "BBB": pd.Series([1.0, 2.0])}
    ) == []


def test_min_observations_default():
    assert MIN_OBSERVATIONS == 60
