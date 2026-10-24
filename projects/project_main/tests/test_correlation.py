"""Tests for return-correlation clustering (src/quant/correlation.py).

Fixtures are seeded-random walks plus exact deterministic multiples, so
correlations are stable where the assertions need them to be and the
near-boundary cases assert against the *sample* correlation (no statistics
gambling).
"""


def _impl_ready() -> bool:
    import importlib.util

    modules = ("src.quant.correlation",)
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import numpy as np
import pandas as pd
import pytest

from src.quant.correlation import (
    CLUSTER_ID_PREFIX,
    correlation_clusters,
    returns_frame,
)

N = 130  # ~6mo of trading days


def _walk(seed: int, n: int = N, start: float = 100.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.01, n)
    return pd.Series(start * np.cumprod(1.0 + rets))


def _noisy_copy(series: pd.Series, seed: int, sigma: float) -> pd.Series:
    """Random walk whose returns are ``series`` returns + fresh noise."""
    rng = np.random.default_rng(seed)
    rets = series.pct_change().dropna().to_numpy() \
        + rng.normal(0.0, sigma, len(series) - 1)
    start = float(series.iloc[0])
    values = np.empty(len(rets) + 1)
    values[0] = start
    values[1:] = start * np.cumprod(1.0 + rets)
    return pd.Series(values)


def _sample_corr(a: pd.Series, b: pd.Series) -> float:
    frame = pd.concat(
        [a.pct_change().dropna(), b.pct_change().dropna()], axis=1
    ).dropna()
    return float(frame.corr().iloc[0, 1])


# ---------------------------------------------------------------------------
# returns_frame
# ---------------------------------------------------------------------------


def test_returns_frame_builds_aligned_returns():
    a, b = _walk(1), _walk(2)
    frame = returns_frame({"AAA": a, "BBB": b})
    assert frame.shape[1] == 2
    assert frame.notna().all().all()
    # Rows = observations - 1 (pct_change) on the aligned index.
    assert len(frame) == len(a) - 1


def test_returns_frame_drops_junk_entries():
    a = _walk(1)
    frame = returns_frame({
        "AAA": a,
        "BBB": None,
        "CCC": "junk",
        "DDD": pd.Series([1.0]),          # too short
        "EEE": pd.DataFrame({"Close": a}),
        "": a,                             # blank ticker dropped
    })
    assert list(frame.columns) == ["AAA", "EEE"]


def test_returns_frame_needs_two_tickers():
    assert returns_frame({"AAA": _walk(1)}).empty
    assert returns_frame(None).empty
    assert returns_frame({}).empty


# ---------------------------------------------------------------------------
# correlation_clusters
# ---------------------------------------------------------------------------


def test_perfect_correlation_clusters():
    a = _walk(1)
    b = a * 2.0                     # exact multiple -> corr exactly 1.0
    d = _walk(2)                    # independent seed
    mapping = correlation_clusters({"AAA": a, "BBB": b, "DDD": d})
    assert set(mapping) == {"AAA", "BBB"}
    assert mapping["AAA"] == mapping["BBB"]


def test_threshold_gates_pairing_around_sample_corr():
    a = _walk(3)
    c = _noisy_copy(a, seed=4, sigma=0.01)  # corr(A, C) ~ 0.7
    sample = _sample_corr(a, c)
    assert 0.5 < sample < 0.9       # sanity on the fixture
    above = correlation_clusters({"AAA": a, "CCC": c}, threshold=sample + 0.05)
    assert above == {}
    below = correlation_clusters(
        {"AAA": a, "CCC": c}, threshold=max(0.0, sample - 0.05)
    )
    assert set(below) == {"AAA", "CCC"}
    assert below["AAA"] == below["CCC"]


def test_negative_correlation_does_not_cluster():
    a = _walk(5)
    mirror = 200.0 - a              # perfect anti-correlation (hedge)
    mapping = correlation_clusters({"AAA": a, "MMM": mirror}, threshold=0.5)
    assert mapping == {}


def test_uncorrelated_tickers_stay_absent():
    mapping = correlation_clusters({"AAA": _walk(6), "BBB": _walk(7)})
    assert mapping == {}


def test_single_linkage_chain_merges_indirect_pair():
    # Deterministic patterns with EXACT correlations: corr(A,B) =
    # corr(B,C) = 1/sqrt(2) ~ 0.707 >= 0.70 threshold, corr(A,C) = 0.
    # Only single-linkage chaining merges all three (B bridges A and C).
    pattern_a = [1.0, -1.0] * 64            # 128 returns, mean 0, var 1
    pattern_c = [1.0, 1.0, -1.0, -1.0] * 32  # orthogonal to pattern_a
    pattern_b = [a + c for a, c in zip(pattern_a, pattern_c)]

    def _prices(pattern):
        rets = pd.Series([0.01 * v for v in pattern])
        prices = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + rets)])
        return pd.Series(prices)

    # pct_change recovers exactly 0.01 * pattern for these paths.
    closes = {"AAA": _prices(pattern_a), "BBB": _prices(pattern_b),
              "CCC": _prices(pattern_c)}
    frame = returns_frame(closes)
    corr = frame.corr()
    assert corr.at["AAA", "BBB"] == pytest.approx(2 ** -0.5)
    assert corr.at["BBB", "CCC"] == pytest.approx(2 ** -0.5)
    assert corr.at["AAA", "CCC"] == pytest.approx(0.0, abs=1e-12)

    mapping = correlation_clusters(closes, threshold=0.70)
    assert set(mapping) == {"AAA", "BBB", "CCC"}
    assert mapping["AAA"] == mapping["BBB"] == mapping["CCC"]


def test_cluster_ids_are_deterministic_and_sorted():
    a = _walk(10)
    mapping = correlation_clusters({"ZZZ": a, "AAA": a * 2.0})
    assert mapping == {
        "AAA": f"{CLUSTER_ID_PREFIX}AAA, ZZZ",
        "ZZZ": f"{CLUSTER_ID_PREFIX}AAA, ZZZ",
    }


def test_min_observations_fail_open():
    a = _walk(11, n=70)
    mapping = correlation_clusters({"AAA": a, "BBB": a * 2.0},
                                   min_observations=100)
    assert mapping == {}
    # At the boundary: 69 aligned returns >= 68 required -> clusters.
    mapping = correlation_clusters({"AAA": a, "BBB": a * 2.0},
                                   min_observations=68)
    assert set(mapping) == {"AAA", "BBB"}


def test_single_ticker_and_tiny_frames_fail_open():
    assert correlation_clusters({"AAA": _walk(12)}) == {}
    assert correlation_clusters({"AAA": pd.Series([1.0, 2.0]),
                                 "BBB": pd.Series([1.0, 2.0])}) == {}


def test_junk_arguments_fail_open():
    a, b = _walk(13), _walk(14)
    for closes, threshold, min_obs in (
        (None, 0.8, 60),
        ("junk", 0.8, 60),
        ({"AAA": None, "BBB": None}, 0.8, 60),
        ({"AAA": "junk", "BBB": 42}, 0.8, 60),
        ({"AAA": a, "BBB": b}, "junk", 60),
        ({"AAA": a, "BBB": b}, float("nan"), 60),
        ({"AAA": a, "BBB": b}, 0.8, "junk"),
        ({"AAA": a, "BBB": b}, 0.8, None),
    ):
        assert correlation_clusters(closes, threshold, min_obs) == {}, \
            (closes, threshold, min_obs)


def test_threshold_clamped_to_unit_interval():
    a = _walk(15)
    # threshold 0 -> any non-negative sample corr links; the independent
    # pair may or may not link, but a perfect pair always must.
    mapping = correlation_clusters({"AAA": a, "BBB": a * 2.0}, threshold=-5)
    assert set(mapping) == {"AAA", "BBB"}


def test_dataframe_close_column_input():
    a = _walk(16)
    frame = pd.DataFrame({"Close": a, "Volume": np.arange(len(a))})
    mapping = correlation_clusters({"AAA": frame, "BBB": a * 2.0})
    assert set(mapping) == {"AAA", "BBB"}


def test_overlap_alignment_uses_intersection():
    a = _walk(17)
    b = (a * 2.0).copy()
    # Give the tickers disjoint trading calendars: no aligned observations
    # -> fail-open with no clusters rather than mis-correlated pairs.
    a.index = pd.date_range("2026-01-01", periods=len(a), freq="B")
    b.index = pd.date_range("2026-06-01", periods=len(b), freq="B")
    mapping = correlation_clusters({"AAA": a, "BBB": b})
    assert mapping == {}


# ---------------------------------------------------------------------------
# Sanity: correlation math is pandas' own (spot check)
# ---------------------------------------------------------------------------


def test_sample_corr_helper_agrees_with_pandas():
    a, b = _walk(18), _walk(19)
    direct = float(pd.concat([a.pct_change().dropna(),
                              b.pct_change().dropna()],
                             axis=1).dropna().corr().iloc[0, 1])
    assert _sample_corr(a, b) == pytest.approx(direct)
