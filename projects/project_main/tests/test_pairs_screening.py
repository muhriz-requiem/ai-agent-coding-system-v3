"""Tests for pair ranking + dedupe (src/pairs/screening.py).

The walk-forward simulation is exercised on synthetic candidates whose
outcome is deterministic by construction: a strongly mean-reverting
planted spread ranks above a random-walk "pair", the greedy dedupe
enforces ticker/sector bounds, and every input-degradation path
degrades to empty.
"""

import numpy as np
import pandas as pd
import pytest

from src.pairs.cointegration import PairCandidate
from src.pairs.screening import (
    RankedPair,
    adv_map,
    dedupe_pairs,
    form_pairs,
    rank_pairs,
)

N = 300


def _candidate(a: str = "AAA", b: str = "BBB", pvalue: float = 0.01,
               sector: str = "Tech") -> PairCandidate:
    return PairCandidate(
        ticker_a=a,
        ticker_b=b,
        hedge_ratio=1.0,
        half_life=10.0,
        adf_pvalue=pvalue,
        lookback_days=252,
        alpha=0.5,
        spread_mean=0.0,
        spread_std=0.02,
        corr=0.9,
        sector_a=sector,
        sector_b=sector,
    )


def _frames() -> dict[str, pd.DataFrame]:
    """OHLCV frames: AAA/BBB mean-revert, EEE/FFF/GGG are noise walks."""
    rng = np.random.default_rng(3)
    log_b = np.cumsum(rng.normal(0.0002, 0.01, N)) + np.log(100.0)
    # phi = 0.90 -> half-life ~6.6 days (inside the default [5, 60] window).
    spread = np.zeros(N)
    for t in range(1, N):
        spread[t] = 0.90 * spread[t - 1] + rng.normal(0.0, 0.008)
    log_a = log_b + spread
    close_a = np.exp(log_a)
    close_b = np.exp(log_b)
    volume = np.full(N, 1_000_000.0)
    idx = pd.date_range("2025-01-01", periods=N, freq="B")
    frames = {
        "AAA": pd.DataFrame({"Close": close_a, "Volume": volume}, index=idx),
        "BBB": pd.DataFrame({"Close": close_b, "Volume": volume}, index=idx),
    }
    for i, nm in enumerate(("EEE", "FFF", "GGG")):
        walk = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.015, N)))
        frames[nm] = pd.DataFrame({"Close": walk, "Volume": volume}, index=idx)
    return frames


# ---------------------------------------------------------------------------
# adv_map
# ---------------------------------------------------------------------------


def test_adv_map_computes_dollar_volume():
    frames = _frames()
    advs = adv_map(frames, lookback_days=252)
    assert set(advs) == {"AAA", "BBB", "EEE", "FFF", "GGG"}
    assert advs["AAA"] == pytest.approx(1_000_000.0 * frames["AAA"]["Close"].mean(),
                                        rel=0.2)


def test_adv_map_requires_volume_column():
    close_only = pd.DataFrame({"Close": np.linspace(1, 2, N)})
    assert adv_map({"AAA": close_only}) == {}
    assert adv_map({"AAA": pd.Series(np.linspace(1, 2, N))}) == {}
    assert adv_map({}) == {}
    assert adv_map(None) == {}


# ---------------------------------------------------------------------------
# rank_pairs
# ---------------------------------------------------------------------------


def test_rank_orders_planted_pair_above_noise():
    frames = _frames()
    candidates = [
        _candidate("AAA", "BBB", pvalue=0.01),
        _candidate("EEE", "FFF", pvalue=0.001, sector="Fin"),
    ]
    ranked = rank_pairs(candidates, frames)
    assert len(ranked) == 2
    assert ranked[0].candidate.pair_id == "AAA/BBB"
    assert ranked[0].sharpe > ranked[1].sharpe
    assert ranked[0].liquidity_score > 0.0
    assert 0.0 <= ranked[0].stability_score <= 1.0


def test_rank_stability_reflects_beta_agreement():
    frames = _frames()
    candidates = [_candidate("AAA", "BBB"), _candidate("EEE", "FFF")]
    ranked = rank_pairs(candidates, frames)
    by_id = {r.candidate.pair_id: r for r in ranked}
    # The mean-reverting pair's halves agree on the hedge ratio far
    # better than two independent walks.
    assert by_id["AAA/BBB"].stability_score > by_id["EEE/FFF"].stability_score


def test_rank_deterministic():
    frames = _frames()
    candidates = [_candidate("AAA", "BBB"), _candidate("EEE", "FFF", sector="Fin")]
    first = rank_pairs(candidates, frames)
    second = rank_pairs(candidates, frames)
    assert [(r.candidate.pair_id, r.sharpe) for r in first] \
        == [(r.candidate.pair_id, r.sharpe) for r in second]


def test_rank_missing_close_data_drops_candidate():
    frames = _frames()
    ranked = rank_pairs([_candidate("AAA", "NOPE")], frames)
    assert ranked == []
    assert rank_pairs([], frames) == []
    assert rank_pairs([None], frames) == []


# ---------------------------------------------------------------------------
# dedupe_pairs
# ---------------------------------------------------------------------------


def _ranked(pair_id: str, sharpe: float, sector: str = "Tech") -> RankedPair:
    a, b = pair_id.split("/")
    return RankedPair(
        candidate=_candidate(a, b, sector=sector),
        sharpe=sharpe,
        stability_score=0.9,
        liquidity_score=1e6,
    )


def test_dedupe_no_reused_tickers():
    ranked = [_ranked("AAA/BBB", 2.0), _ranked("BBB/CCC", 1.5),
              _ranked("CCC/DDD", 1.0)]
    kept = dedupe_pairs(ranked, max_pairs=10, max_per_sector=10)
    assert [r.candidate.pair_id for r in kept] == ["AAA/BBB", "CCC/DDD"]


def test_dedupe_max_pairs():
    ranked = [_ranked(f"T{i}/T{i + 10}", 2.0 - i * 0.1) for i in range(5)]
    kept = dedupe_pairs(ranked, max_pairs=2, max_per_sector=10)
    assert len(kept) == 2


def test_dedupe_sector_cap():
    ranked = [
        _ranked("A1/A2", 2.0),
        _ranked("A3/A4", 1.9),
        _ranked("A5/A6", 1.8),
        _ranked("A7/A8", 1.7),
    ]
    kept = dedupe_pairs(ranked, max_pairs=10, max_per_sector=2)
    assert len(kept) == 2
    assert kept[0].candidate.sector_a == "Tech"


def test_dedupe_unknown_sector_exempt_from_sector_cap():
    ranked = [
        _ranked("A1/A2", 2.0, sector=""),
        _ranked("A3/A4", 1.9, sector=""),
        _ranked("A5/A6", 1.8, sector=""),
    ]
    kept = dedupe_pairs(ranked, max_pairs=10, max_per_sector=1)
    assert len(kept) == 3  # unknown sectors are not concentration-bounded


def test_dedupe_junk_bounds_fail_open():
    ranked = [_ranked("A1/A2", 2.0), _ranked("A3/A4", 1.9)]
    kept = dedupe_pairs(ranked, max_pairs="junk", max_per_sector="junk")
    assert len(kept) == 2
    assert dedupe_pairs([], max_pairs=5) == []


# ---------------------------------------------------------------------------
# form_pairs (full pipeline)
# ---------------------------------------------------------------------------


def test_form_pairs_end_to_end():
    frames = _frames()
    sectors = {"AAA": "Tech", "BBB": "Tech", "EEE": "Fin",
               "FFF": "Fin", "GGG": "Fin"}
    kept = form_pairs(frames, sectors=sectors, min_sharpe=None)
    ids = [r.candidate.pair_id for r in kept]
    assert ids and ids[0] == "AAA/BBB"
    assert "EEE/FFF" not in ids  # noise pair must not survive


def test_form_pairs_min_sharpe_gate():
    frames = _frames()
    # An impossibly high Sharpe floor empties the result.
    assert form_pairs(frames, min_sharpe=1e6) == []
    assert form_pairs(frames, min_sharpe=None)  # gate off keeps pairs


def test_form_pairs_insufficient_data():
    assert form_pairs({}) == []
    assert form_pairs({"AAA": pd.Series([1.0, 2.0]),
                       "BBB": pd.Series([2.0, 3.0])}) == []
