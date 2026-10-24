"""Tests for pairs risk gates (src/pairs/risk.py).

Each gate is exercised in both directions plus its fail-open path
(missing data leaves the gate open); panic must block BUY legs while
SELL_EXIT-only signals always pass.
"""

import numpy as np
import pandas as pd
import pytest

from src.pairs.book import BookEntry
from src.pairs.risk import (
    PairRiskDecision,
    evaluate_pair_risk,
)
from src.pairs.signals import PairSignal


def _entry(**overrides) -> BookEntry:
    payload = dict(
        pair_id="AAA/BBB",
        ticker_a="AAA",
        ticker_b="BBB",
        hedge_ratio=1.0,
        alpha=0.0,
        spread_mean=0.0,
        spread_std=0.02,
        half_life=10.0,
        formed_on="2026-09-01",
        spread_daily_vol=0.004,
        formation_corr=0.80,
        formation_adv={"AAA": 5_000_000.0, "BBB": 5_000_000.0},
    )
    payload.update(overrides)
    return BookEntry(**payload)


def _signal(signal_a="BUY", signal_b="SELL_EXIT", **overrides) -> PairSignal:
    payload = dict(
        pair_id="AAA/BBB",
        ticker_a="AAA",
        ticker_b="BBB",
        hedge_ratio=1.0,
        alpha=0.0,
        z_score=2.5,
        signal_a=signal_a,
        signal_b=signal_b,
        confidence=0.9,
        spread_mean=0.0,
        spread_std=0.02,
        price_a=105.0,
        price_b=100.0,
        spread_daily_vol=0.004,
    )
    payload.update(overrides)
    return PairSignal(**payload)


def _closes(corr: float = 0.8, n: int = 120, seed: int = 9):
    """Two leg price series with a controllable return correlation."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.01, n)
    noise = rng.normal(0.0, 0.01 * np.sqrt(max(1e-9, 1.0 - corr ** 2)) /
                       max(1e-9, abs(corr)) if abs(corr) < 1 else 0.0, n)
    ret_b = base
    ret_a = corr * base + (1.0 - corr) * noise
    close_a = pd.Series(100.0 * np.exp(np.cumsum(ret_a)))
    close_b = pd.Series(100.0 * np.exp(np.cumsum(ret_b)))
    return {"AAA": close_a, "BBB": close_b}


# ---------------------------------------------------------------------------
# decision shape
# ---------------------------------------------------------------------------


def test_decision_shape_and_to_dict():
    decision = PairRiskDecision(pair_id="AAA/BBB", approved=False,
                                veto_reasons=["x"])
    assert decision.to_dict() == {
        "pair_id": "AAA/BBB", "approved": False, "veto_reasons": ["x"],
    }


def test_clean_signal_passes_all_gates():
    decision = evaluate_pair_risk(_signal(), _entry(),
                                  regime={"regime": "neutral"},
                                  current_closes=_closes(0.8),
                                  current_adv={"AAA": 5e6, "BBB": 5e6})
    assert decision.approved
    assert decision.veto_reasons == []


# ---------------------------------------------------------------------------
# gate 1: regime panic
# ---------------------------------------------------------------------------


def test_panic_blocks_buy_legs():
    decision = evaluate_pair_risk(_signal(), _entry(),
                                  regime={"regime": "panic"})
    assert not decision.approved
    assert any("panic" in reason for reason in decision.veto_reasons)


def test_panic_allows_sell_exit_only_signals():
    decision = evaluate_pair_risk(
        _signal(signal_a="SELL_EXIT", signal_b="SELL_EXIT"),
        _entry(), regime={"regime": "panic"})
    assert decision.approved


def test_risk_off_does_not_veto():
    decision = evaluate_pair_risk(_signal(), _entry(),
                                  regime={"regime": "risk_off"})
    assert decision.approved


def test_missing_regime_fails_open():
    assert evaluate_pair_risk(_signal(), _entry(), regime=None).approved
    assert evaluate_pair_risk(_signal(), _entry(), regime="junk").approved


# ---------------------------------------------------------------------------
# gate 2: correlation breakdown
# ---------------------------------------------------------------------------


def test_correlation_breakdown_vetoes():
    # Formation corr 0.80 -> live corr below 0.5*0.8 = 0.40 vetoes.
    decision = evaluate_pair_risk(_signal(), _entry(),
                                  current_closes=_closes(0.10))
    assert not decision.approved
    assert any("correlation" in reason for reason in decision.veto_reasons)


def test_correlation_healthy_passes():
    assert evaluate_pair_risk(
        _signal(), _entry(), current_closes=_closes(0.85)).approved


def test_correlation_gate_fails_open_without_closes():
    assert evaluate_pair_risk(_signal(), _entry(),
                              current_closes=None).approved
    assert evaluate_pair_risk(_signal(), _entry(),
                              current_closes={}).approved


def test_correlation_gate_needs_formation_corr():
    entry = _entry(formation_corr=None)
    assert evaluate_pair_risk(
        _signal(), entry, current_closes=_closes(0.05)).approved


def test_correlation_gate_skipped_for_sell_exit_only():
    decision = evaluate_pair_risk(
        _signal(signal_a="SELL_EXIT", signal_b="SELL_EXIT"),
        _entry(), current_closes=_closes(0.05))
    assert decision.approved


# ---------------------------------------------------------------------------
# gate 3: half-life drift
# ---------------------------------------------------------------------------


def test_half_life_drift_vetoes():
    # A spread that has gone near-random-walk (phi -> 1) drives the live
    # half-life far beyond 2x formation (10d -> >20d).  400 bars keep
    # the AR(1) estimate deterministic and near the unit root.
    entry = _entry()
    closes = _closes(0.9, n=400)
    rng = np.random.default_rng(4)
    log_b = np.log(closes["BBB"].to_numpy())
    drift = np.cumsum(rng.normal(0.0, 0.01, len(log_b)))
    closes["AAA"] = pd.Series(np.exp(log_b * 1.0 + drift))
    decision = evaluate_pair_risk(_signal(), entry,
                                  current_closes=closes)
    assert not decision.approved
    assert any("half-life" in reason for reason in decision.veto_reasons)


def test_half_life_gate_fails_open_without_closes():
    assert evaluate_pair_risk(_signal(), _entry()).approved


def test_half_life_gate_skipped_for_sell_exit_only():
    closes = _closes(0.9, n=400)
    rng = np.random.default_rng(4)
    log_b = np.log(closes["BBB"].to_numpy())
    closes["AAA"] = pd.Series(np.exp(log_b + np.cumsum(rng.normal(0, 0.01, 400))))
    decision = evaluate_pair_risk(
        _signal(signal_a="SELL_EXIT", signal_b="SELL_EXIT"),
        _entry(), current_closes=closes)
    assert decision.approved


# ---------------------------------------------------------------------------
# gate 4: liquidity deterioration
# ---------------------------------------------------------------------------


def test_liquidity_drop_vetoes():
    decision = evaluate_pair_risk(
        _signal(), _entry(),
        current_adv={"AAA": 1_000_000.0, "BBB": 5_000_000.0})
    assert not decision.approved
    assert any("liquidity" in reason for reason in decision.veto_reasons)


def test_liquidity_healthy_passes():
    decision = evaluate_pair_risk(
        _signal(), _entry(),
        current_adv={"AAA": 4_999_999.0, "BBB": 5_000_000.0})
    assert decision.approved


def test_liquidity_gate_fails_open_on_missing_adv():
    assert evaluate_pair_risk(_signal(), _entry(),
                              current_adv=None).approved
    # ADV data for a ticker outside the pair leaves both legs unknown.
    assert evaluate_pair_risk(_signal(), _entry(),
                              current_adv={"XXX": 100.0}).approved
    assert evaluate_pair_risk(
        _signal(), _entry(formation_adv={}),
        current_adv={"AAA": 1.0, "BBB": 1.0}).approved


# ---------------------------------------------------------------------------
# stacking / fail-open envelope
# ---------------------------------------------------------------------------


def test_multiple_gates_stack_reasons():
    decision = evaluate_pair_risk(
        _signal(), _entry(),
        regime={"regime": "panic"},
        current_closes=_closes(0.05),
        current_adv={"AAA": 100.0, "BBB": 5e6},
    )
    assert not decision.approved
    assert len(decision.veto_reasons) >= 3


def test_never_raises_on_junk_inputs():
    assert evaluate_pair_risk(None, None).approved or True
    decision = evaluate_pair_risk(
        _signal(), _entry(),
        current_closes={"AAA": "junk", "BBB": None},
        current_adv={"AAA": "junk"},
        regime=None,
    )
    assert isinstance(decision, PairRiskDecision)
