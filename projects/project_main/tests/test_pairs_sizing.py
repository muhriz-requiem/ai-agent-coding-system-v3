"""Tests for pair-leg order planning (sizing seam).

Covers both entry points:
  * ``src.pairs.sizing.size_pair_legs`` (dataclass adapter), and
  * ``build_order_plan`` with ``pair_decisions``/``pair_signals`` (the
    pipeline path, where pair plans share the single-leg accumulators).

Pinned safety properties: long-only (SELL_EXIT capped at held qty,
skipped when unheld), scale-down-only factors except pairs Kelly,
per-pair / total-pairs / shared-equity caps, dust filter, cap priority
(pair legs skipped first), PDT + panic BUY gates.
"""

import math

import pytest

from src.agents.models import RiskDecision
from src.agents.portfolio_manager import build_order_plan
from src.pairs.risk import PairRiskDecision
from src.pairs.sizing import pair_kelly_from_stats, size_pair_legs

EQUITY = 100_000.0


def _account(**overrides) -> dict:
    payload = {
        "equity": EQUITY,
        "max_position_pct": 0.10,
        "target_daily_vol": 0.02,
        "pair_max_pct": 0.03,
        "pairs_total_pct": 0.30,
        "pair_z_entry": 2.0,
        "cost_buffer_pct": 0.0,
        "min_trade_notional": 10.0,
    }
    payload.update(overrides)
    return payload


def _signal(pair_id="AAA/BBB", z=2.5, a="AAA", b="BBB",
            sig_a="SELL_EXIT", sig_b="BUY", spread_vol=0.01, **extra) -> dict:
    # With beta=1, alpha=0: z=+2.5 means A is expensive, B cheap.
    try:
        z_value = float(z)
    except (TypeError, ValueError):
        z_value = 2.5  # junk z still needs a price to reach the planner
    price_a = 100.0 * math.exp(z_value * 0.02)
    payload = {
        "pair_id": pair_id,
        "ticker_a": a,
        "ticker_b": b,
        "hedge_ratio": 1.0,
        "z_score": z,
        "signal_a": sig_a,
        "signal_b": sig_b,
        "price_a": price_a,
        "price_b": 100.0,
        "spread_daily_vol": spread_vol,
    }
    payload.update(extra)
    return payload


def _decision(pair_id="AAA/BBB", approved=True) -> dict:
    return {"pair_id": pair_id, "approved": approved, "veto_reasons": []}


# ---------------------------------------------------------------------------
# size_pair_legs (adapter)
# ---------------------------------------------------------------------------


def test_buy_leg_budget_is_1pct_of_equity_at_defaults():
    # Base leg = 100k * 0.03 / 2 = 1500.  Conviction |2.5|/2 = 1.0 (cap),
    # spread vol 0.01 < target 0.02 -> vol factor 1.0, no kelly/regime.
    plans = size_pair_legs(_signal(), _decision(), _account())
    assert len(plans) == 1
    order = plans[0]
    assert order.ticker == "BBB"
    assert order.side == "buy"
    assert order.quantity * order.limit_price == pytest.approx(1500.0, rel=1e-6)


def test_conviction_scales_down_near_entry():
    plans = size_pair_legs(_signal(z=1.0), _decision(), _account())
    order = plans[0]
    assert order.quantity * order.limit_price == pytest.approx(1500.0 * 0.5,
                                                              rel=1e-6)


def test_vol_scale_down_only():
    plans = size_pair_legs(_signal(spread_vol=0.04), _decision(), _account())
    order = plans[0]
    assert order.quantity * order.limit_price == pytest.approx(1500.0 * 0.5,
                                                              rel=1e-6)
    # Low spread vol never scales up.
    plans_hi = size_pair_legs(_signal(spread_vol=0.001), _decision(),
                              _account())
    assert plans_hi[0].quantity * plans_hi[0].limit_price \
        == pytest.approx(1500.0, rel=1e-6)


def test_kelly_is_the_only_scale_up():
    plans = size_pair_legs(
        _signal(), _decision(), _account(),
        kelly={"fraction": 0.5, "n": 100})
    order = plans[0]
    # fraction 0.5 / cap 0.5 -> mult 1.5.
    assert order.quantity * order.limit_price == pytest.approx(2250.0, rel=1e-6)
    assert "kelly x1.50" in order.rationale


def test_regime_risk_off_scales_budget():
    plans = size_pair_legs(_signal(), _decision(), _account(),
                           regime={"regime": "risk_off"})
    order = plans[0]
    assert order.quantity * order.limit_price == pytest.approx(750.0, rel=1e-6)
    assert "regime risk_off" in order.rationale


def test_panic_blocks_buy_legs_but_sells_pass():
    plans = size_pair_legs(_signal(), _decision(), _account(),
                           regime={"regime": "panic"})
    assert plans == []
    # A held leg exits mechanically even in panic (SELL_EXIT passes).
    plans = size_pair_legs(
        _signal(sig_a="SELL_EXIT", sig_b="SELL_EXIT"), _decision(),
        _account(), positions={"AAA": 5.0}, regime={"regime": "panic"})
    assert [p.side for p in plans] == ["sell"]
    assert plans[0].quantity == pytest.approx(5.0)


def test_sell_exit_is_mechanical_full_exit():
    plans = size_pair_legs(
        _signal(sig_a="SELL_EXIT", sig_b="SELL_EXIT", z=0.2),
        _decision(), _account(), positions={"AAA": 3.0, "BBB": 2.0})
    assert [p.ticker for p in plans] == ["AAA", "BBB"]
    assert all(p.side == "sell" for p in plans)
    assert plans[0].quantity == pytest.approx(3.0)
    assert plans[1].quantity == pytest.approx(2.0)


def test_sell_exit_skipped_when_unheld():
    plans = size_pair_legs(
        _signal(sig_a="SELL_EXIT", sig_b="SELL_EXIT"),
        _decision(), _account(), positions=None)
    assert plans == []


def test_unapproved_pair_emits_nothing():
    assert size_pair_legs(_signal(), _decision(approved=False), _account()) == []


def test_missing_z_emits_no_buy():
    assert size_pair_legs(_signal(z="junk"), _decision(), _account()) == []


def test_dust_orders_skipped():
    plans = size_pair_legs(
        _signal(), _decision(), _account(equity=300.0))  # leg = 4.5 -> dust
    assert plans == []


def test_junk_account_plans_nothing():
    assert size_pair_legs(_signal(), _decision(), {"equity": "junk"}) == []
    assert size_pair_legs(_signal(), _decision(), {}) == []


# ---------------------------------------------------------------------------
# build_order_plan: pair integration + caps
# ---------------------------------------------------------------------------


def _single_leg_decision(ticker="MSFT", approved=True) -> RiskDecision:
    return RiskDecision(ticker=ticker, approved=approved, veto_reasons=[])


def _single_leg_signal(ticker="MSFT") -> dict:
    return {"signal": "BUY", "reference_price": 100.0, "confidence": 0.95,
            "daily_vol": 0.02}


def test_pipeline_path_emits_single_leg_then_pair_legs():
    plans = build_order_plan(
        [_single_leg_decision()], {"MSFT": _single_leg_signal()},
        _account(), [], "2026-09-12",
        pair_decisions=[_decision()], pair_signals=[_signal()],
    )
    tickers = [p.ticker for p in plans]
    assert tickers[0] == "MSFT"          # single-leg first (cap priority)
    assert "BBB" in tickers


def test_shared_equity_cap_skips_pair_legs_first():
    # Two single-leg BUYs of 50% each exactly exhaust the equity cap;
    # the pair BUY must then be skipped while the (cap-free) SELL_EXIT
    # on a held leg still passes.
    account = _account(max_position_pct=0.50)
    plans = build_order_plan(
        [_single_leg_decision("MSFT"), _single_leg_decision("AAPL")],
        {"MSFT": _single_leg_signal(), "AAPL": _single_leg_signal()},
        account, [], "2026-09-12",
        pair_decisions=[_decision()], pair_signals=[_signal()],
    )
    buy_tickers = [p.ticker for p in plans if p.side == "buy"]
    assert buy_tickers == ["MSFT", "AAPL"]
    assert "BBB" not in buy_tickers  # equity exhausted -> pair leg skipped
    # And the shared cap blocks pair BUYs while SELL_EXITs pass.
    plans = build_order_plan(
        [_single_leg_decision("MSFT"), _single_leg_decision("AAPL")],
        {"MSFT": _single_leg_signal(), "AAPL": _single_leg_signal()},
        account, [], "2026-09-12",
        positions={"AAA": 5.0},
        pair_decisions=[_decision()],
        pair_signals=[_signal(sig_a="SELL_EXIT", sig_b="BUY")],
    )
    pair_orders = [(p.ticker, p.side) for p in plans
                   if p.ticker in {"AAA", "BBB"}]
    assert pair_orders == [("AAA", "sell")]


def test_per_pair_cap_counts_both_legs():
    # Two BUY legs in one pair: combined notional must not exceed 3%.
    plans = build_order_plan(
        [_single_leg_decision("MSFT")], {"MSFT": _single_leg_signal()},
        _account(), [], "2026-09-12",
        pair_decisions=[_decision()],
        pair_signals=[_signal(sig_a="BUY", sig_b="BUY", z=2.5)],
    )
    pair_legs = [p for p in plans if p.ticker in {"AAA", "BBB"}
                 and p.side == "buy"]
    assert len(pair_legs) == 2
    total = sum(p.quantity * p.limit_price for p in pair_legs)
    assert total <= EQUITY * 0.03 + 1e-6


def test_total_pairs_cap():
    # 11 pairs * 1.5% leg = 16.5% > ... use 25 pairs to blow the 30% rail.
    decisions, signals = [], []
    for i in range(25):
        pid = f"P{i}A/P{i}B"
        decisions.append(_decision(pid))
        signals.append(_signal(pair_id=pid, a=f"P{i}A", b=f"P{i}B"))
    plans = build_order_plan(
        [_single_leg_decision("MSFT")], {"MSFT": _single_leg_signal()},
        _account(), [], "2026-09-12",
        pair_decisions=decisions, pair_signals=signals,
    )
    pair_buys = [p for p in plans if p.side == "buy"
                 and p.ticker.startswith("P")]
    total = sum(p.quantity * p.limit_price for p in pair_buys)
    assert total <= EQUITY * 0.30 + 1e-6
    assert len(pair_buys) < 50  # the rail bound the count


def test_shared_sector_cap_consumed_by_pair_legs():
    account = _account(max_sector_pct=0.02)  # 2k sector budget
    plans = build_order_plan(
        [_single_leg_decision("MSFT")], {"MSFT": _single_leg_signal()},
        account, [], "2026-09-12",
        sector_map={"MSFT": "Tech", "BBB": "Tech"},
        pair_decisions=[_decision()], pair_signals=[_signal()],
    )
    # MSFT consumed the 2k Tech budget (10% of equity -> 10k budget...
    # actually max_position 10% = 10k > 2k cap -> MSFT skipped; pair leg
    # (1.5k) then fits under the same 2k accumulator.
    tech_buys = [p for p in plans if p.side == "buy"]
    total = sum(p.quantity * p.limit_price for p in tech_buys)
    assert total <= EQUITY * 0.02 + 1e-6


def test_pdt_restriction_blocks_pair_buys():
    day_trades = ["2026-09-08", "2026-09-09", "2026-09-10"]
    plans = build_order_plan(
        [_single_leg_decision("MSFT")], {"MSFT": _single_leg_signal()},
        _account(), day_trades, "2026-09-12",
        pair_decisions=[_decision()], pair_signals=[_signal()],
    )
    assert [p for p in plans if p.side == "buy"] == []


def test_no_pair_inputs_unchanged_behavior():
    plans = build_order_plan(
        [_single_leg_decision()], {"MSFT": _single_leg_signal()},
        _account(), [], "2026-09-12")
    assert [p.ticker for p in plans] == ["MSFT"]
    # Explicit Nones behave identically.
    plans2 = build_order_plan(
        [_single_leg_decision()], {"MSFT": _single_leg_signal()},
        _account(), [], "2026-09-12",
        pair_decisions=None, pair_signals=None)
    assert [p.ticker for p in plans2] == ["MSFT"]


def test_pair_decision_duck_typing_objects():
    plans = build_order_plan(
        [_single_leg_decision()], {"MSFT": _single_leg_signal()},
        _account(), [], "2026-09-12",
        pair_decisions=[PairRiskDecision(pair_id="AAA/BBB", approved=True)],
        pair_signals=[_signal()],
    )
    assert any(p.ticker == "BBB" for p in plans)


# ---------------------------------------------------------------------------
# pair_kelly_from_stats
# ---------------------------------------------------------------------------


def test_pair_kelly_from_stats(tmp_path):
    import json

    path = tmp_path / "stats.json"
    path.write_text(json.dumps({"aggregate": {"kelly_fraction": 0.1,
                                              "n_trades": 50}}),
                    encoding="utf-8")
    assert pair_kelly_from_stats(path) == {"fraction": 0.1, "n": 50}
    # Below min trades -> None.
    assert pair_kelly_from_stats(path, min_trades=100) is None
    # Non-positive fraction -> None.
    path.write_text(json.dumps({"aggregate": {"kelly_fraction": -0.1,
                                              "n_trades": 50}}),
                    encoding="utf-8")
    assert pair_kelly_from_stats(path) is None
    # Missing file -> None.
    assert pair_kelly_from_stats(tmp_path / "nope.json") is None
