"""Tests for portfolio-level sizing, batching and the PDT rule."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.agents.portfolio_manager", "src.agents.models")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

import math

from src.agents.models import OrderPlan, RiskDecision
from src.agents.portfolio_manager import build_order_plan

_ACCOUNT = {"equity": 100_000.0, "max_position_pct": 0.10}


def _decision(ticker, approved=True) -> RiskDecision:
    return RiskDecision.model_validate({"ticker": ticker,
                                        "approved": approved})


def _signals(ticker, price, signal="BUY") -> dict:
    # confidence 0.95 keeps the conviction multiplier at 1.0 so base sizing
    # tests exercise the fixed-fractional formula unchanged.
    return {ticker: {"signal": signal, "confidence": 0.95,
                     "reference_price": price}}


def test_fixed_fractional_sizing():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1
    expected_qty = math.floor(100_000.0 * 0.10 / 250.0)  # = 40
    assert orders[0].quantity == expected_qty
    assert orders[0].limit_price == 250.0
    assert orders[0].side == "buy"


def test_fractional_sizing_small_account():
    # $1000 account, 10% per position, $250 price -> 0.4 shares
    small_account = {"equity": 1000.0, "max_position_pct": 0.10}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=small_account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1
    assert orders[0].quantity == 0.4
    assert orders[0].limit_price == 250.0
    assert orders[0].side == "buy"


def test_skips_hold_and_unapproved():
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT", approved=False)],
        signals={**_signals("AAPL", 250.0),
                 "MSFT": {"signal": "HOLD", "confidence": 0.5,
                          "reference_price": 400.0}},
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert [o.ticker for o in orders] == ["AAPL"]


def test_capital_cap_limits_total_notional():
    # Tiny account so the first order eats the whole budget.
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT")],
        signals={**_signals("AAPL", 10.0), **_signals("MSFT", 10.0)},
        account={"equity": 100.0, "max_position_pct": 0.90},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    total = sum(o.quantity * o.limit_price for o in orders)
    assert total <= 100.0 + 1e-9
    assert len(orders) == 1  # second order would breach the cap


def test_pdt_restriction_blocks_buys_allows_sells():
    # Rolling 5 business days ending Mon 2026-08-31 (inclusive):
    # 08-31, 08-28, 08-27, 08-26, 08-25.
    # Restriction blocks risk-ADDING buys; risk-reducing sells stay allowed.
    three_trades = ["2026-08-25", "2026-08-26", "2026-08-28"]
    signals = {**_signals("AAPL", 250.0, signal="BUY"),
               **_signals("MSFT", 400.0, signal="SELL")}
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=three_trades,
        as_of="2026-08-31",
        positions={"MSFT": 10.0},
    )
    assert [(o.ticker, o.side) for o in orders] == [("MSFT", "sell")]


def test_pdt_restriction_blocks_buy_even_with_holding():
    three_trades = ["2026-08-25", "2026-08-26", "2026-08-28"]
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=_ACCOUNT,
        day_trade_dates=three_trades,
        as_of="2026-08-31",
        positions={"AAPL": 100.0},
    )
    assert orders == []


def test_pdt_allows_when_under_limit():
    two_trades = ["2026-08-25", "2026-08-26"]
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=two_trades,
        as_of="2026-08-31",
        positions={"AAPL": 5.0},
    )
    assert len(orders) == 1 and orders[0].side == "sell"


def test_old_day_trades_do_not_count():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=["2026-08-10", "2026-08-11", "2026-08-12"],
        as_of="2026-08-31",
        positions={"AAPL": 5.0},
    )
    assert len(orders) == 1


def test_orders_are_valid_orderplans():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert all(isinstance(o, OrderPlan) for o in orders)


# ---------------------------------------------------------------------------
# Holdings-aware SELL sizing (concerns.txt HIGH #1)
# ---------------------------------------------------------------------------


def test_sell_capped_at_held_quantity():
    # budget/price would be 40 shares; only 12.5 held -> sell exactly 12.5.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 12.5},
    )
    assert len(orders) == 1
    assert orders[0].side == "sell"
    assert orders[0].quantity == 12.5


def test_sell_takes_full_budget_when_held_exceeds_budget():
    # held 100 > budget/price 40 -> partial exit sized to budget.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 100.0},
    )
    assert orders[0].quantity == 40.0


def test_sell_skipped_when_nothing_held():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={},
    )
    assert orders == []


def test_sell_skipped_without_positions_data():
    # No positions mapping at all -> conservative: never emit a sell.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders == []


def test_sell_position_lookup_is_case_insensitive():
    orders = build_order_plan(
        decisions=[_decision("aapl")],
        signals=_signals("aapl", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 3.0},
    )
    assert len(orders) == 1 and orders[0].quantity == 3.0


def test_sell_notional_does_not_consume_buy_budget():
    # SELL does not add to the cumulative cap; the later BUY gets the full
    # tiny-account budget instead of being crowded out by the sell.
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT")],
        signals={**_signals("AAPL", 10.0, signal="SELL"),
                 **_signals("MSFT", 10.0, signal="BUY")},
        account={"equity": 100.0, "max_position_pct": 0.90},
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 50.0},
    )
    assert [(o.ticker, o.side) for o in orders] == \
        [("AAPL", "sell"), ("MSFT", "buy")]
    assert orders[0].quantity == 9.0  # min(90/10, 50)
    assert orders[1].quantity == 9.0  # full 90 budget still available


def test_holdings_cap_survives_invalid_position_rows():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": "not-a-number", "BAD": None},
    )
    assert orders == []


# ---------------------------------------------------------------------------
# Volatility + conviction scaling (concerns.txt HIGH #2 / MEDIUM #3)
# ---------------------------------------------------------------------------


def test_high_volatility_scales_budget_down():
    # daily_vol 0.04 vs default target 0.02 -> x0.5 -> qty 20 (not 40).
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.95,
                        "reference_price": 250.0, "daily_vol": 0.04}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 20.0
    assert "vol x0.50" in orders[0].rationale


def test_low_volatility_never_scales_up():
    # 0.02/0.005 = 4.0 but the factor is capped at 1.0 (scale-down only).
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.95,
                        "reference_price": 250.0, "daily_vol": 0.005}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 40.0


def test_target_daily_vol_from_account():
    account = {**_ACCOUNT, "target_daily_vol": 0.01}
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.95,
                        "reference_price": 250.0, "daily_vol": 0.04}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 10.0  # 0.01/0.04 = x0.25 -> 2500/250


def test_lower_confidence_scales_budget_down():
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.475,
                        "reference_price": 250.0}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # conviction = clamp(0.475/0.95, 0.25, 1.0) = 0.5 -> 5000/250
    assert orders[0].quantity == 20.0


def test_conviction_weight_has_floor():
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.01,
                        "reference_price": 250.0}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # weight = max(0.25, 0.01/0.95) = 0.25 -> 2500/250
    assert orders[0].quantity == 10.0


def test_missing_confidence_keeps_base_budget():
    signals = {"AAPL": {"signal": "BUY", "reference_price": 250.0}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 40.0


def test_vol_and_conviction_multiply():
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.475,
                        "reference_price": 250.0, "daily_vol": 0.04}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # budget = 10000 * 0.5 (vol) * 0.5 (conviction) = 2500 -> qty 10
    assert orders[0].quantity == 10.0


# ---------------------------------------------------------------------------
# Sector concentration cap (concerns.txt MEDIUM #4)
# ---------------------------------------------------------------------------


def test_sector_cap_skips_breaching_orders():
    # Default max_sector_pct=0.30 -> $30k per sector on the $100k account.
    # Budget is $10k/ticker: AAPL+MSFT+AMD reach exactly $30k (allowed),
    # TSLA would breach and is skipped; JNJ is a different sector.
    sector_map = {"AAPL": "Tech", "MSFT": "Tech", "AMD": "Tech",
                  "TSLA": "Tech", "JNJ": "Health"}
    signals = {**_signals("AAPL", 10.0), **_signals("MSFT", 10.0),
               **_signals("AMD", 10.0), **_signals("TSLA", 10.0),
               **_signals("JNJ", 10.0)}
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAPL", "MSFT", "AMD", "TSLA",
                                          "JNJ")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        sector_map=sector_map,
    )
    assert [(o.ticker, o.side) for o in orders] == [
        ("AAPL", "buy"), ("MSFT", "buy"), ("AMD", "buy"), ("JNJ", "buy"),
    ]
    tech_total = sum(o.quantity * o.limit_price for o in orders
                     if o.ticker in {"AAPL", "MSFT", "AMD"})
    assert tech_total == 30_000.0  # exactly-equal spend allowed


def test_sector_cap_respects_account_override():
    account = {**_ACCOUNT, "max_sector_pct": 0.15}
    sector_map = {"AAPL": "Tech", "MSFT": "Tech"}
    signals = {**_signals("AAPL", 10.0), **_signals("MSFT", 10.0)}
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT")],
        signals=signals,
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
        sector_map=sector_map,
    )
    # AAPL spends $10k; MSFT would take Tech to $20k > $15k -> skipped.
    assert [o.ticker for o in orders] == ["AAPL"]


def test_sector_cap_exempts_unlabeled_tickers():
    sector_map = {"AAPL": "Tech", "MSFT": "Tech", "AMD": "Tech",
                  "TSLA": "Tech", "SPY": None}
    signals = {**_signals("AAPL", 10.0), **_signals("MSFT", 10.0),
               **_signals("AMD", 10.0), **_signals("TSLA", 10.0),
               **_signals("SPY", 10.0)}
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAPL", "MSFT", "AMD", "TSLA",
                                          "SPY")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        sector_map=sector_map,
    )
    # TSLA breaches the Tech cap; SPY carries no sector label -> exempt.
    assert [o.ticker for o in orders] == ["AAPL", "MSFT", "AMD", "SPY"]


def test_no_sector_map_means_no_sector_cap():
    # Four same-"sector" tickers are only limited by the equity cap when no
    # sector labels are provided (e.g. fundamentals fetch degraded).
    signals = {t: _signals(t, 10.0)[t]
               for t in ("AAPL", "MSFT", "AMD", "TSLA", "GOOGL", "META")}
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAPL", "MSFT", "AMD", "TSLA",
                                          "GOOGL", "META")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 6


def test_sector_sells_bypass_cap_but_stay_holding_capped():
    sector_map = {"AAPL": "Tech"}
    signals = {"AAPL": _signals("AAPL", 10.0, signal="SELL")["AAPL"]}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 5.0},
        sector_map=sector_map,
    )
    assert len(orders) == 1 and orders[0].side == "sell"
    assert orders[0].quantity == 5.0  # capped by holdings, not sector/budget


# ---------------------------------------------------------------------------
# Regime gating (global liquidity & fear layer)
# ---------------------------------------------------------------------------


def test_panic_regime_blocks_buys_allows_sells():
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT")],
        signals={**_signals("AAPL", 10.0, signal="BUY"),
                 **_signals("MSFT", 10.0, signal="SELL")},
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"MSFT": 50.0},
        regime={"regime": "panic"},
    )
    assert [(o.ticker, o.side) for o in orders] == [("MSFT", "sell")]
    # Panic gate is active but leaves sizing untouched (x1.00 in rationale).
    assert "regime panic x1.00" in orders[0].rationale


def test_panic_regime_missing_leaves_buys_untouched():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1


def test_invalid_regime_snapshot_is_fail_open():
    for junk in (None, "junk", {"regime": "unknown"}, {}, 42):
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 10.0, signal="BUY"),
            account=_ACCOUNT,
            day_trade_dates=[],
            as_of="2026-08-31",
            regime=junk,
        )
        assert len(orders) == 1, junk


def test_risk_off_scales_budget_down():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        regime={"regime": "risk_off"},
    )
    assert orders[0].quantity == 20.0  # 10000 * 0.5 / 250
    assert "regime risk_off x0.50" in orders[0].rationale


def test_risk_off_multiplier_is_configurable():
    account = {**_ACCOUNT, "regime_budget_mult": 0.25}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
        regime={"regime": "risk_off"},
    )
    assert orders[0].quantity == 10.0


def test_risk_off_multiplier_never_scales_up():
    account = {**_ACCOUNT, "regime_budget_mult": 4.0}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
        regime={"regime": "risk_off"},
    )
    assert orders[0].quantity == 40.0  # clamped to 1.0


def test_risk_off_also_descales_sells():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 50.0},
        regime={"regime": "risk_off"},
    )
    # SELL sized to min(budget/price, held) = min(5000/10=500, 50) under x0.5.
    assert orders[0].quantity == 50.0


def test_regime_risk_off_with_zero_mult_blocks_all_buys():
    account = {**_ACCOUNT, "regime_budget_mult": 0.0}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
        regime={"regime": "risk_off"},
    )
    assert orders == []


def test_regime_combines_with_vol_and_conviction():
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.475,
                        "reference_price": 250.0, "daily_vol": 0.04}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        regime={"regime": "risk_off"},
    )
    # 10000 * 0.5(vol) * 0.5(conv) * 0.5(regime) = 1250 -> 1250/250 = qty 5
    assert orders[0].quantity == 5.0


def test_pdt_and_panic_gates_compose():
    three_trades = ["2026-08-25", "2026-08-26", "2026-08-28"]
    orders = build_order_plan(
        decisions=[_decision("AAPL"), _decision("MSFT")],
        signals={**_signals("AAPL", 10.0, signal="BUY"),
                 **_signals("MSFT", 10.0, signal="SELL")},
        account=_ACCOUNT,
        day_trade_dates=three_trades,
        as_of="2026-08-31",
        positions={"MSFT": 3.0},
        regime={"regime": "panic"},
    )
    # Both gates block the buy; the sell passes either way.
    assert [(o.ticker, o.side) for o in orders] == [("MSFT", "sell")]


# ---------------------------------------------------------------------------
# Debate-winner conviction (item 10)
# ---------------------------------------------------------------------------

from src.agents.portfolio_manager import (  # noqa: E402
    DEFAULT_DEBATE_OPPOSE_MULT,
    _debate_multiplier,
)


def test_bear_winner_shrinks_buy():
    debates = {"AAPL": {"winner": "bear"}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        debates=debates,
    )
    # 10000 * 0.75 / 250 = 30
    assert orders[0].quantity == 30.0
    assert "debate bear x0.75" in orders[0].rationale


def test_bull_winner_shrinks_sell():
    debates = {"AAPL": {"winner": "bull"}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 5000.0},
        debates=debates,
    )
    # min(10000 * 0.75 / 10, 5000) = 750
    assert orders[0].quantity == 750.0
    assert "debate bull x0.75" in orders[0].rationale


def test_bull_winner_leaves_buy_untouched():
    debates = {"AAPL": {"winner": "bull"}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        debates=debates,
    )
    assert orders[0].quantity == 40.0
    assert "debate" not in orders[0].rationale


def test_undecided_or_absent_debate_is_noop():
    for debates in (None, {}, {"AAPL": {"winner": "undecided"}},
                    {"AAPL": {}}, {"MSFT": {"winner": "bear"}}):
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 250.0, signal="BUY"),
            account=_ACCOUNT,
            day_trade_dates=[],
            as_of="2026-08-31",
            debates=debates,
        )
        assert orders[0].quantity == 40.0, debates
        assert "debate" not in orders[0].rationale, debates


def test_debate_multiplier_is_configurable():
    account = {**_ACCOUNT, "debate_oppose_mult": 0.5}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
        debates={"AAPL": {"winner": "bear"}},
    )
    assert orders[0].quantity == 20.0  # 10000 * 0.5 / 250


def test_debate_multiplier_clamped_both_ways():
    # Above 1.0 clamps down to 1.0 (no token: nothing contributed).
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account={**_ACCOUNT, "debate_oppose_mult": 4.0},
        day_trade_dates=[],
        as_of="2026-08-31",
        debates={"AAPL": {"winner": "bear"}},
    )
    assert orders[0].quantity == 40.0
    assert "debate" not in orders[0].rationale
    # Below the 0.25 floor clamps up.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account={**_ACCOUNT, "debate_oppose_mult": 0.0},
        day_trade_dates=[],
        as_of="2026-08-31",
        debates={"AAPL": {"winner": "bear"}},
    )
    assert orders[0].quantity == 10.0  # 10000 * 0.25 / 250
    assert "debate bear x0.25" in orders[0].rationale


def test_debate_record_duck_typed_object_or_dict():
    class _Debate:
        winner = "bear"

    assert _debate_multiplier(_Debate(), "buy", DEFAULT_DEBATE_OPPOSE_MULT) == 0.75
    assert _debate_multiplier({"winner": "bear"}, "buy",
                              DEFAULT_DEBATE_OPPOSE_MULT) == 0.75


def test_debate_multiplier_unit_rules():
    m = DEFAULT_DEBATE_OPPOSE_MULT
    # Bear winner: adverse for buys, favorable for sells.
    assert _debate_multiplier({"winner": "bear"}, "buy", m) == m
    assert _debate_multiplier({"winner": "bear"}, "sell", m) == 1.0
    # Bull winner: mirrored.
    assert _debate_multiplier({"winner": "bull"}, "sell", m) == m
    assert _debate_multiplier({"winner": "bull"}, "buy", m) == 1.0
    # Junk winners / junk records fail open.
    for junk in ({"winner": "undecided"}, {"winner": "BULL"},
                 {"winner": None}, {}, None, "junk", 42):
        assert _debate_multiplier(junk, "buy", m) == 1.0, junk
    # Junk multiplier config falls back to the default.
    for junk_mult in (None, "junk", float("nan"), float("inf")):
        mult = _debate_multiplier({"winner": "bear"}, "buy", junk_mult)
        assert mult == m, junk_mult


def test_debate_composes_with_regime_and_vol():
    signals = {"AAPL": {"signal": "BUY", "confidence": 0.475,
                        "reference_price": 250.0, "daily_vol": 0.04}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=signals,
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        regime={"regime": "risk_off"},
        debates={"AAPL": {"winner": "bear"}},
    )
    # 10000 * 0.5(vol) * 0.5(conv) * 0.5(regime) * 0.75(debate) = 937.5
    assert orders[0].quantity == 3.75
    assert "debate bear x0.75" in orders[0].rationale


# ---------------------------------------------------------------------------
# Transaction costs (item 9)
# ---------------------------------------------------------------------------

from src.agents.portfolio_manager import (  # noqa: E402
    ALPACA_MIN_NOTIONAL,
    DEFAULT_COST_BUFFER_PCT,
    DEFAULT_MIN_TRADE_NOTIONAL,
    DEFAULT_MIN_TRADE_NOTIONAL_PCT,
    _cost_multiplier,
)


def test_cost_buffer_default_is_noop():
    assert DEFAULT_COST_BUFFER_PCT == 0.0
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 40.0  # unchanged base sizing
    assert "cost" not in orders[0].rationale


def test_cost_buffer_scales_budget_down():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account={**_ACCOUNT, "cost_buffer_pct": 0.10},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # 10000 * 0.90 / 250 = 36
    assert orders[0].quantity == 36.0
    assert "cost x0.900" in orders[0].rationale


def test_cost_buffer_applies_to_sells():
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="SELL"),
        account={**_ACCOUNT, "cost_buffer_pct": 0.50},
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 5000.0},
    )
    # min(10000 * 0.50 / 10, 5000) = 500
    assert orders[0].quantity == 500.0


def test_cost_buffer_clamped_to_unit_interval():
    # Invalid types fail open to no-op; finite values clamp into [0, 1].
    for junk in (None, "junk", float("nan"), float("inf"), True):
        assert _cost_multiplier(junk) == 1.0, junk
    assert _cost_multiplier(-1.0) == 1.0   # clamps to 0 buffer
    assert _cost_multiplier(4.0) == 0.0    # clamps to full buffer
    assert _cost_multiplier(1.0) == 0.0    # explicit full buffer allowed
    assert _cost_multiplier(0.25) == 0.75


def test_min_notional_skips_dust_buys():
    # Budget 1000 * 0.10 * 0.05 = 5.0 notional @ 250 -> below the 10.0 floor.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0, signal="BUY"),
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "cost_buffer_pct": 0.95},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders == []


def test_min_notional_skips_dust_sells():
    # Held 0.0005 shares @ 10.0 -> 0.005 notional: dust exit skipped.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="SELL"),
        account=_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 0.0005},
    )
    assert orders == []


def test_min_notional_configurable_and_zero_disables():
    # Floor 0.0 disables the filter: the same dust order now passes.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="SELL"),
        account={**_ACCOUNT, "min_trade_notional": 0.0},
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 0.0005},
    )
    assert orders[0].quantity == 0.0005
    # Junk falls back to the default floor (which skips the dust order).
    for junk in (None, "junk", float("nan"), float("inf"), -5.0):
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 10.0, signal="SELL"),
            account={**_ACCOUNT, "min_trade_notional": junk},
            day_trade_dates=[],
            as_of="2026-08-31",
            positions={"AAPL": 0.0005},
        )
        assert orders == [], junk
    assert DEFAULT_MIN_TRADE_NOTIONAL == 10.0


def test_min_notional_boundary_is_inclusive():
    # Exactly the effective floor passes; strictly below is dust.
    # Large account: effective floor = min(10, max(1, 100000*0.01)) = 10.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account={"equity": 10_000.0, "max_position_pct": 0.001},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1  # 1.0 shares * 10.0 = 10.0 notional == floor
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account={"equity": 9_999.0, "max_position_pct": 0.001},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders == []  # 0.9999 shares * 10.0 = 9.999 notional < floor


def test_min_notional_floor_scales_down_with_equity():
    # $500 account: effective floor = min(10, max(1, 500*0.01)) = 5.
    # Budget 500*0.10 = 50; cost buffer 0.89 -> 5.5 notional >= floor.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account={"equity": 500.0, "max_position_pct": 0.10,
                 "cost_buffer_pct": 0.89},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1  # 5.5 notional clears the scaled floor
    assert abs(orders[0].quantity - 0.55) < 1e-5
    # Below the scaled floor is dust: buffer 0.91 -> 4.5 < 5.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account={"equity": 500.0, "max_position_pct": 0.10,
                 "cost_buffer_pct": 0.91},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders == []


def test_min_notional_floor_never_below_alpaca_minimum():
    # $100 account: relative floor = max(1, 100*0.01) = 1.0; budget 10.
    # Cost buffer 0.95 -> 0.5 notional: below the $1 hard floor, skipped.
    # Cost buffer 0.80 -> 2.0 notional: clears the floor, passes.
    for buffer, expected_len in ((0.95, 0), (0.80, 1)):
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 10.0, signal="BUY"),
            account={"equity": 100.0, "max_position_pct": 0.10,
                     "cost_buffer_pct": buffer},
            day_trade_dates=[],
            as_of="2026-08-31",
        )
        assert len(orders) == expected_len, buffer


def test_min_notional_pct_configurable_and_junk_falls_back():
    # pct 0.0 -> relative floor max(1, 0) = 1 -> effective floor 1 at a
    # $1k account: a 5.0-notional order the absolute floor 10 would skip
    # now trades.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "min_trade_notional_pct": 0.0, "cost_buffer_pct": 0.95},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 1  # budget 1000*0.10*0.05 = 5.0 >= floor 1.0
    # Junk pct falls back to the default 0.01 (floor 10 at $1k): skipped.
    for junk in (None, "junk", float("nan"), float("inf"), -0.5):
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 10.0, signal="BUY"),
            account={"equity": 1000.0, "max_position_pct": 0.10,
                     "min_trade_notional_pct": junk, "cost_buffer_pct": 0.95},
            day_trade_dates=[],
            as_of="2026-08-31",
        )
        assert orders == [], junk
    assert DEFAULT_MIN_TRADE_NOTIONAL_PCT == 0.01
    assert ALPACA_MIN_NOTIONAL == 1.0


def test_absolute_floor_above_default_is_strict_override():
    # The RUNBOOK kill switch (MIN_TRADE_NOTIONAL=1000000) must keep
    # winning over the equity-relative scaling: the effective floor is
    # the explicit absolute, so every order is skipped as dust.
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="BUY"),
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "min_trade_notional": 1000000.0},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders == []


def test_dust_exit_now_possible_on_small_account():
    # The old absolute floor trapped tiny exits forever (SELL notional
    # < $10 skipped); the scaled floor lets a $100 account exit a
    # 0.2-share position ($2 notional).
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 10.0, signal="SELL"),
        account={"equity": 100.0, "max_position_pct": 0.10},
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAPL": 0.2},
    )
    assert len(orders) == 1
    assert orders[0].quantity == 0.2
    assert orders[0].side == "sell"


# ---------------------------------------------------------------------------
# Correlation-cluster cap (item 11)
# ---------------------------------------------------------------------------

_CLUSTER = "corr:AAA, BBB, CCC"


def _cluster_signals(*tickers, price=10.0):
    signals = {}
    for t in tickers:
        signals.update(_signals(t, price))
    return signals


def test_corr_cluster_cap_blocks_breaching_buy():
    # Budget 1000 * 0.10 = 100 notional each; cluster cap 0.25 -> 250:
    # two buys fit (200), the third would hit 300 and is skipped.
    cluster_map = {t: _CLUSTER for t in ("AAA", "BBB", "CCC")}
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAA", "BBB", "CCC")],
        signals=_cluster_signals("AAA", "BBB", "CCC"),
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "max_corr_pct": 0.25},
        day_trade_dates=[],
        as_of="2026-08-31",
        cluster_map=cluster_map,
    )
    assert [(o.ticker, o.side) for o in orders] == [("AAA", "buy"),
                                                    ("BBB", "buy")]


def test_corr_cluster_cap_absent_map_is_noop():
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAA", "BBB", "CCC")],
        signals=_cluster_signals("AAA", "BBB", "CCC"),
        account={"equity": 1000.0, "max_position_pct": 0.10},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 3


def test_corr_cluster_cap_exempts_unmapped_tickers():
    cluster_map = {"AAA": _CLUSTER, "BBB": _CLUSTER}  # CCC unmapped
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAA", "BBB", "CCC")],
        signals=_cluster_signals("AAA", "BBB", "CCC"),
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "max_corr_pct": 0.15},  # 150: only one cluster buy fits
        day_trade_dates=[],
        as_of="2026-08-31",
        cluster_map=cluster_map,
    )
    # AAA fills the cluster budget; BBB breaches it; CCC is exempt.
    assert [o.ticker for o in orders] == ["AAA", "CCC"]


def test_corr_cluster_cap_junk_pct_is_noop():
    cluster_map = {t: _CLUSTER for t in ("AAA", "BBB", "CCC")}
    for junk in ("junk", float("nan"), 0.0, -1.0, None):
        orders = build_order_plan(
            decisions=[_decision(t) for t in ("AAA", "BBB", "CCC")],
            signals=_cluster_signals("AAA", "BBB", "CCC"),
            account={"equity": 1000.0, "max_position_pct": 0.10,
                     "max_corr_pct": junk},
            day_trade_dates=[],
            as_of="2026-08-31",
            cluster_map=cluster_map,
        )
        assert len(orders) == 3, junk


def test_corr_cluster_cap_ignores_sells():
    cluster_map = {t: _CLUSTER for t in ("AAA", "BBB", "CCC")}
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAA", "BBB", "CCC")],
        signals={t: {"signal": "SELL", "confidence": 0.95,
                     "reference_price": 10.0}
                 for t in ("AAA", "BBB", "CCC")},
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "max_corr_pct": 0.05},
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"AAA": 10.0, "BBB": 10.0, "CCC": 10.0},
        cluster_map=cluster_map,
    )
    # Exits always pass: held-qty cap is the only sell-side limiter.
    assert [(o.ticker, o.side) for o in orders] == [
        ("AAA", "sell"), ("BBB", "sell"), ("CCC", "sell")]


def test_corr_and_sector_caps_compose_independently():
    cluster_map = {"AAA": _CLUSTER, "BBB": _CLUSTER}  # CCC in no cluster
    sector_map = {"AAA": "Tech", "BBB": "Tech", "CCC": "Tech"}
    orders = build_order_plan(
        decisions=[_decision(t) for t in ("AAA", "BBB", "CCC")],
        signals=_cluster_signals("AAA", "BBB", "CCC"),
        account={"equity": 1000.0, "max_position_pct": 0.10,
                 "max_corr_pct": 0.15,   # cluster: 150
                 "max_sector_pct": 0.25},  # sector: 250
        day_trade_dates=[],
        as_of="2026-08-31",
        cluster_map=cluster_map,
        sector_map=sector_map,
    )
    # AAA: 100 (cluster 100/150, sector 100/250) ok.
    # BBB: cluster would hit 200 > 150 -> skipped.
    # CCC: exempt from cluster; sector 200 <= 250 -> ok.
    assert [o.ticker for o in orders] == ["AAA", "CCC"]


# ---------------------------------------------------------------------------
# Batch-level inverse-vol risk parity (item 8)
# ---------------------------------------------------------------------------


def _vol_signals(vols, signal="BUY", price=250.0) -> dict:
    """Signals keyed by ticker with per-ticker daily_vol."""
    out = {}
    for ticker, vol in vols.items():
        out[ticker] = {"signal": signal, "confidence": 0.95,
                       "reference_price": price, "daily_vol": vol}
    return out


# Neutralizes the vol_mult factor (target far above every fixture vol) so
# the parity tests exercise parity math in isolation.
_PARITY_ACCOUNT = {**_ACCOUNT, "target_daily_vol": 0.10}


def test_parity_needs_two_valid_vol_buys():
    # Single BUY candidate -> no-op even with a valid vol.
    orders = build_order_plan(
        decisions=[_decision("AAA")],
        signals=_vol_signals({"AAA": 0.04}),
        account=_PARITY_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 40.0
    assert "invol" not in orders[0].rationale
    # Two candidates but only one carries a vol -> no-op as well.
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB")],
        signals={**_vol_signals({"AAA": 0.04}),
                 "BBB": {"signal": "BUY", "confidence": 0.95,
                         "reference_price": 250.0}},
        account=_PARITY_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert [o.quantity for o in orders] == [40.0, 40.0]
    assert all("invol" not in o.rationale for o in orders)


def test_parity_equalizes_risk_contributions():
    # AAA vol 0.01, BBB vol 0.04 -> weights 0.8 / 0.2 of the 20000 total.
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB")],
        signals=_vol_signals({"AAA": 0.01, "BBB": 0.04}),
        account=_PARITY_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    by_ticker = {o.ticker: o for o in orders}
    # Parity caps: AAA 16000 (>= factor 10000 -> not binding), BBB 4000.
    assert by_ticker["AAA"].quantity == 40.0      # factor budget binds
    assert by_ticker["BBB"].quantity == 16.0      # 4000 / 250
    assert "invol x0.40" in by_ticker["BBB"].rationale
    assert "invol" not in by_ticker["AAA"].rationale


def test_parity_uses_factor_budgets_not_base():
    # BBB carries a low quant conviction (factor budget 5000); parity must
    # redistribute the POST-factor total, not the raw base budgets.
    signals = {
        "AAA": {"signal": "BUY", "confidence": 0.95, "reference_price": 250.0,
                "daily_vol": 0.01},
        "BBB": {"signal": "BUY", "confidence": 0.475, "reference_price": 250.0,
                "daily_vol": 0.04},
    }
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB")],
        signals=signals,
        account=_PARITY_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    by_ticker = {o.ticker: o for o in orders}
    # Factor budgets: AAA 10000, BBB 5000 -> total 15000.
    # Weights: (1/0.01)=100, (1/0.04)=25 -> AAA 12000, BBB 3000.
    assert by_ticker["AAA"].quantity == 40.0   # parity above factor -> cap
    assert by_ticker["BBB"].quantity == 12.0   # 3000 / 250 (vs 20 uncapped)
    assert "invol x0.60" in by_ticker["BBB"].rationale


def test_parity_disabled_is_noop():
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB")],
        signals=_vol_signals({"AAA": 0.01, "BBB": 0.04}),
        account={**_PARITY_ACCOUNT, "invol_enabled": False},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert [o.quantity for o in orders] == [40.0, 40.0]
    assert all("invol" not in o.rationale for o in orders)


def test_parity_ignores_sells_and_gated_names():
    # BBB is a SELL: excluded from weights.  CCC is HOLD: never a
    # candidate, so parity runs only over AAA (single candidate -> no-op).
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB"), _decision("CCC")],
        signals={**_vol_signals({"AAA": 0.04}),
                 "BBB": {"signal": "SELL", "confidence": 0.95,
                         "reference_price": 250.0, "daily_vol": 0.01},
                 "CCC": {"signal": "HOLD", "confidence": 0.95,
                         "reference_price": 250.0, "daily_vol": 0.10}},
        account=_PARITY_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
        positions={"BBB": 100.0},
    )
    by_ticker = {o.ticker: o for o in orders}
    assert by_ticker["AAA"].quantity == 40.0   # lone valid buy -> no parity
    assert by_ticker["BBB"].side == "sell"
    assert set(by_ticker) == {"AAA", "BBB"}


def test_parity_zero_vol_is_not_a_candidate():
    # daily_vol 0.0 is invalid (non-positive) -> treated like missing.
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB")],
        signals={**_vol_signals({"AAA": 0.04}),
                 "BBB": {"signal": "BUY", "confidence": 0.95,
                         "reference_price": 250.0, "daily_vol": 0.0}},
        account=_PARITY_ACCOUNT,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert [o.quantity for o in orders] == [40.0, 40.0]


def test_parity_interacts_with_cumulative_cap():
    # Tiny equity: parity budgets sum far above equity; the cumulative
    # cap still governs actual spend (parity is an additional cap only).
    orders = build_order_plan(
        decisions=[_decision("AAA"), _decision("BBB")],
        signals=_vol_signals({"AAA": 0.02, "BBB": 0.02}, price=100.0),
        account={"equity": 100.0, "max_position_pct": 0.90},
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    total = sum(o.quantity * o.limit_price for o in orders)
    assert total <= 100.0 + 1e-9
    assert len(orders) == 1  # second order breaches equity as before


# ---------------------------------------------------------------------------
# Kelly sizing (item 7)
# ---------------------------------------------------------------------------

from src.agents.portfolio_manager import (  # noqa: E402
    DEFAULT_KELLY_FRACTION_CAP,
    DEFAULT_KELLY_MAX_MULT,
    _kelly_multiplier,
)

import pytest  # noqa: E402


def test_kelly_absent_is_noop():
    for account in (_ACCOUNT, {**_ACCOUNT, "kelly": None}):
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 250.0),
            account=account,
            day_trade_dates=[],
            as_of="2026-08-31",
        )
        assert orders[0].quantity == 40.0, account
        assert "kelly" not in orders[0].rationale, account


def test_kelly_positive_fraction_scales_up():
    # fraction 0.125 = cap/2 -> mult 1.25 -> budget 12500 -> qty 50.
    account = {**_ACCOUNT, "kelly": {"fraction": 0.125, "n": 100}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 50.0
    assert "kelly x1.25 (n=100)" in orders[0].rationale


def test_kelly_full_fraction_hits_max_mult():
    account = {**_ACCOUNT, "kelly": {"fraction": 0.25, "n": 100}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # 10000 * 1.5 = 15000 == the hard per-name cap: exactly allowed.
    assert orders[0].quantity == 60.0
    assert "kelly x1.50" in orders[0].rationale


def test_kelly_above_cap_clamps():
    account = {**_ACCOUNT, "kelly": {"fraction": 0.9, "n": 100}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 60.0
    assert "kelly x1.50" in orders[0].rationale


def test_kelly_nonpositive_fraction_is_inactive():
    for fraction in (0.0, -0.1):
        account = {**_ACCOUNT, "kelly": {"fraction": fraction, "n": 100}}
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 250.0),
            account=account,
            day_trade_dates=[],
            as_of="2026-08-31",
        )
        assert orders[0].quantity == 40.0, fraction
        assert "kelly" not in orders[0].rationale, fraction


def test_kelly_junk_is_inactive():
    for junk in ("junk", {}, {"fraction": "junk"}, {"fraction": True},
                 {"fraction": float("nan")}):
        account = {**_ACCOUNT, "kelly": junk}
        orders = build_order_plan(
            decisions=[_decision("AAPL")],
            signals=_signals("AAPL", 250.0),
            account=account,
            day_trade_dates=[],
            as_of="2026-08-31",
        )
        assert orders[0].quantity == 40.0, junk
        assert "kelly" not in orders[0].rationale, junk


def test_kelly_valid_fraction_with_junk_count_still_active():
    # The fraction carries the edge; an unparseable n degrades to None
    # in the rationale but the multiplier still applies.
    account = {**_ACCOUNT, "kelly": {"fraction": 0.1, "n": "junk"}}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 48.0  # 10000 * 1.2 / 250
    assert "kelly x1.20 (n=None)" in orders[0].rationale


def test_kelly_max_mult_configurable():
    account = {**_ACCOUNT, "kelly": {"fraction": 0.25, "n": 50},
               "kelly_max_mult": 2.0}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # 10000 * 2.0 = 20000 == hard cap (equity * 0.10 * 2.0): allowed.
    assert orders[0].quantity == 80.0
    assert "kelly x2.00" in orders[0].rationale


def test_kelly_max_mult_below_one_neutralizes_scale_up():
    account = {**_ACCOUNT, "kelly": {"fraction": 0.25, "n": 50},
               "kelly_max_mult": 0.5}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_signals("AAPL", 250.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert orders[0].quantity == 40.0
    assert "kelly" not in orders[0].rationale


def test_kelly_unit_mapping():
    m = _kelly_multiplier({"fraction": 0.125, "n": 10}, 0.25, 1.5)
    assert m == {"mult": 1.25, "fraction": 0.125, "n": 10}
    assert _kelly_multiplier({"fraction": 0.25, "n": 10}, 0.25, 1.5)["mult"] == 1.5
    assert _kelly_multiplier({"fraction": 9.0, "n": 10}, 0.25, 1.5)["mult"] == 1.5
    assert _kelly_multiplier(None, 0.25, 1.5) is None
    assert _kelly_multiplier({}, 0.25, 1.5) is None
    # Junk caps/ceilings fall back to defaults; ceiling < 1 clamps to 1.0.
    assert _kelly_multiplier({"fraction": 0.1}, "junk", "junk")["mult"] \
        == pytest.approx(1.0 + 0.5 * (0.1 / 0.25))
    assert _kelly_multiplier({"fraction": 0.1}, 0.25, float("nan"))["mult"] == 1.0
    assert DEFAULT_KELLY_FRACTION_CAP == 0.25
    assert DEFAULT_KELLY_MAX_MULT == 1.5


def test_kelly_composes_with_scale_down_factors():
    account = {**_ACCOUNT, "kelly": {"fraction": 0.125, "n": 100},
               "target_daily_vol": 0.10}
    orders = build_order_plan(
        decisions=[_decision("AAPL")],
        signals=_vol_signals({"AAPL": 0.04}),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    # vol_mult = min(1, 0.10/0.04) = 1.0 -> budget = 10000 * 1.25 = 12500.
    assert orders[0].quantity == 50.0
    assert "kelly x1.25" in orders[0].rationale


def test_kelly_scale_up_still_respects_cumulative_equity_cap():
    # 8 names at kelly-max each want 15% of equity; the 100% cumulative
    # BUY cap stops the seventh.
    account = {"equity": 1000.0, "max_position_pct": 0.10,
               "kelly": {"fraction": 0.25, "n": 100}}
    tickers = [f"T{i}" for i in range(8)]
    orders = build_order_plan(
        decisions=[_decision(t) for t in tickers],
        signals=_cluster_signals(*tickers, price=10.0),
        account=account,
        day_trade_dates=[],
        as_of="2026-08-31",
    )
    assert len(orders) == 6  # 6 * 150 = 900 fits; 7th would hit 1050
    total = sum(o.quantity * o.limit_price for o in orders)
    assert total <= 1000.0 + 1e-9
