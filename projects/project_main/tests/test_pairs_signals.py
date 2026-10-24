"""Tests for pair signal generation (src/pairs/signals.py).

Deterministic spread paths with known z-scores exercise every rule of
the long-only signal table (entry both directions, reversion exit,
stop loss, hold), plus the fail-open degradation paths.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.pairs.book import BookEntry
from src.pairs.signals import generate_pair_signal, generate_pair_signals


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
        formation_corr=0.9,
    )
    payload.update(overrides)
    return BookEntry(**payload)


def _prices(price_a: float, price_b: float) -> dict:
    return {"AAA": price_a, "BBB": price_b}


# With alpha=0, beta=1, mean=0, std=0.02: z = (log A - log B) / 0.02.


def _z_of(price_a: float, price_b: float) -> float:
    return (math.log(price_a) - math.log(price_b)) / 0.02


def test_entry_rule_buys_the_cheap_leg():
    # z = +2.5 -> A expensive, B cheap -> BUY B.
    pa = 105.0
    pb = pa / math.exp(2.5 * 0.02)
    signal = generate_pair_signal(_entry(), _prices(pa, pb))
    assert signal is not None
    assert signal.signal_a == "SELL_EXIT"
    assert signal.signal_b == "BUY"
    assert signal.pair_id == "AAA/BBB"
    assert signal.z_score == pytest.approx(_z_of(pa, pb), abs=1e-9)
    # |z| = 2.5 exceeds the entry reference -> confidence clamps to 1.0.
    assert signal.confidence == 1.0

    # Mirrored: z = -2.5 -> A cheap -> BUY A.
    pb2 = 105.0
    pa2 = pb2 / math.exp(2.5 * 0.02)
    signal2 = generate_pair_signal(_entry(), _prices(pa2, pb2))
    assert signal2 is not None
    assert signal2.signal_a == "BUY"
    assert signal2.signal_b == "SELL_EXIT"


def test_hold_inside_the_dead_zone():
    # z = +1.0 -> between z_exit (0.5) and z_entry (2.0) -> HOLD both.
    pa = 102.0
    pb = pa / math.exp(1.0 * 0.02)
    signal = generate_pair_signal(_entry(), _prices(pa, pb))
    assert signal is not None
    assert signal.signal_a == "HOLD"
    assert signal.signal_b == "HOLD"


def test_reversion_exit_signals_both_legs():
    # z = +0.2 -> |z| < z_exit -> SELL_EXIT both (take the reversion).
    pa = 100.4
    pb = pa / math.exp(0.2 * 0.02)
    signal = generate_pair_signal(_entry(), _prices(pa, pb))
    assert signal is not None
    assert signal.signal_a == "SELL_EXIT"
    assert signal.signal_b == "SELL_EXIT"


def test_stop_loss_overrides_entry():
    # z = +4.0 -> beyond z_stop -> SELL_EXIT both even though the
    # spread is "wide enough to enter".
    pa = 108.0
    pb = pa / math.exp(4.0 * 0.02)
    signal = generate_pair_signal(_entry(), _prices(pa, pb))
    assert signal is not None
    assert signal.signal_a == "SELL_EXIT"
    assert signal.signal_b == "SELL_EXIT"


def test_custom_thresholds():
    entry = _entry()
    pa = 102.0
    pb = pa / math.exp(1.2 * 0.02)  # z = 1.2
    tight = generate_pair_signal(entry, _prices(pa, pb),
                                 z_entry=1.0, z_exit=0.5, z_stop=3.5)
    assert tight is not None and tight.signal_b == "BUY"
    wide = generate_pair_signal(entry, _prices(pa, pb),
                                z_entry=1.5, z_exit=0.5, z_stop=3.5)
    assert wide is not None and wide.signal_a == "HOLD"


def test_leg_signal_lookup():
    signal = generate_pair_signal(
        _entry(), _prices(105.0, 105.0 / math.exp(2.5 * 0.02)))
    assert signal is not None
    assert signal.leg_signal("BBB") == "BUY"
    assert signal.leg_signal("AAA") == "SELL_EXIT"
    assert signal.leg_signal("NOPE") == "HOLD"


def test_confidence_clamped():
    # |z| far beyond entry -> confidence caps at 1.0; tiny z -> 0.
    pa = 108.0
    pb = pa / math.exp(6.0 * 0.02)  # z = 6 -> stop zone
    signal = generate_pair_signal(_entry(), _prices(pa, pb))
    assert signal is not None
    assert signal.confidence == 1.0


def test_price_input_shapes():
    idx = pd.date_range("2026-01-01", periods=5, freq="B")
    frame_a = pd.DataFrame({"Close": [1.0] * 4 + [105.0]}, index=idx)
    series_b = pd.Series([1.0] * 4 + [105.0 / math.exp(2.5 * 0.02)], index=idx)
    signal = generate_pair_signal(_entry(), {"AAA": frame_a, "BBB": series_b})
    assert signal is not None
    assert signal.price_a == pytest.approx(105.0)
    assert signal.z_score == pytest.approx(2.5, abs=1e-6)


def test_fail_open_on_missing_or_junk_prices():
    entry = _entry()
    assert generate_pair_signal(entry, {}) is None
    assert generate_pair_signal(entry, {"AAA": 100.0}) is None
    assert generate_pair_signal(entry, {"AAA": 0.0, "BBB": 100.0}) is None
    assert generate_pair_signal(entry, {"AAA": -5.0, "BBB": 100.0}) is None
    assert generate_pair_signal(entry, {"AAA": "junk", "BBB": 100.0}) is None
    assert generate_pair_signal(None, _prices(100.0, 100.0)) is None


def test_fail_open_on_junk_entry_stats():
    assert generate_pair_signal(_entry(spread_std=0.0),
                                _prices(100.0, 100.0)) is None
    assert generate_pair_signal(_entry(hedge_ratio=-1.0),
                                _prices(100.0, 100.0)) is None
    assert generate_pair_signal(_entry(spread_mean="junk"),
                                _prices(100.0, 100.0)) is None


def test_spread_daily_vol_rides_along():
    signal = generate_pair_signal(
        _entry(), _prices(105.0, 105.0 / math.exp(2.5 * 0.02)))
    assert signal is not None
    assert signal.spread_daily_vol == pytest.approx(0.004)
    assert generate_pair_signal(
        _entry(spread_daily_vol=None),
        _prices(105.0, 105.0 / math.exp(2.5 * 0.02)),
    ).spread_daily_vol is None


def test_generate_pair_signals_iterates_book():
    from src.pairs.book import PairBook

    book = PairBook("/unused")
    book.entries = {
        "ZZZ/YYY": _entry(pair_id="ZZZ/YYY").__class__(**{
            **_entry(pair_id="ZZZ/YYY").__dict__,
            "ticker_a": "ZZZ", "ticker_b": "YYY",
        }),
        "AAA/BBB": _entry(),
    }
    prices = {
        "AAA": 105.0, "BBB": 105.0 / math.exp(2.5 * 0.02),
        "ZZZ": 105.0, "YYY": 105.0 / math.exp(2.5 * 0.02),
    }
    signals = generate_pair_signals(book, prices)
    assert [s.pair_id for s in signals] == ["AAA/BBB", "ZZZ/YYY"]
    assert generate_pair_signals(None, prices) == []
    assert generate_pair_signals(book, {}) == []


def test_spread_math_nontrivial_params():
    # alpha/beta/mean all nonzero: z uses the persisted stats exactly.
    entry = _entry(hedge_ratio=1.5, alpha=0.3, spread_mean=0.05,
                   spread_std=0.04)
    pa, pb = 120.0, 80.0
    spread = math.log(pa) - 0.3 - 1.5 * math.log(pb)
    expected_z = (spread - 0.05) / 0.04
    signal = generate_pair_signal(entry, _prices(pa, pb))
    assert signal is not None
    assert signal.z_score == pytest.approx(expected_z, abs=1e-9)
