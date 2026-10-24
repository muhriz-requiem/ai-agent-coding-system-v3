"""Tests for the execution cost model (src/backtest/costs.py)."""

from __future__ import annotations

import pytest

from src.backtest.costs import ExecutionCosts, costs_from_env


class TestExecutionCosts:
    def test_default_values(self):
        costs = ExecutionCosts()
        assert costs.spread_bps == 10.0
        assert costs.slippage_bps == 3.0
        assert costs.scale_by_vol is True

    def test_round_trip_bps_no_scaling(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        assert costs.round_trip_bps() == 16.0  # 10 + 2*3

    def test_round_trip_bps_with_scaling(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=True)
        # At 2% daily vol (baseline), no scaling
        assert costs.round_trip_bps(daily_vol=0.02) == 16.0
        # At 4% daily vol, 2x scaling
        assert costs.round_trip_bps(daily_vol=0.04) == 32.0
        # At 1% daily vol, 0.5x scaling
        assert costs.round_trip_bps(daily_vol=0.01) == 8.0

    def test_one_way_bps(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        assert costs.one_way_bps() == 8.0  # half of round trip

    def test_adjust_entry_buy(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        price = 100.0
        adjusted = costs.adjust_entry(price, "BUY")
        # 8 bps = 0.08%
        assert adjusted == pytest.approx(100.08)

    def test_adjust_entry_sell(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        price = 100.0
        adjusted = costs.adjust_entry(price, "SELL")
        assert adjusted == pytest.approx(99.92)

    def test_adjust_exit_buy(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        price = 100.0
        adjusted = costs.adjust_exit(price, "BUY")
        assert adjusted == pytest.approx(99.92)

    def test_adjust_exit_sell(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        price = 100.0
        adjusted = costs.adjust_exit(price, "SELL")
        assert adjusted == pytest.approx(100.08)

    def test_adjust_with_vol_scaling(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=True)
        price = 100.0
        # At 4% vol, costs are 2x
        adjusted = costs.adjust_entry(price, "BUY", daily_vol=0.04)
        # 16 bps = 0.16%
        assert adjusted == pytest.approx(100.16)

    def test_negative_price_unchanged(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0)
        assert costs.adjust_entry(-1.0, "BUY") == -1.0
        assert costs.adjust_entry(0.0, "BUY") == 0.0

    def test_unknown_side_unchanged(self):
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0)
        assert costs.adjust_entry(100.0, "HOLD") == 100.0


class TestCostsFromEnv:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("SPREAD_BPS", raising=False)
        monkeypatch.delenv("SLIPPAGE_BPS", raising=False)
        monkeypatch.delenv("COST_SCALE_BY_VOL", raising=False)
        costs = costs_from_env()
        assert costs.spread_bps == 10.0
        assert costs.slippage_bps == 3.0
        assert costs.scale_by_vol is True

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("SPREAD_BPS", "5.0")
        monkeypatch.setenv("SLIPPAGE_BPS", "2.0")
        monkeypatch.setenv("COST_SCALE_BY_VOL", "0")
        costs = costs_from_env()
        assert costs.spread_bps == 5.0
        assert costs.slippage_bps == 2.0
        assert costs.scale_by_vol is False

    def test_invalid_values_fallback(self, monkeypatch):
        monkeypatch.setenv("SPREAD_BPS", "invalid")
        monkeypatch.setenv("SLIPPAGE_BPS", "invalid")
        costs = costs_from_env()
        assert costs.spread_bps == 10.0
        assert costs.slippage_bps == 3.0

    def test_negative_values_clamped(self, monkeypatch):
        monkeypatch.setenv("SPREAD_BPS", "-5.0")
        monkeypatch.setenv("SLIPPAGE_BPS", "-2.0")
        costs = costs_from_env()
        assert costs.spread_bps == 0.0
        assert costs.slippage_bps == 0.0


class TestReplayWithCosts:
    """Integration tests for replay with costs."""

    def test_replay_signals_with_costs(self):
        import pandas as pd
        from src.backtest.replay import replay_signals
        from src.backtest.costs import ExecutionCosts

        # Create a price series that generates BUY signals
        values = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
        closes = pd.Series(values)

        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        trades_no_cost = replay_signals(closes, horizon=5)
        trades_with_cost = replay_signals(closes, horizon=5, costs=costs)

        # With costs, returns should be worse (more negative or less positive)
        for t1, t2 in zip(trades_no_cost, trades_with_cost):
            assert t2["return"] <= t1["return"]

    def test_replay_overnight_with_costs(self):
        import pandas as pd
        from src.backtest.replay import replay_overnight
        from src.backtest.costs import ExecutionCosts

        closes = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
        opens = [100.0] + [c * 1.01 for c in closes[:-1]]
        frame = pd.DataFrame({"Close": closes, "Open": opens})

        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        trades_no_cost = replay_overnight(frame)
        trades_with_cost = replay_overnight(frame, costs=costs)

        for t1, t2 in zip(trades_no_cost, trades_with_cost):
            assert t2["return"] <= t1["return"]

    def test_replay_random_with_costs(self):
        import pandas as pd
        from src.backtest.replay import replay_random
        from src.backtest.costs import ExecutionCosts

        values = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
        closes = pd.Series(values)

        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        trades_no_cost = replay_random(closes, horizon=5, signal_rate=1.0, seed=42)
        trades_with_cost = replay_random(closes, horizon=5, signal_rate=1.0, seed=42, costs=costs)

        assert len(trades_no_cost) == len(trades_with_cost)
        for t1, t2 in zip(trades_no_cost, trades_with_cost):
            assert t2["return"] <= t1["return"]


class TestPortfolioWithCosts:
    """Integration tests for portfolio backtest with costs."""

    def test_simulate_portfolio_overnight_with_costs(self):
        from src.backtest.portfolio import simulate_portfolio_overnight
        from src.backtest.costs import ExecutionCosts
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=10)
        entries = {
            "AAA": {
                dates[0]: {"entry": 100.0, "exit": 101.0, "confidence": 0.9},
                dates[1]: {"entry": 100.0, "exit": 102.0, "confidence": 0.5},
            },
        }

        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=3.0, scale_by_vol=False)
        sim_no_cost = simulate_portfolio_overnight(entries, top_n=2, stake_pct=0.20, cost_bps=0.0)
        sim_with_cost = simulate_portfolio_overnight(entries, top_n=2, stake_pct=0.20, cost_bps=0.0, costs=costs)

        # With costs, returns should be worse
        assert sim_with_cost["final_equity"] < sim_no_cost["final_equity"]

    def test_costs_vs_cost_bps_equivalence(self):
        """Test that cost model with 10bps spread + 0 slippage ≈ cost_bps=10."""
        from src.backtest.portfolio import simulate_portfolio_overnight
        from src.backtest.costs import ExecutionCosts
        import pandas as pd

        dates = pd.bdate_range("2024-01-01", periods=10)
        entries = {
            "AAA": {
                dates[0]: {"entry": 100.0, "exit": 101.0, "confidence": 0.9},
            },
        }

        # Simple cost_bps=10 (10 bps round trip = 1%)
        sim_simple = simulate_portfolio_overnight(entries, top_n=1, stake_pct=1.0, cost_bps=10.0)

        # Cost model: spread=10, slippage=0 -> round trip = 10 bps
        costs = ExecutionCosts(spread_bps=10.0, slippage_bps=0.0, scale_by_vol=False)
        sim_model = simulate_portfolio_overnight(entries, top_n=1, stake_pct=1.0, cost_bps=0.0, costs=costs)

        # Should be very close (same math) - allow small diff due to entry/exit split
        assert sim_simple["final_equity"] == pytest.approx(sim_model["final_equity"], rel=1e-4)


class TestWalkforward:
    """Tests for walk-forward backtest."""

    def test_run_walkforward_basic(self):
        from src.backtest.sweep import run_walkforward

        # Use a simple fetch that returns synthetic data
        def fake_fetch(ticker, period):
            import pandas as pd
            values = [100.0] * 400 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0] * 20
            return pd.DataFrame({"Close": values})

        result = run_walkforward(
            ["TEST"], period="1y", train_ratio=0.7, step_days=63,
            min_train_days=252, horizon=5, min_trades=1, mode="close",
            fetch=fake_fetch,
        )

        assert "folds" in result
        assert len(result["folds"]) >= 1
        for fold in result["folds"]:
            assert "ticker" in fold
            assert "fold" in fold
            assert "train_bars" in fold
            assert "test_bars" in fold

    def test_run_walkforward_overnight(self):
        from src.backtest.sweep import run_walkforward

        def fake_fetch(ticker, period):
            import pandas as pd
            n = 500
            closes = [100.0 * (1.0005 ** i) for i in range(n)]
            opens = [100.0] + [c * 1.0001 for c in closes[:-1]]
            return pd.DataFrame({"Close": closes, "Open": opens})

        result = run_walkforward(
            ["TEST"], period="1y", train_ratio=0.7, step_days=63,
            min_train_days=252, horizon=5, min_trades=1, mode="overnight",
            fetch=fake_fetch,
        )

        assert "folds" in result
        assert len(result["folds"]) >= 1
        assert result["mode"] == "overnight"
        assert result["horizon_days"] == 1

    def test_walkforward_folds_exactly_tile_the_oos_region(self):
        # Folds must be disjoint and, pooled, must contain exactly the
        # trades a single replay with start_index=train_end collects.
        import numpy as np
        import pandas as pd
        from src.backtest.replay import replay_signals
        from src.backtest.sweep import run_walkforward

        rng = np.random.default_rng(7)
        steps = rng.normal(0.0004, 0.02, size=600)
        frame = pd.DataFrame({"Close": list(100.0 * np.cumprod(1.0 + steps))})

        result = run_walkforward(
            ["TEST"], period="1y", train_ratio=0.7, step_days=63,
            min_train_days=252, horizon=5, min_trades=1, mode="close",
            fetch=lambda t, p: frame,
        )
        train_end = max(int(600 * 0.7), 252)
        expected = replay_signals(frame, horizon=5, start_index=train_end)
        assert expected, "noise fixture must produce trades"
        per_fold_total = sum(f["aggregate"]["n_trades"] for f in result["folds"])
        assert per_fold_total == len(expected)
        assert result["aggregate_oos"]["n_trades"] == len(expected)

    def test_walkforward_overnight_folds_tile_exactly(self):
        import numpy as np
        import pandas as pd
        from src.backtest.replay import replay_overnight
        from src.backtest.sweep import run_walkforward

        rng = np.random.default_rng(11)
        steps = rng.normal(0.0004, 0.02, size=600)
        closes = list(100.0 * np.cumprod(1.0 + steps))
        opens = [closes[0]] + closes[:-1]  # Open[t] = Close[t-1]
        frame = pd.DataFrame({"Close": closes, "Open": opens})

        result = run_walkforward(
            ["TEST"], period="1y", train_ratio=0.7, step_days=63,
            min_train_days=252, horizon=5, min_trades=1, mode="overnight",
            fetch=lambda t, p: frame,
        )
        train_end = max(int(600 * 0.7), 252)
        expected = replay_overnight(frame, start_index=train_end)
        assert expected, "noise fixture must produce trades"
        per_fold_total = sum(f["aggregate"]["n_trades"] for f in result["folds"])
        assert per_fold_total == len(expected)
        assert result["aggregate_oos"]["n_trades"] == len(expected)

    def test_walkforward_respects_min_trades(self):
        # min_trades must flow through: a 2-trade fold reports
        # kelly=None/usable=False instead of the old hard-coded n>=1.
        import pandas as pd
        from src.backtest.sweep import run_walkforward

        def fake_fetch(ticker, period):
            return pd.DataFrame({"Close": [100.0 + float(i) for i in range(600)]})

        result = run_walkforward(
            ["TEST"], period="1y", train_ratio=0.7, step_days=63,
            min_train_days=252, horizon=5, min_trades=30, mode="close",
            fetch=fake_fetch,
        )
        assert result["aggregate_oos"]["kelly_fraction"] is None
        assert result["aggregate_oos"]["usable"] is False


class TestRandomBaseline:
    """Tests for the random-baseline null distribution."""

    def test_replay_random_overnight_matches_strategy_contract(self):
        import pandas as pd
        from src.backtest.replay import replay_random

        closes = [100.0] * 30 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0]
        opens = [closes[0]] + [c * 1.01 for c in closes[:-1]]
        frame = pd.DataFrame({"Close": closes, "Open": opens})

        trades = replay_random(frame, mode="overnight", signal_rate=1.0,
                               seed=42)
        assert trades, "signal_rate=1.0 must trade every bar"
        for t in trades:
            # Enter at Close[t], exit at Open[t+1] = Close[t] * 1.01.
            assert t["exit_price"] == pytest.approx(t["entry_price"] * 1.01)
            if t["signal"] == "BUY":
                assert t["return"] == pytest.approx(0.01)
            else:
                assert t["return"] == pytest.approx(-0.01)

    def test_replay_random_is_deterministic_and_seeded(self):
        import pandas as pd
        from src.backtest.replay import replay_random

        values = [100.0] * 40 + [95.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0] * 5
        a = replay_random(pd.Series(values), horizon=5, seed=123)
        b = replay_random(pd.Series(values), horizon=5, seed=123)
        c = replay_random(pd.Series(values), horizon=5, seed=124)
        assert a == b
        assert a != c

    def test_replay_random_unknown_mode_raises(self):
        import pandas as pd
        from src.backtest.replay import replay_random

        with pytest.raises(ValueError):
            replay_random(pd.Series([100.0] * 60), mode="junk")