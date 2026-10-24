"""Tests for the Typer CLI (src/cli.py). All externals are stubbed."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.cli", "src.config", "src.store.decisions")
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from typer.testing import CliRunner

import src.cli as cli
from src.config import reset_settings_cache

runner = CliRunner()


def test_scan_prints_decision_rows(monkeypatch):
    fake_decisions = [{
        "ticker": "FAKE1",
        "quant": {"signal": "HOLD", "confidence": 0.5, "z_score": 0.1},
        "veto": {"approved": False, "veto_reasons": []},
        "reports": {},
        "orders": [],
    }]

    def fake_run_pipeline(tickers, components=None, as_of=None):
        assert "FAKE1" in tickers
        return fake_decisions

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)
    result = runner.invoke(cli.app, ["scan", "--tickers", "FAKE1"])
    assert result.exit_code == 0
    assert "FAKE1" in result.output
    assert "HOLD" in result.output


def test_decisions_reads_store(monkeypatch, tmp_path):
    from src.store.decisions import DecisionStore
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    store.log_decision({"ts": "t", "ticker": "AAPL", "quant": {},
                        "reports": [], "veto": {}, "orders": []})
    monkeypatch.setattr(cli, "_store", lambda: store)
    result = runner.invoke(cli.app, ["decisions"])
    assert result.exit_code == 0
    assert "AAPL" in result.output


def test_show_config_lists_watchlist(monkeypatch):
    reset_settings_cache()
    monkeypatch.setenv("WATCHLIST", "AAA,BBB")
    reset_settings_cache()
    result = runner.invoke(cli.app, ["show-config"])
    assert result.exit_code == 0
    assert "AAA" in result.output and "BBB" in result.output


def test_paper_run_fails_closed_without_keys(monkeypatch):
    reset_settings_cache()
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    result = runner.invoke(cli.app, ["paper-run"])
    assert result.exit_code != 0


def test_app_has_expected_commands():
    registered = {cmd.name for cmd in cli.app.registered_commands} | \
                 {cmd.name for cmd in cli.app.registered_groups}
    assert {"scan", "paper-run", "decisions", "show-config"} <= registered


def test_backtest_writes_stats_and_prints_summary(monkeypatch, tmp_path):
    from src.backtest import replay as replay_mod

    fake_stats = {
        "generated_at": "t", "period": "2y", "horizon_days": 5,
        "min_trades": 30,
        "per_ticker": {"FAKE1": {"n_trades": 40, "win_rate": 0.6,
                                 "kelly_fraction": 0.3}},
        "aggregate": {"n_trades": 40, "win_rate": 0.6, "payoff_ratio": 1.5,
                      "kelly_fraction": 0.3, "usable": True},
    }
    monkeypatch.setattr(replay_mod, "run_backtest",
                        lambda *a, **k: dict(fake_stats))
    out_path = tmp_path / "stats.json"
    monkeypatch.setenv("BACKTEST_STATS_PATH", str(out_path))
    result = runner.invoke(cli.app, ["backtest", "--tickers", "FAKE1"])
    assert result.exit_code == 0
    assert "kelly=0.3" in result.output
    assert out_path.exists()


def test_pairs_backtest_writes_sizer_readable_stats(monkeypatch, tmp_path):
    # Regression (2026-09-14 review): without --param the CLI must write
    # the RAW stats dict (top-level "aggregate") so pair_kelly_from_stats
    # and PAIR_KELLY_* sizing can actually load it. The old sweep-shaped
    # wrapper left pairs Kelly silently neutral forever.
    import json

    from src.pairs import backtest as pairs_backtest_mod
    from src.pairs.sizing import pair_kelly_from_stats

    fake_stats = {
        "generated_at": "t", "mode": "pairs", "period": "2y",
        "formation_window": 252, "rebalance_days": 21,
        "universe_size": 2, "dropped_short_history": [],
        "formations": 1,
        "z_entry": 1.5, "z_exit": 0.25, "z_stop": 3.5,
        "spread_bps": None, "slippage_bps": None, "cost_scale_by_vol": None,
        "per_pair": {},
        "aggregate": {"n_trades": 50, "win_rate": 0.6, "payoff_ratio": 1.4,
                      "kelly_fraction": 0.32, "usable": True},
    }
    monkeypatch.setattr(pairs_backtest_mod, "run_pairs_backtest",
                        lambda *a, **k: dict(fake_stats))
    out_path = tmp_path / "pairs_stats.json"
    result = runner.invoke(cli.app, [
        "pairs-backtest", "--tickers", "AAA,BBB", "--out", str(out_path),
    ])
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert "aggregate" in written, "stats file must carry a top-level aggregate"
    # The consumer the sizer actually uses must load the written file.
    kelly = pair_kelly_from_stats(str(out_path), min_trades=30)
    assert kelly == {"fraction": 0.32, "n": 50}


def test_pairs_backtest_param_writes_sweep_shape(monkeypatch, tmp_path):
    # With --param the output stays the sweep matrix (offline analysis);
    # the stats-file contract above applies only to the plain run.
    import json

    from src.pairs import backtest as pairs_backtest_mod

    monkeypatch.setattr(pairs_backtest_mod, "run_pairs_param_sweep",
                        lambda *a, **k: {"generated_at": "t", "tickers": [],
                                         "results": []})
    out_path = tmp_path / "pairs_sweep.json"
    result = runner.invoke(cli.app, [
        "pairs-backtest", "--tickers", "AAA,BBB", "--out", str(out_path),
        "--param", "z_entry=1.5,2.0",
    ])
    assert result.exit_code == 0, result.output
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert "results" in written and "aggregate" not in written


def test_regime_sweep_prints_fire_rate_table(monkeypatch):
    from src.backtest import regime_sweep as sweep_mod

    fake_report = {
        "days_evaluated": 10,
        "keys": {"fear_z_threshold": [
            {"threshold": 1.0, "evaluated": 10, "fired": 2,
             "fire_rate": 0.2},
            {"threshold": 2.0, "evaluated": 10, "fired": 0,
             "fire_rate": 0.0},
        ]},
    }
    monkeypatch.setattr(sweep_mod, "collect_series",
                        lambda period, cache=None: {})
    monkeypatch.setattr(sweep_mod, "sweep_thresholds",
                        lambda raw, baseline=None: fake_report)
    result = runner.invoke(cli.app, ["regime-sweep"])
    assert result.exit_code == 0
    assert "days evaluated: 10" in result.output
    assert "fear_z_threshold" in result.output
    assert "20.0%" in result.output
