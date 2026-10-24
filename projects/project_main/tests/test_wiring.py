"""Integration wiring tests (TASK-7013, tests-first).

Pins the real-component seams left stubbed by TASK-7011B/TASK-7012:

  * src.cli._components() must return the full 12-key run_pipeline
    contract bound to the REAL project modules (import-path regression
    guard for the hallucinated-module-layout failure class).
  * src.pipeline.smoke.run_smoke must drive the flow through
    src.pipeline.graph.run_pipeline with the injected components and
    append every decision to the SQLite audit log (context.md hard
    rule 6: decision audit).
  * the CLI scan path must persist decisions, JSON-safe even when the
    decision dict carries pydantic models inside ``reports``.

Offline only: every I/O seam is monkeypatched; no test touches the
network, yfinance, OpenRouter or Alpaca.
"""

import asyncio

import pandas as pd
import pytest
from typer.testing import CliRunner

import src.cli as cli
import src.data.fundamentals as fundamentals
import src.data.market_data as market_data
import src.data.news as news
import src.pipeline.smoke as smoke
from src.agents.analysts import run_analysts
from src.agents.debate import run_debate
from src.agents.models import AnalystReport
from src.agents.portfolio_manager import build_order_plan
from src.agents.risk_manager import evaluate_risk
from src.quant.signals import generate_signal
from src.store.decisions import DecisionStore

runner = CliRunner()

#: The exact component contract of src.pipeline.graph.run_pipeline
#: (pinned by TASK-7010B).
_COMPONENT_KEYS = {
    "client",
    "account",
    "day_trade_dates",
    "positions",
    "regime",
    "fetch_ohlcv",
    "fetch_fundamentals",
    "fetch_news",
    "generate_signal",
    "run_analysts",
    "run_debate",
    "evaluate_risk",
    "build_order_plan",
    "broker_submit",
}


def _no_alpaca_keys(monkeypatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    # Keep every _components() call offline: the regime fetch never runs.
    monkeypatch.setenv("REGIME_ENABLED", "0")


# ---------------------------------------------------------------------------
# src.cli._components -- the real component graph
# ---------------------------------------------------------------------------


def test_components_contains_full_contract(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    comps = cli._components()
    assert _COMPONENT_KEYS <= set(comps)
    for name in (
        "fetch_ohlcv",
        "fetch_fundamentals",
        "fetch_news",
        "generate_signal",
        "run_analysts",
        "run_debate",
        "evaluate_risk",
        "build_order_plan",
        "broker_submit",
    ):
        assert callable(comps[name]), name
    assert callable(getattr(comps["client"], "complete_json", None))

    account = comps["account"]
    assert isinstance(account, dict)
    assert float(account["equity"]) > 0
    assert 0 < float(account["max_position_pct"]) <= 1
    assert float(account["target_daily_vol"]) > 0
    assert 0 < float(account["max_sector_pct"]) <= 1
    assert isinstance(comps["day_trade_dates"], list)
    # Broker holdings map (empty without credentials): ticker -> qty.
    assert comps["positions"] == {}


def test_components_fetch_positions_when_creds_present(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("REGIME_ENABLED", "0")

    class FakeClient:
        def __init__(self, key, secret, paper_only=True):
            assert paper_only is True

        def get_account(self):
            return {"equity": "123456.78"}

        def get_positions(self):
            return [
                {"symbol": "AAPL", "qty": "12.5"},
                {"symbol": "msft", "qty": "3"},
                {"symbol": "BAD", "qty": "not-a-number"},
                {"symbol": "ZERO", "qty": "0"},
                {"symbol": "", "qty": "5"},
                {"symbol": "NOQTY"},
            ]

    monkeypatch.setattr(cli, "AlpacaClient", FakeClient)
    comps = cli._components(submit=True)
    assert comps["account"]["equity"] == 123456.78
    assert comps["positions"] == {"AAPL": 12.5, "MSFT": 3.0}


def test_components_reexport_real_agent_functions(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    comps = cli._components()
    assert comps["generate_signal"] is generate_signal
    assert comps["run_analysts"] is run_analysts
    assert comps["run_debate"] is run_debate
    assert comps["evaluate_risk"] is evaluate_risk
    assert comps["build_order_plan"] is build_order_plan


def test_components_fetch_ohlcv_returns_full_frame(monkeypatch):
    # Contract change (volume features): the closure returns the FULL
    # OHLCV frame; graph._close_of derives the close series for the
    # analysts/correlation and generate_signal consumes the frame.
    _no_alpaca_keys(monkeypatch)
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    frame = pd.DataFrame({"Close": [10.0, 11.0, 12.0]}, index=index)
    monkeypatch.setattr(market_data, "_download", lambda ticker, period: frame)

    comps = cli._components()
    out = comps["fetch_ohlcv"]("FAKE")
    assert isinstance(out, pd.DataFrame)
    assert "Close" in out.columns
    assert float(out["Close"].iloc[-1]) == 12.0


def test_components_data_fetchers_use_real_adapters(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    monkeypatch.setattr(
        fundamentals, "_raw_info",
        lambda ticker: {"grossMargins": 0.4, "marketCap": 5},
    )
    monkeypatch.setattr(
        news, "_raw_news",
        lambda ticker: [{"title": "t", "publisher": "p", "link": "u"}],
    )

    comps = cli._components()
    assert comps["fetch_fundamentals"]("FAKE")["gross_margin"] == 0.4
    assert comps["fetch_news"]("FAKE") == [
        {"title": "t", "publisher": "p", "url": "u"}
    ]


def test_components_broker_noop_without_keys(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    comps = cli._components()
    assert comps["broker_submit"]([]) == []
    # Even with submit=True, missing credentials degrade to the no-op
    # (paper-run additionally fails closed before reaching this point).
    paper_comps = cli._components(submit=True)
    assert paper_comps["broker_submit"]([]) == []


def test_components_regime_disabled_is_neutral(monkeypatch):
    _no_alpaca_keys(monkeypatch)  # sets REGIME_ENABLED=0
    comps = cli._components()
    assert comps["regime"] == {"regime": "neutral", "fear_z": None,
                               "liquidity_z": None, "vix": None,
                               "vix3m": None, "vix_term": None,
                               "curve_inverted": False, "components": {},
                               "as_of": None}


def test_components_regime_enabled_uses_fetch_seam(monkeypatch):
    from src.data import regime as regime_mod

    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("REGIME_ENABLED", "1")
    snapshot = {"regime": "risk_off", "fear_z": 1.2, "liquidity_z": None,
                "vix": 25.0, "vix3m": 26.0, "vix_term": 0.96,
                "curve_inverted": False, "components": {}, "as_of": "t"}
    monkeypatch.setattr(
        regime_mod, "fetch_regime",
        lambda cache=None, thresholds=None: dict(snapshot))
    comps = cli._components()
    assert comps["regime"] == snapshot


def test_components_regime_fetch_failure_falls_open(monkeypatch):
    from src.data import regime as regime_mod

    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("REGIME_ENABLED", "1")

    def boom(cache=None, thresholds=None):
        raise RuntimeError("no network")

    monkeypatch.setattr(regime_mod, "fetch_regime", boom)
    comps = cli._components()
    assert comps["regime"]["regime"] == "neutral"  # REGIME_NEUTRAL fallback


def test_components_regime_budget_mult_from_env(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("REGIME_BUDGET_MULT", "0.3")
    comps = cli._components()
    assert comps["account"]["regime_budget_mult"] == 0.3


# ---------------------------------------------------------------------------
# Kelly stats loading (item 7)
# ---------------------------------------------------------------------------


def test_components_kelly_stats_loaded_when_usable(monkeypatch, tmp_path):
    import json

    _no_alpaca_keys(monkeypatch)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "aggregate": {"kelly_fraction": 0.2, "n_trades": 50, "win_rate": 0.6},
    }), encoding="utf-8")
    monkeypatch.setenv("BACKTEST_STATS_PATH", str(stats_path))
    comps = cli._components()
    assert comps["account"]["kelly"] == {"fraction": 0.2, "n": 50}
    assert comps["account"]["kelly_fraction_cap"] == 0.25
    assert comps["account"]["kelly_max_mult"] == 1.5


def test_components_kelly_absent_file_is_none(monkeypatch, tmp_path):
    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("BACKTEST_STATS_PATH", str(tmp_path / "missing.json"))
    comps = cli._components()
    assert comps["account"]["kelly"] is None


def test_components_kelly_low_sample_is_none(monkeypatch, tmp_path):
    import json

    _no_alpaca_keys(monkeypatch)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "aggregate": {"kelly_fraction": 0.2, "n_trades": 10},
    }), encoding="utf-8")
    monkeypatch.setenv("BACKTEST_STATS_PATH", str(stats_path))
    comps = cli._components()
    assert comps["account"]["kelly"] is None


def test_components_kelly_disabled_flag_is_none(monkeypatch, tmp_path):
    import json

    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("KELLY_ENABLED", "0")
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({
        "aggregate": {"kelly_fraction": 0.2, "n_trades": 50},
    }), encoding="utf-8")
    monkeypatch.setenv("BACKTEST_STATS_PATH", str(stats_path))
    comps = cli._components()
    assert comps["account"]["kelly"] is None


def test_components_kelly_junk_file_is_none(monkeypatch, tmp_path):
    _no_alpaca_keys(monkeypatch)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text("not json at all", encoding="utf-8")
    monkeypatch.setenv("BACKTEST_STATS_PATH", str(stats_path))
    comps = cli._components()
    assert comps["account"]["kelly"] is None


# ---------------------------------------------------------------------------
# Whale/flow fetch components (flag-gated)
# ---------------------------------------------------------------------------

_FLOW_FETCHERS = (
    "fetch_insider",
    "fetch_institutional",
    "fetch_options_flow",
    "fetch_dark_pool",
)

_FLOW_FLAGS = (
    "INSIDER_ENABLED",
    "INSTITUTIONAL_ENABLED",
    "OPTIONS_FLOW_ENABLED",
    "DARK_POOL_ENABLED",
)


def test_components_flow_fetchers_registered_by_default(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    comps = cli._components()
    for name in _FLOW_FETCHERS:
        assert callable(comps[name]), name


def test_components_flow_fetchers_absent_when_disabled(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    for flag in _FLOW_FLAGS:
        monkeypatch.setenv(flag, "0")
    comps = cli._components()
    for name in _FLOW_FETCHERS:
        assert name not in comps, name


def test_components_flow_thresholds_in_account(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("CLUSTER_INSIDER_THRESHOLD", "5")
    monkeypatch.setenv("FLOW_VETO_MIN_HOLDERS", "20")
    monkeypatch.setenv("UOA_PUT_SKEW_VETO", "0.25")
    account = cli._components()["account"]
    assert account["cluster_insider_threshold"] == 5
    assert account["flow_veto_min_holders"] == 20
    assert account["uoa_put_skew_veto"] == 0.25


def test_components_insider_fetcher_uses_cache_seam(monkeypatch, tmp_path):
    _no_alpaca_keys(monkeypatch)
    monkeypatch.setenv("EDGAR_IDENTITY", "Tester tester@example.com")
    monkeypatch.setenv("REGIME_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("INSIDER_LOOKBACK_DAYS", "45")
    calls = []
    import src.data.insider as insider_mod

    def seam(ticker, days):
        calls.append((ticker, days))
        return []

    monkeypatch.setattr(insider_mod, "_form4_transactions", seam)
    monkeypatch.setattr(insider_mod, "_form144_proposals",
                        lambda t, d: [])

    comps = cli._components()
    first = comps["fetch_insider"]("FAKE")
    second = comps["fetch_insider"]("FAKE")
    assert calls == [("FAKE", 45)]  # second call served from the disk cache
    assert first["status"] == "no_data"
    assert first == second


# ---------------------------------------------------------------------------
# Smoke-rewiring probe: the three smoke-delegation tests below self-skip
# until src/pipeline/smoke.py is rewired (TASK-7013B). Mirrors the repo's
# tests-first activation pattern.
# ---------------------------------------------------------------------------


def _smoke_rewired() -> bool:
    """True once _build_real_components is the real async delegation."""
    try:
        comps = asyncio.run(smoke._build_real_components(dry_run=True))
        return (
            isinstance(comps, dict)
            and "client" in comps
            and "fetch_ohlcv" in comps
            and "build_order_plan" in comps
        )
    except Exception:
        return False


_needs_smoke = pytest.mark.skipif(
    not _smoke_rewired(),
    reason="smoke rewiring pending (activates when smoke.py delegates to "
    "run_pipeline / cli._components)",
)

# ---------------------------------------------------------------------------
# src.pipeline.smoke.run_smoke -- delegation to run_pipeline + audit log
# ---------------------------------------------------------------------------


def _stub_components(plan_calls, submit_calls, quant):
    def build_order_plan(decisions, signals, account, day_trade_dates,
                         as_of=None, positions=None, sector_map=None,
                         regime=None, analyst_reports=None, debates=None,
                         cluster_map=None):
        plan_calls.append(
            (list(decisions), dict(signals), dict(account),
             list(day_trade_dates), as_of)
        )
        return []

    def broker_submit(orders):
        submit_calls.append(list(orders))
        return []

    return {
        "fetch_ohlcv": lambda ticker, period="6mo": pd.Series(
            [100.0, 101.0, 99.0] * 10
        ),
        "fetch_fundamentals": lambda ticker: {},
        "fetch_news": lambda ticker: [],
        "generate_signal": lambda prices, regime=None: dict(quant),
        "run_analysts": lambda ticker, inputs, client=None: {},
        "run_debate": lambda ticker, reports, client=None: None,
        "evaluate_risk": lambda ticker, reports, debate, quant_dict, account,
        proposed_position_pct=None, flow_data=None: None,
        "build_order_plan": build_order_plan,
        "broker_submit": broker_submit,
        "account": {"equity": 100_000.0, "max_position_pct": 0.10},
        "day_trade_dates": [],
        "client": None,
    }


@_needs_smoke
def test_run_smoke_delegates_to_pipeline_contract(monkeypatch, tmp_path):
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "d.sqlite3"))
    plan_calls = []
    submit_calls = []
    quant = {"signal": "HOLD", "confidence": 0.5, "z_score": 0.0,
             "momentum": 0.0}
    components = _stub_components(plan_calls, submit_calls, quant)

    summary = asyncio.run(smoke.run_smoke(
        ["FAKE1"], dry_run=True, components=components))

    assert summary["dry_run"] is True
    assert summary["orders_submitted"] == 0
    assert summary["tickers"] == ["FAKE1"]

    # The batch contract of the real build_order_plan: one call with the
    # decision list, the signals dict keyed by ticker (annotated with the
    # reference price), the account, the day-trade dates and as_of.
    assert len(plan_calls) == 1
    decisions_arg, signals_arg, account_arg, dtd_arg, as_of_arg = plan_calls[0]
    assert signals_arg == {"FAKE1": {**quant, "reference_price": 99.0}}
    assert isinstance(decisions_arg, list) and len(decisions_arg) == 1
    assert account_arg == {"equity": 100_000.0, "max_position_pct": 0.10}
    assert dtd_arg == []
    assert as_of_arg is None

    # broker submitted exactly once for the batch.
    assert len(submit_calls) == 1

    # Decision audit: exactly one row for the one ticker.
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    assert store.count() == 1
    logged = store.tail(1)[0]
    assert logged["ticker"] == "FAKE1"


@_needs_smoke
def test_run_smoke_does_not_mutate_injected_components(monkeypatch, tmp_path):
    monkeypatch.setenv("DECISIONS_DB", str(tmp_path / "d.sqlite3"))
    components = _stub_components([], [], {"signal": "HOLD"})
    original_broker = components["broker_submit"]

    asyncio.run(smoke.run_smoke(["FAKE1"], dry_run=True,
                                components=components))

    assert components["broker_submit"] is original_broker
    assert set(components) == set(_stub_components([], [], {}).keys())


@_needs_smoke
def test_build_real_components_is_offline_safe(monkeypatch, tmp_path):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("REGIME_ENABLED", "0")
    monkeypatch.chdir(tmp_path)

    components = asyncio.run(smoke._build_real_components(dry_run=True))
    assert _COMPONENT_KEYS <= set(components)
    assert components["broker_submit"]([]) == []
    assert components["regime"]["regime"] == "neutral"


# ---------------------------------------------------------------------------
# CLI scan path -- decision audit (context.md hard rule 6)
# ---------------------------------------------------------------------------


def _fake_decision(reports=None):
    return [{
        "ticker": "FAKE1",
        "quant": {"signal": "HOLD", "confidence": 0.5, "z_score": 0.1},
        "veto": {"approved": False, "veto_reasons": []},
        "reports": reports,
        "orders": [],
    }]


def test_scan_logs_decisions(monkeypatch, tmp_path):
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    monkeypatch.setattr(
        cli, "run_pipeline",
        lambda tickers, components=None, as_of=None: _fake_decision({}),
    )
    monkeypatch.setattr(cli, "_store", lambda: store)

    result = runner.invoke(cli.app, ["scan", "--tickers", "FAKE1"])

    assert result.exit_code == 0
    assert "FAKE1" in result.output
    assert store.count() == 1
    logged = store.tail(1)[0]
    assert logged["ticker"] == "FAKE1"
    assert logged["veto"]["approved"] is False


def test_scan_log_is_json_safe_for_pydantic_reports(monkeypatch, tmp_path):
    store = DecisionStore(db_path=tmp_path / "d.sqlite3")
    report = AnalystReport(
        agent="fundamental", ticker="FAKE1", stance="bullish",
        confidence=0.7, thesis="t", citations=["close=12.0"],
    )
    monkeypatch.setattr(
        cli, "run_pipeline",
        lambda tickers, components=None, as_of=None: _fake_decision(
            {"fundamental": report}),
    )
    monkeypatch.setattr(cli, "_store", lambda: store)

    result = runner.invoke(cli.app, ["scan", "--tickers", "FAKE1"])

    assert result.exit_code == 0
    assert store.count() == 1
    logged = store.tail(1)[0]
    assert isinstance(logged["reports"]["fundamental"], dict)
    assert logged["reports"]["fundamental"]["agent"] == "fundamental"


def test_paper_run_still_fails_closed(monkeypatch):
    _no_alpaca_keys(monkeypatch)
    result = runner.invoke(cli.app, ["paper-run"])
    assert result.exit_code != 0
