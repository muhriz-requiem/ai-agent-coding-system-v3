"""Command line interface for the trading pipeline.

This module wires together the real project components (market data,
fundamentals, news, quant signals, agent analysts/debate/risk/portfolio,
LLM client and broker) into the ``run_pipeline`` contract and exposes a
small Typer CLI on top of it.
"""

import asyncio
import inspect
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from pydantic import BaseModel

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")

from src.agents.models import OrderPlan

from src.agents.analysts import run_analysts
from src.agents.debate import run_debate
from src.agents.llm import LLMClient
from src.agents.portfolio_manager import build_order_plan
from src.agents.risk_manager import evaluate_risk
from src.broker.alpaca import AlpacaClient
from src.data.cache import Cache
from src.config import get_settings
from src.data.fundamentals import fetch_fundamentals
from src.data.market_data import fetch_ohlcv
from src.data.news import fetch_headlines
from src.pipeline.graph import run_pipeline
from src.pipeline.overnight import (exclude_session_bought, exit_orders,
                                    load_overnight_kelly, pending_exits)
from src.pipeline.session import (SubmissionGate, order_notional,
                                  order_side, order_ticker, seconds_until)
from src.quant.signals import generate_signal
from src.store.decisions import DecisionStore
from src.store.session import SessionStore

app = typer.Typer()


def _jsonable(obj: Any) -> Any:
    """Recursively convert pydantic models / containers into JSON-safe data."""
    if isinstance(obj, BaseModel):
        return _jsonable(obj.model_dump())
    if isinstance(obj, dict):
        return {key: _jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(item) for item in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _store() -> DecisionStore:
    db_path = Path(os.environ.get("DECISIONS_DB", "data/decisions.sqlite3"))
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return DecisionStore(db_path=db_path)


def _flag_enabled(name: str, default: str = "1") -> bool:
    """Feature-flag parse: 0/false/no disables, anything else enables."""
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, "")))
    except (TypeError, ValueError):
        return default


def _costs_from_options(spread_bps: Optional[float],
                        slippage_bps: Optional[float],
                        cost_scale_by_vol: Optional[bool]):
    """Build ExecutionCosts from CLI options with env-var fallback.

    ``None`` options fall back to ``SPREAD_BPS`` / ``SLIPPAGE_BPS`` /
    ``COST_SCALE_BY_VOL`` (via :func:`costs_from_env`), so the env vars
    configure backtest costs and explicit flags override them.
    """
    from src.backtest.costs import ExecutionCosts, costs_from_env

    env = costs_from_env()
    return ExecutionCosts(
        spread_bps=env.spread_bps if spread_bps is None else spread_bps,
        slippage_bps=env.slippage_bps if slippage_bps is None else slippage_bps,
        scale_by_vol=(
            env.scale_by_vol if cost_scale_by_vol is None else cost_scale_by_vol
        ),
    )


def _flow_cache(ttl_seconds: float):
    """Per-module disk cache, built at fetch time.

    ``Cache`` TTL is per instance (src/data/cache.py), so every fetch
    module gets its own instance constructed with its module TTL --
    sharing one instance would silently collapse every TTL to the 300 s
    default.  ``None`` degrades to uncached fetches.
    """
    try:
        root = os.environ.get("REGIME_CACHE_ROOT", "data/cache")
        return Cache(root=Path(root), ttl_seconds=ttl_seconds)
    except Exception:
        return None


def _components(submit: bool = False, cache: Optional[Dict[str, Any]] = None,
                with_pairs: bool = False) -> Dict[str, Any]:
    """Build the dict of components consumed by ``run_pipeline``.

    This function must never raise -- any failure while constructing an
    individual component falls back to a safe default so the CLI keeps
    working even without network access or optional credentials.
    """
    components: Dict[str, Any] = {}

    # -- LLM client (lazy, import-safe, no network at construction time) --
    try:
        components["client"] = LLMClient(api_key=os.environ.get("OPENROUTER_API_KEY"))
    except Exception:
        try:
            components["client"] = LLMClient(api_key=None)
        except Exception:
            components["client"] = None

    # -- account configuration --
    try:
        equity = float(os.environ.get("ACCOUNT_EQUITY", "100000"))
    except (TypeError, ValueError):
        equity = 100000.0

    try:
        max_position_pct = float(os.environ.get("MAX_POSITION_PCT", "0.10"))
    except (TypeError, ValueError):
        max_position_pct = 0.10

    # Volatility scaling: scale-down-only target for daily returns.
    try:
        target_daily_vol = float(os.environ.get("TARGET_DAILY_VOL", "0.02"))
    except (TypeError, ValueError):
        target_daily_vol = 0.02

    # Sector concentration cap as a fraction of equity.
    try:
        max_sector_pct = float(os.environ.get("MAX_SECTOR_PCT", "0.30"))
    except (TypeError, ValueError):
        max_sector_pct = 0.30

    # Regime risk-off budget multiplier (clamped in portfolio_manager).
    try:
        regime_budget_mult = float(os.environ.get("REGIME_BUDGET_MULT", "0.5"))
    except (TypeError, ValueError):
        regime_budget_mult = 0.5

    # Debate-winner opposition multiplier (scale-down-only; item 10):
    # a bear debate winner shrinks BUYs by this factor, a bull winner
    # shrinks SELLs.  Clamped to [0.25, 1.0] in portfolio_manager.
    debate_oppose_mult = _env_float("DEBATE_OPPOSE_MULT", 0.75)

    # Transaction-cost awareness (item 9): budget buffer for estimated
    # round-trip costs + dust filter on sub-minimum orders.  The buffer
    # defaults to 0.0 (no-op); enabling it is an explicit operator choice.
    cost_buffer_pct = _env_float("COST_BUFFER_PCT", 0.0)
    min_trade_notional = _env_float("MIN_TRADE_NOTIONAL", 10.0)
    # Equity-relative floor component: the effective dust floor is the
    # smaller of the absolute floor and max($1, equity * pct), so small
    # accounts trade while >= $1k accounts are unchanged.
    min_trade_notional_pct = _env_float("MIN_TRADE_NOTIONAL_PCT", 0.01)

    # Correlation-cluster cap (item 11): BUY notional per correlated
    # basket capped like a pseudo-sector.
    max_corr_pct = _env_float("MAX_CORR_PCT", 0.30)

    # Inverse-vol risk parity (item 8): batch-level equal-risk caps for
    # same-day BUY candidates carrying a valid daily_vol.
    invol_enabled = _flag_enabled("INVOL_ENABLED")

    # Kelly sizing (item 7): load backtest stats when present and usable;
    # absent/stale/junk stats leave sizing untouched (factor 1.0).
    kelly: Optional[Dict[str, Any]] = None
    if _flag_enabled("KELLY_ENABLED"):
        stats_path = Path(
            os.environ.get("BACKTEST_STATS_PATH", "data/backtest_stats.json")
        )
        min_trades = _env_int("KELLY_MIN_TRADES", 30)
        try:
            raw = json.loads(stats_path.read_text(encoding="utf-8"))
            aggregate = raw.get("aggregate") if isinstance(raw, dict) else None
            if isinstance(aggregate, dict):
                fraction = float(aggregate.get("kelly_fraction"))
                n_trades = int(aggregate.get("n_trades"))
                if math.isfinite(fraction) and fraction > 0 and n_trades >= min_trades:
                    kelly = {"fraction": fraction, "n": n_trades}
        except Exception:
            kelly = None
    kelly_fraction_cap = _env_float("KELLY_FRACTION_CAP", 0.25)
    kelly_max_mult = _env_float("KELLY_MAX_MULT", 1.5)

    # Whale/flow risk-veto thresholds (consumed by evaluate_risk via the
    # account mapping; the rules fail open when these keys are absent).
    cluster_insider_threshold = _env_int("CLUSTER_INSIDER_THRESHOLD", 3)
    flow_veto_min_holders = _env_int("FLOW_VETO_MIN_HOLDERS", 10)
    uoa_put_skew_veto = _env_float("UOA_PUT_SKEW_VETO", 0.30)

    # Try to fetch live equity + positions from Alpaca when credentials
    # are available.  Degradation is one-directional: a failure anywhere
    # inside the account fetch (equity or positions) keeps the fallback
    # equity and no positions -- sells will be skipped, buys unaffected.
    positions: Dict[str, float] = {}
    try:
        settings = get_settings()
        api_key = getattr(settings, "alpaca_api_key", None)
        secret_key = getattr(settings, "alpaca_secret_key", None)
        if api_key and secret_key:
            broker_client = AlpacaClient(key=api_key, secret=secret_key, paper_only=True)
            acct = broker_client.get_account()
            live_equity = float(acct.get("equity", 0))
            if live_equity > 0:
                equity = live_equity
            try:
                for row in broker_client.get_positions():
                    symbol = row.get("symbol")
                    qty = row.get("qty")
                    if not symbol or qty is None:
                        continue
                    try:
                        held = float(qty)
                    except (TypeError, ValueError):
                        continue
                    if held > 0:
                        positions[str(symbol).upper()] = held
            except Exception:
                positions = {}  # keep empty: sells will be skipped, buys unaffected
    except Exception:
        pass  # keep fallback equity + no positions

    components["account"] = {
        "equity": equity,
        "max_position_pct": max_position_pct,
        "target_daily_vol": target_daily_vol,
        "max_sector_pct": max_sector_pct,
        "regime_budget_mult": regime_budget_mult,
        "debate_oppose_mult": debate_oppose_mult,
        "cost_buffer_pct": cost_buffer_pct,
        "min_trade_notional": min_trade_notional,
        "min_trade_notional_pct": min_trade_notional_pct,
        "max_corr_pct": max_corr_pct,
        "invol_enabled": invol_enabled,
        "kelly": kelly,
        "kelly_fraction_cap": kelly_fraction_cap,
        "kelly_max_mult": kelly_max_mult,
        "cluster_insider_threshold": cluster_insider_threshold,
        "flow_veto_min_holders": flow_veto_min_holders,
        "uoa_put_skew_veto": uoa_put_skew_veto,
    }

    # -- pairs / stat-arb account config (consumed by the planner) --------
    components["account"]["pair_max_pct"] = _env_float("MAX_PAIR_PCT", 0.03)
    components["account"]["pairs_total_pct"] = _env_float("MAX_PAIRS_PCT", 0.30)
    components["account"]["pair_z_entry"] = _env_float("PAIR_Z_ENTRY", 2.0)
    components["account"]["pair_kelly_fraction_cap"] = _env_float(
        "PAIR_KELLY_FRACTION_CAP", 0.5)
    components["account"]["pair_kelly_max_mult"] = _env_float(
        "PAIR_KELLY_MAX_MULT", 1.5)
    components["account"]["pair_kelly"] = None
    if _flag_enabled("PAIR_KELLY_ENABLED"):
        try:
            from src.pairs.sizing import pair_kelly_from_stats

            components["account"]["pair_kelly"] = pair_kelly_from_stats(
                os.environ.get(
                    "PAIR_BACKTEST_STATS_PATH", "data/pairs_backtest_stats.json"
                ),
                min_trades=_env_int("KELLY_MIN_TRADES", 30),
            )
        except Exception:
            components["account"]["pair_kelly"] = None

    components["positions"] = positions

    # -- correlation clustering config (item 11; consumed by graph.py) --
    # Pure computation on already-fetched closes; CORR_CLUSTERING_ENABLED=0
    # disables it entirely (graph then passes no cluster map).
    components["correlation"] = {
        "enabled": _flag_enabled("CORR_CLUSTERING_ENABLED"),
        "threshold": _env_float("CORR_CLUSTER_THRESHOLD", 0.80),
        "min_observations": _env_int("CORR_MIN_OBSERVATIONS", 60),
    }

    # -- global liquidity & fear regime (fail-open to neutral) --
    # REGIME_ENABLED=0 skips fetching entirely (offline runs/tests); the
    # snapshot then stays neutral so no regime gate can fire.
    regime_enabled = os.environ.get("REGIME_ENABLED", "1").strip().lower() \
        not in ("0", "false", "no")
    # Gate-threshold overrides (item 12a): defaults match the module
    # constants in src/data/regime.py; junk env values fall back there.
    regime_thresholds = {
        "vix_backwardation": _env_float("VIX_BACKWARDATION", 1.0),
        "vix_panic_level": _env_float("VIX_PANIC_LEVEL", 30.0),
        "fear_z_threshold": _env_float("FEAR_Z_THRESHOLD", 1.0),
        "liquidity_z_floor": _env_float("LIQUIDITY_Z_FLOOR", -1.0),
    }
    if not regime_enabled:
        try:
            from src.data.regime import REGIME_NEUTRAL

            components["regime"] = dict(REGIME_NEUTRAL)
        except Exception:
            components["regime"] = None
    else:
        try:
            from src.data.regime import REGIME_NEUTRAL, fetch_regime

            cache_root = os.environ.get("REGIME_CACHE_ROOT", "data/cache")
            try:
                regime_cache = Cache(root=Path(cache_root))
            except Exception:
                regime_cache = None
            try:
                components["regime"] = fetch_regime(
                    cache=regime_cache, thresholds=regime_thresholds
                )
            except Exception:
                components["regime"] = dict(REGIME_NEUTRAL)
        except Exception:
            components["regime"] = None

    # -- pairs / stat-arb pipeline components (flag-gated, additive) --------
    # Registered ONLY when the caller passes --with-pairs and the
    # PAIRS_ENABLED flag is on; absent components leave the pipeline
    # pairs-free (graph degrades to no pair signals, single-leg unchanged).
    if with_pairs and _flag_enabled("PAIRS_ENABLED", "1"):
        try:
            from dataclasses import asdict

            from src.pairs.book import (
                PairBook,
                entry_from_candidate,
                pairs_book_path,
            )
            from src.pairs.risk import evaluate_pair_risk
            from src.pairs.screening import adv_map, form_pairs
            from src.pairs.signals import generate_pair_signals
            from src.pairs.universe import refresh_universe

            pair_settings = get_settings()
            rebalance_days = pair_settings.pair_rebalance_days
            regime_snapshot = components.get("regime")
            # Per-cycle price cache shared by the signals and risk
            # closures (the drift gates reuse the frames the signals
            # fetched; the "_frames" key cannot collide with a ticker).
            pair_price_cache: Dict[str, Any] = {}

            def _pair_price_frames(tickers, period=None):
                period = period or pair_settings.pair_price_period
                cache = _flow_cache(86400.0)  # 1-day TTL for daily bars
                frames: Dict[str, Any] = {}
                for ticker in tickers:
                    cached = pair_price_cache.get(ticker)
                    if cached is not None:
                        frames[ticker] = cached
                        continue
                    try:
                        frames[ticker] = fetch_ohlcv(
                            ticker, period=period, cache=cache
                        )
                    except Exception:
                        continue  # missing ticker -> dropped from the universe
                return frames

            def _pairs_form_if_due(book, as_of):
                from datetime import date as _date

                anchor = as_of or _date.today().isoformat()
                try:
                    due = book.formation_due(anchor, rebalance_days)
                except Exception:
                    due = True
                if not due:
                    return book
                try:
                    universe = refresh_universe(
                        cache_ttl_days=pair_settings.universe_refresh_days,
                        cache=_flow_cache(
                            pair_settings.universe_refresh_days * 86400.0
                        ),
                        fallback_path=pair_settings.universe_fallback_path,
                    )
                    frames = _pair_price_frames(sorted(universe))
                    advs = adv_map(
                        frames, lookback_days=pair_settings.coint_lookback_days
                    )
                    ranked = form_pairs(
                        frames,
                        sectors=universe,
                        lookback_days=pair_settings.coint_lookback_days,
                        min_half_life=pair_settings.coint_min_half_life,
                        max_half_life=pair_settings.coint_max_half_life,
                        max_pvalue=pair_settings.coint_adf_pvalue,
                        max_lag=pair_settings.coint_max_lag,
                        prescreen_corr=pair_settings.pair_prescreen_corr,
                        z_entry=pair_settings.pair_z_entry,
                        z_exit=pair_settings.pair_z_exit,
                        z_stop=pair_settings.pair_z_stop,
                        max_pairs=pair_settings.max_pairs,
                        max_per_sector=pair_settings.max_pairs_per_sector,
                        min_sharpe=pair_settings.min_pair_sharpe,
                    )
                    entries = [
                        entry_from_candidate(
                            item.candidate, closes=frames, advs=advs
                        )
                        for item in ranked
                    ]
                    book.replace(entries, as_of=anchor)
                    print(f"pairs: formed {len(entries)} pair(s) from "
                          f"{len(frames)} universe name(s)")
                except Exception as exc:
                    print(f"warning: pairs formation failed ({exc}); "
                          f"continuing with the existing book")
                return book

            def _pairs_signals(book):
                try:
                    tickers = book.tickers()
                    if not tickers:
                        return []
                    frames = _pair_price_frames(tickers, period="5d")
                    pair_price_cache["_frames"] = frames
                    prices: Dict[str, float] = {}
                    for ticker, frame in frames.items():
                        close: Any = frame
                        if hasattr(close, "columns") and "Close" in close.columns:
                            close = close["Close"]
                        if hasattr(close, "iloc"):
                            close = close.iloc[-1]
                        try:
                            value = float(close)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(value) and value > 0.0:
                            prices[ticker] = value
                    signals = generate_pair_signals(
                        book,
                        prices,
                        z_entry=pair_settings.pair_z_entry,
                        z_exit=pair_settings.pair_z_exit,
                        z_stop=pair_settings.pair_z_stop,
                    )
                    return [asdict(signal) for signal in signals]
                except Exception as exc:
                    print(f"warning: pairs signal generation failed ({exc})")
                    return []

            def _pairs_risk(signal, book):
                pair_id = (
                    signal.get("pair_id")
                    if isinstance(signal, dict)
                    else getattr(signal, "pair_id", "")
                )
                try:
                    frames = pair_price_cache.get("_frames") or {}
                    entry = book.get(pair_id) if hasattr(book, "get") else None
                    decision = evaluate_pair_risk(
                        signal,
                        entry,
                        regime=regime_snapshot,
                        current_closes=frames,
                        current_adv=adv_map(frames, lookback_days=60),
                    )
                    return decision.to_dict()
                except Exception as exc:
                    return {
                        "pair_id": pair_id,
                        "approved": False,
                        "veto_reasons": [f"risk evaluation error: {exc}"],
                    }

            components["pair_book"] = PairBook.load(
                pairs_book_path(str(pair_settings.pair_book_path)))
            components["pairs_form_if_due"] = _pairs_form_if_due
            components["pairs_signals"] = _pairs_signals
            components["pairs_risk"] = _pairs_risk
        except Exception as exc:
            print(f"warning: pairs module unavailable ({exc}); "
                  f"continuing without pairs")

    # -- day trade dates --
    dates_str = os.environ.get("DAY_TRADE_DATES", "")
    try:
        components["day_trade_dates"] = [d.strip() for d in dates_str.split(",") if d.strip()]
    except Exception:
        components["day_trade_dates"] = []

    # -- data fetch closures --
    # Returns the FULL OHLCV frame (fetch_ohlcv already flattens yfinance
    # MultiIndex columns): graph.py derives the close series for the
    # analysts/correlation and hands the whole frame to generate_signal so
    # the advisory volume features (volume_z / volume_trend) are computed.
    def _fetch_ohlcv(ticker: str, period: str = "6mo"):
        return fetch_ohlcv(ticker, period=period)

    components["fetch_ohlcv"] = _fetch_ohlcv

    def _fetch_fundamentals(ticker: str):
        return fetch_fundamentals(ticker)

    components["fetch_fundamentals"] = _fetch_fundamentals

    def _fetch_news(ticker: str):
        return fetch_headlines(ticker, limit=10)

    components["fetch_news"] = _fetch_news

    # -- whale/flow fetch closures (flag-gated) -----------------------------
    # Each closure is registered ONLY when its *_ENABLED flag is on; absent
    # component keys mean graph.process_ticker leaves that slice at None and
    # the corresponding analyst is not dispatched.  Modules import their
    # heavy dependencies lazily, so these imports never touch the network.
    if _flag_enabled("INSIDER_ENABLED"):
        try:
            from src.data.insider import TTL_SECONDS as _INSIDER_TTL
            from src.data.insider import fetch_insider_activity

            def _fetch_insider(ticker: str):
                days = _env_int("INSIDER_LOOKBACK_DAYS", 90)
                return fetch_insider_activity(
                    ticker, cache=_flow_cache(_INSIDER_TTL), days=days
                )

            components["fetch_insider"] = _fetch_insider
        except Exception:
            pass  # fetchers are optional; keep the CLI working without them

    if _flag_enabled("INSTITUTIONAL_ENABLED"):
        try:
            from src.data.institutional import TTL_SECONDS as _INSTITUTIONAL_TTL
            from src.data.institutional import fetch_institutional_holdings

            def _fetch_institutional(ticker: str):
                return fetch_institutional_holdings(
                    ticker, cache=_flow_cache(_INSTITUTIONAL_TTL)
                )

            components["fetch_institutional"] = _fetch_institutional
        except Exception:
            pass

    if _flag_enabled("OPTIONS_FLOW_ENABLED"):
        try:
            from src.data.options_flow import TTL_SECONDS as _OPTIONS_FLOW_TTL
            from src.data.options_flow import fetch_unusual_options

            async def _fetch_options_flow(ticker: str):
                return await fetch_unusual_options(
                    ticker,
                    cache=_flow_cache(_OPTIONS_FLOW_TTL),
                    min_vol_oi=_env_float("UOA_MIN_VOL_OI_RATIO", 3.0),
                    min_premium=_env_float("UOA_MIN_PREMIUM", 25000.0),
                    max_dte=_env_int("UOA_MAX_DTE", 45),
                )

            components["fetch_options_flow"] = _fetch_options_flow
        except Exception:
            pass

    if _flag_enabled("DARK_POOL_ENABLED"):
        try:
            from src.data.dark_pool import TTL_SECONDS as _DARK_POOL_TTL
            from src.data.dark_pool import fetch_dark_pool_proxy

            def _fetch_dark_pool(ticker: str):
                return fetch_dark_pool_proxy(ticker, cache=_flow_cache(_DARK_POOL_TTL))

            components["fetch_dark_pool"] = _fetch_dark_pool
        except Exception:
            pass

    # -- real functions by identity (no wrapping) --
    components["generate_signal"] = generate_signal
    components["run_analysts"] = run_analysts
    components["run_debate"] = run_debate
    components["evaluate_risk"] = evaluate_risk
    components["build_order_plan"] = build_order_plan

    # -- broker submission --
    def _broker_noop(orders):
        return []

    broker_submit = _broker_noop

    if submit:
        try:
            settings = get_settings()
            api_key = getattr(settings, "alpaca_api_key", None)
            secret_key = getattr(settings, "alpaca_secret_key", None)
            if api_key and secret_key:
                broker_client = AlpacaClient(key=api_key, secret=secret_key, paper_only=True)

                def _broker_submit(orders, _client=broker_client):
                    return _client.submit_limit_orders(orders)

                broker_submit = _broker_submit
        except Exception:
            broker_submit = _broker_noop

    components["broker_submit"] = broker_submit

    return components


def _print_decision(decision: dict) -> None:
    if decision.get("kind") == "pair":
        pair = decision.get("pair") or {}
        veto = decision.get("veto") or {}
        approved = veto.get("approved", False)
        z = pair.get("z_score")
        z_text = f"{float(z):.2f}" if isinstance(z, (int, float)) else "?"
        print(f"PAIR {pair.get('pair_id', decision.get('ticker', '?'))} "
              f"z={z_text} approved={approved}"
              + (f" vetoes={veto.get('veto_reasons')}"
                 if veto.get("veto_reasons") else ""))
        return
    ticker = decision.get("ticker", "?")
    quant = decision.get("quant") or {}
    signal = quant.get("signal", "?")
    confidence = quant.get("confidence", 0.0)
    veto = decision.get("veto") or {}
    approved = veto.get("approved", False)
    print(f"{ticker} {signal} {confidence} approved={approved}")


async def _await_and_close(awaitable: Any, components: Dict[str, Any]) -> Any:
    """Await a pipeline coroutine, then close the LLM client on the loop.

    ``asyncio.run`` builds and closes a fresh event loop per call; an LLM
    client cached across cycles would have its connection-close tasks
    finalized on a later (already-closed) loop, surfacing as
    ``RuntimeError: Event loop is closed`` noise.  Closing the client
    HERE -- inside the loop -- prevents that for every entry point.
    """
    try:
        return await awaitable
    finally:
        closer = getattr(components.get("client"), "aclose", None)
        if callable(closer):
            try:
                await closer()
            except Exception:
                pass


def _run_and_print(tickers: List[str], submit: bool = False,
                   with_pairs: bool = False) -> None:
    components = _components(submit=submit, with_pairs=with_pairs)
    result = run_pipeline(tickers, components)
    if inspect.isawaitable(result):
        result = asyncio.run(_await_and_close(result, components))

    store = _store()
    for decision in result or []:
        _print_decision(decision)
        try:
            if decision.get("kind") == "pair":
                store.log_pair_decision(_jsonable(decision))
            else:
                store.log_decision(_jsonable(decision))
        except Exception as exc:
            print(f"warning: failed to persist decision: {exc}")


def _parse_tickers(tickers: Optional[str]) -> List[str]:
    if tickers is None:
        return list(get_settings().watchlist)
    return [t.strip() for t in tickers.split(",") if t.strip()]


@app.command("scan")
def scan(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    with_pairs: bool = typer.Option(
        False, "--with-pairs",
        help="Include the pairs (stat-arb) module in the batch",
    ),
) -> None:
    ticker_list = _parse_tickers(tickers)
    _run_and_print(ticker_list, submit=False, with_pairs=with_pairs)


@app.command("paper-run")
def paper_run(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    with_pairs: bool = typer.Option(
        False, "--with-pairs",
        help="Include the pairs (stat-arb) module in the batch",
    ),
) -> None:
    settings = get_settings()
    if not (getattr(settings, "alpaca_api_key", None) and getattr(settings, "alpaca_secret_key", None)):
        print("error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for paper-run")
        raise typer.Exit(code=1)
    ticker_list = _parse_tickers(tickers)
    _run_and_print(ticker_list, submit=True, with_pairs=with_pairs)


@app.command("decisions")
def decisions() -> None:
    rows = _store().tail(20)
    for row in rows:
        print(f"{row.get('ticker')} {row}")


@app.command("backtest")
def backtest(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    period: str = typer.Option("2y", "--period", help="History window per ticker"),
    horizon: int = typer.Option(5, "--horizon", help="Forward days a signal is scored over"),
    mode: str = typer.Option(
        "close", "--mode",
        help="close: close->close over --horizon; overnight: close->next open (halo/drift)",
    ),
    primary: bool = typer.Option(
        False, "--primary",
        help="Overnight mode only: also write the BUY-side stats to "
             "BACKTEST_STATS_PATH so intraday Kelly sizing uses the "
             "overnight edge (the production regime).",
    ),
    spread_bps: Optional[float] = typer.Option(
        None, "--spread-bps",
        help="Bid-ask spread in basis points (default: SPREAD_BPS env or 10.0)"),
    slippage_bps: Optional[float] = typer.Option(
        None, "--slippage-bps",
        help="Market order slippage in basis points (default: SLIPPAGE_BPS env or 3.0)"),
    cost_scale_by_vol: Optional[bool] = typer.Option(
        None, "--cost-scale-by-vol/--no-cost-scale-by-vol",
        help="Scale costs by daily volatility (default: COST_SCALE_BY_VOL env or on)"),
    random_baseline: bool = typer.Option(False, "--random-baseline", help="Also run random signal baseline for comparison"),
    random_signal_rate: float = typer.Option(0.1, "--random-signal-rate", help="Signal rate for random baseline (fraction of bars)"),
    random_seed: int = typer.Option(42, "--random-seed", help="RNG seed for random baseline"),
) -> None:
    """Deterministic signal replay: writes Kelly stats for the sizer.

    Replays the quant signal generator over historical prices (no LLM,
    no broker) and persists aggregate win-rate/payoff/Kelly stats, where
    paper-run/scan pick them up when KELLY_ENABLED=1.  ``--mode close``
    (default) scores close->close moves over --horizon and writes to
    BACKTEST_STATS_PATH (default data/backtest_stats.json); ``--mode
    overnight`` scores the halo/overnight-drift strategy (enter at the
    signal-day close, exit next open) and writes to OVERNIGHT_STATS_PATH
    (default data/backtest_stats_overnight.json).  The two files are
    separate on purpose: overnight experiments never feed intraday sizing.

    With ``--spread-bps``/``--slippage-bps``, execution costs are modeled
    (Alpaca paper has no commission).  With ``--random-baseline``, a
    random signal baseline is run alongside for statistical validation.
    """
    from src.backtest.replay import run_backtest

    if mode not in ("close", "overnight"):
        print(f"error: unknown backtest mode {mode!r} (expected 'close' or 'overnight')")
        raise typer.Exit(code=1)

    costs = _costs_from_options(spread_bps, slippage_bps, cost_scale_by_vol)

    ticker_list = _parse_tickers(tickers)
    min_trades = _env_int("KELLY_MIN_TRADES", 30)
    if mode == "overnight":
        out_path = Path(os.environ.get(
            "OVERNIGHT_STATS_PATH", "data/backtest_stats_overnight.json"))
    else:
        out_path = Path(os.environ.get("BACKTEST_STATS_PATH", "data/backtest_stats.json"))
    stats = run_backtest(ticker_list, period=period, horizon=horizon,
                         min_trades=min_trades, mode=mode, costs=costs)
    
    # Random baseline comparison (null distribution scored under the SAME
    # contract as the strategy: close->close for --mode close, close->next
    # open for --mode overnight).
    if random_baseline:
        from src.backtest.replay import replay_random, summarize_trades

        print(f"\n--- Random Baseline (signal_rate={random_signal_rate}, "
              f"seed={random_seed}) ---")
        horizon_eff = 1 if mode == "overnight" else horizon
        random_all_trades: list[dict] = []
        for t in ticker_list:
            try:
                data = fetch_ohlcv(t, period=period)
                random_all_trades.extend(replay_random(
                    data, horizon=horizon_eff, signal_rate=random_signal_rate,
                    seed=random_seed, costs=costs, mode=mode,
                ))
            except Exception:
                pass  # degrade per ticker, same as the strategy replay
        random_agg = summarize_trades(
            random_all_trades, min_trades=min_trades, horizon=horizon_eff,
        )
        print(f"Random: n={random_agg.get('n_trades', 0)} "
              f"win_rate={random_agg.get('win_rate')} "
              f"kelly={random_agg.get('kelly_fraction')} "
              f"sharpe={random_agg.get('sharpe')}")
        # Record the baseline alongside the strategy stats for the audit trail.
        stats["random_baseline"] = {
            "signal_rate": random_signal_rate,
            "seed": random_seed,
            "aggregate": random_agg,
        }

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"warning: failed to persist stats: {exc}")

    aggregate = stats.get("aggregate") or {}
    print(f"mode={mode} aggregate: n={aggregate.get('n_trades', 0)} "
          f"win_rate={aggregate.get('win_rate')} "
          f"payoff={aggregate.get('payoff_ratio')} "
          f"kelly={aggregate.get('kelly_fraction')} "
          f"usable={aggregate.get('usable')} "
          f"sharpe={aggregate.get('sharpe')} "
          f"sortino={aggregate.get('sortino')} "
          f"calmar={aggregate.get('calmar')} "
          f"omega={aggregate.get('omega')}")
    for ticker, summary in (stats.get("per_ticker") or {}).items():
        print(f"{ticker}: n={summary.get('n_trades', 0)} "
              f"win_rate={summary.get('win_rate')} "
              f"kelly={summary.get('kelly_fraction')} "
              f"sharpe={summary.get('sharpe')} "
              f"sortino={summary.get('sortino')} "
              f"calmar={summary.get('calmar')} "
              f"omega={summary.get('omega')}")
    print(f"stats written to {out_path}")
    if primary:
        from src.backtest.replay import primary_payload

        payload = primary_payload(stats)
        if payload is None:
            print("warning: --primary requires --mode overnight; skipped")
        else:
            primary_path = Path(os.environ.get(
                "BACKTEST_STATS_PATH", "data/backtest_stats.json"))
            try:
                primary_path.parent.mkdir(parents=True, exist_ok=True)
                primary_path.write_text(json.dumps(payload, indent=2),
                                        encoding="utf-8")
                buy = payload.get("aggregate") or {}
                print(f"primary (overnight BUY) written to {primary_path}: "
                      f"n={buy.get('n_trades')} kelly={buy.get('kelly_fraction')} "
                      f"sharpe={buy.get('sharpe')}")
            except Exception as exc:
                print(f"warning: failed to persist primary stats: {exc}")


@app.command("backtest-sweep")
def backtest_sweep(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    periods: str = typer.Option(
        "1y,2y,3y,5y,10y,max", "--periods",
        help="Comma separated history windows"),
    horizons: str = typer.Option(
        "1,5,10,20", "--horizons",
        help="Comma separated horizons (close mode only; overnight is always 1)"),
    modes: str = typer.Option(
        "close,overnight", "--modes",
        help="Comma separated replay modes"),
    param: list[str] = typer.Option(
        None, "--param",
        help="Parameter grid: NAME=v1,v2,... (repeatable). Allowed: Z_BUY_ENTRY, Z_SELL_ENTRY, MOMENTUM_FLOOR_PCT, VOLUME_Z_MIN_BUY. Empty value means unset."),
    out: Optional[str] = typer.Option(
        None, "--out", help="Output JSON path (default: data/backtest_sweep.json or data/backtest_param_sweep.json if --param used)"),
    spread_bps: Optional[float] = typer.Option(
        None, "--spread-bps",
        help="Bid-ask spread in basis points (default: SPREAD_BPS env or 10.0)"),
    slippage_bps: Optional[float] = typer.Option(
        None, "--slippage-bps",
        help="Market order slippage in basis points (default: SLIPPAGE_BPS env or 3.0)"),
    cost_scale_by_vol: Optional[bool] = typer.Option(
        None, "--cost-scale-by-vol/--no-cost-scale-by-vol",
        help="Scale costs by daily volatility (default: COST_SCALE_BY_VOL env or on)"),
) -> None:
    """Grid sweep of the replay backtest (mode x period x horizon).

    Reuses the deterministic replay (no LLM, no broker) and writes the
    full aggregate + per-ticker stats matrix as JSON for offline
    analysis.  Slow on purpose: 'max' fetches full history per ticker
    and every combo replays the signal bar by bar.

    With --param, the sweep also iterates over environment-variable
    combinations (whitelisted signal thresholds).  Each combo temporarily
    sets the env vars, runs the replay on pre-fetched data, and restores
    the original environment.

    With ``--spread-bps``/``--slippage-bps``, execution costs are modeled
    for all combos in the sweep.
    """
    from src.backtest.sweep import parse_list, run_sweep, run_param_sweep, parse_param_option

    ticker_list = _parse_tickers(tickers)
    period_list = parse_list(periods, ["2y"])

    def _to_int(raw: str) -> Optional[int]:
        try:
            value = int(float(raw))
        except (TypeError, ValueError):
            return None
        return value if value >= 1 else None

    horizon_list = [v for v in (_to_int(h) for h in parse_list(horizons, ["5"]))
                    if v is not None] or [5]
    mode_list = parse_list(modes, ["close", "overnight"])
    min_trades = _env_int("KELLY_MIN_TRADES", 30)

    costs = _costs_from_options(spread_bps, slippage_bps, cost_scale_by_vol)

    # Parse --param options into param_grid
    param_grid: dict[str, list[str]] = {}
    if param:
        for p in param:
            name, values = parse_param_option(p)
            if name in param_grid:
                param_grid[name].extend(values)
            else:
                param_grid[name] = values

    # Determine default output path
    if out is None:
        out = "data/backtest_param_sweep.json" if param_grid else "data/backtest_sweep.json"

    if param_grid:
        sweep = run_param_sweep(
            ticker_list, periods=period_list, horizons=horizon_list,
            modes=mode_list, param_grid=param_grid, min_trades=min_trades,
            costs=costs,
        )
    else:
        sweep = run_sweep(ticker_list, periods=period_list, horizons=horizon_list,
                          modes=mode_list, min_trades=min_trades, costs=costs)

    out_path = Path(out)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"warning: failed to persist sweep: {exc}")
    for row in sweep.get("results") or []:
        if row.get("error"):
            params_str = f" params={row.get('params')}" if row.get("params") else ""
            print(f"{row.get('mode')}/{row.get('period')}{params_str}: error {row['error']}")
            continue
        agg = row.get("aggregate") or {}
        params_str = f" params={row.get('params')}" if row.get("params") else ""
        print(f"{row.get('mode')}/{row.get('period')}/h{row.get('horizon_days')}{params_str}: "
              f"n={agg.get('n_trades', 0)} win_rate={agg.get('win_rate')} "
              f"kelly={agg.get('kelly_fraction')} sharpe={agg.get('sharpe')} "
              f"sortino={agg.get('sortino')} calmar={agg.get('calmar')} "
              f"omega={agg.get('omega')}")
    print(f"sweep written to {out_path}")


@app.command("backtest-walkforward")
def backtest_walkforward(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    period: str = typer.Option("5y", "--period", help="History window per ticker"),
    train_ratio: float = typer.Option(0.7, "--train-ratio", help="Fraction of data for initial training (0-1)"),
    step_days: int = typer.Option(63, "--step-days", help="Step size in days (re-optimize/evaluate frequency)"),
    min_train_days: int = typer.Option(252, "--min-train-days", help="Minimum training bars before first test"),
    horizon: int = typer.Option(5, "--horizon", help="Forward days a signal is scored over"),
    min_trades: int = typer.Option(30, "--min-trades", help="Minimum trades per fold for Kelly"),
    mode: str = typer.Option("close", "--mode", help="close or overnight"),
    out: Optional[str] = typer.Option(None, "--out", help="Output JSON path (default: data/backtest_walkforward.json)"),
    spread_bps: Optional[float] = typer.Option(
        None, "--spread-bps",
        help="Bid-ask spread in basis points (default: SPREAD_BPS env or 10.0)"),
    slippage_bps: Optional[float] = typer.Option(
        None, "--slippage-bps",
        help="Market order slippage in basis points (default: SLIPPAGE_BPS env or 3.0)"),
    cost_scale_by_vol: Optional[bool] = typer.Option(
        None, "--cost-scale-by-vol/--no-cost-scale-by-vol",
        help="Scale costs by daily volatility (default: COST_SCALE_BY_VOL env or on)"),
) -> None:
    """Walk-forward / out-of-sample backtest with expanding windows.

    For each ticker, splits price history into sequential folds:
    - Train window: first ``train_ratio`` of data (min ``min_train_days``)
    - Test window: next ``step_days`` bars
    - Then expand train by ``step_days`` and repeat until end of data

    Only trades whose entry falls inside the test window are collected
    (genuinely out-of-sample; the signal generator still sees the full
    history up to each bar).  Evaluates fixed parameters -- for
    re-optimization use ``backtest-sweep --param`` per train window.
    """
    from src.backtest.sweep import run_walkforward

    if mode not in ("close", "overnight"):
        print(f"error: unknown mode {mode!r} (expected 'close' or 'overnight')")
        raise typer.Exit(code=1)

    costs = _costs_from_options(spread_bps, slippage_bps, cost_scale_by_vol)

    ticker_list = _parse_tickers(tickers)
    min_trades_env = _env_int("KELLY_MIN_TRADES", 30)
    min_trades = max(min_trades, min_trades_env)

    out_path = Path(out) if out else Path("data/backtest_walkforward.json")
    result = run_walkforward(
        ticker_list, period=period, train_ratio=train_ratio,
        step_days=step_days, min_train_days=min_train_days,
        horizon=horizon, min_trades=min_trades, mode=mode,
        costs=costs,
    )

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"warning: failed to persist walkforward: {exc}")

    print(f"walkforward: mode={result.get('mode')} period={result.get('period')}")
    print(f"  train_ratio={result.get('train_ratio')} step_days={result.get('step_days')}")
    print(f"  folds={len(result.get('folds', []))}")
    for fold in result.get("folds", []):
        if "error" in fold:
            print(f"  fold {fold.get('fold')}: {fold.get('ticker')} ERROR={fold['error']}")
        else:
            agg = fold.get("aggregate") or {}
            print(f"  fold {fold.get('fold')}: {fold.get('ticker')} "
                  f"train={fold.get('train_bars')} test={fold.get('test_bars')} "
                  f"n={agg.get('n_trades', 0)} win_rate={agg.get('win_rate')} "
                  f"kelly={agg.get('kelly_fraction')} sharpe={agg.get('sharpe')}")
    oos_agg = result.get("aggregate_oos") or {}
    print(f"  OOS aggregate: n={oos_agg.get('n_trades', 0)} "
          f"win_rate={oos_agg.get('win_rate')} kelly={oos_agg.get('kelly_fraction')} "
          f"sharpe={oos_agg.get('sharpe')}")
    print(f"walkforward written to {out_path}")


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_num(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_dd(value: Any) -> str:
    """Format a positive drawdown fraction as a negative percentage."""
    if value is None:
        return "n/a"
    try:
        return f"{-float(value) * 100:+.1f}%"
    except (TypeError, ValueError):
        return "n/a"


@app.command("portfolio-backtest")
def portfolio_backtest(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    period: str = typer.Option("2y", "--period", help="History window per ticker"),
    top_n: int = typer.Option(5, "--top-n", help="Max concurrent overnight entries per day"),
    stakes: str = typer.Option("0.20,0.10", "--stakes", help="Comma separated per-trade equity stakes"),
    cost_bps: float = typer.Option(10.0, "--cost-bps", help="Round-trip cost per trade (basis points; used when --spread-bps/--slippage-bps are absent)"),
    spread_bps: Optional[float] = typer.Option(
        None, "--spread-bps",
        help="Bid-ask spread in basis points (enables the spread+slippage cost model)"),
    slippage_bps: Optional[float] = typer.Option(
        None, "--slippage-bps",
        help="Market order slippage in basis points (enables the spread+slippage cost model)"),
    cost_scale_by_vol: Optional[bool] = typer.Option(
        None, "--cost-scale-by-vol/--no-cost-scale-by-vol",
        help="Scale costs by daily volatility (default: COST_SCALE_BY_VOL env or on)"),
    benchmark: str = typer.Option("SPY", "--benchmark", help="Buy-and-hold benchmark ticker"),
    out: Optional[str] = typer.Option(None, "--out", help="Output JSON path (default: data/portfolio_backtest.json)"),
) -> None:
    """Portfolio-level sim of the overnight BUY edge vs buy-and-hold.

    Simulates the production overnight strategy as an actual portfolio:
    top --top-n BUY candidates per day (confidence-ranked), each sized
    --stakes of current equity, entering at the signal-day close and
    exiting at the next open with costs, equity compounding daily.
    Benchmarked against a --benchmark buy-and-hold over the same dates.

    Costs can be specified either as a simple round-trip ``--cost-bps``
    or as separate ``--spread-bps`` and ``--slippage-bps`` (which also
    enables volatility scaling via ``--cost-scale-by-vol``).
    """
    from src.backtest.portfolio import run_portfolio_backtest

    ticker_list = _parse_tickers(tickers)
    stake_list = []
    for raw in stakes.split(","):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            stake_list.append(value)
    if not stake_list:
        stake_list = [0.20, 0.10]

    # Cost model only when spread/slippage explicitly requested (either
    # flag); otherwise the flat --cost-bps round-trip applies.
    costs = None
    if spread_bps is not None or slippage_bps is not None:
        costs = _costs_from_options(spread_bps, slippage_bps, cost_scale_by_vol)

    payload = run_portfolio_backtest(
        ticker_list, period=period, top_n=top_n, stakes=tuple(stake_list),
        cost_bps=cost_bps, benchmark=benchmark, costs=costs,
    )

    out_path = Path(out) if out else Path("data/portfolio_backtest.json")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"warning: failed to persist portfolio backtest: {exc}")

    if payload.get("error"):
        print(f"error: {payload['error']}")
        raise typer.Exit(code=1)

    bench = payload.get("benchmark") or {}
    spread_info = f" spread_bps={payload.get('spread_bps')}" if payload.get('spread_bps') else ""
    slippage_info = f" slippage_bps={payload.get('slippage_bps')}" if payload.get('slippage_bps') else ""
    print(f"portfolio-backtest: mode={payload.get('mode')} period={payload.get('period')} "
          f"top_n={payload.get('top_n')} cost_bps={payload.get('cost_bps')}{spread_info}{slippage_info} "
          f"benchmark={payload.get('benchmark_ticker')}")
    print(f"tickers used={payload.get('n_tickers_used')}/{payload.get('n_tickers_requested')} "
          f"grid_days={payload.get('grid_days')} "
          f"({payload.get('first_date')} .. {payload.get('last_date')})")
    for stake_key, result in (payload.get("stakes") or {}).items():
        port = result.get("portfolio") or {}
        comp = result.get("comparison") or {}
        print(f"stake {stake_key}: portfolio total={_fmt_pct(port.get('total_return'))} "
              f"cagr={_fmt_pct(port.get('cagr'))} sharpe={_fmt_num(port.get('sharpe'))} "
              f"sortino={_fmt_num(port.get('sortino'))} maxdd={_fmt_dd(port.get('max_drawdown'))} "
              f"exposure={_fmt_pct(port.get('exposure'))} trades={port.get('n_trades', 0)}")
        print(f"            benchmark total={_fmt_pct(bench.get('total_return'))} "
              f"cagr={_fmt_pct(bench.get('cagr'))} sharpe={_fmt_num(bench.get('sharpe'))} "
              f"maxdd={_fmt_dd(bench.get('max_drawdown'))}")
        print(f"            vs benchmark: excess_cagr={_fmt_pct(comp.get('excess_cagr'))} "
              f"sharpe_delta={_fmt_num(comp.get('sharpe_delta'))} "
              f"maxdd_delta={_fmt_pct(comp.get('maxdd_delta'))} "
              f"corr={_fmt_num(comp.get('daily_return_corr'))}")
    print(f"portfolio backtest written to {out_path}")


# --------------------------------------------------------------------------- #
# Pairs (stat-arb) commands                                                    #
# --------------------------------------------------------------------------- #


@app.command("pairs-discover")
def pairs_discover(
    force: bool = typer.Option(
        False, "--force",
        help="Re-form now even if the rebalance schedule is not due",
    ),
) -> None:
    """Form / re-form the persisted pair book.

    Runs the full discovery pipeline (S&P 400 universe -> Engle-Granger
    with pre-screen -> walk-forward ranking -> dedupe) and atomically
    replaces ``PAIR_BOOK_PATH``.  Without ``--force`` the formation only
    runs when the book is missing/stale per ``PAIR_REBALANCE_DAYS``.
    """
    from datetime import date

    from src.pairs.book import PairBook

    components = _components(with_pairs=True)
    book = components.get("pair_book")
    if book is None:
        print("error: pairs module unavailable "
              "(check PAIRS_ENABLED / statsmodels / lxml)")
        raise typer.Exit(code=1)
    if force:
        book.entries = {}
        book.formed_on = None
    form = components.get("pairs_form_if_due")
    if form is not None:
        book = form(book, date.today().isoformat())
    print(f"pair book: {len(book)} pair(s) formed_on={book.formed_on}")
    for entry in book.entries.values():
        print(f"  {entry.pair_id}: beta={entry.hedge_ratio:.3f} "
              f"half_life={entry.half_life:.1f}d "
              f"adf_p={entry.adf_pvalue} "
              f"corr={entry.formation_corr} sector={entry.sector_a}")
    print(f"book written to {book.path}")


@app.command("pairs-scan")
def pairs_scan() -> None:
    """Analysis only: spread signals + risk vetoes for the active book."""
    components = _components(with_pairs=True)
    book = components.get("pair_book")
    if book is None:
        print("error: pairs module unavailable "
              "(check PAIRS_ENABLED / statsmodels / lxml)")
        raise typer.Exit(code=1)
    if len(book) == 0:
        print("no active pairs (run pairs-discover first)")
        raise typer.Exit(code=1)
    signals_fn = components.get("pairs_signals")
    risk_fn = components.get("pairs_risk")
    signals = signals_fn(book) if signals_fn else []
    if not signals:
        print("no tradable pair signals (missing/stale prices?)")
        return
    for signal in signals:
        decision = (
            risk_fn(signal, book)
            if risk_fn
            else {"pair_id": signal.get("pair_id"), "approved": True,
                  "veto_reasons": []}
        )
        status = "APPROVED" if decision.get("approved") else "VETOED"
        print(f"{signal.get('pair_id')} z={float(signal.get('z_score')):.2f} "
              f"{signal.get('ticker_a')}={signal.get('signal_a')} "
              f"{signal.get('ticker_b')}={signal.get('signal_b')} -> {status}")
        for reason in decision.get("veto_reasons") or []:
            print(f"    veto: {reason}")


@app.command("pairs-backtest")
def pairs_backtest(
    tickers: Optional[str] = typer.Option(
        None, "--tickers",
        help="Comma separated tickers (default: the S&P 400 universe; required when --param is used)"),
    period: str = typer.Option("2y", "--period", help="History window"),
    formation: int = typer.Option(
        252, "--formation", help="Formation window in trading days"),
    rebalance: int = typer.Option(
        21, "--rebalance", help="Re-discovery interval in trading days"),
    param: list[str] = typer.Option(
        None, "--param",
        help="Parameter grid: NAME=v1,v2,... (repeatable). Allowed: z_entry, z_exit, z_stop, max_pvalue, min_half_life, max_half_life, prescreen_corr, rebalance_days, formation_window, max_pairs, max_per_sector, min_sharpe. Requires --tickers."),
    out: Optional[str] = typer.Option(
        None, "--out", help="Output JSON path (default: data/pairs_backtest_stats.json or data/pairs_param_sweep.json if --param used)"),
    spread_bps: Optional[float] = typer.Option(
        None, "--spread-bps",
        help="Bid-ask spread in basis points (default: SPREAD_BPS env or 10.0)"),
    slippage_bps: Optional[float] = typer.Option(
        None, "--slippage-bps",
        help="Market order slippage in basis points (default: SLIPPAGE_BPS env or 3.0)"),
    cost_scale_by_vol: Optional[bool] = typer.Option(
        None, "--cost-scale-by-vol/--no-cost-scale-by-vol",
        help="Scale costs by daily volatility (default: COST_SCALE_BY_VOL env or on)"),
) -> None:
    """Walk-forward pairs backtest; writes PAIR_BACKTEST_STATS_PATH.

    Re-forms pairs every --rebalance bars and trades the long-only
    conditioned mean-reversion strategy on the following bars (no LLM,
    no broker).  The aggregate Kelly stats feed PAIR_KELLY_* sizing
    when PAIR_KELLY_ENABLED=1.  Slow on purpose: the default universe
    fetches ~400 names over --period.

    With --param, the sweep iterates over parameter combinations.
    Each combo runs the full walk-forward backtest on pre-fetched data.
    Requires explicit --tickers (full S&P 400 universe is too slow for sweeps).

    With ``--spread-bps``/``--slippage-bps``, execution costs are modeled
    for each leg trade (no vol scaling for pairs).
    """
    from src.data.market_data import fetch_ohlcv
    from src.pairs.backtest import run_pairs_backtest, run_pairs_param_sweep, parse_pairs_param_option
    from src.pairs.universe import refresh_universe

    settings = get_settings()
    # No --tickers means the full S&P 400 universe (per the help text);
    # the project watchlist default of _parse_tickers must not win here.
    ticker_list = (
        [t.strip() for t in tickers.split(",") if t.strip()]
        if tickers else []
    )
    universe: Dict[str, Any] = {}
    if not ticker_list:
        universe = refresh_universe(
            cache_ttl_days=settings.universe_refresh_days,
            cache=_flow_cache(settings.universe_refresh_days * 86400.0),
            fallback_path=settings.universe_fallback_path,
        )
        ticker_list = sorted(universe)
    if not ticker_list:
        print("error: no universe available (scrape failed and no "
              "cache/fallback file); pass --tickers explicitly")
        raise typer.Exit(code=1)
    sectors = {t: universe.get(t) for t in ticker_list} if universe else None

    # Parse --param options into param_grid
    param_grid: dict[str, list[str]] = {}
    if param:
        for p in param:
            name, values = parse_pairs_param_option(p)
            if name in param_grid:
                param_grid[name].extend(values)
            else:
                param_grid[name] = values

    # When sweeping, require explicit tickers (full universe too slow)
    if param_grid and not tickers:
        print("error: --param requires explicit --tickers (full S&P 400 universe is too slow for sweeps)")
        raise typer.Exit(code=1)
    if param_grid and len(ticker_list) > 80:
        print(f"warning: sweeping {len(ticker_list)} tickers x "
              f"{sum(len(v) for v in param_grid.values())} grid points; "
              "each combo re-runs pair formation (ADF) -- expect a long run")

    price_cache = _flow_cache(86400.0)

    def _fetch(ticker: str, fetch_period: str):
        return fetch_ohlcv(ticker, period=fetch_period, cache=price_cache)

    base_params = {
        "period": period,
        "formation_window": formation,
        "rebalance_days": rebalance,
        "min_trades": _env_int("KELLY_MIN_TRADES", 30),
        "sectors": sectors,
        "min_sharpe": settings.min_pair_sharpe,
        "max_pairs": settings.max_pairs,
        "max_per_sector": settings.max_pairs_per_sector,
        "z_entry": settings.pair_z_entry,
        "z_exit": settings.pair_z_exit,
        "z_stop": settings.pair_z_stop,
        "min_half_life": settings.coint_min_half_life,
        "max_half_life": settings.coint_max_half_life,
        "max_pvalue": settings.coint_adf_pvalue,
        "max_lag": settings.coint_max_lag,
        "prescreen_corr": settings.pair_prescreen_corr,
    }

    costs = _costs_from_options(spread_bps, slippage_bps, cost_scale_by_vol)

    # Determine default output path
    if out is None:
        out = "data/pairs_param_sweep.json" if param_grid else "data/pairs_backtest_stats.json"

    if param_grid:
        sweep = run_pairs_param_sweep(
            ticker_list,
            base_params=base_params,
            param_grid=param_grid,
            fetch=_fetch,
            costs=costs,
        )
    else:
        stats = run_pairs_backtest(
            ticker_list,
            period=period,
            formation_window=formation,
            rebalance_days=rebalance,
            min_trades=_env_int("KELLY_MIN_TRADES", 30),
            sectors=sectors,
            min_sharpe=settings.min_pair_sharpe,
            max_pairs=settings.max_pairs,
            max_per_sector=settings.max_pairs_per_sector,
            z_entry=settings.pair_z_entry,
            z_exit=settings.pair_z_exit,
            z_stop=settings.pair_z_stop,
            min_half_life=settings.coint_min_half_life,
            max_half_life=settings.coint_max_half_life,
            max_pvalue=settings.coint_adf_pvalue,
            max_lag=settings.coint_max_lag,
            prescreen_corr=settings.pair_prescreen_corr,
            fetch=_fetch,
            costs=costs,
        )
        sweep = {
            "generated_at": stats.get("generated_at"),
            "tickers": ticker_list,
            "results": [{
                "params": {k: v for k, v in base_params.items() if k in ("z_entry", "z_exit", "z_stop", "max_pvalue", "min_half_life", "max_half_life", "prescreen_corr", "rebalance_days", "formation_window", "max_pairs", "max_per_sector", "min_sharpe")},
                "period": stats.get("period"),
                "formation_window": stats.get("formation_window"),
                "rebalance_days": stats.get("rebalance_days"),
                "universe_size": stats.get("universe_size"),
                "formations": stats.get("formations"),
                "per_pair": stats.get("per_pair"),
                "aggregate": stats.get("aggregate"),
            }],
        }

    out_path = Path(out)
    if param_grid:
        # Sweep output: the full matrix document (offline analysis shape).
        payload = sweep
    else:
        # Stats-file output: the RAW replay dict (top-level "aggregate") so
        # pair_kelly_from_stats / PAIR_KELLY_* sizing can read it directly.
        payload = stats
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"warning: failed to persist sweep: {exc}")

    for row in sweep.get("results") or []:
        if row.get("error"):
            params_str = f" params={row.get('params')}" if row.get("params") else ""
            print(f"{row.get('period')}{params_str}: error {row['error']}")
            continue
        agg = row.get("aggregate") or {}
        params_str = f" params={row.get('params')}" if row.get("params") else ""
        spread_info = f" spread={row.get('spread_bps')}" if row.get('spread_bps') else ""
        slippage_info = f" slippage={row.get('slippage_bps')}" if row.get('slippage_bps') else ""
        print(f"{row.get('period')}{params_str}{spread_info}{slippage_info}: "
              f"universe={row.get('universe_size')} formations={row.get('formations')} "
              f"n={agg.get('n_trades', 0)} "
              f"win_rate={agg.get('win_rate')} "
              f"kelly={agg.get('kelly_fraction')} "
              f"usable={agg.get('usable')} "
              f"sharpe={agg.get('sharpe')}")
        for pair_id, summary in (row.get("per_pair") or {}).items():
            print(f"  {pair_id}: n={summary.get('n_trades', 0)} "
                  f"win_rate={summary.get('win_rate')} "
                  f"kelly={summary.get('kelly_fraction')} "
                  f"sharpe={summary.get('sharpe')}")
    kind = "sweep" if param_grid else "stats"
    print(f"pairs {kind} written to {out_path}")


@app.command("regime-sweep")
def regime_sweep(
    period: str = typer.Option("1y", "--period", help="History window for the regime series"),
) -> None:
    """Regime-threshold sweep: gate fire rates over history (item 12).

    Evaluates the threshold-independent regime quantities per historical
    date, then reports how often each candidate threshold would have
    fired its gate.  Diagnostic only -- prints a table, writes nothing.
    """
    from src.backtest.regime_sweep import collect_series, sweep_thresholds

    cache_root = os.environ.get("REGIME_CACHE_ROOT", "data/cache")
    try:
        cache = Cache(root=Path(cache_root))
    except Exception:
        cache = None
    raw = collect_series(period=period, cache=cache)
    report = sweep_thresholds(raw)
    print(f"days evaluated: {report.get('days_evaluated', 0)}")
    for key, rows in (report.get("keys") or {}).items():
        print(f"\n{key}:")
        for row in rows:
            rate = row["fire_rate"]
            rate_text = f"{rate:.1%}" if isinstance(rate, float) else "n/a"
            print(f"  {row['threshold']:>8}: fired {row['fired']:>4} / "
                  f"{row['evaluated']:>4} ({rate_text})")


@app.command("show-config")
def show_config() -> None:
    settings = get_settings()
    for ticker in settings.watchlist:
        print(ticker)
    print(f"paper_only={getattr(settings, 'paper_only', True)}")


# --------------------------------------------------------------------------- #
# Session loop: market-clock-aware continuous operation                        #
# --------------------------------------------------------------------------- #

_SLEEP_CHUNK_SECONDS = 60.0


def _fetch_clock(broker: AlpacaClient) -> Optional[dict]:
    """Fetch the Alpaca market clock; ``None`` on failure."""
    try:
        clock = broker.get_clock()
    except Exception as exc:
        print(f"warning: clock fetch failed ({exc}); retrying")
        return None
    return clock if isinstance(clock, dict) else None


def _sleep_with_chunks(seconds: float) -> None:
    """Sleep in <=60s chunks so Ctrl+C stays responsive."""
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        chunk = min(_SLEEP_CHUNK_SECONDS, remaining)
        time.sleep(chunk)
        remaining -= chunk


def _wait_until_open(broker: AlpacaClient) -> dict:
    """Block until the market is open; prints progress.  Ctrl+C aborts."""
    while True:
        clock = _fetch_clock(broker)
        if clock is not None and clock.get("is_open"):
            return clock
        wait = None
        if clock is not None:
            wait = seconds_until(clock.get("next_open"))
            if wait is None:
                wait = seconds_until(clock.get("next_close"))
        print(f"market closed; sleeping {max(60, int(wait)) if wait else 60}s "
              f"(Ctrl+C aborts)")
        _sleep_with_chunks(min(wait, 3600.0) if wait else _SLEEP_CHUNK_SECONDS)


def _open_order_pairs(broker: AlpacaClient) -> set:
    """Set of ``(SYMBOL, side)`` pairs with open broker orders (fail-open)."""
    try:
        rows = broker.get_open_orders()
    except Exception:
        return set()
    pairs = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").upper()
        side = str(row.get("side") or "").lower()
        if symbol and side in ("buy", "sell"):
            pairs.add((symbol, side))
    return pairs


@app.command("session")
def session(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    interval: int = typer.Option(30, "--interval", help="Minutes between cycles"),
    llm_refresh: int = typer.Option(120, "--llm-refresh", help="Minutes between full LLM analyses"),
    submit: bool = typer.Option(False, "--submit", help="Actually submit paper orders"),
    max_cycles: int = typer.Option(0, "--max-cycles", help="Stop after N cycles (0 = run until close)"),
    no_wait: bool = typer.Option(
        False, "--no-wait",
        help="Test mode: never sleep, run cycles back-to-back even when closed (submit disabled)",
    ),
) -> None:
    """Continuous intraday operation: cycles analysis (+orders) until close.

    Sleeps until the market opens (Alpaca /v2/clock), then runs cycles:
    a FULL pipeline (all LLM analysts + debate) at open and every
    --llm-refresh minutes, CHEAP quant-only cycles in between (the last
    full analysis' reports/debate are reused; fresh prices drive fresh
    quant signals).  Orders are gated cross-cycle by the session store:
    at most 1 BUY + 1 SELL per ticker per day (SESSION_MAX_*_PER_TICKER),
    no re-submission while an open order exists, and session-wide BUY
    notional capped at live equity.  Ctrl+C is safe at any time; state
    persists per action in the decisions DB.
    """
    settings = get_settings()
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)
    if not (api_key and secret_key):
        print("error: session requires ALPACA_API_KEY / ALPACA_SECRET_KEY (market clock)")
        raise typer.Exit(code=1)

    if no_wait and submit:
        print("no-wait test mode: --submit disabled (analysis-only cycles)")
        submit = False

    broker = AlpacaClient(key=api_key, secret=secret_key, paper_only=True)
    ticker_list = _parse_tickers(tickers)
    decision_store = _store()
    session_store = SessionStore(
        db_path=Path(os.environ.get("DECISIONS_DB", "data/decisions.sqlite3"))
    )
    gate = SubmissionGate(
        max_buys_per_ticker=_env_int("SESSION_MAX_BUYS_PER_TICKER", 1),
        max_sells_per_ticker=_env_int("SESSION_MAX_SELLS_PER_TICKER", 1),
    )
    interval_seconds = max(_SLEEP_CHUNK_SECONDS, float(interval) * 60.0)
    refresh_seconds = max(interval_seconds, float(llm_refresh) * 60.0)

    print(f"session: tickers={','.join(ticker_list)} submit={submit} "
          f"interval={interval}m llm_refresh={llm_refresh}m"
          + (" no-wait" if no_wait else ""))
    if no_wait:
        clock = _fetch_clock(broker) or {}
    else:
        clock = _wait_until_open(broker)
    trade_date = datetime.now(timezone.utc).date().isoformat()
    print(f"market open; trade date {trade_date}"
          if clock.get("is_open")
          else f"market closed (no-wait test mode); trade date {trade_date}")

    cycles = 0
    last_full_monotonic = -math.inf
    last_full_decisions: Optional[List[dict]] = None
    try:
        while True:
            clock = _fetch_clock(broker)
            if not no_wait and (clock is None or not clock.get("is_open")):
                print("market closed — session ending")
                break
            if max_cycles and cycles >= max_cycles:
                print(f"max cycles ({max_cycles}) reached — stopping")
                break

            now_mono = time.monotonic()
            is_full = (cycles == 0) or (now_mono - last_full_monotonic) >= refresh_seconds
            components = _components(submit=submit)
            if not is_full and last_full_decisions is not None:
                components["precomputed"] = {
                    str(d.get("ticker") or "").upper(): {
                        "reports": d.get("reports") or {},
                        "debate": d.get("debate"),
                        "sector": d.get("sector"),
                        "flow_data": d.get("flow_data") or {},
                    }
                    for d in last_full_decisions
                }

            # Cross-cycle budget state, refreshed from the session store.
            counts = session_store.action_counts_for_date(trade_date)
            used_notional = session_store.buy_notional_for_date(trade_date)
            budget = float(components["account"].get("equity", 0.0))
            if submit:
                # The planner budgets against remaining session equity.
                try:
                    components["account"]["equity"] = max(0.0, budget - used_notional)
                except Exception:
                    pass

            # Wrap the planner so cross-cycle gating happens before the
            # (single) submission step inside run_pipeline.  Applied in
            # every mode: in dry-runs the gate filters what *would* be
            # submitted, so the output stays realistic.
            planner = components["build_order_plan"]
            gate_state = {
                "gate": gate,
                "counts": counts,
                "open_pairs": _open_order_pairs(broker) if submit else set(),
                "used": used_notional,
                "budget": budget,
            }

            def _gated_plan(*args, _planner=planner, _state=gate_state, **kwargs):
                orders = _planner(*args, **kwargs)
                allowed, skipped = _state["gate"].gate(
                    list(orders or []),
                    _state["counts"],
                    _state["open_pairs"],
                    _state["used"],
                    _state["budget"],
                )
                for order, reason in skipped:
                    print(f"skipped {order_ticker(order)} "
                          f"{order_side(order)}: {reason}")
                return allowed

            components["build_order_plan"] = _gated_plan

            label = "full" if is_full else "cheap"
            print(f"--- cycle {cycles + 1} ({label}) "
                  f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC ---")
            result = run_pipeline(ticker_list, components)
            if inspect.isawaitable(result):
                result = asyncio.run(_await_and_close(result, components))
            decisions = list(result or [])
            for decision in decisions:
                _print_decision(decision)
                try:
                    decision_store.log_decision(_jsonable(decision))
                except Exception as exc:
                    print(f"warning: failed to persist decision: {exc}")

            if is_full:
                last_full_monotonic = now_mono
                last_full_decisions = decisions
            if submit:
                for decision in decisions:
                    for order in decision.get("orders") or []:
                        try:
                            session_store.log_action(
                                trade_date=trade_date,
                                ticker=str(order.get("ticker") or ""),
                                side=str(order.get("side") or ""),
                                notional=order_notional(order),
                            )
                        except Exception as exc:
                            print(f"warning: failed to log session action: {exc}")
            cycles += 1

            # -- inter-cycle wait (market-aware, Ctrl+C responsive) ---------
            if max_cycles and cycles >= max_cycles:
                continue
            if no_wait:
                continue
            waited = 0.0
            while waited < interval_seconds:
                clock = _fetch_clock(broker)
                if clock is not None and not clock.get("is_open"):
                    break
                chunk = min(_SLEEP_CHUNK_SECONDS, interval_seconds - waited)
                _sleep_with_chunks(chunk)
                waited += chunk
    except KeyboardInterrupt:
        print("\nCtrl+C — session ending (state persisted per action)")
    finally:
        actions = session_store.actions_for_date(trade_date)
        buys = sum(1 for a in actions if a["side"] == "buy")
        sells = sum(1 for a in actions if a["side"] == "sell")
        notional = session_store.buy_notional_for_date(trade_date)
        print(f"session summary: cycles={cycles} buys={buys} sells={sells} "
              f"buy_notional={notional:.2f}")


# --------------------------------------------------------------------------- #
# Overnight cycle: enter at the close, mechanically exit at the next open      #
# --------------------------------------------------------------------------- #

def _recent_actions(session_store: SessionStore, days: int = 7) -> List[dict]:
    """Chronological actions across the last ``days`` trade dates."""
    today = datetime.now(timezone.utc).date()
    actions: List[dict] = []
    for offset in range(days, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        actions.extend(session_store.actions_for_date(day))
    return actions


def _exit_reference_prices(tickers: set) -> Dict[str, float]:
    """Last close per ticker for exit-order notional (fail-open to skip)."""
    prices: Dict[str, float] = {}
    for ticker in tickers:
        try:
            frame = fetch_ohlcv(ticker, period="5d")
            close = frame["Close"] if hasattr(frame, "columns") else frame
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            value = float(close.iloc[-1])
            if math.isfinite(value) and value > 0:
                prices[ticker] = value
        except Exception:
            continue
    return prices


@app.command("overnight")
def overnight(
    tickers: Optional[str] = typer.Option(None, "--tickers", help="Comma separated list of tickers"),
    max_names: int = typer.Option(
        0, "--max-names",
        help="Max concurrent overnight names (0 = OVERNIGHT_MAX_NAMES, default 5)",
    ),
    entry_before_close_min: int = typer.Option(
        10, "--entry-before-close",
        help="Run the entry cycle when this many minutes remain until close",
    ),
    exit_after_open_utc: str = typer.Option(
        "13:35", "--exit-after-open-utc",
        help="Earliest UTC time to exit overnight entries (09:35 ET in summer)",
    ),
    submit: bool = typer.Option(False, "--submit", help="Actually submit paper orders"),
    no_wait: bool = typer.Option(
        False, "--no-wait",
        help="Test mode: one entry-planning pass + pending-exit listing, no sleeps (submit disabled)",
    ),
) -> None:
    """Halo/overnight-drift cycle: BUY approved signals at the close,
    sell at the next open.

    Sizing uses the BUY-side Kelly stats from OVERNIGHT_STATS_PATH
    (`backtest --mode overnight`) when usable; entry names are capped at
    OVERNIGHT_MAX_NAMES by quant confidence (filtered at the planner, so
    the pipeline submits exactly once).  Exits are mechanical: every open
    overnight entry is sold at held quantity the next morning regardless
    of signal, idempotently (state lives in the session store, so a
    restart mid-overnight still yields a complete exit).  The panic
    regime gate already blocks entries.
    """
    settings = get_settings()
    api_key = getattr(settings, "alpaca_api_key", None)
    secret_key = getattr(settings, "alpaca_secret_key", None)
    if not (api_key and secret_key):
        print("error: overnight requires ALPACA_API_KEY / ALPACA_SECRET_KEY (market clock)")
        raise typer.Exit(code=1)

    if no_wait and submit:
        print("no-wait test mode: --submit disabled (analysis-only pass)")
        submit = False

    broker = AlpacaClient(key=api_key, secret=secret_key, paper_only=True)
    ticker_list = _parse_tickers(tickers)
    decision_store = _store()
    session_store = SessionStore(
        db_path=Path(os.environ.get("DECISIONS_DB", "data/decisions.sqlite3"))
    )
    max_names_effective = max_names or _env_int("OVERNIGHT_MAX_NAMES", 5)
    try:
        exit_hh, exit_mm = (int(part) for part in exit_after_open_utc.split(":", 1))
    except ValueError:
        print(f"error: bad --exit-after-open-utc {exit_after_open_utc!r}")
        raise typer.Exit(code=1)

    print(f"overnight: tickers={','.join(ticker_list)} submit={submit} "
          f"max_names={max_names_effective} exit_after={exit_after_open_utc} UTC"
          + (" no-wait" if no_wait else ""))

    kelly = load_overnight_kelly(
        os.environ.get("OVERNIGHT_STATS_PATH", "data/backtest_stats_overnight.json"),
        min_trades=_env_int("KELLY_MIN_TRADES", 30),
    )
    print("overnight kelly: "
          + (f"fraction={kelly['fraction']:.4f} (n={kelly['n']})"
             if kelly else "unusable (no scale-up)"))

    try:
        while True:
            clock = _fetch_clock(broker)
            if clock is None:
                if no_wait:
                    print("clock unavailable (no-wait) — stopping")
                    break
                _sleep_with_chunks(_SLEEP_CHUNK_SECONDS)
                continue
            now_utc = datetime.now(timezone.utc)
            trade_date = now_utc.date().isoformat()
            is_open = bool(clock.get("is_open"))
            closes_in = seconds_until(clock.get("next_close"))
            actions = _recent_actions(session_store)

            # -- morning exit (mechanical, idempotent) -----------------------
            pending = pending_exits(actions, trade_date)
            exit_due = (now_utc.hour > exit_hh
                        or (now_utc.hour == exit_hh and now_utc.minute >= exit_mm))
            if is_open and pending and not exit_due:
                print(f"{len(pending)} overnight entr(ies) held; "
                      f"exit window opens {exit_after_open_utc} UTC")
            if is_open and pending and exit_due:
                if not submit:
                    print(f"[dry-run] would exit overnight entries: "
                          f"{', '.join(sorted(pending))}")
                else:
                    positions: Dict[str, float] = {}
                    try:
                        for row in broker.get_positions():
                            try:
                                positions[str(row.get("symbol") or "").upper()] = \
                                    float(row.get("qty") or 0.0)
                            except (TypeError, ValueError):
                                continue
                    except Exception:
                        positions = {}
                    price_map = _exit_reference_prices(pending)
                    orders = exit_orders(pending, positions, price_map)
                    if not orders:
                        print("pending exits resolved (no live positions)")
                    else:
                        broker.submit_limit_orders([
                            OrderPlan.model_validate(order) for order in orders
                        ])
                        for order in orders:
                            session_store.log_action(
                                trade_date=trade_date,
                                ticker=str(order["ticker"]),
                                side="sell",
                                notional=float(order["quantity"]) * float(order["limit_price"]),
                                kind="overnight_exit",
                            )
                        print(f"overnight exit submitted: "
                              f"{', '.join(o['ticker'] for o in orders)}")

            # -- evening entry (single pipeline pass; planner filters) --------
            entered_today = any(
                a.get("kind") == "overnight_entry" and a.get("trade_date") == trade_date
                for a in actions
            )
            entry_window_open = no_wait or (
                is_open and closes_in is not None
                and closes_in <= entry_before_close_min * 60.0
            )
            if entry_window_open and not entered_today:
                components = _components(submit=submit)
                if kelly:
                    components["account"]["kelly"] = dict(kelly)
                print(f"--- overnight entry cycle {now_utc.strftime('%H:%M:%S')} UTC ---")

                def _overnight_plan(decisions, signals, account, day_trade_dates,
                                    _planner=components["build_order_plan"],
                                    _max=max_names_effective, **kwargs):
                    ranked = []
                    for veto in decisions or []:
                        ticker = str(getattr(veto, "ticker", "") or "").upper()
                        quant = signals.get(ticker) or {}
                        if str(quant.get("signal") or "").upper() != "BUY":
                            continue
                        if not getattr(veto, "approved", False):
                            continue
                        try:
                            confidence = float(quant.get("confidence"))
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(confidence):
                            ranked.append((confidence, ticker))
                    ranked.sort(reverse=True)
                    selected = {ticker for _, ticker in ranked[:_max]}
                    # Cross-mode guard: never buy a ticker whose per-day
                    # BUY budget the session loop already consumed today.
                    session_counts = session_store.action_counts_for_date(trade_date)
                    session_max_buys = _env_int("SESSION_MAX_BUYS_PER_TICKER", 1)
                    eligible = exclude_session_bought(
                        selected, session_counts, session_max_buys)
                    guarded = selected - eligible
                    if guarded:
                        print(f"overnight guard: session already bought today, "
                              f"skipping {', '.join(sorted(guarded))}")
                    selected = eligible
                    orders = _planner(decisions, signals, account, day_trade_dates,
                                      **kwargs)
                    filtered = [
                        order for order in (orders or [])
                        if order_ticker(order) in selected
                        and order_side(order) == "buy"
                    ]
                    dropped = [order_ticker(o) for o in (orders or [])
                               if o not in filtered]
                    if dropped:
                        print(f"overnight cap: dropped {', '.join(sorted(set(dropped)))} "
                              f"(max_names={_max})")
                    return filtered

                components["build_order_plan"] = _overnight_plan
                result = run_pipeline(ticker_list, components)
                if inspect.isawaitable(result):
                    result = asyncio.run(_await_and_close(result, components))
                decisions = list(result or [])
                for decision in decisions:
                    _print_decision(decision)
                    try:
                        decision_store.log_decision(_jsonable(decision))
                    except Exception as exc:
                        print(f"warning: failed to persist decision: {exc}")
                if not submit:
                    print("[dry-run] entry analysis only (no orders submitted)")
                else:
                    # The pipeline already submitted the filtered orders;
                    # record them as overnight entries for the morning exit.
                    for decision in decisions:
                        for order in decision.get("orders") or []:
                            try:
                                session_store.log_action(
                                    trade_date=trade_date,
                                    ticker=order_ticker(order),
                                    side="buy",
                                    notional=order_notional(order),
                                    kind="overnight_entry",
                                )
                            except Exception as exc:
                                print(f"warning: failed to log overnight entry: {exc}")
                if no_wait:
                    print("no-wait: one pass complete — stopping")
                    break

            if no_wait:
                print("no-wait: one pass complete — stopping")
                break
            if not is_open:
                print(f"market closed; next open "
                      f"{clock.get('next_open', '?')} — sleeping (Ctrl+C aborts)")
                wait = seconds_until(clock.get("next_open")) or _SLEEP_CHUNK_SECONDS
                _sleep_with_chunks(min(wait, 3600.0))
                continue
            _sleep_with_chunks(_SLEEP_CHUNK_SECONDS)
    except KeyboardInterrupt:
        print("\nCtrl+C — overnight cycle ending (state persisted)")
    finally:
        actions = _recent_actions(session_store)
        entries = sum(1 for a in actions if a.get("kind") == "overnight_entry")
        exits = sum(1 for a in actions if a.get("kind") == "overnight_exit")
        print(f"overnight summary: entries={entries} exits={exits}")


if __name__ == "__main__":
    app()
