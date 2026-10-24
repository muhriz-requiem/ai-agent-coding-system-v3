"""DAG wiring for the trading pipeline.

This module exposes a single public entry point, :func:`run_pipeline`,
that wires together the ingestion, quant-signal, analyst, debate, risk,
order-planning and broker-submission stages of the trading pipeline.

The wiring is intentionally "dumb": every piece of behaviour (data
fetching, analysis, risk evaluation, order construction, broker
submission) is injected via the ``components`` mapping so that this
module performs no I/O of its own and can be fully exercised with
fakes/mocks in tests.

Per-ticker flow (executed concurrently across tickers via
``asyncio.gather``)::

    fetch_ohlcv / fetch_fundamentals / fetch_news
        -> generate_signal (quant)
        -> run_analysts
        -> run_debate
        -> evaluate_risk

Once for the whole batch, after every ticker has produced a risk
decision::

    build_order_plan -> broker_submit
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pandas as pd

from src.agents.flow_analyst import build_flow_report
from src.quant.correlation import correlation_clusters
from src.quant.signals import attach_regime_features


def _close_of(prices: Any) -> Any:
    """Derive the close-price Series from a fetch result.

    The OHLCV fetch component may return the full frame (production) or a
    bare close Series (offline tests / older fetchers); everything that is
    not ``generate_signal`` (analysts, reference price, correlation) keeps
    consuming the close series exactly as before.
    """
    if isinstance(prices, pd.DataFrame):
        if "Close" in prices.columns:
            close = prices["Close"]
        else:
            close = prices.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close
    return prices

try:
    # TASK-7010A introduced the shared pipeline state container.
    from src.pipeline.state import new_state
except Exception:  # pragma: no cover - defensive fallback only

    def new_state(tickers: list[str], as_of: str | None = None) -> Any:  # type: ignore[no-redef]
        return {
            "tickers": list(tickers),
            "reports": [],
            "debates": [],
            "decisions": [],
            "orders": [],
            "as_of": as_of,
        }


async def _maybe_await(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``func`` and await the result if it is awaitable."""

    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _append(state: Any, key: str, value: Any) -> None:
    """Best-effort append to a list-shaped field on ``state``.

    ``state`` may be a ``TypedDict``/plain ``dict`` or an object exposing
    the same attributes; we support both without depending on the exact
    shape produced by ``new_state``.
    """

    if isinstance(state, dict):
        container = state.get(key)
        if container is None:
            state[key] = [value]
        else:
            container.append(value)
        return

    container = getattr(state, key, None)
    if container is None:
        setattr(state, key, [value])
    else:
        container.append(value)


def _set(state: Any, key: str, value: Any) -> None:
    """Best-effort assignment of a field on ``state``."""

    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)


def _dump(value: Any) -> Any:
    """Convert pydantic-style models to plain dicts, otherwise pass through."""

    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _new_state(tickers: list[str], as_of: str | None) -> Any:
    """Construct the shared :class:`PipelineState`, tolerating either
    ``new_state(tickers)`` or ``new_state(tickers, as_of=...)`` signatures.
    """

    try:
        return new_state(tickers, as_of=as_of)
    except TypeError:
        state = new_state(tickers)
        _set(state, "as_of", as_of)
        return state


async def run_pipeline(
    tickers: list[str],
    components: dict,
    as_of: str | None = None,
) -> list[dict]:
    """Run the full trading pipeline for ``tickers`` concurrently.

    Per ticker (executed concurrently via :func:`asyncio.gather`):
      1. Ingest price/fundamentals/news data.
      2. Generate the quant signal (annotated with ``reference_price``).
      3. Run the analyst crew.
      4. Run the debate.
      5. Evaluate risk.

    Once for the whole batch (after every ticker has produced a risk
    decision):
      6. Build the order plan.
      7. Submit orders to the broker.

    Returns one decision dict per ticker of the shape::

        {"ticker": ..., "quant": ..., "reports": ..., "veto": ...,
         "orders": [...], "regime": ...}

    where ``orders`` carries this ticker's emitted orders (JSON-safe
    dicts, for the SQLite audit log) and ``regime`` is the batch
    regime snapshot (``None`` when the regime layer is disabled).
    """

    client = components["client"]
    account = components["account"]
    day_trade_dates = components["day_trade_dates"]
    positions = components.get("positions") or {}
    regime = components.get("regime")

    fetch_ohlcv = components["fetch_ohlcv"]
    fetch_fundamentals = components["fetch_fundamentals"]
    fetch_news = components["fetch_news"]
    generate_signal = components["generate_signal"]
    run_analysts = components["run_analysts"]
    run_debate = components["run_debate"]
    evaluate_risk = components["evaluate_risk"]
    build_order_plan = components["build_order_plan"]
    broker_submit = components["broker_submit"]

    state = _new_state(tickers, as_of)

    async def process_ticker(ticker: str) -> dict:
        # -- cheap-cycle injection (session loop) ---------------------------
        # ``precomputed`` carries the last full analysis' reports, debate,
        # sector label and flow envelopes for this ticker, so a cheap
        # session cycle re-runs only ingestion of prices + quant + risk
        # + planning with zero LLM calls.  Ingestion of the slow-moving
        # slices (fundamentals/news/whale-flow) is skipped entirely; the
        # morning's flow envelopes ride along so flow vetoes/sizing stay
        # as fresh as the full analysis that produced them.
        precomputed = (components.get("precomputed") or {}).get(ticker)

        # -- ingestion (must precede analysis) -----------------------------
        prices = await _maybe_await(fetch_ohlcv, ticker, period="6mo")
        if precomputed is not None:
            fundamentals = None
            news = None
            insider = None
            institutional = None
            options_flow = None
            dark_pool = None
            flow_data = dict(precomputed.get("flow_data") or {})
        else:
            fundamentals = await _maybe_await(fetch_fundamentals, ticker)
            news = await _maybe_await(fetch_news, ticker)

            # -- optional whale/flow slices ------------------------------------
            # Fetchers come from components, never direct imports (graph.py
            # performs no I/O; pinned by tests/test_wiring.py).  Absent
            # components (feature flag off / disabled data source) leave the
            # slice at None so the corresponding analysts are not dispatched.
            fetch_insider = components.get("fetch_insider")
            fetch_institutional = components.get("fetch_institutional")
            fetch_options_flow = components.get("fetch_options_flow")
            fetch_dark_pool = components.get("fetch_dark_pool")

            insider = await _maybe_await(fetch_insider, ticker) if fetch_insider else None
            institutional = (
                await _maybe_await(fetch_institutional, ticker)
                if fetch_institutional
                else None
            )
            options_flow = (
                await _maybe_await(fetch_options_flow, ticker) if fetch_options_flow else None
            )
            dark_pool = await _maybe_await(fetch_dark_pool, ticker) if fetch_dark_pool else None
            flow_data = {
                name: data
                for name, data in (
                    ("insider", insider),
                    ("institutional", institutional),
                    ("options_flow", options_flow),
                )
                if data is not None
            }

        # -- quant signal ----------------------------------------------------
        # The full OHLCV frame (when the fetcher provides one) feeds the
        # advisory volume features; the close series drives everything else
        # (reference price, analysts, correlation) exactly as before.
        close = _close_of(prices)
        # The regime snapshot rides along so the BUY-side regime gates
        # (fear_z / gold-eq breadth) evaluate; fail-open when absent.
        quant = await _maybe_await(generate_signal, prices, regime)
        if isinstance(quant, dict):
            quant["reference_price"] = float(close.iloc[-1])
            # Advisory breadth features from the batch regime snapshot
            # (fear_z / liquidity_z composites + breadth component
            # z-scores).  Fail-open: no regime data adds nothing.
            quant = attach_regime_features(quant, regime)

        # -- analysis ---------------------------------------------------------
        # The dark-pool proxy (when fetched successfully) rides along in the
        # macro slice as regime context; the regime snapshot itself is
        # untouched so a disabled feature leaves the macro slice unchanged.
        macro_slice: Any = regime
        if isinstance(dark_pool, dict) and dark_pool.get("status") == "ok":
            macro_slice = {"regime": regime, "dark_pool": dark_pool}
        inputs = {
            "prices": close,
            "ohlcv": close,
            "fundamentals": fundamentals,
            "news": news,
            "quant": quant,
            # The macro analyst consumes the batch regime snapshot (global
            # liquidity & fear composites); when absent it degrades to {}.
            "macro": macro_slice,
            # Whale/flow slices: None -> analyst not dispatched (gating).
            "insider": insider,
            "institutional": institutional,
            "options_flow": options_flow,
        }
        if precomputed is not None:
            # Cheap cycle: reuse the last full analysis verbatim.
            reports = precomputed.get("reports") or {}
            debate = precomputed.get("debate")
        else:
            reports = await _maybe_await(run_analysts, ticker, inputs, client=client)
            # Deterministic flow-synthesis report (advisory, zero LLM cost):
            # only added when at least one whale/flow source fetched OK, so
            # flags-off runs are bit-identical to before.  It joins the
            # debate prompt; the risk vetoes read flow_data directly and the
            # sizing blend filters on per-source names, so no double count.
            flow_report = build_flow_report(ticker, flow_data)
            if flow_report is not None:
                reports = dict(reports)
                reports["flow"] = flow_report
            debate = await _maybe_await(run_debate, ticker, reports, client=client)
        risk_decision = await _maybe_await(
            evaluate_risk,
            ticker,
            reports,
            debate,
            quant,
            account,
            proposed_position_pct=None,
            flow_data=flow_data or None,
        )

        return {
            "ticker": ticker,
            "quant": quant,
            "reports": reports,
            "debate": debate,
            "veto": risk_decision,
            "fundamentals": fundamentals,
            "flow_data": flow_data,
            # Sector label carried through for the batch sector cap even on
            # cheap cycles (fundamentals fetch is skipped there).
            "sector": (
                (fundamentals or {}).get("sector")
                if fundamentals is not None
                else (precomputed or {}).get("sector")
            ),
            # Retained for the batch correlation clustering (item 11);
            # in-memory only, never persisted.
            "closes": close,
        }

    per_ticker_results = list(
        await asyncio.gather(*(process_ticker(ticker) for ticker in tickers))
    )

    # -- populate the shared pipeline state --------------------------------
    for result in per_ticker_results:
        _append(
            state,
            "reports",
            {
                "ticker": result["ticker"],
                "quant": result["quant"],
                "reports": result["reports"],
            },
        )
        _append(
            state,
            "debates",
            {"ticker": result["ticker"], "debate": result["debate"]},
        )
        _append(
            state,
            "decisions",
            {"ticker": result["ticker"], "veto": result["veto"]},
        )

    # -- batch-level order planning + submission (exactly once) -----------
    decisions = [result["veto"] for result in per_ticker_results]
    signals = {result["ticker"]: result["quant"] for result in per_ticker_results}
    # Sector labels come straight from the fetched fundamentals; tickers
    # without a sector (e.g. ETFs) map to None and are exempt from the cap.
    # Cheap cycles skip the fundamentals fetch, so the precomputed sector
    # label from the last full analysis is the fallback.
    sector_map = {
        result["ticker"]: (
            (result.get("fundamentals") or {}).get("sector")
            if result.get("fundamentals") is not None
            else result.get("sector")
        )
        for result in per_ticker_results
    }

    # Flow-conviction reports for sizing: only reports backed by a
    # successful fetch are passed, so sentinel/error runs (and flags-off
    # runs, where the analysts are never dispatched) leave sizing at 1.0.
    flow_analysts = ("insider", "institutional", "options_flow")
    analyst_reports: dict[str, dict] = {}
    for result in per_ticker_results:
        flow = result.get("flow_data") or {}
        analyst_reports[result["ticker"]] = {
            name: report
            for name, report in (result.get("reports") or {}).items()
            if name in flow_analysts
            and isinstance(flow.get(name), dict)
            and flow[name].get("status") == "ok"
        }

    # Debate records (item 10): duck-typed by build_order_plan; a bear
    # winner shrinks BUYs, a bull winner shrinks SELLs, undecided no-ops.
    debates = {
        result["ticker"]: result.get("debate")
        for result in per_ticker_results
    }

    # Correlation clusters (item 11): computed once per batch from the
    # closes already fetched per ticker.  Pure computation on in-memory
    # data; insufficient overlap yields no clusters (fail-open no-op).
    closes = {
        result["ticker"]: result.get("closes") for result in per_ticker_results
    }
    corr_cfg = components.get("correlation")
    if isinstance(corr_cfg, dict):
        corr_enabled = bool(corr_cfg.get("enabled", True))
    else:
        corr_enabled = True
    if corr_enabled and len(closes) >= 2:
        cfg = corr_cfg if isinstance(corr_cfg, dict) else {}
        cluster_map = correlation_clusters(
            closes,
            threshold=cfg.get("threshold", 0.80),
            min_observations=cfg.get("min_observations", 60),
        )
    else:
        cluster_map = None

    # -- pairs (stat-arb, batch level) ---------------------------------------
    # Everything heavy lives behind injected components (graph.py performs
    # no I/O and no direct imports, the same seam as the fetchers):
    #   pair_book          -- loaded PairBook (None -> pairs disabled)
    #   pairs_form_if_due  -- (book, as_of) -> book (re-forms when due)
    #   pairs_signals      -- (book) -> list[PairSignal]
    #   pairs_risk         -- (signal, book) -> pair risk decision
    # Absent components / empty book leave the batch pairs-free (fail-open).
    pair_signals: list = []
    pair_decisions: list = []
    pair_book = components.get("pair_book")
    if pair_book is not None:
        form_if_due = components.get("pairs_form_if_due")
        if form_if_due is not None:
            pair_book = await _maybe_await(form_if_due, pair_book, as_of)
        pairs_signals_fn = components.get("pairs_signals")
        if pairs_signals_fn is not None and pair_book is not None:
            pair_signals = await _maybe_await(pairs_signals_fn, pair_book) or []
        pairs_risk_fn = components.get("pairs_risk")
        if pairs_risk_fn is not None and pair_signals:
            pair_decisions = [
                await _maybe_await(pairs_risk_fn, signal, pair_book)
                for signal in pair_signals
            ]

    # Pair kwargs are passed to the planner ONLY when the pairs layer is
    # active, so flag-off runs are bit-identical to pre-pairs behavior
    # (including planner fakes with the legacy signature).
    plan_kwargs: dict = {}
    if pair_decisions or pair_signals:
        plan_kwargs = {
            "pair_decisions": pair_decisions,
            "pair_signals": pair_signals,
        }
    orders = await _maybe_await(
        build_order_plan,
        decisions,
        signals,
        account,
        day_trade_dates,
        as_of=as_of,
        positions=positions,
        sector_map=sector_map,
        regime=regime,
        analyst_reports=analyst_reports,
        debates=debates,
        cluster_map=cluster_map,
        **plan_kwargs,
    )
    await _maybe_await(broker_submit, orders)

    _set(state, "orders", orders)

    def _order_ticker(order: Any) -> Any:
        if isinstance(order, dict):
            return order.get("ticker")
        return getattr(order, "ticker", None)

    orders_by_ticker: dict[Any, list] = {}
    for order in orders or []:
        orders_by_ticker.setdefault(_order_ticker(order), []).append(_dump(order))

    results = [
        {
            "ticker": result["ticker"],
            "quant": result["quant"],
            "reports": result["reports"],
            # Carried for the session loop's cheap-cycle reuse (and now
            # persisted in the audit log too).  The model is kept intact:
            # evaluate_risk/build_order_plan duck-type its attributes.
            "debate": result.get("debate"),
            "sector": result.get("sector"),
            "flow_data": _dump(result.get("flow_data")),
            "veto": _dump(result["veto"]),
            "orders": orders_by_ticker.get(result["ticker"], []),
            # Regime snapshot stamped for the SQLite audit log.
            "regime": _dump(regime) if regime is not None else None,
        }
        for result in per_ticker_results
    ]

    # Pair audit entries: one per pair signal, carrying the pair's leg
    # orders (leg tickers are not per-ticker results, so their orders
    # would otherwise drop out of the audit trail).  The CLI persists
    # these through DecisionStore.log_pair_decision (kind="pair").
    def _pair_field_of(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    for signal, decision in zip(pair_signals, pair_decisions):
        pair_id = _pair_field_of(signal, "pair_id") or ""
        legs = (_pair_field_of(signal, "ticker_a"), _pair_field_of(signal, "ticker_b"))
        leg_orders = [
            order
            for leg in legs
            for order in orders_by_ticker.get(leg, [])
        ]
        results.append({
            "kind": "pair",
            "ticker": f"PAIR:{pair_id}",
            "pair": _dump(signal),
            "veto": _dump(decision),
            "orders": leg_orders,
            "regime": _dump(regime) if regime is not None else None,
        })

    return results
