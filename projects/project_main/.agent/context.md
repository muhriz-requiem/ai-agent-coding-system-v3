# project_main — Production-Grade Multi-Agent AI Stock Trading System

Derived from the technical blueprint. Read this before building any task.

## Pipeline (lean asyncio DAG — no LangGraph in v1)

```
Layer 1  Ingestion (yfinance OHLCV / news / fundamentals) + disk cache
   |     asyncio.Queue decouples ingestion from LLM work (rate-limit safety)
Layer 2  Specialist analysts in parallel, INFORMATION ISOLATED:
         Fundamental, Technical (text/indicators in v1 — vision deferred to v2),
         Sentiment, Macro.  Deterministic Quant node runs alongside.
Layer 3  Bull/Bear debate  ->  Risk Manager VETO GATE (deterministic validator)
Layer 4  Portfolio Manager: whole-watchlist view, ranking, sizing, batching
         -> Alpaca PAPER broker (limit orders only, 200 req/min limiter)
```

## Hard rules

1. **Quant node is mathematically absolute.** SELL with confidence > 0.80
   (extreme z-score) downgrades any bullish thesis to HOLD. Implemented as a
   deterministic check in the Risk Manager — never prompt-hoped.
2. **Evidence gating.** Every analyst report carries `citations: list[str]`;
   the Risk Manager hard-vetoes theses lacking numeric citations.
3. **Paper only.** Broker client hard-locks the paper endpoint while
   `config.paper_only` is true; market orders are rejected by validation —
   limit orders bound to thesis reference prices only.
4. **Alpaca limits.** 200 req/min client-side limiter; orders batched through
   the Portfolio Manager; PDT rule: < $25k margin accounts max 3 intraday
   round-trips per rolling 5 business days.
5. **Offline-first tests.** Every pytest run must be network-free: yfinance /
   OpenAI / Alpaca sit behind thin adapters that tests monkeypatch or stub.
   This is what makes the coding system's real-test gate deterministic.
6. **Decision audit.** Every run appends to the SQLite decision log
   (timestamp, ticker, quant metrics, reports, veto result, orders).

## Model routing (trading system)

z-ai/glm-5.3-flash via OpenRouter for all agent roles; reasoning effort
`low` for extraction/screening, `max` for Risk Manager and Portfolio Manager.
Keys live in `project_main/.env` (see `.env.example`), separate from the
coding workspace root.

## Build order — tests-first workflow

Task board (TASK-7001..7013B) is dependency-chained and **complete** — all
tasks done, suite fully green (113 tests, zero skips). The final integration
wiring (TASK-7013A/B) bound `cli._components()` to the real modules and made
`run_smoke` delegate to `run_pipeline` with the decision audit on both paths.
**The test files are already written** (by the human PM) and live in `tests/`;
each test file skips itself until its implementation module exists. Therefore:

1. Each task = implement exactly its `target_files` (ONE file each — a
   two-file integration task escalated three cycles before being split;
   see TASK-7013A/B on the board).
2. Do NOT create or modify test files — they are the fixed specification.
3. The real gate demands the task's own tests run and pass
   (`requires_new_tests`): a green suite without new tests passing counts
   as CHANGES_REQUESTED.
4. Follow the exact API signatures in the task objective — the tests pin
   them precisely.
5. The wired entry points: `python -m src.cli scan` (analysis + decision
   log, no keys needed), `paper-run` (fails closed without Alpaca keys),
   and `src.pipeline.smoke.run_smoke` (manual paper session; components
   built via `src.cli._components(submit=not dry_run)`). Account sizing
   comes from `ACCOUNT_EQUITY` / `MAX_POSITION_PCT` / `DAY_TRADE_DATES`
   env vars (see `.env.example`).

v2 candidates: vision chart analysis, SEC-filing ingestion, free-tier
routing (Groq/Gemini/Ollama), dashboards.
