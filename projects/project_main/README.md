# project_main

LLM-driven paper-trading pipeline for equities. Fetches market data, runs quant signals + AI analysts through bull/bear debate, applies deterministic risk veto, sizes orders, submits to Alpaca paper API.

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill ALPACA_API_KEY, ALPACA_SECRET_KEY, OPENROUTER_API_KEY
                       # (whale/flow keys are optional — see .env.example)
python -m src.cli scan           # analysis only
python -m src.cli paper-run      # analysis + submit orders
python -m src.cli decisions      # show audit log
python -m src.cli show-config    # print settings
python -m src.cli backtest       # replay quant signals -> Kelly stats
python -m src.cli backtest --mode overnight --primary
                                  # overnight replay -> BUY-side Kelly stats
                                  # promoted to the primary stats file
python -m src.cli backtest --spread-bps 10 --slippage-bps 3 --random-baseline
                                  # backtest with spread/slippage + random baseline
python -m src.cli backtest-walkforward --tickers AAPL --period 5y
                                  # walk-forward OOS validation
python -m src.cli backtest-sweep # grid sweep: mode x period x horizon -> JSON
python -m src.cli backtest-sweep --param "MOMENTUM_FLOOR_PCT=-2.0,-0.5"
                                  # threshold grid on top (repeatable --param;
                                  # whitelisted signal/pair parameters only)
python -m src.cli portfolio-backtest
                                  # portfolio-level overnight-BUY sim vs SPY
                                  # buy-and-hold (top-5/day, stakes, costs)
python -m src.cli portfolio-backtest --spread-bps 8 --slippage-bps 2
                                  # portfolio sim with spread/slippage model
python -m src.cli regime-sweep   # regime threshold fire rates (diagnostic)
python -m src.cli session        # continuous intraday cycles until close
python -m src.cli overnight      # enter at the close, exit at the next open
python -m src.cli pairs-discover # form/re-form the cointegrated pair book
python -m src.cli pairs-backtest # walk-forward pairs replay -> pairs Kelly stats
python -m src.cli pairs-backtest --spread-bps 10 --slippage-bps 3
                                  # pairs backtest with execution costs
python -m src.cli pairs-scan     # pairs analysis only (signals + vetoes)
```

`.env` is loaded automatically at startup (python-dotenv) — no need to
export variables; real environment variables still take precedence.
pytest runs skip the `.env` load so tests stay hermetic.

Daily paper-trading routine (start-up → review → safe shutdown):
see `RUNBOOK.txt`.

The whale/flow layer (insider Form 4/144, 13F holders, unusual options,
dark-pool proxy) is enabled by default; without the optional keys it
degrades to "no data" sentinels and stays neutral. See the
Configuration table below and `.env.example` for where to get each key.

## Architecture

```
yfinance (cached) → quant (z-score + momentum + daily vol + volume
        features; BUYs pass fail-open confirmation gates: volume z /
        regime fear_z / gold-eq breadth)
    → regime (once per run: bonds, USD/JPY, GLD/SPY, VIX term structure,
        credit HYG/IEF, DXY, copper/gold, breadth → fear/liquidity z-scores
        → panic | risk_off | neutral)
    → 4 isolated LLM analysts (fundamental/technical/sentiment/macro)
        (macro analyst reads the regime snapshot)
    → 3 optional whale/flow analysts (insider Form 4/144, 13F holders,
        unusual options) — dispatched only when their data flag is on;
        per-analyst failures degrade to neutral placeholders
    → bull/bear debate
    → risk veto (citations / high-conf SELL / position cap /
        insider cluster sell / institutional exodus / put skew —
        the flow vetoes block BUYs only)
    → fixed-fractional sizing (base = equity × MAX_POSITION_PCT,
        scaled down by volatility (TARGET_DAILY_VOL / daily_vol, capped at 1),
        conviction (confidence / 0.95, floored at 0.25), regime
        (risk_off × REGIME_BUDGET_MULT), directional flow conviction
        (bearish flow reports shrink the budget; never scales up),
        debate winner (bear shrinks BUYs / bull shrinks SELLs) and an
        optional transaction-cost buffer (COST_BUFFER_PCT); capped by
        batch-level inverse-vol risk parity (INVOL_ENABLED), per-sector
        and per-correlation-cluster notional caps, and Kelly
        (the only scale-up factor, bounded by KELLY_MAX_MULT);
        SELLs capped at held qty; dust orders skipped;
        cumulative BUY notional ≤ equity)
    → Alpaca paper (market/day: BUY notional, SELL fractional qty)
    → SQLite audit log (decisions + emitted orders + regime snapshot)

Optional batch-level branch (--with-pairs; see src/pairs/):
    S&P 400 universe (Wikipedia + 7d cache + static fallback) →
        Engle-Granger cointegration on log prices (ADF ≤ COINT_ADF_PVALUE,
        half-life in [5, 60]d; same-sector + |corr| ≥ 0.70 pre-screen)
    → walk-forward Sharpe rank → dedupe (no reused ticker, per-sector cap)
    → persisted pair book (data/pairs_active.json; re-formed every
        PAIR_REBALANCE_DAYS — signals never re-fit between formations)
    → long-only spread signals (z from book stats: buy the cheap leg at
        |z| ≥ 1.5, mechanical exit at |z| < 0.25, stop at |z| > 3.5;
        .env-tuned 2026-09-13 via pairs-backtest --param sweeps)
    → risk vetoes (correlation breakdown / half-life drift / liquidity
        deterioration — deterministic, fail-open on absent data)
    → pair legs into the SAME planner: shared equity/sector/cluster caps,
        per-pair 3% (both legs) + total 30% rails, pairs Kelly (separate
        stats file, the only scale-up); pair legs skipped first when the
        equity cap binds; kind="pair" entries in the audit log
```

### Regime layer (deterministic, never LLM-decided)

Computed once per run from free yfinance data (see `src/data/regime.py`):

| Composite | Series | Signal |
|-----------|--------|--------|
| VIX term | ^VIX / ^VIX3M | ≥1.0 backwardation; +VIX ≥ 30 → **panic** |
| Fear | VIX level, GLD/SPY slope, copper/gold (inverted) | z > +1 → risk_off |
| Liquidity | USD/JPY slope, DXY slope (inverted), TLT slope, HYG/IEF slope (inverted), RSP & IWM vs SPY | z < −1 → risk_off |
| Curve | ^TNX − ^IRX | inverted → risk_off |

Gate thresholds are env-configurable (`VIX_BACKWARDATION`,
`VIX_PANIC_LEVEL`, `FEAR_Z_THRESHOLD`, `LIQUIDITY_Z_FLOOR`); junk values
fall back to the defaults above. Use `python -m src.cli regime-sweep` to
see each candidate threshold's historical fire rate before changing one.

Gates (same shape as the PDT gate — deterministic, fail-open on missing data):
- **panic** blocks BUYs (risk-adding); SELLs (exits) always pass
- **risk_off** multiplies the budget by `REGIME_BUDGET_MULT` (both sides; never scales up)
- any fetch failure degrades that component; total failure → neutral snapshot

### Quant BUY-side confirmation gates

Beyond the entry thresholds, BUYs must pass three fail-open
confirmation gates (SELLs are always ungated; missing data never
blocks a BUY):

| Gate | Env var | Reads | Blocked when |
|------|---------|-------|--------------|
| Volume confirmation | `VOLUME_Z_MIN_BUY` | bar volume z-score vs 20d | volume z below threshold |
| Fear regime | `FEAR_Z_MAX_BUY` | regime `fear_z` composite | fear_z above threshold |
| Breadth / risk-off | `BREADTH_GOLD_EQ_MIN_BUY` | regime gold/equity breadth z | gold_eq z below threshold |

The per-gate pass map is recorded in the quant dict (`gates` key) and
lands in the SQLite audit trail. The fear/breadth gates need the batch
regime snapshot (wired in `graph.py`); without one they pass.

### Pairs module (stat-arb, long-only — opt-in via `--with-pairs`)

`src/pairs/` adds cointegrated-pairs trading on a dynamic S&P 400
universe (Wikipedia scrape + 7-day cache + static fallback).  Pairs are
formed every `PAIR_REBALANCE_DAYS` by Engle-Granger on log prices
(ADF ≤ `COINT_ADF_PVALUE`, half-life within `[COINT_MIN_HALF_LIFE,
COINT_MAX_HALF_LIFE]`, same-sector + |corr| ≥ `PAIR_PRESCREEN_CORR`
pre-screen), ranked by walk-forward Sharpe, deduped (no reused ticker,
`MAX_PAIRS_PER_SECTOR`), and persisted to `PAIR_BOOK_PATH` — the
formation stats survive restarts and z-scores never re-fit between
formations.  Signals are **long-only conditioned mean reversion**: buy
the cheap leg at |z| ≥ `PAIR_Z_ENTRY`, mechanically exit held legs at
|z| < `PAIR_Z_EXIT` or stop at |z| > `PAIR_Z_STOP`; nothing is ever
sold short (SELL_EXIT is capped at held quantity).  BUY legs pass the
same PDT/panic gates as single-leg BUYs and share the equity, sector,
and cluster caps, plus per-pair (`MAX_PAIR_PCT`, both legs) and
total-pairs (`MAX_PAIRS_PCT`) rails; pair legs are skipped first when
the equity cap binds.  Sizing factors are scale-down-only except pairs
Kelly (`PAIR_KELLY_*`, separate `pairs_backtest_stats.json` from
`pairs-backtest` — never feeds intraday/overnight sizing).  Risk vetoes
(correlation breakdown, half-life drift, liquidity deterioration) are
deterministic and fail-open on absent data; a missing/stale book simply
trades no pairs.  The module is additive: `PAIRS_ENABLED=0` (or omitting
`--with-pairs`) leaves the pipeline bit-identical to before.

### Backtest layer (deterministic, quant signals only)

`python -m src.cli backtest --tickers AAPL,MSFT --period 2y --horizon 5`
replays the quant signal generator over historical bars (no LLM calls;
the full OHLCV frame feeds the volume gates) and writes win-rate /
payoff / Kelly-fraction stats **plus risk-adjusted metrics (Sharpe,
Sortino, Calmar, Omega — annualized from trade-horizon returns)** to
`BACKTEST_STATS_PATH` (default `data/backtest_stats.json`). With
`KELLY_ENABLED=1` (default) and a usable aggregate sample
(`n ≥ KELLY_MIN_TRADES`, positive fraction), paper-run/scan scale
budgets by a bounded Kelly multiplier — the only factor that may exceed
1.0 (see Safety).

`--mode overnight` scores the halo/overnight-drift strategy instead
(enter at the signal-day close, exit the next open) and writes to
`OVERNIGHT_STATS_PATH` (default `data/backtest_stats_overnight.json`),
with a `per_side` BUY/SELL split — the overnight strategy trades the
BUY side only.

**Primary Kelly source (production regime):** the overnight BUY side is
the production edge candidate. `python -m src.cli backtest --mode
overnight --primary` additionally promotes the BUY-side stats into
`BACKTEST_STATS_PATH` (as `mode: "overnight_buy"`), so intraday Kelly
sizing scales on the overnight edge. **Refreshed 2026-09-14** after the
overnight replay was fixed to apply the volume-confirmation gate exactly
as live does (it previously hid the `Volume` column, pinning
`volume_z` to 0.0 and validating a larger population than production
trades): with `VOLUME_Z_MIN_BUY=0.0` and the default 16 bps round-trip
cost model, the 2y 28-name BUY side is n=1123, win 0.489,
kelly=None (usable=False) — the cost-free edge (kelly 0.086, Sharpe
0.84) does not survive ~16 bps round-trip spread+slippage, consistent
with the portfolio backtest's break-even ≈ 10 bps finding. The sizer
treats a non-positive/absent Kelly as a no-op (factor 1.0), so sizing
stays neutral until an edge survives costs.

**Threshold sweeps:** `backtest-sweep` and `pairs-backtest` accept
repeatable `--param NAME=v1,v2` options that grid a whitelisted set of
signal/pair parameters on top of the mode × period × horizon matrix
(price frames are fetched once and reused across all combos; results
land in `data/backtest_param_sweep.json` / `data/pairs_param_sweep.json`
unless `--out` is given). 2026-09-13 sweep findings: relaxing
`MOMENTUM_FLOOR_PCT` to −2.0 improves the overnight BUY edge on 2y, 5y
and max (kelly 0.127/0.138/0.110 vs baseline 0.085/0.081/0.096); pairs
z-thresholds `1.5/0.25/3.5` beat `2.0/0.5/3.5` on both 2y and 5y.

**Grid sweeps:** `python -m src.cli backtest-sweep --periods
1y,2y,3y,5y,10y,max --horizons 1,5,10,20 --modes close,overnight` runs
the full mode × period × horizon matrix and writes aggregate +
per-ticker stats (all metrics) to `data/backtest_sweep.json`
(`--out` to change). Slow on purpose — `max` fetches full history and
every combo replays bar by bar. Key sweep findings (2026-09-08):
the aggregate edge is short-horizon only; the durable multi-decade
overnight BUY cohort is SPY / XLP / COST / CRM / ADBE (Sharpe 0.6–0.95,
n>1200 over 25y).

**Portfolio-level benchmark:** `python -m src.cli portfolio-backtest`
(`--period`, `--top-n`, `--stakes`, `--cost-bps`, `--spread-bps`, `--slippage-bps`,
`--benchmark`, `--out`; default `data/portfolio_backtest.json`) simulates the overnight
BUY edge as an actual portfolio — top-5 confidence-ranked candidates per
day, each a fixed equity stake (20%/10% variants), close→next-open with
round-trip costs, equity compounding daily — and scores it against a
SPY buy-and-hold on the same dates (total return, CAGR, Sharpe, Sortino,
max drawdown, daily-return correlation). The BUY rule is a vectorized
mirror of the replay's `generate_signal` path (equivalence
property-tested, including the volume gate). **Refreshed 2026-09-14**
(after the volume-gate alignment; 2y, 10 bps flat RT, 28 names): the
portfolio LOSES money — stake 0.20: total −5.4%, Sharpe −0.40, maxDD
−12.8% vs SPY total +33.8%, Sharpe 1.23, maxDD −18.6%; daily-return
corr ~0.07. With the volume gate now live-faithful, the overnight edge
no longer covers costs at the portfolio level either — the sleeve is
not deployable as-is on market orders (limit-order execution or a
stronger signal is required before revisit).

### Execution cost model (spread + slippage)

All backtest commands (`backtest`, `backtest-sweep`, `portfolio-backtest`,
`pairs-backtest`, `backtest-walkforward`) support a realistic execution
cost model via `--spread-bps` / `--slippage-bps`. Unset flags fall back
to the `SPREAD_BPS` / `SLIPPAGE_BPS` / `COST_SCALE_BY_VOL` env vars
(defaults 10 / 3 / on) — so `.env` configures the costs and explicit
flags override them. (`portfolio-backtest` additionally keeps its flat
`--cost-bps` round-trip default when no spread/slippage is requested.)
Since Alpaca paper has no commission, these are the primary execution
costs:

- **Spread** (bid-ask): paid once per round-trip (cross the spread)
- **Slippage**: paid on entry AND exit (market order impact)
- **Volatility scaling**: when `COST_SCALE_BY_VOL=1` (default), costs scale
  linearly with the signal's `daily_vol` relative to a 2% baseline

Round-trip cost = `spread_bps + 2 × slippage_bps` (scaled by vol if enabled).
For the default 10 bps spread + 3 bps slippage = 16 bps round-trip at 2% vol.

The cost model applies per-trade using each signal's `daily_vol` (for single-leg
and overnight) or without vol scaling (for pairs, which lack signal-level vol).

**Cost findings (2026-09-14, 2y overnight BUY, 28 names):** cost-free the
BUY side shows kelly 0.086 / Sharpe 0.84 (n=1123 with the volume gate);
with the default 16 bps round-trip the same population shows
kelly=None / negative Sharpe at every `VOLUME_Z_MIN_BUY` setting
(−2 … +1). The strategy-level overnight edge does not survive realistic
spread+slippage — treat cost-free backtests as upper bounds.

### Random baseline comparison

`backtest --random-baseline` runs a random signal generator alongside the
quant strategy to validate statistical edge. Random signals match a configurable
signal rate (default 10% of bars) and are scored under the SAME contract as
the strategy (close→close for `--mode close`, close→next-open for
`--mode overnight`), so the null distribution is comparable. Use
`--random-signal-rate` and `--random-seed` to control the baseline; the
aggregate is recorded in the stats JSON as a `random_baseline` block.

```bash
python -m src.cli backtest --tickers AAPL --period 2y --random-baseline
```

2026-09-14 run (2y overnight, 28 names, 16 bps RT): random null
n=1232, win 0.356, Sharpe −3.61 vs strategy aggregate n=3328, win
0.418, Sharpe −1.45 — the signal is better than random, but both are
negative after costs.

### Walk-forward / out-of-sample backtest

`backtest-walkforward` evaluates fixed parameters on unseen data using
expanding windows:

```bash
python -m src.cli backtest-walkforward --tickers AAPL --period 5y \
  --train-ratio 0.7 --step-days 63 --min-train-days 252
```

- Train on first `train_ratio` of data (min `min_train_days`)
- Test on next `step_days` bars
- Expand train window by `step_days` and repeat until end of data

Only trades whose ENTRY bar falls inside the test window are collected
(the signal generator still sees the full history up to each bar — the
live no-lookahead contract; fold slices get `horizon` bars of
look-ahead room so every in-window entry's outcome can be scored).
Fold trade populations are disjoint and exactly tile the post-train
region (property-tested), and the pooled `aggregate_oos` block is the
honest OOS read. 2026-09-14 run (2y overnight, 28 names, 16 bps RT):
84 folds, OOS aggregate n=990, win 0.433, kelly=None — no out-of-sample
edge after costs, matching the full-sample replay.

For parameter re-optimization per fold, run `backtest-sweep --param` on
each train window manually.

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| ALPACA_API_KEY | — | Paper trading key |
| ALPACA_SECRET_KEY | — | Paper trading secret |
| PAPER_ONLY | true | Hard-locked to paper |
| OPENROUTER_API_KEY | — | LLM routing |
| WATCHLIST | 10 default tickers | Comma-separated (`.env` currently ships a 28-name expanded universe: mega-cap tech + sector ETFs) |
| Z_BUY_ENTRY | -1.5 | Quant z-score below which a BUY fires |
| Z_SELL_ENTRY | 1.5 | Quant z-score above which a SELL fires |
| MOMENTUM_FLOOR_PCT | -0.5 | Min 10-day-MA slope (%/day) for a BUY (falling-knife gate) |
| VOLUME_Z_MIN_BUY | unset (off) | BUY gate: bar volume z-score must be ≥ threshold (fail-open without a Volume column) |
| FEAR_Z_MAX_BUY | unset (off) | BUY gate: regime fear_z must be ≤ threshold (fail-open without a regime snapshot) |
| BREADTH_GOLD_EQ_MIN_BUY | unset (off) | BUY gate: gold/equity breadth z must be ≥ threshold (extreme risk-off block) |
| ACCOUNT_EQUITY | 1000 | Fallback equity (live fetch used when keys present) |
| MAX_POSITION_PCT | 0.10 | Per-position budget fraction (hard cap: × kelly_max_mult when Kelly is active) |
| TARGET_DAILY_VOL | 0.02 | Daily-return vol at which the budget is unscaled; higher-vol names scale down only |
| MAX_SECTOR_PCT | 0.30 | Max BUY notional per sector (fraction of equity); sectorless tickers (ETFs) exempt |
| MAX_CORR_PCT | 0.30 | Max BUY notional per correlation cluster (pseudo-sector cap) |
| CORR_CLUSTER_THRESHOLD | 0.80 | Pairwise daily-return Pearson corr at or above which tickers cluster |
| CORR_MIN_OBSERVATIONS | 60 | Min aligned return observations before clustering (fail-open below) |
| CORR_CLUSTERING_ENABLED | 1 | 0 disables correlation clustering entirely |
| INVOL_ENABLED | 1 | 0 disables batch-level inverse-vol risk-parity caps |
| REGIME_BUDGET_MULT | 0.5 | Budget multiplier in risk_off regime (clamped to [0, 1]) |
| REGIME_ENABLED | 1 | 0 disables the regime fetch (offline runs/tests stay neutral) |
| REGIME_CACHE_ROOT | data/cache | Disk cache root for regime series |
| VIX_BACKWARDATION | 1.0 | VIX/VIX3M ratio at or above which the term structure is backwardated |
| VIX_PANIC_LEVEL | 25 | Spot VIX at or above which a backwardated term structure means panic (tuned 2026-09-08 from regime sweeps; 30 fired <3% of days) |
| FEAR_Z_THRESHOLD | 1.5 | fear_z above this → risk_off (tuned 2026-09-08; 1.0 fired 17.5% of 1y days) |
| LIQUIDITY_Z_FLOOR | −0.5 | liquidity_z below this → risk_off (tuned 2026-09-08; −1.0 never fired in 2y/5y) |
| DEBATE_OPPOSE_MULT | 0.75 | Factor applied when the debate winner opposes the signal (clamped [0.25, 1]) |
| COST_BUFFER_PCT | 0.0 | Estimated round-trip cost buffer (opt-in; scales budgets by 1 − buffer) |
| MIN_TRADE_NOTIONAL | 10 | Absolute dust floor: orders below this notional are skipped |
| MIN_TRADE_NOTIONAL_PCT | 0.01 | Equity-relative floor: effective floor = min(MIN_TRADE_NOTIONAL, max(1.0, equity × pct)); an absolute floor above the default wins over this scaling |
| SPREAD_BPS | 10.0 | Bid-ask spread in basis points (1 bp = 0.01%) for backtest cost model |
| SLIPPAGE_BPS | 3.0 | Market order slippage in basis points for backtest cost model |
| COST_SCALE_BY_VOL | 1 | Scale execution costs by daily_vol relative to 2% baseline (1=on, 0=off) |
| KELLY_ENABLED | 1 | 0 ignores backtest stats in sizing |
| KELLY_FRACTION_CAP | 0.25 | Backtest fraction treated as "full Kelly" for the multiplier mapping |
| KELLY_MAX_MULT | 1.5 | Ceiling of the Kelly multiplier (≤ 1.0 neutralizes scale-up) |
| KELLY_MIN_TRADES | 30 | Min aggregate replayed trades before stats are usable |
| BACKTEST_STATS_PATH | data/backtest_stats.json | Where `backtest` writes / sizing reads Kelly stats |
| OVERNIGHT_STATS_PATH | data/backtest_stats_overnight.json | `backtest --mode overnight` output (BUY-side stats drive the overnight cycle) |
| SESSION_MAX_BUYS_PER_TICKER | 1 | Max BUY submissions per ticker per session day |
| SESSION_MAX_SELLS_PER_TICKER | 1 | Max SELL submissions per ticker per session day |
| OVERNIGHT_MAX_NAMES | 5 | Max concurrent overnight entry names (confidence-ranked) |
| DAY_TRADE_DATES | — | ISO dates for PDT gate (restriction blocks BUYs, allows SELLs) |
| DECISIONS_DB | data/decisions.sqlite3 | Audit log |
| EDGAR_IDENTITY | — | Required by SEC for edgartools insider fetch (name + email) |
| SECURITIESDB_API_KEY | — | 13F institutional holder data (absent key → fail-open no-data) |
| FINRA_API_KEY | — | FINRA dark-pool ATS proxy (absent key → fail-open no-data) |
| INSIDER_ENABLED | 1 | 0 disables the insider fetch + analyst (behavior identical) |
| INSTITUTIONAL_ENABLED | 1 | 0 disables the 13F fetch + analyst |
| OPTIONS_FLOW_ENABLED | 1 | 0 disables the UOA scan + analyst |
| DARK_POOL_ENABLED | 1 | 0 disables the FINRA dark-pool proxy (macro context) |
| UOA_MIN_VOL_OI_RATIO | 3.0 | Unusual-options volume/OI threshold |
| UOA_MIN_PREMIUM | 25000 | Unusual-options dollar premium threshold |
| UOA_MAX_DTE | 45 | Max days-to-expiry scanned |
| INSIDER_LOOKBACK_DAYS | 90 | Form 4/144 lookback window |
| CLUSTER_INSIDER_THRESHOLD | 3 | Insiders selling within 30d that veto a BUY |
| FLOW_VETO_MIN_HOLDERS | 10 | Min 13F holders before the exodus veto can fire |
| UOA_PUT_SKEW_VETO | 0.30 | Call/put premium ratio below which a BUY is vetoed |
| PAIRS_ENABLED | 1 | 0 disables the pairs module (pipeline is then pairs-free) |
| UNIVERSE_REFRESH_DAYS | 7 | S&P 400 universe cache TTL (`UNIVERSE_*` bounds filter by ADV/mcap) |
| COINT_LOOKBACK_DAYS | 252 | Pair formation window (needs `PAIR_PRICE_PERIOD` ≥ it) |
| COINT_ADF_PVALUE | 0.05 | Engle-Granger ADF acceptance threshold |
| COINT_MIN/MAX_HALF_LIFE | 5 / 60 | Acceptable spread half-life window (days) |
| PAIR_PRESCREEN_CORR | 0.70 | \|return corr\| floor before ADF is run (runtime + multiplicity control) |
| PAIR_REBALANCE_DAYS | 21 | Pair (re-)formation cadence; signals are daily off the persisted book |
| MAX_PAIRS / MAX_PAIRS_PER_SECTOR | 10 / 3 | Concurrent-pair and per-sector pair limits |
| MIN_PAIR_SHARPE | 1.0 | Walk-forward Sharpe floor for a pair to enter the book |
| PAIR_Z_ENTRY / PAIR_Z_EXIT / PAIR_Z_STOP | 2.0 / 0.5 / 3.5 | Spread z-score entry / reversion-exit / stop-loss thresholds |
| MAX_PAIR_PCT | 0.03 | Per-pair BUY notional cap (BOTH legs combined) |
| MAX_PAIRS_PCT | 0.30 | Total pairs BUY notional cap (fraction of equity) |
| PAIR_BOOK_PATH | data/pairs_active.json | Persisted pair book (formation stats) |
| PAIR_KELLY_ENABLED | 1 | 0 ignores pairs backtest stats in pair sizing |
| PAIR_KELLY_FRACTION_CAP / PAIR_KELLY_MAX_MULT | 0.5 / 1.5 | Pairs Kelly mapping (ceiling pending Phase 4 validation) |
| PAIR_BACKTEST_STATS_PATH | data/pairs_backtest_stats.json | `pairs-backtest` output / pairs sizing input |

## Continuous operation (session & overnight cycles)

`python -m src.cli session --interval 30 --llm-refresh 120 [--submit]`
sleeps until the market opens (Alpaca `/v2/clock`), then cycles until
close: a FULL pipeline (all LLM analysts + debate) at open and every
`--llm-refresh` minutes, cheap quant-only cycles in between that reuse
the last full analysis' reports/debate (fresh prices drive fresh quant
signals; no LLM calls).  Cross-cycle discipline lives in a session
store (`session_actions` table in the decisions DB): at most
`SESSION_MAX_BUYS_PER_TICKER` BUYs / `SESSION_MAX_SELLS_PER_TICKER`
SELLs per ticker per day, no re-submission while an open order exists,
and a session-wide BUY cap — the planner budgets against live equity
minus today's submitted BUY notional.  Ctrl+C is safe (state persists
per action).

`python -m src.cli overnight [--submit]` implements the halo/overnight-
drift cycle validated by `backtest --mode overnight` (BUY-side edge is
positive; SELL-side is not): entries are the top `OVERNIGHT_MAX_NAMES`
approved BUYs by confidence, sized with the BUY-side overnight Kelly
stats (separate stats file — overnight experiments never feed intraday
sizing), submitted in the last `--entry-before-close` minutes; exits are
mechanical next-open sales at held quantity (from
`--exit-after-open-utc`), idempotent across restarts because entries
persist in the session store.  `--no-wait` runs one analysis-only pass
for testing (never submits).

## Safety

- Paper-only broker (live raises)
- Citation-gated LLM outputs (min_length=1) + NO-DATA sentinel citations on empty slices
- LLM list outputs tolerate dict-shaped entries: `citations`/debate points are
  coerced to plain strings at validation (the non-empty citation contract is
  still enforced); empty citations are still rejected
- Deterministic veto gates (no model risk)
- Regime gates deterministic + fail-open (panic blocks buys, risk_off scales down)
- Whale/flow vetoes deterministic, fail-open on absent data, and BUY-gated (exits always pass)
- Per-analyst failure isolation (one analyst error degrades to a neutral placeholder)
- Flow sizing is scale-down-only and only from successfully fetched data
- Debate-winner sizing is scale-down-only (a bear winner shrinks BUYs, never inflates)
- Kelly is the ONLY scale-up factor: bounded by KELLY_MAX_MULT (default 1.5×) and by the
  hard per-name cap `equity × MAX_POSITION_PCT × kelly_max_mult`; the cumulative BUY cap
  stays at equity; absent/junk/non-positive stats are a no-op
- Inverse-vol parity and the cost buffer only ever shrink budgets
- SELL orders capped at held quantity (skipped when unheld)
- Volatility scale-down + conviction weighting (never scales up)
- Per-sector and per-correlation-cluster notional caps (independent accumulators)
- BUY orders are submitted as dollar notional (actual spend equals the
  planned budget regardless of price drift); SELLs as fractional qty
  (exact position exits)
- Dust filter skips sub-minimum orders on both sides; the effective floor
  is min(MIN_TRADE_NOTIONAL, max($1, equity × MIN_TRADE_NOTIONAL_PCT)) so
  accounts of any size can trade and dust positions remain exitable
- Append-only SQLite decisions log (incl. emitted orders + regime)
- Cumulative BUY notional capped at equity
- `.env` auto-load (python-dotenv) is skipped under pytest so tests stay hermetic
- Console output is cp1252-safe: unencodable characters are replaced, never fatal
  (Windows consoles)
- Session/overnight cycles add cross-cycle caps on top of every per-run rail:
  per-ticker daily action budgets, no re-submission while an open order exists,
  session-wide BUY notional ≤ equity, and planner-level filtering so the pipeline
  submits exactly once per cycle; the daily BUY budget is enforced in BOTH
  directions (session gate counts overnight entries, overnight entries skip
  tickers the session already bought) — running both modes the same day can
  never buy the same ticker twice
- Overnight exits are mechanical and idempotent (state persisted in SQLite);
  entries never skip the veto/regime gates, and `--no-wait` test mode can never
  submit
- Pairs module (opt-in `--with-pairs`) adds no new risk direction: pair legs
  are long-only (SELL_EXIT capped at held qty, skipped when unheld — never a
  short), exits are mechanical full-quantity, BUY legs pass the same PDT/panic
  gates as single-leg BUYs and share the equity/sector/cluster accumulators,
  plus per-pair (MAX_PAIR_PCT, both legs) and total-pairs (MAX_PAIRS_PCT) rails
- Pair legs are skipped FIRST when the equity cap binds (single-leg edge with
  the validated Kelly stats has priority); sizing factors are scale-down-only
  except pairs Kelly (separate stats file, ceiling pending walk-forward
  validation — Sharpe < 1 keeps the multiplier neutral)
- Pairs formation stats persist in the pair book; z-scores never re-fit between
  formations; a missing/stale book trades no pairs (single-leg pipeline
  unaffected); risk vetoes (correlation breakdown, half-life drift, liquidity)
  fail open on absent data

## Testing

```bash
python -m pytest tests/ -v
```

See `concerns.txt` for the sizing-concerns registry (all items addressed
as of 2026-09-07; full covariance-based risk parity / concentration
remains open future work). 666 tests passing as of 2026-09-14 (includes
the pairs module, the `--param` sweep extensions, the portfolio-level
backtest, execution cost model, random baseline, walk-forward OOS
validation, and the overnight volume-gate live-alignment regression
tests).