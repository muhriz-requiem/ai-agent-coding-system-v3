"""Pairs trading module (stat-arb, long-only conditioned mean reversion).

Submodules:
    universe       -- dynamic S&P 400 universe (Wikipedia scrape + cache)
    cointegration  -- Engle-Granger engine (log prices, ADF, half-life)
    book           -- persisted pair book (formation stats survive restarts)
    screening      -- pair ranking + dedupe (walk-forward Sharpe)
    signals        -- spread z-score signals (long-only, from book stats)
    risk           -- pairs veto gates (BUY-side only, fail-open)
    sizing         -- dataclass adapter over the pure pair planner
    backtest       -- walk-forward pairs backtest (Kelly stats file)

Import safety: importing ``src.pairs`` is side-effect free; the heavy
statsmodels dependency is imported by ``cointegration`` (a hard
dependency per the plan).  All degradation is fail-open: an empty
universe yields no pairs, a missing/stale book yields no signals, and
the single-leg pipeline is unaffected in every path.
"""
