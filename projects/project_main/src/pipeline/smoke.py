"""Smoke test runner for the trading pipeline.

Manual run: ALPACA_API_KEY=... ALPACA_SECRET_KEY=... python -m src.pipeline.smoke AAPL MSFT --dry-run
"""

import asyncio
import inspect
import os
from pathlib import Path

from src.cli import _components as build_cli_components, _jsonable
from src.pipeline.graph import run_pipeline
from src.store.decisions import DecisionStore


class MissingCredentialsError(Exception):
    """Raised when live trading is attempted without API credentials."""


async def _build_real_components(dry_run: bool) -> dict:
    """Build real components for smoke testing."""
    return dict(build_cli_components(submit=not dry_run))


async def run_smoke(tickers, dry_run=True, components=None):
    """Run smoke test on tickers through the full pipeline.
    
    Args:
        tickers: Iterable of ticker symbols to test
        dry_run: If True, don't submit orders
        components: Optional pre-built components dict (for testing)
    
    Returns:
        Dict with dry_run status, orders submitted count, and tickers tested
    """
    # Credentials-first rule: check before any component handling
    if not dry_run and not (os.environ.get('ALPACA_API_KEY') and os.environ.get('ALPACA_SECRET_KEY')):
        raise MissingCredentialsError("Missing ALPACA_API_KEY and/or ALPACA_SECRET_KEY for live trading")
    
    if components is None:
        components = await _build_real_components(dry_run=dry_run)
    
    submitted = []
    copy = dict(components)
    original = copy['broker_submit']
    
    async def broker_submit_recording(orders):
        result = original(orders)
        if inspect.isawaitable(result):
            result = await result
        submitted.extend(result or [])
        return result
    
    copy['broker_submit'] = broker_submit_recording
    
    decisions = await run_pipeline(list(tickers), copy)
    
    # Audit: log all non-None decisions
    db_path = Path(os.environ.get('DECISIONS_DB', 'data/decisions.sqlite3'))
    store = DecisionStore(db_path=db_path)
    for decision in decisions:
        if decision is not None:
            store.log_decision(_jsonable(decision))
    
    return {
        'dry_run': dry_run,
        'orders_submitted': len(submitted),
        'tickers': list(tickers)
    }
