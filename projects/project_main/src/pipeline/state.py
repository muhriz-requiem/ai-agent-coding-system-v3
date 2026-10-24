"""Shared pipeline state container.

Provides a TypedDict describing the state passed between pipeline stages,
plus a factory that returns a fresh state instance with empty buckets.
"""

from typing import TypedDict


class PipelineState(TypedDict):
    """Mutable state threaded through the pipeline stages.

    Keys:
        tickers: Tickers under consideration for the current run.
        reports: Analyst research reports accumulated so far.
        debates: Bull/bear debate transcripts accumulated so far.
        decisions: Final trading decisions accumulated so far.
        orders: Orders derived from decisions accumulated so far.
        as_of: Optional ISO timestamp anchoring the run, if set.
    """

    tickers: list[str]
    reports: list
    debates: list
    decisions: list
    orders: list
    as_of: str | None


def new_state(tickers: list[str]) -> PipelineState:
    """Return a fresh PipelineState for the given tickers.

    Every call produces a new dict with its own list instances (no shared
    mutable defaults), with ``tickers`` copied from the input and all other
    buckets empty.
    """
    return PipelineState(
        tickers=list(tickers),
        reports=[],
        debates=[],
        decisions=[],
        orders=[],
        as_of=None,
    )
