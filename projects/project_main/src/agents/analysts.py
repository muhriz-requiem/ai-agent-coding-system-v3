"""Specialist analysts: information-isolated reports, run concurrently.

``run_analysts`` fans a market-data snapshot out to the specialist analysts
(fundamental, technical, sentiment, macro, plus the optional whale/flow
analysts insider, institutional, options_flow).  Information isolation is
enforced by construction: each analyst's prompt is rendered from exactly one
slice of ``inputs`` -- the fundamental analyst sees only
``inputs['fundamentals']``, the technical analyst only ``inputs['prices']``,
and so on.  A report can therefore only ever cite evidence from its own
slice.  All LLM calls are dispatched concurrently through
:func:`asyncio.gather`.

Dispatch gating (whale/flow feature flags): the optional analysts run only
when their input slice is present.  With the fetch components absent (flag
off / disabled data source) the slice is ``None`` and no prompt is built,
so flags-off behavior and LLM cost are identical to the four-analyst run.

Failure isolation: a single analyst raising (LLMError after retries,
validation, transport) degrades to a neutral placeholder report and never
kills the batch -- matching the regime layer's fail-open philosophy.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol, cast

import pandas as pd

try:  # Canonical home of the shared report schema.
    from src.agents.models import AnalystReport
except ImportError:  # pragma: no cover - alternate layout fallback
    try:
        from src.agents.schemas import AnalystReport  # type: ignore[no-redef]
    except ImportError:  # pragma: no cover - keeps the module importable standalone
        from pydantic import BaseModel

        class AnalystReport(BaseModel):  # type: ignore[no-redef]
            """Minimal stand-in used only when the shared schema is unavailable."""

            analyst: str
            ticker: str
            stance: str = "neutral"
            confidence: float = 0.5
            summary: str = ""
            citations: list[str] = []


logger = logging.getLogger(__name__)

__all__ = ["ANALYSTS", "run_analysts"]

#: Fixed execution (and result-keying) order of the specialist analysts.
CORE_ANALYSTS: tuple[str, ...] = ("fundamental", "technical", "sentiment", "macro")

#: Whale/flow analysts: dispatched ONLY when their input slice is present
#: (see :func:`_analyst_specs`); the slice key matches the analyst name.
OPTIONAL_ANALYSTS: tuple[str, ...] = ("insider", "institutional", "options_flow")

#: Fixed execution (and result-keying) order of all specialist analysts.
ANALYSTS: tuple[str, ...] = CORE_ANALYSTS + OPTIONAL_ANALYSTS

#: Sampling effort requested from the LLM client on every analyst call.
_EFFORT = "low"

#: Cap on price observations embedded in the technical prompt (most recent kept).
_MAX_PRICE_POINTS = 500

#: Cap on list entries embedded in whale/flow prompts (most significant kept).
_MAX_SLICE_ROWS = 10

#: Per-analyst mandate rendered into the system prompt.
_MANDATES: dict[str, str] = {
    "fundamental": (
        "assess valuation, profitability, growth and balance-sheet strength "
        "from company fundamentals"
    ),
    "technical": (
        "assess trend, momentum, volatility and key levels from the historical "
        "price series"
    ),
    "sentiment": (
        "assess news tone, catalysts and information flow from the supplied "
        "news items"
    ),
    "macro": (
        "assess the macroeconomic backdrop and regime from the supplied macro "
        "indicators"
    ),
    "insider": (
        "assess insider conviction from Form 4/144 clusters and net flow"
    ),
    "institutional": (
        "assess smart money positioning from 13F holdings changes and whale "
        "alerts"
    ),
    "options_flow": (
        "assess informed money signals from unusual options volume, premium, "
        "and call/put skew"
    ),
}

#: Label of the single data block each analyst receives.
_DATA_LABELS: dict[str, str] = {
    "fundamental": "FUNDAMENTALS",
    "technical": "PRICE SERIES",
    "sentiment": "NEWS ITEMS",
    "macro": "MACRO INDICATORS",
    "insider": "INSIDER FILINGS",
    "institutional": "13F HOLDINGS",
    "options_flow": "OPTIONS FLOW",
}


class AnalystClient(Protocol):
    """Minimal LLM client contract used by the analysts."""

    def complete_json(
        self,
        system: str,
        user: str,
        model: type[AnalystReport],
        *,
        effort: str = "low",
    ) -> AnalystReport | Awaitable[AnalystReport]:
        """Return a validated report for the prompt pair (or an awaitable of one)."""
        ...


# --------------------------------------------------------------------------- #
# Serialization helpers -- each touches exactly ONE input slice.              #
# --------------------------------------------------------------------------- #


def _dumps(data: Any) -> str:
    """JSON-encode ``data``, coercing pandas/numpy scalars via ``str``."""
    return json.dumps(data, default=str)


def _number(value: Any) -> Any:
    """Coerce a scalar to a JSON-safe float (``None`` when missing)."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _serialize_fundamentals(fundamentals: Any) -> str:
    """Serialize the fundamental slice and nothing else.

    Empty/absent slices carry a NO-DATA sentinel so the analyst always has
    a literal line to cite (citations ``min_length=1``).
    """
    if fundamentals is None:
        return _dumps({"notice": "NO FUNDAMENTALS DATA: slice absent"})
    payload = dict(fundamentals) if isinstance(fundamentals, Mapping) else {}
    if not payload:
        payload["notice"] = "NO FUNDAMENTALS DATA: no fields available"
    return _dumps(payload)


def _serialize_macro(macro: Any) -> str:
    """Serialize the macro slice and nothing else (sentinel when empty)."""
    if macro is None:
        return _dumps({"notice": "NO MACRO DATA: slice absent"})
    payload = dict(macro) if isinstance(macro, Mapping) else {}
    if not payload:
        payload["notice"] = "NO MACRO DATA: no indicators available"
    return _dumps(payload)


def _serialize_news(news: Any) -> str:
    """Serialize the news slice and nothing else (sentinel when empty)."""
    if news is None:
        return _dumps({"notice": "NO NEWS ITEMS: slice absent"})
    if isinstance(news, Mapping):
        payload = dict(news)
        if not payload:
            payload["notice"] = "NO NEWS ITEMS: no items available"
        return _dumps(payload)
    items = list(news)
    if not items:
        return _dumps({"notice": "NO NEWS ITEMS: news=[]"})
    return _dumps(items)


def _serialize_prices(prices: Any) -> str:
    """Serialize the price slice (summary stats + per-date points) and nothing else."""
    if prices is None:
        return _dumps({"observations": 0})
    if not isinstance(prices, pd.Series):
        return _dumps(prices)
    series = prices.dropna()
    if series.empty:
        return _dumps({"observations": 0})
    window = series.iloc[-_MAX_PRICE_POINTS:]
    try:
        low: Any = _number(series.min())
        high: Any = _number(series.max())
    except (TypeError, ValueError):
        low, high = None, None
    return _dumps(
        {
            "observations": int(len(series)),
            "points_shown": int(len(window)),
            "start": str(series.index[0]),
            "end": str(series.index[-1]),
            "first": _number(series.iloc[0]),
            "last": _number(series.iloc[-1]),
            "min": low,
            "max": high,
            "points": {str(index): _number(value) for index, value in window.items()},
        }
    )


def _top_by(rows: Any, key: str, cap: int = _MAX_SLICE_ROWS) -> list:
    """Keep the ``cap`` most significant dict rows by ``key`` magnitude."""
    usable = [row for row in (rows or []) if isinstance(row, dict)]
    return sorted(
        usable,
        key=lambda row: (
            abs(row.get(key)) if isinstance(row.get(key), (int, float)) else 0.0
        ),
        reverse=True,
    )[:cap]


def _flow_payload(
    data: Any, absent_notice: str, notices: dict[str, Any]
) -> dict:
    """Copy a fetch envelope and attach the NO-DATA sentinel when needed.

    The sentinel (a line starting with ``NO ``) is what the analyst must
    cite when the slice carries no usable data -- it satisfies the
    ``citations`` ``min_length=1`` contract without fabricating evidence.
    """
    payload = dict(data) if isinstance(data, Mapping) else {}
    if data is None:
        payload["notice"] = absent_notice
        return payload
    builder = notices.get(payload.get("status"))
    if builder is not None:
        payload["notice"] = builder(payload.get("reason"))
    return payload


def _window_days(data: Any) -> Any:
    if isinstance(data, Mapping):
        return data.get("window_days", "?")
    return "?"


def _serialize_insider(data: Any) -> str:
    """Serialize the insider slice (sentinel-aware, size-capped)."""
    payload = _flow_payload(
        data,
        "NO INSIDER DATA AVAILABLE: slice absent",
        {
            "no_data": lambda reason: (
                "NO INSIDER FILINGS: form4=[], form144=[] "
                f"(window: {_window_days(data)}d)"
            ),
            "error": lambda reason: (
                f"NO INSIDER DATA AVAILABLE: {reason or 'fetch failed'}"
            ),
        },
    )
    payload["form4"] = _top_by(payload.get("form4"), "value")
    payload["form144"] = list(payload.get("form144") or [])[:_MAX_SLICE_ROWS]
    return _dumps(payload)


def _serialize_institutional(data: Any) -> str:
    """Serialize the institutional slice (sentinel-aware, size-capped)."""
    payload = _flow_payload(
        data,
        "NO 13F DATA AVAILABLE: slice absent",
        {
            "no_data": lambda reason: f"NO 13F HOLDINGS DATA: {reason or 'no coverage'}",
            "error": lambda reason: (
                f"NO 13F DATA AVAILABLE: {reason or 'fetch failed'}"
            ),
        },
    )
    payload["top_holders"] = _top_by(payload.get("top_holders"), "value")
    payload["whale_alerts"] = list(payload.get("whale_alerts") or [])[:_MAX_SLICE_ROWS]
    return _dumps(payload)


def _serialize_options_flow(data: Any) -> str:
    """Serialize the options-flow slice (sentinel-aware, size-capped)."""
    payload = _flow_payload(
        data,
        "NO OPTIONS DATA AVAILABLE: slice absent",
        {
            "no_data": lambda reason: (
                f"NO UNUSUAL OPTIONS ACTIVITY: {reason or 'no contracts above thresholds'}"
            ),
            "error": lambda reason: (
                f"NO OPTIONS DATA AVAILABLE: {reason or 'fetch failed'}"
            ),
        },
    )
    payload["unusual_contracts"] = _top_by(payload.get("unusual_contracts"), "premium")
    return _dumps(payload)


# --------------------------------------------------------------------------- #
# Prompt builders -- isolation invariant: each prompt embeds exactly one      #
# serialized slice and no other slice is ever read while building it.         #
# --------------------------------------------------------------------------- #


def _system_prompt(name: str, ticker: str) -> str:
    """Role definition plus the mandatory evidence-gated citation rules."""
    return (
        f"You are the {name} analyst covering {ticker}. Mandate: {_MANDATES[name]}. "
        "You receive ONLY the data supplied in the user message; you have no access "
        "to any other market data and must not assume, recall or invent any.\n"
        "\n"
        "EVIDENCE-GATED CITATIONS (mandatory):\n"
        "1. Ground every claim in `summary` in the supplied data only.\n"
        "2. Every entry in `citations` must point at evidence that literally appears "
        "in the supplied data (a field name, value, date or headline). Never "
        "fabricate a citation and never cite data you were not given.\n"
        "3. No evidence, no claim: if the supplied data is insufficient, say so in "
        "`summary` and lower `confidence`.\n"
        "4. NO-DATA sentinel: if the supplied data contains a line starting with "
        "`NO ` (for example `NO INSIDER FILINGS: ...`), your `citations` must be "
        "exactly that sentinel line -- it is the evidence that the data is "
        "unavailable. Never fabricate other evidence and never return an empty "
        "`citations` list.\n"
        "5. Each entry in `citations` must be a plain string (one string per "
        "piece of evidence). Never return citations as objects or nested "
        "arrays.\n"
        "\n"
        "Respond with exactly one JSON object conforming to the AnalystReport schema."
    )


def _user_prompt(name: str, ticker: str, data_json: str) -> str:
    """Task instructions plus the analyst's single, isolated data block."""
    label = _DATA_LABELS[name]
    return (
        f"TICKER: {ticker}\n"
        f"ANALYST ROLE: {name}\n"
        "\n"
        f"{label} (JSON):\n"
        f"{data_json}\n"
        "\n"
        "TASK:\n"
        f"1. Analyze the {label.lower()} above for {ticker} strictly on its own; "
        "do not speculate about anything that is not contained in it.\n"
        "2. Write `thesis` so that every claim is backed by evidence from the "
        "data above.\n"
        "3. Build `citations` under the evidence-gating rules: each citation must "
        "reference the exact field, value, date or headline from the data above "
        "that supports the claim it is attached to.\n"
        f'4. Set `agent` to "{name}", `ticker` to "{ticker}", `stance` to '
        '"bullish", "bearish" or "neutral", and `confidence` to a float in '
        "[0.0, 1.0].\n"
        "5. Return only the JSON object."
    )


#: Input key and serializer for every analyst (one slice, one serializer).
_INPUT_KEYS: dict[str, str] = {
    "fundamental": "fundamentals",
    "technical": "prices",
    "sentiment": "news",
    "macro": "macro",
    "insider": "insider",
    "institutional": "institutional",
    "options_flow": "options_flow",
}

_SERIALIZERS: dict[str, Any] = {
    "fundamental": _serialize_fundamentals,
    "technical": _serialize_prices,
    "sentiment": _serialize_news,
    "macro": _serialize_macro,
    "insider": _serialize_insider,
    "institutional": _serialize_institutional,
    "options_flow": _serialize_options_flow,
}


def _analyst_specs(ticker: str, inputs: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    """Render ``(analyst, system, user)`` triples, one per specialist analyst.

    Isolation invariant: the ``user`` prompt of each analyst embeds exactly one
    serialized slice of ``inputs`` and no other slice is read while building it.

    Dispatch gating: the optional whale/flow analysts (:data:`OPTIONAL_ANALYSTS`)
    are included ONLY when their input slice is present (non-``None``).  Absent
    slices mean the corresponding fetch component is disabled/absent, so no
    prompt is built and no LLM call is made for them -- feature flags off leave
    the four core analysts unchanged.
    """
    specs: list[tuple[str, str, str]] = []
    for name in ANALYSTS:
        key = _INPUT_KEYS[name]
        if name in OPTIONAL_ANALYSTS and inputs.get(key) is None:
            continue  # dispatch gating (flag off / component absent)
        serializer = _SERIALIZERS[name]
        data_json = serializer(inputs.get(key))
        specs.append((name, _system_prompt(name, ticker), _user_prompt(name, ticker, data_json)))
    return specs


# --------------------------------------------------------------------------- #
# Execution                                                                    #
# --------------------------------------------------------------------------- #


async def _complete(client: AnalystClient, system: str, user: str) -> AnalystReport:
    """Invoke ``complete_json`` once, awaiting async clients transparently."""
    outcome: AnalystReport | Awaitable[AnalystReport] = client.complete_json(
        system, user, AnalystReport, effort=_EFFORT
    )
    if inspect.isawaitable(outcome):
        return cast(AnalystReport, await outcome)
    return outcome


def _placeholder_report(name: str, ticker: str) -> AnalystReport:
    """Neutral placeholder for a failed analyst (never raises).

    Keeps ``citations`` at exactly one entry (the ``min_length=1`` contract)
    and ``confidence`` at 0.0 so downstream sizing treats the analyst as
    having contributed nothing.
    """
    label = _DATA_LABELS.get(name, name.upper())
    citation = f"{label}: unavailable this run"
    payload = {
        "agent": name,
        "ticker": ticker,
        "stance": "neutral",
        "confidence": 0.0,
        "thesis": citation,
        "citations": [citation],
    }
    try:
        return AnalystReport.model_validate(payload)
    except Exception:  # minimal fallback schema uses `summary`, not `thesis`
        payload["summary"] = payload.pop("thesis")
        return AnalystReport.model_validate(payload)


async def _guarded_complete(
    client: AnalystClient, name: str, ticker: str, system: str, user: str
) -> AnalystReport:
    """One analyst call with failure isolation.

    Any failure (``LLMError`` after retries, validation, transport) degrades
    to a neutral placeholder report so one analyst can never kill the batch
    -- matching the regime layer's fail-open philosophy.
    """
    try:
        return await _complete(client, system, user)
    except Exception as exc:
        logger.warning("analyst %s failed for %s: %s", name, ticker, exc)
        return _placeholder_report(name, ticker)


async def run_analysts(
    ticker: str,
    inputs: dict,
    client: AnalystClient,
) -> dict[str, AnalystReport]:
    """Run the specialist analysts concurrently over isolated input slices.

    Each analyst is prompted with exactly one slice of ``inputs``:

    * ``fundamental``    -- ``inputs['fundamentals']`` (dict)
    * ``technical``      -- ``inputs['prices']`` (``pandas.Series``)
    * ``sentiment``      -- ``inputs['news']`` (list of dicts)
    * ``macro``          -- ``inputs['macro']`` (dict)
    * ``insider``        -- ``inputs['insider']`` (status envelope; optional)
    * ``institutional``  -- ``inputs['institutional']`` (envelope; optional)
    * ``options_flow``   -- ``inputs['options_flow']`` (envelope; optional)

    The optional whale/flow analysts are dispatched only when their slice is
    present (feature flags on); missing slices degrade to no dispatch, and
    any extra keys are ignored (never leaked into any prompt).  Every prompt
    instructs evidence-gated citations, including the NO-DATA sentinel rule.

    Args:
        ticker: Instrument symbol the reports are about.
        inputs: Market-data snapshot; missing optional slices skip their
            analysts entirely.
        client: LLM client exposing
            ``complete_json(system, user, model, effort=...)``.

    Returns:
        Mapping of analyst name to its ``AnalystReport``, keyed in
        :data:`ANALYSTS` order (one entry per dispatched analyst).
    """
    snapshot: Mapping[str, Any] = inputs or {}
    specs = _analyst_specs(ticker, snapshot)
    logger.debug("run_analysts: dispatching %s for %s", [name for name, _, _ in specs], ticker)

    reports = await asyncio.gather(
        *(
            _guarded_complete(client, name, ticker, system, user)
            for name, system, user in specs
        )
    )

    result: dict[str, AnalystReport] = {
        name: report for (name, _, _), report in zip(specs, reports)
    }
    logger.debug("run_analysts: completed analysts for %s", sorted(result))
    return result
