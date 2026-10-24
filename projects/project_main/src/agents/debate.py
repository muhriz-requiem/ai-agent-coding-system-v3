"""Bull/Bear research debate over analyst reports.

Import contract (pinned by tests/test_debate.py): this module may import ONLY
stdlib (asyncio/typing) plus ``from src.agents.models import ...``.  It must
never import ``src.agents.llm``, any ``src.broker.*`` module, or third-party
packages, so that importing this module can never break test collection.
"""

import asyncio
import typing

from src.agents.models import AnalystReport, DebateRecord

_SYSTEM_PROMPT = (
    "You are a rigorous research debate moderator. Given analyst reports for a "
    "ticker, you argue BOTH the bull and the bear perspectives, grounding every "
    "claim in the reports' citations. Respond only with structured data.\n"
    "\n"
    "Respond with exactly one JSON object with exactly these fields: "
    '`ticker` (string), `bull_points` (list of plain strings), `bear_points` '
    '(list of plain strings), `winner` (one of "bull", "bear", "undecided"), and '
    '`citations` (non-empty list of plain strings quoting the analyst citations '
    "above; strings only, never objects). Do not add any other fields."
)


def _build_user_prompt(ticker: str, reports: dict[str, AnalystReport]) -> str:
    """Build the single user prompt for the debate.

    Contains, for EVERY report in ``reports``, the report's agent name and its
    thesis text verbatim (no case-folding, truncation, or reformatting), plus
    the ticker and explicit instructions to argue bull and bear sides with
    citations drawn from the reports.
    """
    lines: list[str] = [
        f"Ticker: {ticker}",
        "",
        "Analyst reports to consider:",
    ]

    for key, report in reports.items():
        agent_name = getattr(report, "agent", key)
        thesis = getattr(report, "thesis", "")
        citations = getattr(report, "citations", [])
        lines.append("- agent: " + str(agent_name))
        lines.append("  thesis: " + str(thesis))
        if citations:
            lines.append("  citations: " + "; ".join(str(c) for c in citations))

    lines.append("")
    lines.append(
        "Debate instructions: argue the bull perspective AND the bear "
        "perspective for this ticker. For each side, explicitly cite the "
        "citations from the analyst reports above (bull case with supporting "
        "evidence, bear case with counter-evidence). Cover every report's "
        "thesis."
    )

    return "\n".join(lines)


async def run_debate(
    ticker: str,
    reports: dict[str, AnalystReport],
    client,
) -> DebateRecord:
    """Run one bull/bear debate for ``ticker`` over ``reports``.

    Builds a single user prompt carrying every report's agent name and thesis
    verbatim plus bull/bear instructions, then delegates to the injected
    ``client`` via ``client.complete_json(system, user, DebateRecord,
    effort='low')`` and returns the resulting DebateRecord unchanged.
    """
    system = _SYSTEM_PROMPT
    user = _build_user_prompt(ticker, reports)
    result = await client.complete_json(system, user, DebateRecord, effort="low")
    return typing.cast(DebateRecord, result)


__all__ = ["run_debate"]
