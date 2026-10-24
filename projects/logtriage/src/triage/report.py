"""Report composition with a reconciling event ledger."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.triage import cost, failures, latency, providers
from src.triage.reader import count_messages, load_log


def analyse(events: Sequence[dict], malformed: int = 0) -> dict[str, Any]:
    """
    Assemble a complete report from parsed events.

    Args:
        events: Sequence of parsed event dictionaries.
        malformed: Count of malformed lines during parsing.

    Returns:
        Dict with exactly five top-level keys: events, latency, failures,
        providers, cost. All values are JSON-serialisable and deterministic.
    """
    total = len(events)
    ledger = count_messages(events)

    # Verify reconciliation: sum of ledger counts must equal total events
    # This is a sanity check; the contract guarantees it by construction.
    assert sum(ledger.values()) == total, "Ledger reconciliation failed"

    return {
        "events": {
            "total": total,
            "malformed": malformed,
            "ledger": ledger,
        },
        "latency": {
            "roles": latency.role_latency(events),
            "outliers": latency.outliers(events),
        },
        "failures": {
            "shapes": failures.failure_shapes(events),
            "generic_signature": failures.generic_signature(events),
        },
        "providers": {
            "stats": providers.provider_stats(events),
            "cooldowns": providers.cooldown_events(events),
            "fallback_rate": providers.fallback_rate(events),
        },
        "cost": {
            "totals": cost.token_totals(events),
            "per_role": cost.role_tokens(events),
            "sizes": cost.exchange_sizes(events),
        },
    }


def analyse_file(path: str | Path) -> dict[str, Any]:
    """
    Load a log file and return its analysed report.

    Args:
        path: Path to the log file.

    Returns:
        Report dict from analyse(). Missing file yields empty report with
        malformed == 0.
    """
    try:
        parsed = load_log(path)
        return analyse(parsed["events"], malformed=parsed["malformed"])
    except FileNotFoundError:
        return analyse([], 0)
