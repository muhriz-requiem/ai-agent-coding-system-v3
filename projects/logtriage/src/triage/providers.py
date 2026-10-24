"""Provider reliability metrics for triage."""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Sequence


def _split(model: str) -> tuple[str, str]:
    """Split a model string into (provider, rest).

    Uses partition on the first colon. If no colon is present,
    returns (model, model).

    Args:
        model: The model string, e.g., "openrouter:qwen/qwen3.8-27b:free"

    Returns:
        Tuple of (provider, rest). If no colon, both elements are the model.
    """
    provider, sep, rest = str(model).partition(":")
    if not rest:
        return (model, model)
    return (provider, rest)


def provider_stats(events: Sequence[dict]) -> dict[str, dict]:
    """Compute provider statistics from events.

    Args:
        events: Sequence of event dictionaries.

    Returns:
        Dictionary mapping provider to statistics with keys:
        calls, errors, timeouts, success_rate, median_s.
        Keys are sorted ascending. Empty input returns {}.
    """
    if not events:
        return {}

    # Track per-provider data
    provider_data = defaultdict(lambda: {
        "calls": 0,
        "errors": 0,
        "timeouts": 0,
        "durations": []
    })

    for event in events:
        message = event.get("message")
        data = event.get("data", {})

        if message == "llm_exchange":
            duration = data.get("duration_s")
            model = data.get("model")
            if isinstance(duration, (int, float)) and model is not None:
                provider = _split(model)[0]
                provider_data[provider]["calls"] += 1
                provider_data[provider]["durations"].append(float(duration))

        elif message == "llm_model_failed_trying_fallback":
            error_text = data.get("error", "")
            model = data.get("model")
            if model is not None:
                provider = _split(model)[0]
                if "wall-clock budget" in str(error_text):
                    provider_data[provider]["timeouts"] += 1
                else:
                    provider_data[provider]["errors"] += 1

    # Build result with sorted keys
    result = {}
    for provider in sorted(provider_data.keys()):
        data = provider_data[provider]
        calls = data["calls"]
        errors = data["errors"]
        timeouts = data["timeouts"]
        # Success rate counts EVERY attempt that produced no answer, timeouts
        # included: a provider that times out half the time is not reliable.
        # (This previously excluded timeouts to satisfy a test that asserted
        # 1.0 for one success plus one timeout -- a spec/test contradiction the
        # Reviewer caught across three attempts.  The test was wrong: the
        # objective's calls/(calls+errors+timeouts) is the defensible semantic.)
        total_attempts = calls + errors + timeouts
        if total_attempts == 0:
            success_rate = 1.0
        else:
            success_rate = calls / total_attempts

        durations = data["durations"]
        if durations:
            median_s = median(durations)
        else:
            median_s = 0.0

        result[provider] = {
            "calls": calls,
            "errors": errors,
            "timeouts": timeouts,
            "success_rate": success_rate,
            "median_s": median_s
        }

    return result


def cooldown_events(events: Sequence[dict]) -> list[dict]:
    """Extract cooldown event counts by provider and reason.

    Args:
        events: Sequence of event dictionaries.

    Returns:
        List of dictionaries with keys provider, reason, count.
        Sorted by descending count, then ascending provider, then ascending reason.
        Events missing provider or reason are skipped. Empty input returns [].
    """
    if not events:
        return []

    counts = defaultdict(int)

    for event in events:
        if event.get("message") != "provider_cooldown":
            continue

        data = event.get("data", {})
        provider = data.get("provider")
        reason = data.get("reason")

        if provider is None or reason is None:
            continue

        counts[(provider, reason)] += 1

    result = [
        {"provider": provider, "reason": reason, "count": count}
        for (provider, reason), count in sorted(
            counts.items(),
            key=lambda x: (-x[1], x[0][0], x[0][1])
        )
    ]

    return result


def fallback_rate(events: Sequence[dict]) -> float:
    """Compute overall fallback rate across all providers.

    Failed calls (same definition as provider_stats) divided by total calls
    (successful + failed), as a float. Returns 0.0 when there is no evidence
    at all -- no evidence is not evidence of failure.

    Args:
        events: Sequence of event dictionaries.

    Returns:
        Fallback rate as float between 0.0 and 1.0.
    """
    if not events:
        return 0.0

    total_calls = 0
    total_failed = 0

    for event in events:
        message = event.get("message")
        data = event.get("data", {})

        if message == "llm_exchange":
            duration = data.get("duration_s")
            model = data.get("model")
            if isinstance(duration, (int, float)) and model is not None:
                total_calls += 1

        elif message == "llm_model_failed_trying_fallback":
            model = data.get("model")
            if model is not None:
                total_calls += 1
                total_failed += 1

    if total_calls == 0:
        return 0.0

    return total_failed / total_calls
