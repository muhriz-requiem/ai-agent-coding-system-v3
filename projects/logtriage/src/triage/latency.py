"""Latency analysis for LLM exchange events."""

import math
from collections.abc import Sequence


def percentile(values: Sequence[float], q: float) -> float:
    """Return the nearest-rank percentile of ``values`` without interpolation."""
    ordered = sorted(values)
    if not ordered:
        return 0.0

    q_value = float(q)
    if math.isnan(q_value):
        q_value = 0.0
    if q_value < 0.0:
        q_value = 0.0
    elif q_value > 100.0:
        q_value = 100.0

    n = len(ordered)
    index = math.ceil(q_value / 100.0 * n) - 1
    if index < 0:
        index = 0
    elif index >= n:
        index = n - 1

    return float(ordered[index])


def role_latency(events: Sequence[dict]) -> dict[str, dict]:
    """Aggregate eligible llm_exchange events by model role."""
    groups: dict[str, list[float]] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("message") != "llm_exchange":
            continue

        data = event.get("data")
        if not isinstance(data, dict):
            continue

        duration = data.get("duration_s")
        role = data.get("model_role")

        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        if not isinstance(role, str):
            continue

        groups.setdefault(role, []).append(float(duration))

    result: dict[str, dict] = {}
    for role in sorted(groups):
        samples = groups[role]
        result[role] = {
            "count": len(samples),
            "median": percentile(samples, 50),
            "p90": percentile(samples, 90),
            "max": float(max(samples)),
        }

    return result


def outliers(events: Sequence[dict], factor: float = 1.5) -> list[dict]:
    """Return IQR-based outlier entries for roles with at least four samples."""
    groups: dict[str, list[float]] = {}

    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("message") != "llm_exchange":
            continue

        data = event.get("data")
        if not isinstance(data, dict):
            continue

        duration = data.get("duration_s")
        role = data.get("model_role")

        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            continue
        if not isinstance(role, str):
            continue

        groups.setdefault(role, []).append(float(duration))

    found: list[dict] = []
    for role, samples in groups.items():
        if len(samples) < 4:
            continue

        q1 = percentile(samples, 25)
        q3 = percentile(samples, 75)
        iqr = q3 - q1
        threshold = q3 + factor * iqr
        max_duration = float(max(samples))

        if max_duration > threshold:
            found.append(
                {
                    "model_role": role,
                    "duration_s": max_duration,
                    "threshold": float(threshold),
                }
            )

    found.sort(key=lambda item: (-item["duration_s"], item["model_role"]))
    return found
