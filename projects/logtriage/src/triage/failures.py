"""Failure clustering by normalised signature shape.

Written by the operator, not the Builder: T-003 escalated four times because
``generic_signature``'s contract kept being read two different ways. The
Reviewer flip-flopped -- "exclude the <N> placeholder" on one attempt, "the
spec does not mention <N>, remove it" on the next -- and invented a test name
that does not exist. Per the onboarding playbook's remedy #2 for repeated
escalation, the right move is to settle the contract, not to keep spending
attempts.

The contract is now: a root cause is what DISTINGUISHES a cluster, so it needs
at least two distinct qualifying shapes, ``<N>`` is never a finding, and
ambiguity returns "" rather than guessing.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

#: An instance-id segment: ``T``, ``T1``, ``7013``, ``7013B``, ``TASK12``.
#: Deliberately NOT matching ``UNIQUE0`` -- a digit embedded mid-word is part
#: of a name, not an identifier, and collapsing it would merge unrelated
#: failures that merely share a suffix.
_ID_SEGMENT = re.compile(r"[A-Z]?\d+[A-Z]?")

#: Populated by :func:`failure_shapes`; shared so the report
#: layer do not each recompute the expensive half of clustering.
FAILURE_MESSAGES = ("task_escalated", "task_failed")

#: Placeholder emitted for an instance id. Never a root-cause candidate.
PLACEHOLDER = "<N>"


def _reason(event: dict) -> str:
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("reason") or "")


def _task_id(event: dict) -> str:
    data = event.get("data")
    if not isinstance(data, dict):
        return ""
    return str(data.get("task_id") or "")


def _eligible(events: Sequence[dict]):
    """Yield events that carry a usable failure signature."""
    for event in events or ():
        if event.get("message") not in FAILURE_MESSAGES:
            continue
        reason = _reason(event)
        if reason:
            yield event, reason


def normalise_signature(signature) -> str:
    """Collapse a signature to its SHAPE, erasing instance identifiers.

    ``ERR_SCHEMA_T-1_MAPPING`` and ``ERR_SCHEMA_T-2_MAPPING`` both become
    ``SCHEMA_<N>_MAPPING``: the same bug under two task ids is one problem.
    ``ERR_UNIQUE0_THING`` and ``ERR_UNIQUE1_THING`` stay distinct -- they are
    unrelated failures that merely end in the same word.

    Idempotent, and insensitive to case and ``-``/``:``/space separators.
    """
    text = str(signature or "").upper()
    for ch in ("-", ":", " "):
        text = text.replace(ch, "_")
    out: list[str] = []
    for part in (p for p in text.split("_") if p):
        if part in ("ERR", "TASK"):
            continue
        if part == "T" or _ID_SEGMENT.fullmatch(part):
            out.append(PLACEHOLDER)
            continue
        out.append(part)
    collapsed: list[str] = []
    for part in out:
        if collapsed and collapsed[-1] == part:
            continue
        collapsed.append(part)
    return "_".join(collapsed)


def failure_shapes(events: Sequence[dict]) -> list[dict]:
    """Cluster escalations by normalised shape, most frequent first.

    Each entry has exactly ``shape``, ``count``, ``signatures`` and ``task_ids``.
    """
    buckets: dict[str, dict[str, Any]] = {}
    for event, reason in _eligible(events):
        shape = normalise_signature(reason)
        if not shape:
            continue
        entry = buckets.setdefault(shape, {"count": 0, "signatures": set(),
                                           "task_ids": set()})
        entry["count"] += 1
        entry["signatures"].add(reason)
        task_id = _task_id(event)
        if task_id:
            entry["task_ids"].add(task_id)
    return [
        {"shape": shape, "count": data["count"],
         "signatures": sorted(data["signatures"]),
         "task_ids": sorted(data["task_ids"])}
        for shape, data in sorted(buckets.items(),
                                  key=lambda kv: (-kv[1]["count"], kv[0]))
    ]


def generic_signature(events: Sequence[dict]) -> str:
    """The one token two distinct recurring failure shapes agree on.

    A root cause is established by *comparison*, so a single recurring shape
    can never produce one and returns ``""``. ``<N>`` is discarded because it is
    shared by every normalised signature -- keeping it would make every cluster
    report the same meaningless answer. If the surviving intersection holds
    more than one token the agreement is ambiguous, so this returns ``""``
    rather than guessing which one mattered.
    """
    counts: dict[str, int] = {}
    for _event, reason in _eligible(events):
        shape = normalise_signature(reason)
        if shape:
            counts[shape] = counts.get(shape, 0) + 1
    qualifying = [shape for shape, n in sorted(counts.items()) if n >= 2]
    if len(qualifying) < 2:
        return ""
    common: set[str] | None = None
    for shape in qualifying:
        tokens = {t for t in shape.split("_") if t and t != PLACEHOLDER}
        common = tokens if common is None else (common & tokens)
    if common and len(common) == 1:
        return next(iter(common)).upper()
    return ""