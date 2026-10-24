from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date as _date
from pathlib import Path


def _is_iso8601_timestamp(value: object) -> bool:
    """Return True when value is an ISO-8601 timestamp string."""
    if not isinstance(value, str):
        return False

    pattern = (
        r"^(\d{4})-(\d{2})-(\d{2})"
        r"T(\d{2}):(\d{2})"
        r"(?::(\d{2})(?:\.\d+)?)?"
        r"(?:[Zz]|[+-]\d{2}:?\d{2})?$"
    )

    try:
        match = re.match(pattern, value)
        if match is None:
            return False

        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        _date(year, month, day)

        hour = int(match.group(4))
        minute = int(match.group(5))
        second_group = match.group(6)
        second = int(second_group) if second_group is not None else 0

        if hour > 23 or minute > 59 or second > 59:
            return False

        return True
    except Exception:
        return False


def _normalize_event(event: dict) -> dict:
    """Normalise a parsed JSON object to the exact four-key event schema."""
    if not isinstance(event, dict):
        event = {}

    ts = event.get("ts")
    normalized_ts = ts if _is_iso8601_timestamp(ts) else None

    level = event.get("level")
    if level is None or (isinstance(level, str) and level == ""):
        normalized_level = "INFO"
    else:
        normalized_level = str(level).upper()

    message = event.get("message")
    normalized_message = "" if message is None else str(message)

    if "data" not in event or event.get("data") is None:
        normalized_data = {}
    else:
        data = event.get("data")
        if isinstance(data, dict):
            normalized_data = data
        else:
            normalized_data = {"_raw": str(data)}

    return {
        "ts": normalized_ts,
        "level": normalized_level,
        "message": normalized_message,
        "data": normalized_data,
    }


def parse_rows(lines: Sequence[str]) -> dict:
    """Parse JSONL lines into normalised events.

    Blank or whitespace-only lines are ignored. Invalid JSON lines are counted
    as malformed and do not produce events. This function never raises.
    """
    events = []
    malformed = 0

    try:
        iterator = iter(lines)
    except Exception:
        return {"events": events, "malformed": malformed}

    try:
        for raw_line in iterator:
            try:
                if isinstance(raw_line, str):
                    line = raw_line.strip()
                    if not line:
                        continue
                elif isinstance(raw_line, (bytes, bytearray)):
                    line = raw_line.strip()
                    if not line:
                        continue
                else:
                    line = raw_line

                parsed = json.loads(line)
            except Exception:
                malformed += 1
                continue

            if not isinstance(parsed, dict):
                malformed += 1
                continue

            try:
                events.append(_normalize_event(parsed))
            except Exception:
                malformed += 1
    except Exception:
        pass

    return {"events": events, "malformed": malformed}


def load_log(path: str | Path) -> dict:
    """Load a JSONL log file and parse its rows.

    Missing or unreadable files return the empty result and never raise.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
            lines = handle.readlines()
        return parse_rows(lines)
    except Exception:
        return {"events": [], "malformed": 0}


def count_messages(events: Sequence[dict]) -> dict[str, int]:
    """Count messages, sorted by descending count then ascending message."""
    counts = {}

    try:
        iterator = iter(events)
    except Exception:
        return {}

    try:
        for event in iterator:
            try:
                if not isinstance(event, dict):
                    continue

                message = event.get("message")
                key = "" if message is None else str(message)
            except Exception:
                continue

            counts[key] = counts.get(key, 0) + 1
    except Exception:
        pass

    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
