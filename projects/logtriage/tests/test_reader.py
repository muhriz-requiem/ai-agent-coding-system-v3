"""Tests for src/triage/reader.py -- JSONL ingestion + normalisation.

Expected values were computed independently in a scratch reference script; they
are not derived from the implementation.
"""

import json


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.triage.reader") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.triage.reader import (count_messages, load_log,  # noqa: E402
                              parse_rows)

_ROWS = [
    '{"ts": "2026-10-04T12:00:00.000+00:00", "level": "INFO", '
    '"message": "task_started", "data": {"task_id": "T-1"}}',
    '{"ts": "2026-10-04T12:00:01.000+00:00", "level": "WARNING", '
    '"message": "task_changes_requested", "data": {"task_id": "T-1"}}',
    '{"ts": "2026-10-04T12:00:02.000+00:00", "level": "INFO", '
    '"message": "task_started", "data": {"task_id": "T-2"}}',
]


# ------------------------------------------------------------------ happy
def test_parse_rows_returns_one_event_per_valid_line():
    out = parse_rows(_ROWS)
    assert out["malformed"] == 0
    assert len(out["events"]) == 3
    assert out["events"][0]["message"] == "task_started"
    assert out["events"][0]["data"]["task_id"] == "T-1"


def test_level_is_uppercased_and_message_coerced_to_str():
    rows = ['{"ts": "2026-01-01T00:00:00+00:00", "level": "info", '
            '"message": "x", "data": {}}']
    assert parse_rows(rows)["events"][0]["level"] == "INFO"

    rows = ['{"ts": "2026-01-01T00:00:00+00:00", "level": "INFO", '
            '"message": 42, "data": {}}']
    assert parse_rows(rows)["events"][0]["message"] == "42"


def test_missing_level_defaults_to_info_and_missing_data_to_empty():
    rows = ['{"ts": "2026-01-01T00:00:00+00:00", "message": "bare"}']
    event = parse_rows(rows)["events"][0]
    assert event["level"] == "INFO"
    assert event["data"] == {}


# --------------------------------------------------------------- malformed
def test_malformed_json_is_counted_not_fatal():
    rows = _ROWS[:1] + ["{not json"] + _ROWS[1:]
    out = parse_rows(rows)
    assert out["malformed"] == 1
    assert len(out["events"]) == 3


def test_non_iso_timestamp_yields_none_and_is_not_malformed():
    """The real reference log contains {"ts": "probe"} -- a tool that dies on
    it cannot diagnose the log that contains it."""
    rows = ['{"ts": "probe", "level": "INFO", "message": "external_probe", '
            '"data": {}}']
    out = parse_rows(rows)
    assert out["malformed"] == 0
    assert out["events"][0]["ts"] is None
    assert out["events"][0]["message"] == "external_probe"


def test_non_dict_data_is_wrapped_under_raw():
    rows = ['{"ts": "2026-01-01T00:00:00+00:00", "level": "INFO", '
            '"message": "odd", "data": [1, 2]}']
    event = parse_rows(rows)["events"][0]
    assert isinstance(event["data"], dict)
    assert event["data"]["_raw"] == "[1, 2]"


def test_blank_lines_are_ignored_entirely():
    out = parse_rows(["", "   ", *_ROWS, ""])
    assert len(out["events"]) == 3
    assert out["malformed"] == 0


# -------------------------------------------------------------------- empty
def test_empty_input_is_valid():
    out = parse_rows([])
    assert out == {"events": [], "malformed": 0}


def test_count_messages_sorts_by_count_then_key():
    events = parse_rows(_ROWS)["events"]
    counts = count_messages(events)
    # task_started=2 outranks task_changes_requested=1
    assert list(counts) == ["task_started", "task_changes_requested"]
    assert counts["task_started"] == 2


def test_count_messages_on_empty_is_empty_dict():
    assert count_messages([]) == {}


def test_load_log_reads_a_file(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(_ROWS) + "\n", encoding="utf-8")
    out = load_log(path)
    assert len(out["events"]) == 3
    assert out["malformed"] == 0


def test_load_log_on_missing_file_is_empty_not_an_exception(tmp_path):
    out = load_log(tmp_path / "nope.jsonl")
    assert out == {"events": [], "malformed": 0}


def test_load_log_accepts_a_str_path(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text(_ROWS[0] + "\n", encoding="utf-8")
    assert len(load_log(str(path))["events"]) == 1


def test_every_event_has_exactly_the_four_keys():
    for event in parse_rows(_ROWS)["events"]:
        assert set(event) == {"ts", "level", "message", "data"}


def test_parse_rows_is_deterministic():
    a = parse_rows(_ROWS)
    b = parse_rows(_ROWS)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)