"""Tests for src/triage/report.py -- composition.

The composition contract is that one call gives a complete picture, and that
the event ledger is reconciled against the parts: every event must be accounted
for by exactly one category, or `unclassified` grows. That reconciliation is the
only thing that makes a whole-report number trustworthy.
"""

import json

import pytest


def _impl_ready() -> bool:
    import importlib.util

    try:
        return (importlib.util.find_spec("src.triage.report") is not None
                and importlib.util.find_spec("src.triage.reader") is not None)
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.triage.report import analyse, analyse_file  # noqa: E402


@pytest.fixture
def analyse_rows(tmp_path):
    """Write JSONL *text* to a temp log and return the parsed report.

    A fixture rather than a module-level helper: the helper was originally
    defined at the bottom of the file while eight call sites above referenced a
    slightly different name, so every test in this file died with
    ``NameError``. The Reviewer caught it; the fix is to stop hand-rolling a
    temp-file dance in each test.
    """
    def _run(rows):
        path = tmp_path / "agent_log.jsonl"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return analyse_file(path)
    return _run


_MIXED = [
    '{"ts": "2026-10-04T12:00:00+00:00", "level": "INFO", '
    '"message": "llm_exchange", "data": {"model_role": "builder_initial", '
    '"model": "groq:qwen", "duration_s": 10.0, "prompt_head": "abc", '
    '"response_head": "def"}}',
    '{"ts": "2026-10-04T12:00:01+00:00", "level": "INFO", '
    '"message": "llm_usage", "data": {"model_role": "builder_initial", '
    '"model": "groq:qwen", "prompt_tokens": 100, "completion_tokens": 40, '
    '"total_tokens": 140}}',
    '{"ts": "2026-10-04T12:00:02+00:00", "level": "WARNING", '
    '"message": "llm_model_failed_trying_fallback", "data": '
    '{"model": "groq:qwen", "error": "rate limited"}}',
    '{"ts": "2026-10-04T12:00:03+00:00", "level": "INFO", '
    '"message": "provider_cooldown", "data": {"provider": "groq", '
    '"reason": "http_429"}}',
    '{"ts": "2026-10-04T12:00:04+00:00", "level": "ERROR", '
    '"message": "task_escalated", "data": {"task_id": "T-1", '
    '"reason": "ERR_SCHEMA_T-1_MAPPING"}}',
    '{"ts": "2026-10-04T12:00:05+00:00", "level": "ERROR", '
    '"message": "task_escalated", "data": {"task_id": "T-2", '
    '"reason": "ERR_SCHEMA_T-2_MAPPING"}}',
    # generic_signature needs TWO DISTINCT recurring shapes that share exactly
    # one meaningful token; a single shape proves nothing by comparison.
    '{"ts": "2026-10-04T12:00:05.5+00:00", "level": "ERROR", '
    '"message": "task_escalated", "data": {"task_id": "T-3", '
    '"reason": "ERR_PIPELINE_T-1_SCHEMA"}}',
    '{"ts": "2026-10-04T12:00:05.7+00:00", "level": "ERROR", '
    '"message": "task_escalated", "data": {"task_id": "T-4", '
    '"reason": "ERR_PIPELINE_T-2_SCHEMA"}}',
    '{"ts": "2026-10-04T12:00:06+00:00", "level": "INFO", '
    '"message": "task_started", "data": {"task_id": "T-1"}}',
    "not json at all",
]


def test_analyse_top_level_keys(analyse_rows):
    report = analyse([])
    assert set(report) == {"events", "latency", "failures", "providers",
                           "cost"}


def test_analyse_counts_events_and_malformed(analyse_rows):
    report = analyse_rows(_MIXED)
    assert report["events"]["total"] == 9
    assert report["events"]["malformed"] == 1


def test_analyse_event_ledger_reconciles(analyse_rows):
    """Every parsed event lands in exactly one bucket, or in `unclassified`."""
    report = analyse_rows(_MIXED)
    ledger = report["events"]["ledger"]
    assert sum(ledger.values()) == report["events"]["total"]
    assert ledger["llm_exchange"] == 1
    assert ledger["llm_usage"] == 1
    assert ledger["task_escalated"] == 4
    assert ledger["task_started"] == 1


def test_analyse_surfaces_the_failure_cluster(analyse_rows):
    report = analyse_rows(_MIXED)
    shapes = report["failures"]["shapes"]
    # Two distinct recurring shapes, each seen twice.
    assert len(shapes) == 2
    assert all(s["count"] == 2 for s in shapes)
    # They share exactly one meaningful token once <N> is discarded.
    assert report["failures"]["generic_signature"] == "SCHEMA"


def test_analyse_surfaces_provider_reliability(analyse_rows):
    providers = analyse_rows(_MIXED)["providers"]
    assert providers["stats"]["groq"]["calls"] == 1
    assert providers["stats"]["groq"]["errors"] == 1
    assert providers["cooldowns"][0]["count"] == 1
    assert providers["fallback_rate"] == pytest.approx(0.5)


def test_analyse_surfaces_cost(analyse_rows):
    cost = analyse_rows(_MIXED)["cost"]
    assert cost["totals"]["total"] == 140
    assert cost["per_role"]["builder_initial"] == 140
    assert cost["sizes"]["count"] == 1


def test_analyse_on_empty_input_is_a_well_formed_empty_report(analyse_rows):
    """Every empty value's TYPE is pinned.

    "Well-formed empty value" in the objective left the Builder guessing, and
    it guessed wrong twice (``outliers: {}`` and ``cooldowns: {}`` where the
    underlying functions return lists). Both readings are falsy, so the
    escalation was entirely my specification's fault.
    """
    report = analyse_rows([])
    assert report["events"]["total"] == 0
    assert report["events"]["malformed"] == 0
    assert report["events"]["ledger"] == {}
    assert report["latency"]["roles"] == {}          # dict: role -> stats
    assert report["latency"]["outliers"] == []       # list, per latency.outliers
    assert report["failures"]["shapes"] == []        # list, per failures.failure_shapes
    assert report["failures"]["generic_signature"] == ""
    assert report["providers"]["stats"] == {}        # dict: provider -> stats
    assert report["providers"]["cooldowns"] == []    # list, per providers.cooldown_events
    assert report["providers"]["fallback_rate"] == 0.0
    assert report["cost"]["totals"] == {
        "prompt": 0, "completion": 0, "total": 0, "records": 0}
    assert report["cost"]["per_role"] == {}          # dict: role -> tokens
    assert report["cost"]["sizes"] == {
        "max_prompt_head": 0, "max_response_head": 0,
        "mean_prompt_head": 0.0, "count": 0}


def test_analyse_is_json_serialisable(analyse_rows):
    json.dumps(analyse_rows(_MIXED), sort_keys=True)


def test_analyse_is_deterministic(analyse_rows):
    a = analyse_rows(_MIXED)
    b = analyse_rows(_MIXED)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_analyse_file_reads_from_disk(tmp_path, analyse_rows):
    path = tmp_path / "agent_log.jsonl"
    path.write_text("\n".join(_MIXED) + "\n", encoding="utf-8")
    report = analyse_file(path)
    assert report["events"]["total"] == 9
    assert report["events"]["malformed"] == 1


def test_analyse_file_on_missing_path_is_an_empty_report(tmp_path, analyse_rows):
    report = analyse_file(tmp_path / "nope.jsonl")
    assert report["events"]["total"] == 0
    assert report["events"]["malformed"] == 0


def _analyse_file_text(rows):
    """Helper: analyse JSONL *text* lines (write to tmp then read)."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "log.jsonl"
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return analyse_file(p)