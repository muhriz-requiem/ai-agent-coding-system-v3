"""Tests for src/triage/failures.py -- failure clustering.

The normalisation rule is pinned by examples because it is the crux: two
escalations that differ only in a task id are ONE design problem, while two
that merely end in the same word are two.
"""

import pytest


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.triage.failures") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.triage.failures import (failure_shapes,  # noqa: E402
                                 generic_signature, normalise_signature)


def _escalated(signature, task_id="T-1", message="task_escalated"):
    return {"ts": "2026-10-04T12:00:00.000+00:00", "level": "ERROR",
            "message": message, "data": {"task_id": task_id,
                                         "reason": signature}}


def _changes(task_id="T-1", signature=None):
    data = {"task_id": task_id}
    if signature is not None:
        data["error_signature"] = signature
    return {"ts": "2026-10-04T12:00:00.000+00:00", "level": "WARNING",
            "message": "task_changes_requested", "data": data}


# ---------------------------------------------------------- normalisation
@pytest.mark.parametrize("raw,expected", [
    ("ERR_PIPELINE_WIRING", "PIPELINE_WIRING"),
    ("TASK:T-7013B:pipeline wiring", "<N>_PIPELINE_WIRING"),
    ("ERR_SCHEMA_T-1_MAPPING", "SCHEMA_<N>_MAPPING"),
    ("ERR_SCHEMA_T-2_MAPPING", "SCHEMA_<N>_MAPPING"),
    ("ERR_UNIQUE0_THING", "UNIQUE0_THING"),
    ("ERR_UNIQUE1_THING", "UNIQUE1_THING"),
    ("ERR_A_B", "A_B"),
    ("", ""),
    (None, ""),
])
def test_normalise_signature(raw, expected):
    assert normalise_signature(raw) == expected


def test_task_ids_collapse_so_the_same_bug_is_one_shape():
    assert normalise_signature("ERR_SCHEMA_T-1_MAPPING") == \
        normalise_signature("ERR_SCHEMA_T-9_MAPPING")


def test_a_shared_generic_suffix_does_not_collapse_distinct_failures():
    """The false-positive this rule exists to avoid: three unrelated failures
    that merely end in the same word are NOT one systemic problem."""
    assert normalise_signature("ERR_UNIQUE0_THING") != \
        normalise_signature("ERR_UNIQUE1_THING")


def test_normalisation_is_idempotent():
    once = normalise_signature("ERR_SCHEMA_T-1_MAPPING")
    assert normalise_signature(once) == once


def test_normalisation_is_case_and_separator_insensitive():
    assert normalise_signature("err-schema-t-1-mapping") == \
        normalise_signature("ERR_SCHEMA_T_1_MAPPING")


# --------------------------------------------------------- failure_shapes
def test_failure_shapes_counts_escalations_by_shape():
    events = [_escalated("ERR_SCHEMA_T-1_MAPPING", "T-1"),
              _escalated("ERR_SCHEMA_T-2_MAPPING", "T-2"),
              _escalated("ERR_SCHEMA_T-3_MAPPING", "T-3")]
    shapes = failure_shapes(events)
    assert len(shapes) == 1
    assert shapes[0]["count"] == 3
    assert shapes[0]["shape"] == "SCHEMA_<N>_MAPPING"
    assert shapes[0]["task_ids"] == ["T-1", "T-2", "T-3"]


def test_failure_shapes_sorts_by_count_then_shape():
    events = [_escalated("ERR_ALPHA_X", "T-1"),
              _escalated("ERR_BETA_Y", "T-2"),
              _escalated("ERR_BETA_Y", "T-3")]
    shapes = failure_shapes(events)
    assert [s["shape"] for s in shapes] == ["BETA_Y", "ALPHA_X"]
    assert shapes[0]["count"] == 2


def test_failure_shapes_keeps_signatures_sorted_and_deduped():
    events = [_escalated("ERR_SCHEMA_T-2_MAPPING", "T-2"),
              _escalated("ERR_SCHEMA_T-1_MAPPING", "T-1"),
              _escalated("ERR_SCHEMA_T-1_MAPPING", "T-1")]
    entry = failure_shapes(events)[0]
    assert entry["signatures"] == ["ERR_SCHEMA_T-1_MAPPING",
                                   "ERR_SCHEMA_T-2_MAPPING"]


def test_failure_shapes_ignores_events_without_a_reason():
    events = [{"ts": None, "level": "ERROR", "message": "task_escalated",
               "data": {"task_id": "T-9"}}]
    assert failure_shapes(events) == []


def test_failure_shapes_ignores_changing_events():
    assert failure_shapes([_changes(signature="ERR_X")]) == []


def test_failure_shapes_on_empty_is_empty_list():
    assert failure_shapes([]) == []


def test_failure_shapes_entry_keys_are_exact():
    entry = failure_shapes([_escalated("ERR_A_B")])[0]
    assert set(entry) == {"shape", "count", "signatures", "task_ids"}


# ------------------------------------------------------- generic_signature
def test_generic_signature_finds_the_token_two_shapes_share():
    """The root cause is what DISTINGUISHES a cluster, so it needs >= 2 shapes.

    SCHEMA_<N>_MAPPING and PIPELINE_<N>_SCHEMA share exactly one meaningful
    token once the instance-id placeholder is discarded: SCHEMA.
    """
    events = [_escalated("ERR_SCHEMA_T-1_MAPPING", "T-1"),
              _escalated("ERR_SCHEMA_T-2_MAPPING", "T-2"),
              _escalated("ERR_PIPELINE_T-1_SCHEMA", "T-3"),
              _escalated("ERR_PIPELINE_T-2_SCHEMA", "T-4")]
    assert generic_signature(events) == "SCHEMA"


def test_generic_signature_ignores_the_placeholder_token():
    """Two shapes can agree on nothing but ``<N>`` -- that is not a root cause.

    A Builder returned ``<N>`` here once, because the placeholder is trivially
    shared by every normalised signature. It must be excluded or every cluster
    would report the same meaningless answer.
    """
    events = [_escalated("ERR_SCHEMA_T-1_MAPPING", "T-1"),
              _escalated("ERR_SCHEMA_T-2_MAPPING", "T-2"),
              _escalated("ERR_QUANT_T-1_SIGNAL", "T-3"),
              _escalated("ERR_QUANT_T-2_SIGNAL", "T-4")]
    assert generic_signature(events) == ""


def test_generic_signature_needs_two_distinct_qualifying_shapes():
    """One shape cannot establish a root cause -- there is nothing to compare."""
    events = [_escalated("ERR_SCHEMA_T-1_MAPPING", "T-1"),
              _escalated("ERR_SCHEMA_T-2_MAPPING", "T-2"),
              _escalated("ERR_SCHEMA_T-3_MAPPING", "T-3")]
    assert generic_signature(events) == ""


def test_generic_signature_is_empty_when_several_tokens_are_shared():
    """Ambiguous agreement is not a finding.

    ``A_<N>_B`` and ``A_<N>_B_C`` share both A and B.  Guessing which one
    mattered is exactly the sort of confident invention this function exists to
    avoid, so it declines.
    """
    events = [_escalated("ERR_A_T-1_B", "T-1"),
              _escalated("ERR_A_T-2_B", "T-2"),
              _escalated("ERR_A_T-3_B_C", "T-3"),
              _escalated("ERR_A_T-4_B_C", "T-4")]
    assert generic_signature(events) == ""


def test_generic_signature_on_empty_is_empty_string():
    assert generic_signature([]) == ""