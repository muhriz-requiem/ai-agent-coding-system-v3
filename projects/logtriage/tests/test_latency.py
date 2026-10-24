"""Tests for src/triage/latency.py -- percentiles and outliers.

Expected percentile values come from an independent scratch reference using
``math.ceil(q/100*n)`` nearest-rank; the convention is pinned deliberately
because interpolation would give different answers and the whole point is
reproducibility.
"""

import pytest


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.triage.latency") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.triage.latency import (outliers, percentile,  # noqa: E402
                                role_latency)


def _event(message="llm_exchange", role="builder_initial", duration=5.0,
           model="m"):
    return {"ts": "2026-10-04T12:00:00.000+00:00", "level": "INFO",
            "message": message, "data": {"model_role": role,
                                         "model": model,
                                         "duration_s": duration}}


# -------------------------------------------------------------- percentile
@pytest.mark.parametrize("values,q,expected", [
    ([1, 2, 3, 4, 5], 50, 3.0),
    ([1, 2, 3, 4, 5], 90, 5.0),
    ([1, 2, 3, 4, 5], 0, 1.0),
    ([1, 2, 3, 4], 50, 2.0),
    ([1, 2, 3, 4], 25, 1.0),
    ([7], 90, 7.0),
    ([3, 1, 2], 50, 2.0),
])
def test_percentile_is_nearest_rank(values, q, expected):
    assert percentile(values, q) == expected


def test_percentile_sorts_its_input():
    assert percentile([9, 1, 5, 3], 50) == percentile([1, 3, 5, 9], 50)


def test_percentile_of_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_percentile_clamps_out_of_range_quantiles():
    assert percentile([1, 2, 3], -50) == 1.0
    assert percentile([1, 2, 3], 500) == 3.0


def test_percentile_does_not_mutate_its_argument():
    values = [3, 1, 2]
    percentile(values, 50)
    assert values == [3, 1, 2]


# ------------------------------------------------------------ role_latency
def test_role_latency_reports_count_median_p90_max():
    events = [_event(duration=d) for d in (10, 20, 30, 40)]
    stats = role_latency(events)
    assert stats["builder_initial"]["count"] == 4
    assert stats["builder_initial"]["median"] == 20.0
    assert stats["builder_initial"]["p90"] == 40.0
    assert stats["builder_initial"]["max"] == 40.0


def test_role_latency_groups_by_role():
    events = [_event(role="builder_initial", duration=10),
              _event(role="reviewer_feature", duration=90)]
    stats = role_latency(events)
    assert set(stats) == {"builder_initial", "reviewer_feature"}
    assert stats["reviewer_feature"]["max"] == 90.0


def test_role_latency_ignores_events_without_a_duration():
    events = [_event(duration=10),
              {"ts": None, "level": "INFO", "message": "llm_exchange",
               "data": {"model_role": "ghost"}}]
    stats = role_latency(events)
    assert "ghost" not in stats
    assert stats["builder_initial"]["count"] == 1


def test_role_latency_skips_non_llm_messages():
    events = [_event(message="task_started", duration=5)]
    assert role_latency(events) == {}


def test_role_latency_on_empty_is_empty_dict():
    assert role_latency([]) == {}


def test_role_latency_keys_are_sorted():
    events = [_event(role="zeta", duration=1), _event(role="alpha", duration=1),
              _event(role="mid", duration=1)]
    assert list(role_latency(events)) == ["alpha", "mid", "zeta"]


# ----------------------------------------------------------------- outliers
def test_outliers_uses_the_iqr_rule():
    # q1=11 q3=13 iqr=2 -> threshold 16.0; 300 is the only value above it.
    events = [_event(duration=d) for d in (10, 12, 11, 13, 12, 300)]
    found = outliers(events)
    assert len(found) == 1
    assert found[0]["duration_s"] == 300.0
    assert found[0]["model_role"] == "builder_initial"


def test_outliers_are_empty_for_a_tight_distribution():
    events = [_event(duration=d) for d in (10, 11, 12, 11, 10)]
    assert outliers(events) == []


def test_outliers_need_at_least_four_samples():
    """IQR is meaningless on 3 points; the implementation must not invent one."""
    events = [_event(duration=d) for d in (1, 2, 1000)]
    assert outliers(events) == []


def test_outliers_on_empty_is_empty_list():
    assert outliers([]) == []


def test_outliers_factor_is_honoured():
    events = [_event(duration=d) for d in (10, 12, 11, 13, 30)]
    assert outliers(events, factor=1000.0) == []
    assert outliers(events, factor=0.0)


def test_outlier_entries_expose_role_duration_and_threshold():
    events = [_event(duration=d) for d in (10, 12, 11, 13, 12, 300)]
    entry = outliers(events)[0]
    assert set(entry) == {"model_role", "duration_s", "threshold"}
    assert entry["threshold"] == 16.0  # q3 13 + 1.5 * iqr 2


def test_outliers_are_sorted_by_duration_descending():
    events = [_event(duration=d) for d in (10, 11, 500, 900)]
    found = outliers(events)
    durations = [f["duration_s"] for f in found]
    assert durations == sorted(durations, reverse=True)