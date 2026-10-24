"""Tests for src/triage/cost.py -- token and prompt accounting.

Note the asymmetry this module must handle: ``llm_usage`` carries real SDK
numbers, while ``llm_exchange`` only carries *truncated* heads. Mixing them
silently would understate cost, so the two are reported separately.
"""

import pytest


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.triage.cost") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.triage.cost import (exchange_sizes,  # noqa: E402
                             role_tokens, token_totals)


def _usage(role="builder_initial", model="m", prompt=100, completion=50,
           total=150):
    return {"ts": "2026-10-04T12:00:00.000+00:00", "level": "INFO",
            "message": "llm_usage",
            "data": {"model_role": role, "model": model,
                     "prompt_tokens": prompt, "completion_tokens": completion,
                     "total_tokens": total}}


def _exchange(role="builder_initial", prompt_head="x" * 300,
              response_head="y" * 500):
    return {"ts": "2026-10-04T12:00:00.000+00:00", "level": "INFO",
            "message": "llm_exchange",
            "data": {"model_role": role, "model": "m", "duration_s": 5.0,
                     "prompt_head": prompt_head,
                     "response_head": response_head}}


# ------------------------------------------------------------ token_totals
def test_token_totals_sums_all_three_columns():
    events = [_usage(prompt=100, completion=50, total=150),
              _usage(prompt=200, completion=20, total=220)]
    totals = token_totals(events)
    assert totals["prompt"] == 300
    assert totals["completion"] == 70
    assert totals["total"] == 370
    assert totals["records"] == 2


def test_token_totals_derives_total_when_absent():
    events = [_usage(prompt=10, completion=5, total=0)]
    assert token_totals(events)["total"] == 15


def test_token_totals_on_empty_is_all_zero():
    assert token_totals([]) == {"prompt": 0, "completion": 0, "total": 0,
                                "records": 0}


def test_token_totals_ignores_events_without_token_data():
    assert token_totals([_exchange()])["records"] == 0


def test_token_totals_ignores_negative_values():
    events = [_usage(prompt=-5, completion=-5, total=-10)]
    assert token_totals(events)["total"] == 0


# ------------------------------------------------------------ role_tokens
def test_role_tokens_groups_by_model_role():
    events = [_usage(role="builder_initial", total=150),
              _usage(role="builder_initial", total=50),
              _usage(role="reviewer_feature", total=20)]
    per_role = role_tokens(events)
    assert per_role == {"builder_initial": 200, "reviewer_feature": 20}


def test_role_tokens_keys_are_sorted():
    events = [_usage(role="zeta"), _usage(role="alpha")]
    assert list(role_tokens(events)) == ["alpha", "zeta"]


def test_role_tokens_on_empty_is_empty_dict():
    assert role_tokens([]) == {}


# ---------------------------------------------------------- exchange_sizes
def test_exchange_sizes_reports_max_and_mean_heads():
    events = [_exchange(prompt_head="a" * 100, response_head="b" * 200),
              _exchange(prompt_head="c" * 300, response_head="d" * 400)]
    sizes = exchange_sizes(events)
    assert sizes["max_prompt_head"] == 300
    assert sizes["max_response_head"] == 400
    assert sizes["mean_prompt_head"] == pytest.approx(200.0)
    assert sizes["count"] == 2


def test_exchange_sizes_mean_is_float_even_on_exact_division():
    events = [_exchange(prompt_head="a" * 10), _exchange(prompt_head="b" * 20)]
    assert exchange_sizes(events)["mean_prompt_head"] == pytest.approx(15.0)


def test_exchange_sizes_on_empty_is_zeroed():
    assert exchange_sizes([]) == {"max_prompt_head": 0, "max_response_head": 0,
                                  "mean_prompt_head": 0.0, "count": 0}


def test_exchange_sizes_ignores_missing_heads():
    event = _exchange()
    event["data"].pop("prompt_head")
    assert exchange_sizes([event])["max_prompt_head"] == 0


def test_exchange_sizes_entry_keys_are_exact():
    assert set(exchange_sizes([_exchange()])) == {
        "max_prompt_head", "max_response_head", "mean_prompt_head", "count"}