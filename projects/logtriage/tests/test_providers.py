"""Tests for src/triage/providers.py -- reliability and fallback behaviour."""

import pytest


def _impl_ready() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("src.triage.providers") is not None
    except Exception:
        return False


if not _impl_ready():
    import pytest

    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.triage.providers import (cooldown_events,  # noqa: E402
                                  fallback_rate, provider_stats)


def _event(message, data):
    return {"ts": "2026-10-04T12:00:00.000+00:00", "level": "INFO",
            "message": message, "data": data}


def _ok(provider="groq", model="qwen", duration=12.0):
    return _event("llm_exchange", {"model": f"{provider}:{model}",
                                   "duration_s": duration})


def _failed(provider="groq", model="qwen", error="rate limited"):
    return _event("llm_model_failed_trying_fallback",
                  {"model": f"{provider}:{model}", "error": error})


def _split(model):
    """``provider:model`` -> (provider, model); models may contain colons."""
    provider, _, rest = str(model).partition(":")
    return provider, rest or model


# ---------------------------------------------------------- provider_stats
def test_provider_stats_counts_ok_and_failed_calls():
    events = [_ok(), _ok(), _failed()]
    stats = provider_stats(events)
    assert stats["groq"]["calls"] == 2
    assert stats["groq"]["errors"] == 1
    assert stats["groq"]["success_rate"] == pytest.approx(2 / 3)


def test_provider_stats_separates_timeouts_from_other_errors():
    events = [_ok(), _failed(error="call exceeded 360s wall-clock budget")]
    stats = provider_stats(events)
    assert stats["groq"]["timeouts"] == 1
    assert stats["groq"]["errors"] == 0
    # A timeout is a failed call: the request produced no answer, so a provider
    # that times out half the time is not 100% reliable.  Timeouts therefore
    # count against success_rate.
    assert stats["groq"]["success_rate"] == pytest.approx(0.5)


def test_provider_stats_median_is_over_successes_only():
    events = [_ok(duration=10), _ok(duration=20), _ok(duration=30)]
    assert provider_stats(events)["groq"]["median_s"] == 20.0


def test_provider_stats_keys_are_sorted():
    events = [_ok("zed"), _ok("alpha"), _ok("mid")]
    assert list(provider_stats(events)) == ["alpha", "mid", "zed"]


def test_provider_stats_on_only_failures_has_zero_median():
    assert provider_stats([_failed()])["groq"]["median_s"] == 0.0


def test_provider_stats_on_empty_is_empty_dict():
    assert provider_stats([]) == {}


def test_provider_stats_entry_keys_are_exact():
    entry = provider_stats([_ok()])["groq"]
    assert set(entry) == {"calls", "errors", "timeouts", "success_rate",
                          "median_s"}


# --------------------------------------------------------- cooldown_events
def test_cooldown_events_returns_provider_reason_and_count():
    events = [
        _event("provider_cooldown", {"provider": "groq", "reason": "http_429"}),
        _event("provider_cooldown", {"provider": "groq", "reason": "http_429"}),
        _event("provider_cooldown", {"provider": "openrouter",
                                     "reason": "http_429"}),
    ]
    cooldowns = cooldown_events(events)
    assert cooldowns == [{"provider": "groq", "reason": "http_429",
                          "count": 2},
                         {"provider": "openrouter", "reason": "http_429",
                          "count": 1}]


def test_cooldown_events_sorts_by_count_then_provider():
    events = [
        _event("provider_cooldown", {"provider": "a", "reason": "r"}),
        _event("provider_cooldown", {"provider": "b", "reason": "r"}),
        _event("provider_cooldown", {"provider": "b", "reason": "r"}),
    ]
    assert [c["provider"] for c in cooldown_events(events)] == ["b", "a"]


def test_cooldown_events_ignores_other_messages():
    assert cooldown_events([_ok()]) == []


def test_cooldown_events_on_empty_is_empty_list():
    assert cooldown_events([]) == []


# ----------------------------------------------------------- fallback_rate
def test_fallback_rate_is_failures_over_total_attempts():
    events = [_ok(), _ok(), _ok(), _failed()]
    assert fallback_rate(events) == pytest.approx(0.25)


def test_fallback_rate_is_zero_when_nothing_fails():
    assert fallback_rate([_ok(), _ok()]) == 0.0


def test_fallback_rate_is_one_when_everything_fails():
    assert fallback_rate([_failed(), _failed()]) == 1.0


def test_fallback_rate_on_empty_is_zero():
    """No evidence is not evidence of failure."""
    assert fallback_rate([]) == 0.0


def test_fallback_rate_ignores_unrelated_events():
    assert fallback_rate([_event("task_started", {"task_id": "T-1"})]) == 0.0


def test_split_handles_models_without_a_provider_prefix():
    assert _split("bare-model") == ("bare-model", "bare-model")
    assert _split("groq:qwen/qwen3.8-27b") == ("groq", "qwen/qwen3.8-27b")
    assert _split("openrouter:qwen/qwen3.8-27b:free") == \
        ("openrouter", "qwen/qwen3.8-27b:free")