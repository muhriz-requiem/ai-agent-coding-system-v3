"""LLMClient free-routing: failover, cache, 429 cooldown, deferral helper."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

import config
import model_router
from agent_roles import LLMClient
from project_orchestrator import ProjectOrchestrator
from providers import LLMUnavailableError
from quota_governor import get_governor
from validation import TaskSpec


def _planner_task(**kwargs) -> TaskSpec:
    base = {"id": "T1", "title": "t", "objective": "o", "target_files": []}
    base.update(kwargs)
    return TaskSpec(**base)


class _FakeUsage:
    """Stand-in for the SDK's response.usage object."""

    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


def test_free_complete_success_health_and_cache(clean_env, tmp_path, monkeypatch):
    clean_env.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    calls = {"n": 0}

    async def fake_call(self, pdef, model, system, user, tools):
        calls["n"] += 1
        # v2.8: _call_provider returns (text, usage) so real token accounting
        # is possible.  This fake reports usage so the logging path is covered.
        return '{"ok": true}', _FakeUsage(11, 7, 18)

    monkeypatch.setattr(LLMClient, "_call_provider", fake_call)
    llm = LLMClient(tmp_path)
    first = asyncio.run(llm.complete("sys", "user", model_role="planner"))
    assert first == '{"ok": true}'
    assert llm.last_usage and llm.last_usage["provider"] == "groq"

    # Exact-match cache: the second identical call spends no provider call.
    second = asyncio.run(llm.complete("sys", "user", model_role="planner"))
    assert second == first
    assert calls["n"] == 1

    # Parse recording lands in the health registry.
    llm.record_parse(True)
    health = (tmp_path / ".system" / "model_health.json").read_text("utf-8")
    assert "groq" in health and "parses" in health


def test_unavailable_raises_without_candidates(clean_env, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    monkeypatch.setattr(config, "FREE_MODELS_ALLOW_PAID_FALLBACK", False)
    llm = LLMClient(tmp_path)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(llm.complete("sys", "user", model_role="planner"))


def test_429_sets_cooldown_and_defers(clean_env, tmp_path, monkeypatch):
    clean_env.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(config, "DEMO_MODE", False)
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    monkeypatch.setattr(config, "FREE_MODELS_ALLOW_PAID_FALLBACK", False)

    class _Response:
        status_code = 429
        headers = {"retry-after": "30"}

    class _RateLimited(Exception):
        status_code = 429
        response = _Response()

    async def fake_call(self, pdef, model, system, user, tools):
        raise _RateLimited("rate limited")

    monkeypatch.setattr(LLMClient, "_call_provider", fake_call)
    llm = LLMClient(tmp_path)
    with pytest.raises(LLMUnavailableError):
        asyncio.run(llm.complete("sys", "user", model_role="planner"))
    # Every groq planner model was 429'd -> each is on a model-level
    # cooldown, so the router has no candidate left for the role.
    governor = get_governor(tmp_path)
    cooldowns = governor.ledger["providers"]["groq"]["cooldowns"]
    assert cooldowns and all(until > time.time() for until in cooldowns.values())
    assert model_router.rank_candidates(
        "planner", workspace_root=tmp_path) == []


def test_deferred_helper_window():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert ProjectOrchestrator._deferred(_planner_task(defer_until=future))
    assert not ProjectOrchestrator._deferred(_planner_task(defer_until=past))
    assert not ProjectOrchestrator._deferred(_planner_task(defer_until=None))
    assert not ProjectOrchestrator._deferred(_planner_task(defer_until="junk"))
