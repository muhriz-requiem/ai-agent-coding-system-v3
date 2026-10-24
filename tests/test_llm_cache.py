"""Exact-match LLM response cache behaviour."""

from __future__ import annotations

import config
import llm_cache


def test_roundtrip_and_key_separation(tmp_path):
    llm_cache.put("planner", "groq", "m1", "sys", "hello", "resp-1",
                  workspace_root=tmp_path)
    assert llm_cache.get("planner", "groq", "m1", "sys", "hello",
                         workspace_root=tmp_path) == "resp-1"
    # Different provider/model/prompt -> separate key, no cross-talk.
    assert llm_cache.get("planner", "groq", "m2", "sys", "hello",
                         workspace_root=tmp_path) is None
    assert llm_cache.get("planner", "gemini", "m1", "sys", "hello",
                         workspace_root=tmp_path) is None


def test_ttl_expiry(tmp_path, monkeypatch):
    llm_cache.put("planner", "groq", "m1", "sys", "hello", "resp",
                  workspace_root=tmp_path)
    monkeypatch.setattr(config, "LLM_CACHE_TTL_H", 0)
    assert llm_cache.get("planner", "groq", "m1", "sys", "hello",
                         workspace_root=tmp_path) is None


def test_disabled_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)
    llm_cache.put("planner", "groq", "m1", "sys", "hello", "resp",
                  workspace_root=tmp_path)
    assert llm_cache.get("planner", "groq", "m1", "sys", "hello",
                         workspace_root=tmp_path) is None


def test_empty_response_never_cached(tmp_path):
    llm_cache.put("planner", "groq", "m1", "sys", "hello", "   ",
                  workspace_root=tmp_path)
    assert llm_cache.get("planner", "groq", "m1", "sys", "hello",
                         workspace_root=tmp_path) is None


def test_clear_and_stats(tmp_path):
    llm_cache.put("planner", "groq", "m1", "sys", "a", "resp",
                  workspace_root=tmp_path)
    llm_cache.put("builder", "groq", "m1", "sys", "b", "resp",
                  workspace_root=tmp_path)
    assert llm_cache.stats(tmp_path)["entries"] == 2
    assert llm_cache.clear(tmp_path) == 2
    assert llm_cache.stats(tmp_path)["entries"] == 0
