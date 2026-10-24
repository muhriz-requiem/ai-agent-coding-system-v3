"""Model router: capability filters, health ranking, cooldown exclusion."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import config
import model_router
import providers
import pytest
from quota_governor import get_governor


def _enable(clean_env, *keys: str):
    clean_env.setenv("FREE_MODEL_ROUTING", "true")
    for key in keys:
        clean_env.setenv(f"{key.upper()}_API_KEY", "test-key")
        if key == "cloudflare":
            clean_env.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    return clean_env


def test_json_roles_require_structured_output(clean_env, tmp_path):
    _enable(clean_env, "groq")
    # Every groq seed supports response_format -> json candidates exist.
    ranked = model_router.rank_candidates("planner", workspace_root=tmp_path)
    assert ranked
    assert {c.pdef.key for c in ranked} == {"groq"}
    assert all(c.model.response_format or c.model.structured_outputs
               for c in ranked)


def test_longform_excludes_low_tpm_provider(clean_env, tmp_path):
    _enable(clean_env, "groq", "gemini")
    ranked = model_router.rank_candidates("builder", workspace_root=tmp_path)
    assert ranked
    keys = {c.pdef.key for c in ranked}
    # Groq's 8k TPM disqualifies it for long builder generations.
    assert "groq" not in keys
    assert "gemini" in keys


def test_disabled_free_routing_returns_empty(clean_env, tmp_path, monkeypatch):
    _enable(clean_env, "groq")
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", False)
    assert model_router.rank_candidates("planner",
                                        workspace_root=tmp_path) == []


def test_cooldown_excludes_provider(clean_env, tmp_path):
    _enable(clean_env, "groq")
    governor = get_governor(tmp_path)
    governor.set_cooldown("groq", 120, reason="test")
    assert model_router.rank_candidates("planner",
                                        workspace_root=tmp_path) == []


def test_health_degrades_ranking(clean_env, tmp_path):
    """A model that mostly works loses rank but stays in rotation.

    The ratio is chosen so the outcome lands between "untried" (health 0.60)
    and the quarantine line (success <= 0.10): degraded enough to lose first
    place, not so degraded that it is thrown out of the pool.
    """
    _enable(clean_env, "groq")
    before = model_router.rank_candidates("planner", workspace_root=tmp_path)
    top = before[0]
    for _ in range(10):
        model_router.record_call(top.pdef.key, top.model.model, "planner",
                                 "ok", workspace_root=tmp_path)
    for _ in range(10):
        model_router.record_call(top.pdef.key, top.model.model, "planner",
                                 "error", workspace_root=tmp_path)
    after = model_router.rank_candidates("planner", workspace_root=tmp_path)
    assert after
    failed_score = next(c.score for c in after
                        if c.model.model == top.model.model)
    assert failed_score < top.score
    assert not model_router.is_quarantined(top.pdef.key, top.model.model,
                                           workspace_root=tmp_path)
    assert after[0].model.model != top.model.model


def test_never_responded_model_leaves_rotation(clean_env, tmp_path):
    """0 successes is a fact at a low sample size, not a noisy rate (v2.9.1).

    The 20-call quarantine bar exists because a *rate* over few observations is
    noise.  "Has never once answered" is not a rate estimate, so it acts at
    ``QUARANTINE_MIN_CALLS_ZERO_OK``.  This matters operationally: two nvidia
    models each held an authoring attempt for the full 360 s call budget on
    repeat, because a timeout counts as a transient failure and 2 < 20.
    """
    _enable(clean_env, "groq")
    before = model_router.rank_candidates("planner", workspace_root=tmp_path)
    top = before[0]
    for _ in range(model_router.QUARANTINE_MIN_CALLS_ZERO_OK):
        model_router.record_call(top.pdef.key, top.model.model, "planner",
                                 "timeout", workspace_root=tmp_path)
    assert model_router.is_quarantined(top.pdef.key, top.model.model,
                                       workspace_root=tmp_path)
    after = model_router.rank_candidates("planner", workspace_root=tmp_path)
    assert all(c.model.model != top.model.model for c in after)


def test_latency_score_is_not_a_reward_for_silence(clean_env, tmp_path):
    """An unmeasured latency must be neutral, never perfect (v2.9.1).

    ``latency_ewma`` starts at 0.0 and is only written when a call returns, so
    the old ``latency <= 0`` clause scored a model that had never responded as
    the fastest in the pool -- on 15% of its health.
    """
    never_answered = {"calls": 3, "ok": 0, "latency_ewma": 0.0}
    assert model_router._latency_score(never_answered) < 0.5
    fast = {"calls": 3, "ok": 3, "latency_ewma": 4.0}
    assert model_router._latency_score(fast) == 1.0
    untried = {"calls": 0, "ok": 0, "latency_ewma": 0.0}
    assert model_router._latency_score(untried) == 0.5


def test_record_outcome_and_parse(clean_env, tmp_path):
    usage = {"provider": "groq", "model": "openai/gpt-oss-120b",
             "role": "planner"}
    model_router.record_parse(False, usage, workspace_root=tmp_path)
    model_router.record_outcome(usage, True, workspace_root=tmp_path)
    health = model_router._load_json(
        model_router._health_path(tmp_path), {})
    entry = health["entries"]["groq|openai/gpt-oss-120b|planner"]
    assert entry["parse_fail"] == 1
    assert entry["outcomes_ok"] == 1


def test_next_reset_iso_is_future(tmp_path, clean_env):
    _enable(clean_env, "groq")
    reset = model_router.next_reset_iso(workspace_root=tmp_path)
    assert datetime.fromisoformat(reset) > datetime.now(timezone.utc)


def test_catalog_conversion_filters_nonzero_pricing():
    entries = [
        {"id": "free/model:free", "pricing": {"prompt": "0", "completion": "0"},
         "context_length": 100000, "supported_parameters": ["structured_outputs"],
         "top_provider": {"max_completion_tokens": 32000},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "paid/model", "pricing": {"prompt": "0.1", "completion": "0.2"},
         "context_length": 100000,
         "architecture": {"output_modalities": ["text"]}},
        {"id": "audio/model:free", "pricing": {"prompt": "0", "completion": "0"},
         "context_length": 100000,
         "architecture": {"output_modalities": ["audio"]}},
        {"id": "openrouter/free", "pricing": {"prompt": "0", "completion": "0"}},
    ]
    models = model_router._openrouter_models_from_catalog(entries)
    assert [m.model for m in models] == ["free/model:free"]


def test_verify_providers_reports_ok_and_errors(clean_env, tmp_path):
    _enable(clean_env, "groq")

    class _FakeLLM:
        async def raw_complete(self, provider, model, system, user):
            if provider == "groq":
                return "OK"
            raise RuntimeError("boom")

    results = asyncio.run(
        model_router.verify_providers(_FakeLLM(), tmp_path))
    assert len(results) == 1
    assert results[0]["provider"] == "groq" and results[0]["ok"]


# ============== A probe is the designated cleanup path, so it must clean up
def test_probe_takes_a_withdrawn_model_out_of_rotation(clean_env, tmp_path,
                                                      monkeypatch):
    """Found on the first live probe run.

    ``cloudflare:@cf/moonshotai/kimi-k2.6`` answered HTTP 403 "is not available
    on the Workers Free plan".  That is an availability fact about the model,
    not a transient error -- but ``raw_complete`` (the probe path) recorded it
    as a generic failure and left the model in the catalog.  So the one command
    an operator runs to clean the roster up did not clean it.
    """
    from agent_roles import LLMClient

    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    monkeypatch.setattr(config, "MODEL_UNAVAILABLE_COOLDOWN_S", 900)
    client = LLMClient(tmp_path)

    class Gone(Exception):
        status_code = 403

    async def boom(*a, **k):
        raise Gone("Model @cf/moonshotai/kimi-k2.6 is not available on the "
                   "Workers Free plan")

    monkeypatch.setattr(LLMClient, "_call_provider", boom)
    monkeypatch.setattr(model_router.providers, "provider_by_key",
                        lambda key: providers.ProviderDef(key="cloudflare", label="cf", base_url="u", api_key_env="X", models=(), rpm=30, priority=25))
    monkeypatch.setattr(model_router.providers, "has_credentials",
                        lambda p: True)

    with pytest.raises(Gone):
        asyncio.run(client.raw_complete("cloudflare",
                                        "@cf/moonshotai/kimi-k2.6",
                                        "s", "u"))
    assert model_router.unavailable_models(tmp_path), (
        "a probe must take a withdrawn model out of rotation")


def test_probe_does_not_mark_a_rate_limit_unavailable(clean_env, tmp_path,
                                                      monkeypatch):
    """429 is congestion. Marking it dead would empty the pool under load."""
    from agent_roles import LLMClient

    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    client = LLMClient(tmp_path)

    class Busy(Exception):
        status_code = 429

    async def boom(*a, **k):
        raise Busy("rate limit exceeded")

    monkeypatch.setattr(LLMClient, "_call_provider", boom)
    monkeypatch.setattr(model_router.providers, "provider_by_key",
                        lambda key: providers.ProviderDef(key="groq", label="g", base_url="u", api_key_env="X", models=(), rpm=30, priority=20))
    monkeypatch.setattr(model_router.providers, "has_credentials",
                        lambda p: True)

    with pytest.raises(Busy):
        asyncio.run(client.raw_complete("groq", "some/model", "s", "u"))
    assert not model_router.unavailable_models(tmp_path), (
        "a 429 must never be recorded as unavailability")

def test_refresh_returns_a_real_count(clean_env, tmp_path):
    """``--models refresh`` used to print "0 free models" after adding 610.

    ``refresh_catalog`` never set a ``count`` key and the CLI read
    ``result.get("count", 0)``, so the one command whose entire job is to report
    what changed reported a confident zero on every run -- while the real diff
    (+610 / -7) went only to the JSONL log.  Pinned because the value is now
    what an operator reads before trusting their roster.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setattr(model_router.providers, "all_provider_defs", list)
    monkey.setattr(model_router.providers, "has_credentials", lambda p: False)
    result = model_router.refresh_catalog(workspace_root=tmp_path)
    assert "count" in result, "the CLI reads result['count']; it must exist"
    assert isinstance(result["count"], int)
    assert "added_count" in result and "removed_count" in result
    assert result["added_count"] == len(result["added"])
    assert result["removed_count"] == len(result["removed"])
    monkey.undo()


def test_refresh_count_matches_the_per_provider_counts(clean_env, tmp_path):
    monkey = pytest.MonkeyPatch()

    def one_provider():
        return [providers.ProviderDef(
            key="groq", label="g", base_url="u", api_key_env="X",
            models=(), rpm=30, priority=20,
            supports_models_endpoint=True)]

    monkey.setattr(model_router.providers, "all_provider_defs", one_provider)
    monkey.setattr(model_router.providers, "has_credentials", lambda p: True)
    monkey.setattr(model_router, "_models_url",
                   lambda p: "https://example.invalid/v1/models")

    class M:
        def __init__(self, name):
            self.model = name

    monkey.setattr(model_router, "_fetch_models",
                   lambda p, t: [{"id": "a/model"}, {"id": "b/model"}])
    monkey.setattr(model_router, "_models_from_generic_list",
                   lambda p, entries: [M(e["id"]) for e in entries])
    result = model_router.refresh_catalog(workspace_root=tmp_path)
    assert result["ok"] is True
    assert result["count"] == 2, result
    assert result["providers"]["groq"]["count"] == 2
    monkey.undo()