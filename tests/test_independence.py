"""v2.9: auditing roles must not run on the Builder's weights.

The defect this fixes is a specific one, and it is worth naming because it is
the kind that never shows up as a failure.  ``_reviewer_exclusions`` decided
membership with ``role.startswith("reviewer")``, so when v2.9 added two more
roles that judge Builder output -- ``spec_critic`` (audits the objective against
the tests that will grade the Builder) and ``diagnosis`` (judges Builder output
after the fact) -- both were silently opted out.  Nothing errored.  The Spec
Critic simply agreed with whatever family the Builder used, which is the exact
defect the exclusion was written to prevent.

Exact-id exclusion is also insufficient on its own: exclude
``anthropic/claude-sonnet-5`` and the router still offers
``anthropic/claude-opus-4``, the same opinion from the same lineage.  So family
matching is part of this, and it has one subtlety that a naive implementation
gets wrong in both directions -- see :func:`model_router.model_family`.

These tests are about routing and derivation, so they run against the real
provider roster rather than a hand-built fixture.  A fixture would not have
caught the ``@cf/`` routing prefix or the aggregator case, both of which only
show up against actual model ids.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
import model_router
import providers
from agent_roles import AUDITING_ROLES, LLMClient, _BUILDER_MODELS


@pytest.fixture
def builder_model(monkeypatch):
    """Pin the recorded Builder weights and the exclusion toggle."""
    monkeypatch.setattr(config, "REVIEWER_MODEL_EXCLUSION", True)
    _BUILDER_MODELS.clear()
    _BUILDER_MODELS["builder"] = "anthropic/claude-sonnet-5"
    yield "anthropic/claude-sonnet-5"
    _BUILDER_MODELS.clear()


@pytest.fixture
def routing_workspace(tmp_path, monkeypatch):
    """A workspace with an empty health registry and all providers routable."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".system" / "model_health.json").write_text(
        json.dumps({"entries": {}}), encoding="utf-8")
    # Credential checks are blanked suite-wide by tests/conftest.py, which is
    # correct (a test must not spend quota) and means routing has to be opted
    # into explicitly here.
    monkeypatch.setattr(providers, "enabled_providers",
                        lambda: list(providers.all_provider_defs()))
    monkeypatch.setattr(model_router.providers, "enabled_providers",
                        lambda: list(providers.all_provider_defs()))
    return tmp_path


# ============================================== which roles must be excluded
@pytest.mark.parametrize("role", sorted(AUDITING_ROLES))
def test_every_auditing_role_excludes_the_builder_model(role, builder_model):
    assert LLMClient._reviewer_exclusions(role) == [builder_model]


@pytest.mark.parametrize("role", ["spec_author", "planner", "judge",
                                  "merger", "campaign", "builder_initial", ""])
def test_non_auditing_roles_keep_the_whole_pool(role, builder_model):
    """Excluding too much is the other failure mode, and it starves the pool."""
    assert LLMClient._reviewer_exclusions(role) == []
    assert LLMClient._reviewer_family_exclusions(role) == []


def test_auditing_roles_are_an_explicit_set_not_a_prefix_test():
    """A prefix test is what let two v2.9 roles slip through."""
    assert isinstance(AUDITING_ROLES, frozenset)
    assert "spec_critic" in AUDITING_ROLES
    assert "diagnosis" in AUDITING_ROLES


def test_exclusion_respects_the_kill_switch(builder_model, monkeypatch):
    monkeypatch.setattr(config, "REVIEWER_MODEL_EXCLUSION", False)
    for role in AUDITING_ROLES:
        assert LLMClient._reviewer_exclusions(role) == []
        assert LLMClient._reviewer_family_exclusions(role) == []


# ================================================= family derivation, real ids
@pytest.mark.parametrize("provider,model_id,expected,why", [
    ("groq", "qwen/qwen3.8-27b", "qwen", "vendor is the first path segment"),
    ("groq", "openai/gpt-oss-120b", "openai", "vendor is the first path segment"),
    # The two cases a naive split gets wrong, both taken from the live roster.
    ("nvidia", "z-ai/glm-5.3", "z-ai",
     "hyphenated vendor: must not collapse to 'z'"),
    ("cloudflare", "@cf/openai/gpt-oss-120b", "openai",
     "'@cf' is a routing prefix, not a vendor"),
    ("cloudflare", "@cf/moonshotai/kimi-k2.6", "moonshotai",
     "'@cf' stripped before reading the vendor"),
    # OpenRouter resells five different vendors; its provider key says nothing.
    ("openrouter", "google/gemma-4-31b-it", "google",
     "aggregator: family comes from the model id, not the provider"),
    ("openrouter", "qwen/qwen3.8-27b:free", "qwen",
     "aggregator with a nested path and a ':free' suffix"),
    ("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", "nvidia",
     "aggregator serving a third party"),
    ("openrouter", "dots-studio/dots-3-note-preview:free", "dots-studio",
     "aggregator, hyphenated vendor"),
    ("gemini", "gemini-3.5-flash-lite", "gemini",
     "no slash: split on the first hyphen"),
    ("zai", "glm-4.7-flash", "glm", "no slash"),
    ("mistral", "codestral-latest", "codestral", "no slash"),
    ("local", "qwen3-coder", "qwen3",
     "qwen3 is a distinct family from qwen"),
    ("cerebras", "qwen-3.8-27b", "qwen", "hyphen split for a slash-less id"),
])
def test_family_derivation(provider, model_id, expected, why):
    assert model_router.model_family(provider, model_id) == expected, why


def test_same_vendor_through_two_providers_is_one_family():
    """The id-swap defence depends on this.

    Excluding ``groq/qwen/...`` must also exclude ``openrouter/qwen/...``: they
    are the same weights behind two different doors, and treating them as
    unrelated families would defeat the whole rule.
    """
    assert (model_router.model_family("groq", "qwen/qwen3.8-27b")
            == model_router.model_family("openrouter", "qwen/qwen3.8-27b:free")
            == "qwen")


def test_aggregator_does_not_conflate_vendors():
    """The inverse: OpenRouter must not be one blob family."""
    qwen = model_router.model_family("openrouter", "qwen/qwen3.8-27b:free")
    google = model_router.model_family("openrouter", "google/gemma-4-31b-it")
    assert qwen != google, (
        "treating every OpenRouter model as one family would exclude the whole "
        "provider whenever the Builder used any of it")


def test_provider_is_not_used_as_the_family():
    """NVIDIA serves Z.AI and DeepSeek; the provider is not the vendor."""
    assert model_router.model_family("nvidia", "z-ai/glm-5.3") == "z-ai"
    assert model_router.model_family(
        "nvidia", "deepseek-ai/deepseek-v4.1-flash") == "deepseek-ai"


def test_unreadable_family_returns_none_rather_than_guessing():
    """None means "do not exclude"; a guess here would empty a provider."""
    assert model_router.model_family("groq", "") is None
    assert model_router.model_family("groq", "   ") is None
    assert model_router.model_family("groq", "///") is None


# ============================================== the filter actually filters
def test_family_exclusion_removes_a_renamed_sibling(routing_workspace,
                                                    monkeypatch):
    """The id-swap defence, end to end through the real ranking.

    Note the two halves are separate on purpose.  ``anthropic/claude-sonnet-5``
    is a *premium* model and is not in the free seed roster at all, so id
    exclusion of it necessarily changes nothing -- which is why asserting "the
    id exclusion empties the pool" would be wrong.  What has to hold is that
    family exclusion removes the whole family, including the ids that are present.
    """
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    everything = model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace)
    assert everything, "precondition: candidates exist"

    # No model is served under this id in the free roster, so excluding it is
    # correctly a no-op rather than an error.
    assert model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace,
        exclude_models=["anthropic/claude-sonnet-5"])

    kept = model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace,
        exclude_families=["anthropic"])
    assert kept, "excluding one family emptied the whole pool"
    survivors = {c.model.model for c in kept}
    for pdef in providers.all_provider_defs():
        for model in pdef.models:
            if model.model in survivors:
                assert model_router.model_family(pdef.key, model.model) != \
                    "anthropic", f"excluded family survived: {model.model}"


def test_id_exclusion_removes_a_model_that_is_actually_routable(
        routing_workspace, monkeypatch):
    """The id half of the rule, against an id that really is in the roster."""
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    baseline = model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace)
    assert baseline, "precondition: candidates exist"
    victim = baseline[0].model.model

    kept = model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace,
        exclude_models=[victim])
    assert victim not in {c.model.model for c in kept}
    assert kept, "excluding one id emptied the pool"


def test_unknown_family_does_not_exclude_everything(routing_workspace,
                                                    monkeypatch):
    """A parse failure must never silently empty the ranking."""
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    baseline = model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace)
    with_junk = model_router.rank_candidates(
        "reviewer_feature", limit=500, workspace_root=routing_workspace,
        exclude_families=["definitely-not-a-real-family"])
    assert {c.model.model for c in with_junk} == \
        {c.model.model for c in baseline}


def test_exclusion_filters_realistically_rather_than_starving(routing_workspace,
                                                              monkeypatch):
    """Both failure modes at once: no-op exclusion, or an emptied pool.

    Measured against the real roster, excluding a Builder that used three
    families leaves 30 of 31 reviewer-capable models.  The bounds are loose on
    purpose -- the point is that the rule is *doing something* without starving
    a free tier where one healthy family is often all there is.
    """
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    _BUILDER_MODELS.clear()
    _BUILDER_MODELS["builder_initial"] = "anthropic/claude-sonnet-5"
    _BUILDER_MODELS["builder_patch"] = "deepseek/deepseek-chat-v3.1"
    _BUILDER_MODELS["builder_escalate"] = "z-ai/glm-5.3"
    try:
        excluded = set(LLMClient._reviewer_exclusions("spec_critic"))
        families = set(LLMClient._reviewer_family_exclusions("spec_critic"))
        assert len(families) >= 3, families
        baseline = model_router.rank_candidates(
            "spec_critic", limit=500, workspace_root=routing_workspace)
        filtered = model_router.rank_candidates(
            "spec_critic", limit=500, workspace_root=routing_workspace,
            exclude_models=excluded, exclude_families=families)

        assert baseline, "precondition: candidates exist"
        assert filtered, "three-family exclusion emptied the pool"
        assert len(filtered) < len(baseline), (
            "exclusion was a no-op; it filtered nothing at all")
        assert len(filtered) > len(baseline) * 0.5, (
            f"exclusion was too broad: kept {len(filtered)} of {len(baseline)}. "
            f"On a free tier this starves the pool and turns collapses into "
            f"stalls.")
    finally:
        _BUILDER_MODELS.clear()


# ================================================= collapse reporting
def test_exclusion_is_a_harmless_noop_for_a_premium_builder(routing_workspace,
                                                            monkeypatch):
    """A measured property worth pinning, because it looks like a bug.

    ``builder_initial`` defaults to ``anthropic/claude-sonnet-5``, and **no
    anthropic model exists in the free seed roster** -- so excluding the
    anthropic family removes nothing and the pool is unchanged.

    That is correct rather than broken: there is nothing anthropic to exclude.
    It does mean the rule is inert until the Builder actually runs on a free
    model, which is what happens under ``FREE_MODEL_ROUTING=true`` (the default)
    where routing sends the Builder to whichever flash model is healthy.
    """
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    baseline = model_router.rank_candidates(
        "spec_critic", limit=999, workspace_root=routing_workspace)
    anthropic = [m for p in providers.all_provider_defs() for m in p.models
                 if model_router.model_family(p.key, m.model) == "anthropic"]
    assert not anthropic, (
        "the free roster now contains an anthropic model; this test's premise "
        "no longer holds and the no-op needs re-measuring")

    kept = model_router.rank_candidates(
        "spec_critic", limit=999, workspace_root=routing_workspace,
        exclude_families=["anthropic"])
    assert len(kept) == len(baseline)


def test_exclusion_bites_when_the_builder_runs_a_free_model(routing_workspace,
                                                            monkeypatch):
    """The case that actually matters, measured rather than assumed."""
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    baseline = len(model_router.rank_candidates(
        "spec_critic", limit=999, workspace_root=routing_workspace))
    assert baseline, "precondition: candidates exist"
    try:
        for builder in ("z-ai/glm-5.3", "gemini-3.5-flash-lite",
                        "qwen/qwen3.8-27b"):
            _BUILDER_MODELS.clear()
            _BUILDER_MODELS["builder"] = builder
            kept = len(model_router.rank_candidates(
                "spec_critic", limit=999, workspace_root=routing_workspace,
                exclude_models=LLMClient._reviewer_exclusions("spec_critic"),
                exclude_families=LLMClient._reviewer_family_exclusions(
                    "spec_critic")))
            assert kept < baseline, (
                f"excluding the family of {builder} filtered nothing")
            assert kept > baseline * 0.5, (
                f"excluding {builder}'s family was too broad: {kept}/{baseline}")
    finally:
        _BUILDER_MODELS.clear()


def test_client_records_independence_collapse():
    """A degraded audit has to say so; the flag is how the agent knows."""
    llm = LLMClient(Path(__file__).resolve().parents[1])
    assert llm.last_independence_collapsed is False


def test_documented_role_table_has_no_auditing_collision():
    """Under non-free routing the table is live, so collisions are real.

    Free-tier routing overrides the table entirely (which is why the exclusion
    rule exists at all), but ``FREE_MODEL_ROUTING=false`` is a supported mode
    and the table must not contradict itself there.
    """
    routing = config.DEFAULT_MODEL_ROUTING
    builder_family = model_router.model_family(
        "", routing["builder_initial"])
    assert builder_family
    for role in AUDITING_ROLES:
        configured = routing.get(role)
        if not configured:
            continue
        family = model_router.model_family("", configured)
        assert family != builder_family, (
            f"{role} is configured on the same family as builder_initial "
            f"({configured}); under FREE_MODEL_ROUTING=false this audit would "
            f"be an echo")
