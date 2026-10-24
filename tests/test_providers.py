"""Provider registry: credential gating, enable/disable precedence."""

from __future__ import annotations

import providers


def test_no_keys_means_no_enabled_providers(clean_env):
    assert providers.enabled_providers() == []
    assert not providers.has_credentials(
        providers.provider_by_key("groq"))
    assert not providers.has_credentials(
        providers.provider_by_key("local"))


def test_single_key_enables_only_that_provider(clean_env):
    clean_env.setenv("GROQ_API_KEY", "test-key")
    enabled = [p.key for p in providers.enabled_providers()]
    assert enabled == ["groq"]


def test_default_disabled_requires_explicit_enable(clean_env):
    clean_env.setenv("CEREBRAS_API_KEY", "test-key")
    assert "cerebras" not in [p.key for p in providers.enabled_providers()]
    clean_env.setenv("PROVIDERS_ENABLED", "cerebras")
    assert "cerebras" in [p.key for p in providers.enabled_providers()]


def test_explicit_enable_beats_default_disabled(clean_env):
    clean_env.setenv("LOCAL_LLM_ENABLED", "true")
    clean_env.setenv("PROVIDERS_ENABLED", "local")
    assert "local" in [p.key for p in providers.enabled_providers()]


def test_disable_wins_over_key(clean_env):
    clean_env.setenv("GROQ_API_KEY", "test-key")
    clean_env.setenv("PROVIDERS_DISABLED", "groq")
    assert "groq" not in [p.key for p in providers.enabled_providers()]


def test_cloudflare_requires_account_id(clean_env):
    cloudflare = providers.provider_by_key("cloudflare")
    clean_env.setenv("CLOUDFLARE_API_TOKEN", "token")
    assert not providers.has_credentials(cloudflare)
    clean_env.setenv("CLOUDFLARE_ACCOUNT_ID", "acct-123")
    assert providers.has_credentials(cloudflare)
    assert "acct-123" in providers.resolve_base_url(cloudflare)


def test_env_quota_override(clean_env):
    clean_env.setenv("GROQ_RPD", "77")
    groq = providers.provider_by_key("groq")
    assert groq.rpd == 77


def test_priority_ordering(clean_env):
    clean_env.setenv("GROQ_API_KEY", "a")
    clean_env.setenv("GEMINI_API_KEY", "b")
    enabled = [p.key for p in providers.enabled_providers()]
    assert enabled.index("gemini") < enabled.index("groq")
