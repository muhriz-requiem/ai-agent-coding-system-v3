"""Multi-provider free-inference registry (v2.6 capacity layer).

Every provider below exposes an **OpenAI-compatible Chat Completions API**,
so the existing ``AsyncOpenAI`` machinery can talk to all of them by
swapping ``base_url`` + API key.  Design rules:

* data-driven -- provider definitions are plain frozen dataclasses;
  environment variables enable/disable providers and supply keys,
* offline-safe -- importing this module performs no network I/O,
* fail-open -- a provider without a configured key is simply unavailable,
  never an error.

The registry is consumed by :mod:`model_router` (ranking) and by
``agent_roles.LLMClient`` (dispatch).  ``LLMUnavailableError`` is the
fail-closed signal raised when no provider can serve a call right now;
orchestrators convert it into a quota deferral (never an escalation).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from logger import log_event


class LLMUnavailableError(RuntimeError):
    """No healthy free candidate can serve a call (quota/cooldown/key).

    Distinct from a model failure: the task did not fail, the *capacity*
    ran out.  Orchestrators defer the task until the earliest reset
    instead of escalating it.
    """


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProviderModel:
    """One model offered by a provider (metadata used for role routing)."""

    provider: str
    model: str
    context_length: int = 32768
    max_output: int = 8192
    structured_outputs: bool = False
    response_format: bool = False
    tools: bool = False
    coding_index: Optional[float] = None
    trains_data: bool = False
    free: bool = True
    notes: str = ""


@dataclass(frozen=True)
class ProviderDef:
    """A provider endpoint plus its quota shape and seed models."""

    key: str
    label: str
    base_url: str
    api_key_env: str
    models: Tuple[ProviderModel, ...]
    rpm: int = 30                 # requests/minute (0 = unknown)
    tpm: int = 0                  # tokens/minute (0 = unknown)
    rpd: int = 0                  # requests/day (0 = unknown)
    tpd: int = 0                  # tokens/day (0 = unknown)
    monthly_requests: int = 0     # requests/month (0 = unknown)
    monthly_tokens: int = 0       # tokens/month (0 = unknown)
    reset_tz: str = "UTC"         # UTC | America/Los_Angeles
    priority: int = 50            # lower = preferred on equal scores
    quota_style: str = "daily"    # daily | monthly | trial | local
    supports_models_endpoint: bool = False
    extra_headers: Dict[str, str] = field(default_factory=dict)


def _m(provider: str, model: str, **kwargs) -> ProviderModel:
    return ProviderModel(provider=provider, model=model, **kwargs)


# ---------------------------------------------------------------------------
# Provider catalog -- seeded from the research (Sep/Oct 2026).
# Model lists are refreshed from live `/models` endpoints where available;
# the seeds keep the system functional offline.
# ---------------------------------------------------------------------------
OPENROUTER_MODELS: Tuple[ProviderModel, ...] = (
    _m("openrouter", "qwen/qwen3.8-27b:free", context_length=262144,
       max_output=235929, structured_outputs=True, response_format=True,
       tools=True, coding_index=68.1),
    _m("openrouter", "dots-studio/dots-3-note-preview:free", context_length=512000,
       max_output=460800, structured_outputs=True, response_format=True,
       tools=True),
    _m("openrouter", "apodex/apodex-1.1-mini:free", context_length=262144,
       max_output=235929, structured_outputs=True, response_format=True,
       tools=True),
    _m("openrouter", "nvidia/nemotron-3-super-120b-a12b:free", context_length=262144,
       max_output=235929, structured_outputs=True, response_format=True,
       tools=True, coding_index=37.7),
    _m("openrouter", "google/gemma-4-31b-it:free", context_length=262144,
       max_output=32768, response_format=True, tools=True, coding_index=43.4),
    _m("openrouter", "thinkingmachines/inkling:free", context_length=1048576,
       max_output=262144, tools=True, coding_index=52.1),
    _m("openrouter", "nvidia/nemotron-3-ultra-550b-a55b:free", context_length=1000000,
       max_output=65536, tools=True, coding_index=49.3),
    _m("openrouter", "cohere/north-mini-code:free", context_length=256000,
       max_output=64000, tools=True, coding_index=36.5),
    _m("openrouter", "poolside/laguna-s-2.1:free", context_length=262144,
       max_output=32768, tools=True),
    _m("openrouter", "stealth/space-bunny-alpha", context_length=1000000,
       max_output=524288, tools=True),
    _m("openrouter", "google/gemma-4-26b-a4b-it:free", context_length=262144,
       max_output=32768, response_format=True, tools=True, coding_index=39.3),
    _m("openrouter", "inclusionai/ling-3.1-flash", context_length=262144,
       max_output=32768, tools=True),
)

GROQ_MODELS: Tuple[ProviderModel, ...] = (
    _m("groq", "openai/gpt-oss-120b", context_length=131072, max_output=8192,
       response_format=True, tools=True, coding_index=49.0),
    _m("groq", "openai/gpt-oss-20b", context_length=131072, max_output=8192,
       response_format=True, tools=True),
    _m("groq", "qwen/qwen3.8-27b", context_length=131072, max_output=8192,
       response_format=True, tools=True, coding_index=60.0),
)

GEMINI_MODELS: Tuple[ProviderModel, ...] = (
    _m("gemini", "gemini-3.5-flash-lite", context_length=1048576,
       max_output=65536, structured_outputs=True, response_format=True,
       tools=True, coding_index=40.0),
    _m("gemini", "gemini-3.1-flash-lite", context_length=1048576,
       max_output=65536, structured_outputs=True, response_format=True,
       tools=True),
    _m("gemini", "gemini-3-flash-preview", context_length=1048576,
       max_output=65536, structured_outputs=True, response_format=True,
       tools=True),
)

CLOUDFLARE_MODELS: Tuple[ProviderModel, ...] = (
    _m("cloudflare", "@cf/openai/gpt-oss-120b", context_length=131072,
       max_output=32768, response_format=True, tools=True, coding_index=49.0),
    _m("cloudflare", "@cf/openai/gpt-oss-20b", context_length=131072,
       max_output=32768, response_format=True, tools=True),
    _m("cloudflare", "@cf/moonshotai/kimi-k2.6", context_length=131072,
       max_output=32768, response_format=True, tools=True),
)

ZAI_MODELS: Tuple[ProviderModel, ...] = (
    _m("zai", "glm-4.7-flash", context_length=131072, max_output=16384,
       response_format=True, tools=True, coding_index=55.0),
    _m("zai", "glm-4.5-flash", context_length=131072, max_output=16384,
       response_format=True, tools=True),
)

NVIDIA_MODELS: Tuple[ProviderModel, ...] = (
    # Refreshed against GET integrate.api.nvidia.com/v1/models (Oct 2026):
    # the old gpt-oss-120b/llama-3.3 IDs are end-of-life (410).
    _m("nvidia", "z-ai/glm-5.3", context_length=131072, max_output=32768,
       response_format=True, tools=True, coding_index=55.0),
    _m("nvidia", "z-ai/glm-5.3-flash", context_length=131072, max_output=32768,
       response_format=True, tools=True, coding_index=50.0),
    _m("nvidia", "deepseek-ai/deepseek-v4.1-flash", context_length=131072,
       max_output=32768, response_format=True, tools=True, coding_index=50.0),
    _m("nvidia", "nvidia/nemotron-3-super-120b-a12b", context_length=131072,
       max_output=32768, response_format=True, tools=True, coding_index=40.0),
    _m("nvidia", "nvidia/nemotron-3-ultra-550b-a55b", context_length=131072,
       max_output=32768, tools=True, coding_index=49.0),
    _m("nvidia", "google/gemma-4-31b-it", context_length=131072,
       max_output=32768, response_format=True, tools=True, coding_index=43.4),
    _m("nvidia", "openai/gpt-oss-20b", context_length=131072, max_output=8192,
       response_format=True, tools=True),
)

MISTRAL_MODELS: Tuple[ProviderModel, ...] = (
    _m("mistral", "codestral-latest", context_length=262144, max_output=32768,
       response_format=True, tools=True, coding_index=45.0, trains_data=True),
    _m("mistral", "mistral-small-latest", context_length=131072, max_output=32768,
       response_format=True, tools=True, trains_data=True),
)

COHERE_MODELS: Tuple[ProviderModel, ...] = (
    _m("cohere", "command-a-plus-05-2026", context_length=131072,
       max_output=8192, response_format=True, tools=True),
)

ALIBABA_MODELS: Tuple[ProviderModel, ...] = (
    _m("alibaba", "qwen3-coder-plus", context_length=1000000, max_output=32768,
       response_format=True, tools=True, coding_index=60.0),
    _m("alibaba", "qwen-plus", context_length=1000000, max_output=32768,
       response_format=True, tools=True),
)

CEREBRAS_MODELS: Tuple[ProviderModel, ...] = (
    _m("cerebras", "gpt-oss-120b", context_length=65536, max_output=32768,
       response_format=True, tools=True, coding_index=49.0),
    _m("cerebras", "qwen-3.8-27b", context_length=65536, max_output=32768,
       response_format=True, tools=True),
)

LOCAL_MODELS: Tuple[ProviderModel, ...] = (
    _m("local", os.getenv("LOCAL_LLM_MODEL", "qwen3-coder"), context_length=262144,
       max_output=32768, response_format=True, tools=True),
)

DEFAULT_PROVIDERS: Tuple[ProviderDef, ...] = (
    ProviderDef(
        key="openrouter", label="OpenRouter (free roster)",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY", models=OPENROUTER_MODELS,
        rpm=20, rpd=50, priority=10, supports_models_endpoint=True,
        extra_headers={"HTTP-Referer": "https://localhost",
                       "X-Title": "EnterpriseMultiAgentCodingSystem"},
    ),
    ProviderDef(
        key="groq", label="Groq (free tier)",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY", models=GROQ_MODELS,
        rpm=30, tpm=8000, rpd=1000, tpd=200000, priority=20,
        supports_models_endpoint=True,
    ),
    ProviderDef(
        key="gemini", label="Google Gemini (AI Studio free tier)",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY", models=GEMINI_MODELS,
        rpm=15, tpm=250000, priority=15, reset_tz="America/Los_Angeles",
        supports_models_endpoint=True,
    ),
    ProviderDef(
        key="cloudflare", label="Cloudflare Workers AI (free neurons)",
        base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        api_key_env="CLOUDFLARE_API_TOKEN", models=CLOUDFLARE_MODELS,
        rpm=30, priority=25,
        # 10k neurons/day ~= ~150k output tokens on gpt-oss-120b (approx)
        tpd=200000,
    ),
    ProviderDef(
        key="zai", label="Z.ai (GLM Flash free)",
        base_url="https://api.z.ai/api/paas/v4/",
        api_key_env="ZAI_API_KEY", models=ZAI_MODELS,
        rpm=30, priority=30,
        supports_models_endpoint=True,
    ),
    ProviderDef(
        key="nvidia", label="NVIDIA NIM (free trial)",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY", models=NVIDIA_MODELS,
        rpm=40, priority=35,
        # v2.9: this is the provider whose roster changed underneath us once
        # already (gpt-oss-120b/llama-3.3 went 410 and the harness only found
        # out via a hand-edit). See MODEL_MODELS_URL in model_router.
        supports_models_endpoint=True,
    ),
    ProviderDef(
        key="mistral", label="Mistral (PAID plan required -- opt-in only)",
        base_url="https://api.mistral.ai/v1",
        api_key_env="MISTRAL_API_KEY", models=MISTRAL_MODELS,
        rpm=30, monthly_requests=500, priority=40, quota_style="monthly",
    ),
    ProviderDef(
        key="cohere", label="Cohere (trial key)",
        base_url="https://api.cohere.ai/compatibility/v1",
        api_key_env="COHERE_API_KEY", models=COHERE_MODELS,
        rpm=20, monthly_requests=1000, priority=45, quota_style="monthly",
    ),
    ProviderDef(
        key="alibaba", label="Alibaba Model Studio (region/account info required -- opt-in only)",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY", models=ALIBABA_MODELS,
        rpm=30, monthly_tokens=1000000, priority=50, quota_style="monthly",
    ),
    ProviderDef(
        key="cerebras", label="Cerebras ($5 trial -- card required)",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY", models=CEREBRAS_MODELS,
        rpm=5, tpm=30000, tpd=1000000, priority=90, quota_style="trial",
    ),
    ProviderDef(
        key="local", label="Local OpenAI-compatible server",
        base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key_env="LOCAL_LLM_API_KEY", models=LOCAL_MODELS,
        rpm=0, priority=5, quota_style="local",
    ),
)

#: Providers disabled unless explicitly enabled: paid-plan/trial-card
#: requirements (cerebras, mistral), missing account info (alibaba), or
#: hardware (local).  A key alone never activates them.
DEFAULT_DISABLED = {"cerebras", "local", "mistral", "alibaba"}

_ACCOUNT_ID_ENV = "CLOUDFLARE_ACCOUNT_ID"


#: Single source of truth for the harness version.  v2.9 introduced this after
#: verification found that ``main.py`` and ``README.md`` carried the version
#: only as prose that could drift apart silently -- there was no constant to
#: assert against and no test that read it.
VERSION = "2.9.1"


# ---------------------------------------------------------------------------
# Environment-driven resolution
# ---------------------------------------------------------------------------
def _split_env(name: str) -> List[str]:
    raw = os.getenv(name, "") or ""
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def _int_override(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw.strip()))
    except ValueError:
        return default


def all_provider_defs() -> List[ProviderDef]:
    """Return every provider definition (enabled or not), env-tuned."""
    defs: List[ProviderDef] = []
    for pdef in DEFAULT_PROVIDERS:
        overrides = {
            "rpm": _int_override(f"{pdef.key.upper()}_RPM", pdef.rpm),
            "tpm": _int_override(f"{pdef.key.upper()}_TPM", pdef.tpm),
            "rpd": _int_override(f"{pdef.key.upper()}_RPD", pdef.rpd),
            "tpd": _int_override(f"{pdef.key.upper()}_TPD", pdef.tpd),
        }
        base_override = os.getenv(f"{pdef.key.upper()}_BASE_URL", "").strip()
        if base_override:
            overrides["base_url"] = base_override
        defs.append(replace(pdef, **overrides))
    return defs


def provider_by_key(key: str) -> Optional[ProviderDef]:
    for pdef in all_provider_defs():
        if pdef.key == key:
            return pdef
    return None


def has_credentials(pdef: ProviderDef) -> bool:
    """True when the provider can be called (key present or local)."""
    if pdef.quota_style == "local":
        return os.getenv("LOCAL_LLM_ENABLED", "").strip().lower() in {
            "1", "true", "yes", "on"}
    if pdef.key == "cloudflare" and not os.getenv(_ACCOUNT_ID_ENV, "").strip():
        return False
    return bool(os.getenv(pdef.api_key_env, "").strip())


def api_key(pdef: ProviderDef) -> str:
    if pdef.quota_style == "local":
        return os.getenv(pdef.api_key_env, "") or "local"
    return os.getenv(pdef.api_key_env, "").strip()


def resolve_base_url(pdef: ProviderDef) -> str:
    """Substitute per-provider placeholders (Cloudflare account id)."""
    if "{account_id}" in pdef.base_url:
        account = os.getenv(_ACCOUNT_ID_ENV, "").strip()
        return pdef.base_url.replace("{account_id}", account)
    return pdef.base_url


def enabled_providers() -> List[ProviderDef]:
    """Providers that are enabled AND credentialed, best priority first."""
    allow = set(_split_env("PROVIDERS_ENABLED"))
    deny = set(_split_env("PROVIDERS_DISABLED")) | set(DEFAULT_DISABLED)
    if allow:
        deny -= allow  # an explicit enable wins over the default disable set
    out: List[ProviderDef] = []
    for pdef in all_provider_defs():
        if pdef.key in deny and pdef.key not in allow:
            continue
        if not has_credentials(pdef):
            continue
        out.append(pdef)
    out.sort(key=lambda p: p.priority)
    return out


def provider_status() -> List[Dict[str, object]]:
    """Compact per-provider status for the ``--models status`` CLI."""
    allow = set(_split_env("PROVIDERS_ENABLED"))
    rows: List[Dict[str, object]] = []
    for pdef in all_provider_defs():
        explicit = pdef.key in allow
        enabled = any(p.key == pdef.key for p in enabled_providers())
        rows.append({
            "key": pdef.key,
            "label": pdef.label,
            "enabled": enabled,
            "explicit": explicit,
            "key_env": pdef.api_key_env,
            "credentialed": has_credentials(pdef),
            "rpm": pdef.rpm,
            "rpd": pdef.rpd,
            "monthly_requests": pdef.monthly_requests,
            "priority": pdef.priority,
            "trains_data": any(m.trains_data for m in pdef.models),
            "models": len(pdef.models),
        })
    return rows


def log_enabled_providers() -> None:
    keys = [p.key for p in enabled_providers()]
    log_event("INFO", "providers_resolved", {"enabled": keys})
