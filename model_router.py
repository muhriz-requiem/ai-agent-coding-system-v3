"""Free-model router: catalog, health registry, ranking, probes (v2.6).

This is the "automatically choose reliable free models" layer.  It

* discovers free models (seeded + refreshed from OpenRouter's catalog),
* filters by role capability (JSON roles need structured output support;
  long-form roles need output headroom and a usable TPM),
* ranks survivors by capability prior + observed health + quota headroom,
* records every call/parse/task outcome so reliability is learned from
  real runs (EWMA), not guessed,
* probes cold-start models with canaries on refresh (bounded),
* fails closed (``LLMUnavailableError``) when nothing usable remains, so
  orchestrators defer instead of burning attempts.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config
import llm_cache
import providers
from providers import LLMUnavailableError, ProviderDef, ProviderModel
from logger import log_event
from quota_governor import get_governor

# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------
_JSON_ROLE_PREFIXES = ("planner", "reviewer", "judge", "strategist",
                       "campaign", "submission", "integration",
                       "super_orchestrator", "project_orchestrator")
_LONGFORM_ROLE_PREFIXES = ("builder", "modeler", "feature", "ensembler",
                           "red_team", "eda", "merger")

#: v2.9.1 -- roles that emit **whole files** inside a JSON envelope.
#:
#: ``spec_author`` writes each task's complete test file inline, so its output
#: budget is a function of the plan size, not of the response shape.  It matched
#: no prefix above and therefore fell through to the default ``json`` kind,
#: whose only length gate is ``MODEL_MIN_MAX_OUTPUT`` (4096).  On the observed
#: roster that admitted ``groq:qwen/qwen3.8-27b`` with ``max_output=8192`` --
#: a budget that cannot hold three small test files, let alone eight.  It was
#: ranked first on measured health, answered with a bare planner-shaped
#: ``{"task_id": ..., "steps": [...]}`` object, and the Spec Author reported
#: "authored zero tasks" while the loop retried it for ten minutes.
#:
#: So the role kind now distinguishes "returns a small structured verdict" from
#: "returns files in a structured envelope", and only the latter needs the
#: larger floor below.
_FILE_BEARING_ROLE_PREFIXES = ("spec_author",)

#: Providers whose TPM is too small for long generations (Groq: 8k TPM).
_LONGFORM_MIN_PROVIDER_TPM = 30000


def _role_kind(role_key: str) -> str:
    role = (role_key or "planner").lower()
    if role.startswith(_FILE_BEARING_ROLE_PREFIXES):
        return "file_bearing"
    if role.startswith(_LONGFORM_ROLE_PREFIXES):
        return "longform"
    if role.startswith(_JSON_ROLE_PREFIXES):
        return "json"
    return "json"


# ---------------------------------------------------------------------------
# Small storage helpers
# ---------------------------------------------------------------------------
def _system_dir(workspace_root: Optional[Path] = None) -> Path:
    root = Path(workspace_root) if workspace_root else config.WORKSPACE_ROOT
    return root / ".system"


def _load_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _save_json(path: Path, payload: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


#: Model-id segments that are routing artefacts rather than a vendor name.
#: Cloudflare addresses every model as ``@cf/<vendor>/<model>``, so ``@cf`` has
#: to be skipped before the real vendor is read.
_ROUTING_SEGMENT_RE = re.compile(r"^@[a-z0-9]+$", re.IGNORECASE)


def model_family(provider_key: str, model_id: str) -> Optional[str]:
    """Best-effort vendor family for a ``(provider, model)`` pair.

    Used by v2.9's independence rule: excluding the Builder's *exact ids* is
    defeated by one rename, so an auditing role also has to avoid the Builder's
    family.

    ``provider_key`` is accepted and then mostly ignored, which is the point.
    The naive approach -- treat the provider as the family -- is wrong in both
    directions, and both errors were found by running it against the real roster
    rather than by reading it:

    * **The provider is often not the vendor.**  ``nvidia`` serves
      ``z-ai/glm-5.3`` and ``deepseek-ai/deepseek-v4.1-flash``; ``openrouter``
      serves ``google/gemma``, ``qwen``, ``nvidia``, ``cohere`` and
      ``mistral``.  Deriving family from the provider would mark a dozen
      unrelated OpenRouter models as one family and wipe out the whole provider
      whenever the Builder used any of them.
    * **The vendor is often *in the id*, in the first position.**
      ``openrouter/google/gemma-4-31b-it`` is Google's model, and its id says
      so.  Taking the last segment instead yields ``gemma``.

    So the rule is: skip any leading ``@x`` routing segment, then take the first
    ``/``-separated segment; if there is no ``/``, take the first
    hyphen-separated segment.

    That last fallback is what keeps ``gemini-3.5-flash-lite`` -> ``gemini``
    while ``z-ai/glm-5.3`` -> ``z-ai`` (hyphenated vendor, not ``z``) and
    ``qwen3-coder`` -> ``qwen3`` (distinct from ``qwen``).

    Returns None when no family can be read, which callers treat as *not
    excluded*: an unknown family must never silently exclude a provider.
    """
    del provider_key  # deliberately unused; see the docstring
    segments = [seg for seg in (model_id or "").strip().split("/") if seg]
    while segments and _ROUTING_SEGMENT_RE.match(segments[0]):
        segments.pop(0)
    if not segments:
        return None
    head = segments[0].strip().lower()
    if not head:
        return None
    return head.split("-")[0] if len(segments) == 1 else head


def _health_path(workspace_root: Optional[Path] = None) -> Path:
    return _system_dir(workspace_root) / "model_health.json"


def _registry_path(workspace_root: Optional[Path] = None) -> Path:
    return _system_dir(workspace_root) / "model_registry.json"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------
def _openrouter_models_from_catalog(entries: List[Dict[str, Any]]) -> List[ProviderModel]:
    """Convert raw OpenRouter catalog entries into free ProviderModels."""
    models: List[ProviderModel] = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for entry in entries:
        try:
            pricing = entry.get("pricing") or {}
            if str(pricing.get("prompt")) != "0" or str(pricing.get("completion")) != "0":
                continue
            model_id = str(entry.get("id", ""))
            if not model_id or model_id == "openrouter/free":
                continue
            expiry = entry.get("expiration_date")
            if expiry and str(expiry) < today:
                continue  # already dead
            arch = entry.get("architecture") or {}
            outputs = arch.get("output_modalities") or []
            if outputs and "text" not in outputs:
                continue
            top = entry.get("top_provider") or {}
            params = set(entry.get("supported_parameters") or [])
            benchmarks = entry.get("benchmarks") or {}
            aa = benchmarks.get("artificial_analysis") or {}
            coding = aa.get("coding_index")
            models.append(ProviderModel(
                provider="openrouter",
                model=model_id,
                context_length=int(entry.get("context_length") or 32768),
                max_output=int(top.get("max_completion_tokens") or 8192),
                structured_outputs="structured_outputs" in params,
                response_format="response_format" in params,
                tools="tools" in params,
                coding_index=float(coding) if isinstance(coding, (int, float)) else None,
                free=True,
            ))
        except (TypeError, ValueError):
            continue
    models.sort(key=lambda m: (-(m.coding_index or 0), -m.max_output))
    return models


# ---------------------------------------------------------------------------
# Catalog refresh (v2.9): every credentialed provider, not just OpenRouter
# ---------------------------------------------------------------------------
#: Only providers that actually expose a listable ``/models`` endpoint.  Each
#: of these is OpenAI-compatible, so the shape is the same; they differ in
#: authentication and in how much metadata they expose.
MODELS_ENDPOINT_PROVIDERS = ("openrouter", "groq", "gemini", "nvidia", "zai",
                             "mistral", "cerebras")

#: ``{account_id}`` in a base_url is substituted per-request (see
#: ``providers.request_headers``); a literal brace must never reach the wire.
_ACCOUNT_PLACEHOLDER = "{account_id}"


def _models_url(pdef: ProviderDef) -> Optional[str]:
    """The live model list URL for *pdef*, or None if it has none."""
    if not pdef.supports_models_endpoint:
        return None
    if pdef.key not in MODELS_ENDPOINT_PROVIDERS:
        return None
    if _ACCOUNT_PLACEHOLDER in pdef.base_url:
        return None  # needs an account id we do not resolve for catalog reads
    if pdef.base_url.rstrip("/").endswith("/v1beta/openai"):
        # Gemini's OpenAI-compat base already carries the version prefix.
        return pdef.base_url.rstrip("/") + "/models"
    return pdef.base_url.rstrip("/") + "/models"


def _fetch_models(pdef: ProviderDef, timeout: int) -> List[Dict[str, Any]]:
    """GET *pdef*'s ``/models``.  Raises on any failure."""
    url = _models_url(pdef)
    if not url:
        raise ValueError(f"{pdef.key} has no models endpoint")
    headers: Dict[str, str] = {
        "User-Agent": ("EnterpriseMultiAgentCodingSystem/"
                       f"{providers.VERSION}"),
        "Accept": "application/json",
    }
    key = os.getenv(pdef.api_key_env, "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    for name, value in (pdef.extra_headers or {}).items():
        headers[name] = value
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    if isinstance(payload, dict):
        for field in ("data", "models"):
            if isinstance(payload.get(field), list):
                return payload[field]
    return payload if isinstance(payload, list) else []


def _models_from_generic_list(pdef: ProviderDef,
                              entries: List[Dict[str, Any]]) -> List[ProviderModel]:
    """Convert a plain OpenAI-style ``/models`` list into ProviderModels.

    These endpoints report an id and little else, so there is no ``coding_index``
    to read and no pricing to filter on.  That is precisely the problem this
    function exists to *stop* being permanent: a discovered model lands with
    ``coding_index=None`` and is then carried by the discovery-order probe
    (:func:`_discovery_order`) rather than by a hand-typed number.
    """
    models: List[ProviderModel] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or entry.get("name") or "").strip()
        if not model_id or model_id in (":free", "auto"):
            continue
        # Skip anything already retired: several of these list tombstones.
        if str(entry.get("status") or "").lower() in ("retired", "deprecated"):
            continue
        context = (entry.get("context_length") or entry.get("context_window")
                   or entry.get("max_context_length") or 0)
        try:
            context_length = int(context) if context else 32768
        except (TypeError, ValueError):
            context_length = 32768
        if context_length <= 0:
            context_length = 32768
        models.append(ProviderModel(
            provider=pdef.key,
            model=model_id,
            context_length=context_length,
            max_output=32768,
            # Every provider in scope is OpenAI-compatible, but a bare /models
            # listing cannot confirm structured output support, and assuming it
            # is how JSON-role calls end up routed at models that reject them.
            # Probe fills this in from evidence.
            structured_outputs=False,
            response_format=False,
            tools=True,
            coding_index=None,
            free=pdef.key in MODELS_ENDPOINT_PROVIDERS,
            # v2.9.1: mark the capability fields as UNKNOWN rather than False.
            # These endpoints report an id and little else, so "no structured
            # output" was never a fact about the model -- it was missing data
            # wearing a False. And the gate believed it: `_capability_ok`
            # requires `structured_outputs or response_format` for every JSON
            # role, so a refresh eliminated the entire provider from the
            # planner, reviewer, spec_author, spec_critic and diagnosis pools.
            #
            # Observed after one `--models refresh`: OpenRouter contributed
            # 466 discovered models to the roster and **zero** to any JSON role;
            # the spec_author pool collapsed to the 3 cloudflare models that
            # still carried hand-typed flags, and the author gave up with
            # "all free candidates exhausted". The flags are also advisory only --
            # nothing ever sends `response_format` as a request parameter
            # (checked: zero call sites), so hard-failing on one eliminated
            # providers over a preference the code does not use.
            notes="discovered-capabilities-unverified",
        ))
    models.sort(key=lambda m: m.model)
    return models


def _registry_for(registry: Dict[str, Any], pdef: ProviderDef) -> Dict[str, Any]:
    """The ``{key}_models`` slot for *pdef* inside a registry dict."""
    return dict(registry.get(f"{pdef.key}_models") or [])


def refresh_catalog(timeout: int = 15,
                    workspace_root: Optional[Path] = None,
                    only: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Refresh every credentialed provider's catalog and persist the result.

    v2.8 did this for OpenRouter alone, on the stated grounds that "other
    providers churn slowly".  They do not: NVIDIA retired the whole
    gpt-oss-120b/llama-3.3 line mid-roster and the only reason the harness
    noticed was a hand-edit in ``providers.py`` after a 410.

    Per-provider best-effort: one provider failing (no key, no endpoint, a
    network blip) never discards another provider's successful refresh, and a
    total failure leaves every seed roster exactly where it was.
    """
    wanted = {str(k).strip().lower() for k in (only or ()) if k}
    registry = dict(_load_json(_registry_path(workspace_root), {}))
    results: Dict[str, Any] = {}
    added_all: List[str] = []
    removed_all: List[str] = []

    for pdef in providers.all_provider_defs():
        if wanted and pdef.key not in wanted:
            continue
        if _models_url(pdef) is None:
            continue
        if not providers.has_credentials(pdef):
            results[pdef.key] = {"ok": False, "skipped": "no_credentials"}
            continue
        # The seeds this live list will be diffed against.
        before = {m.model for m in pdef.models}
        try:
            entries = _fetch_models(pdef, timeout)
            models = _models_from_generic_list(pdef, entries)
            if not models:
                raise ValueError("empty model list")
        except Exception as exc:  # noqa: BLE001 -- per-provider isolation
            results[pdef.key] = {"ok": False, "error": str(exc)[:200]}
            log_event("WARNING", "model_catalog_provider_failed",
                      {"provider": pdef.key, "error": str(exc)[:200]})
            continue

        registry[f"{pdef.key}_models"] = [m.__dict__ for m in models]
        after = {m.model for m in models}
        added = sorted(after - before)
        removed = sorted(before - after)
        added_all.extend(f"{pdef.key}:{m}" for m in added)
        removed_all.extend(f"{pdef.key}:{m}" for m in removed)
        results[pdef.key] = {"ok": True, "count": len(models),
                             "added": added, "removed": removed}

    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    registry["catalog_source"] = "live"
    _save_json(_registry_path(workspace_root), registry)

    # The roster diff. A retirement that is merely *absent* from a provider's
    # list is not proof of death, so removals are reported, never auto-applied
    # to the seeds -- that stays an operator decision.
    if added_all or removed_all:
        log_event("INFO", "model_catalog_changed",
                  {"added": added_all[:40], "removed": removed_all[:40],
                   "added_count": len(added_all),
                   "removed_count": len(removed_all)})
    total = sum(1 for r in results.values() if r.get("ok"))
    log_event("INFO", "model_catalog_refreshed",
              {"providers_ok": total, "providers": list(results)})
    # v2.9.1: ``count`` used to be absent from this return, and the CLI read
    # ``result.get("count", 0)`` -- so ``--models refresh`` printed "catalog
    # refreshed: 0 free models" after a refresh that added 610 models and
    # removed 7. The command whose entire job is to report what changed
    # reported zero, confidently, and the diff was only visible in the JSONL
    # log. A number that is always zero is worse than no number.
    total_models = sum(int(r.get("count", 0) or 0)
                       for r in results.values() if r.get("ok"))
    return {"ok": total > 0, "providers": results, "count": total_models,
            "providers_ok": total,
            "added": added_all, "removed": removed_all,
            "added_count": len(added_all), "removed_count": len(removed_all)}


def _persisted_models(pdef: ProviderDef,
                      workspace_root: Optional[Path]) -> List[ProviderModel]:
    """A provider's live-discovered models if the registry is fresh enough."""
    registry = _load_json(_registry_path(workspace_root), {})
    updated = str(registry.get("updated_at") or "")
    if not updated:
        return []
    try:
        age_h = (datetime.now(timezone.utc)
                 - datetime.fromisoformat(updated)).total_seconds() / 3600.0
    except ValueError:
        return []
    if age_h > config.MODEL_REGISTRY_TTL_H:
        return []
    models: List[ProviderModel] = []
    for entry in registry.get(f"{pdef.key}_models") or []:
        if not isinstance(entry, dict) or not entry.get("model"):
            continue
        try:
            models.append(ProviderModel(**entry))
        except TypeError:
            continue
    return models


def _openrouter_models(workspace_root: Optional[Path] = None) -> List[ProviderModel]:
    """Persisted catalog if fresh enough, else built-in seeds."""
    models = _persisted_models(
        ProviderDef(key="openrouter", label="", base_url="", api_key_env="",
                    models=()), workspace_root)
    if models:
        return models
    return [m for m in providers.OPENROUTER_MODELS]


def _merge_models(seeds: Sequence[ProviderModel],
                  discovered: Sequence[ProviderModel]) -> List[ProviderModel]:
    """Hand-written seeds with discovery *added*, not substituted for.

    v2.9.1.  ``_models_for`` returned the persisted live list whenever one
    existed, so a single ``--models refresh`` silently replaced every hand-typed
    seed.  That is the opposite of what ``refresh_catalog`` promises in its own
    comment -- "removals are reported, never auto-applied to the seeds, that
    stays an operator decision" -- because the registry *is* what gets read.

    The cost was concrete.  A bare ``/models`` listing carries no capability
    annotations, so the refreshed roster was 630 unannotated models scoring a
    flat prior, and the annotated ones were gone: the spec_author pool became
    five Gemini *preview* models that the same account answers ``400 Developer
    instruction is not enabled`` for, and the author gave up with "all free
    candidates exhausted".  The verified model that had been working all session
    was no longer in the pool at all.

    So: seeds keep their annotations and their identity, discovery contributes
    everything new, and a discovered entry may only fill a capability field a
    seed left unset.  A seed the provider has genuinely dropped still stays
    visible -- if it has also gone away, the health registry and
    :func:`mark_model_unavailable` retire it on evidence, which is the mechanism
    that is actually allowed to make that call.
    """
    merged: Dict[str, ProviderModel] = {}
    for seed in seeds:
        merged[seed.model] = seed
    for fresh in discovered:
        existing = merged.get(fresh.model)
        if existing is None:
            merged[fresh.model] = fresh
            continue
        # Fill only what the seed left unknown; never overwrite a typed value.
        updates: Dict[str, Any] = {}
        if existing.coding_index is None and fresh.coding_index is not None:
            updates["coding_index"] = fresh.coding_index
        if not existing.structured_outputs and fresh.structured_outputs:
            updates["structured_outputs"] = True
        if not existing.response_format and fresh.response_format:
            updates["response_format"] = True
        if (existing.context_length or 0) < config.MODEL_MIN_CONTEXT \
                and (fresh.context_length or 0) > (existing.context_length or 0):
            updates["context_length"] = fresh.context_length
        if updates:
            merged[fresh.model] = replace(existing, **updates)
    return sorted(merged.values(), key=lambda m: m.model)


def _models_for(pdef: ProviderDef,
                workspace_root: Optional[Path] = None) -> List[ProviderModel]:
    """Hand-written seeds *plus* discovery -- see :func:`_merge_models`.

    Merging rather than substituting is the whole point: discovery adds
    candidates, it does not erase the annotations (and therefore the rankings)
    that were hand-verified.
    """
    seeds = list(pdef.models)
    if pdef.key == "openrouter":
        return _merge_models(seeds, _openrouter_models(workspace_root))
    discovered = _persisted_models(pdef, workspace_root)
    if discovered:
        return _merge_models(seeds, discovered)
    return seeds


# ---------------------------------------------------------------------------
# Dead-model quarantine (v2.9)
# ---------------------------------------------------------------------------
#: Minimum observations before a model may be quarantined on its success rate.
#: Below this the "rate" is mostly noise, and quarantining a model that failed
#: twice is how you end up with an empty pool.
QUARANTINE_MIN_CALLS = 20
#: Success rate at or below which a model is considered unusable.
QUARANTINE_MAX_SUCCESS = 0.10
#: A model whose failures are *all* transient (rate limits, timeouts) is not
#: dead, it is throttled or slow -- very different problems with very different
#: fixes, and conflating them is how a usable model gets thrown away.
_TRANSIENT_OUTCOMES = {"rate_limited", "timeout"}

#: v2.9.1: a lower bar for the specific case of **zero** successes.  The
#: 20-call threshold exists because a *rate* over few observations is noise --
#: but "has never once answered" is not a rate estimate, it is a fact, and 0/5
#: says as much as 0/55.  Observed: two ``nvidia`` models held an authoring
#: attempt for the full 360 s budget each, on repeat, because a timeout counts
#: as transient and 2 < 20.
QUARANTINE_MIN_CALLS_ZERO_OK = 5


def _quarantine_decision(entry: Dict[str, Any]) -> Optional[str]:
    """Why *entry* should be quarantined, or None to keep it.

    Returns a short reason string so the decision is auditable in the log and
    legible in the manifest.
    """
    calls = int(entry.get("calls", 0) or 0)
    ok = int(entry.get("ok", 0) or 0)
    # Never responded?  That is a fact at a much lower sample size than a rate.
    if ok == 0:
        if calls < QUARANTINE_MIN_CALLS_ZERO_OK:
            return None
        hard = (int(entry.get("error", 0) or 0) + int(entry.get("empty", 0) or 0)
                + int(entry.get("parse_fail", 0) or 0))
        transient = (int(entry.get("rate_limited", 0) or 0)
                     + int(entry.get("timeout", 0) or 0))
        # Zero successes in N attempts, whatever the mix.
        #
        # The "all failures are transient, so keep it" rule below is about a
        # model that *works* and is being throttled.  It must not extend to one
        # that has never once responded: rate-limiting a model 54 times out of
        # 55 attempts is not congestion, it is unavailability wearing a 429.
        # groq/gpt-oss-120b and groq/gpt-oss-20b are exactly this -- 0/55 each,
        # 54 rate limits -- and every call spent on them was wasted.
        return (f"0/{calls} ok; never responded "
                f"({hard} hard, {transient} transient)")
    if calls < QUARANTINE_MIN_CALLS:
        return None
    success = ok / calls
    if success > QUARANTINE_MAX_SUCCESS:
        return None
    hard = (int(entry.get("error", 0) or 0) + int(entry.get("empty", 0) or 0)
            + int(entry.get("parse_fail", 0) or 0))
    transient = (int(entry.get("rate_limited", 0) or 0)
                 + int(entry.get("timeout", 0) or 0))
    if hard == 0 and transient > 0:
        return None  # works, just throttled or slow
    return (f"{ok}/{calls} ok ({success:.0%}); {hard} hard failure(s), "
            f"{transient} transient")


def quarantined_models(workspace_root: Optional[Path] = None
                       ) -> Dict[str, str]:
    """``provider|model`` -> reason, for every model worth quarantining."""
    health = _load_json(_health_path(workspace_root), {"entries": {}})
    entries = health.get("entries") or {}
    # The registry keys per role; a model's fate should not depend on which
    # role happened to try it, so aggregate across roles first.
    totals: Dict[str, Dict[str, int]] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        parts = key.split("|")
        if len(parts) < 2:
            continue
        ident = f"{parts[0]}|{parts[1]}"
        agg = totals.setdefault(ident, {})
        for field, value in entry.items():
            if isinstance(value, (int, float)):
                agg[field] = agg.get(field, 0) + int(value)
    out: Dict[str, str] = {}
    for ident, agg in totals.items():
        reason = _quarantine_decision(agg)
        if reason:
            out[ident] = reason
    return out


def is_quarantined(provider: str, model: str,
                   workspace_root: Optional[Path] = None) -> bool:
    return f"{provider}|{model}" in quarantined_models(workspace_root)


# ---------------------------------------------------------------------------
# Model unavailability (v2.9.1) -- availability, as distinct from quality
# ---------------------------------------------------------------------------
#: Substrings that mean "this model is not there", across the providers the
#: system actually talks to.  Matched against a lowercased error string.
_MODEL_GONE_MARKERS = (
    "model not found",
    "no such model",
    "model_unavailable",
    "model is unavailable",
    "model unavailable",
    "model_not_found",
    "does not exist",
    "unknown model",
    "invalid model",
    "invalid_model",
    "unsupported model",
    "is not available",
    "no longer available",
)
#: Stems rather than phrases: providers word the same fact as "is deprecated",
#: "was deprecated", "has been deprecated" and "deprecated on ...", and chasing
#: every tense with exact phrases loses the moment a provider picks a fourth.
#: Each stem is specific enough not to collide with rate limits, timeouts or
#: quota errors, which are transient and handled elsewhere.
_MODEL_GONE_STEMS = (
    "deprecat",
    "retired",
    "decommission",
    "shut down",
    "taken down",
    "pulled from",
)


def error_says_model_gone(error: Any) -> bool:
    """True when *error* reports the model itself as gone.

    Deliberately narrow.  A 429 is congestion and a 5xx is a bad minute; neither
    says anything about whether the model still exists.  This only fires on the
    wording providers use when a model id has been withdrawn, renamed or never
    existed, which is a fact about the *model* rather than about one call.
    """
    text = str(error or "").lower()
    return (any(marker in text for marker in _MODEL_GONE_MARKERS)
            or any(stem in text for stem in _MODEL_GONE_STEMS))


def _unavailable_path(workspace_root: Optional[Path] = None) -> Path:
    return _system_dir(workspace_root) / "model_unavailable.json"


def mark_model_unavailable(provider: str, model: str, reason: str,
                           workspace_root: Optional[Path] = None) -> None:
    """Take one model out of every rotation for a bounded cooldown.

    Why this exists, and why quarantine was not enough.  ``quarantined_models``
    answers "has this model been *consistently* bad", which is the right
    question about quality and the wrong one about existence -- and it needs 20
    observations, aggregated across every role the model has ever served.

    The failure that motivated it: ``openrouter|qwen/qwen3.8-27b:free`` had 23
    recorded planner calls with 16 successes, so it ranked **first** for four
    different roles, and then answered every live call with
    ``404 This model is unavailable``.  Its two failures had just landed in the
    *spec_author* slice, where a 2-call sample cannot move a threshold set at
    20, so nothing in the registry noticed.  The router has no memory of
    recency at all: a model that worked last week outranks one that is failing
    right now.

    So absence gets its own fast, expiring signal.  It is deliberately *not*
    written into the health registry (which is a quality record and is
    summarised in the README), and it expires on its own so a catalog glitch or
    a brief provider outage does not permanently shrink the pool.
    ``--models refresh`` and ``--models probe`` are how an operator re-admits
    a model early.
    """
    if not provider or not model:
        return
    path = _unavailable_path(workspace_root)
    payload = _load_json(path, {})
    if not isinstance(payload, dict):
        payload = {}
    key = f"{provider}|{model}"
    previous = payload.get(key) or {}
    hits = int(previous.get("hits", 0) or 0) + 1
    # Repeated confirmations extend the cooldown (capped), so a model that is
    # genuinely gone stays out while a one-off blip costs one cooldown.
    cooldown = float(config.MODEL_UNAVAILABLE_COOLDOWN_S) * min(4, hits)
    payload[key] = {
        "until": time.time() + min(cooldown, 24 * 3600),
        "cooldown_s": cooldown,
        "hits": hits,
        "reason": str(reason)[:200],
        "marked_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        _save_json(path, payload)
    except OSError:
        return
    log_event("WARNING", "model_marked_unavailable",
              {"model": f"{provider}:{model}", "hits": hits,
               "cooldown_s": cooldown, "reason": str(reason)[:160],
               "note": "excluded from every role until the cooldown expires"})


def unavailable_models(workspace_root: Optional[Path] = None
                       ) -> Dict[str, str]:
    """``provider|model`` -> reason, for models inside an *active* cooldown."""
    path = _unavailable_path(workspace_root)
    payload = _load_json(path, {})
    if not isinstance(payload, dict):
        return {}
    now = time.time()
    out: Dict[str, str] = {}
    for key, entry in payload.items():
        if not isinstance(entry, dict):
            continue
        try:
            until = float(entry.get("until", 0) or 0)
        except (TypeError, ValueError):
            until = 0.0
        if until <= now:
            continue
        out[key] = (f"{entry.get('reason', 'unavailable')} "
                    f"(cooldown {until - now:.0f}s, "
                    f"{int(entry.get('hits', 1) or 1)} hit(s))")
    return out


def clear_model_unavailable(provider: Optional[str] = None,
                            model: Optional[str] = None,
                            workspace_root: Optional[Path] = None) -> int:
    """Re-admit models. All of them, or one. Returns how many were cleared."""
    path = _unavailable_path(workspace_root)
    payload = _load_json(path, {})
    if not isinstance(payload, dict):
        return 0
    if provider and model:
        removed = 1 if payload.pop(f"{provider}|{model}", None) is not None else 0
    else:
        removed = len(payload)
        payload = {}
    if removed:
        try:
            _save_json(path, payload)
        except OSError:
            return 0
        log_event("INFO", "model_unavailability_cleared",
                  {"count": removed,
                   "scope": f"{provider}:{model}" if provider else "all"})
    return removed


# ---------------------------------------------------------------------------
# Health registry
# ---------------------------------------------------------------------------
def _health_entry(health: Dict[str, Any], provider: str, model: str,
                  role_key: str) -> Dict[str, Any]:
    key = f"{provider}|{model}|{role_key}"
    entry = health.setdefault("entries", {}).setdefault(key, {})
    entry.setdefault("calls", 0)
    entry.setdefault("ok", 0)
    entry.setdefault("empty", 0)
    entry.setdefault("timeout", 0)
    entry.setdefault("error", 0)
    entry.setdefault("rate_limited", 0)
    entry.setdefault("parses", 0)
    entry.setdefault("parse_fail", 0)
    entry.setdefault("outcomes_ok", 0)
    entry.setdefault("outcomes_bad", 0)
    entry.setdefault("latency_ewma", 0.0)
    return entry


def record_call(provider: str, model: str, role_key: str,
                outcome: str, latency_s: float = 0.0,
                workspace_root: Optional[Path] = None) -> None:
    """Record one LLM call outcome (ok | empty | timeout | error | 429)."""
    path = _health_path(workspace_root)
    health = _load_json(path, {"entries": {}})
    entry = _health_entry(health, provider, model, role_key)
    entry["calls"] += 1
    if outcome in entry:
        entry[outcome] += 1
    elif outcome == "429":
        entry["rate_limited"] += 1
    if latency_s > 0:
        prev = float(entry.get("latency_ewma") or 0.0)
        entry["latency_ewma"] = round(
            latency_s if prev <= 0 else prev * 0.7 + latency_s * 0.3, 3)
    entry["last_used"] = datetime.now(timezone.utc).isoformat()
    health["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(path, health)


def record_parse(ok: bool, usage: Optional[Dict[str, Any]],
                 workspace_root: Optional[Path] = None) -> None:
    """Record whether a JSON-role response parsed (parse adherence signal)."""
    if not usage:
        return
    path = _health_path(workspace_root)
    health = _load_json(path, {"entries": {}})
    entry = _health_entry(health, str(usage.get("provider", "")),
                          str(usage.get("model", "")),
                          str(usage.get("role", "")))
    entry["parses"] += 1
    if not ok:
        entry["parse_fail"] += 1
    _save_json(path, health)


def record_outcome(usage: Optional[Dict[str, Any]], approved: bool,
                   workspace_root: Optional[Path] = None) -> None:
    """Record the gate outcome attributed to a model usage snapshot.

    ``approved`` is "did this model's output turn into something usable", not
    "did the HTTP call succeed".  The Builder's Reviewer verdict and the Spec
    Author's usable-plan count both land here, so a role-specific quality
    signal exists for every role that has one -- previously only the Builder
    had one, and a model could be ranked top for authoring while having never
    produced a plan.
    """
    if not usage:
        return
    path = _health_path(workspace_root)
    health = _load_json(path, {"entries": {}})
    entry = _health_entry(health, str(usage.get("provider", "")),
                          str(usage.get("model", "")),
                          str(usage.get("role", "")))
    if approved:
        entry["outcomes_ok"] += 1
    else:
        entry["outcomes_bad"] += 1
    detail = usage.get("outcome_detail")
    if isinstance(detail, dict):
        entry["last_outcome"] = {str(k): detail[k] for k in sorted(detail)[:12]}
    _save_json(path, health)


def _latency_score(entry: Dict[str, Any]) -> float:
    """1.0 for fast, decaying to a floor for slow, and **neutral when unknown**.

    v2.9.1 -- this used to be the inline expression
    ``1.0 if latency <= 20.0 or latency <= 0 else max(0.1, 20.0 / latency)``, and
    the ``latency <= 0`` clause made the signal backwards.  ``latency_ewma``
    starts at ``0.0`` and is only updated when a call actually *returns*, so a
    model that had never once responded -- one that times out or errors on
    every attempt -- kept the initial value and scored a perfect 1.0 on 15% of
    its health.  The router was rewarding silence as speed.

    Observed cost: ``nvidia:deepseek-ai/deepseek-v4.1-flash`` and
    ``nvidia:z-ai/glm-5.3-flash`` each held an authoring attempt for the full
    360 s budget, twice per attempt, while the ranking treated them as fast.
    """
    calls = int(entry.get("calls", 0))
    ok = int(entry.get("ok", 0))
    latency = float(entry.get("latency_ewma") or 0.0)
    if calls == 0:
        return 0.5  # never tried: neutral prior, not a free win
    if ok == 0:
        return 0.1  # tried, and never came back: this is not fast
    if latency <= 0:
        return 0.5  # inconsistent record; refuse to guess in either direction
    return 1.0 if latency <= 20.0 else max(0.1, 20.0 / latency)


def _health_score(entry: Dict[str, Any]) -> float:
    """Laplace-smoothed blend of success, parse adherence and outcomes."""
    calls = int(entry.get("calls", 0))
    if calls == 0:
        return 0.60  # neutral prior: new models deserve a try
    ok = int(entry.get("ok", 0))
    success = (ok + 1.0) / (calls + 2.0)
    parses = int(entry.get("parses", 0))
    parse_rate = 1.0 - (int(entry.get("parse_fail", 0)) + 1.0) / (parses + 2.0)
    out_ok = int(entry.get("outcomes_ok", 0))
    out_bad = int(entry.get("outcomes_bad", 0))
    outcome = (out_ok + 1.0) / (out_ok + out_bad + 2.0)
    return (0.40 * success + 0.25 * parse_rate + 0.20 * outcome
            + 0.15 * _latency_score(entry))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    pdef: ProviderDef
    model: ProviderModel
    score: float = 0.0
    #: v2.9: which kind of evidence produced this score -- ``measured``,
    #: ``sparse``, or ``asserted`` (no observations, ranking on the typed-in
    #: ``coding_index`` alone).  Surfaced so a routing decision can be audited
    #: without re-deriving it.
    evidence: str = "asserted"


#: Marker written by :func:`_models_from_generic_list` on models whose
#: capability fields are unknown rather than known-false.
UNVERIFIED_NOTE = "discovered-capabilities-unverified"


#: Below this many candidates, and only when relaxing would at least multiply
#: the pool by this much, the capability filter is treated as over-tight.
_MIN_POOL_BEFORE_RELAX = 3
_RELAX_MULTIPLIER = 2


def kind_needs_structured(role_key: str) -> bool:
    """True for role kinds whose eligibility depends on a capability flag."""
    return _role_kind(role_key) in ("json", "file_bearing")


def _capabilities_unverified(model: ProviderModel) -> bool:
    """True when this model's capability flags are missing data, not verdicts.

    A model discovered from a bare ``/models`` listing carries
    ``structured_outputs=False`` because the endpoint said nothing about it.
    Treating that as "cannot do JSON" removes the model from every structured
    role, which is how one ``--models refresh`` emptied the spec_author pool
    (466 OpenRouter models discovered, zero eligible). The flag is advisory
    anyway -- nothing sends ``response_format`` as a request parameter -- so the
    honest treatment is to admit the model with a lower score, not to exclude it.

    Two signals, because the marker only exists after a refresh:

    * the explicit ``notes`` marker written by
      :func:`_models_from_generic_list`; and
    * **no** ``coding_index``. Every hand-verified seed carries a typed
      capability number, so its absence means "never annotated" -- which is what
      the capability score already treats as a flat prior. Keying on this means
      a registry written *before* this fix is still handled correctly, rather
      than requiring the operator to refresh again.
    """
    if UNVERIFIED_NOTE in (model.notes or ""):
        return True
    return model.coding_index is None


def _capability_ok(role_key: str, pdef: ProviderDef, model: ProviderModel) -> bool:
    kind = _role_kind(role_key)
    if model.context_length < config.MODEL_MIN_CONTEXT:
        return False
    unverified = _capabilities_unverified(model)
    if kind == "file_bearing":
        # Emits complete files inline, so the floor is per-plan, not per-reply.
        if model.max_output < config.MODEL_FILE_BEARING_MIN_MAX_OUTPUT:
            return False
        # An unverified model is admitted: it may well emit JSON, and there is
        # no evidence either way. The rank falls out of `_capability_score`,
        # where a missing `coding_index` already scores a flat prior.
        if unverified:
            return True
        if not (model.structured_outputs or model.response_format):
            return False
        return True
    if kind == "json":
        if unverified:
            return True
        if not (model.structured_outputs or model.response_format):
            return False
        return model.max_output >= config.MODEL_MIN_MAX_OUTPUT
    # longform
    if model.max_output < config.MODEL_LONGFORM_MIN_MAX_OUTPUT:
        return False
    if pdef.tpm and pdef.tpm < _LONGFORM_MIN_PROVIDER_TPM:
        return False
    return True


#: Weight of *measured* evidence versus *asserted* capability in the ranking
#: blend.  v2.9 flipped this.
#:
#: v2.8 scored 0.35 capability / 0.30 health.  Because ``_capability_score``
#: spans 0.175..0.341 across the observed roster while a health score clusters
#: around 0.5..0.9, the *spread* of the hand-typed term exceeded the entire
#: usable spread of the measured one.  Concretely: groq/qwen3.8-27b was scored
#: 0.857 on a ``coding_index=60.0`` typed into providers.py months earlier,
#: while its measured success was 0.67 across 202 calls with 66 hard errors.
#: A stale assertion was outranking 200 observations.
#:
#: Evidence is evidence.  The asserted index is a prior, useful only until a
#: model has been tried.
HEALTH_WEIGHT = 0.50
CAPABILITY_WEIGHT = 0.20
HEADROOM_WEIGHT = 0.20
PRIORITY_WEIGHT = 0.10

#: Below this many recorded calls a health score is mostly prior.  Laplace
#: smoothing already handles the arithmetic; this exists so a 1-call model
#: cannot be treated as measured.
HEALTH_MATURE_CALLS = 5


def _capability_score(role_key: str, model: ProviderModel) -> float:
    coding = model.coding_index
    base = 0.5 if coding is None else max(0.0, min(1.0, float(coding) / 70.0))
    if _role_kind(role_key) == "json" and model.structured_outputs:
        base = min(1.0, base + 0.05)
    return base


def _evidence_label(entry: Dict[str, Any]) -> str:
    """``measured`` once a model has enough calls to be worth believing."""
    if int(entry.get("calls", 0) or 0) >= HEALTH_MATURE_CALLS:
        return "measured"
    if int(entry.get("calls", 0) or 0) > 0:
        return "sparse"
    return "asserted"


def rank_candidates(role_key: str, task_type: Optional[str] = None,
                    limit: Optional[int] = None, reserve: bool = False,
                    workspace_root: Optional[Path] = None,
                    exclude_models: Optional[Sequence[str]] = None,
                    exclude_families: Optional[Sequence[str]] = None
                    ) -> List[Candidate]:
    """Ordered healthy candidates for a role (best first). [] = none.

    ``exclude_models`` drops specific model ids and ``exclude_families`` drops
    vendor families.  Both exist for one reason: **an auditing role must not be
    the same weights as the Builder.**  Free-tier routing collapses roles onto
    whatever flash model is currently healthy -- in the reference run the
    Reviewer reviewed the Builder's work with the Builder's own model in 64 of 65
    live calls, which makes the review an echo rather than a check.

    Family exclusion exists because id exclusion is defeated by a rename:
    excluding ``anthropic/claude-sonnet-5`` still leaves
    ``anthropic/claude-opus-4`` in the pool, which is the same opinion from the
    same training lineage.  The families are computed here, where ``pdef`` is in
    scope, rather than in the caller, so the aggregator rules in
    :func:`model_family` are applied consistently.
    """
    if not config.FREE_MODEL_ROUTING:
        return []
    excluded = {str(m) for m in (exclude_models or ()) if m}
    excluded_families = {str(f).lower() for f in (exclude_families or ()) if f}
    governor = get_governor(workspace_root)
    health = _load_json(_health_path(workspace_root), {"entries": {}})
    entries = health.get("entries") or {}
    top_k = limit if limit is not None else config.MODEL_ROUTER_TOP_K
    out: List[Candidate] = []
    dead = quarantined_models(workspace_root)
    # v2.9.1: absence is enforced separately from quality, and acts on every
    # role at once.  See :func:`mark_model_unavailable` for why the per-role
    # health record cannot see this.
    gone = unavailable_models(workspace_root)
    for pdef in providers.enabled_providers():
        for model in _models_for(pdef, workspace_root):
            if model.model in excluded:
                continue
            if excluded_families:
                family = model_family(pdef.key, model.model)
                # An unreadable family is never excluded: defaulting to
                # "exclude" on a parse failure would silently empty a provider.
                if family and family in excluded_families:
                    continue
            if not _capability_ok(role_key, pdef, model):
                continue
            if f"{pdef.key}|{model.model}" in dead:
                continue
            if f"{pdef.key}|{model.model}" in gone:
                continue
            if governor.cooldown_active(pdef.key, model.model):
                continue
            allowed, _reason = governor.can_call(pdef, tokens=0,
                                                 reserve=reserve)
            if not allowed:
                continue
            key = f"{pdef.key}|{model.model}|{role_key}"
            entry = entries.get(key) or {}
            health_score = _health_score(entry)
            utilization = governor.utilization(pdef)
            headroom = 0.5 if utilization < 0 else max(0.0, 1.0 - utilization)
            priority = max(0.0, min(1.0, (100.0 - pdef.priority) / 100.0))
            score = (CAPABILITY_WEIGHT * _capability_score(role_key, model)
                     + HEALTH_WEIGHT * health_score
                     + HEADROOM_WEIGHT * headroom
                     + PRIORITY_WEIGHT * priority)
            out.append(Candidate(pdef=pdef, model=model, score=round(score, 4),
                                 evidence=_evidence_label(entry)))
    out.sort(key=lambda c: (-c.score, c.pdef.priority, c.model.model))
    ranked = out[:max(0, top_k)]

    # Safety net.  Every filter above can legitimately empty the pool -- a
    # provider quota exhausted, an over-tight capability floor, a refreshed
    # roster that lost its annotations.  A pool of one provider with no
    # redundancy is the worst outcome, because a single provider's bad minute
    # then fails every call for that role with "all free candidates exhausted",
    # which is exactly how the slatepress author gave up.
    #
    # So: if capability filtering left a small pool and a *less* filtered one
    # would be materially larger, fall back and say so.  The same principle as
    # `--preflight scan --committed` refusing to degrade silently.
    if len(ranked) < _MIN_POOL_BEFORE_RELAX and kind_needs_structured(role_key):
        relaxed: List[Candidate] = []
        for pdef in providers.enabled_providers():
            for model in _models_for(pdef, workspace_root):
                if model.model in excluded:
                    continue
                key = f"{pdef.key}|{model.model}"
                if key in dead or key in gone:
                    continue
                if governor.cooldown_active(pdef.key, model.model):
                    continue
                if _role_kind(role_key) == "file_bearing" and \
                        model.max_output < \
                        config.MODEL_FILE_BEARING_MIN_MAX_OUTPUT:
                    continue
                if model.context_length < config.MODEL_MIN_CONTEXT:
                    continue
                entry = entries.get(f"{pdef.key}|{model.model}|{role_key}") or {}
                utilization = governor.utilization(pdef)
                headroom = 0.5 if utilization < 0 else max(
                    0.0, 1.0 - utilization)
                priority = max(0.0, min(1.0, (100.0 - pdef.priority) / 100.0))
                score = (CAPABILITY_WEIGHT * _capability_score(role_key, model)
                         + HEALTH_WEIGHT * _health_score(entry)
                         + HEADROOM_WEIGHT * headroom
                         + PRIORITY_WEIGHT * priority)
                relaxed.append(Candidate(pdef=pdef, model=model,
                                         score=round(score, 4),
                                         evidence=_evidence_label(entry)))
        if len(relaxed) > len(ranked) * _RELAX_MULTIPLIER:
            relaxed.sort(key=lambda c: (-c.score, c.pdef.priority,
                                        c.model.model))
            log_event("WARNING", "capability_pool_relaxed",
                      {"role": role_key,
                       "filtered": len(ranked),
                       "relaxed": len(relaxed),
                       "kept": min(len(relaxed), max(0, top_k)),
                       "note": "structured-output filtering left too small a "
                               "pool; admitted models whose capability flags "
                               "are unknown rather than verified false"})
            return relaxed[:max(0, top_k)]
    return ranked


def models_used_for(project_tasks: str = "") -> Dict[str, List[str]]:
    """Model ids recently used, per role, from the health registry.

    Used to keep the Reviewer off the Builder's model.  The registry already
    records role -> model for every call, so no new bookkeeping is needed.
    """
    result: Dict[str, List[str]] = {}
    health = _load_json(_health_path(None), {"entries": {}})
    for key in (health.get("entries") or {}):
        parts = key.split("|")
        if len(parts) != 3:
            continue
        _provider, model, role = parts
        bucket = result.setdefault(role, [])
        if model not in bucket:
            bucket.append(model)
    return result


# ---------------------------------------------------------------------------
# Canary probes (bounded, optional)
# ---------------------------------------------------------------------------
_JSON_PROBE_SYSTEM = (
    "You are a capability probe. Reply with ONLY one valid JSON object: "
    '{"ok": true, "value": 1}. No markdown, no commentary.')
_JSON_PROBE_USER = "Return the probe JSON object now."
_FENCE_PROBE_SYSTEM = (
    "You are a capability probe. Reply with ONLY one fenced code block in "
    'this exact shape: ```python path=probe.py\\nVALUE = 1\\n```')
_FENCE_PROBE_USER = "Emit the fenced file block now."


def _parse_json_object(text: str) -> bool:
    if not text:
        return False
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return False
    try:
        parsed = json.loads(text[start:end + 1])
        return isinstance(parsed, dict)
    except json.JSONDecodeError:
        return False


def _has_fenced_path_block(text: str) -> bool:
    return bool(re.search(r"```[a-zA-Z0-9_+-]*\s+path=\S+", text or ""))


def _discovery_order(workspace_root: Optional[Path] = None,
                     cap: int = 24) -> List[Candidate]:
    """Least-observed models first, so probing cannot confirm its own priors.

    This is the fix for a circularity that made 13 of the 17 unannotated seeds
    permanently invisible to evidence.

    ``probe_models`` used to build its targets from ``rank_candidates``.  That
    reads as harmless -- probe the best candidates -- but the ranking weights a
    hand-typed ``coding_index``, and a model with none scores a flat 0.5, so it
    ranks below every annotated model, falls outside the probe cutoff, is never
    called, so ``calls`` stays 0, so ``_health_score`` returns its 0.60 neutral
    prior, so it keeps ranking below the cutoff.  Self-reinforcing, and
    unbreakable from inside the loop it was part of.

    So probe order is *observation count*, ascending, independent of the
    ranking it is meant to inform.  Every model with zero calls comes first,
    and only once a model has real observations does ranking get a vote.
    """
    health = _load_json(_health_path(workspace_root), {"entries": {}})
    entries = health.get("entries") or {}
    #: provider|model -> total calls across every role.  The registry keys on
    #: ``provider|model|role``, so a model probed as ``planner`` but built as
    #: ``builder_initial`` has two rows; summing them is what makes "has this
    #: model been tried at all" answerable, which is the whole question here.
    seen: Dict[str, int] = {}
    for key, entry in entries.items():
        if not isinstance(entry, dict):
            continue
        parts = key.split("|")
        if len(parts) < 2:
            continue
        ident = f"{parts[0]}|{parts[1]}"
        seen[ident] = seen.get(ident, 0) + int(entry.get("calls", 0) or 0)

    pool: List[Candidate] = []
    dead = quarantined_models(workspace_root)
    # v2.9.1: a probe is how an operator re-admits a model, so the unavailable
    # set is deliberately NOT applied here.  Anything the cooldown removed is
    # exactly what should be re-tested, least-observed first as before.
    for pdef in providers.enabled_providers():
        for model in _models_for(pdef, workspace_root):
            key = f"{pdef.key}|{model.model}"
            # A quarantined model is not a probe target: spending a call to
            # re-confirm it is dead is exactly the waste quarantine exists to
            # stop.  It keeps its health entry, so recovery stays possible.
            if key in dead:
                continue
            pool.append(Candidate(
                pdef=pdef, model=model, score=0.0,
                evidence="measured" if seen.get(key, 0) >= HEALTH_MATURE_CALLS
                else ("sparse" if seen.get(key, 0) else "asserted")))
    # Fewest observations first; alphabetical within a band so the order is
    # deterministic across runs (a rotating probe order makes logs unreadable).
    pool.sort(key=lambda c: (seen.get(f"{c.pdef.key}|{c.model.model}", 0),
                             c.pdef.key, c.model.model))
    return pool[:max(1, cap)]


async def probe_models(llm: Any, workspace_root: Optional[Path] = None,
                       max_models: Optional[int] = None) -> List[Dict[str, Any]]:
    """Run bounded canaries and feed the health registry.

    ``llm`` must expose ``raw_complete(provider_key, model, system, user)``.

    v2.9: targets come from :func:`_discovery_order` -- least-observed first --
    rather than from ``rank_candidates``.  Probing the ranking's own favourites
    can only ever confirm the ranking.  Cost is unchanged (still bounded by
    ``MODEL_PROBE_MAX_MODELS``); only the choice of which models to spend those
    calls on is different.
    """
    cap = max_models if max_models is not None else config.MODEL_PROBE_MAX_MODELS
    targets = _discovery_order(workspace_root, cap=cap)
    seen: set = set()
    results: List[Dict[str, Any]] = []
    for cand in targets:
        key = f"{cand.pdef.key}|{cand.model.model}"
        if key in seen:
            continue
        seen.add(key)
        try:
            json_text = await llm.raw_complete(
                cand.pdef.key, cand.model.model,
                _JSON_PROBE_SYSTEM, _JSON_PROBE_USER)
            json_ok = _parse_json_object(json_text)
            record_parse(json_ok, {"provider": cand.pdef.key,
                                   "model": cand.model.model,
                                   "role": "probe_json"},
                         workspace_root=workspace_root)
            fence_text = await llm.raw_complete(
                cand.pdef.key, cand.model.model,
                _FENCE_PROBE_SYSTEM, _FENCE_PROBE_USER)
            fence_ok = _has_fenced_path_block(fence_text)
            record_parse(fence_ok, {"provider": cand.pdef.key,
                                    "model": cand.model.model,
                                    "role": "probe_fence"},
                         workspace_root=workspace_root)
            results.append({"provider": cand.pdef.key, "model": cand.model.model,
                            "json": json_ok, "fence": fence_ok})
        except LLMUnavailableError as exc:
            results.append({"provider": cand.pdef.key, "model": cand.model.model,
                            "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- probes are best-effort
            results.append({"provider": cand.pdef.key, "model": cand.model.model,
                            "error": str(exc)})
    return results


# ---------------------------------------------------------------------------
# Provider verification (one minimal live call per provider)
# ---------------------------------------------------------------------------
async def verify_providers(llm: Any, workspace_root: Optional[Path] = None
                           ) -> List[Dict[str, Any]]:
    """Auth/endpoint/model smoke test: one minimal call per provider.

    Tries up to the three lightest configured models per provider so a
    single upstream rate-limited model does not mark the whole provider
    failed.  Results also feed the health registry (role ``verify``).
    Never raises: every failure becomes ``{ok: false, error}``.
    """
    results: List[Dict[str, Any]] = []
    for pdef in providers.enabled_providers():
        models = sorted(_models_for(pdef, workspace_root),
                        key=lambda m: m.max_output)[:3]
        if not models:
            results.append({"provider": pdef.key, "model": None, "ok": False,
                            "error": "no models configured"})
            continue
        row: Dict[str, Any] = {"provider": pdef.key, "ok": False}
        errors: List[str] = []
        for model in models:
            started = time.monotonic()
            try:
                text = await llm.raw_complete(pdef.key, model.model,
                                              "You are a connectivity probe.",
                                              "Reply with exactly: OK")
                row.update({"model": model.model,
                            "ok": bool((text or "").strip()),
                            "latency_s": round(time.monotonic() - started, 2),
                            "response_head": (text or "").strip()[:80]})
                if row["ok"]:
                    break
            except Exception as exc:  # noqa: BLE001 -- try the next model
                errors.append(f"{model.model}: {str(exc)[:160]}")
        if not row["ok"]:
            row.setdefault("model", models[0].model)
            row["error"] = " | ".join(errors)[:400]
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def next_reset_iso(workspace_root: Optional[Path] = None) -> str:
    """Earliest provider quota reset, bounded by ``LLM_DEFER_MAX_S``."""
    reset = get_governor(workspace_root).next_reset_iso()
    try:
        bound = datetime.now(timezone.utc).timestamp() + config.LLM_DEFER_MAX_S
        if datetime.fromisoformat(reset).timestamp() > bound:
            return datetime.fromtimestamp(bound, timezone.utc).isoformat()
    except (TypeError, ValueError):
        pass
    return reset


def capacity_summary(workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    """Everything ``--models status`` prints, in one dict.

    ``unavailable_models`` and ``quarantined_models`` are both reported because
    they mean different things and an operator needs to tell them apart: one is
    "the provider says this id no longer exists" (expiring, recoverable), the
    other is "we tried it a lot and it kept failing" (evidence-based, needs
    re-probing).
    """
    gone = unavailable_models(workspace_root)
    dead = quarantined_models(workspace_root)
    payload = _capacity_summary(workspace_root)
    if isinstance(payload, dict):
        payload["unavailable_models"] = gone
        payload["quarantined_models"] = dead
    return payload


def _capacity_summary(workspace_root: Optional[Path] = None) -> Dict[str, Any]:
    governor = get_governor(workspace_root)
    health = _load_json(_health_path(workspace_root), {"entries": {}})
    entries = health.get("entries") or {}
    ranked = rank_candidates("builder", limit=5, workspace_root=workspace_root)
    return {
        "free_model_routing": bool(config.FREE_MODEL_ROUTING),
        "providers": providers.provider_status(),
        "quota": governor.snapshot(),
        "top_builder_candidates": [
            {"provider": c.pdef.key, "model": c.model.model, "score": c.score}
            for c in ranked
        ],
        "health_entries": len(entries),
        "cache": llm_cache.stats(workspace_root),
        "registered_at": _load_json(_registry_path(workspace_root), {})
                                 .get("updated_at"),
    }
