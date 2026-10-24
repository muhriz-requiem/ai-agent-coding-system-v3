"""Central configuration and idempotent workspace bootstrap.

This module is the **single source of truth** for:

* model routing / escalation / fallback policy (``select_model``),
* feature toggles (integration agent, prompt caching, demo mode),
* the on-disk layout of the multi-agent workspace (``ensure_workspace``).

It is deliberately dependency-light (only ``python-dotenv``, which is
optional at import time) so every other component can import it safely.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Environment loading (.env support) -- optional dependency, offline safe
# ---------------------------------------------------------------------------
# v2.8: set ``AGENTS_NO_DOTENV=1`` to import this module *without* loading
# ``.env``.  ``.env`` holds live provider credentials, so importing ``config``
# otherwise pulls them into ``os.environ`` for every process that touches it --
# including every pytest run, where a single stray read (or a model echoing a
# value into an assertion) can bake a real key into a committed test file.
# The test suite sets this before importing config; nothing in production does.
DOTENV_DISABLED = str(os.getenv("AGENTS_NO_DOTENV", "")).strip().lower() in (
    "1", "true", "yes", "on")
if not DOTENV_DISABLED:
    try:  # python-dotenv is in requirements.txt, but the system must stay
        # runnable even if it is absent: .env loading then silently no-ops.
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
else:  # pragma: no cover - exercised by the test suite
    # Belt and braces: drop any provider key already present in the ambient
    # environment so an opt-out really means "no credentials in this process".
    for _name in list(os.environ):
        if _name.endswith(("_API_KEY", "_API_TOKEN", "_ACCOUNT_ID")):
            os.environ.pop(_name, None)

# ---------------------------------------------------------------------------
# Paths and core settings
# ---------------------------------------------------------------------------
#: Root of the multi-agent workspace (the directory that contains this file).
WORKSPACE_ROOT: Path = Path(__file__).resolve().parent

#: OpenRouter API key.  When empty the system automatically runs in DEMO_MODE.
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "").strip()


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable with a fallback default."""
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable ("1/true/yes/on")."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


#: All provider key env vars the free capacity layer understands.  Kept
#: inline (not imported from ``providers``) to avoid a circular import:
#: DEMO_MODE must resolve at config import time.
_PROVIDER_KEY_ENVS = (
    "OPENROUTER_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
    "CLOUDFLARE_API_TOKEN", "ZAI_API_KEY", "NVIDIA_API_KEY",
    "MISTRAL_API_KEY", "COHERE_API_KEY", "DASHSCOPE_API_KEY",
    "CEREBRAS_API_KEY",
)


def any_provider_key() -> bool:
    """True when at least one LLM provider key is configured."""
    return any(os.getenv(name, "").strip() for name in _PROVIDER_KEY_ENVS)


def _resolve_demo_mode() -> bool:
    """DEMO_MODE=auto -> enabled iff no provider API key is configured.

    Multi-provider free routing means *any* known provider key (not only
    OpenRouter) switches the system to live mode.  Explicit
    ``true``/``false`` values always win over the auto detection.
    """
    raw = _env_str("DEMO_MODE", "auto").lower()
    if raw in {"", "auto"}:
        return not any_provider_key()
    return raw in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Model routing / escalation policy  (single source of truth)
#
# Slugs refreshed against the live OpenRouter catalog (Aug 2026): the old
# ``anthropic/claude-3.5-sonnet`` / ``google/gemini-1.5-pro`` /
# ``deepseek/deepseek-r1`` routes now 404.  Per-role env overrides via
# ``MODEL_OVERRIDES`` always win over these defaults.
# ---------------------------------------------------------------------------
#: Default logical-role -> model mapping (quality-first: strong planner and a
#: strong first-attempt builder, cheaper patch model on retries).
DEFAULT_MODEL_ROUTING: Dict[str, str] = {
    "planner": "anthropic/claude-sonnet-5",
    "super_orchestrator": "anthropic/claude-sonnet-5",
    "project_orchestrator": "google/gemini-2.5-pro",
    "builder_initial": "anthropic/claude-sonnet-5",
    "builder_patch": "deepseek/deepseek-chat-v3.1",
    #: Attempt-3 auto-escalation: when cheap builder attempts fail twice,
    #: the final attempt routes here (premium model, bounded cost).
    "builder_escalate": "anthropic/claude-sonnet-5",
    "reviewer_feature": "deepseek/deepseek-r1-0528",
    "reviewer_bugfix": "openai/gpt-5.4-mini",
    "reviewer_experiment": "deepseek/deepseek-r1-0528",
    "integration_agent": "anthropic/claude-sonnet-5",
    # v2.9: the two Council roles are gone with the Council, replaced by two
    # surfaces that run *before* the failure they used to explain.
    #
    # ``spec_critic`` audits a task's objective against the tests that will
    # grade it, before any Builder attempt.  ``diagnosis`` explains an
    # escalation after the retry budget is gone.  Both are auditing roles and
    # both are excluded from the Builder's model ids *and* its family -- see
    # ``agent_roles.AUDITING_ROLES``.
    "spec_critic": "deepseek/deepseek-chat-v3.1",
    # An auditing role, so it must not share the Builder's family.  It was
    # briefly pointed at claude-sonnet-5 on the reasoning that it only narrates
    # a post-mortem -- which is wrong: above DIAGNOSIS_MIN_CONFIDENCE it
    # auto-rewrites the failing task's ``objective``.  That is a judgement with
    # consequences, not a report.
    "diagnosis": "openai/gpt-5.4",
    # The author of the specification.  Pointed at a strong reasoning model
    # rather than the cheapest available one: this output becomes the contract
    # every downstream Builder is graded against, so its cost is a fraction of
    # the waste it prevents.
    #
    # Shares a family with ``builder_initial`` **on purpose**.  It is not an
    # auditing role -- it authors the plan rather than judging a diff -- and
    # continuity between whoever wrote the spec and whoever implements it is
    # defensible.  Excluding it would shrink the candidate pool the most of any
    # role, and on a free tier that converts collapses into starvation.
    "spec_author": "anthropic/claude-sonnet-5",
}

#: Ordered fallback lists per resolved role.  ``LLMClient`` walks these in
#: order whenever the primary model call fails.
MODEL_FALLBACKS: Dict[str, List[str]] = {
    "planner": [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
        "google/gemini-2.5-pro",
    ],
    "super_orchestrator": [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
    ],
    "project_orchestrator": [
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-5",
    ],
    "builder_initial": [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
        "google/gemini-2.5-pro",
    ],
    "builder_patch": [
        "deepseek/deepseek-chat-v3.1",
        "openai/gpt-5.4-mini",
        "anthropic/claude-sonnet-5",
    ],
    "builder_escalate": [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
    ],
    "reviewer_feature": [
        "deepseek/deepseek-r1-0528",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
    ],
    "reviewer_bugfix": [
        "anthropic/claude-sonnet-5",
        "deepseek/deepseek-r1-0528",
    ],
    "reviewer_experiment": [
        "deepseek/deepseek-r1-0528",
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
    ],
    "integration_agent": [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4",
    ],
    "spec_critic": [
        "deepseek/deepseek-chat-v3.1",
        "openai/gpt-5.4-mini",
        "anthropic/claude-sonnet-5",
    ],
    "diagnosis": [
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "anthropic/claude-sonnet-5",
    ],
    "spec_author": [
        "anthropic/claude-sonnet-5",
        "openai/gpt-5.4-mini",
        "deepseek/deepseek-chat-v3.1",
    ],
}


def _load_model_overrides() -> Dict[str, str]:
    """Load ``MODEL_OVERRIDES`` JSON from the environment (e.g. ``{"planner":
    "provider/model"}``).  Invalid JSON is ignored so the system keeps the
    defaults rather than crashing at import time."""
    raw = _env_str("MODEL_OVERRIDES", "")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except (json.JSONDecodeError, TypeError):
        pass  # keep defaults; misconfiguration must never block startup
    return {}


#: Effective routing table (defaults merged with env overrides).
MODEL_ROUTING: Dict[str, str] = {**DEFAULT_MODEL_ROUTING, **_load_model_overrides()}


def resolve_role(role: str, attempt: Optional[int] = None,
                 task_type: Optional[str] = None) -> str:
    """Resolve a *logical* role into a concrete routing key.

    * ``builder``  -> ``builder_initial`` on attempt 1, ``builder_patch``
      on attempt 2, ``builder_escalate`` (premium model) on the final
      attempt -- cheap-first with a bounded escalation cost.
    * ``reviewer`` -> ``reviewer_bugfix`` for bugfix tasks, else
      ``reviewer_feature``.
    * anything else maps to itself.
    """
    role = (role or "planner").lower()
    if role == "builder":
        if attempt is not None and attempt >= MAX_ATTEMPTS:
            return "builder_escalate"
        return "builder_initial" if (attempt or 1) <= 1 else "builder_patch"
    if role == "reviewer":
        if (task_type or "feature") == "bugfix":
            return "reviewer_bugfix"
        if (task_type or "feature") == "experiment":
            return "reviewer_experiment"
        return "reviewer_feature"
    return role


def select_model(role: str, attempt: Optional[int] = None,
                 task_type: Optional[str] = None) -> str:
    """Return the primary model for a role.

    Escalation policy (this function is the *only* place that decides):
      * builder attempt 1        -> ``builder_initial``
      * builder retries (2+)     -> ``builder_patch``
      * reviewer on bugfix tasks -> ``reviewer_bugfix``
      * reviewer on features     -> ``reviewer_feature``
    """
    key = resolve_role(role, attempt=attempt, task_type=task_type)
    return MODEL_ROUTING.get(key, DEFAULT_MODEL_ROUTING["planner"])


def fallback_models(role: str, attempt: Optional[int] = None,
                    task_type: Optional[str] = None) -> List[str]:
    """Return the ordered fallback list for a role (excluding the primary)."""
    key = resolve_role(role, attempt=attempt, task_type=task_type)
    primary = select_model(role, attempt=attempt, task_type=task_type)
    return [m for m in MODEL_FALLBACKS.get(key, []) if m != primary]


# ---------------------------------------------------------------------------
# Feature toggles and tuning knobs
# ---------------------------------------------------------------------------
#: v2.8: DEPRECATED and inert.  ``CACHE_ENABLED`` gated
#: ``SharedWorkspaceEngine.apply_cache_headers``, which was verified to be a
#: no-op against the real Chat-Completions payload shape and has been removed.
#: The knob is kept so existing ``.env`` files keep parsing, but it no longer
#: does anything.  Use ``LLM_CACHE_ENABLED`` for the exact-match response cache.
CACHE_ENABLED: bool = _env_bool("CACHE_ENABLED", True)

#: Master switch for the cross-project IntegrationAgent.
ENABLE_INTEGRATION_AGENT: bool = _env_bool("ENABLE_INTEGRATION_AGENT", True)

# ---------------------------------------------------------------------------
# Circuit breaker (v2.9): the Council's one useful reflex, without a model
# ---------------------------------------------------------------------------
#: The Council was deleted in v2.9 -- 47 arbiter failures, 9 decisions held
#: for a human, and zero interventions that actually repaired anything, at the
#: price of the most expensive prompt in the system.  What it was genuinely
#: good for was *stopping a doomed run*, and that needs no model at all.
#: These two knobs drive a deterministic check in ``SuperOrchestrator``:
#: zero completions plus too many tokens spent means the project is not
#: learning, it is looping, and gets paused with the evidence attached.
BREAKER_ENABLED: bool = _env_bool("BREAKER_ENABLED", True)
#: Tokens a project may spend in the window with zero tasks completed before
#: the breaker pauses it.  The logtriage reference build spent 766k tokens
#: across 273 calls for 6 delivered modules, so a healthy project clears this
#: comfortably; a genuinely wedged one trips it early.  0 disables the check.
BREAKER_TOKENS_PER_WINDOW: int = max(
    0, int(_env_str("BREAKER_TOKENS_PER_WINDOW", "250000")))

# ---------------------------------------------------------------------------
# Spec Critic (v2.9): verification before the Builder, not after a failure
# ---------------------------------------------------------------------------
#: Run the deterministic spec-vs-tests checks before scheduling a task.  These
#: are free (no model call) and they catch the mechanical failures -- a test
#: importing a name nothing defines, a task promising tests it does not name --
#: that would otherwise each cost up to three Builder attempts and a rejection.
#: Leaving this on is close to free; turning it off only saves an ``ast`` pass.
SPEC_CRITIC_ENABLED: bool = _env_bool("SPEC_CRITIC_ENABLED", True)
#: Run the model's own judgement pass as well.  Costs one call per task, so it
#: is the part worth disabling under quota pressure -- and the part that cannot
#: substitute for the deterministic tier, only add to it.
SPEC_CRITIC_LLM_ENABLED: bool = _env_bool("SPEC_CRITIC_LLM_ENABLED", True)
# ---------------------------------------------------------------------------
# Spec Author (v2.9): the system writes its own specification
# ---------------------------------------------------------------------------
#: Author a tests-first task breakdown from a one-paragraph goal.
#:
#: This transfers real authority: the system begins writing the contract that
#: its own Builder is graded against.  What keeps that honest is structural, not
#: procedural -- the authored test is written *before* the task is baselined, so
#: ``TestIntegrityGuard`` freezes it and refuses Builder edits to it (verified by
#: probe), and every authored task passes ``TaskSpec`` validation plus the Spec
#: Critic before it can be scheduled.
SPEC_AUTHOR_ENABLED: bool = _env_bool("SPEC_AUTHOR_ENABLED", True)
#: Hard cap on authored tasks per goal.  Bounds the blast radius of one bad
#: generation and keeps a single prompt from producing a 40-task backlog.
SPEC_AUTHOR_MAX_TASKS: int = max(
    1, min(24, int(_env_str("SPEC_AUTHOR_MAX_TASKS", "8"))))
#: Write the authored tests to disk.  Turning this off produces a dry-run plan
#: that a human can read and approve first -- the right setting the first time
#: this runs on a real goal.
SPEC_AUTHOR_WRITE_TESTS: bool = _env_bool("SPEC_AUTHOR_WRITE_TESTS", True)
#: v2.9.1: bounded authoring attempts per goal.  Without a budget the loop
#: retried an unsatisfiable goal forever, replaying the identical cached reply
#: every couple of seconds (observed: 10 minutes of wall-clock to learn one
#: thing once).  Each retry carries the previous attempt's rejection notes, so
#: the attempts are genuinely different requests.  On exhaustion the state is
#: written to ``projects/<name>/.agent/spec_author_state.json`` with
#: ``gave_up: true`` and a ``remedy``; delete that file to let it try again.
SPEC_AUTHOR_MAX_ATTEMPTS: int = max(
    1, min(10, int(_env_str("SPEC_AUTHOR_MAX_ATTEMPTS", "3"))))
#: Linear backoff between authoring attempts (attempt N waits N-1 times this),
#: capped at 60 s.  A goal the author cannot satisfy should cost seconds, not
#: the whole ``--once`` budget.
SPEC_AUTHOR_RETRY_BACKOFF_S: float = max(
    0.0, min(60.0, float(_env_str("SPEC_AUTHOR_RETRY_BACKOFF_S", "15"))))

# ---------------------------------------------------------------------------
# Escalation diagnosis (v2.9)
# ---------------------------------------------------------------------------
#: Diagnose a task that has exhausted its retry budget.  One model call per
#: escalation, on evidence the loop already recorded.
#:
#: Escalation used to be terminal: the task arrived at a human as a stack trace,
#: and auto-requeue then handed the identical objective back to the Builder --
#: observed repeating seven times in the logtriage build.  The escalator
#: recorded *that* a task failed and nothing about *why*, so the retry had no new
#: information to act on.  This supplies the missing half.
DIAGNOSIS_ENABLED: bool = _env_bool("DIAGNOSIS_ENABLED", True)
#: Minimum confidence before an auto-remedy touches the board.  Below this the
#: diagnosis is recorded as advice only.  Two reasons for the bar: the
#: diagnosing model also participated in the work that failed, and a wrong
#: auto-applied rewrite is harder to notice than a wrong suggestion.
DIAGNOSIS_MIN_CONFIDENCE: float = 0.6
try:
    DIAGNOSIS_MIN_CONFIDENCE = min(1.0, max(0.0, float(
        _env_str("DIAGNOSIS_MIN_CONFIDENCE", "0.6"))))
except ValueError:
    pass

#: Severity at which the Critic holds a task instead of annotating it.  Only
#: ``high`` blocks; a Critic that stops the workspace on every nit is worse than
#: no Critic.  A held task is surfaced on the board, never silently dropped.
SPEC_CRITIC_BLOCK_SEVERITY: str = _env_str("SPEC_CRITIC_BLOCK_SEVERITY", "high")
#: Cap on the test source sent to the model, in characters.  Generous on
#: purpose: a half-read test file invites a confident wrong verdict, which is
#: more expensive than the tokens.
SPEC_CRITIC_MAX_SOURCE_CHARS: int = max(
    2000, int(_env_str("SPEC_CRITIC_MAX_SOURCE_CHARS", "24000")))

#: Keep the Reviewer off the model that produced the diff it is reviewing.
#: A reviewer sharing the builder's weights is an echo, not a check: free-tier
#: routing sends every role to whichever model is currently healthy, which in
#: the reference run put the Reviewer on the Builder's model in 64 of 65 live
#: calls.  When no independent model is available the exclusion is given up
#: and the collapse is logged as ``reviewer_independence_unavailable``.
REVIEWER_MODEL_EXCLUSION: bool = _env_bool("REVIEWER_MODEL_EXCLUSION", True)

#: Demo mode: deterministic canned LLM responses so the whole system runs
#: offline.  Auto-enabled when no API key is present (see ``_resolve_demo_mode``).
DEMO_MODE: bool = _resolve_demo_mode()

#: Minimum level for the structured JSONL logger.
LOG_LEVEL: str = _env_str("LOG_LEVEL", "INFO").upper()

#: Echo one compact, colored event line to the console for every logged
#: event.  The JSONL file remains the source of truth either way.
CONSOLE_ECHO: bool = _env_bool("CONSOLE_ECHO", True)

#: Real execution mode: Builder diffs are actually applied to project files
#: and the Reviewer gate runs REAL pytest (instead of simulated tests).
#: Always disabled in DEMO_MODE (offline runs never touch files).
REAL_EXECUTION: bool = _env_bool("REAL_EXECUTION", True)

#: Content-mode building: for tests-first tasks (``requires_new_tests``) the
#: Builder emits COMPLETE file contents in fenced blocks (```python path=...)
#: instead of unified diffs -- mid-tier models are far more reliable at
#: "write this whole file" than at diff hunk arithmetic.  Blocks are
#: converted to synthetic new-file diffs, so every safety rail (snapshot,
#: scope guard, rollback, real-test gate) still applies unchanged.
CONTENT_MODE: bool = _env_bool("CONTENT_MODE", True)

#: Per-file generation: multi-file tests-first tasks issue one focused
#: request per target file (small outputs -- the sweet spot of cheap
#: models) instead of one large multi-block response.
CONTENT_PER_FILE: bool = _env_bool("CONTENT_PER_FILE", True)

#: Subprocess timeouts for the real execution gate (seconds).
TEST_TIMEOUT_S: int = 120
LINT_TIMEOUT_S: int = 60
#: Objective-gate subprocess timeout (training/benchmark runs, seconds).
OBJECTIVE_TIMEOUT_S: int = 1800
try:
    OBJECTIVE_TIMEOUT_S = max(10, int(_env_str("OBJECTIVE_TIMEOUT_S", "1800")))
except ValueError:
    pass

#: Budget (seconds) for the single-pass ``--once`` orchestration window.
#: Overridable for long live builds; the default loop has no budget.
ONCE_BUDGET_S: int = 600
try:
    ONCE_BUDGET_S = int(_env_str("ONCE_BUDGET_S", "600"))
except ValueError:
    pass

#: Grace period (seconds) ``--once`` waits for every project loop to report its
#: first scheduling decision.  Without it a wedged or starved loop held the
#: pass open for the whole ``ONCE_BUDGET_S``; with it the pass degrades to
#: "proceed with the loops that did answer" and logs ``once_boot_timeout``.
ONCE_BOOT_S: int = 120
try:
    ONCE_BOOT_S = int(_env_str("ONCE_BOOT_S", "120"))
except ValueError:
    pass

#: Wall-clock budget (seconds) per LLM candidate call. The SDK's per-read
#: timeout never fires on a slow-drip response (observed: a 926 s empty
#: reply), so the await is additionally bounded with ``asyncio.wait_for``.
LLM_CALL_TIMEOUT_S: int = 360
try:
    LLM_CALL_TIMEOUT_S = int(_env_str("LLM_CALL_TIMEOUT_S", "360"))
except ValueError:
    pass

#: Bounded auto-requeue policy for failed (escalated) tasks (v2.5.5).  When
#: a project loop has no eligible work, a ``failed`` task is requeued to
#: ``pending`` after ``FAILED_RETRY_COOLDOWN_S`` -- at most
#: ``FAILED_AUTO_REQUEUES`` times per task (0 disables the feature).  This
#: only ever fires at starvation time (never preempts healthy work) and a
#: manual ``retry`` (dashboard/operator) always grants a fresh budget.
FAILED_RETRY_COOLDOWN_S: int = 600
try:
    FAILED_RETRY_COOLDOWN_S = int(_env_str("FAILED_RETRY_COOLDOWN_S", "600"))
except ValueError:
    pass
FAILED_AUTO_REQUEUES: int = 1
try:
    FAILED_AUTO_REQUEUES = int(_env_str("FAILED_AUTO_REQUEUES", "1"))
except ValueError:
    pass

#: Environment self-healing (v2.5.6): when the real-test gate fails on
#: missing imports, run ``pip install -r <project>/requirements.txt`` once
#: per task and re-run the gate.  Requirements-only policy -- the project's
#: own file is the single source of truth for what may be installed.
ENV_AUTOHEAL_ENABLED: bool = _env_bool("ENV_AUTOHEAL_ENABLED", True)

#: Subprocess timeout (seconds) for the self-heal ``pip install``.
PIP_TIMEOUT_S: int = 300
try:
    PIP_TIMEOUT_S = int(_env_str("PIP_TIMEOUT_S", "300"))
except ValueError:
    pass

#: Builder-Reviewer cycle cap (attempts 1..3) per task.
MAX_ATTEMPTS: int = 3

# ---------------------------------------------------------------------------
# Secrets and egress (redaction + outbound gating)
# ---------------------------------------------------------------------------
#: Master switch for secret redaction.  When true, ``redaction`` replaces the
#: literal values of secrets this machine holds with ``[REDACTED]`` at every
#: sink: the JSONL log, the ``llm_exchange`` heads, outbound model prompts,
#: the shared pattern/anti-pattern memory, the LLM cache, divergence
#: candidate archives and Kaggle submission payloads.  Set false only to
#: debug a redaction false positive -- the safe default is on.
REDACTION_ENABLED: bool = _env_bool("REDACTION_ENABLED", True)

#: Master switch for secret *scanning*.  Independent of ``REDACTION_ENABLED``:
#: scanning detects credential-shaped patterns in files bound to leave the
#: machine (Kaggle submissions) and in the tree before a git commit.  It never
#: mutates text.
SECRET_SCAN_ENABLED: bool = _env_bool("SECRET_SCAN_ENABLED", True)

#: Minimum length for an environment value to be treated as a redactable
#: secret.  Guards against pathological matches on very short values that
#: would appear incidentally in ordinary prose.
SECRET_SCAN_MIN_LEN: int = 8
try:
    SECRET_SCAN_MIN_LEN = max(4, int(_env_str("SECRET_SCAN_MIN_LEN", "8")))
except ValueError:
    pass

#: Comma-separated workspace-relative JSON files that ``--preflight scrub``
#: inspects for leaked credentials (dry-run by default; ``--apply`` rewrites).
SECRET_SCRUB_TARGETS: str = _env_str(
    "SECRET_SCRUB_TARGETS",
    ".shared_workspace/anti_patterns.json,"
    ".shared_workspace/pattern_memory.json")

#: Allow a domain pack to execute an arbitrary ``run_command`` shell string.
#: Off by default: the objective gate's ``run_command`` is LLM-authored and a
#: hypothesis such as ``env > reports/metrics.json`` would otherwise satisfy
#: the metrics check while dumping every credential in the environment into a
#: file that is then submitted to Kaggle.
KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND: bool = _env_bool(
    "KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND", False)

#: Default per-role rate limits (requests per minute) used when
#: ``.system/rate_limits.json`` does not define a role explicitly.
RATE_LIMIT_DEFAULTS: Dict[str, int] = {
    "planner": 30,
    "builder_initial": 30,
    "builder_patch": 60,
    "reviewer_feature": 20,
    "reviewer_bugfix": 20,
    "reviewer_experiment": 20,
    "integration_agent": 15,
    "super_orchestrator": 10,
    "project_orchestrator": 10,
}
DEFAULT_RATE_LIMIT_RPM: int = 30

# ---------------------------------------------------------------------------
# v2.6 free multi-provider routing / capacity layer
# ---------------------------------------------------------------------------
#: Master switch: when true, ``LLMClient`` routes through
#: ``model_router.rank_candidates`` across all credentialed free providers.
#: When false, the legacy static ``MODEL_ROUTING``/``MODEL_FALLBACKS`` path
#: is used byte-for-byte as before.
FREE_MODEL_ROUTING: bool = _env_bool("FREE_MODEL_ROUTING", True)

#: When free routing is on and every free candidate is unhealthy/exhausted,
#: optionally fall through to the static (usually paid) routing tables.
#: Off by default: quota exhaustion defers the task instead of spending.
FREE_MODELS_ALLOW_PAID_FALLBACK: bool = _env_bool(
    "FREE_MODELS_ALLOW_PAID_FALLBACK", False)

#: How many ranked candidates one LLM call may try.
MODEL_ROUTER_TOP_K: int = 5
try:
    MODEL_ROUTER_TOP_K = max(1, int(_env_str("MODEL_ROUTER_TOP_K", "5")))
except ValueError:
    pass

#: Capability floors used by the router's role filters.
MODEL_MIN_CONTEXT: int = 32768
MODEL_MIN_MAX_OUTPUT: int = 4096
MODEL_LONGFORM_MIN_MAX_OUTPUT: int = 16384
#: v2.9.1: floor for roles that emit **whole files** inside a structured
#: envelope (``spec_author`` writes each task's complete test inline).  32768
#: holds roughly three ordinary test files, so a model below it cannot produce
#: a multi-task plan no matter how healthy it looks.  Observed reason this
#: exists: ``groq:qwen/qwen3.8-27b`` (``max_output=8192``) was ranked first for
#: ``spec_author`` and could not fit a single-task plan's test body.
MODEL_FILE_BEARING_MIN_MAX_OUTPUT: int = 32768
#: v2.9.1: how long a model stays out of every rotation after answering with
#: "this model is unavailable / not found / deprecated".  Expiry is the point:
#: a catalog glitch must not permanently shrink the pool, and unlike the
#: health registry this acts on the first observation instead of waiting for 20
#: aggregated calls.  Repeated confirmations multiply it (capped at 4x).
MODEL_UNAVAILABLE_COOLDOWN_S: int = max(
    0, int(_env_str("MODEL_UNAVAILABLE_COOLDOWN_S", "900")))

#: Hybrid reliability: canary probes on refresh + observed run outcomes.
MODEL_PROBE_ENABLED: bool = _env_bool("MODEL_PROBE_ENABLED", True)
MODEL_PROBE_MAX_MODELS: int = 6
try:
    MODEL_PROBE_MAX_MODELS = max(1, int(_env_str("MODEL_PROBE_MAX_MODELS", "6")))
except ValueError:
    pass

#: Catalog cache TTL (hours); preview/stealth models are allowed by policy
#: (``MODEL_ALLOW_PREVIEW`` is informational -- expired models are always
#: dropped from a refreshed catalog).
MODEL_REGISTRY_TTL_H: int = 24
try:
    MODEL_REGISTRY_TTL_H = max(1, int(_env_str("MODEL_REGISTRY_TTL_H", "24")))
except ValueError:
    pass
MODEL_ALLOW_PREVIEW: bool = _env_bool("MODEL_ALLOW_PREVIEW", True)

# ---------------------------------------------------------------------------
# Self-tuning RPM (v2.9)
# ---------------------------------------------------------------------------
#: Each provider's configured RPM (``GROQ_RPM`` etc.) is a hand-typed guess.  It
#: is frequently wrong, and wrong in the expensive direction: groq's configured
#: 30 requests/minute produced 217 genuine http_429 refusals in one reference
#: build, 281 of 325 cooldowns in total.  Backoff was never the problem -- the
#: limit was.  So the governor learns the real limit from the provider's own
#: refusals instead of trusting the constant.
#:
#: On a 429 the effective limit is multiplied by :data:`RPM_RATCHET_DOWN` and
#: floored at :data:`RPM_LEARNED_FLOOR`; it recovers by a smaller step only after
#: :data:`RPM_RECOVER_AFTER_CLEAN` consecutive clean calls.  The asymmetry is
#: deliberate: guessing low costs some throughput, guessing high costs a round
#: trip plus a cooldown plus a wasted retry on a different model.
RPM_RATCHET_DOWN: float = 0.7
try:
    RPM_RATCHET_DOWN = min(0.95, max(0.3, float(
        _env_str("RPM_RATCHET_DOWN", "0.7"))))
except ValueError:
    pass
#: Never ratchet below this, however many 429s arrive.  A provider that
#: refuses even at this rate is being detected by ``verify_providers`` instead,
#: which is a slower and more expensive signal than one more backoff step.
RPM_LEARNED_FLOOR: int = max(1, int(_env_str("RPM_LEARNED_FLOOR", "4")))
#: Multiplier applied to a ratcheted limit when it recovers.  Deliberately not
#: the reciprocal of RPM_RATCHET_DOWN: recovery has to be slower than the
#: tightening, or a provider that oscillates around its true limit ratchets
#: every few calls and never settles.
RPM_RATCHET_UP: float = 1.25
try:
    RPM_RATCHET_UP = max(1.05, min(2.0, float(
        _env_str("RPM_RATCHET_UP", "1.25"))))
except ValueError:
    pass
#: Consecutive successful calls required before a ratcheted limit recovers.
RPM_RECOVER_AFTER_CLEAN: int = max(
    1, int(_env_str("RPM_RECOVER_AFTER_CLEAN", "40")))
#: Set 0 to trust the configured RPM numbers and never learn from 429s.
RPM_SELF_TUNING: bool = _env_bool("RPM_SELF_TUNING", True)

# ---------------------------------------------------------------------------
# Unbroken lines (v2.9)
# ---------------------------------------------------------------------------
#: Carry a bounded summary of completed tasks into the next task's prompt, so a
#: line does not re-derive the API decisions an earlier task already settled.
#:
#: The Builder object is already reused across tasks, so continuity existed in
#: the trivial sense; what was missing was *intent*.  Each task arrived with no
#: idea of what the previous one established, and re-derived the same decisions.
#: That is the waste this removes.
LINE_CONTINUITY: bool = _env_bool("LINE_CONTINUITY", True)
#: How many finished tasks to keep in the cross-task preamble.  Bounded on
#: purpose: unbounded notes crowd out the current task, and the stalest entries
#: are the least relevant while costing the most.
LINE_NOTE_LIMIT: int = max(
    1, int(_env_str("LINE_NOTE_LIMIT", "6")))
#: Independent lines per project.  1 is exactly the v2.8 behaviour: one
#: sequential chain of tasks, so this is a safe setting to try first.
#:
#: Beyond 1 the lines contend for the same shared quota governor, which is
#: correct -- it queues rather than failing -- but it does mean a higher peak
#: request rate.  RPM is the binding constraint on this account (217 genuine
#: 429s in one reference build), so raise this only after the learned limits
#: have settled.
LINE_COUNT: int = max(1, min(8, int(_env_str("LINE_COUNT", "1"))))

#: Fraction of every provider quota held back for probes and in-call
#: retries; ordinary first-choice calls may only use the remainder.
QUOTA_RESERVE_PCT: float = 0.10
try:
    QUOTA_RESERVE_PCT = min(0.9, max(0.0, float(
        _env_str("QUOTA_RESERVE_PCT", "0.10"))))
except ValueError:
    pass

#: Optional per-role daily call caps (JSON), e.g. {"builder": 200}.
try:
    _parsed_caps = json.loads(_env_str("QUOTA_ROLE_CAPS", "") or "{}")
    QUOTA_ROLE_CAPS: Dict[str, int] = {
        str(k): int(v) for k, v in (_parsed_caps or {}).items()
        if isinstance(v, (int, float)) and int(v) > 0
    }
except (json.JSONDecodeError, TypeError, ValueError):
    QUOTA_ROLE_CAPS = {}

#: Exact-match response cache: dedupes byte-identical LLM calls (retries,
#: probes, repeated patterns) so duplicate prompts never spend quota twice.
LLM_CACHE_ENABLED: bool = _env_bool("LLM_CACHE_ENABLED", True)
LLM_CACHE_TTL_H: int = 24
try:
    LLM_CACHE_TTL_H = max(1, int(_env_str("LLM_CACHE_TTL_H", "24")))
except ValueError:
    pass
LLM_CACHE_MAX_ENTRIES: int = 500
try:
    LLM_CACHE_MAX_ENTRIES = max(10, int(
        _env_str("LLM_CACHE_MAX_ENTRIES", "500")))
except ValueError:
    pass

#: When all free capacity is exhausted, tasks are deferred to the earliest
#: provider reset (bounded by ``LLM_DEFER_MAX_S``) instead of escalating.
LLM_DEFER_MAX_S: int = 3600
try:
    LLM_DEFER_MAX_S = max(60, int(_env_str("LLM_DEFER_MAX_S", "3600")))
except ValueError:
    pass

# ---------------------------------------------------------------------------
# v2.6 divergence / convergence
# ---------------------------------------------------------------------------
#: Global divergent-candidate default.  Individual tasks opt in with
#: ``divergent: true`` and/or a per-project ``.agent/divergence.json``
#: (``{"enabled": true, "n_candidates": 3}``) -- task flag wins over board,
#: board wins over this global default.  Off by default: single-candidate
#: behavior is byte-for-byte preserved when nothing opts in.
DIVERGENCE_ENABLED: bool = _env_bool("DIVERGENCE_ENABLED", False)
DIVERGENCE_DEFAULT_CANDIDATES: int = 3
try:
    DIVERGENCE_DEFAULT_CANDIDATES = max(
        1, int(_env_str("DIVERGENCE_DEFAULT_CANDIDATES", "3")))
except ValueError:
    pass
#: Hard cap on candidates per task (quota protection).
DIVERGENCE_MAX_CANDIDATES: int = 4
try:
    DIVERGENCE_MAX_CANDIDATES = max(
        1, int(_env_str("DIVERGENCE_MAX_CANDIDATES", "4")))
except ValueError:
    pass
#: Top-2 scores within this epsilon trigger the blind judge (research:
#: treat differences below measurement resolution as ties).
DIVERGENCE_TIE_EPSILON: float = 0.02
try:
    DIVERGENCE_TIE_EPSILON = max(0.0, float(
        _env_str("DIVERGENCE_TIE_EPSILON", "0.02")))
except ValueError:
    pass
DIVERGENCE_JUDGE_ENABLED: bool = _env_bool("DIVERGENCE_JUDGE_ENABLED", True)
#: Optional third convergence step: an LLM merger combines the gate-passing
#: near-tie candidates into one implementation that must re-pass the gate
#: and match/beat the winner's deterministic score to be accepted.
DIVERGENCE_MERGE_ENABLED: bool = _env_bool("DIVERGENCE_MERGE_ENABLED", False)
#: Plan-level divergence: 2-3 Planner runs (correctness/performance/
#: simplicity lenses) merged deterministically before building.
PLAN_DIVERGENCE: bool = _env_bool("PLAN_DIVERGENCE", False)

# ---------------------------------------------------------------------------
# v2.6 Kaggle domain pack (optional vertical; inert unless a project opts in
# via .agent/domain.json -- the core stays domain-agnostic)
# ---------------------------------------------------------------------------
#: Auto-submit validated submissions only when out-of-fold CV improves.
#: Off by default: the pack prepares + validates submission files, a human
#: (or this flag) decides to spend a daily submission.
KAGGLE_AUTO_SUBMIT: bool = _env_bool("KAGGLE_AUTO_SUBMIT", False)
KAGGLE_DAILY_SUBMISSION_CAP: int = 5
try:
    KAGGLE_DAILY_SUBMISSION_CAP = max(
        0, int(_env_str("KAGGLE_DAILY_SUBMISSION_CAP", "5")))
except ValueError:
    pass
KAGGLE_MIN_CV_IMPROVEMENT: float = 0.0
try:
    KAGGLE_MIN_CV_IMPROVEMENT = max(0.0, float(
        _env_str("KAGGLE_MIN_CV_IMPROVEMENT", "0.0")))
except ValueError:
    pass
#: ``kaggle competitions submit --wait`` timeout (scoring wait budget).
KAGGLE_SUBMIT_WAIT_S: int = 600
try:
    KAGGLE_SUBMIT_WAIT_S = max(0, int(_env_str("KAGGLE_SUBMIT_WAIT_S", "600")))
except ValueError:
    pass
#: Campaign director: generate the next experiment batch when the board
#: starves.  Off until a campaign has proven itself against a human board.
CAMPAIGN_AUTO_GENERATE: bool = _env_bool("CAMPAIGN_AUTO_GENERATE", False)
#: LLM-driven campaign director: generates new experiment hypotheses
#: after each result.  When disabled the pack uses the static queue.
CAMPAIGN_DIRECTOR_ENABLED: bool = _env_bool("CAMPAIGN_DIRECTOR_ENABLED", False)
CAMPAIGN_MAX_NEW_TASKS: int = 3
try:
    CAMPAIGN_MAX_NEW_TASKS = max(
        1, int(_env_str("CAMPAIGN_MAX_NEW_TASKS", "3")))
except ValueError:
    pass

# ---------------------------------------------------------------------------
# Seed file contents (written by ``ensure_workspace`` ONLY if the file is
# missing -- existing state is never overwritten).
# ---------------------------------------------------------------------------
SHARED_UTIL_SOURCE = '''"""Shared token-normalization utility exported by project_alpha.

Other projects (e.g. project_beta) import ``normalize_token`` from this
module, which is what the IntegrationAgent's cross-project conflict
heuristic watches.
"""

from __future__ import annotations


def normalize_token(raw: str) -> str:
    """Return *raw* stripped of surrounding whitespace and lower-cased.

    Raises ``TypeError`` when *raw* is not a string so callers fail fast
    instead of silently normalizing numbers or ``None``.
    """
    if not isinstance(raw, str):
        raise TypeError("raw token must be a string")
    return raw.strip().lower()
'''

TEST_SHARED_UTIL_SOURCE = '''"""Seeded tests for src/shared_util.py (must pass offline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared_util import normalize_token  # noqa: E402


def test_normalize_strips_whitespace():
    assert normalize_token("  HeLLo  ") == "hello"


def test_normalize_is_lowercase():
    assert normalize_token("ABC") == "abc"


def test_normalize_rejects_non_string():
    try:
        normalize_token(None)  # type: ignore[arg-type]
    except TypeError:
        assert True
    else:
        raise AssertionError("expected TypeError for non-string input")
'''

CONSUMER_SOURCE = '''"""Consumer module demonstrating cross-project reuse of shared_util.

project_beta imports ``normalize_token`` from project_alpha's
``shared_util`` module.  When project_alpha changes that module, the
IntegrationAgent detects the dependency via its import-scan heuristic.
"""

from __future__ import annotations

from src.shared_util import normalize_token


def consume(raw: str) -> str:
    """Normalize *raw* via ``shared_util.normalize_token``."""
    return normalize_token(raw)
'''

TEST_CONSUMER_SOURCE = '''"""Seeded tests for src/consumer.py (must pass offline)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.consumer import consume  # noqa: E402


def test_consume_normalizes_token():
    assert consume("  MiXeD Case  ") == "mixed case"


def test_consume_handles_empty_string():
    assert consume("") == ""
'''

#: Exact seed task board for project_alpha.
TASK_BOARD_ALPHA: Dict[str, Any] = {
    "tasks": [
        {
            "id": "TASK-1001",
            "title": "Implement JWT middleware",
            "objective": "Add JWT verification with RSA256 support",
            "target_files": ["src/middleware/auth.py"],
            "dependencies": [],
            "priority": "high",
            "acceptance_criteria": ["Validate token", "Return 401 on invalid"],
            "workspace_patterns_referenced": [],
            "retry_count": 0,
            "status": "pending",
            "last_feedback": None,
            "error_signature": "ERR_JWT_EXPIRED_SIGNATURE_UNCHECKED",
            "task_type": "feature",
            "should_pass": True,
        }
    ]
}

#: Exact seed task board for project_beta.
TASK_BOARD_BETA: Dict[str, Any] = {
    "tasks": [
        {
            "id": "TASK-2001",
            "title": "Consume shared_util",
            "objective": "Use shared_util.normalize_token in src/consumer.py",
            "target_files": ["src/consumer.py"],
            "dependencies": [],
            "priority": "medium",
            "acceptance_criteria": ["Uses normalize_token", "Tests pass"],
            "workspace_patterns_referenced": [],
            "retry_count": 0,
            "status": "pending",
            "last_feedback": None,
            "error_signature": None,
            "task_type": "feature",
            "should_pass": True,
        }
    ]
}


# ---------------------------------------------------------------------------
# Research lane (v3 Phase 4): REA evidence + GitHub prior-art digests
# ---------------------------------------------------------------------------
#: Passive prompt injection of reverse-engineering / prior-art context.
#: Both default OFF: when OFF, prompts and resolved tool sets are byte-for-
#: byte identical to the pre-research-lane ones.  Enabling is an explicit
#: config choice, not a prompt-time decision.  Implementation: ``research.py``;
#: seams: Planner/Builder prompt assembly in ``agent_roles.py``.
#:
#: The REA path shells out to the pinned global package.  The full Evidence
#: envelope measured ~313 MB of stdout for an 8.7 MB asar, so the executor
#: pins ``--filter-output`` to the small evidence fields AND the module caps
#: the packed block at ``REA_OUTPUT_MAX_BYTES`` bytes.
REA_RESEARCH_ENABLED: bool = _env_bool("REA_RESEARCH_ENABLED", False)
#: GitHub prior-art digests to learn ideas from (licenses pinned, no token,
#: per-query cache in ``.shared_workspace/prior_art_cache.json``).
PRIOR_ART_ENABLED: bool = _env_bool("PRIOR_ART_ENABLED", False)
REA_TIMEOUT_S: int = max(30, int(_env_str("REA_TIMEOUT_S", "300")))
PRIOR_ART_TIMEOUT_S: int = max(5, int(_env_str("PRIOR_ART_TIMEOUT_S", "60")))
REA_OUTPUT_MAX_BYTES: int = max(1024,
                                int(_env_str("REA_OUTPUT_MAX_BYTES", "20000")))
PRIOR_ART_MAX_BYTES: int = max(
    1024, int(_env_str("PRIOR_ART_MAX_BYTES", "20000")))
#: Pin the published latest at install time (npm view rea-agents
#: dist-tags.latest) -- registrations move fast and phases pin deliberately.
REA_AGENTS_VERSION: str = _env_str("REA_AGENTS_VERSION", "6.5.0")


# ---------------------------------------------------------------------------
# Export gate (v3 Phase 8a): nothing leaves the workspace without a manifest
# ---------------------------------------------------------------------------
#: The gate *refuses* rather than exports: it stages artifacts, records a
#: manifest, and requires operator approval before anything is exportable.
#: Unlike the research gates above, ON here is the fail-closed direction
#: (False disables the machinery entirely; there is no other export path
#: that could silently take over).  Implementation: ``export_gate.py``.
EXPORT_GATE_ENABLED: bool = _env_bool("EXPORT_GATE_ENABLED", True)
#: Fresh test run allowed at staging time; a project suite that needs longer
#: than this cannot be staged (a stuck suite is not a green suite).
EXPORT_TEST_TIMEOUT_S: int = max(60,
                                 int(_env_str("EXPORT_TEST_TIMEOUT_S", "300")))
#: Comma-separated destinations an operator may approve (e.g. a NEW repo).
#: Empty by design: an empty allowlist approves nothing, so "" means
#: "staging possible, approval impossible until the operator names one".
EXPORT_DESTINATION_ALLOWLIST: str = _env_str("EXPORT_DESTINATION_ALLOWLIST",
                                             "")
#: Operator-configurable blacklist additions; the built-in blacklist (legacy
#: repo, broker endpoints) lives in ``export_gate.BUILTIN_FORBIDDEN_DESTS``
#: and bypasses configuration entirely.
EXPORT_FORBIDDEN_DESTS: str = _env_str("EXPORT_FORBIDDEN_DESTS", "")


# ---------------------------------------------------------------------------
# workspace MCP (v3 Phase 8f): read-only structured state for host sessions
# ---------------------------------------------------------------------------
#: Local stdio MCP server exposing roster/boards/log-tail/steering/export-
#: staging/cache-stats, structured and bounded.  READ-ONLY by construction:
#: there is no write surface, mutations stay on the CLI path behind the
#: orchestrator lock and the export gate.  Default OFF like the research
#: lane: enabling is an explicit operator config choice (.env), never a
#: prompt-time one.  Implementation: ``workspace_mcp.py``.
MCP_WORKSPACE_ENABLED: bool = _env_bool("MCP_WORKSPACE_ENABLED", False)


# ---------------------------------------------------------------------------
# Idempotent workspace bootstrap
# ---------------------------------------------------------------------------
def ensure_workspace(root: Optional[Path] = None) -> Path:
    """Create the full workspace tree and seed files, **only if missing**.

    Safe to call any number of times: existing state (task boards, pattern
    memory, logs, ...) is never overwritten.  Returns the workspace root.
    """
    base = Path(root) if root is not None else WORKSPACE_ROOT

    # -- directory tree -----------------------------------------------------
    directories = [
        base / ".system",
        base / ".shared_workspace" / "draft_logs",
        base / "projects" / "project_alpha" / ".agent",
        base / "projects" / "project_alpha" / "src" / "middleware",
        base / "projects" / "project_alpha" / "tests",
        base / "projects" / "project_beta" / ".agent",
        base / "projects" / "project_beta" / "src",
        base / "projects" / "project_beta" / "tests",
        base / "logs",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # -- seed helpers --------------------------------------------------------
    def _seed_text(path: Path, content: str) -> None:
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    def _seed_json(path: Path, payload: Any) -> None:
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # -- .system/ ------------------------------------------------------------
    _seed_json(base / ".system" / "manifest.json", {})
    _seed_text(base / ".system" / "global_rules.md", "")
    _seed_json(base / ".system" / "rate_limits.json", {})

    # -- .shared_workspace/ ---------------------------------------------------
    _seed_json(base / ".shared_workspace" / "pattern_memory.json", {"patterns": []})
    _seed_json(base / ".shared_workspace" / "anti_patterns.json",
               {"anti_patterns": []})
    _seed_text(base / ".shared_workspace" / "canonical_rules.md", "")

    # -- project_alpha --------------------------------------------------------
    alpha = base / "projects" / "project_alpha"
    _seed_json(alpha / ".agent" / "task_board.json", TASK_BOARD_ALPHA)
    _seed_text(alpha / ".agent" / "context.md", "")
    _seed_text(alpha / ".agent" / "review_history.log", "")
    _seed_json(alpha / ".agent" / "cycle_summary.json", {})
    _seed_text(alpha / "src" / "shared_util.py", SHARED_UTIL_SOURCE)
    _seed_text(alpha / "tests" / "test_shared_util.py", TEST_SHARED_UTIL_SOURCE)

    # -- project_beta ---------------------------------------------------------
    beta = base / "projects" / "project_beta"
    _seed_json(beta / ".agent" / "task_board.json", TASK_BOARD_BETA)
    _seed_text(beta / ".agent" / "context.md", "")
    _seed_text(beta / ".agent" / "review_history.log", "")
    _seed_json(beta / ".agent" / "cycle_summary.json", {})
    _seed_text(beta / "src" / "consumer.py", CONSUMER_SOURCE)
    _seed_text(beta / "tests" / "test_consumer.py", TEST_CONSUMER_SOURCE)

    return base


def scaffold_new_project(name: str, root: Optional[Path] = None) -> Path:
    """Create ``projects/<name>/`` with an empty task board (``--new-project``).

    Idempotent for an existing project with an empty board; refuses to
    overwrite a board that already contains tasks.  Returns the project dir.
    Raises ``ValueError`` on invalid names (path-safety).
    """
    import re as _re

    if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,63}", name or ""):
        raise ValueError(
            "project name must be 2-64 chars: letters, digits, '_' or '-'")
    base = Path(root) if root is not None else WORKSPACE_ROOT
    ensure_workspace(base)
    project_dir = base / "projects" / name
    agent_dir = project_dir / ".agent"
    for directory in (agent_dir, project_dir / "src", project_dir / "tests"):
        directory.mkdir(parents=True, exist_ok=True)

    board_path = agent_dir / "task_board.json"
    if board_path.exists():
        try:
            existing = json.loads(board_path.read_text(encoding="utf-8"))
            if existing.get("tasks"):
                raise FileExistsError(
                    f"refusing to clobber non-empty board: {board_path}")
        except json.JSONDecodeError as exc:
            raise FileExistsError(
                f"refusing to clobber unreadable board: {exc}") from exc

    seeds = {
        board_path: json.dumps({"tasks": []}, indent=2) + "\n",
        agent_dir / "context.md": "",
        agent_dir / "review_history.log": "",
        agent_dir / "cycle_summary.json": "{}\n",
    }
    for path, content in seeds.items():
        if not path.exists():
            path.write_text(content, encoding="utf-8")
    return project_dir
