"""Secret redaction at every sink: log, model exchange, prompts, artifacts."""

from __future__ import annotations

import asyncio
import json

import pytest

import config
import divergence
import llm_cache
import logger
import redaction
import shared_workspace_engine
from agent_roles import LLMClient

# Synthetic stand-in for a live key -- never the operator's real credentials.
FAKE = "Zz9qW3eR7tYuIoP1aSdF4gH6jKlZxCvB2nM5"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch, restore_test_log_dir):
    """Redirect the workspace root and pin a synthetic secret into the env."""
    monkeypatch.setenv("SINK_TEST_API_KEY", FAKE)
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(config, "REDACTION_ENABLED", True)
    monkeypatch.setattr(config, "CONSOLE_ECHO", False)
    monkeypatch.setattr(config, "LOG_LEVEL", "DEBUG")
    # v2.8: tests/conftest.py redirects the logger so a test run cannot
    # interleave synthetic events into the production log.  These tests assert
    # on what they wrote, so they opt back into their own tmp_path; the
    # ``restore_test_log_dir`` fixture puts the shared redirection back.
    logger.set_log_dir(tmp_path / "logs")
    redaction.reset_cache()
    yield tmp_path
    redaction.reset_cache()


def _log_entries(root):
    path = root / "logs" / "agent_log.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# logger
# ---------------------------------------------------------------------------
def test_log_event_redacts_payload_values(tmp_path):
    logger.log_event("INFO", "llm_model_failed",
                     {"error": f"auth failed for key {FAKE}", "model": "m1"})
    blob = json.dumps(_log_entries(tmp_path))
    assert FAKE not in blob
    assert "[REDACTED]" in blob
    # Non-secret fields survive so the event stays diagnosable.
    assert '"model": "m1"' in blob


def test_log_event_redacts_nested_structures(tmp_path):
    logger.log_event("WARNING", "weird",
                     {"detail": [{"stdout": f"TOKEN={FAKE}"}], "n": 3})
    blob = json.dumps(_log_entries(tmp_path))
    assert FAKE not in blob
    assert '"n": 3' in blob


def test_logger_stays_json_serializable(tmp_path):
    logger.log_event("INFO", "odd_types", {"obj": object(), "n": 1})
    entries = _log_entries(tmp_path)
    assert entries and entries[-1]["data"]["n"] == 1


# ---------------------------------------------------------------------------
# llm_exchange heads
# ---------------------------------------------------------------------------
def test_log_exchange_heads_are_redacted(tmp_path):
    LLMClient._log_exchange("builder", "demo",
                            f"set KEY={FAKE} now", f"done {FAKE}", 0.1)
    entries = [e for e in _log_entries(tmp_path) if e["message"] == "llm_exchange"]
    assert entries
    head = entries[-1]["data"]
    assert FAKE not in json.dumps(head)
    # The identifier is still there: the dashboard shows the model thinking.
    assert "KEY=" in head["prompt_head"]


def test_log_exchange_keeps_code_readable(tmp_path):
    code = 'key = os.getenv("ALPACA_API_KEY")\n'
    LLMClient._log_exchange("builder", "demo", code, code, 0.1)
    entries = [e for e in _log_entries(tmp_path) if e["message"] == "llm_exchange"]
    assert 'os.getenv("ALPACA_API_KEY")' in entries[-1]["data"]["response_head"]


# ---------------------------------------------------------------------------
# outbound prompts
# ---------------------------------------------------------------------------
def test_complete_redacts_the_outbound_prompt(tmp_path, monkeypatch):
    captured = {}

    def _capture(self, role, system, user, tools):
        captured["user"] = user
        return "ok"

    monkeypatch.setattr(LLMClient, "_demo_response", _capture, raising=True)
    monkeypatch.setattr(config, "DEMO_MODE", True)
    client = LLMClient(tmp_path)
    asyncio.run(client.complete("sys", f"the key is {FAKE}"))

    assert captured["user"]
    assert FAKE not in captured["user"]
    assert "the key is" in captured["user"]


# ---------------------------------------------------------------------------
# persisted artifacts
# ---------------------------------------------------------------------------
def test_shared_workspace_write_redacts(tmp_path):
    engine = shared_workspace_engine.SharedWorkspaceEngine(tmp_path)
    engine.add_anti_pattern(signature="ERR_X",
                            description=f"provider rejected key {FAKE}")
    blob = (tmp_path / ".shared_workspace" / "anti_patterns.json").read_text(
        encoding="utf-8")
    assert FAKE not in blob
    # Still valid, still merged, still queryable.
    payload = json.loads(blob)
    assert payload["anti_patterns"]
    assert engine.query_anti_patterns("ERR_X")


def test_llm_cache_stores_redacted_response(tmp_path):
    llm_cache.put("planner", "groq", "m1", "sys", "prompt", f"leaked {FAKE}",
                  workspace_root=tmp_path)
    stored = llm_cache.get("planner", "groq", "m1", "sys", "prompt",
                           workspace_root=tmp_path)
    assert FAKE not in (stored or "")
    # Keying is unaffected: same prompt still hits.
    assert stored is not None


def test_divergence_archive_redacts(tmp_path):
    from validation import StrategySpec

    record = divergence.CandidateRecord(
        label="A", strategy=StrategySpec(name="baseline", approach="do it"),
        model_role="builder_initial", diff=f"+KEY = '{FAKE}'\n",
        feedback=f"rejected {FAKE}", gate_ok=True, tests_passed=1,
        tests_failed=0, lint_count=0, metric=0.9, direction="maximize")
    path = divergence.store_candidate(tmp_path, "TASK-1", record)
    assert path is not None
    blob = path.read_text(encoding="utf-8")
    assert FAKE not in blob
    assert json.loads(blob)["label"] == "A"


# ---------------------------------------------------------------------------
# end-to-end: a leak attempt must not survive the log -> prompt chain
# ---------------------------------------------------------------------------
def test_leaked_exception_text_cannot_reach_a_prompt(tmp_path):
    """The realized leak: provider exception -> log -> anti-pattern -> prompt.

    Guards the whole chain.  A credential in a provider error body must not
    come back out in a later model prompt.
    """
    account_id = "user_ZZZFAKEFAKEFAKEACCT123456789"
    # 1. Provider raises with the key AND the account id in the error body --
    #    the shape of a real OpenAI-compatible billing/auth error.
    reason = f"401 invalid key {FAKE} for {account_id}"
    logger.log_event("WARNING", "llm_model_failed_trying_fallback",
                     {"model": "m", "role": "builder", "error": reason})
    # 2. Orchestrator turns that reason into an anti-pattern description.
    engine = shared_workspace_engine.SharedWorkspaceEngine(tmp_path)
    engine.add_anti_pattern(
        signature="ERR_LEAK",
        description=f"max build-review cycles failed: {reason}")
    # 3. The next prompt is assembled from that memory.
    injected = json.dumps(engine.query_anti_patterns("ERR_LEAK"))

    # The configured secret is gone at both sinks.
    assert FAKE not in json.dumps(_log_entries(tmp_path))
    assert FAKE not in injected

    # v2.8 POLICY CHANGE (found by dogfooding, not by inspection).  This test
    # used to assert the account identifier SURVIVED into the prompt, pinning
    # the "exact-values-only at sinks" rule on purpose: applying every
    # credential *shape* to prompts would blind a Reviewer that has to judge
    # code handling credentials.
    #
    # That reasoning is sound for credential shapes and wrong for account
    # identifiers.  The `logtriage` project -- built by this harness to
    # diagnose itself -- found 48 raw `user_*` and 11 raw `org_*` values still
    # sitting in the production log, and this exact chain (provider error ->
    # anti-pattern -> prompt) is how the id reached third parties in the first
    # place.  An account id carries no diagnostic value: "the provider returned
    # 402" is the signal, not *whose* account it was.
    #
    # So identifiers are now the narrow, deliberate exception.  Every OTHER
    # credential shape is still exact-value-only at sinks, which keeps the
    # Reviewer able to read credential-handling code.
    assert account_id not in injected
    assert "REDACTED" in injected

    # 4. The remediation still works, and finds nothing left to do.
    scrubbed, found = redaction.scrub_secrets(injected)
    assert account_id not in scrubbed