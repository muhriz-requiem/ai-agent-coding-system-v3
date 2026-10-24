"""Phase 4: reviewer independence, hard call bound, token accounting,
node-level tests-first gate.

Each test targets a specific defect found by analysing the reference run's
36,646 log events.
"""

import asyncio
import json
import types

import pytest

import config
import executors
import logger
import model_router
from project_orchestrator import real_tests_block
# Aliased: pytest would collect a bare `test_set_gate_block` import as a
# test function and error on the missing fixture.
from project_orchestrator import test_set_gate_block as _gate_block


# ================================================== reviewer independence
def test_rank_candidates_honours_model_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FREE_MODEL_ROUTING", True)
    monkeypatch.setattr(model_router, "_models_for",
                        lambda pdef, ws=None: list(pdef.models))
    monkeypatch.setattr(model_router, "_capability_ok",
                        lambda role, pdef, model: True)
    monkeypatch.setattr(model_router, "_load_json",
                        lambda path, default=None: {"entries": {}})
    monkeypatch.setattr(model_router.providers, "enabled_providers",
                        lambda: list(model_router.providers.DEFAULT_PROVIDERS))

    gov = types.SimpleNamespace(
        cooldown_active=lambda *a, **k: False,
        can_call=lambda *a, **k: (True, ""),
        utilization=lambda p: 0.0)
    monkeypatch.setattr(model_router, "get_governor", lambda ws=None: gov)

    everything = model_router.rank_candidates("builder", limit=200,
                                              workspace_root=tmp_path)
    assert everything, "precondition: candidates exist"
    victim = everything[0].model.model

    kept = model_router.rank_candidates("reviewer", limit=200,
                                       workspace_root=tmp_path,
                                       exclude_models=[victim])
    assert victim not in {c.model.model for c in kept}
    assert kept, "exclusion must not empty the list when others exist"


def test_reviewer_exclusions_apply_to_every_auditing_role(monkeypatch):
    """v2.9: this assertion used to be the bug.

    The old test asserted that ``spec_critic`` was *not* excluded, on a
    ``startswith("reviewer")`` prefix test.  But the Spec Critic audits the
    objective against the tests that will grade the Builder -- exactly the
    judgement the exclusion exists for -- so it was silently opted out of the
    independence rule the moment v2.8 added it.  ``diagnosis`` had the same
    problem.
    """
    from agent_roles import LLMClient, _BUILDER_MODELS

    monkeypatch.setattr(config, "REVIEWER_MODEL_EXCLUSION", True)
    _BUILDER_MODELS.clear()
    _BUILDER_MODELS["builder"] = "some/flash-model"
    try:
        for role in ("reviewer_feature", "reviewer_bugfix", "spec_critic",
                     "diagnosis"):
            assert LLMClient._reviewer_exclusions(role) == ["some/flash-model"], (
                f"{role} judges Builder output but is not excluded from the "
                f"Builder's weights")
        # Roles that do not judge a diff must keep the full pool.
        for role in ("builder_initial", "planner", "spec_author", "judge",
                     "merger", "campaign", ""):
            assert LLMClient._reviewer_exclusions(role) == [], role
    finally:
        _BUILDER_MODELS.clear()


def test_auditing_roles_are_declared_explicitly():
    """A prefix test silently opts out every role added later.

    That is precisely how ``spec_critic`` and ``diagnosis`` ended up
    unexcluded, and it will happen again to the next role unless membership is
    an explicit set that a reviewer has to look at.
    """
    from agent_roles import AUDITING_ROLES

    assert isinstance(AUDITING_ROLES, frozenset)
    assert {"reviewer_feature", "reviewer_bugfix", "spec_critic",
            "diagnosis"} <= AUDITING_ROLES


def test_every_resolvable_reviewer_route_is_an_auditing_role():
    """Enumerating has a cost, and this is what pays it.

    ``resolve_role("reviewer", ...)`` can return three different keys depending
    on ``task_type``.  The v2.8 prefix test covered all of them implicitly; an
    explicit set has to name them, and this is the test that catches forgetting
    one.  Without it, the switch from prefix to set would have *silently narrowed*
    coverage rather than widening it.
    """
    from agent_roles import AUDITING_ROLES

    resolved = set()
    for task_type in ("feature", "bugfix", "experiment"):
        resolved.add(config.resolve_role("reviewer", task_type=task_type))
    assert resolved <= AUDITING_ROLES, (
        f"reachable reviewer keys missing from AUDITING_ROLES: "
        f"{sorted(resolved - AUDITING_ROLES)}")


def test_reviewer_exclusion_can_be_disabled(monkeypatch):
    from agent_roles import LLMClient, _BUILDER_MODELS

    monkeypatch.setattr(config, "REVIEWER_MODEL_EXCLUSION", False)
    _BUILDER_MODELS.clear()
    _BUILDER_MODELS["builder"] = "m"
    try:
        assert LLMClient._reviewer_exclusions("reviewer_feature") == []
    finally:
        _BUILDER_MODELS.clear()


# ======================================================== hard call bound
def test_call_provider_returns_text_and_usage(tmp_path, monkeypatch):
    from agent_roles import LLMClient

    class Usage:
        prompt_tokens, completion_tokens, total_tokens = 5, 7, 12

    class Message:
        content = "hello"

    class Choice:
        message = Message()

    class Resp:
        choices = [Choice()]
        usage = Usage()

    async def fake_create(**kwargs):
        return Resp()

    llm = LLMClient(tmp_path)
    monkeypatch.setattr(llm, "_provider_client",
                        lambda pdef: types.SimpleNamespace(
                            chat=types.SimpleNamespace(
                                completions=types.SimpleNamespace(
                                    create=fake_create))))
    pdef = types.SimpleNamespace(key="k")
    text, usage = asyncio.run(
        llm._call_provider(pdef, "m", "sys", "usr", None))
    assert text == "hello"
    assert usage.total_tokens == 12


def test_slow_provider_is_abandoned_at_the_budget_not_awaited(tmp_path,
                                                               monkeypatch):
    """A provider that ignores cancellation must not hold the orchestrator.

    The reference run logged 8 *successful* calls at up to 1614s against a
    360s budget, because ``wait_for`` waits for the inner task to unwind.
    """
    from agent_roles import LLMClient

    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_S", 1)
    started = asyncio.Event()
    released = asyncio.Event()

    async def hang(**kwargs):
        started.set()
        await released.wait()      # never completes on its own
        raise AssertionError("should have been abandoned")

    llm = LLMClient(tmp_path)
    monkeypatch.setattr(llm, "_provider_client",
                        lambda pdef: types.SimpleNamespace(
                            chat=types.SimpleNamespace(
                                completions=types.SimpleNamespace(
                                    create=hang))))
    pdef = types.SimpleNamespace(key="k")

    async def run():
        with pytest.raises(asyncio.TimeoutError):
            await llm._call_provider(pdef, "m", "sys", "usr", None)

    # Must return promptly (~1s), not block for the provider's own timeout.
    asyncio.run(asyncio.wait_for(run(), timeout=5))
    released.set()


# ====================================================== token accounting
def test_usage_is_logged_at_info(monkeypatch, tmp_path):
    """INFO, not DEBUG: DEBUG is filtered at the default log level."""
    import agent_roles

    seen = {}
    # agent_roles did `from logger import log_event`, so patch it there.
    monkeypatch.setattr(agent_roles, "log_event",
                        lambda level, msg, data=None: seen.update(
                            level=level, msg=msg, data=data or {}))

    class U:
        prompt_tokens, completion_tokens, total_tokens = 100, 50, 150

    LLMClient = agent_roles.LLMClient
    LLMClient._log_usage("builder_initial", "k:m", U)
    assert seen["level"] == "INFO"
    assert seen["msg"] == "llm_usage"
    assert seen["data"]["total_tokens"] == 150
    assert seen["data"]["completion_tokens"] == 50


def test_absent_usage_is_not_logged_as_zero(monkeypatch):
    """An invented zero would silently flatter the quota governor."""
    import agent_roles

    calls = []
    monkeypatch.setattr(agent_roles, "log_event",
                        lambda level, msg, data=None: calls.append(msg))
    agent_roles.LLMClient._log_usage("builder", "k:m", None)
    agent_roles.LLMClient._log_usage("builder", "k:m",
                                     types.SimpleNamespace(
                                         prompt_tokens=0, completion_tokens=0,
                                         total_tokens=0))
    assert calls == []


def test_cache_hit_is_visible_at_default_log_level(monkeypatch, tmp_path):
    """Was DEBUG, so "cache never hits" and "cache never reports" looked alike."""
    import llm_cache

    seen = []
    monkeypatch.setattr(llm_cache, "log_event",
                        lambda level, msg, data=None: seen.append((level, msg)))
    llm_cache.get("planner", "groq", "m", "s", "u", None, tmp_path)
    assert seen == [], "a miss must log nothing"
    # Now a real hit.
    entry = llm_cache._entry_path(
        llm_cache._digest("planner", "groq", "m", "s", "u", None), tmp_path)
    entry.parent.mkdir(parents=True, exist_ok=True)
    entry.write_text(json.dumps({"created_ts": __import__("time").time(),
                                 "response": "cached"}), encoding="utf-8")
    llm_cache.get("planner", "groq", "m", "s", "u", None, tmp_path)
    assert ("INFO", "llm_cache_hit") in seen


# ================================================ log message redaction
def test_log_message_is_redacted(monkeypatch, tmp_path, restore_test_log_dir):
    """``data`` was redacted; ``message`` was not. Both are now."""
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(config, "CONSOLE_ECHO", False)
    monkeypatch.setattr(config, "LOG_LEVEL", "INFO")
    # conftest redirects the logger away from the production log; this test
    # asserts on the file it writes itself, so opt back into tmp_path.  The
    # ``restore_test_log_dir`` fixture puts the shared redirect back afterwards.
    logger.set_log_dir(tmp_path / "logs")

    class FakeRedaction:
        @staticmethod
        def redact_known_secrets(text, workspace_root=None):
            return str(text).replace("sk-live-SECRET", "[REDACTED]")

        @staticmethod
        def redact_obj(data):
            return data

    monkeypatch.setitem(__import__("sys").modules, "redaction", FakeRedaction)
    logger.log_event("INFO", "token is sk-live-SECRET here", {"k": "v"})
    line = (tmp_path / "logs" / "agent_log.jsonl").read_text(
        encoding="utf-8").strip()
    entry = json.loads(line)
    assert "sk-live-SECRET" not in entry["message"]
    assert "[REDACTED]" in entry["message"]


def test_production_log_is_not_polluted_by_the_test_suite(restore_test_log_dir):
    """A test run must not interleave synthetic events into the run log.

    Interleaved test rows make an audit of ``logs/agent_log.jsonl`` quietly
    wrong -- counting real failures, measuring pause spin, spotting a
    resurrected event -- and ~3000 rows are indistinguishable from real ones.
    """
    prod = config.WORKSPACE_ROOT / "logs" / "agent_log.jsonl"
    before = prod.stat().st_size if prod.exists() else 0
    logger.log_event("INFO", "synthetic_test_event", {"probe": True})
    after = prod.stat().st_size if prod.exists() else 0
    assert after == before, "test event reached the production log"


# ======================================= node-level tests-first gate
def test_gate_blocks_removed_tests():
    baseline = ["tests/test_a.py::test_one", "tests/test_a.py::test_two"]
    current = ["tests/test_a.py::test_one"]
    reason = _gate_block(baseline, current, required=True)
    assert reason and "removed" in reason
    assert "test_two" in reason


def test_gate_blocks_adding_nothing():
    baseline = ["tests/test_a.py::test_one"]
    reason = _gate_block(baseline, list(baseline), required=True)
    assert reason and "added no new test" in reason


def test_gate_accepts_a_pure_addition():
    baseline = ["tests/test_a.py::test_one"]
    current = baseline + ["tests/test_b.py::test_new"]
    assert _gate_block(baseline, current, required=True) is None


def test_gate_is_a_noop_when_not_required():
    assert _gate_block([], [], required=False) is None


def test_collect_test_ids_and_deltas(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_one():\n    assert True\n\n\n"
        "def test_two():\n    assert True\n", encoding="utf-8")
    ids = executors.collect_test_ids(tmp_path)
    assert any("test_one" in i for i in ids)
    assert any("test_two" in i for i in ids)

    deltas = executors.test_set_deltas(ids, ids + ["tests/test_x.py::test_three"])
    assert deltas["added"] == ["tests/test_x.py::test_three"]
    assert deltas["removed"] == []

    deltas = executors.test_set_deltas(ids, [])
    assert len(deltas["removed"]) == len(ids)


def test_collect_test_ids_is_fail_closed(tmp_path):
    """A missing pytest must yield an empty list, never an exception."""
    assert executors.collect_test_ids(tmp_path) == []


def test_count_rule_still_exists_as_the_cheap_check():
    """The cheap integer check is retained, not replaced."""
    assert real_tests_block({"passed": 5}, 5) is True
    assert real_tests_block({"passed": 6}, 5) is False