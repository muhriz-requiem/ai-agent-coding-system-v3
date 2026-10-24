"""Regression tests for bugs found by running the system on a real project.

Every test here corresponds to an observed failure during the first live
dogfood run (project ``logtriage``), not to a hypothetical.
"""

import asyncio
import json

import pytest

import config
import steering
from steering import AttemptRecord, SteeringMonitor


@pytest.fixture(autouse=True)
def _clean():
    steering.reset_monitors()
    yield
    steering.reset_monitors()


def _done_board(n=1):
    return [{"id": f"T-{i}", "status": "done", "retry_count": 0}
            for i in range(n)]


def _pending_board(n=1):
    return [{"id": f"T-{i}", "status": "pending", "retry_count": 0}
            for i in range(n)]


# =========================================================================
# BUG 1 (mine, v2.8 Phase 2): a COMPLETE project was reported as stalled,
# forever, on every idle tick.
#
# Observed: project_alpha / project_beta / project_main -- all boards 100%
# done -- emitted `steering_stall` at 50-58 idle cycles, three projects, several
# times a second, for the whole run.  Same log-spam class as the pause poll
# that v2.8 had just fixed, reintroduced by the module written to replace it.
# It would also have re-tripped the breaker on every heartbeat forever.
# =========================================================================
def test_completed_board_is_not_stalled(tmp_path):
    monitor = SteeringMonitor(tmp_path)
    for _ in range(steering.STALL_CYCLES_N * 3):
        monitor.note_board("p", _done_board())
    assert not [s for s in monitor.evaluate("p", _done_board())
                if s.kind == steering.STALL], \
        "a finished project must never report a stall"


def test_a_failed_task_counts_as_actionable_work(tmp_path):
    """A failed task IS actionable -- auto-requeue may claim it -- so a board
    that is only 'done + failed' may legitimately be called stalled.  This
    pins the boundary of the fix so it cannot be widened by accident."""
    monitor = SteeringMonitor(tmp_path)
    board = [{"id": "T-1", "status": "done", "retry_count": 0},
             {"id": "T-2", "status": "failed", "retry_count": 3,
              "failed_at": "2030-01-01T00:00:00+00:00"}]
    for _ in range(steering.STALL_CYCLES_N + 1):
        monitor.note_board("p", board)
    assert [s for s in monitor.evaluate("p", board)
            if s.kind == steering.STALL], \
        "a board whose only remaining work is a failed task can stall"


def test_completed_project_does_not_bank_idle_cycles(tmp_path):
    """The counter means 'cycles where work was available but nothing moved'.

    Banking ticks on a finished board is what let it reach 58 and stay there.
    """
    monitor = SteeringMonitor(tmp_path)
    for _ in range(steering.STALL_CYCLES_N * 4):
        monitor.note_board("p", _done_board())
    state = monitor.snapshot("p")
    assert state["idle_cycles"] == 0


def test_a_genuinely_stalled_pending_board_still_reports(tmp_path):
    """The fix must not silence the detector it was supposed to keep."""
    monitor = SteeringMonitor(tmp_path)
    board = _pending_board()
    for _ in range(steering.STALL_CYCLES_N + 1):
        monitor.note_board("p", board)
    assert [s for s in monitor.evaluate("p", board)
            if s.kind == steering.STALL]


# =========================================================================
# BUG 2 (mine): the stall signal was level-triggered, so a genuinely stuck
# project emitted one WARNING per idle tick -- which is precisely how a signal
# stops being read.
# =========================================================================
def test_stall_is_edge_triggered_not_level_triggered(tmp_path):
    monitor = SteeringMonitor(tmp_path)
    board = _pending_board()
    for _ in range(steering.STALL_CYCLES_N + 1):
        monitor.note_board("p", board)
    first = [s for s in monitor.evaluate("p", board)
             if s.kind == steering.STALL]
    assert len(first) == 1
    monitor.report_emitted("p", [steering.STALL])
    # Many more idle cycles: the signal must stay quiet.
    for _ in range(steering.STALL_CYCLES_N * 5):
        monitor.note_board("p", board)
    assert not [s for s in monitor.evaluate("p", board)
                if s.kind == steering.STALL], \
        "a reported stall must not repeat every tick"


def test_stall_is_rearmed_by_real_progress(tmp_path):
    monitor = SteeringMonitor(tmp_path)
    board = _pending_board()
    for _ in range(steering.STALL_CYCLES_N + 1):
        monitor.note_board("p", board)
    assert [s for s in monitor.evaluate("p", board)
            if s.kind == steering.STALL]
    monitor.report_emitted("p", [steering.STALL])
    assert not [s for s in monitor.evaluate("p", board)
                if s.kind == steering.STALL]
    # Progress re-arms it.
    monitor.note_board("p", [{"id": "T-1", "status": "done",
                              "retry_count": 0}])
    stalled_again = [{"id": "T-2", "status": "pending", "retry_count": 0}]
    for _ in range(steering.STALL_CYCLES_N + 1):
        monitor.note_board("p", stalled_again)
    assert [s for s in monitor.evaluate("p", stalled_again)
            if s.kind == steering.STALL], \
        "a stall must be able to fire again after progress"


def test_report_emitted_only_stamps_stall(tmp_path):
    monitor = SteeringMonitor(tmp_path)
    monitor.note_board("p", _pending_board())
    before = monitor.snapshot("p")["idle_cycles"]
    monitor.report_emitted("p", [steering.BURN_RATE])
    assert monitor.snapshot("p")["idle_cycles"] == before
    monitor.report_emitted("p", [steering.STALL])
    assert monitor.snapshot("p")["idle_cycles"] == before  # value unchanged


def test_evaluate_stays_pure(tmp_path):
    """evaluate() must not mutate state -- read-side mutation is the defect
    this module was written to replace (the old TTL deleted on read)."""
    monitor = SteeringMonitor(tmp_path)
    for _ in range(steering.STALL_CYCLES_N + 1):
        monitor.note_board("p", _pending_board())
    raw = json.loads((tmp_path / ".system" / "steering_state.json")
                     .read_text(encoding="utf-8"))
    snapshot_a = json.dumps(raw, sort_keys=True)
    monitor.evaluate("p", _pending_board())
    monitor.evaluate("p", _pending_board())
    raw2 = json.loads((tmp_path / ".system" / "steering_state.json")
                      .read_text(encoding="utf-8"))
    assert json.dumps(raw2, sort_keys=True) == snapshot_a


def _planner(tmp_path):
    """A PlannerAgent with a stub LLMClient over a real project root."""
    from agent_roles import LLMClient, PlannerAgent

    llm = LLMClient.__new__(LLMClient)
    llm.last_usage = None
    llm.record_parse = lambda ok: None
    return PlannerAgent(llm, tmp_path, tmp_path)


# =========================================================================
# BUG 3 (pre-existing): the Planner had no retry, so one malformed reply
# produced a ZERO-STEP plan and the build proceeded with no planning.
# Observed live: planner_unparseable_response -> plan_created steps=0.
# =========================================================================
def test_planner_retries_once_before_degrading(monkeypatch, tmp_path):
    from agent_roles import PlannerAgent

    calls = {"n": 0}

    async def fake_complete(self, model_role, user, tools):
        calls["n"] += 1
        return "I think we should probably just write the module."

    monkeypatch.setattr(PlannerAgent, "_complete", fake_complete)
    from validation import TaskSpec

    agent = _planner(tmp_path)

    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["src/x.py"])
    plan = asyncio.run(agent.run(task))
    assert calls["n"] == 2, "the planner must re-ask once"
    assert plan.steps == [], "an unusable reply degrades to an empty plan"


def test_planner_uses_the_first_parseable_reply(monkeypatch, tmp_path):
    from agent_roles import PlannerAgent
    from validation import TaskSpec

    calls = {"n": 0}
    good = json.dumps({"task_id": "T-1", "steps": ["do the thing"],
                       "data_structures": [], "edge_cases": [],
                       "test_approach": []})

    async def fake_complete(self, model_role, user, tools):
        calls["n"] += 1
        return "garbage" if calls["n"] == 1 else good

    monkeypatch.setattr(PlannerAgent, "_complete", fake_complete)
    agent = _planner(tmp_path)
    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["src/x.py"])
    plan = asyncio.run(agent.run(task))
    assert calls["n"] == 2
    assert plan.steps == ["do the thing"]


def test_planner_does_not_retry_a_good_reply(monkeypatch, tmp_path):
    from agent_roles import PlannerAgent
    from validation import TaskSpec

    calls = {"n": 0}
    good = json.dumps({"task_id": "T-1", "steps": ["s"], "data_structures": [],
                       "edge_cases": [], "test_approach": []})

    async def fake_complete(self, model_role, user, tools):
        calls["n"] += 1
        return good

    monkeypatch.setattr(PlannerAgent, "_complete", fake_complete)
    agent = _planner(tmp_path)
    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["src/x.py"])
    asyncio.run(agent.run(task))
    assert calls["n"] == 1, "a parseable plan must not cost a retry"


# =========================================================================
# BUG 4 (mine, v2.8 Phase 4): the hard call bound could reintroduce an
# unbounded hang via `asyncio.shield` inside `wait_for`.
# `wait_for` awaits cancellation of the future it is given; a shield future
# does not complete until the INNER task does, so a provider that ignores
# cancellation would block forever -- the exact stall the bound exists to stop.
# =========================================================================
def test_call_provider_gives_up_even_if_cancellation_is_ignored(
        tmp_path, monkeypatch):
    """A provider that swallows cancellation must not pin the orchestrator.

    The first attempt at this bound used ``wait_for(asyncio.shield(req))``.
    ``wait_for`` awaits cancellation of the future it is handed, and a shield
    future does not complete until the inner task does -- so a stubborn
    provider hung the caller forever.  This test is the reason that was found:
    with the shield version it never returned and had to be killed.

    Note the care needed at the *end*: ``asyncio.run`` cancels stragglers and
    waits for them, so a task that ignores cancellation would hang loop
    shutdown.  The gate is released before returning for exactly that reason --
    and that is also a real operational hazard worth knowing about.
    """
    import agent_roles
    from agent_roles import LLMClient

    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_S", 1)
    released = asyncio.Event()

    async def hang(**kwargs):
        # Swallow cancellation exactly like a badly-behaved SDK, but RETURN
        # once released -- an already-set Event.wait() never yields, so a bare
        # `while True` here would starve the loop instead of ending.
        while True:
            try:
                await released.wait()
                return "late"
            except asyncio.CancelledError:
                continue

    llm = LLMClient(tmp_path)
    monkeypatch.setattr(llm, "_provider_client",
                        lambda pdef: type("C", (), {
                            "chat": type("Chat", (), {
                                "completions": type("Comp", (), {
                                    "create": staticmethod(hang)})})})())
    pdef = type("P", (), {"key": "k"})()

    async def scenario():
        try:
            with pytest.raises(asyncio.TimeoutError):
                await llm._call_provider(pdef, "m", "sys", "usr", None)
        finally:
            released.set()  # let the abandoned task finish so shutdown works

    loop = asyncio.new_event_loop()
    try:
        # Hard outer bound: if _call_provider blocks past the 1s budget plus
        # slack, this fails instead of hanging the suite.
        loop.run_until_complete(asyncio.wait_for(scenario(), timeout=8))
    finally:
        loop.close()
    assert agent_roles._ABANDONED is not None


def test_call_provider_returns_promptly_for_a_stubborn_provider(
        tmp_path, monkeypatch):
    """Timing assertion: the budget is a ceiling, not a suggestion."""
    import time as _time

    from agent_roles import LLMClient

    monkeypatch.setattr(config, "LLM_CALL_TIMEOUT_S", 1)
    released = asyncio.Event()

    async def hang(**kwargs):
        while True:
            try:
                await released.wait()
                return "late"
            except asyncio.CancelledError:
                continue

    llm = LLMClient(tmp_path)
    monkeypatch.setattr(llm, "_provider_client",
                        lambda pdef: type("C", (), {
                            "chat": type("Chat", (), {
                                "completions": type("Comp", (), {
                                    "create": staticmethod(hang)})})})())
    pdef = type("P", (), {"key": "k"})()

    async def scenario():
        started = _time.monotonic()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await llm._call_provider(pdef, "m", "sys", "usr", None)
            return _time.monotonic() - started
        finally:
            released.set()

    loop = asyncio.new_event_loop()
    try:
        elapsed = loop.run_until_complete(scenario())
    finally:
        loop.close()
    assert elapsed < 5, f"gave up after {elapsed:.1f}s against a 1s budget"

# =========================================================================
# BUG 5 (mine, v2.8 Phase 2): `steering_doom_loop` was level-triggered, so a
# single stuck task emitted the same ERROR over and over.
# Observed live: 16 identical `steering_doom_loop` rows for one task, each also
# driving the breaker. Same class of defect as the stall false positive.
# =========================================================================
def _loop(ws, n, sig="ERR_STUCK_LOOP", task="T-1"):
    m = SteeringMonitor(ws)
    for i in range(n):
        m.record_attempt("p", AttemptRecord(
            task_id=task, outcome="CHANGES_REQUESTED", signature=sig,
            diff_hash=f"h{i}"))
    return m


def test_doom_loop_is_not_re_logged_every_evaluation(tmp_path):
    monitor = _loop(tmp_path, steering.DOOM_LOOP_SIGNATURE_N)
    first = [s for s in monitor.evaluate("p") if s.kind == steering.DOOM_LOOP]
    assert len(first) == 1
    monitor.report_emitted("p", [steering.DOOM_LOOP], first)
    # Many further evaluations with the SAME evidence: must stay quiet.
    for _ in range(20):
        assert not [s for s in monitor.evaluate("p")
                    if s.kind == steering.DOOM_LOOP], \
            "an unchanged doom loop must not re-log on every evaluation"


def test_doom_loop_reports_again_when_it_worsens(tmp_path):
    """Edge-triggering must not silence a genuinely worsening loop."""
    monitor = _loop(tmp_path, steering.DOOM_LOOP_SIGNATURE_N)
    first = [s for s in monitor.evaluate("p") if s.kind == steering.DOOM_LOOP]
    monitor.report_emitted("p", [steering.DOOM_LOOP], first)
    assert not [s for s in monitor.evaluate("p")
                if s.kind == steering.DOOM_LOOP]
    for _ in range(steering.REPEAT_STEP):
        monitor.record_attempt("p", AttemptRecord(
            task_id="T-1", outcome="CHANGES_REQUESTED",
            signature="ERR_STUCK_LOOP", diff_hash=f"w{_}"))
    again = [s for s in monitor.evaluate("p") if s.kind == steering.DOOM_LOOP]
    assert again, "a loop that keeps getting worse must be re-reported"
    assert again[0].evidence["consecutive_failures"] >= \
        steering.DOOM_LOOP_SIGNATURE_N + steering.REPEAT_STEP


def test_doom_loop_is_rearmed_by_a_success(tmp_path):
    monitor = _loop(tmp_path, steering.DOOM_LOOP_SIGNATURE_N)
    first = [s for s in monitor.evaluate("p") if s.kind == steering.DOOM_LOOP]
    monitor.report_emitted("p", [steering.DOOM_LOOP], first)
    monitor.record_attempt("p", AttemptRecord(task_id="T-1",
                                             outcome="APPROVED", diff_hash="ok"))
    for i in range(steering.DOOM_LOOP_SIGNATURE_N):
        monitor.record_attempt("p", AttemptRecord(
            task_id="T-1", outcome="CHANGES_REQUESTED",
            signature="ERR_STUCK_LOOP", diff_hash=f"r{i}"))
    assert [s for s in monitor.evaluate("p") if s.kind == steering.DOOM_LOOP], \
        "a fresh loop after progress must report again"


def test_repeated_diff_doom_loop_is_also_edge_triggered(tmp_path):
    m = SteeringMonitor(tmp_path)
    for _ in range(steering.DOOM_LOOP_DIFF_N):
        m.record_attempt("p", AttemptRecord(
            task_id="T-1", outcome="CHANGES_REQUESTED", signature="ERR_A_B",
            diff_hash="identical"))
    first = [s for s in m.evaluate("p") if s.kind == steering.DOOM_LOOP]
    assert first
    m.report_emitted("p", [steering.DOOM_LOOP], first)
    for _ in range(20):
        assert not [s for s in m.evaluate("p")
                    if s.kind == steering.DOOM_LOOP]


def test_evaluate_remains_pure_with_edge_triggering(tmp_path):
    """Edge-triggering must not be bought by mutating during a read."""
    import json as _json

    monitor = _loop(tmp_path, steering.DOOM_LOOP_SIGNATURE_N)
    for _ in range(5):
        monitor.evaluate("p")
    raw = _json.loads((tmp_path / ".system" / "steering_state.json")
                      .read_text(encoding="utf-8"))
    before = _json.dumps(raw, sort_keys=True)
    for _ in range(5):
        monitor.evaluate("p")
    after = _json.dumps(
        _json.loads((tmp_path / ".system" / "steering_state.json")
                    .read_text(encoding="utf-8")), sort_keys=True)
    assert after == before, "evaluate() wrote to the store"


def test_signals_carry_a_dedupe_key(tmp_path):
    monitor = _loop(tmp_path, steering.DOOM_LOOP_SIGNATURE_N)
    for s in monitor.evaluate("p"):
        if s.kind == steering.DOOM_LOOP:
            assert s.key, "doom-loop signals need a stable identity"


# =========================================================================
# BUG 6 (found by the logtriage project this harness built to diagnose
# itself): provider account/org identifiers were still RAW in the production
# log -- 48 `user_*` and 11 `org_*` inside provider exception bodies.
#
# The scan rules for these shapes already existed, but they only ran at the
# egress gates. Sink redaction is exact-value-only by design, so nothing
# blanked them on the way to disk.
# =========================================================================
def test_account_identifiers_are_blanked_at_the_sink():
    import redaction

    out = redaction.redact_known_secrets(
        "Error 402 for user_ZZZFAKEFAKEFAKEACCT123456789 on org_ABCDEFGHIJKLMNOPQ")
    assert "3IQTF0XMV5Q7WOYKJAOPYOSMQEJ" not in out
    assert "ABCDEFGHIJKLMNOPQ" not in out
    assert "REDACTED" in out


def test_identifiers_are_blanked_inside_nested_structures():
    import redaction

    out = redaction.redact_obj({"err": {"detail": "acct user_ZZZFAKEFAKEFAKEACCT123456789"},
                                "n": 7})
    assert "3IQTF0XMV5Q7WOYKJAOPYOSMQEJ" not in json.dumps(out)


def test_ordinary_words_containing_user_are_not_blanked():
    """``username=alice`` must survive; the pattern needs the underscore plus
    a long opaque tail, or every log mentioning a username would be mangled."""
    import redaction

    for clean in ("username=alice", "user_id=7", "superuser_abcdef",
                  "user_abc", "end_user_short"):
        assert redaction.redact_known_secrets(clean) == clean, clean


def test_clean_text_is_returned_unchanged_and_fast():
    import redaction

    text = "task_completed :: project=project_main attempts=1"
    assert redaction.redact_known_secrets(text) is text or \
        redaction.redact_known_secrets(text) == text


def test_identifier_redaction_survives_non_string_input():
    import redaction

    assert redaction.redact_identifiers(None) is None
    assert redaction.redact_identifiers(42) == 42
    assert redaction.redact_identifiers("") == ""


# =========================================================================
# BUG 7 (mine, v2.8 Phase 2): `steering_burn_rate` was ALSO level-triggered.
# Observed live: 12,258 rows -- the single largest source of log noise in the
# entire project, four times worse than the structural-block flood.
#
# I fixed stall, then doom_loop, then structural_block, each time by hand, and
# missed burn_rate all three times. The lesson is that "fix the one I just saw"
# is not a strategy; the invariant below is.
# =========================================================================
def test_no_signal_type_can_flood_when_evidence_is_unchanged(tmp_path):
    """Evaluate 300 times with static evidence; output must stay bounded.

    This is the general form of the bug. Any signal whose evidence has not
    changed since it was last reported must not be re-emitted, whatever its
    kind. Hammering the monitor is how the flood reproduced.
    """
    monitor = SteeringMonitor(tmp_path)
    # Enough state to trip EVERY detector at once.
    for i in range(6):
        monitor.record_attempt("p", AttemptRecord(
            task_id=f"T-{i}", outcome="ESCALATED",
            signature=f"ERR_SCHEMA_T-{i}_MAPPING", diff_hash=f"h{i}",
            duration_s=steering.BURN_TASK_S + 10, tokens=5000))
    board = [{"id": f"T-{i}", "status": "pending", "retry_count": 0}
             for i in range(6)]

    total = 0
    for _ in range(300):
        signals = monitor.evaluate("p", board)
        total += len(signals)
        monitor.report_emitted("p", [s.kind for s in signals], signals)
    assert total <= 40, \
        f"300 evaluations of unchanged evidence emitted {total} signals"


def test_burn_rate_does_not_flood(tmp_path):
    monitor = SteeringMonitor(tmp_path)
    for _ in range(3):
        monitor.record_attempt("p", AttemptRecord(
            task_id="T-1", outcome="CHANGES_REQUESTED", signature="ERR_A_B",
            duration_s=steering.BURN_TASK_S + 5, tokens=1000))
    emitted = 0
    for _ in range(50):
        signals = monitor.evaluate("p", [])
        emitted += len([s for s in signals if s.kind == steering.BURN_RATE])
        monitor.report_emitted("p", [s.kind for s in signals], signals)
    assert emitted <= 4, f"burn_rate emitted {emitted} times on static evidence"


def test_every_signal_kind_has_a_stampable_key(tmp_path):
    """report_emitted can only suppress a signal that carries a key.

    A new signal kind added without one would be silently level-triggered, so
    this pins that any signal the detector produces is stampable.
    """
    monitor = SteeringMonitor(tmp_path)
    for i in range(4):
        monitor.record_attempt("p", AttemptRecord(
            task_id=f"T-{i}", outcome="ESCALATED",
            signature=f"ERR_A_T-{i}_B", diff_hash=f"h{i}",
            duration_s=steering.BURN_TASK_S + 5, tokens=100))
    board = [{"id": f"T-{i}", "status": "pending", "retry_count": 0}
             for i in range(4)]
    signals = monitor.evaluate("p", board)
    assert signals
    unkeyed = [s.kind for s in signals if not s.key]
    assert not unkeyed, f"signals without a dedupe key: {unkeyed}"
