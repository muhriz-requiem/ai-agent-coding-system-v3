"""Deterministic steering telemetry (v2.8).

The point of these tests is that every signal is a *fact about counters*, not
an LLM opinion.  Each detector is therefore exercised by feeding it the exact
sequence that should trip it, and its absence in the sequence just before the
threshold.
"""

import json

import pytest

import steering
from steering import (DOOM_LOOP, STALL, STRUCTURAL_BLOCK, AttemptRecord,
                      SteeringMonitor, diff_fingerprint, signature_tokens)


@pytest.fixture(autouse=True)
def _clean_monitors():
    steering.reset_monitors()
    yield
    steering.reset_monitors()


@pytest.fixture
def ws(tmp_path):
    (tmp_path / ".system").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _fail(monitor, project, sig, task="T-1", dhash="h1", duration=10.0):
    monitor.record_attempt(project, AttemptRecord(
        task_id=task, outcome="CHANGES_REQUESTED", attempt=1,
        signature=sig, diff_hash=dhash, duration_s=duration))


# ---------------------------------------------------------------- signatures
def test_signature_tokens_drops_the_shared_ERR_prefix():
    assert signature_tokens("ERR_PIPELINE_WIRING") == ["PIPELINE", "WIRING"]
    # "ERR" is on every signature, so comparing it would make all failures
    # look identical.
    assert "ERR" not in signature_tokens("ERR_FOO")
    assert signature_tokens(None) == []


def test_diff_fingerprint_ignores_hunk_line_numbers():
    a = """--- a/src/x.py
+++ b/src/x.py
@@ -10,3 +10,3 @@
-old
+new
"""
    b = """--- a/src/x.py
+++ b/src/x.py
@@ -99,3 +99,3 @@
-old
+new
"""
    assert diff_fingerprint(a) == diff_fingerprint(b)
    c = a.replace("+new", "+different")
    assert diff_fingerprint(a) != diff_fingerprint(c)


# ------------------------------------------------------------------ doom loop
def test_repeated_signature_trips_doom_loop_at_threshold(ws):
    m = SteeringMonitor(ws)
    # Distinct diff hashes, so only the *signature* counter can trip here.
    for i in range(steering.DOOM_LOOP_SIGNATURE_N - 1):
        _fail(m, "p", "ERR_QUANT_SIGNAL_MATH", dhash=f"h{i}")
    assert not [s for s in m.evaluate("p") if s.kind == DOOM_LOOP
                and "same diff" not in s.detail]

    _fail(m, "p", "ERR_QUANT_SIGNAL_MATH", dhash="h9")
    signals = [s for s in m.evaluate("p") if s.kind == DOOM_LOOP]
    assert signals, "third consecutive same-signature failure must trip"
    assert signals[0].severity == "high"


def test_a_success_clears_the_signature_counters(ws):
    """Progress resets the doom-loop evidence -- no sticky false positives."""
    m = SteeringMonitor(ws)
    for i in range(steering.DOOM_LOOP_SIGNATURE_N - 1):
        _fail(m, "p", "ERR_A_B", dhash=f"d{i}")
    m.record_attempt("p", AttemptRecord(task_id="T-1", outcome="APPROVED",
                                        diff_hash="ok"))
    _fail(m, "p", "ERR_A_B", dhash="z")
    assert not [s for s in m.evaluate("p") if s.kind == DOOM_LOOP
                and "same diff" not in s.detail]


def test_same_diff_twice_trips_doom_loop(ws):
    m = SteeringMonitor(ws)
    _fail(m, "p", "ERR_X_Y", dhash="identical")
    assert not [s for s in m.evaluate("p") if s.kind == DOOM_LOOP]
    _fail(m, "p", "ERR_X_Y", dhash="identical")
    signals = [s for s in m.evaluate("p") if s.kind == DOOM_LOOP]
    assert signals and "same diff" in signals[0].detail


# --------------------------------------------------------------------- stall
def test_stall_requires_consecutive_unchanged_boards(ws):
    m = SteeringMonitor(ws)
    board = [{"id": "T-1", "status": "pending", "retry_count": 0}]
    # The first observation establishes the baseline and cannot itself be a
    # change, so N unchanged observations need N+1 calls.
    for _ in range(steering.STALL_CYCLES_N):
        m.note_board("p", board)
    assert not [s for s in m.evaluate("p") if s.kind == STALL]

    m.note_board("p", board)
    assert [s for s in m.evaluate("p") if s.kind == STALL]


def test_board_progress_resets_the_stall_counter(ws):
    m = SteeringMonitor(ws)
    for _ in range(steering.STALL_CYCLES_N):
        m.note_board("p", [{"id": "T-1", "status": "pending", "retry_count": 0}])
    # A retry bump is progress, even though status is unchanged.
    m.note_board("p", [{"id": "T-1", "status": "pending", "retry_count": 1}])
    assert not [s for s in m.evaluate("p") if s.kind == STALL]


# ---------------------------------------------------------- structural block
def test_dependency_deadlock_is_structural(ws):
    m = SteeringMonitor(ws)
    board = [
        {"id": "T-1", "status": "pending", "dependencies": ["T-2"]},
        {"id": "T-2", "status": "pending", "dependencies": ["T-1"]},
    ]
    signals = [s for s in m.evaluate("p", board) if s.kind == STRUCTURAL_BLOCK]
    assert signals and "deadlocked" in signals[0].detail


def test_a_runnable_task_is_not_a_deadlock(ws):
    m = SteeringMonitor(ws)
    board = [
        {"id": "T-1", "status": "pending", "dependencies": []},
        {"id": "T-2", "status": "pending", "dependencies": ["T-1"]},
    ]
    assert not [s for s in m.evaluate("p", board) if s.kind == STRUCTURAL_BLOCK]


def test_shared_signature_token_across_tasks_is_systemic(ws):
    """Three tasks failing the same way is one design problem, not three bugs.

    This is the distinction that matters for steering: retrying each task
    individually cannot fix a shared root cause.
    """
    m = SteeringMonitor(ws)
    for i, task in enumerate(("T-1", "T-2", "T-3")):
        m.record_attempt("p", AttemptRecord(
            task_id=task, outcome="ESCALATED",
            signature=f"ERR_SCHEMA_{task}_MAPPING", diff_hash=f"d{i}"))
    signals = [s for s in m.evaluate("p", []) if s.kind == STRUCTURAL_BLOCK]
    assert signals
    assert any("design-level" in s.detail for s in signals)


def test_distinct_signatures_are_not_structural(ws):
    m = SteeringMonitor(ws)
    for i, task in enumerate(("T-1", "T-2", "T-3")):
        m.record_attempt("p", AttemptRecord(
            task_id=task, outcome="ESCALATED",
            signature=f"ERR_UNIQUE{i}_THING", diff_hash=f"d{i}"))
    assert not [s for s in m.evaluate("p", []) if s.kind == STRUCTURAL_BLOCK]


# ---------------------------------------------------------------- persistence
def test_state_survives_a_restart(ws):
    """86 orchestrator launches for 101 tasks: in-memory detection is useless."""
    first = SteeringMonitor(ws)
    for _ in range(steering.DOOM_LOOP_SIGNATURE_N):
        _fail(first, "p", "ERR_PERSIST_ME", dhash=f"h{_}")
    assert [s for s in first.evaluate("p") if s.kind == DOOM_LOOP]

    reopened = SteeringMonitor(ws)  # fresh instance, same disk state
    assert [s for s in reopened.evaluate("p") if s.kind == DOOM_LOOP]
    raw = json.loads((ws / ".system" / "steering_state.json").read_text(
        encoding="utf-8"))
    assert "p" in raw["projects"]


def test_state_is_bounded(ws):
    m = SteeringMonitor(ws)
    for i in range(400):
        _fail(m, "p", f"ERR_S{i}_X", task=f"T-{i}", dhash=f"d{i}")
    raw = json.loads((ws / ".system" / "steering_state.json").read_text(
        encoding="utf-8"))
    assert len(raw["projects"]["p"]["attempts"]) <= 200


def test_reset_project_clears_accumulators(ws):
    m = SteeringMonitor(ws)
    for _ in range(steering.DOOM_LOOP_SIGNATURE_N):
        _fail(m, "p", "ERR_GONE", dhash=f"h{_}")
    m.reset_project("p")
    assert not [s for s in m.evaluate("p") if s.kind == DOOM_LOOP]


# ------------------------------------------------------------------ snapshot
def test_snapshot_is_bounded_and_json_safe(ws):
    m = SteeringMonitor(ws)
    for i in range(5):
        _fail(m, "p", f"ERR_S{i}_X", dhash=f"d{i}", duration=5.0 * i)
    snap = m.snapshot("p")
    json.dumps(snap)  # must not raise
    assert snap["attempts_recorded"] == 5
    assert len(snap["recent_escalations"]) <= 5
    assert snap["median_cycle_s"] >= 0


# -------------------------------------------------------- retired promotion
def test_pattern_promotion_is_no_longer_reachable(ws):
    """The v2.6 'learning' path stored task.objective verbatim and was never
    read back into any prompt, so promotion is removed from the hot path.

    Asserted at the *call site* rather than on the store, because the store
    may legitimately not exist in a fresh workspace.
    """
    import inspect

    import project_orchestrator

    source = inspect.getsource(project_orchestrator)
    assert "propose_and_verify_pattern" not in source, \
        "the orchestrator must not promote patterns any more"
    assert "decrement_pattern_confidence" not in source
    # Retry payloads must not carry pattern text into a Builder prompt.
    assert 'builder_context["patterns"]' not in source
    assert 'builder_context["anti_patterns"]' not in source


def test_querying_patterns_does_not_delete_them(ws):
    """Regression: a *read* used to run the 30-day TTL prune.

    That is how 11 of 23 promoted patterns disappeared with no write intent.
    """
    from shared_workspace_engine import SharedWorkspaceEngine

    (ws / ".shared_workspace").mkdir(parents=True, exist_ok=True)
    store = ws / ".shared_workspace" / "pattern_memory.json"
    payload = {"patterns": [{
        "id": "P-0001", "signature": "ERR_OLD_THING", "fix_summary": "s",
        "diff_hash": "d", "confidence": 1.0,
        "created_at": "2020-01-01T00:00:00Z", "last_used": "2020-01-01T00:00:00Z",
    }]}
    store.write_text(json.dumps(payload), encoding="utf-8")

    SharedWorkspaceEngine(ws).query_patterns("ERR_OLD_THING")
    after = json.loads(store.read_text(encoding="utf-8"))
    assert len(after["patterns"]) == 1, "a read must not prune the store"