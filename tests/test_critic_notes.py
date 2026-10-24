"""Pins for the authored-plan repair loop (v2.9.1 notes lane).

Blocking critic findings used to be only ``log_event``ed: the next
authoring attempt could not see why attempt 1 was rejected and reproduced
the same defect.  Observed on slatepress -- a plan with no task owning
``src/slatepress/__init__.py`` was blocked, and the retry proposed the
same board again.

Three contracts are held here:

1. ``_note_blocking``: a blocking finding rides in ``plan.notes`` exactly
   once per kind (the critic may surface the same board-level defect under
   several tasks or across repeated calls), capped at four findings, and
   no-ops without a plan.
2. ``_critique_authored``: the deterministic tier's blocking verdict is
   final on that task and the LLM tier is not spent re-deriving it; the
   semantic tier's blocking verdict lands in the notes as well.
3. ``_try_author_from_goal``: when every authored task is held, the notes
   survive to disk marked ``gave_up: False`` -- and the next attempt
   receives them verbatim as its ``repair`` context.  That is the loop
   closing; without it the author retries blind.
"""
import asyncio
import json
from types import SimpleNamespace

import config
import spec_author
from project_orchestrator import ProjectOrchestrator

PO = ProjectOrchestrator


class _Finding:
    """A critic finding: ``kind``/``detail`` plus the ``to_dict()`` that the
    ``spec_author_task_blocked`` log line serializes."""

    def __init__(self, kind="package_init_unowned",
                 detail="no task owns src/demo/__init__.py"):
        self.kind = kind
        self.detail = detail

    def to_dict(self):
        return {"kind": self.kind, "detail": self.detail}


def _finding(**kwargs):
    return _Finding(**kwargs)


def _report(blocking):
    """The deterministic tier's report shape: only ``blocking`` is consulted."""
    return SimpleNamespace(blocking=list(blocking))


# ---------------------------------------------------------------------------
# 1. _note_blocking
# ---------------------------------------------------------------------------

def test_blocking_note_is_appended_once_per_kind():
    """Same kind, different phrasings -> one note.  New kind -> appended."""
    plan = SimpleNamespace(notes=[])
    PO._note_blocking(plan, _report([_finding(detail="first phrasing")]))
    PO._note_blocking(plan, _report([_finding(detail="second phrasing")]))
    PO._note_blocking(plan, _report([_finding(kind="other_kind", detail="x")]))
    critic_notes = [n for n in plan.notes
                    if n.startswith("critic blocked activation [")]
    assert critic_notes == [
        "critic blocked activation [package_init_unowned]: first phrasing",
        "critic blocked activation [other_kind]: x",
    ], critic_notes


def test_note_is_clipped_to_400_chars():
    plan = SimpleNamespace(notes=[])
    PO._note_blocking(plan, _report([_finding(detail="x" * 500)]))
    note = plan.notes[-1]
    assert len(note) == 400, "the detail is clipped, never spilled raw"
    assert note.startswith(
        "critic blocked activation [package_init_unowned]: ")


def test_note_blocking_caps_at_four_findings():
    plan = SimpleNamespace(notes=[])
    PO._note_blocking(plan, _report(
        [_finding(kind=f"kind_{i}", detail="d") for i in range(5)]))
    critic_notes = [n for n in plan.notes
                    if n.startswith("critic blocked activation [")]
    assert len(critic_notes) == 4, critic_notes


def test_note_blocking_without_plan_is_a_noop():
    # A planless call must not raise: the gate holds without anywhere to note.
    PO._note_blocking(None, _report([_finding()]))
    # So must an object with no notes attribute at all (F3: the reader is
    # defensive, so the writer must be too -- _critique_authored never raises).
    PO._note_blocking(SimpleNamespace(), _report([_finding()]))


def test_note_blocking_tolerates_unrelated_and_malformed_notes():
    plan = SimpleNamespace(notes=[
        "critic blocked activation [no-closing-bracket",  # never re-parsed
        "some prior author note -- with a separator",
    ])
    PO._note_blocking(plan, _report([_finding()]))
    critic_notes = [n for n in plan.notes
                    if n.startswith("critic blocked activation [")]
    assert len(critic_notes) == 2, critic_notes  # one malformed + one fresh


def test_note_blocking_dedupes_a_kind_containing_bracket():
    """A kind may itself contain ``]``: recovery delimits on ``]: ``."""
    plan = SimpleNamespace(notes=[])
    PO._note_blocking(plan, _report(
        [_finding(kind="owning]kind", detail="d1")]))
    PO._note_blocking(plan, _report(
        [_finding(kind="owning]kind", detail="d2")]))
    critic_notes = [n for n in plan.notes
                    if n.startswith("critic blocked activation [")]
    assert len(critic_notes) == 1, critic_notes
    assert critic_notes[0] == "critic blocked activation [owning]kind]: d1"


# ---------------------------------------------------------------------------
# 2. _critique_authored routing
# ---------------------------------------------------------------------------

def test_deterministic_block_is_final_and_does_not_spend_the_llm_call(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(config, "SPEC_CRITIC_ENABLED", True)
    monkeypatch.setattr(config, "SPEC_CRITIC_LLM_ENABLED", True)
    llm_calls = []

    class FakeCritic:
        async def run(self, task, root):
            llm_calls.append(task)
            raise AssertionError("deterministic hold must end the review")

    orc = PO(workspace_root=tmp_path, project_name="demo")
    orc.spec_critic = FakeCritic()
    deterministic = _report([_finding()])
    orc._deterministic_spec_report = lambda task: deterministic

    plan = SimpleNamespace(notes=[])
    got = asyncio.run(
        orc._critique_authored(SimpleNamespace(id="T-1"), plan))
    assert got is deterministic
    assert llm_calls == [], "a held verdict is not re-derived"
    assert plan.notes[-1] == (
        "critic blocked activation [package_init_unowned]: "
        "no task owns src/demo/__init__.py"), plan.notes


def test_semantic_block_also_lands_in_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SPEC_CRITIC_ENABLED", True)
    monkeypatch.setattr(config, "SPEC_CRITIC_LLM_ENABLED", True)

    class FakeFinds:
        async def run(self, task, root):
            finding = _finding(kind="missing_task_owns_module")
            return SimpleNamespace(ok=False, blocking=[finding],
                                   findings=[finding],
                                   to_dict=lambda: {"ok": False})

    orc = PO(workspace_root=tmp_path, project_name="demo")
    orc.spec_critic = FakeFinds()
    orc._deterministic_spec_report = lambda task: _report([])

    plan = SimpleNamespace(notes=[])
    full = asyncio.run(
        orc._critique_authored(SimpleNamespace(id="T-2"), plan))
    assert full.blocking[0].kind == "missing_task_owns_module"
    assert plan.notes[-1] == (
        "critic blocked activation [missing_task_owns_module]: "
        "no task owns src/demo/__init__.py"), plan.notes
    assert orc.status["spec_critic"] == {"ok": False}


def test_critique_authored_without_plan_still_gates(tmp_path, monkeypatch):
    """The plan argument is optional; gating without it must still hold."""
    monkeypatch.setattr(config, "SPEC_CRITIC_ENABLED", True)
    orc = PO(workspace_root=tmp_path, project_name="demo")
    blocking = _report([_finding()])
    orc._deterministic_spec_report = lambda task: blocking
    got = asyncio.run(orc._critique_authored(SimpleNamespace(id="T-3")))
    assert got is blocking


# ---------------------------------------------------------------------------
# 3. the loop closes: notes -> disk -> next attempt's repair context
# ---------------------------------------------------------------------------

class FakeAuthor:
    """Records every call; emits one authored, critic-holdable task."""

    def __init__(self, notes_per_attempt: int = 1):
        self.notes_per_attempt = notes_per_attempt
        self.calls = []

    async def run(self, goal, **kwargs):
        attempt = kwargs["attempt"]
        self.calls.append({"attempt": attempt,
                           "repair": list(kwargs.get("repair") or [])})
        authored = SimpleNamespace(
            task=SimpleNamespace(id=f"T-{attempt}-A"), usable=True)
        notes = [f"attempt {attempt} note {i}"
                 for i in range(self.notes_per_attempt)]
        return SimpleNamespace(usable=[authored], tasks=[], notes=notes)


def _empty_demo_project(tmp_path):
    """A project with an empty board and a goal, exactly as --new-project."""
    root = tmp_path / "projects" / "demo"
    (root / ".agent").mkdir(parents=True, exist_ok=True)
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": []}), encoding="utf-8")
    (root / "goal.txt").write_text("Build a small site.", encoding="utf-8")
    return root


def test_blocked_board_persists_notes_and_next_attempt_receives_repair(
        tmp_path, monkeypatch):
    _empty_demo_project(tmp_path)
    monkeypatch.setattr(config, "SPEC_AUTHOR_ENABLED", True)
    monkeypatch.setattr(config, "SPEC_AUTHOR_MAX_ATTEMPTS", 2)
    monkeypatch.setattr(config, "SPEC_AUTHOR_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(config, "SPEC_CRITIC_ENABLED", True)
    monkeypatch.setattr(config, "SPEC_CRITIC_LLM_ENABLED", False)
    # Stage 2 (tests-first) is not the subject here: keep it off the disk.
    monkeypatch.setattr(spec_author, "write_tests", lambda *a, **k: [])

    orc = PO(workspace_root=tmp_path, project_name="demo")
    fake_author = FakeAuthor()
    orc.spec_author = fake_author
    orc._deterministic_spec_report = lambda task: _report([
        _finding(detail="nobody owns the package initializer")])

    first = asyncio.run(orc._try_author_from_goal())
    assert first is False  # every task was held by the critic

    state = orc._load_author_state()
    assert state["attempts"] == 1
    assert state["gave_up"] is False, "a critic hold is convergable, not fatal"
    assert state["last_notes"] == [
        "attempt 1 note 0",
        "critic blocked activation [package_init_unowned]: "
        "nobody owns the package initializer",
    ], state["last_notes"]
    assert "Spec Critic" in state["remedy"]

    second = asyncio.run(orc._try_author_from_goal())
    assert second is False
    assert len(fake_author.calls) == 2
    second_call = fake_author.calls[1]
    assert second_call["attempt"] == 2
    # The loop closing: attempt 2 receives attempt 1's rejection notes.
    assert second_call["repair"] == [str(n) for n in state["last_notes"]]
    state2 = orc._load_author_state()
    assert state2["attempts"] == 2
    # F1 pin: attempt 2 spent the whole MAX_ATTEMPTS=2 budget, so the
    # persisted state must read terminal -- like the sibling
    # ``not plan.usable`` path -- instead of telling the operator to wait
    # on a retry that can never run.
    assert state2["gave_up"] is True
    assert "exhausted" in state2["remedy"]

    # The budget still holds: no third authoring call after MAX_ATTEMPTS=2.
    third = asyncio.run(orc._try_author_from_goal())
    assert third is False
    assert len(fake_author.calls) == 2


def test_critic_note_survives_a_note_heavy_plan(tmp_path, monkeypatch):
    """The persisted repair context is the TAIL of the notes, not the head.

    Author notes are emitted first and the critic appends after them, so a
    note-heavy plan pushes the blocking reasons past a head slice's window.
    Persisting ``notes[:6]`` there would hand the retry zero critic context
    and reproduce the blind-retry behavior this lane exists to stop.
    """
    _empty_demo_project(tmp_path)
    monkeypatch.setattr(config, "SPEC_AUTHOR_ENABLED", True)
    monkeypatch.setattr(config, "SPEC_AUTHOR_MAX_ATTEMPTS", 3)
    monkeypatch.setattr(config, "SPEC_AUTHOR_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(config, "SPEC_CRITIC_ENABLED", True)
    monkeypatch.setattr(config, "SPEC_CRITIC_LLM_ENABLED", False)
    monkeypatch.setattr(spec_author, "write_tests", lambda *a, **k: [])

    orc = PO(workspace_root=tmp_path, project_name="demo")
    fake_author = FakeAuthor(notes_per_attempt=8)
    orc.spec_author = fake_author
    orc._deterministic_spec_report = lambda task: _report([
        _finding(detail="nobody owns the package initializer")])

    assert asyncio.run(orc._try_author_from_goal()) is False

    state = orc._load_author_state()
    # 8 author notes + 1 critic note: 9 entries, so a [:6] head slice would
    # keep note 0 and lose the critic line entirely.
    critic = [n for n in state["last_notes"]
              if n.startswith("critic blocked activation [")]
    assert len(critic) == 1, state["last_notes"]
    assert not any("note 0" in n for n in state["last_notes"]), (
        "a head slice would keep the oldest author note instead")
    assert state["gave_up"] is False, "attempt 1 of 3 still converges"

    # And the retry receives the surviving tail, critic reasons included.
    second = asyncio.run(orc._try_author_from_goal())
    assert second is False
    repair = fake_author.calls[1]["repair"]
    assert any(n.startswith("critic blocked activation [") for n in repair), \
        repair
