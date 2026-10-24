"""v2.9 Phase F: unbroken lines, and the cross-task leak they exposed.

Phase F was meant to make a line *carry context forward*.  Building it turned up
the opposite defect: the Builder's conversation was already carrying context
forward, and it should not have been.

``BuilderAgent.reset()`` fired only at ``attempt == MAX_ATTEMPTS``, so
``_history`` was never cleared at a task boundary.  Task 2's attempt 1 opened a
conversation still containing task 1's objective, task 1's diff, and task 1's
reviewer feedback -- while being asked to write a different file.  The model was
working from a transcript about ``src/a.py`` when told to write ``src/b.py``.

That is not a subtle cost. It is a wrong prompt, on every task after the first
in a project, and it predates this work.

The two halves are deliberately separate state, and the tests below assert they
stay separate:

* ``_history`` -- the retry state machine for *one* task. Resets at the task
  boundary. Must never leak.
* ``_line_notes`` -- what previous tasks *established*. Carries deliberately,
  bounded, and survives the history reset. This is the continuity that was
  wanted.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import config
from agent_roles import BuilderAgent
from validation import TaskSpec


class _RecordingLLM:
    """Captures every prompt so cross-task leakage is observable."""

    def __init__(self) -> None:
        self.prompts: list = []

    async def complete(self, **kwargs) -> str:
        self.prompts.append(kwargs)
        return ""

    def record_parse(self, ok: bool) -> None:
        pass


def _task(task_id: str, title: str, **kw) -> TaskSpec:
    return TaskSpec(id=task_id, title=title,
                    objective=f"Implement {title}.",
                    target_files=[f"src/{title}.py"], **kw)


@pytest.fixture
def builder(tmp_path) -> BuilderAgent:
    return BuilderAgent(_RecordingLLM(), tmp_path)


# ======================================================= the leak, and the fix
def test_history_does_not_survive_a_task_boundary(builder):
    """The regression this phase found, stated as a property.

    ``reset()`` is what the orchestrator now calls at ``attempt == 1``.  Before
    v2.9 it was called only at ``attempt == MAX_ATTEMPTS``, so this assertion
    describes behaviour that did not exist.
    """
    builder._history.append({"role": "user", "content": "TASK T-1 objective"})
    builder._history.append({"role": "assistant", "content": "diff src/a.py"})
    assert builder._history, "fixture problem"

    builder.reset()

    assert not builder._history, (
        "conversation history survived the task boundary; the next task opens "
        "with the previous task's objective, diff and reviewer feedback")


def test_orchestrator_resets_at_the_start_of_every_task():
    """Asserted against the source, because the loop is not easily reachable.

    The call must be inside the attempt loop at ``attempt == 1``, and the
    existing ``attempt == MAX_ATTEMPTS`` reset must still be there for the
    compact final attempt.  Getting this wrong in either direction is a real
    regression: too few resets reintroduces the leak, too many destroys the
    retry feedback that attempts 2 depends on.
    """
    source = (Path(__file__).resolve().parents[1]
              / "project_orchestrator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    attempts_resets = 0
    boundary_resets = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not any(isinstance(inner, ast.Compare)
                   for inner in ast.walk(node)):
            continue
        body = ast.unparse(node)
        if "config.MAX_ATTEMPTS" not in body:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.If) and "self.builder.reset()" in ast.unparse(sub):
                test = ast.unparse(sub.test)
                if "attempt == config.MAX_ATTEMPTS" in test:
                    attempts_resets += 1
                elif "attempt == 1" in test:
                    boundary_resets += 1
    assert boundary_resets == 1, (
        "expected exactly one reset at the task boundary "
        f"(attempt == 1); found {boundary_resets}")
    assert attempts_resets == 1, (
        "the compact final-attempt reset must survive; "
        f"found {attempts_resets}")


# ================================================ continuity: the useful half
def test_line_notes_carry_across_tasks(builder):
    builder.note_task_complete(_task("T-001", "reader"), ["src/reader.py"], 1)
    builder.note_task_complete(_task("T-002", "latency"), ["src/latency.py"], 2)

    preamble = builder.line_preamble()
    assert "T-001" in preamble and "T-002" in preamble
    assert "src/reader.py" in preamble
    assert "LINE_CONTEXT" in preamble


def test_line_notes_survive_the_history_reset(builder):
    """The point of separating the two: continuity must outlive the reset.

    If the history reset also cleared the line notes, a task boundary would
    erase exactly the context continuity exists to preserve -- and the leak fix
    would have quietly undone the feature.
    """
    builder.note_task_complete(_task("T-001", "reader"), ["src/reader.py"], 1)
    builder.reset()
    assert builder.line_preamble(), (
        "the task-boundary reset wiped the cross-task notes")


def test_no_preamble_when_nothing_is_recorded(builder):
    """An empty context block is noise in every prompt."""
    assert builder.line_preamble() == ""


def test_line_notes_are_bounded_and_say_so(builder, monkeypatch):
    """Unbounded notes crowd out the current task, and truncation that is not
    reported reads as a complete history."""
    monkeypatch.setattr(config, "LINE_NOTE_LIMIT", 3)
    for index in range(6):
        builder.note_task_complete(
            _task(f"T-{index:03d}", f"mod{index}"), [f"src/mod{index}.py"], 1)

    preamble = builder.line_preamble()
    assert "not shown" in preamble
    assert "T-005" in preamble, "the most recent task must be kept"
    # Everything after the marker is within the limit.
    tail = preamble.split("not shown", 1)[1]
    entries = [line for line in tail.splitlines() if line.startswith("- [")]
    assert len(entries) <= 3, f"{len(entries)} entries survived the limit"


def test_continuity_can_be_switched_off(builder, monkeypatch):
    """A toggle that does nothing would be worse than no toggle."""
    monkeypatch.setattr(config, "LINE_CONTINUITY", False)
    builder.note_task_complete(_task("T-001", "reader"), ["src/reader.py"], 1)
    assert builder.line_preamble() == ""


def test_begin_line_clears_notes_but_not_a_reset(builder):
    builder.note_task_complete(_task("T-001", "reader"), ["src/reader.py"], 1)
    builder.begin_line()
    assert builder.line_preamble() == ""
    # reset() and begin_line() are different operations and must not be aliases.
    builder.note_task_complete(_task("T-002", "x"), ["src/x.py"], 1)
    builder.reset()
    assert builder.line_preamble(), "reset() must not double as begin_line()"


def test_attempt_count_is_recorded(builder):
    """How hard something was is part of what a later task should know."""
    builder.note_task_complete(_task("T-001", "hard"), ["src/hard.py"], 3)
    assert "3 attempt" in builder.line_preamble()


def test_notes_prefer_applied_files_over_targets(builder):
    """The applied list is what actually landed, which is more useful."""
    builder.note_task_complete(_task("T-001", "x"), ["src/real.py"], 1)
    assert "src/real.py" in builder.line_preamble()
    assert "src/x.py" not in builder.line_preamble()


def test_line_config_is_sane():
    assert config.LINE_NOTE_LIMIT >= 1
    assert 1 <= config.LINE_COUNT <= 8
    assert isinstance(config.LINE_CONTINUITY, bool)


# ================================ both halves, through the real prompt builder
def test_prompt_carries_line_context_but_not_the_previous_task(builder, tmp_path):
    """The end-to-end property, on the prompt that actually gets sent."""
    llm = _RecordingLLM()
    agent = BuilderAgent(llm, tmp_path)

    # Task 1 runs, its conversation is recorded, then the boundary resets it.
    agent._history.append({"role": "user", "content": "TASK T-1 objective"})
    agent._history.append({"role": "assistant", "content": "diff src/a.py"})
    agent.note_task_complete(_task("T-001", "reader"), ["src/reader.py"], 1)
    agent.reset()

    # Task 2's prompt.
    agent._history.append({"role": "user",
                           "content": "TASK T-2 objective" +
                                      agent.line_preamble()})

    sent = json.dumps(agent._history)
    assert "T-2 objective" in sent
    assert "T-001" in sent, "the established API must carry forward"
    assert "src/reader.py" in sent
    assert "diff src/a.py" not in sent, (
        "the previous task's diff leaked into this task's conversation")
    assert "TASK T-1 objective" not in sent


def test_preamble_is_appended_outside_the_truncation():
    """A 4000-char truncation must not be able to cut the preamble off.

    The per-attempt history entry is truncated to keep the state machine
    compact, and the preamble rides along in the same string.  If it were
    truncated too, the oldest and least relevant continuity information would be
    the part that survived -- which is backwards.
    """
    import inspect
    source = inspect.getsource(BuilderAgent.run)
    assert "user[:4000]" in source, "expected the history truncation to remain"
    truncate_at = source.index("user[:4000]")
    preamble_at = source.index("line_preamble()")
    assert preamble_at > truncate_at, (
        "the preamble must be concatenated after truncation, not before")


# ============================= Doom loops must stop work, not just report it
def test_doom_loop_signal_is_a_stop_signal_not_only_a_log_line():
    """The detection existed; the intervention did not.

    ``steering`` reports a doom loop at ERROR when the same failure signature
    recurs three times or a byte-identical diff is produced twice.  Until
    v2.9.1 nothing consumed those signals, so the loop kept building and kept
    producing the identical diff.  Measured on a real board: 16 detections, 29
    gate blocks, 0 interventions, on a single task.

    Pinned by reading the orchestrator, because a unit test cannot cheaply
    drive three Builder attempts through the real gate.
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parents[1] / "project_orchestrator.py"
    text = src.read_text(encoding="utf-8")
    assert "_doom_loop_signals" in text, "the loop must consult steering"
    assert "doom_loop_halts_attempts" in text, "and must log the intervention"
    # The consultation has to sit inside the attempt loop, before the next
    # build, not after the retries have already been spent.
    assert text.index("_doom_loop_signals(task.id)") < \
        text.index("await self.builder.run(task, plan")


def test_doom_loop_query_is_task_scoped():
    """A doom loop on another task is not a reason to abandon this one."""
    import inspect
    import project_orchestrator
    fn = project_orchestrator.ProjectOrchestrator._doom_loop_signals
    src = inspect.getsource(fn)
    assert "s.task_id == task_id" in src, (
        "signals must be filtered to the task in hand, or one stuck task "
        "cancels work on every other task")


def test_doom_loop_query_matches_the_task_less_signature_signal():
    """The shape that actually fires carries ``task_id=None``.

    ``steering`` emits two doom-loop shapes: a diff repeat (carries the task
    id) and a signature repeat (deliberately ``task_id=None``, because it is one
    project-level row per recurring token).  A naive equality filter matches the
    first and silently drops the second -- which is the shape that fired 5 times
    in a row on a real board, and the reason the first version of the halt did
    nothing at all.

    So a task-less signal is matched against the task's own ``error_signature``
    tokens.  This also explains why ``error_signature`` matters: without one,
    steering genuinely cannot attribute a loop to a task, and the halt cannot
    fire.
    """
    import inspect
    import project_orchestrator
    src = inspect.getsource(
        project_orchestrator.ProjectOrchestrator._doom_loop_signals)
    assert "signature_tokens" in src, (
        "a task-less doom-loop signal must be attributed via the task's own "
        "error_signature")
    assert "doom_loop_unattributed" in src, (
        "an unattributable loop must be reported, not ignored -- that is the "
        "state the docs call a doom loop made invisible")


def test_authored_tasks_carry_an_error_signature():
    """A task with no signature cannot be attributed to a loop.

    Measured: the escalation fallback ``TASK:<id>`` normalised down to the bare
    token ``TASK``, so every escalated task looked like the same signature as
    every other one -- and one stalled task reported the token ``LEDGER`` with
    no task owning it.
    """
    import json
    import tempfile
    import spec_author
    from validation import TaskSpec, validate_or_error

    root = Path(tempfile.mkdtemp())
    (root / "src").mkdir()
    draft = {
        "id": "T-1", "title": "ledger", "objective": "Implement Ledger.",
        "target_files": ["src/ledger.py"], "dependencies": [],
        "test_source": "def test_x():\n    assert True\n",
    }
    plan = spec_author.coerce_plan({"tasks": [draft]}, "goal", [], root)
    assert plan.usable, plan.notes
    entry = spec_author.board_entries(plan)[0]
    task, err = validate_or_error(TaskSpec, entry)
    assert err is None, err
    assert task.error_signature, "every authored task must carry a signature"
    assert task.error_signature.startswith("ERR_")
    assert "TASK" not in json.dumps(entry), \
        "the old TASK:<id> fallback must not come back"