"""v2.9 Phase G: partitioning a board into independent lines.

The claim under test is narrow and falsifiable: *line assignment is graph
colouring, not a model decision*.  Two tasks may share a line only if they
cannot conflict, and both ways of conflicting are readable from the board alone --
a shared ``target_files`` entry, and a dependency that has to resolve first.
Neither needs a model, so neither gets one.

The measured justification, from the real boards in this workspace:

* ``logtriage`` -- 6 tasks, 6 distinct files, **0 shared**. Splits cleanly into
  2 or 3 balanced lines with nothing serialised.
* ``project_main`` -- 20 tasks, 21 files, **3 shared files**
  (``src/config.py``, ``src/cli.py``, ``src/pipeline/smoke.py``). Those pairs
  must serialise, and it is colour placement that enforces it rather than luck.

Two real bugs in the partitioner were found by running it against those boards,
and both are pinned here because both produced *plausible, wrong* output rather
than an error:

1. **Dependencies were treated as placement constraints.**  That collapsed the
   fully parallel logtriage board into a single line at every line count.  A
   dependency constrains ordering, not placement -- the scheduler's existing
   dependency gate handles waiting.
2. **Components were ordered independently and concatenated.**  When several
   components share a line, a dependency pointing from an early component to a
   late one (``TASK-7011`` depends on ``TASK-7010B``) produced an invalid
   execution order.  The line now is re-ordered topologically as a whole.
"""

from __future__ import annotations

import json
from pathlib import Path


import lines


def t(task_id: str, files, deps=()) -> dict:
    return {"id": task_id, "title": task_id,
            "objective": f"do {task_id}",
            "target_files": list(files), "dependencies": list(deps),
            "status": "pending"}


# ============================================ the parallel case, measured
def test_a_board_with_no_shared_files_splits_fully():
    """The logtriage shape: 6 tasks, 6 files, nothing in common.

    If this does not split, the partitioner has no parallelism to offer and
    every wall-clock claim about Phase G is empty.
    """
    board = [t(f"T-{i:03d}", [f"src/triage/mod{i}.py"]) for i in range(1, 7)]
    result = lines.partition(board, max_lines=3)
    assert result.count == 3, "a conflict-free board did not split"
    assert not result.serialised, "nothing needed serialising"
    assert not lines.validate(result, board)
    # And it is actually parallel: no task waits on another line's work.
    assert all(not line.internal_deps for line in result.lines)


def test_balanced_distribution():
    board = [t(f"T-{i:03d}", [f"src/m{i}.py"]) for i in range(6)]
    result = lines.partition(board, max_lines=3)
    sizes = sorted(len(line) for line in result.lines)
    assert sizes == [2, 2, 2], f"poorly balanced: {sizes}"


def test_line_count_one_is_exactly_the_old_behaviour():
    """LINE_COUNT=1 must be today's behaviour, so it is the safe default."""
    board = [t(f"T-{i:03d}", [f"src/m{i}.py"]) for i in range(4)]
    result = lines.partition(board, max_lines=1)
    assert result.count == 1
    assert len(result.lines[0]) == 4
    assert any("LINE_COUNT=1" in note for note in result.notes)


def test_more_lines_than_tasks_does_not_invent_lines():
    result = lines.partition([t("T-1", ["src/a.py"])], max_lines=4)
    assert result.count == 1, "empty lines were emitted"


def test_empty_board():
    result = lines.partition([], max_lines=3)
    assert result.count == 0
    assert result.notes


# ================================== safety: shared files must serialise
def test_tasks_sharing_a_file_land_on_the_same_line():
    """The project_main shape, and the whole reason colouring exists.

    Two concurrent Builders writing one file race the snapshot chain: the loser's
    diff applies to a tree it never saw.
    """
    board = [t("T-1", ["src/config.py"]),
             t("T-2", ["src/cli.py"]),
             t("T-3", ["src/config.py"]),
             t("T-4", ["src/cli.py"])]
    result = lines.partition(board, max_lines=3)
    assert not lines.validate(result, board), "an unsafe partition was produced"
    assert result.serialised, "the shared-file pairs were not recorded"
    shared = {path for _, _, path in result.serialised}
    assert shared == {"src/config.py", "src/cli.py"}


def test_transitive_file_sharing_is_held_together():
    """A->B share a file, B->C share another; the ends do not overlap.

    Without transitive closure A and C would be placed independently and could
    still end up racing through B.
    """
    board = [t("A", ["src/one.py"]),
             t("B", ["src/one.py", "src/two.py"]),
             t("C", ["src/two.py"])]
    result = lines.partition(board, max_lines=2)
    assert not lines.validate(result, board)
    assert result.count == 1, (
        "a transitively connected component was split across lines")


def test_test_files_do_not_count_as_conflicts():
    """Two tasks may both have tests.

    Each authored test is written once at authoring time and then frozen, so
    test content is never written concurrently by two Builders.  Treating
    ``tests/`` as a shared resource would serialise the whole board for no
    reason -- and on a tests-first board, every task has tests.
    """
    board = [t("T-1", ["src/a.py", "tests/test_a.py"]),
             t("T-2", ["src/b.py", "tests/test_b.py"]),
             t("T-3", ["src/c.py", "tests/test_c.py"])]
    result = lines.partition(board, max_lines=3)
    assert not result.serialised, "test files were treated as conflicts"
    assert not lines.validate(result, board)


# ============================ dependencies: ordering, not placement
def test_dependencies_do_not_collapse_the_partition():
    """The first version of the partitioner unioned on dependencies.

    That is a category error -- a dependency says *when*, not *where* -- and it
    cost the entire parallelism opportunity: logtriage has 8 dependency edges
    across 6 tasks and collapsed to one line at every line count.
    """
    board = [t("T-1", ["src/a.py"]),
             t("T-2", ["src/b.py"], deps=["T-1"]),
             t("T-3", ["src/c.py"])]
    result = lines.partition(board, max_lines=2)
    assert result.count == 2, (
        "a dependency edge collapsed independent tasks onto one line")


def test_intra_line_order_respects_dependencies():
    """Even across components that share a line.

    The second real bug: components were ordered independently and
    concatenated, so on project_main ``TASK-7011`` (an early component) landed
    before ``TASK-7010B`` (a late one) despite depending on it.  The line is now
    re-ordered topologically as a whole.
    """
    board = [t("EARLY", ["src/a.py"]),
             t("LATE", ["src/b.py"]),
             t("NEEDS_LATE", ["src/c.py"], deps=["LATE"]),
             t("NEEDS_EARLY", ["src/d.py"], deps=["EARLY"])]
    # Force one line so every edge is intra-line.
    result = lines.partition(board, max_lines=1)
    order = {task_id: position
             for position, task_id in enumerate(result.lines[0].task_ids)}
    assert order["EARLY"] < order["NEEDS_EARLY"]
    assert order["LATE"] < order["NEEDS_LATE"]
    assert not lines.validate(result, board), lines.validate(result, board)


def test_cross_line_dependencies_are_allowed():
    """Waiting across lines is the scheduler's job, not a partition error."""
    board = [t("T-1", ["src/a.py"]),
             t("T-2", ["src/b.py"]),
             t("T-3", ["src/c.py"], deps=["T-1", "T-2"])]
    result = lines.partition(board, max_lines=3)
    assert not lines.validate(result, board), (
        "a legitimate cross-line dependency was reported as a conflict")


def test_internal_deps_are_reported_per_line():
    board = [t("A", ["src/a.py"]), t("B", ["src/b.py"], deps=["A"])]
    result = lines.partition(board, max_lines=1)
    assert result.lines[0].internal_deps.get("B") == ["A"]


def test_dependency_cycle_does_not_hang():
    """Reported, not looped on. The scheduler refuses the cycle anyway."""
    board = [t("A", ["src/a.py"], deps=["B"]),
             t("B", ["src/b.py"], deps=["A"])]
    result = lines.partition(board, max_lines=1)
    assert result.notes, "a cycle produced no note"
    assert any("cycle" in note for note in result.notes)


def test_dependency_on_a_missing_task_is_ignored():
    """A dangling id must not invent a component or crash the ordering."""
    board = [t("A", ["src/a.py"], deps=["GHOST"])]
    result = lines.partition(board, max_lines=1)
    assert result.count == 1
    assert not lines.validate(result, board)


# ============================================== determinism and the validator
def test_partition_is_deterministic():
    """A partition that changed between runs would make every log diff useless."""
    board = [t(f"T-{i:03d}", [f"src/m{i % 4}.py"]) for i in range(9)]
    first = lines.partition(board, max_lines=3)
    second = lines.partition(board, max_lines=3)
    assert ([line.task_ids for line in first.lines]
            == [line.task_ids for line in second.lines])


def test_board_order_is_preserved_within_equal_components():
    board = [t("Z", ["src/z.py"]), t("A", ["src/a.py"])]
    result = lines.partition(board, max_lines=1)
    assert result.lines[0].task_ids == ["Z", "A"], (
        "ties were broken by id rather than board order, making the result "
        "depend on naming")


def test_validator_catches_a_forged_partition():
    """The validator re-derives everything; it must reject a bad partition."""
    board = [t("T-1", ["src/shared.py"]), t("T-2", ["src/shared.py"])]
    forged = lines.Partition(lines=[
        lines.Line(index=0, task_ids=["T-1"]),
        lines.Line(index=1, task_ids=["T-2"]),
    ])
    problems = lines.validate(forged, board)
    assert problems, "a partition with a shared-file race passed validation"
    assert any("shared.py" in problem for problem in problems)


def test_validator_catches_a_missing_task():
    board = [t("T-1", ["src/a.py"]), t("T-2", ["src/b.py"])]
    partial = lines.Partition(lines=[lines.Line(index=0, task_ids=["T-1"])])
    problems = lines.validate(partial, board)
    assert any("not placed" in problem for problem in problems)


def test_line_for_lookup():
    board = [t("T-1", ["src/a.py"]), t("T-2", ["src/b.py"])]
    result = lines.partition(board, max_lines=2)
    assert result.line_for("T-1") is not None
    assert result.line_for("NOPE") is None


def test_describe_is_readable():
    board = [t(f"T-{i:03d}", [f"src/m{i}.py"]) for i in range(3)]
    text = lines.describe(lines.partition(board, max_lines=2))
    assert "line 0" in text and "line 1" in text


# ============================================ against the real boards
def test_every_real_board_partitions_validly():
    """Run against the boards in this workspace, not against fixtures.

    Fixtures written to pass prove nothing about a partitioner; these are the
    boards that produced 766k tokens and 12 escalations.
    """
    root = Path(__file__).resolve().parents[1]
    boards = sorted((root / "projects").glob("*/.agent/task_board.json"))
    assert boards, "no boards found -- this test has silently stopped testing"
    for path in boards:
        board = json.loads(path.read_text(encoding="utf-8"))
        tasks = board.get("tasks", [])
        for count in (1, 2, 3):
            result = lines.partition(tasks, max_lines=count)
            problems = lines.validate(result, tasks)
            assert not problems, (
                f"{path.parents[1].name} at {count} line(s): {problems[:3]}")
            assert sum(len(line) for line in result.lines) == len(tasks), (
                f"{path.parents[1].name}: tasks lost in the partition")


def test_logtriage_really_is_fully_parallel():
    """The specific claim behind the 2x wall-clock estimate."""
    path = (Path(__file__).resolve().parents[1] / "projects" / "logtriage"
            / ".agent" / "task_board.json")
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    result = lines.partition(tasks, max_lines=3)
    assert not result.serialised, (
        "logtriage was supposed to need no serialisation at all")
    assert result.count == 3
    assert not lines.validate(result, tasks)


def test_project_main_serialises_exactly_its_three_shared_files():
    path = (Path(__file__).resolve().parents[1] / "projects" / "project_main"
            / ".agent" / "task_board.json")
    tasks = json.loads(path.read_text(encoding="utf-8"))["tasks"]
    result = lines.partition(tasks, max_lines=3)
    shared = {path for _, _, path in result.serialised}
    assert shared == {"src/config.py", "src/cli.py", "src/pipeline/smoke.py"}, (
        f"unexpected shared-file set: {sorted(shared)}")
    assert not lines.validate(result, tasks)
