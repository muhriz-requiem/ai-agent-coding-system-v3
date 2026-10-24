"""A task's authored test path of record is discoverable by the critic.

2026-10-17, follow-up to the coerce-side test-path disambiguation (6ba6819):
renaming T-RENDERER's test to a disambiguated path was invisible to the
spec critic -- it resolves task tests from target_files plus the
``test_<module>.py`` convention only -- so the task was graded against a
missing file and blocked at scheduling time forever (observed live: the
boardscan build spent its entire 2100 s budget in the blocked loop).

The task now carries ``authored_test_path`` end-to-end: coerce_task sets
it, board entries round-trip it via model_dump, and ``_test_files_for``
honors it above conventions, existence optional (tests-first).
"""

import spec_author
import spec_critic
import validation


def _draft(task_id: str, title: str, test_path: str | None = None) -> dict:
    draft = {"id": task_id, "title": title, "objective": f"{title} work",
             "target_files": ["src/boardscan.py"]}
    if test_path:
        draft["test_path"] = test_path
    return draft


def test_coerce_task_records_authored_test_path(tmp_path):
    plan = spec_author.coerce_plan(
        {"tasks": [_draft("T-1", "Loader"), _draft("T-2", "Renderer")]},
        goal="g", existing_ids=[], project_root=tmp_path, require_tests=False)
    tasks = [t.task for t in plan.usable]
    assert tasks[0].authored_test_path == "tests/test_boardscan.py"
    assert tasks[1].authored_test_path.startswith(
        "tests/test_boardscan_"), tasks[1].authored_test_path
    # board entries round-trip the path (model_dump).
    entries = spec_author.board_entries(plan)
    assert entries[0]["authored_test_path"] == tasks[0].authored_test_path


def test_critic_resolves_deduped_authored_test(tmp_path):
    """The failed build's exact shape: derived name missing, authored file
    present under the disambiguated name -- resolution must find it."""
    authored_rel = "tests/test_boardscan_task_row_formatting.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / authored_rel).write_text(
        "def test_row_formatting():\n    assert True\n", encoding="utf-8")
    task = validation.TaskSpec(
        id="T-RENDERER", title="Renderer",
        objective="format rows", target_files=["src/boardscan.py"],
        requires_new_tests=True,
        authored_test_path=authored_rel)
    tests = spec_critic._test_files_for(task, tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in tests]
    assert rels[0] == authored_rel, rels


def test_critic_named_path_semantics_preserved(tmp_path):
    """Existence is not required -- same declared contract as explicitly
    named paths in target_files."""
    task = validation.TaskSpec(
        id="T-X", title="T", objective="work",
        target_files=["src/mod.py"], requires_new_tests=True,
        authored_test_path="tests/test_mod_extra.py")
    tests = spec_critic._test_files_for(task, tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in tests]
    assert rels[0] == "tests/test_mod_extra.py"  # may not exist yet


def test_legacy_task_without_field_still_conventional(tmp_path):
    """Old boards (field None) keep the unchanged convention behaviour."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_mod.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    task = validation.TaskSpec(
        id="T-Y", title="T", objective="work",
        target_files=["src/mod.py"], requires_new_tests=True)
    assert task.authored_test_path is None
    tests = spec_critic._test_files_for(task, tmp_path)
    rels = [p.relative_to(tmp_path).as_posix() for p in tests]
    assert rels == ["tests/test_mod.py"], rels
