"""T-DATALOADER contract -- hand-pinned 2026-10-17 (operator closeout).

The machine-staged draft (llm cache 55a2fae) asserted SystemExit(1) on the
first unreadable board, contradicting its own objective ("continue other
projects"); the spec critic blocked it correctly. This file pins the real
contract of record instead:

* ``load_projects(root, project=None) -> list[tuple[str, object]]`` where the
  value is the parsed board dict for readable boards and the raised
  Exception for unreadable ones (invalid JSON is DATA, never a mid-scan
  abort).
* missing project directories (root itself, or a --project target) raise
  SystemExit(2) -- the CLI surfaces it, it is not an in-loop exit code.
* boards are read-only: nothing here ever writes.
"""

import importlib.util
import json

import pytest

if importlib.util.find_spec("src.boardscan") is None:
    pytest.skip("implementation not written yet", allow_module_level=True)
try:
    from src.boardscan import load_projects
except ImportError:
    pytest.skip("loader symbol not implemented yet", allow_module_level=True)


def _make_board(tmp_path, name, payload=None, raw=None):
    """Create <tmp_path>/projects/<name>/.agent/task_board.json."""
    agent = tmp_path / "projects" / name / ".agent"
    agent.mkdir(parents=True, exist_ok=True)
    board = agent / "task_board.json"
    if raw is not None:
        board.write_text(raw, encoding="utf-8")
    else:
        board.write_text(json.dumps(payload or {"tasks": []}),
                         encoding="utf-8")
    return board


def test_load_projects_returns_sorted_read_boards(tmp_path):
    _make_board(tmp_path, "b_proj", {"tasks": [{"id": "T-2"}]})
    _make_board(tmp_path, "a_proj", {"tasks": [{"id": "T-1"}]})
    # A directory without a board must not shift the sorted names.
    (tmp_path / "projects" / "zz_empty").mkdir(parents=True)

    result = load_projects(str(tmp_path))

    assert result == [("a_proj", {"tasks": [{"id": "T-1"}]}),
                      ("b_proj", {"tasks": [{"id": "T-2"}]})]


def test_load_projects_skips_missing_boards(tmp_path):
    _make_board(tmp_path, "only", {"tasks": []})
    (tmp_path / "projects" / "no_board" / "not_agent").mkdir(parents=True)

    result = load_projects(str(tmp_path))

    assert [name for name, _value in result] == ["only"]


def test_load_projects_invalid_json_is_mtuple_not_abort(tmp_path):
    _make_board(tmp_path, "a_good", {"tasks": [{"id": "T-1"}]})
    _make_board(tmp_path, "z_bad", raw="{ not json")

    result = load_projects(str(tmp_path))

    assert [name for name, _value in result] == ["a_good", "z_bad"]
    assert isinstance(result[0][1], dict)
    assert isinstance(result[1][1], Exception)  # captured, scan continued


def test_load_projects_missing_root_exits_two(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        load_projects(str(tmp_path / "nope"))
    assert excinfo.value.code == 2


def test_load_projects_filter_keeps_one(tmp_path):
    _make_board(tmp_path, "a_proj", {"tasks": []})
    _make_board(tmp_path, "b_proj", {"tasks": [{"id": "T-2"}]})

    result = load_projects(str(tmp_path), project="b_proj")

    assert result == [("b_proj", {"tasks": [{"id": "T-2"}]})]


def test_load_projects_unknown_project_exits_two_with_stderr(
        tmp_path, capsys):
    _make_board(tmp_path, "a_proj", {"tasks": []})

    with pytest.raises(SystemExit) as excinfo:
        load_projects(str(tmp_path), project="nope")

    assert excinfo.value.code == 2
    assert "no such project: nope" in capsys.readouterr().err
