"""T-CLI contract -- hand-pinned 2026-10-17 (operator closeout).

main(argv: list[str] | None = None) -> int wiring argparse
[--root PATH] [--project NAME] [--json] over the loader + renderer:

* 0 = normal, 1 = at least one board unreadable (message per project on
  stderr, the readable sections still printed first / json still printed
  for readable projects only), 2 = missing root or unknown --project
  (SystemExit(2) from load_projects propagates; stderr carries
  "no such project: NAME"), argparse errors exit 2 by themselves.
* --json stdout parses as an array of readable projects only, each
  {"project", "tasks", "diagnoses"}; tasks carry
  [id, title, status, retry_count, dependencies] in board order; well-formed
  diagnoses carry [task_id, cause, confidence, remedy, raw].
"""

import importlib.util
import json

import pytest

if importlib.util.find_spec("src.boardscan") is None:
    pytest.skip("implementation not written yet", allow_module_level=True)
try:
    from src.boardscan import main
except ImportError:
    pytest.skip("cli symbol not implemented yet", allow_module_level=True)


DIAGNOSIS = "DIAGNOSIS: spec_wrong (confidence=0.98) remedy=needs_human"


def _workspace(tmp_path):
    """One healthy project, one unreadable board, one empty-board project."""
    good = tmp_path / "projects" / "a_good" / ".agent"
    good.mkdir(parents=True)
    (good / "task_board.json").write_text(json.dumps({
        "tasks": [
            {"id": "T-1", "title": "Plain", "status": "done",
             "retry_count": 0, "dependencies": []},
            {"id": "T-2", "title": "Escalated", "status": "failed",
             "retry_count": 3, "dependencies": ["T-1"],
             "last_feedback": DIAGNOSIS},
        ]}), encoding="utf-8")

    bad = tmp_path / "projects" / "m_bad" / ".agent"
    bad.mkdir(parents=True)
    (bad / "task_board.json").write_text("{ not json", encoding="utf-8")

    empty = tmp_path / "projects" / "z_empty" / ".agent"
    empty.mkdir(parents=True)
    (empty / "task_board.json").write_text("{\"tasks\": []}",
                                           encoding="utf-8")
    return tmp_path


def test_main_prints_sections_and_continues_past_unreadable(tmp_path,
                                                            capsys):
    root = _workspace(tmp_path)

    assert main(["--root", str(root)]) == 1

    captured = capsys.readouterr()
    # Readable projects render; the unreadable one is reported on stderr.
    assert "== a_good ==" in captured.out
    assert "T-1 | Plain | done | 0 | -" in captured.out
    assert "  DIAGNOSIS: spec_wrong (confidence=0.98) remedy=needs_human" \
        in captured.out
    assert "== m_bad ==" not in captured.out
    assert "m_bad" in captured.err
    assert captured.err.strip() != ""


def test_main_json_round_trip(tmp_path, capsys):
    healthy = tmp_path / "projects" / "a_good" / ".agent"
    healthy.mkdir(parents=True)
    (healthy / "task_board.json").write_text(json.dumps({
        "tasks": [
            {"id": "T-1", "title": "X", "status": "zzz",
             "retry_count": 0, "dependencies": []},
            {"id": "T-2", "title": "Y", "status": "pending",
             "retry_count": 1, "dependencies": ["T-1"],
             "last_feedback": DIAGNOSIS},
        ]}), encoding="utf-8")

    assert main(["--root", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == [{
        "project": "a_good",
        "tasks": [
            {"id": "T-1", "title": "X", "status": "zzz",
             "retry_count": 0, "dependencies": []},
            {"id": "T-2", "title": "Y", "status": "pending",
             "retry_count": 1, "dependencies": ["T-1"]},
        ],
        "diagnoses": [{
            "task_id": "T-2",
            "cause": "spec_wrong", "confidence": "0.98",
            "remedy": "needs_human",
            "raw": DIAGNOSIS,
        }],
    }]


def test_main_json_skips_unreadable_and_returns_one(tmp_path, capsys):
    root = _workspace(tmp_path)

    assert main(["--root", str(root), "--json"]) == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert [entry["project"] for entry in payload] == ["a_good", "z_empty"]
    assert "m_bad" in captured.err


def test_main_empty_projects_dir_returns_zero(tmp_path, capsys):
    (tmp_path / "projects").mkdir()
    assert main(["--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_main_unknown_project_exits_two_with_stderr(tmp_path, capsys):
    root = _workspace(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(root), "--project", "nope"])

    assert excinfo.value.code == 2
    assert "no such project: nope" in capsys.readouterr().err


def test_main_missing_root_exits_two(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(tmp_path / "nope")])
    assert excinfo.value.code == 2


def test_main_invalid_flag_exits_two(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--root", str(tmp_path), "--invalid-flag"])
    assert excinfo.value.code == 2
