"""T-RENDERER contract -- hand-pinned 2026-10-17 (operator closeout).

The machine-staged draft (llm cache 6fef) pinned a parse_diagnosis that
returned ``{"diagnosis": text}``/None and a render_table of loose task-row
dicts named "projects_data" -- neither matches the goal nor the objective.
This file is the contract of record, coherent with the loader output that
T-DATALOADER builds and with T-CLI's consumption of render_table:

* format_task_row: columns [id, TITLE, STATUS, ATTEMPTS, DEPS], joined " | ";
  status verbatim only for done|pending|failed, else '?'; newlines in id and
  title replaced by spaces; ATTEMPTS is retry_count (default 0); DEPS is
  ", "-joined dependencies or '-'.
* parse_diagnosis: only feedback STARTING with "DIAGNOSIS:" and matching
  'DIAGNOSIS: <cause> (confidence=<confidence>) remedy=<remedy>' parses, to
  {'cause', 'confidence', 'remedy'}; anything else (including None) -> None,
  and a None never produces a line.
* render_table consumes load_projects' output (the loader contract): an
  (name, exception) tuple is skipped silently; each readable board gets a
  '== <name> ==' header, one row per task, then one indented diagnosis line
  per parsed diagnosis; lines are '\n'-joined without a trailing newline.
"""

import importlib.util

import pytest

if importlib.util.find_spec("src.boardscan") is None:
    pytest.skip("implementation not written yet", allow_module_level=True)
try:
    from src.boardscan import format_task_row, parse_diagnosis, render_table
except ImportError:
    pytest.skip("renderer symbols not implemented yet",
                allow_module_level=True)


def test_format_task_row_happy_path():
    row = format_task_row({"id": "T-1", "title": "Fix bug",
                           "status": "done", "retry_count": 2,
                           "dependencies": ["T-0", "T-1"]})
    assert row == "T-1 | Fix bug | done | 2 | T-0, T-1"


def test_format_task_row_status_verbatim_only_for_known():
    assert format_task_row({"id": "A", "title": "T", "status": "pending"}
                           ).split(" | ")[2] == "pending"
    assert format_task_row({"id": "A", "title": "T", "status": "failed"}
                           ).split(" | ")[2] == "failed"
    # Unknown and uppercase statuses are not the board's vocabulary.
    assert format_task_row({"id": "A", "title": "T", "status": "DONE"}
                           ).split(" | ")[2] == "?"
    assert format_task_row({"id": "A", "title": "T", "status": "zzz"}
                           ).split(" | ")[2] == "?"


def test_format_task_row_newlines_become_spaces():
    row = format_task_row({"id": "X\nY", "title": "Line1\r\nLine2",
                           "retry_count": 0, "dependencies": []})
    assert row == "X Y | Line1 Line2 | ? | 0 | -"


def test_format_task_row_defaults():
    row = format_task_row({"id": "A", "title": "Only"})
    assert row == "A | Only | ? | 0 | -"


def test_parse_diagnosis_well_formed():
    parsed = parse_diagnosis(
        "DIAGNOSIS: spec_wrong (confidence=0.98) remedy=needs_human")
    assert parsed == {"cause": "spec_wrong", "confidence": "0.98",
                      "remedy": "needs_human"}


def test_parse_diagnosis_cause_keeps_inner_spaces():
    parsed = parse_diagnosis(
        "DIAGNOSIS: spec wrong v2 (confidence=1.0) remedy=fix_the_pin")
    assert parsed == {"cause": "spec wrong v2", "confidence": "1.0",
                      "remedy": "fix_the_pin"}


def test_parse_diagnosis_rejects_everything_else():
    assert parse_diagnosis(None) is None
    assert parse_diagnosis("") is None
    assert parse_diagnosis("no diagnosis here") is None
    assert parse_diagnosis("DIAGNOSIS: missing-the-grammar") is None
    assert parse_diagnosis(
        "prefix DIAGNOSIS: x (confidence=1) remedy=y") is None


def test_render_table_sections_rows_and_diagnoses():
    diagnosis = ("DIAGNOSIS: spec_wrong (confidence=0.98) "
                 "remedy=needs_human")
    board = {"tasks": [
        {"id": "T-1", "title": "Plain task", "status": "done",
         "retry_count": 0, "dependencies": []},
        {"id": "T-2", "title": "Escalated task", "status": "failed",
         "retry_count": 3, "dependencies": ["T-1"],
         "last_feedback": diagnosis},
    ]}
    output = render_table([("a_proj", board)])

    lines = output.split("\n")
    assert lines == [
        "== a_proj ==",
        "T-1 | Plain task | done | 0 | -",
        "T-2 | Escalated task | failed | 3 | T-1",
        "  DIAGNOSIS: spec_wrong (confidence=0.98) remedy=needs_human",
    ]


def test_render_table_skips_unreadable_and_boards_without_tasks():
    output = render_table([("bad", ValueError("boom")),
                           ("empty", {}),
                           ("good", {"tasks": [
                               {"id": "T-1", "title": "X"}]})])
    assert output.split("\n") == [
        "== empty ==",
        "== good ==",
        "T-1 | X | ? | 0 | -",
    ]


def test_render_table_empty_input():
    assert render_table([]) == ""
