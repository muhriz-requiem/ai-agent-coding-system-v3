"""unresolvable_import honors the objective's own scope -- 2026-10-17.

Follow-up to the authored-test-path chain of fixes: with the target module
ALREADY existing (task N-1 built it), a tests-first task N was blocked at
scheduling time forever because its authored test imported the symbols task
N is defined to create. Observed live on boardscan: T-RENDERER blocked the
whole 2100 s run on ``unresolvable_import`` for format_task_row /
parse_diagnosis / render_table while its objective named exactly those.

The exemption is narrow on purpose: a symbol counts as in-scope only when
the OBJECTIVE ITSELF names it (word-boundary match). The dogfood typo case
(test imports ``_analyse_file_text``, objective names ``analyse_file_text``)
still fires -- pinned by the frozen suite and re-pinned here at the
boundary.

The tasks here carry ``authored_test_path`` the way real authored boardscan
tasks do (f5e4889); without it, ``_test_files_for`` resolves the
conventional ``test_boardscan.py`` and the check would examine nothing.
"""

import spec_critic
import validation

TEST_BODY_OK = (
    "from src.boardscan import format_task_row, parse_diagnosis\n\n\n"
    "def test_x():\n    assert parse_diagnosis(None) is None\n")
TEST_BODY_TYPO = (
    "from src.boardscan import _parse_diagnosis\n\n\n"
    "def test_x():\n    assert _parse_diagnosis(1) == 1\n")


def _boardscan_module(tmp_path):
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/boardscan.py").write_text(
        "def load_projects(root, project=None):\n    return []\n",
        encoding="utf-8")


def _task(tmp_path, objective, test_relative, test_body):
    _boardscan_module(tmp_path)
    test_file = tmp_path / test_relative
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text(test_body, encoding="utf-8")
    return validation.TaskSpec(
        id="T-RENDERER", title="renderer", objective=objective,
        target_files=["src/boardscan.py"], requires_new_tests=True,
        authored_test_path=test_relative)


def test_symbol_named_by_objective_is_in_scope(tmp_path):
    task = _task(
        tmp_path,
        "Add format_task_row and parse_diagnosis to src/boardscan.py.",
        "tests/test_rendering.py",
        TEST_BODY_OK)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "unresolvable_import" not in {f.kind for f in report.findings}, (
        report.summary())


def test_neighbor_spelling_outside_objective_still_fires(tmp_path):
    task = _task(
        tmp_path,
        "Add parse_diagnosis to src/boardscan.py.",
        "tests/test_rendering.py",
        TEST_BODY_TYPO)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "unresolvable_import" in {f.kind for f in report.findings}, (
        report.summary())


def test_no_objective_keeps_prior_semantics(tmp_path):
    task = _task(
        tmp_path, "", "tests/test_rendering.py",
        "from src.boardscan import missing_symbol\n")
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "unresolvable_import" in {f.kind for f in report.findings}, (
        report.summary())
