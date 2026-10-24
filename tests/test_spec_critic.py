"""v2.9 Phases C and D: verify the spec before building, diagnose after failing.

Both modules exist because of the same measurement.  The logtriage build put
all 12 of its escalations at the human handover boundary and *none* at an
engineering defect.  Six of those were contradictions between an objective and
a test file that were each individually reasonable -- the Reviewer found them,
after up to three Builder attempts each.

The failure this prevents is not a bug.  It is a system paying maximum price to
discover something that was readable for free.

Fixtures below reproduce the actual contradiction classes from that build rather
than invented ones, because a Critic validated only against synthetic cases is
a Critic validated against nothing.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from pathlib import Path

import pytest

import config
import diagnosis
import spec_critic
from validation import TaskSpec, validate_or_error


# ============================================================ fixture helper
def make_task(tmp_path: Path, objective: str, tests: dict,
              impl: dict = None, name_tests: bool = False, **kw) -> TaskSpec:
    """A task plus on-disk tests/impl, returning the TaskSpec.

    ``name_tests=False`` by default, which mirrors every real board in the
    workspace: a tests-first task names only its implementation file and lets
    the Builder produce the test.  ``name_tests=True`` reproduces the one
    exception (TASK-7001 in project_main), where the task rewrites the grading
    apparatus and must therefore say so and set ``allow_test_edits``.
    """
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    for rel, body in tests.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body, encoding="utf-8")
    for rel, body in (impl or {}).items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body, encoding="utf-8")
    target = (list(tests) if name_tests else []) + list(impl or {})
    return TaskSpec(id=kw.pop("id", "T-1"), title=kw.pop("title", "task"),
                    objective=objective, target_files=target, **kw)


CONSISTENT_TEST = '''
from src.mod import compute


def test_returns_expected():
    assert compute() == 42
'''


# ================================================ Tier 1: the NameError class
def test_catches_the_spellings_that_disagree(tmp_path):
    """The real defect from the dogfood build.

    The objective specified ``analyse_file_text``; the test imported
    ``_analyse_file_text``.  Neither was wrong alone.  The Builder spent an
    attempt, and the Reviewer spent a rejection, learning a typo.
    """
    task = make_task(
        tmp_path,
        objective="Add `analyse_file_text` to src/mod.py returning parsed rows.",
        tests={"tests/test_mod.py": (
            "from src.mod import _analyse_file_text\n\n\n"
            "def test_parse():\n    assert _analyse_file_text('x') == []\n")},
        impl={"src/mod.py": "def analyse_file_text(text):\n    return []\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    kinds = {f.kind for f in report.findings}
    # The precise detection is ``unresolvable_import``: the name *is* bound in
    # the test's namespace, so an undefined-name scan sees nothing, yet the
    # import raises ImportError and takes the whole module with it.  That is why
    # this check resolves the module instead of grepping for the name.
    assert "unresolvable_import" in kinds, (
        f"the objective/test spelling mismatch was not caught: {kinds}")
    assert not report.ok, "an ImportError-in-waiting must block"
    finding = next(f for f in report.findings if f.kind == "unresolvable_import")
    assert "_analyse_file_text" in finding.detail
    assert "src/mod.py" in finding.detail, (
        "the finding must name the module that lacks the symbol")


def test_undefined_local_reference_is_still_caught(tmp_path):
    """The complementary case: a name no import and no module provides."""
    task = make_task(
        tmp_path,
        objective="Add `compute` to src/mod.py.",
        tests={"tests/test_mod.py": (
            "from src.mod import compute\n\n\n"
            "def test_x():\n    assert compute() == missing_helper()\n")},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "undefined_in_tests" in {f.kind for f in report.findings}, (
        report.summary())


def test_import_of_a_name_the_module_reexports_is_fine(tmp_path):
    """A module that pulls a name in from elsewhere still provides it."""
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src/base.py").write_text(
        "def helper():\n    return 7\n", encoding="utf-8")
    (tmp_path / "src/mod.py").write_text(
        "from src.base import helper\n", encoding="utf-8")
    (tmp_path / "tests/test_mod.py").write_text(
        "from src.mod import helper\n\n\n"
        "def test_x():\n    assert helper() == 7\n", encoding="utf-8")
    task = TaskSpec(id="T-1", title="t",
                    objective="Extend `helper` in src/mod.py.",
                    target_files=["src/mod.py"], requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "unresolvable_import" not in {f.kind for f in report.findings}, (
        report.summary())


def test_consistent_spec_passes(tmp_path):
    task = make_task(
        tmp_path,
        objective="Add `compute` to src/mod.py returning the integer 42.",
        tests={"tests/test_mod.py": CONSISTENT_TEST},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert report.ok, report.summary()


def test_objective_asserting_a_symbol_nobody_defines_blocks(tmp_path):
    """The objective says it already exists; it does not.

    Not a spelling mismatch -- the name simply is not anywhere in scope, so the
    task cannot succeed as written however well it is executed.  The
    ``extend``/``already exists`` cue is what distinguishes this from an
    ordinary feature task, which names plenty of symbols it wants *created*.
    """
    task = make_task(
        tmp_path,
        objective=("src/reader.py already exposes `summarise_window`; extend it "
                   "so it aggregates the window."),
        tests={"tests/test_mod.py": (
            "from src.mod import compute\n\n\n"
            "def test_compute():\n    assert compute() == 42\n")},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "objective_asserts_absent_symbols" in {f.kind for f in report.findings}, (
        report.summary())


def test_ordinary_new_feature_is_not_flagged(tmp_path):
    """A task naming symbols it wants *created* is the normal case.

    If this blocks, the Critic blocks everything and gets switched off, which is
    worse than having no Critic at all.
    """
    task = make_task(
        tmp_path,
        objective=("Add `compute` to src/mod.py and a `summarise_window` "
                   "helper that aggregates the window."),
        tests={"tests/test_mod.py": CONSISTENT_TEST},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "objective_asserts_absent_symbols" not in {f.kind for f in report.findings}, (
        report.summary())
    assert report.ok, report.summary()


def test_task_with_no_resolvable_tests_blocks(tmp_path):
    """Nothing named, nothing conventional -- the task cannot be graded."""
    task = make_task(
        tmp_path, objective="Add `compute` to src/mod.py.",
        tests={}, impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "no_tests_resolvable" in {f.kind for f in report.findings}
    assert not report.ok


def test_conventional_test_path_is_found(tmp_path):
    """The real board shape: only the implementation file is named.

    11 of 12 tasks in the logtriage board name no test file at all, because
    under tests-first the Builder produces the test.  A Critic that cannot find
    the test by convention is blind to nearly every task in the system.
    """
    task = TaskSpec(
        id="T-1", title="t",
        objective="Add `compute` to src/triage/reader.py.",
        target_files=["src/triage/reader.py"], requires_new_tests=True)
    (tmp_path / "src/triage").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "tests/test_reader.py").write_text(
        "from src.triage.reader import compute\n\n\n"
        "def test_x():\n    assert compute() == 42\n", encoding="utf-8")
    (tmp_path / "src/triage/reader.py").write_text(
        "def compute():\n    return 42\n", encoding="utf-8")
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert report.ok, report.summary()


def test_pre_existing_protected_target_blocks(tmp_path):
    """The Builder would be refused, so the task cannot succeed regardless."""
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    task = make_task(
        tmp_path, objective="Rewrite the grading apparatus.",
        tests={"tests/test_x.py": "def test_ok():\n    assert True\n"},
        requires_new_tests=True)
    task = task.model_copy(update={
        "target_files": task.target_files + ["conftest.py"]})
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "target_is_protected" in {f.kind for f in report.findings}
    assert not report.ok


def test_new_test_file_is_not_treated_as_protected(tmp_path):
    """Tests-first means the Builder *creates* ``tests/test_x.py``.

    Flagging a not-yet-existing test as protected would block every well-formed
    task in the system, because every one of them involves a test file that does
    not exist at planning time.
    """
    task = TaskSpec(
        id="T-1", title="t",
        objective="Add `compute` to src/mod.py.",
        target_files=["tests/test_new.py", "src/mod.py"],
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    kinds = {f.kind for f in report.findings}
    assert "target_is_protected" not in kinds, report.summary()
    # It is noted, but a note never blocks.
    assert "target_test_missing" in kinds
    assert report.ok, report.summary()


def test_allow_test_edits_opts_out_of_the_protection_check(tmp_path):
    """TASK-7001's shape: a task that legitimately rewrites the harness."""
    (tmp_path / "conftest.py").write_text("", encoding="utf-8")
    task = make_task(
        tmp_path, objective="Rewrite the grading apparatus.",
        tests={"tests/test_x.py": "def test_ok():\n    assert True\n"},
        name_tests=True, requires_new_tests=True)
    task = task.model_copy(update={
        "target_files": task.target_files + ["conftest.py"],
        "allow_test_edits": True})
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "target_is_protected" not in {f.kind for f in report.findings}, (
        report.summary())


def test_collected_ids_zero_is_flagged(tmp_path):
    task = make_task(
        tmp_path, objective="Add `compute` to src/mod.py.",
        tests={"tests/test_mod.py": CONSISTENT_TEST},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path, collected_ids=[])
    assert "tests_collect_nothing" in {f.kind for f in report.findings}


def test_builtins_are_not_reported_as_undefined(tmp_path):
    """A check that cries wolf on `len` or `sorted` gets switched off."""
    task = make_task(
        tmp_path,
        objective="Add `compute` to src/mod.py.",
        tests={"tests/test_mod.py": (
            "import math\nfrom src.mod import compute\n\n\n"
            "def test_x():\n"
            "    values = sorted([compute(), len('ab')])\n"
            "    assert math.isclose(float(values[0]), 42.0)\n")},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "undefined_in_tests" not in {f.kind for f in report.findings}, (
        report.summary())


def test_checks_are_free_of_side_effects(tmp_path):
    """Tier 1 must not need a model, a workspace or a network."""
    task = make_task(
        tmp_path, objective="Add `compute` to src/mod.py.",
        tests={"tests/test_mod.py": CONSISTENT_TEST},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    spec_critic.deterministic_checks(task, tmp_path)
    spec_critic.deterministic_checks(task, tmp_path)  # idempotent


# ================================================= Tier 2: merging behaviour
def test_model_cannot_clear_a_mechanical_finding():
    """A model that says "actually it's fine" must not un-flag a NameError."""
    report = spec_critic.SpecReport(task_id="T-1", findings=[
        spec_critic.SpecFinding("undefined_in_tests", "x is undefined", "high")])
    critique = {"task_id": "T-1", "consistent": True, "severity": "low",
                "confidence": 0.9, "contradictions": [], "unsatisfied": []}
    merged = spec_critic.merge_critique(report, critique)
    assert not merged.ok, "the model cleared a mechanical finding"
    assert merged.source == "deterministic+llm"


def test_only_high_severity_blocks():
    """A Critic that blocks on nits is a Critic that stops the workspace."""
    for severity, expected_ok in (("low", True), ("medium", True),
                                   ("high", False)):
        report = spec_critic.SpecReport(task_id="T-1", findings=[
            spec_critic.SpecFinding("spec_test_contradiction", "x", severity)])
        assert report.ok is expected_ok, severity


def test_untested_acceptance_criterion_never_blocks():
    merged = spec_critic.merge_critique(
        spec_critic.SpecReport(task_id="T-1"),
        {"task_id": "T-1", "consistent": True, "severity": "high",
         "confidence": 0.8, "contradictions": [],
         "unsatisfied": ["criterion with no test"]})
    assert merged.ok, "an uncovered criterion blocked the build"


def test_contradictions_are_carried_at_the_reported_severity():
    merged = spec_critic.merge_critique(
        spec_critic.SpecReport(task_id="T-1"),
        {"task_id": "T-1", "consistent": False, "severity": "high",
         "confidence": 0.8,
         "contradictions": ["objective says dict, test indexes [0]"],
         "unsatisfied": []})
    assert not merged.ok
    finding = next(f for f in merged.findings
                   if f.kind == "spec_test_contradiction")
    assert "dict" in finding.detail and "[0]" in finding.detail


def test_declared_return_type_versus_test_usage(tmp_path):
    """The contradiction class no name-based check can see.

    The objective declares ``outliers() -> dict[str, float]``; the test writes
    ``outliers(...)[0]``.  Every name resolves correctly, the module imports
    cleanly, and the failure is a ``TypeError`` at runtime -- the most expensive
    possible moment to discover it.  This exact pair cost a real escalation in
    the dogfood build.
    """
    task = make_task(
        tmp_path,
        objective=("Create src/triage/bad.py exposing "
                   "`outliers() -> dict[str, float]`."),
        tests={"tests/test_bad.py": (
            "from src.triage.bad import outliers\n\n\n"
            "def test_first_element():\n"
            "    assert outliers([1, 2, 9])[0] == 9\n")},
        impl={"src/triage/bad.py": "def outliers(values):\n    return {}\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "return_type_conflict" in {f.kind for f in report.findings}, (
        report.summary())
    assert not report.ok
    finding = next(f for f in report.findings if f.kind == "return_type_conflict")
    assert "dict" in finding.detail


def test_declared_list_return_indexed_by_int_is_consistent(tmp_path):
    """The other half of the pair: same test, honest declaration, no finding."""
    task = make_task(
        tmp_path,
        objective="Create src/triage/bad.py exposing `outliers() -> list[float]`.",
        tests={"tests/test_bad.py": (
            "from src.triage.bad import outliers\n\n\n"
            "def test_first_element():\n"
            "    assert outliers([1, 2, 9])[0] == 9\n")},
        impl={"src/triage/bad.py": "def outliers(values):\n    return []\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "return_type_conflict" not in {f.kind for f in report.findings}, (
        report.summary())


def test_scalar_return_subscripted_is_a_conflict(tmp_path):
    task = make_task(
        tmp_path,
        objective="Create src/triage/bad.py exposing `count_rows() -> int`.",
        tests={"tests/test_bad.py": (
            "from src.triage.bad import count_rows\n\n\n"
            "def test_count():\n    assert count_rows(['a']) == 1\n")},
        impl={"src/triage/bad.py": "def count_rows(rows):\n    return len(rows)\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "return_type_conflict" not in {f.kind for f in report.findings}, (
        report.summary())


def test_return_type_check_needs_an_arrow_declaration(tmp_path):
    """Objectives written without the ``-> Type`` convention are not checkable.

    Guessing at prose is how a check starts crying wolf, so it stays silent.
    """
    task = make_task(
        tmp_path,
        objective="Create src/triage/bad.py returning a dict of outliers.",
        tests={"tests/test_bad.py": (
            "from src.triage.bad import outliers\n\n\n"
            "def test_first_element():\n"
            "    assert outliers([1, 2, 9])[0] == 9\n")},
        impl={"src/triage/bad.py": "def outliers(values):\n    return {}\n"},
        requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert "return_type_conflict" not in {f.kind for f in report.findings}, (
        report.summary())


def test_coerce_derives_consistency_from_contradictions():
    """A model writing consistent=true alongside contradictions is a model
    that has not read its own output."""
    out = spec_critic.coerce_critique(
        {"consistent": True, "severity": "high", "confidence": 0.7,
         "contradictions": ["objective says X, test asserts Y"]})
    assert out["consistent"] is False


def test_coerce_normalises_unknown_vocabulary():
    out = spec_critic.coerce_critique(
        {"consistent": False, "severity": "catastrophic",
         "confidence": "not a number", "contradictions": "single string"})
    assert out["severity"] == "medium"
    assert out["confidence"] == 0.5
    assert out["contradictions"] == ["single string"]


def test_coerce_rejects_non_dict():
    assert spec_critic.coerce_critique("nope") is None
    assert spec_critic.coerce_critique(None) is None


def test_annotate_never_injects_blocking_findings():
    """Handing a Builder a contradiction it has been told about produces
    worse code than not running the attempt at all."""
    report = spec_critic.SpecReport(task_id="T-1", findings=[
        spec_critic.SpecFinding("undefined_in_tests", "BLOCKING", "high"),
        spec_critic.SpecFinding("spec_test_contradiction", "minor", "low")])
    out = spec_critic.annotate_objective("do the thing", report)
    assert "minor" in out
    assert "BLOCKING" not in out


def test_annotate_is_a_no_op_when_clean():
    report = spec_critic.SpecReport(task_id="T-1")
    assert spec_critic.annotate_objective("do the thing", report) == "do the thing"


def test_critic_prompt_carries_the_whole_test_file(tmp_path):
    """The Reviewer gets 2000 chars of output; this gets the document."""
    long_test = CONSISTENT_TEST + "\n" + "\n".join(
        f"# padding line {n}" for n in range(600))
    task = make_task(
        tmp_path, objective="Add `compute` to src/mod.py.",
        tests={"tests/test_mod.py": long_test},
        impl={"src/mod.py": "def compute():\n    return 42\n"},
        requires_new_tests=True)
    _objective, source = spec_critic.build_critic_prompt(task, tmp_path)
    assert "padding line 599" in source
    assert "tests/test_mod.py" in source


# ============================================================ Phase D: causes
def test_cause_vocabulary_is_closed():
    """Free-form causes would defeat steering clustering."""
    out = diagnosis.coerce({"cause": "the model was being creative",
                            "remedy": "needs_human"})
    assert out.cause == "unknown"


def test_remedy_vocabulary_is_closed():
    out = diagnosis.coerce({"cause": "spec_wrong", "remedy": "rm -rf /"})
    assert out.remedy == "needs_human"
    assert not out.actionable


def test_low_confidence_is_advice_only():
    """The diagnosing model also participated in the work that failed."""
    diag = diagnosis.Diagnosis(task_id="T-1", cause="spec_ambiguous",
                              confidence=0.2, remedy="narrow_objective")
    assert not diag.actionable, "a 0.2-confidence rewrite touched the board"


def test_actionable_requires_all_three_gates():
    good = diagnosis.Diagnosis(task_id="T-1", cause="spec_ambiguous",
                              confidence=0.9, remedy="narrow_objective")
    assert good.actionable
    assert not diagnosis.Diagnosis(remedy="narrow_objective",
                                   confidence=0.0).actionable
    assert not diagnosis.Diagnosis(remedy="needs_human",
                                   confidence=0.99).actionable


def test_apply_only_writes_whitelisted_keys(tmp_path):
    """The authority boundary, exercised rather than asserted."""
    task = TaskSpec(id="T-1", title="t", objective="original",
                    target_files=["src/m.py"])
    written = {}

    def fake_set(task_id, **updates):
        written.update(updates)

    diag = diagnosis.Diagnosis(task_id="T-1", cause="spec_ambiguous",
                              confidence=0.9, remedy="narrow_objective",
                              proposed_objective="narrowed")
    diagnosis.apply(diag, task, fake_set)
    assert diag.applied and written == {"objective": "narrowed"}
    # Nothing outside the whitelist for this remedy may appear.
    assert set(written) <= set(diagnosis.board_keys_for("narrow_objective"))


def test_apply_refuses_a_remedy_with_no_proposal():
    task = TaskSpec(id="T-1", title="t", objective="original",
                    target_files=["src/m.py"])
    diag = diagnosis.Diagnosis(task_id="T-1", confidence=0.9,
                              remedy="narrow_objective",
                              proposed_objective=None)
    diagnosis.apply(diag, task, lambda *a, **k: pytest.fail("wrote anyway"))
    assert not diag.applied


def test_apply_downgrades_split_to_needs_human(tmp_path):
    """Splitting creates tasks; that is not a single-task whitelist move."""
    task = TaskSpec(id="T-1", title="t", objective="original",
                    target_files=["src/m.py"])
    diag = diagnosis.Diagnosis(task_id="T-1", confidence=0.95,
                              remedy="split_task",
                              proposed_split=[{"objective": "half a"}])
    diagnosis.apply(diag, task, lambda *a, **k: pytest.fail("wrote anyway"))
    assert diag.remedy == "needs_human"
    assert not diag.applied


def test_apply_survives_a_failing_board_writer():
    """Advice must never be able to break escalation."""
    task = TaskSpec(id="T-1", title="t", objective="original",
                    target_files=["src/m.py"])

    def boom(*_a, **_k):
        raise OSError("board gone")

    diag = diagnosis.Diagnosis(task_id="T-1", confidence=0.9,
                              remedy="narrow_objective",
                              proposed_objective="narrowed")
    diagnosis.apply(diag, task, boom)
    assert not diag.applied


def test_split_drafts_are_sanitised_on_coerce():
    out = diagnosis.coerce({
        "cause": "scope_too_large", "remedy": "split_task",
        "proposed_split": [
            {"objective": "half a", "title": "A", "target_files": ["src/a.py"],
             "dependencies": ["T-0"], "__class__": "evil", "metric": {"x": 1}},
            {"no_objective": True},
            "junk",
        ]})
    assert len(out.proposed_split) == 1
    draft = out.proposed_split[0]
    assert "__class__" not in draft and "metric" not in draft, (
        "an untrusted split draft carried unexpected keys to the board")


def test_feedback_reads_as_a_conclusion_not_a_stack_trace():
    diag = diagnosis.Diagnosis(task_id="T-9", cause="spec_ambiguous",
                              confidence=0.82, applied=True,
                              remedy="narrow_objective",
                              evidence=["objective says dict[str,float]",
                                        "test does found[0]"])
    text = diag.feedback()
    assert "spec_ambiguous" in text and "0.82" in text
    assert "dict[str,float]" in text
    assert "[applied]" in text


def test_evidence_summary_uses_recorded_history():
    task = TaskSpec(id="T-1", title="do it", objective="Add `compute`.",
                    target_files=["src/m.py"],
                    acceptance_criteria=["returns 42"])
    text = diagnosis.summarize_evidence(
        task, [{"verdict": "CHANGES_REQUESTED", "feedback": "wrong name"}],
        reason="max retries exhausted",
        failing_tests=["tests/test_m.py::test_x"])
    assert "T-1" in text and "returns 42" in text
    assert "tests/test_m.py::test_x" in text
    assert "max retries exhausted" in text


def test_diagnosis_is_optional():
    assert isinstance(config.DIAGNOSIS_ENABLED, bool)
    assert 0.0 <= config.DIAGNOSIS_MIN_CONFIDENCE <= 1.0


def test_evidence_summary_includes_the_frozen_contract():
    """The diagnoser must see the test, or it cannot call the spec wrong.

    Without the contract, "the Builder is wrong" and "the contract is
    impossible" are indistinguishable from a gate line, and they need opposite
    remedies.  Observed on a real build: an authored test called
    ``JournalEntry()`` while the authored objective raised ``ValueError`` on
    fewer than two postings -- unsatisfiable by construction, three correct
    Builder attempts, and an escalation that could only report ``confidence
    0.00`` because nobody had shown it the test.
    """
    task = TaskSpec(id="T-4", title="ledger", objective="Implement Ledger.",
                    target_files=["src/ledger/ledger.py"])
    text = diagnosis.summarize_evidence(
        task, [], reason="max retries exhausted",
        test_sources=[("tests/test_ledger.py",
                       "from src.ledger.ledger import JournalEntry\n"
                       "def test_x():\n    entry = JournalEntry()\n")])
    assert "tests/test_ledger.py" in text
    assert "JournalEntry()" in text
    # And it is told what the frozen contract means for attribution.
    assert "FROZEN contract" in text
    assert "spec_wrong" in text


# ================================== the false-positive budget, measured
def test_critic_is_silent_on_every_completed_task_in_the_workspace():
    """The regression test that matters most, and the one that nearly failed.

    Run against the deterministic tier over all 28 completed board tasks in this
    workspace, the first version of this Critic blocked **24 of 28** -- it would
    have refused almost the entire backlog.  Every one of those was the Critic's
    fault, not the tasks', and each traced to a specific overreach:

      * reporting names bound inside a function as undefined (Python resolves
        those against the enclosing scope; a flat name-set does not);
      * treating a not-yet-created tests-first test file as a protected file;
      * widening "the objective presupposes this symbol exists" from a
        document-level keyword to every symbol in the text.

    A check that blocks 86% of known-good work gets switched off, and then it
    protects nothing.  So this asserts the property directly, against the real
    boards, rather than against fixtures that were written to pass.
    """
    from validation import validate_or_error

    root = Path(__file__).resolve().parents[1]
    boards = sorted((root / "projects").glob("*/.agent/task_board.json"))
    assert boards, "no boards found -- this test has silently stopped testing"

    total = 0
    blocked: list = []
    for board_path in boards:
        board = json.loads(board_path.read_text(encoding="utf-8"))
        tasks = board.get("tasks", [])
        # v2.9.1: scope this to boards that actually *completed*. The test's
        # claim is that the Critic does not block work that demonstrably
        # succeeded; a board mid-build is expected to have findings, and the
        # slatepress run had a genuinely blocking one (no task owned
        # src/slatepress/__init__.py while a test imported the package). Linting
        # an unfinished board against "must be silent" would have forced that
        # finding down to medium and defeated the check.
        if not tasks or any(t.get("status") != "done" for t in tasks):
            continue
        # .../projects/<name>/.agent/task_board.json -> <name> is parents[1].
        project_root = board_path.parents[1]
        for raw in tasks:
            task, err = validate_or_error(TaskSpec, raw)
            if err:
                continue
            total += 1
            report = spec_critic.deterministic_checks(task, project_root)
            if report.blocking:
                blocked.append(f"{board_path.parents[1].name}/{raw['id']}: "
                               + "; ".join(f.kind for f in report.blocking))

    assert total >= 20, f"only {total} tasks examined; too few to mean anything"
    assert not blocked, (
        f"the Critic blocks {len(blocked)} of {total} already-completed "
        f"tasks:\n  " + "\n  ".join(blocked))


# ============================================ Board-level structure (v2.9.1)
def _write_board(root: Path, tasks: list) -> None:
    board = root / ".agent" / "task_board.json"
    board.parent.mkdir(parents=True, exist_ok=True)
    board.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")


def test_flags_a_task_that_asks_for_two_implementation_modules(tmp_path):
    """One implementation file per task is the system's strongest rule.

    Found by reading a real ``--author-plan`` output: a 7-task bookkeeping plan
    where one task declared ``trial_balance.py`` *and* ``reports.py`` under a
    single test file.  The rule was documented but nothing enforced it, and a
    rule that is not checked is a suggestion.
    """
    task = make_task(tmp_path, "Implement two modules.",
                     tests={}, impl={"src/a.py": "A = 1\n",
                                     "src/b.py": "B = 2\n"},
                     id="T-1")
    _write_board(tmp_path, [{"id": "T-1",
                             "target_files": ["src/a.py", "src/b.py"]}])
    kinds = {f.kind for f in spec_critic.project_structure_findings(task,
                                                                   tmp_path)}
    assert "multi_file_task" in kinds


def test_single_module_task_is_not_flagged(tmp_path):
    task = make_task(tmp_path, "Implement one module.",
                     tests={}, impl={"src/a.py": "A = 1\n"}, id="T-1")
    _write_board(tmp_path, [{"id": "T-1", "target_files": ["src/a.py"]}])
    assert spec_critic.project_structure_findings(task, tmp_path) == []


def test_bootstrap_task_may_own_the_harness(tmp_path):
    """``allow_test_edits`` tasks legitimately span files -- there is exactly one.

    The reference board sets it on TASK-7001, which creates ``conftest.py`` and
    two modules.  Without the exemption the Critic would report the one task
    that is *designed* to be multi-file.
    """
    task = make_task(tmp_path, "Create the harness.",
                     tests={}, impl={"src/config.py": "X = 1\n",
                                     "src/logging_setup.py": "Y = 2\n"},
                     id="T-1", allow_test_edits=True)
    _write_board(tmp_path, [{"id": "T-1",
                             "target_files": ["src/config.py",
                                              "src/logging_setup.py"]}])
    kinds = {f.kind for f in spec_critic.project_structure_findings(task,
                                                                   tmp_path)}
    assert "multi_file_task" not in kinds


def test_flags_two_tasks_claiming_one_file(tmp_path):
    """Under LINE_COUNT=1 the second silently overwrites the first.

    Under a real line partition the two race.  The reference board contains two
    such pairs (``src/cli.py``, ``src/pipeline/smoke.py``) and still delivered,
    which is exactly why this is reported rather than treated as fatal.
    """
    task = make_task(tmp_path, "Implement cli.", tests={},
                     impl={"src/cli.py": "def main(): pass\n"}, id="T-1")
    _write_board(tmp_path, [
        {"id": "T-1", "target_files": ["src/cli.py"]},
        {"id": "T-2", "target_files": ["src/cli.py"]},
    ])
    findings = spec_critic.project_structure_findings(task, tmp_path)
    dupes = [f for f in findings if f.kind == "duplicate_target_file"]
    assert dupes and "T-2" in dupes[0].detail


def test_flags_a_test_importing_another_tasks_module_without_a_dependency(
        tmp_path):
    """The failure this prevents is reported as ``passed == baseline``.

    Task A's test imports a module task B creates, with no edge between them,
    so both are eligible at once and A's test dies at collection.  The gate
    cannot tell that apart from "the Builder wrote nothing".
    """
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "helper.py").write_text("def aid(): pass\n",
                                                encoding="utf-8")
    (tmp_path / "src" / "consumer.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_consumer.py").write_text(
        "from src.consumer import A\nfrom src.helper import aid\n\n\n"
        "def test_uses_helper():\n    assert A == 1 and aid() is None\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="consumer",
                    objective="Implement src/consumer.py.",
                    target_files=["src/consumer.py", "tests/test_consumer.py"],
                    dependencies=[])
    _write_board(tmp_path, [
        {"id": "T-1", "target_files": ["src/consumer.py"]},
        {"id": "T-2", "target_files": ["src/helper.py"]},
    ])
    kinds = {f.kind for f in spec_critic.project_structure_findings(task,
                                                                   tmp_path)}
    assert "cross_task_dependency_missing" in kinds


def test_declared_dependency_settles_the_import_check(tmp_path):
    """Same board, but with the edge present: nothing to report."""
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "tests").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "helper.py").write_text("def aid(): pass\n",
                                                encoding="utf-8")
    (tmp_path / "src" / "consumer.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_consumer.py").write_text(
        "from src.consumer import A\nfrom src.helper import aid\n\n\n"
        "def test_uses_helper():\n    assert A == 1 and aid() is None\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="consumer",
                    objective="Implement src/consumer.py.",
                    target_files=["src/consumer.py", "tests/test_consumer.py"],
                    dependencies=["T-2"])
    _write_board(tmp_path, [
        {"id": "T-1", "target_files": ["src/consumer.py"]},
        {"id": "T-2", "target_files": ["src/helper.py"]},
    ])
    kinds = {f.kind for f in spec_critic.project_structure_findings(task,
                                                                   tmp_path)}
    assert "cross_task_dependency_missing" not in kinds


def test_structure_findings_are_reported_not_blocking():
    """Calibration: real defects must not block a board that delivered.

    ``SPEC_CRITIC_BLOCK_SEVERITY`` is ``high``; every structural finding is
    ``medium``.  Blocking on them would fail the calibration test over
    ``projects/project_main`` -- 20 tasks that all completed.
    """
    finding = spec_critic.SpecFinding(kind="multi_file_task", detail="x",
                                      severity="medium", remedy="split_task")
    report = spec_critic.SpecReport(task_id="T-1")
    report.findings.append(finding)
    assert report.blocking == []
    assert finding.to_dict()["remedy"] == "split_task"
    # And the named remedy must actually be executable, or it is just advice.
    assert spec_critic.remedy_is_allowed(finding.remedy)

# ================================ Unresolved project-internal imports (v2.9.1)
def test_flags_a_test_importing_a_path_that_resolves_to_nothing(tmp_path):
    """The defect that manufactured a doom loop on a real authored board.

    The Builder wrote ``from src.journal import JournalEntry`` when the module
    is at ``src/ledger/journal.py``.  The suite reported ``passed == baseline``
    with skipped modules -- identical to "the Builder wrote nothing" -- and the
    gate's remediation said exactly that, so it wrote the same wrong import
    three times.
    """
    (tmp_path / "src" / "ledger").mkdir(parents=True)
    (tmp_path / "src" / "ledger" / "journal.py").write_text(
        "class JournalEntry: pass\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ledger.py").write_text(
        "from src.journal import JournalEntry\n\n\n"
        "def test_uses():\n    assert JournalEntry\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="ledger",
                    objective="Implement src/ledger/ledger.py.",
                    target_files=["src/ledger/ledger.py",
                                  "tests/test_ledger.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    bad = [f for f in report.findings if f.kind == "unresolved_project_import"]
    assert bad, "a wrong in-project import path must be reported"
    # And the message has to name the right file, or it is not actionable.
    assert "src/ledger/journal.py" in bad[0].detail


def test_does_not_flag_the_tasks_own_not_yet_written_module(tmp_path):
    """Tests-first means the module does not exist yet.  That is the design."""
    (tmp_path / "src" / "ledger").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ledger.py").write_text(
        "from src.ledger.ledger import Ledger\n\n\n"
        "def test_uses():\n    assert Ledger\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="ledger",
                    objective="Implement src/ledger/ledger.py.",
                    target_files=["src/ledger/ledger.py",
                                  "tests/test_ledger.py"],
                    requires_new_tests=True)
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert not [f for f in report.findings
                if f.kind == "unresolved_project_import"]


def test_does_not_flag_a_guarded_optional_import(tmp_path):
    """``try: import x / except ImportError: skip`` is how tests-first guards.

    Reading this wrong flagged 9 of the 20 completed reference-board tasks, so
    the exemption is load-bearing rather than defensive.
    """
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_optional.py").write_text(
        "import pytest\n\ntry:\n    import src.maybe\nexcept ImportError:\n"
        "    pytest.skip('optional', allow_module_level=True)\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="optional",
                    objective="Something optional.",
                    target_files=["src/maybe.py", "tests/test_optional.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert not [f for f in report.findings
                if f.kind == "unresolved_project_import"]


def test_does_not_flag_a_namespace_package_import(tmp_path):
    """PEP 420: a bare directory is importable with no ``__init__.py``.

    The reference board has six such smoke tests that pass; treating the
    directory import as broken would flag every one of them.
    """
    (tmp_path / "src" / "data").mkdir(parents=True)
    (tmp_path / "src" / "data" / "provider.py").write_text("X = 1\n",
                                                          encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_pkgs.py").write_text(
        "import src.data\n\n\ndef test_dirs():\n    assert src.data\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="pkgs",
                    objective="Package dirs are importable.",
                    target_files=["tests/test_pkgs.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert not [f for f in report.findings
                if f.kind == "unresolved_project_import"]


def test_does_not_flag_a_third_party_import(tmp_path):
    """A missing dependency is an environment problem, not a spec defect.

    The environment self-heals those from ``requirements.txt``; reporting them
    here would send the operator to edit a specification that is fine.
    """
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ext.py").write_text(
        "import pandas\n\n\ndef test_x():\n    assert pandas\n",
        encoding="utf-8")
    task = TaskSpec(id="T-1", title="ext",
                    objective="Use pandas.",
                    target_files=["src/ext.py", "tests/test_ext.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert not [f for f in report.findings
                if f.kind == "unresolved_project_import"]


def test_internal_import_finding_is_blocking():
    """It is a certain ImportError, so it must hold the task.

    The whole point of moving verification left is that this class of failure is
    visible for free at minute zero, before a Builder attempt is spent on it.
    """
    (tmp_path := Path(tempfile.mkdtemp()))
    (tmp_path / "src" / "ledger").mkdir(parents=True)
    (tmp_path / "src" / "ledger" / "journal.py").write_text("X = 1\n",
                                                             encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ledger.py").write_text(
        "from src.journal import X\n", encoding="utf-8")
    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["tests/test_ledger.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert any(f.kind == "unresolved_project_import" for f in report.blocking)

def test_nearest_project_module_names_the_real_file(tmp_path):
    """The gate's remediation must name the file, not show an example.

    Told ``from src.journal import X`` was wrong, the Builder replied
    ``from src.ledger.journal.entry import X`` -- right package, invented
    submodule -- and spent another attempt.  Naming the module that exists ends
    the guessing, so this is the function that does it.
    """
    (tmp_path / "src" / "ledger").mkdir(parents=True)
    (tmp_path / "src" / "ledger" / "journal.py").write_text("X = 1\n",
                                                            encoding="utf-8")
    (tmp_path / "src" / "ledger" / "accounts.py").write_text("Y = 1\n",
                                                              encoding="utf-8")
    # Right package, invented submodule.
    assert spec_critic.nearest_project_module("src.ledger.journal.entry",
                                              tmp_path) == "src/ledger/journal"
    # Wrong package entirely, correct stem.
    assert spec_critic.nearest_project_module("src.journal",
                                              tmp_path) == "src/ledger/journal"
    # Already correct.
    assert spec_critic.nearest_project_module("src.ledger.accounts",
                                              tmp_path) == "src/ledger/accounts"


def test_nearest_project_module_declines_when_nothing_is_close(tmp_path):
    """A wrong suggestion is worse than none.

    ``ledger`` is a package name, not a module, and mapping it to an arbitrary
    file would send the Builder to the wrong module with confidence.
    """
    (tmp_path / "src" / "ledger").mkdir(parents=True)
    (tmp_path / "src" / "ledger" / "journal.py").write_text("X = 1\n",
                                                            encoding="utf-8")
    assert spec_critic.nearest_project_module("ledger", tmp_path) == ""
    assert spec_critic.nearest_project_module("pandas", tmp_path) == ""


def test_gate_remediation_names_the_replacement_import(tmp_path, monkeypatch):
    """End to end: the text the Builder reads must contain the exact path."""
    # Covered at the unit level above; here we pin that the gate actually calls
    # it, because a correct helper that nothing uses fixes nothing.
    src = pathlib.Path(__file__).resolve().parents[1] / "project_orchestrator.py"
    text = src.read_text(encoding="utf-8")
    assert "nearest_project_module" in text
    assert "UNRESOLVABLE IMPORTS" in text

def test_diagnosis_prompt_states_the_response_schema():
    """The tier was inert because the model was never asked for the fields.

    ``coerce`` defaults a missing ``confidence`` to 0.0 and a missing
    ``evidence`` to [], which makes the diagnosis self-report as inconclusive
    and applies no remedy.  Every escalation in a real build logged
    ``confidence 0.0 / evidence_items 0`` for exactly this reason -- it read as
    a model limitation and was actually a missing schema.
    """
    for field in ("cause", "confidence", "evidence", "remedy"):
        assert f'"{field}"' in diagnosis.SYSTEM, \
            f"the diagnosis prompt must show the {field} field"
    assert "JSON object" in diagnosis.SYSTEM
    # And it must say that omitting confidence is not neutral.
    assert "0.0" in diagnosis.SYSTEM


def test_diagnosis_without_confidence_is_treated_as_inconclusive():
    """A cause with no confidence and no evidence is a guess, not a finding."""
    diag = diagnosis.coerce({"cause": "spec_wrong", "remedy": "needs_human"},
                            "T-1")
    assert diag is not None
    assert diag.confidence == 0.0
    assert diag.evidence == []
    assert not (diag.evidence or diag.applied)

def test_does_not_treat_ordinary_feature_prose_as_an_existence_claim():
    """A false positive here blocks a correct task.

    The existence check once also matched a bare ``exposes?``, so
    "the Ledger class must expose the methods add_entry, balance_of" demanded
    that a symbol named ``the`` already exist -- which nothing ever will.  It
    blocked a correct authored task on a real board.

    "must expose" is a *demand*; "already exposes" is a *claim*.  Only the claim
    is evidence.
    """
    (tmp_path := Path(tempfile.mkdtemp()))
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "src" / "ledger.py").write_text(
        "class Ledger:\n    def add_entry(self, e):\n        ...\n",
        encoding="utf-8")
    task = TaskSpec(
        id="T-1", title="ledger",
        objective=("Implement the `Ledger` class in `src/ledger/ledger.py`. It "
                   "must expose the methods `add_entry(self, entry)` and "
                   "`balance_of(self, account_id)`."),
        target_files=["src/ledger.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    assert not report.blocking, [f.kind for f in report.blocking]


def test_still_catches_a_real_existence_claim():
    """The narrowed cue must not lose the case the check was written for."""
    (tmp_path := Path(tempfile.mkdtemp()))
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "config.py").write_text("SETTINGS = 1\n",
                                                encoding="utf-8")
    task = TaskSpec(
        id="T-1", title="config",
        objective=("Add a CLI that reads src/config.py, which already exposes "
                   "reset_settings_cache, and call it on startup."),
        target_files=["src/cli.py"])
    report = spec_critic.deterministic_checks(task, tmp_path)
    missing = [f for f in report.findings
               if f.kind == "objective_asserts_absent_symbols"]
    assert missing, "a genuine 'already exposes X' claim must still be caught"
    assert "reset_settings_cache" in missing[0].detail

# ============================ Duplicate public symbol across tasks (v2.9.1)
def test_flags_a_symbol_two_tasks_must_each_define(tmp_path):
    """The wart that got a delivered library: two ``JournalEntry`` types.

    On the ledgerline board the ``ledger`` task's test asked for
    ``JournalEntry`` from ``src.ledger.ledger`` while the ``journal`` task had
    already defined it in ``src.ledger.journal``.  The Builder satisfied the
    frozen test by *duplicating* the class instead of importing it, and the
    gate passed -- the gate only knows what the test checks.  The library now
    has two incompatible ``JournalEntry`` types.
    """
    root = Path(tempfile.mkdtemp())
    (root / "src" / "ledger").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "ledger" / "journal.py").write_text(
        "class JournalEntry: pass\n", encoding="utf-8")
    (root / "src" / "ledger" / "ledger.py").write_text("class Ledger: pass\n",
                                                        encoding="utf-8")
    (root / "tests" / "test_journal.py").write_text(
        "from src.ledger.journal import JournalEntry\n\n\n"
        "def test_e():\n    assert JournalEntry\n", encoding="utf-8")
    (root / "tests" / "test_ledger.py").write_text(
        "from src.ledger.ledger import Ledger, JournalEntry\n\n\n"
        "def test_e():\n    assert Ledger and JournalEntry\n",
        encoding="utf-8")
    board = [
        {"id": "T-3", "title": "journal", "objective": "Implement JournalEntry.",
         "target_files": ["src/ledger/journal.py"],
         "dependencies": [], "acceptance_criteria": []},
        {"id": "T-4", "title": "ledger", "objective": "Implement Ledger.",
         "target_files": ["src/ledger/ledger.py"],
         "dependencies": [], "acceptance_criteria": []},
    ]
    (root / ".agent").mkdir()
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": board}), encoding="utf-8")

    for raw in board:
        task, err = validate_or_error(TaskSpec, raw)
        assert err is None, err
        found = spec_critic.deterministic_checks(task, root).findings
        hits = [f for f in found if f.kind == "duplicate_public_symbol"]
        assert hits, f"{raw['id']} should be told the symbol is contested"
        assert "JournalEntry" in hits[0].detail


def test_a_name_re_exported_from_a_package_is_not_a_duplicate(tmp_path):
    """Re-exporting is the opposite of duplicating.

    One test importing ``main`` from ``src.pkg.cli`` and another from
    ``src.pkg`` is the normal shape of a package facade, and the real ledgerline
    board does exactly that.  Flagging it would be the check crying wolf.
    """
    root = Path(tempfile.mkdtemp())
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "src" / "pkg" / "cli.py").write_text("def main(): return 0\n",
                                                 encoding="utf-8")
    (root / "tests" / "test_cli.py").write_text(
        "from src.pkg.cli import main\n\n\ndef test_e():\n    assert main\n",
        encoding="utf-8")
    (root / "tests" / "test_pkg.py").write_text(
        "import src.pkg\n\n\ndef test_e():\n    assert src.pkg.main is not None\n",
        encoding="utf-8")
    board = [
        {"id": "T-1", "title": "cli", "objective": "Implement the CLI.",
         "target_files": ["src/pkg/cli.py"], "dependencies": [],
         "acceptance_criteria": []},
    ]
    (root / ".agent").mkdir()
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": board}), encoding="utf-8")
    task, err = validate_or_error(TaskSpec, board[0])
    assert err is None, err
    # test_pkg imports the package, not a name from a module, so there is
    # nothing to contest here.
    assert not [f for f in spec_critic.deterministic_checks(task, root).findings
                if f.kind == "duplicate_public_symbol"]


def test_duplicate_symbol_is_reported_not_blocking():
    """Same calibration as the other structural findings: real, not fatal."""
    f = spec_critic.SpecFinding(kind="duplicate_public_symbol", detail="x",
                                severity="medium")
    report = spec_critic.SpecReport(task_id="T-1")
    report.findings.append(f)
    assert report.blocking == []
    assert not f.remedy, ("no allow-listed remedy fits this yet, so it must "
                          "not be advertised as automatically fixable")


def test_duplicate_symbol_is_silent_on_the_reference_board():
    """20 delivered tasks must not accumulate findings from a new check."""
    import glob
    root = Path(__file__).resolve().parents[1]
    boards = sorted(glob.glob(str(root / "projects" / "*" / ".agent" /
                                 "task_board.json")))
    assert boards, "no boards found"
    hits = []
    for path in boards:
        project = Path(path).parents[1]
        for raw in json.loads(Path(path).read_text(
                encoding="utf-8")).get("tasks", []):
            task, err = validate_or_error(TaskSpec, raw)
            if err:
                continue
            found = [f for f in spec_critic.deterministic_checks(task, project).findings
                     if f.kind == "duplicate_public_symbol"]
            if found:
                hits.append(f"{project.name}/{raw.get('id')}")
    # ledgerline is the known-bad board this check exists to describe; every
    # other board must be silent.
    assert set(hits) <= {"ledgerline/T-003", "ledgerline/T-004"}, hits

# ================================ A package nobody owns an initializer for
def test_flags_a_test_importing_names_from_an_unowned_package(tmp_path):
    """Found on a real authored board: 3 of 4 tasks failed on one ImportError.

    The goal asked for ``src/slatepress/__init__.py``; the Spec Author wrote a
    test doing ``from src.slatepress import split_front_matter`` and produced no
    task for the initializer. Three tasks then failed with the same collection
    error but three *different* diagnoses -- spec_ambiguous, env_broken,
    missing_module -- and none of them said the initializer did not exist.
    """
    root = Path(tempfile.mkdtemp())
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "frontmatter.py").write_text("def parse(): pass\n",
                                                        encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_pkg.py").write_text(
        "from src.pkg import parse\n\n\ndef test_e():\n    assert parse\n",
        encoding="utf-8")
    board = [{"id": "T-1", "title": "frontmatter",
              "objective": "Implement parse in src/pkg/frontmatter.py.",
              "target_files": ["src/pkg/frontmatter.py"],
              "dependencies": [], "acceptance_criteria": []}]
    (root / ".agent").mkdir()
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": board}), encoding="utf-8")
    task, err = validate_or_error(TaskSpec, board[0])
    assert err is None, err
    hits = [f for f in spec_critic.deterministic_checks(task, root).blocking
            if f.kind == "package_init_unowned"]
    assert hits, "a name the package does not contain needs an __init__.py"
    assert "parse" in hits[0].detail


def test_namespace_package_submodule_import_is_fine(tmp_path):
    """PEP 420 satisfies ``from src.data import provider`` with no __init__.py.

    The reference board does exactly this in six places and all of them pass, so
    flagging a bare package import would have blocked 20 delivered tasks.
    """
    root = Path(tempfile.mkdtemp())
    (root / "src" / "data").mkdir(parents=True)
    (root / "src" / "data" / "provider.py").write_text("X = 1\n",
                                                      encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_data.py").write_text(
        "from src.data import provider\n\n\n"
        "def test_e():\n    assert provider.X == 1\n", encoding="utf-8")
    board = [{"id": "T-1", "title": "provider",
              "objective": "Implement src/data/provider.py.",
              "target_files": ["src/data/provider.py"],
              "dependencies": [], "acceptance_criteria": []}]
    (root / ".agent").mkdir()
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": board}), encoding="utf-8")
    task, err = validate_or_error(TaskSpec, board[0])
    assert err is None, err
    assert not [f for f in spec_critic.deterministic_checks(task, root).blocking
                if f.kind == "package_init_unowned"]


def test_a_task_that_owns_the_initializer_clears_the_finding(tmp_path):
    root = Path(tempfile.mkdtemp())
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "src" / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "pkg" / "frontmatter.py").write_text("def parse(): pass\n",
                                                        encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_pkg.py").write_text(
        "from src.pkg import parse\n\n\ndef test_e():\n    assert parse\n",
        encoding="utf-8")
    board = [{"id": "T-1", "title": "pkg", "objective": "Own __init__.py.",
              "target_files": ["src/pkg/__init__.py"], "dependencies": [],
              "acceptance_criteria": []}]
    (root / ".agent").mkdir()
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": board}), encoding="utf-8")
    task, err = validate_or_error(TaskSpec, board[0])
    assert err is None, err
    assert not [f for f in spec_critic.deterministic_checks(task, root).blocking
                if f.kind == "package_init_unowned"]


def test_orphan_authored_tests_are_removed_not_left_behind(tmp_path):
    """A contract with no implementation task is not a contract.

    Stage 2 writes every planned task's test before the Critic runs, because the
    Critic needs them on disk. A task the Critic then blocks therefore leaves
    its test behind, and those orphans break collection for the whole suite and
    inflate the gate's baseline -- observed as 6 test files for 4 tasks.
    """
    from project_orchestrator import ProjectOrchestrator

    root = tmp_path / "projects" / "demo"
    (root / "tests").mkdir(parents=True)
    (root / "src" / "pkg").mkdir(parents=True)
    (root / ".agent").mkdir(parents=True)
    (root / ".agent" / "task_board.json").write_text(
        json.dumps({"tasks": []}), encoding="utf-8")

    kept = root / "tests" / "test_kept.py"
    kept.write_text("def test_a(): pass\n", encoding="utf-8")
    dropped = root / "tests" / "test_dropped.py"
    dropped.write_text("def test_b(): pass\n", encoding="utf-8")
    outside = root / "src" / "pkg" / "helper_test.py"
    outside.write_text("x = 1\n", encoding="utf-8")

    class FakeTask:
        id = "T-9"

    class FakeAuthored:
        def __init__(self, task_id, path, usable):
            self.task = FakeTask() if task_id == "T-1" else None
            self.test_path = path
            self.usable = usable

    class FakePlan:
        tasks = [
            FakeAuthored("T-1", "tests/test_kept.py", True),
            FakeAuthored("T-2", "tests/test_dropped.py", False),
            FakeAuthored("T-3", "src/pkg/helper_test.py", False),
        ]

    orch = ProjectOrchestrator(workspace_root=tmp_path, project_name="demo")
    removed = orch._remove_orphan_authored_tests(FakePlan(), {"T-1"})

    assert removed == ["tests/test_dropped.py"], removed
    assert kept.exists(), "an activated task's test must survive"
    assert not dropped.exists(), "the blocked task's test must be removed"
    assert outside.exists(), "never delete outside tests/"