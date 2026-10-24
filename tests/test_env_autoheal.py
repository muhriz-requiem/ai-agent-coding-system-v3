"""Environment self-heal: it must heal dependencies and nothing else.

This subsystem had **zero** test references, and it was wrong.

``ENV_AUTOHEAL_ENABLED`` fired on any ``missing_modules`` from pytest, with no
distinction between the two causes that produce one:

* a third-party package that is genuinely not installed -- which
  ``pip install -r requirements.txt`` fixes, and which is the only thing this
  code should ever do; and
* the project's **own** code under a name that does not exist -- a wrong import
  path, which no install can fix and which a ``requirements.txt`` cannot even
  mention.

On the ledgerline build the healer woke 55 times, every single one for a path
*inside the project it was building* (``src.ledger.account``,
``src.ledger.journal.entry``, ``ledger``). That project has no
``requirements.txt`` -- it is standard-library only -- so all 55 attempts burned
a cycle and then failed identically with ``no requirements.txt``, a message that
points at the environment when the problem is in the source.

The distinction itself was already written for the Spec Critic
(``is_internal_module``). This file pins the shared predicate and the partition,
so the two call sites cannot drift apart again -- which is exactly how they came
to disagree in the first place.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import config
import executors
import project_orchestrator
import spec_critic


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def project(tmp_path):
    """A standard-library project laid out the way the orchestrator expects.

    ``ProjectOrchestrator`` derives ``project_root`` as
    ``<workspace_root>/projects/<name>``, so a flat temp dir would make
    ``src/`` invisible and every module look third-party -- which is precisely
    how this bug hides.
    """
    root = tmp_path / "projects" / "demo"
    (root / "src" / "ledger").mkdir(parents=True)
    (root / "src" / "ledger" / "journal.py").write_text(
        "class JournalEntry: pass\n", encoding="utf-8")
    (root / "tests").mkdir()
    return root


def _orch(project):
    return project_orchestrator.ProjectOrchestrator(
        workspace_root=project.parents[1], project_name=project.name)


# ------------------------------------------------- the shared internal predicate
def test_project_internal_path_is_internal(project):
    assert spec_critic.is_internal_module("src.ledger.account", project)
    assert spec_critic.is_internal_module("src.ledger.journal.entry", project)


def test_third_party_is_not_internal(project):
    for name in ("pandas", "numpy", "sklearn", "click"):
        assert not spec_critic.is_internal_module(name, project), name


def test_stdlib_is_not_internal(project):
    """``json``/``decimal``/``datetime`` are never installable from this file."""
    for name in ("json", "csv", "decimal", "datetime", "argparse"):
        assert not spec_critic.is_internal_module(name, project), name


def test_empty_name_is_not_internal(project):
    assert not spec_critic.is_internal_module("", project)


# ------------------------------------------------------------------- partitioning
def test_partition_separates_dependencies_from_wrong_paths(project):
    installable, internal = executors.partition_missing(
        ["pandas", "src.ledger.account", "csv", "src.ledger.missing"], project)
    assert installable == ["pandas", "csv"]
    assert internal == ["src.ledger.account", "src.ledger.missing"]


def test_partition_of_nothing_is_empty_but_not_an_error(project):
    assert executors.partition_missing([], project) == ([], [])
    assert executors.partition_missing(None, project) == ([], [])


def test_partition_ignores_blank_entries(project):
    installable, internal = executors.partition_missing(
        ["", "   ", "pandas"], project)
    assert installable == ["pandas"]
    assert internal == []


def test_partition_survives_a_broken_predicate(project, monkeypatch):
    """Triage must never be the thing that breaks the gate."""

    def boom(dotted, root):
        raise RuntimeError("predicate exploded")

    monkeypatch.setattr(spec_critic, "is_internal_module", boom)
    # Falls back to "installable" rather than crashing the build.
    installable, internal = executors.partition_missing(["pandas"], project)
    assert installable == ["pandas"] and internal == []


# ------------------------------------------------- the gate must not heal these
@pytest.mark.parametrize("missing,expect_heal", [
    (["src.ledger.account"], False),          # the observed 55x case
    (["src.ledger.journal.entry"], False),
    (["src.ledger"], False),                  # real directory: namespace package
    (["src.ledger.account", "pandas"], True),  # mixed: heal the real one only
    (["pandas"], True),                       # genuine dependency
    (["click"], True),
])
def test_heal_decision(project, monkeypatch, missing, expect_heal):
    calls = []

    def fake_install(root, timeout=None):
        calls.append(root)
        return {"ok": True, "detail": "installed"}

    monkeypatch.setattr(executors, "run_pip_install", fake_install)
    monkeypatch.setattr(config, "ENV_AUTOHEAL_ENABLED", True)

    orch = _orch(project)
    healed = asyncio.run(orch._maybe_autoheal(
        _task(), list(missing)))
    assert healed is expect_heal
    assert bool(calls) is expect_heal, (
        f"pip was invoked for {missing} even though expect_heal="
        f"{expect_heal}")


def test_bare_package_name_is_treated_as_installable_by_design(project):
    """A genuinely ambiguous name keeps the only safe answer.

    The ledgerline Builder wrote ``import ledger`` -- and ``ledger`` is also a
    real package on PyPI.  ``is_internal_module`` is deliberately narrow,
    because the Spec Critic uses it to *block* a task: a false positive there
    stops correct work, while a false negative only costs one pointless
    ``pip install`` that then fails harmlessly.  So the ambiguity resolves
    toward "maybe installable", and this test pins that choice rather than
    pretending the predicate is sharper than it is.
    """
    assert not spec_critic.is_internal_module("ledger", project)
    installable, internal = executors.partition_missing(["ledger"], project)
    assert installable == ["ledger"] and internal == []


def test_heal_is_skipped_when_disabled(project, monkeypatch):
    monkeypatch.setattr(config, "ENV_AUTOHEAL_ENABLED", False)
    monkeypatch.setattr(executors, "run_pip_install",
                        lambda *a, **k: pytest.fail("pip ran while disabled"))
    orch = _orch(project)
    assert asyncio.run(orch._maybe_autoheal(_task(), ["pandas"])) is False


def test_heal_happens_at_most_once_per_task(project, monkeypatch):
    """The one-shot budget is what stops a retry storm."""
    calls = []
    monkeypatch.setattr(executors, "run_pip_install",
                        lambda root, timeout=None: (calls.append(1),
                                                    {"ok": True,
                                                     "detail": ""})[1])
    monkeypatch.setattr(config, "ENV_AUTOHEAL_ENABLED", True)
    orch = _orch(project)
    task = _task()
    first = asyncio.run(orch._maybe_autoheal(task, ["pandas"]))
    second = asyncio.run(orch._maybe_autoheal(task, ["numpy"]))
    assert first is True
    assert second is False
    assert len(calls) == 1, "the second attempt must not reach pip"


def test_failed_install_reports_false(project, monkeypatch):
    monkeypatch.setattr(executors, "run_pip_install",
                        lambda *a, **k: {"ok": False,
                                         "detail": "no requirements.txt"})
    monkeypatch.setattr(config, "ENV_AUTOHEAL_ENABLED", True)
    orch = _orch(project)
    assert asyncio.run(orch._maybe_autoheal(_task(), ["pandas"])) is False


# ------------------------------------------------------ the shape of the fix
def test_both_gate_sites_route_through_the_one_helper():
    """Two call sites, one helper -- that is the fix.

    The pytest gate and the objective gate are separate code paths with
    separate re-run logic, which is exactly the shape that produced two copies
    of a wrong condition.  They must share the classification.
    """
    src = Path(__file__).resolve().parents[1] / "project_orchestrator.py"
    text = src.read_text(encoding="utf-8")
    assert text.count("_maybe_autoheal(") >= 3, (
        "expected the helper plus both call sites")
    assert "ENV_AUTOHEAL_ENABLED \\\n" not in text, (
        "the old inline gate condition must be gone from the call sites")
    assert text.count("env_autoheal_started") == 1, (
        "the autoheal_started event must be emitted by the helper only, so the "
        "log has one meaning")


def _task():
    from validation import TaskSpec
    return TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["src/ledger/ledger.py"])