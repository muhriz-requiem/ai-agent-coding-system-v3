"""Real subprocess execution: pytest + ruff.

The real-execution gate runs the project's actual test suite against the
actually-applied diff.  Design rules:

* never raises -- every failure mode becomes structured data,
* deterministic offline behaviour for tests: no network in test suites is a
  task-board convention, the executor itself just runs pytest,
* lint is advisory: ``ruff`` (if installed) reports issues that the
  orchestrator surfaces to the Reviewer, but style noise never vetoes,
* outbound calls (the Kaggle CLI) are secret-gated fail-closed: nothing that
  leaves the machine carries a credential.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import config
import redaction
import spec_critic
from logger import log_event

_OUTPUT_TAIL = 1200  # chars of pytest output kept for logs/feedback


def _tail(text: str) -> str:
    flat = " ".join((text or "").split())
    return flat[-_OUTPUT_TAIL:]


#: Shell metacharacters refused in any harness-executed command string.  With
#: ``shell=False`` these have no meaning, so their presence means the command
#: was written expecting a shell and must be rejected rather than reinterpreted.
_SHELL_METACHARS = set("|&;<>$`\n\r")


def _safe_argv(command: str) -> List[str]:
    """Parse *command* into an argv, or ``[]`` if it is not a plain invocation.

    Allows only: an interpreter/entry executable plus plain arguments.  Any
    shell metacharacter, redirection, or substitution refuses the command.
    """
    text = str(command or "").strip()
    if not text:
        return []
    if any(ch in _SHELL_METACHARS for ch in text):
        return []
    try:
        argv = shlex.split(text, posix=True)
    except ValueError:
        return []
    argv = [a for a in argv if a != ""]
    return argv if argv else []


def collect_test_ids(project_root: Path,
                     timeout: Optional[int] = None) -> List[str]:
    """Node IDs of every collectable test, sorted.

    v2.8: the tests-first gate used to compare a single integer (``passed``
    before vs after), which cannot tell "added two real tests" from "added one
    trivial test" or "deleted the two hard ones and added one easy one".  A
    node-ID diff can: it distinguishes a *grew* suite from a *replaced* one.

    Fail-closed on a collection error: an unreadable id set is returned as an
    empty list rather than raising, and callers treat empty as "cannot verify",
    never as "nothing was removed".
    """
    root = Path(project_root)
    ids: List[str] = []
    if importlib.util.find_spec("pytest") is None:
        return ids
    cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q",
           "-p", "no:cacheprovider", "--continue-on-collection-errors"]
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout or config.TEST_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return ids
    for line in ((proc.stdout or "") + "\n" + (proc.stderr or "")).splitlines():
        line = line.strip()
        if not line or "::" not in line:
            continue
        if line.lower().startswith(("error", "usage")):
            continue
        ids.append(line)
    return sorted(set(ids))


def test_set_deltas(baseline_ids: Sequence[str],
                    current_ids: Sequence[str]) -> Dict[str, List[str]]:
    """``{added, removed}`` node IDs between two collection runs."""
    before, after = set(baseline_ids or ()), set(current_ids or ())
    return {"added": sorted(after - before), "removed": sorted(before - after)}


def run_tests(project_root: Path, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Run the project's pytest suite; return {passed, failed, available, detail}.

    Fail-closed policy: if pytest is missing, the run times out or the
    process cannot start, ``failed`` is set to >= 1 so a real-execution
    Reviewer can never approve an unverifiable change.
    """
    root = Path(project_root)
    result: Dict[str, Any] = {"passed": 0, "failed": 1, "skipped": 0,
                              "available": True, "detail": "",
                              "missing_modules": [], "failing_tests": []}
    if importlib.util.find_spec("pytest") is None:
        result.update(available=False,
                      detail="pytest is not installed (pip install pytest)")
        log_event("WARNING", "real_tests_unavailable", {"project": root.name})
        return result

    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line",
           "-p", "no:cacheprovider",
           # A single broken module must not abort the whole session: let
           # the rest of the suite run, then fail-closed on the error count.
           # --tb=line (v2.5.6) keeps one line per failure so
           # ModuleNotFoundError reasons reach the missing_modules evidence.
           "--continue-on-collection-errors"]
    try:
        proc = subprocess.run(
            cmd, cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout or config.TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        result["detail"] = f"pytest timed out after {timeout or config.TEST_TIMEOUT_S}s"
        log_event("ERROR", "real_tests_timeout", {"project": root.name})
        return result
    except OSError as exc:
        result["detail"] = f"pytest could not start: {exc}"
        log_event("ERROR", "real_tests_error", {"project": root.name, "error": str(exc)})
        return result

    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    passed_match = re.search(r"(\d+) passed", output)
    failed_match = re.search(r"(\d+) failed", output)
    skipped_match = re.search(r"(\d+) skipped", output)
    errors_match = re.search(r"(\d+) errors?", output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    skipped = int(skipped_match.group(1)) if skipped_match else 0
    errors = int(errors_match.group(1)) if errors_match else 0

    if "no tests ran" in output or proc.returncode == 5:
        # Nothing collected: treat as an unverifiable (failing) state.
        failed = max(failed, 1)
        result["detail"] = "no tests collected"
    elif proc.returncode == 4 and not output.strip():
        # Silent usage-exit: almost always a conftest calling sys.exit()
        # or raising SystemExit during load.
        failed = max(failed, 1)
        result["detail"] = ("pytest exited 4 with no output -- a conftest or "
                            "imported module likely called sys.exit() or "
                            "raised SystemExit during load")
    elif proc.returncode == 0:
        failed = 0  # trust the exit code on green runs
    elif failed == 0:
        failed = 1  # non-zero exit without a parseable summary

    # Collection errors (broken/missing modules) are failures too --
    # fail-closed, but without aborting the rest of the suite.
    if errors:
        failed = max(failed, errors)

    # v2.5.6 structured evidence: WHY a run failed, machine-readable --
    # consumed by gate feedback, the Reviewer context and (later) the
    # SpecDoctor autonomy layer.
    missing_modules = sorted(set(re.findall(
        r"ModuleNotFoundError: No module named '([^']+)'", output)))
    # v2.9.1: the *other* import failure, and the one the Builder actually makes.
    #
    # ``ModuleNotFoundError`` means the module is absent.  ``ImportError:
    # cannot import name 'X' from 'Y'`` means the module is right there and does
    # not define ``X`` -- because the Builder invented a name.  Only the first
    # was extracted, so the gate reported "tests did not run" with no missing
    # modules and its remediation said "implement the target module", which the
    # Builder had already done.  Measured cost: nine wasted Builder attempts on
    # one task whose only problem was ``from src.ledger.journal import JournalEntry,
    # EntryType`` against a module defining ``Side, Posting, JournalEntry``.
    missing_names = sorted({
        (name, module)
        for name, module in re.findall(
            r"ImportError: cannot import name '([^']+)' from '([^']+)'", output)})
    failing_tests = sorted(set(re.findall(
        r"^(?:FAILED|ERROR) (tests/\S+)", output, re.MULTILINE)))[:10]

    result.update(passed=passed, failed=failed, skipped=skipped,
                  missing_modules=missing_modules,
                  missing_names=missing_names,
                  failing_tests=failing_tests,
                  detail=_tail(output) or result.get("detail", ""))
    log_event("INFO", "real_tests_run",
              {"project": root.name, "passed": passed, "failed": failed,
               "skipped": skipped, "missing_modules": missing_modules,
               "missing_names": [f"{n} from {m}" for n, m in missing_names][:6],
               "exit_code": proc.returncode})
    return result


def run_lint(project_root: Path, files: List[str],
             timeout: Optional[int] = None) -> List[str]:
    """Run ``ruff check`` over *files* (project-relative); return issue lines.

    Returns [] when ruff is not installed or nothing to check -- lint is
    advisory context for the Reviewer, never a hard gate.
    """
    root = Path(project_root)
    ruff = shutil.which("ruff")
    if not ruff:
        return []
    targets = [str(f) for f in files if f][:50]
    if not targets:
        return []
    try:
        proc = subprocess.run(
            [ruff, "check", "--select", "E9,F", "--", *targets],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout or config.LINT_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as exc:
        log_event("WARNING", "real_lint_error", {"project": root.name,
                                                 "error": str(exc)})
        return []
    issues = [line.strip() for line in (proc.stdout or "").splitlines()
              if line.strip()][:20]
    if issues:
        log_event("WARNING", "real_lint_issues",
                  {"project": root.name, "count": len(issues)})
    return issues


#: Requirement lines that install code rather than a package: editable installs,
#: direct URLs, and VCS references all run arbitrary code at install time.
_UNSAFE_REQUIREMENT_RE = re.compile(
    r"(?:^\s*-e\b)|(?:^\s*--editable\b)|(?:git\+)|(?:https?://)|(?:file://)|"
    r"(?:\.whl\b)|(?:\.tar\.gz\b)|(?:^\s*\.)",
    re.IGNORECASE)


def partition_missing(modules: Sequence[str], project_root: Path
                      ) -> Tuple[List[str], List[str]]:
    """Split missing module names into ``(installable, internal)``.

    A missing module has two completely different causes, and only one of them
    is a missing dependency:

    * ``pandas`` -- a third-party package. ``pip install -r requirements.txt``
      is the right and only remedy, and it is what the self-heal does.
    * ``src.ledger.account`` -- the project's *own* code, under a name that does
      not exist. Nothing installs it; it is a wrong import path and the remedy
      is to change the import. Worse, a project with no ``requirements.txt``
      (true of every standard-library project) makes the self-heal fail with
      "no requirements.txt" -- a message that points at the environment when the
      problem is in the source.

    Measured cost of not distinguishing them: the ledgerline build woke the
    auto-healer 55 times, every one of them for a path inside the project being
    built, each attempt burning a cycle before failing identically.

    ``is_internal_module`` is the shared predicate, so this classification and
    the Spec Critic's cannot drift apart -- which is how the two call sites came
    to disagree in the first place.
    """
    installable: List[str] = []
    internal: List[str] = []
    for raw in modules or ():
        name = str(raw or "").strip()
        if not name:
            continue
        try:
            is_internal = spec_critic.is_internal_module(
                name, Path(project_root))
        except Exception:  # noqa: BLE001 -- triage must never break the gate
            is_internal = False
        (internal if is_internal else installable).append(name)
    return installable, internal


def run_pip_install(project_root: Path,
                    timeout: Optional[int] = None) -> Dict[str, Any]:
    """Environment self-heal (v2.5.6): ``pip install -r requirements.txt``.

    Requirements-only policy: installing the project's own requirements
    file resolves the listed packages AND their missing transitive deps
    (the ``click``-under-``typer`` case).  Never raises; the caller gates
    on ``config.ENV_AUTOHEAL_ENABLED`` and re-runs the gate afterwards.
    """
    req = Path(project_root) / "requirements.txt"
    if not req.exists():
        return {"ok": False, "detail": "no requirements.txt"}
    # v2.8: refuse editable installs and any other directive that executes code
    # at install time.  A requirements.txt is board- and Builder-influenced, so
    # ``-e .`` or a VCS/direct URL requirement is arbitrary code execution
    # running with the harness's full live environment.  Plain named
    # requirements only.
    try:
        lines = req.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return {"ok": False, "detail": f"requirements.txt unreadable: {exc}"}
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        # Option lines (-e / --editable / --index-url ...) are checked too, not
        # skipped: ``-e .`` is the most direct way to install arbitrary code.
        if _UNSAFE_REQUIREMENT_RE.search(line):
            log_event("WARNING", "env_autoheal_requirement_refused",
                      {"project": Path(project_root).name,
                       "requirement": line[:80]})
            return {"ok": False,
                    "detail": "requirements.txt contains an editable / VCS / "
                              "URL requirement; auto-heal installs plain "
                              "named packages only"}
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "-r", str(req)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout or config.PIP_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": False,
                "detail": (f"pip install timed out after "
                           f"{timeout or config.PIP_TIMEOUT_S}s")}
    except OSError as exc:
        return {"ok": False, "detail": f"pip could not start: {exc}"}
    detail = " ".join(((proc.stdout or "") + " " + (proc.stderr or "")).split())
    return {"ok": proc.returncode == 0, "detail": detail[-_OUTPUT_TAIL:]}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort process-tree kill (Windows training runs spawn children)."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:  # pragma: no cover -- POSIX
            proc.kill()
    except Exception:  # noqa: BLE001 -- killing must never raise
        pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def run_objective(project_root: Path, command: str,
                  results_path: str = "reports/metrics.json",
                  timeout: Optional[int] = None,
                  required_fields: Optional[List[str]] = None
                  ) -> Dict[str, Any]:
    """Run an experiment/benchmark command and parse its metrics JSON (v2.6).

    The generic objective gate: the task pins ``run_command`` and the
    ``results_path`` the command must write.  Fail-closed on every anomaly
    (missing command, non-zero exit, timeout, unreadable/invalid JSON,
    missing required fields) so an unverifiable run can never pass.

    Returns ``{ok, metrics, primary, runtime_s, split_hash, detail,
    missing_modules, available}``.
    """
    root = Path(project_root)
    result: Dict[str, Any] = {
        "ok": False, "metrics": {}, "primary": None, "runtime_s": 0.0,
        "split_hash": None, "detail": "", "missing_modules": [],
        "available": True,
    }
    if not command or not str(command).strip():
        result["detail"] = "task declares no run_command"
        return result

    # v2.8: ``shell=True`` handed a board-authored string to the OS shell with
    # the full live environment, so ``run_command = "python x.py; curl ..."``
    # or a backtick/`$(...)`` substitution executed arbitrary commands with
    # every provider credential in scope.  The Kaggle pack already validated
    # its own commands; this is the generic path and it had no such check.
    # Parse to an argv and run without a shell -- metacharacters then have no
    # meaning.  Unparseable input is refused rather than guessed at.
    argv = _safe_argv(command)
    if not argv:
        result["detail"] = ("run_command is not a plain executable invocation "
                            f"({command!r}); chaining, substitution and "
                            "redirection are not allowed here")
        log_event("WARNING", "objective_run_command_refused",
                  {"project": root.name, "reason": "unparseable_or_unsafe"})
        return result

    # Stale-evidence guard: the metrics file must be (re)written by THIS
    # run.  Without deleting it first, a command that exits 0 without
    # writing (crash before the write, wrong path, swallowed exception)
    # would pass on the previous run's metrics -- and in divergence
    # rounds, candidate N would inherit candidate N-1's score.
    metrics_file = root / results_path
    try:
        metrics_file.unlink(missing_ok=True)
    except OSError:
        pass

    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(root), shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        result["detail"] = f"experiment could not start: {exc}"
        log_event("ERROR", "objective_run_error",
                  {"project": root.name, "error": str(exc)})
        return result

    try:
        output, _ = proc.communicate(
            timeout=timeout or config.OBJECTIVE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        _kill_process_tree(proc)
        result["runtime_s"] = round(time.monotonic() - started, 2)
        result["detail"] = (
            f"experiment timed out after {timeout or config.OBJECTIVE_TIMEOUT_S}s "
            "(process tree killed)")
        log_event("ERROR", "objective_run_timeout",
                  {"project": root.name, "timeout_s":
                   timeout or config.OBJECTIVE_TIMEOUT_S})
        return result

    output = output or ""
    result["runtime_s"] = round(time.monotonic() - started, 2)
    result["detail"] = _tail(output)
    result["missing_modules"] = sorted(set(re.findall(
        r"ModuleNotFoundError: No module named '([^']+)'", output)))

    metrics_file = root / results_path
    try:
        # Freshness check (belt-and-braces after the unlink): a file older
        # than this run started cannot be evidence produced by it.
        if metrics_file.stat().st_mtime < started - 2.0:
            raise OSError("metrics file predates this run (stale evidence)")
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result["detail"] = (f"metrics file unreadable ({results_path}): {exc}; "
                            + result["detail"])[:_OUTPUT_TAIL + 200]
        log_event("WARNING", "objective_metrics_unreadable",
                  {"project": root.name, "path": results_path,
                   "exit_code": proc.returncode})
        return result
    if not isinstance(payload, dict):
        result["detail"] = "metrics JSON must be an object"
        return result

    missing = [field for field in (required_fields or [])
               if field not in payload]
    if missing:
        result["detail"] = f"metrics missing required fields: {missing}"
        result["metrics"] = payload
        log_event("WARNING", "objective_metrics_missing_fields",
                  {"project": root.name, "missing": missing})
        return result

    result["metrics"] = payload
    result["split_hash"] = payload.get("split_hash")
    if proc.returncode != 0:
        result["detail"] = (f"experiment exited {proc.returncode}; "
                            + result["detail"])[:_OUTPUT_TAIL + 200]
        log_event("WARNING", "objective_run_nonzero_exit",
                  {"project": root.name, "exit_code": proc.returncode})
        return result

    result["ok"] = True
    log_event("INFO", "objective_run_completed",
              {"project": root.name, "runtime_s": result["runtime_s"],
               "metrics_keys": sorted(payload.keys())[:12]})
    return result


def validate_submission(project_root: Path, submission_path: str,
                        sample_path: str = "data/raw/sample_submission.csv"
                        ) -> Tuple[bool, str]:
    """Validate a submission file against the competition's sample (v2.6).

    stdlib-only (csv): header set equality (order-insensitive), identical
    row count, identical id-column values as sets, and NO empty cells.
    Fail-open ONLY when no sample file exists (some competitions do not
    ship one) -- an absent sample cannot be validated, and the Kaggle
    scorer is the final authority.
    """
    import csv

    submission = Path(project_root) / submission_path
    sample = Path(project_root) / sample_path
    if not submission.exists():
        return False, f"submission file not found: {submission_path}"
    if not sample.exists():
        return True, "no sample submission found; format checks skipped"
    try:
        with open(submission, newline="", encoding="utf-8") as handle:
            sub_rows = list(csv.reader(handle))
        with open(sample, newline="", encoding="utf-8") as handle:
            sample_rows = list(csv.reader(handle))
    except OSError as exc:
        return False, f"could not read submission: {exc}"

    if not sub_rows or not sample_rows:
        return False, "empty submission or sample file"
    sub_header, sample_header = sub_rows[0], sample_rows[0]
    if set(sub_header) != set(sample_header):
        return False, (f"header mismatch: submission={sub_header} "
                       f"sample={sample_header}")
    if len(sub_rows) != len(sample_rows):
        return False, (f"row count mismatch: submission={len(sub_rows)-1} "
                       f"sample={len(sample_rows)-1}")
    for row in sub_rows[1:]:
        if len(row) != len(sub_header) or any(c.strip() == "" for c in row):
            return False, "submission contains empty or malformed cells"

    sub_index = {name: i for i, name in enumerate(sub_header)}
    sample_index = {name: i for i, name in enumerate(sample_header)}
    id_column = sample_header[0]
    if id_column in sub_index and id_column in sample_index:
        sub_ids = {row[sub_index[id_column]] for row in sub_rows[1:]}
        sample_ids = {row[sample_index[id_column]] for row in sample_rows[1:]}
        if sub_ids != sample_ids:
            # Cell values are echoed back to the caller and persisted into
            # campaign.json, so a stray secret in an id column is redacted here
            # rather than copied into the campaign ledger.
            return False, (f"{id_column!r} values differ from the sample "
                           "(missing/extra ids): "
                           f"submission={redaction.redact_known_secrets(sorted(sub_ids)[:5])} "
                           f"sample={redaction.redact_known_secrets(sorted(sample_ids)[:5])}")
    return True, "submission format valid"


def run_kaggle_submit(project_root: Path, competition: str, file_path: str,
                      message: str = "agent submission",
                      wait_s: Optional[int] = None,
                      timeout: Optional[int] = None) -> Dict[str, Any]:
    """Submit via the Kaggle CLI (``competitions submit --wait``); v2.6.

    Never raises: every anomaly becomes ``{ok: False, detail}``.  On
    success the parsed submission ref (and public score when the wait
    completed) are returned for the campaign audit trail.
    """
    result: Dict[str, Any] = {"ok": False, "ref": None, "score": None,
                              "detail": ""}
    kaggle = shutil.which("kaggle")
    if not kaggle:
        result["detail"] = "kaggle CLI not installed (pip install kaggle)"
        return result
    if not competition:
        result["detail"] = "no competition configured"
        return result
    target = Path(project_root) / file_path
    if not target.exists():
        result["detail"] = f"submission file not found: {file_path}"
        return result
    wait = config.KAGGLE_SUBMIT_WAIT_S if wait_s is None else int(wait_s)
    cmd = [kaggle, "competitions", "submit", "-c", competition,
           "-f", str(target), "-m", message or "agent submission"]
    if wait > 0:
        cmd += ["--wait", str(wait)]

    # --- egress gate: the last point before bytes leave the machine ---------
    # The submission file is written by an agent-authored experiment script, so
    # its contents are entirely model-determined. A hypothesis that dumped the
    # environment (or a stray hardcoded key) into a prediction column would
    # otherwise be published to a public competition. Fail closed: refuse and
    # report, never "submit anyway".
    blocked = _secret_gate(target, message)
    if blocked:
        result["detail"] = blocked
        log_event("WARNING", "kaggle_submission_blocked_secrets",
                  {"competition": competition, "path": str(target),
                   "reason": blocked})
        return result

    try:
        proc = subprocess.run(
            cmd, cwd=str(project_root), capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=(timeout or (wait + 120 if wait else 300)))
    except (subprocess.TimeoutExpired, OSError) as exc:
        result["detail"] = f"kaggle submit failed to run: {exc}"
        return result
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ref_match = re.search(r"Submission ref:\s*(\d+)", output)
    if ref_match:
        result["ref"] = ref_match.group(1)
    score_match = re.search(r"score[:\s]+([0-9.]+)", output, re.IGNORECASE)
    if score_match:
        try:
            result["score"] = float(score_match.group(1))
        except ValueError:
            pass
    result["ok"] = proc.returncode == 0 and result["ref"] is not None
    # The combined CLI output is persisted verbatim into campaign.json by the
    # domain pack, so redact before it lands there.
    result["detail"] = redaction.redact_known_secrets(
        " ".join(output.split())[-_OUTPUT_TAIL:])
    log_event("INFO" if result["ok"] else "WARNING", "kaggle_submit_result",
              {"competition": competition, "ok": result["ok"],
               "ref": result["ref"], "score": result["score"]})
    return result


def submission_secret_findings(project_root: Path, submission_path: str,
                               message: str = "") -> List[Dict[str, Any]]:
    """Credential findings for a submission file (empty list means "clear").

    Split out from :func:`_secret_gate` so the domain pack can *report* a block
    at the point of decision, while the executor remains the single place that
    actually refuses.
    """
    if not getattr(config, "SECRET_SCAN_ENABLED", True):
        return []
    try:
        findings: List[Dict[str, Any]] = []
        target = Path(project_root) / submission_path
        if target.is_file():
            findings.extend(redaction.scan_path(target))
        if message:
            findings.extend(redaction.scan_secrets(message))
        return findings
    except Exception:  # noqa: BLE001 -- callers treat "unknown" as "not clear"
        return [{"rule": "scan_failed", "line": 0, "hint": "?"}]


def _secret_gate(target: Path, message: str = "") -> str:
    """Return a refusal reason when *target*/*message* would leak a credential.

    Empty string means "clear to send".  Both a configured-secret check
    (exact, no false positives) and a pattern check run over the file and the
    ``-m`` message, which also travels off-box.  Only rule names and hints are
    ever reported -- never the matched text.
    """
    if not getattr(config, "SECRET_SCAN_ENABLED", True):
        return ""
    try:
        findings = submission_secret_findings(target.parent, target.name, message)
    except Exception as exc:  # noqa: BLE001 -- an undecidable gate must fail shut
        return f"secret scan could not complete, refusing to submit ({type(exc).__name__})"
    if not findings:
        return ""
    rules = sorted({str(f.get("rule", "?")) for f in findings})
    hints = sorted({str(f.get("hint", "?")) for f in findings})
    return (f"submission blocked: credential-shaped content detected "
            f"(rules={','.join(rules)}; hints={','.join(hints[:3])}; "
            f"findings={len(findings)}). Move the secret to an environment "
            f"variable and rebuild the submission file.")
