"""ProjectOrchestrator: one per project; runs the Builder-Reviewer loop.

Loop per cycle:
    honor pause flag -> pick next pending task -> plan -> build -> review
    -> complete or retry (max 3 attempts) -> write summaries -> notify the
    IntegrationAgent.
"""

from __future__ import annotations

import asyncio
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import config
import divergence
import domain_registry
import executors
import model_router
import patcher
import diagnosis
import lines as lines_mod
import redaction
import spec_author
import spec_critic
import steering
import test_integrity
from agent_roles import (BuilderAgent, IntegrationEvent, JudgeAgent,
                         LLMClient, MergerAgent, PlannerAgent,
                         ReviewerAgent, SpecAuthorAgent, SpecCriticAgent,
                         _extract_json)
from logger import log_event
from providers import LLMUnavailableError
from quota_governor import get_governor
from shared_workspace_engine import SharedWorkspaceEngine
from tool_registry import resolve_required_tools
from validation import (AutomatedTests, BuilderOutput, CycleSummary, Plan,
                        MetricSpec, ObjectiveGateResult, ReviewFeedbackItem,
                        ReviewVerdict, StrategySpec, TaskSpec,
                        validate_or_error)

#: Ordering used when sorting tasks by priority.
_PRIORITY_RANK = {"high": 0, "medium": 1, "low": 2}

#: Board-read retry schedule (attempt 0 tries immediately).  A transient
#: read failure must never become a total loss of the task backlog, so a
#: short backoff rides out Windows file locks from editors / AV / indexers.
_BOARD_READ_BACKOFF_S: Tuple[float, ...] = (0.0, 0.2, 0.5)


def _usage_tokens(usage: Optional[Dict[str, Any]]) -> int:
    """Best-effort token count from an ``LLMClient`` usage snapshot.

    v2.8 reads the *real* SDK-reported figure when present.  ``_estimate_tokens``
    (a character-count heuristic that ignores completions) remains the
    fallback so the steering burn-rate detector works before any real usage
    data exists.
    """
    if not isinstance(usage, dict):
        return 0
    for key in ("total_tokens", "tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value > 0:
            return value
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, int) or isinstance(completion, int):
        return int(prompt or 0) + int(completion or 0)
    return 0


def real_tests_block(tests: Dict[str, Any], baseline_passed: int) -> bool:
    """Tests-first rule: the task's own tests must RUN (not skip).

    A green run whose passed count did not grow over the pre-task baseline
    means the implementation module never came into existence.

    v2.8: this integer comparison is kept as the *cheap* check, but it is no
    longer the whole rule -- see :func:`test_set_gate_block`.  A count cannot
    distinguish "added two real tests" from "deleted the hard ones and added
    one trivial one".
    """
    return int(tests.get("passed", 0)) <= int(baseline_passed)


def test_set_gate_block(baseline_ids: Sequence[str], current_ids: Sequence[str],
                        required: bool) -> Optional[str]:
    """Node-ID level check for the tests-first gate. ``None`` means pass.

    Closes two holes the count-only rule could not see:

    * **deleted tests** -- removing a failing assertion and adding one trivial
      passing test keeps ``passed`` growing, so the count rule approves it.
      Here the removed set would be non-empty.
    * **no tests added at all** -- the Builder satisfies "passed grew" by
      fixing an unrelated flake instead of writing the required test.

    The pre-existing-test *content* is protected separately by
    :mod:`test_integrity`; this checks the shape of the suite.

    Known limit, stated plainly: this does not verify that a new test's
    *assertions* are meaningful.  That needs mutation testing, which the
    system does not have.
    """
    if not required:
        return None
    deltas = executors.test_set_deltas(baseline_ids, current_ids)
    if deltas["removed"]:
        return ("the patch removed pre-existing test(s): "
                + ", ".join(deltas["removed"][:5])
                + " -- tests are the fixed specification")
    if not deltas["added"]:
        return ("requires_new_tests is set but the patch added no new test; "
                "add a test that exercises the new behaviour")
    return None


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _primary_value(payload: Dict[str, Any], name: str) -> Optional[float]:
    """Extract the primary metric value from an experiment's metrics JSON."""
    for key in (name, "primary", "score", "metric"):
        if key and isinstance(payload.get(key), (int, float)):
            return float(payload[key])
    return None


def objective_gate_block(metric: Optional[MetricSpec], outcome: Dict[str, Any],
                         baseline: Optional[float]) -> Tuple[bool, str]:
    """Objective-gate rule (v2.6): all-or-nothing on the pinned metric.

    Returns ``(blocked, reason)``.  Fail-closed: unparseable/missing
    results block; a split-hash mismatch blocks (scores from different
    splits are never comparable); target and min-improvement are enforced.
    """
    if metric is None:
        return False, ""
    if not outcome.get("ok"):
        return True, str(outcome.get("detail") or "objective run failed")
    value = outcome.get("primary")
    if value is None:
        return True, f"primary metric {metric.name!r} missing from results"
    if metric.split_key is not None and \
            outcome.get("split_hash") != metric.split_key:
        return True, (f"split_hash mismatch: expected {metric.split_key!r}, "
                      f"got {outcome.get('split_hash')!r}")
    direction = (metric.direction or "maximize").lower()
    if metric.target is not None:
        hit = value >= metric.target if direction == "maximize" \
            else value <= metric.target
        if not hit:
            return True, (f"{metric.name}={_fmt_metric(value)} misses target "
                          f"{_fmt_metric(metric.target)} ({direction})")
    if baseline is not None and metric.min_improvement > 0:
        improvement = (value - baseline) if direction == "maximize" \
            else (baseline - value)
        if improvement < metric.min_improvement:
            return True, (f"{metric.name}={_fmt_metric(value)} improves only "
                          f"{_fmt_metric(improvement)} over baseline "
                          f"{_fmt_metric(baseline)} (requires "
                          f"{_fmt_metric(metric.min_improvement)})")
    return False, ""


def compute_board_stats(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Board summary surfaced through the manifest (v2.5.5).

    ``blocked`` counts pending tasks with at least one dependency that is
    not ``done`` -- the same eligibility rule the scheduler applies -- so
    an idle loop is self-explanatory ("starved by TASK-X") instead of
    looking dead.  ``failed_ids`` is capped to keep the manifest compact.
    """
    done_ids = {t.get("id") for t in tasks if t.get("status") == "done"}
    failed_ids = [str(t.get("id")) for t in tasks
                  if t.get("status") == "failed"]
    counts = {"done": 0, "failed": 0, "pending": 0, "blocked": 0}
    for task in tasks:
        status = task.get("status")
        if status in counts:
            counts[status] += 1
        if status == "pending":
            deps = task.get("dependencies") or []
            if any(dep not in done_ids for dep in deps):
                counts["blocked"] += 1
    counts["failed_ids"] = failed_ids[:10]
    return counts


class ProjectOrchestrator:
    """Owns one project directory and all of its task cycles."""

    def __init__(self, workspace_root: Path, project_name: str,
                 engine: Optional[SharedWorkspaceEngine] = None,
                 llm: Optional[LLMClient] = None,
                 integration_queue: Optional[asyncio.Queue] = None,
                 status: Optional[Dict[str, Any]] = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.project_name = project_name
        self.project_root = self.workspace_root / "projects" / project_name
        self.agent_dir = self.project_root / ".agent"
        self.board_path = self.agent_dir / "task_board.json"
        self.engine = engine or SharedWorkspaceEngine(self.workspace_root)
        self.llm = llm or LLMClient(self.workspace_root)
        self.planner = PlannerAgent(self.llm, self.project_root, self.workspace_root)
        self.builder = BuilderAgent(self.llm, self.project_root)
        self.builder.workspace_root = self.workspace_root
        self.reviewer = ReviewerAgent(self.llm, self.project_root, self.workspace_root)
        self.judge = JudgeAgent(self.llm, self.project_root, self.workspace_root)
        self.merger = MergerAgent(self.llm, self.project_root, self.workspace_root)
        # v2.9: two pre-Builder surfaces.  ``spec_critic`` audits an objective
        # against its tests before any attempt is spent; ``spec_author`` fills
        # an empty board from the project's goal.  Both are per-project because
        # both are about this project's specification, and a shared instance
        # would leak one project's contract into another's prompt.
        self.spec_critic = SpecCriticAgent(self.llm, self.project_root,
                                           self.workspace_root)
        self.spec_author = SpecAuthorAgent(self.llm, self.project_root,
                                           self.workspace_root)
        self.integration_queue = integration_queue
        self.status = status if status is not None else {}
        self.current_task_id: Optional[str] = None
        self.poll_interval_s = 1.0
        self.idle_interval_s = 3.0
        self._env_heal_done = False  # v2.5.6: one self-heal per task
        self._divergence_done = False  # v2.6: divergence only on attempt 1
        # v2.8: frozen test baseline, captured once per task.
        self._test_guard_cache: Optional[test_integrity.TestIntegrityGuard] = None
        self._current_guard_for: str = ""
        # v2.8: deterministic progress telemetry (doom loops, stalls, blocks).
        self.steering = steering.monitor_for(self.workspace_root)

    # ------------------------------------------------------------------
    # Task board
    # ------------------------------------------------------------------
    def _load_board(self) -> Optional[Dict[str, Any]]:
        """Read the board, or ``None`` when it could not be parsed.

        ``None`` is load-bearing: a caller MUST NOT hand the result to
        :meth:`_save_board`.  Returning a placeholder like ``{"tasks": []}``
        turns any transient read failure -- a torn read, an AV/indexer file
        lock, a partial write from a non-atomic writer -- into a *silent
        total loss of the backlog*, because the placeholder satisfies the
        board schema and gets written straight back over the real file.

        Transient failures are retried with a short backoff: on Windows an
        editor, virus scanner or indexer can hold the file for a few hundred
        milliseconds, which is not worth discarding a board over.
        """
        last_error: str = ""
        for attempt, delay in enumerate(_BOARD_READ_BACKOFF_S):
            if attempt:
                time.sleep(delay)
            try:
                return json.loads(self.board_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                last_error = str(exc)
        log_event("ERROR", "task_board_unreadable",
                  {"project": self.project_name, "error": last_error,
                   "attempts": len(_BOARD_READ_BACKOFF_S)})
        return None

    def _load_board_or_empty(self) -> Dict[str, Any]:
        """Read-only convenience: the board, or ``{"tasks": []}`` if unreadable.

        Only for call sites that immediately discard the result (statistics,
        next-task selection).  Anything that writes the board back must use
        :meth:`_load_board` and handle ``None`` explicitly.
        """
        return self._load_board() or {"tasks": []}

    def _save_board(self, board: Dict[str, Any]) -> None:
        """Atomic board write (tmp-file + replace) so pollers (dashboard,
        operators, the orchestrator's own next tick) never see a torn read.

        Shares :func:`redaction.write_json_atomic` with every other state
        writer in the system so there is exactly one write discipline.
        """
        try:
            redaction.write_json_atomic(self.board_path, board)
        except (OSError, TypeError, ValueError) as exc:
            log_event("ERROR", "task_board_write_failed",
                      {"project": self.project_name, "error": str(exc)})

    def get_next_pending_task(self) -> Optional[Tuple[TaskSpec, Dict[str, Any]]]:
        """First eligible pending task: dependencies done, then priority
        (high > medium > low), then board list order."""
        board = self._load_board_or_empty()
        tasks = board.get("tasks", [])
        done_ids = {t.get("id") for t in tasks if t.get("status") == "done"}
        eligible: List[Tuple[int, int, TaskSpec, Dict[str, Any]]] = []
        for index, task in enumerate(tasks):
            obj, err = validate_or_error(TaskSpec, task)
            if err:
                # Concise structured error upstream -- never a raw traceback.
                log_event("ERROR", "task_validation_failed",
                          {"project": self.project_name, "error": err["error"]})
                continue
            if obj.status != "pending":
                continue
            if self._deferred(obj):
                continue
            if any(dep not in done_ids for dep in obj.dependencies):
                continue  # blocked on unfinished dependency
            eligible.append((_PRIORITY_RANK.get(obj.priority, 1), index, obj, task))
        if not eligible:
            return None
        eligible.sort(key=lambda item: (item[0], item[1]))
        chosen, raw = eligible[0][2], eligible[0][3]
        # v2.9: the deterministic spec checks run *here*, in the synchronous
        # scheduler, so a task whose objective disagrees with its tests is
        # reported before it is ever handed to a Builder.  Only the free
        # mechanical tier -- no model call in the sync path, which has no
        # LLMClient to make one with.  The model's own pass runs in
        # ``process_task`` before attempt 1.
        if config.SPEC_CRITIC_ENABLED:
            report = self._deterministic_spec_report(chosen)
            if report is not None and report.blocking:
                log_event("ERROR", "spec_critic_blocked",
                          {"project": self.project_name, "task_id": chosen.id,
                           "findings": [f.to_dict() for f in report.blocking]})
                return None
            if report is not None and report.findings:
                # Non-blocking notes ride along on the objective so the Builder
                # knows what the Critic saw without the loop losing the task.
                raw = dict(raw)
                raw["objective"] = spec_critic.annotate_objective(
                    chosen.objective, report)
                chosen = chosen.model_copy(
                    update={"objective": raw["objective"]})
                log_event("INFO", "spec_critic_notes",
                          {"project": self.project_name, "task_id": chosen.id,
                           "notes": len(report.findings)})
        return chosen, raw

    async def _maybe_autoheal(self, task: TaskSpec,
                              modules: Sequence[str]) -> bool:
        """Self-heal the environment, but only for genuinely missing packages.

        Returns True when an install was attempted and succeeded, so the caller
        knows whether re-running its gate is worthwhile.

        v2.9.1 -- this used to fire on *any* missing module.  The distinction
        that matters is not spelled anywhere in the code: a missing
        ``pandas`` is a dependency problem that ``pip install -r
        requirements.txt`` fixes, and a missing ``src.ledger.account`` is the
        project's own code under a name that does not exist, which no install
        can fix and which no ``requirements.txt`` can even mention.  Measured on
        the ledgerline build: the healer woke 55 times, every one for a path
        *inside the project it was building*, and a standard-library project
        with no ``requirements.txt`` made every one of them fail identically --
        burning a cycle per attempt to report an environment problem that was
        actually a wrong import.

        Internal paths are now reported as what they are, under their own event,
        so the operator and the Builder's feedback both see the real cause.
        """
        if not modules or not config.ENV_AUTOHEAL_ENABLED or self._env_heal_done:
            return False
        installable, internal = executors.partition_missing(
            modules, self.project_root)
        if internal:
            log_event("WARNING", "env_autoheal_skipped_internal",
                      {"project": self.project_name, "task_id": task.id,
                       "internal_modules": internal[:6],
                       "note": "these resolve into this project's own "
                               "namespace, so they are wrong import paths, "
                               "not missing dependencies; nothing to install"})
        if not installable:
            return False
        self._env_heal_done = True
        log_event("INFO", "env_autoheal_started",
                  {"project": self.project_name, "task_id": task.id,
                   "missing_modules": installable[:8]})
        heal = await asyncio.to_thread(executors.run_pip_install,
                                       self.project_root)
        if heal["ok"]:
            log_event("INFO", "env_autoheal_completed",
                      {"project": self.project_name, "task_id": task.id})
            return True
        log_event("WARNING", "env_autoheal_failed",
                  {"project": self.project_name, "task_id": task.id,
                   "missing_modules": installable[:8],
                   "detail": heal["detail"][:200]})
        return False

    def _deterministic_spec_report(self, task: TaskSpec) -> Optional[Any]:
        """The free tier of the Spec Critic. Never raises."""
        try:
            return spec_critic.deterministic_checks(task, self.project_root)
        except Exception as exc:  # noqa: BLE001 -- advisory only
            log_event("WARNING", "spec_critic_deterministic_failed",
                      {"task_id": task.id, "error": str(exc)[:200]})
            return None

    async def _critique_authored(self, task: TaskSpec,
                                 plan: Any = None) -> Optional[Any]:
        """Full Spec Critic on one machine-authored task. Never raises.

        Runs the deterministic tier and then, when
        ``SPEC_CRITIC_LLM_ENABLED`` is on, the semantic judgement pass -- the
        same two tiers a hand-written board gets, which is the point: an
        authored task becomes the contract, so it gets the same scrutiny as one
        a human wrote.

        The judgement tier is advisory in the Critic's own design (it can add
        findings but the deterministic tier remains authoritative).  For
        authored tasks, a *blocking* semantic finding still holds activation:
        the cost of dropping one task from a plan of eight is far lower than
        the cost of three Builder attempts and an escalation.

        **Blocking findings are also appended to ``plan.notes``.**  That is the
        difference between a loop that converges and one that repeats itself:
        ``_try_author_from_goal`` feeds ``last_notes`` back into the next
        attempt as repair context, and until this the findings were only
        ``log_event``ed -- so attempt 2 had no idea why attempt 1 was rejected
        and reproduced the same defect.  Observed exactly that way on
        slatepress: a plan with no task owning ``src/slatepress/__init__.py``
        was blocked, and the retry proposed the same board again.
        """
        report = self._deterministic_spec_report(task)
        if report is None or not config.SPEC_CRITIC_ENABLED:
            return report
        if report.blocking:
            self._note_blocking(plan, report)
            return report  # do not spend a call re-deriving it
        if not config.SPEC_CRITIC_LLM_ENABLED:
            return report
        try:
            full = await self.spec_critic.run(task, self.project_root)
        except Exception as exc:  # noqa: BLE001 -- advisory only
            log_event("WARNING", "spec_author_critic_failed",
                      {"project": self.project_name, "task_id": task.id,
                       "error": str(exc)[:200]})
            return report
        try:
            self.status["spec_critic"] = full.to_dict()
        except AttributeError:
            pass
        log_event("INFO", "spec_author_task_critic",
                  {"project": self.project_name, "task_id": task.id,
                   "ok": full.ok,
                   "judgement_independent":
                       getattr(self.spec_critic, "last_independence_collapsed",
                               None),
                   "findings": [f.kind for f in full.findings][:6]})
        if full.blocking:
            self._note_blocking(plan, full)
        return full

    @staticmethod
    def _note_blocking(plan: Any, report: Any) -> None:
        """Push blocking findings into the plan's notes, once each.

        They become the next attempt's repair context.  Deduplicated by kind
        so one board-level defect (no task owns the package initializer) is
        stated once rather than once per task.  The kinds already recorded are
        recovered from the notes this helper itself appends (delimiting the
        kind with ``]: ``, so a kind may itself contain ``]``), and the gate
        holds across tasks *and* across repeated calls on the same plan.
        """
        notes = getattr(plan, "notes", None)
        if notes is None or not isinstance(notes, list):
            return
        prefix = "critic blocked activation ["
        noted = set()
        for existing in notes:
            end = existing.find("]: ", len(prefix))
            if existing.startswith(prefix) and end >= 0:
                noted.add(existing[len(prefix):end])
        for finding in report.blocking[:4]:
            if finding.kind in noted:
                continue
            noted.add(finding.kind)
            notes.append(f"critic blocked activation [{finding.kind}]: "
                         f"{finding.detail}"[:400])

    # ------------------------------------------------------------------
    # Shared memory queries
    # ------------------------------------------------------------------
    @staticmethod
    def _deferred(task: TaskSpec) -> bool:
        """v2.6: skip tasks deferred by quota exhaustion until their reset."""
        if not task.defer_until:
            return False
        try:
            return (datetime.fromisoformat(task.defer_until)
                    > datetime.now(timezone.utc))
        except (TypeError, ValueError):
            return False

    def _query_shared_memory(self, task: TaskSpec) -> Dict[str, Any]:
        query = task.error_signature or f"{task.title} {task.objective}"
        patterns = self.engine.query_patterns(query)
        anti_patterns = self.engine.query_anti_patterns(query)
        return {"patterns": patterns, "anti_patterns": anti_patterns}

    # ------------------------------------------------------------------
    # Small persistence helpers
    # ------------------------------------------------------------------
    def _write_cycle_summary(self, summary: CycleSummary) -> None:
        """Overwrite cycle_summary.json with the compact summary."""
        try:
            path = self.agent_dir / "cycle_summary.json"
            path.write_text(summary.model_dump_json(indent=2) + "\n",
                            encoding="utf-8")
        except OSError as exc:
            log_event("ERROR", "cycle_summary_write_failed",
                      {"project": self.project_name, "error": str(exc)})

    def _append_review_history(self, entry: Dict[str, Any]) -> None:
        try:
            path = self.agent_dir / "review_history.log"
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            log_event("ERROR", "review_history_write_failed",
                      {"project": self.project_name, "error": str(exc)})

    def _review_history_for(self, task_id: str,
                            limit: int = 12) -> List[Dict[str, Any]]:
        """This task's recorded review history, oldest first.

        v2.9: escalation diagnosis reads this instead of re-running anything.
        The file already holds every prior verdict and its feedback, so the
        diagnosis costs one call rather than a re-execution.

        Deliberately bounded and filtered: the log is append-only across the
        whole project lifetime, and feeding all of it to a model would spend
        the diagnosis budget on history that has nothing to do with this task.
        """
        path = self.agent_dir / "review_history.log"
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        rows: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("task_id") or "") != str(task_id):
                continue
            rows.append(row)
        return rows[-limit:]

    def _set_task_status(self, task_id: str, **updates: Any) -> None:
        board = self._load_board()
        if board is None:
            # Never write back a board we could not read: doing so would
            # replace the entire backlog with whatever placeholder we hold.
            log_event("WARNING", "task_status_update_skipped",
                      {"project": self.project_name, "task_id": task_id})
            return
        for task in board.get("tasks", []):
            if task.get("id") == task_id:
                task.update(updates)
                break
        self._save_board(board)

    # ------------------------------------------------------------------
    # IntegrationAgent notification
    # ------------------------------------------------------------------
    def _notify_integration(self, task: TaskSpec, diff: str,
                            files_changed: Optional[List[str]] = None) -> None:
        """Fire-and-forget event; additive data only, never blocks the loop."""
        if self.integration_queue is None:
            return
        changed = list(files_changed) if files_changed else list(task.target_files)
        modules = {Path(f).stem for f in changed}
        for match in __import__("re").findall(r"(?:from|import)\s+([\w\.]+)", diff):
            for part in match.split("."):
                if part:
                    modules.add(part)
        event = IntegrationEvent(
            source_project=self.project_name,
            task_id=task.id,
            files_changed=changed,
            status="completed",
            modules_touched=sorted(modules),
        )
        try:
            self.integration_queue.put_nowait(event.model_dump(mode="json"))
            log_event("INFO", "integration_notified",
                      {"task_id": task.id, "source_project": self.project_name})
        except Exception as exc:
            log_event("WARNING", "integration_notify_failed",
                      {"task_id": task.id, "error": str(exc)})

    # ------------------------------------------------------------------
    # Real execution gate: apply diff to disk + run REAL tests
    # ------------------------------------------------------------------
    def _real_execution(self) -> bool:
        """Real apply+test only in live mode; DEMO_MODE never touches files."""
        return bool(config.REAL_EXECUTION) and not config.DEMO_MODE

    @staticmethod
    def _allowed_targets(task: TaskSpec) -> Set[str]:
        """Declared target files, normalized to forward-slash relative paths."""
        return {str(p).replace("\\", "/") for p in (task.target_files or [])}

    def _out_of_scope(self, paths: Sequence[str],
                      task: TaskSpec) -> List[str]:
        """Entries of *paths* the task is not permitted to write.

        Package-marker ``__init__.py`` files are exempt only as declared
        targets, siblings of targets, or package ancestors of targets
        (``patcher.scaffolding_allowed``) -- trivial scaffolding, nothing else.
        """
        allowed = self._allowed_targets(task)
        return [f for f in paths
                if f not in allowed
                and not patcher.scaffolding_allowed(f, task.target_files)]

    def _test_guard(self, task: TaskSpec) -> test_integrity.TestIntegrityGuard:
        """The frozen test baseline this task is graded against.

        Captured once per task (not per attempt) so a retry is measured
        against the same specification the first attempt saw.
        """
        if self._current_guard_for != task.id:
            self._test_guard_cache = test_integrity.TestIntegrityGuard.for_task(
                self.project_root, task)
            self._current_guard_for = task.id
        return self._test_guard_cache

    def _integrity_violations(self, paths: Sequence[str],
                              task: TaskSpec) -> List[str]:
        """Protected-and-baselined paths this task must not rewrite."""
        return [v["path"] for v in
                self._test_guard(task).violations(paths)]

    async def _apply_and_test(self, task: TaskSpec, diff: str) -> Dict[str, Any]:
        """Snapshot -> apply diff -> run real pytest (+ lint context).

        Returns ``{ok, snapshot, applied, tests, lint, feedback}`` where
        ``feedback`` carries structured patch-failure items for the retry
        path.  On application failure the snapshot is rolled back here;
        on review-failure the caller rolls back before the next attempt.
        v2.5.6: on missing imports, self-heals the environment
        (``pip install -r requirements.txt``, once per task) and re-runs
        the gate with the files still applied.
        """
        out: Dict[str, Any] = {"ok": False, "snapshot": None, "applied": [],
                               "tests": None, "lint": [], "feedback": []}
        parsed = patcher.parse_diff(diff)
        if not parsed:
            out["feedback"] = [ReviewFeedbackItem(
                file="(patch)", line=0,
                issue="diff contains no applicable file sections",
                remediation=("output a real unified diff (---/+++/@@ hunks) "
                             "for the target files"))]
            return out
        touched = [fp.new_path or fp.old_path for fp in parsed
                   if (fp.new_path or fp.old_path)]
        # Pre-apply scope guard.  The post-apply guard below stays the
        # authority (it inspects what actually landed), but it runs *after*
        # ``apply_diff`` has written to disk and relies on rollback to undo an
        # out-of-scope write.  The declared paths are already known here from
        # ``parse_diff``, so refuse before touching the filesystem at all: an
        # overreaching diff then never exists on disk, not even briefly.
        outside = self._out_of_scope(touched, task)
        if outside:
            out["feedback"] = [ReviewFeedbackItem(
                file=f, line=0,
                issue="diff touched a file outside the task scope",
                remediation=("emit hunks for the declared target files only: "
                             + ", ".join(sorted(self._allowed_targets(task)))))
                for f in outside]
            return out

        # Pre-apply test-integrity guard: refuse a rewrite of the tests or
        # pytest config this task is graded against, before anything is
        # written.  ``patcher.apply_diff`` re-checks this (it is the layer
        # the divergence winner path goes through); this pass exists so the
        # Builder gets a precise, actionable rejection.
        tampered = self._integrity_violations(touched, task)
        if tampered:
            self._test_guard(task).log_refusal(tampered, task.id)
            out["feedback"] = [ReviewFeedbackItem(
                file=f, line=0,
                issue="diff modifies a pre-existing test or pytest config",
                remediation=("the tests are the fixed specification for this "
                             "task and must not be edited; implement the "
                             "source change instead, or add a NEW test file"))
                for f in tampered]
            return out

        snapshot = patcher.snapshot_files(self.project_root, touched)
        result = patcher.apply_diff(self.project_root, diff,
                                    guard=self._test_guard(task))
        if not result.ok:
            patcher.restore_snapshot(self.project_root, snapshot)
            out["feedback"] = [
                ReviewFeedbackItem(
                    file=f.file, line=0, issue=f"patch failed: {f.reason}",
                    remediation=("regenerate the diff so hunks match the "
                                 "current file content exactly"))
                for f in result.failed]
            return out

        out["ok"] = True
        out["applied"] = result.applied

        # Scope guard (post-apply, authoritative): the diff may only touch the
        # task's declared files.  Prevents an overreaching diff from corrupting
        # already-approved modules (observed failure mode with weaker builder
        # models).  Package-marker __init__.py files are exempt ONLY as
        # declared targets, siblings of targets, or package ancestors of
        # targets (patcher.scaffolding_allowed) -- trivial scaffolding only.
        outside = self._out_of_scope(result.applied, task)
        if outside:
            patcher.restore_snapshot(self.project_root, snapshot)
            out["ok"] = False
            out["applied"] = []
            out["feedback"] = [ReviewFeedbackItem(
                file=f, line=0,
                issue="diff touched a file outside the task scope",
                remediation=("emit hunks for the declared target files only: "
                             + ", ".join(sorted(self._allowed_targets(task)))))
                for f in outside]
            return out

        out["snapshot"] = snapshot
        tests = executors.run_tests(self.project_root)

        # v2.5.6 environment self-heal: missing imports -> install the
        # project's own requirements once per task, then re-run the gate
        # with the files still applied.  Requirements-only policy, and (since
        # v2.9.1) only for modules that are actually installable.
        if await self._maybe_autoheal(task, tests.get("missing_modules") or ()):
            tests = executors.run_tests(self.project_root)

        out["tests"] = {"passed": tests["passed"], "failed": tests["failed"],
                        "skipped": tests.get("skipped", 0),
                        "available": tests["available"],
                        "detail": tests["detail"],
                        "missing_modules": tests.get("missing_modules", []),
                        "failing_tests": tests.get("failing_tests", []),
                        "baseline_passed": 0}
        out["lint"] = executors.run_lint(self.project_root, result.applied)
        return out

    async def _apply_and_objective_gate(self, task: TaskSpec,
                                        diff: str) -> Dict[str, Any]:
        """Snapshot -> apply -> run the task's experiment command (v2.6).

        Same safety rails as :meth:`_apply_and_test` (parse, snapshot,
        scope guard, rollback, env self-heal); the difference is the gate:
        instead of pytest the task's ``run_command`` executes and its
        metrics JSON (``results_path``) becomes the authoritative evidence.

        Returns ``{ok, snapshot, applied, objective, lint, feedback}``.
        """
        out: Dict[str, Any] = {"ok": False, "snapshot": None, "applied": [],
                               "objective": None, "lint": [], "feedback": []}
        parsed = patcher.parse_diff(diff)
        if not parsed:
            out["feedback"] = [ReviewFeedbackItem(
                file="(patch)", line=0,
                issue="diff contains no applicable file sections",
                remediation=("output a real unified diff (---/+++/@@ hunks) "
                             "for the target files"))]
            return out
        touched = [fp.new_path or fp.old_path for fp in parsed
                   if (fp.new_path or fp.old_path)]
        # Pre-apply scope guard -- see _apply_and_test for the rationale.
        outside = self._out_of_scope(touched, task)
        if outside:
            out["feedback"] = [ReviewFeedbackItem(
                file=f, line=0,
                issue="diff touched a file outside the task scope",
                remediation=("emit hunks for the declared target files only: "
                             + ", ".join(sorted(self._allowed_targets(task)))))
                for f in outside]
            return out
        tampered = self._integrity_violations(touched, task)
        if tampered:
            self._test_guard(task).log_refusal(tampered, task.id)
            out["feedback"] = [ReviewFeedbackItem(
                file=f, line=0,
                issue="diff modifies a pre-existing test or pytest config",
                remediation=("the tests are the fixed specification for this "
                             "task and must not be edited; implement the "
                             "source change instead, or add a NEW test file"))
                for f in tampered]
            return out

        snapshot = patcher.snapshot_files(self.project_root, touched)
        result = patcher.apply_diff(self.project_root, diff,
                                    guard=self._test_guard(task))
        if not result.ok:
            patcher.restore_snapshot(self.project_root, snapshot)
            out["feedback"] = [
                ReviewFeedbackItem(
                    file=f.file, line=0, issue=f"patch failed: {f.reason}",
                    remediation=("regenerate the diff so hunks match the "
                                 "current file content exactly"))
                for f in result.failed]
            return out

        out["ok"] = True
        out["applied"] = result.applied

        outside = self._out_of_scope(result.applied, task)
        if outside:
            patcher.restore_snapshot(self.project_root, snapshot)
            out["ok"] = False
            out["applied"] = []
            out["feedback"] = [ReviewFeedbackItem(
                file=f, line=0,
                issue="diff touched a file outside the task scope",
                remediation=("emit hunks for the declared target files only: "
                             + ", ".join(sorted(self._allowed_targets(task)))))
                for f in outside]
            return out

        out["snapshot"] = snapshot
        metric = task.metric
        objective = executors.run_objective(
            self.project_root, task.run_command or "",
            results_path=task.results_path,
            timeout=task.max_runtime_s,
            required_fields=(metric.required_fields if metric else None))

        # Same environment self-heal as the pytest gate (once per task).
        if await self._maybe_autoheal(
                task, objective.get("missing_modules") or ()):
            objective = executors.run_objective(
                self.project_root, task.run_command or "",
                results_path=task.results_path,
                timeout=task.max_runtime_s,
                required_fields=(metric.required_fields if metric else None))

        objective["primary"] = _primary_value(
            objective.get("metrics") or {},
            metric.name if metric else "primary")
        out["objective"] = objective
        out["lint"] = executors.run_lint(self.project_root, result.applied)
        return out

    # ------------------------------------------------------------------
    # Main task processing (Builder-Reviewer loop, max 3 attempts)
    # ------------------------------------------------------------------
    async def process_task(self, task: TaskSpec) -> None:
        started = time.monotonic()
        self._env_heal_done = False
        self._divergence_done = False
        self.current_task_id = task.id
        self.status["current_task_id"] = task.id
        self.status["status"] = "running"
        log_event("INFO", "task_started",
                  {"project": self.project_name, "task_id": task.id,
                   "task_type": task.task_type, "priority": task.priority,
                   "real_execution": self._real_execution()})

        shared = self._query_shared_memory(task)
        # v2.8: the queries are retained only as a *read-only* context summary.
        # v2.9: the Council was the consumer and it is gone, so this is now
        # purely an audit row -- nothing from here reaches a Builder prompt.
        # See the note at the attempt-3 retry payload.
        if shared.get("patterns") or shared.get("anti_patterns"):
            log_event("INFO", "shared_memory_context_available",
                      {"task_id": task.id,
                       "pattern_count": len(shared.get("patterns") or []),
                       "anti_pattern_count": len(
                           shared.get("anti_patterns") or []),
                       "consumed_by": "audit_only"})

        plan: Optional[Plan] = None
        last_diff = ""
        last_feedback = ""
        verdict: Optional[ReviewVerdict] = None
        model_used = ""
        builder_usage: Optional[Dict[str, Any]] = None
        objective_result: Optional[ObjectiveGateResult] = None
        objective_task = task.metric is not None and bool(task.run_command)
        active_snapshot: Optional[Path] = None  # pre-apply backup for rollback
        files_applied_count = 0
        applied_files: List[str] = []
        # Tests-first gate baseline: pre-task green test count.
        baseline_passed = 0
        baseline_skipped = 0
        baseline_test_ids: List[str] = []
        if self._real_execution() and task.requires_new_tests \
                and not objective_task:
            baseline = executors.run_tests(self.project_root)
            baseline_passed = baseline["passed"]
            baseline_skipped = baseline.get("skipped", 0)
            # Node IDs, so the gate can tell a *grown* suite from a *replaced*
            # one (see test_set_gate_block).
            baseline_test_ids = executors.collect_test_ids(self.project_root)
            log_event("INFO", "baseline_tests_recorded",
                      {"task_id": task.id, "passed": baseline_passed,
                       "skipped": baseline_skipped,
                       "collected_ids": len(baseline_test_ids)})

        def _rollback() -> None:
            """Undo any applied files; guarantees a clean tree between attempts."""
            nonlocal active_snapshot
            if active_snapshot is not None:
                patcher.restore_snapshot(self.project_root, active_snapshot)
                active_snapshot = None

        try:
            for attempt in range(1, config.MAX_ATTEMPTS + 1):
                # Per-task conversation state, kept separate from cross-task
                # continuity (which is ``BuilderAgent._line_notes``, not this).
                #
                # v2.9: this reset is new, and it fixes a real bug.  ``reset()``
                # used to fire *only* at ``attempt == MAX_ATTEMPTS``, so
                # ``_history`` was never cleared between tasks: task 2's attempt
                # 1 opened with task 1's objective and task 1's diff still in the
                # conversation. The model was being asked to write src/b.py while
                # looking at a transcript about src/a.py, and the reviewer
                # feedback from the previous task was silently in scope.
                #
                # Attempts 1 and 2 still share a conversation within a task, which
                # is what retry feedback needs; only the task boundary resets.
                if attempt == 1:
                    self.builder.reset()
                    if config.LINE_CONTINUITY:
                        log_event("DEBUG", "builder_task_boundary",
                                  {"task_id": task.id,
                                   "line_notes": len(self.builder._line_notes)})
                # Final attempt: fresh conversation, compact state only.
                if attempt == config.MAX_ATTEMPTS:
                    self.builder.reset()
                    log_event("INFO", "builder_conversation_reset",
                              {"task_id": task.id, "attempt": attempt})

                # v2.9.1: a doom loop is a *stop* signal, not a log line.
                #
                # ``steering`` detects two shapes of "another attempt cannot
                # help" -- the same failure signature three times, or a
                # byte-identical diff twice -- and reports both at ERROR.  Until
                # now nothing consumed them: the loop kept planning, kept
                # building, kept getting the identical diff, and handed the
                # task back again.  Measured on a real board: 16 doom-loop
                # detections, 29 gate blocks, 0 interventions, all of it on one
                # task whose test could not pass by construction.
                #
                # So the signal now ends the attempt loop and goes straight to
                # escalation with the diagnosis, which is the one thing that can
                # actually change the outcome.  Cheap, deterministic, and it
                # converts wasted quota into an actionable board entry.
                doom = self._doom_loop_signals(task.id)
                if doom:
                    log_event("WARNING", "doom_loop_halts_attempts",
                              {"task_id": task.id, "attempt": attempt,
                               "signals": [s.kind for s in doom],
                               "evidence": [s.evidence for s in doom][:2],
                               "note": "further identical attempts cannot "
                                       "help; escalating with a diagnosis "
                                       "instead of retrying"})
                    await self._escalate(
                        task, verdict, self.builder.strategy_model_role
                        or "builder", 0.0,
                        reason="doom loop: " + "; ".join(
                            s.detail for s in doom)[:400])
                    return

                # 1. Plan (mandatory before Builder).  Plan-level divergence:
                #    multiple Planner lenses merged deterministically.
                if plan is None:
                    if config.PLAN_DIVERGENCE:
                        plan = await self._plan_with_lenses(task)
                    else:
                        plan = await self.planner.run(task)
                    log_event("INFO", "plan_created",
                              {"task_id": task.id, "steps": len(plan.steps),
                               "plan_divergence": bool(config.PLAN_DIVERGENCE)})

                # 2. Build.  Divergent tasks fan out candidate strategies on
                #    the first attempt and converge on the deterministic
                #    winner; everything else uses the single-builder path
                #    (model escalation: attempt 1 -> builder_initial,
                #    retries -> builder_patch).
                required_tools = resolve_required_tools(task, "builder")
                precomputed_gate: Optional[Dict[str, Any]] = None
                build: Optional[BuilderOutput] = None
                if attempt == 1 and not self._divergence_done \
                        and divergence.enabled(task, self.project_root):
                    self._divergence_done = True
                    selected = await self._run_divergence(task, plan)
                    if selected is not None:
                        build, precomputed_gate, builder_usage = selected
                if build is None:
                    self.builder.current_attempt = attempt
                    self.builder.is_retry = attempt > 1
                    builder_context = None
                    if attempt > 1:
                        # COMPACT retry payload -- never full history.
                        # Attempt 2: exactly task_id / current_diff /
                        # last_feedback / max_retries_left / original plan.
                        builder_context = {
                            "task_id": task.id,
                            "current_diff": last_diff[:4000],
                            "last_feedback": last_feedback,
                            "max_retries_left": config.MAX_ATTEMPTS - attempt,
                            "plan": plan.model_dump(mode="json"),
                        }
                        if attempt == config.MAX_ATTEMPTS:
                            # Final attempt gets the fresh-builder compact
                            # state: spec + latest feedback + plan.
                            #
                            # v2.8: patterns/anti-patterns are NO LONGER
                            # injected.  The v2.6 injection appended a verbatim
                            # copy of ``task.objective`` (already delivered in
                            # <<<TASK_JSON>>> on every attempt) plus raw
                            # provider error prose -- ~1.3 KB per call for zero
                            # information.  Retry guidance now comes from the
                            # actual gate evidence in ``last_feedback``, and
                            # systemic looping is detected deterministically
                            # by :mod:`steering` instead of being re-explained
                            # to a Builder that already saw the text.
                            builder_context["spec"] = task.model_dump(
                                include={"id", "title", "objective", "target_files",
                                         "acceptance_criteria", "task_type"})
                        log_event("INFO", "retry_compact_context",
                                  {"task_id": task.id, "attempt": attempt,
                                   "fields": sorted(builder_context.keys()),
                                   "max_retries_left": builder_context["max_retries_left"]})
                    build = await self.builder.run(task, plan, required_tools,
                                                   context=builder_context)
                    builder_usage = self.llm.usage_snapshot()
                last_diff = build.diff
                model_used = "divergence" if precomputed_gate is not None \
                    else (self.builder.model_role or "builder_unknown")

                # 3. Real execution gate: apply to disk + run REAL pytest (or
                #    the task's objective gate).  Divergent winners arrive
                #    with their gate precomputed from the candidate round.
                real_tests: Optional[Dict[str, int]] = None
                review_context: Optional[str] = None
                patch_failed = False
                if self._real_execution() or precomputed_gate is not None:
                    if precomputed_gate is not None:
                        gate = precomputed_gate
                    elif objective_task:
                        gate = await self._apply_and_objective_gate(task,
                                                                    build.diff)
                    else:
                        gate = await self._apply_and_test(task, build.diff)
                    if not gate["ok"]:
                        # Patch application failed: skip the LLM review and
                        # feed the structured patch errors straight back.
                        patch_failed = True
                        verdict = ReviewVerdict(
                            task_id=task.id, verdict="CHANGES_REQUESTED",
                            automated_tests=AutomatedTests(passed=0, failed=1),
                            feedback=gate["feedback"],
                            reviewer_notes="real-execution: patch not applicable")
                        log_event("WARNING", "patch_application_failed",
                                  {"task_id": task.id,
                                   "feedback": [f.issue for f in gate["feedback"]]})
                    elif objective_task:
                        obj = gate["objective"] or {}
                        blocked, reason = objective_gate_block(
                            task.metric, obj, task.metric.baseline)
                        if blocked:
                            # Objective gate failed: keep the snapshot live
                            # so the CHANGES_REQUESTED rollback below removes
                            # the applied-but-unverified files.
                            patch_failed = True
                            active_snapshot = gate["snapshot"]
                            verdict = ReviewVerdict(
                                task_id=task.id,
                                verdict="CHANGES_REQUESTED",
                                automated_tests=AutomatedTests(passed=0,
                                                               failed=1),
                                feedback=[ReviewFeedbackItem(
                                    file="(gate)", line=0, issue=reason,
                                    remediation=(
                                        "fix the experiment so the objective "
                                        "gate passes: metric / split_hash / "
                                        "target / improvement"))],
                                reviewer_notes=("real-execution: objective "
                                                "gate"))
                            log_event("WARNING", "objective_gate_blocked",
                                      {"task_id": task.id, "reason": reason,
                                       "metrics": obj.get("metrics"),
                                       "detail": str(obj.get("detail", ""))[:300]})
                        else:
                            active_snapshot = gate["snapshot"]
                            applied_files = gate["applied"]
                            files_applied_count = len(applied_files)
                            objective_result = ObjectiveGateResult(
                                task_id=task.id, passed=True,
                                metrics=obj.get("metrics") or {},
                                primary=obj.get("primary"),
                                baseline=task.metric.baseline,
                                reason=reason or "objective gate passed",
                                runtime_s=float(obj.get("runtime_s", 0.0)),
                                split_hash=obj.get("split_hash"))
                            context_parts = [
                                f"REAL EXECUTION: applied files: {applied_files}",
                                ("OBJECTIVE GATE: passed -- "
                                 f"{task.metric.name}={obj.get('primary')} "
                                 f"({task.metric.direction}), "
                                 f"target={task.metric.target}, "
                                 f"baseline={task.metric.baseline}, "
                                 f"runtime_s={obj.get('runtime_s')}"),
                                ("METRICS JSON: "
                                 + json.dumps(obj.get("metrics") or {})[:600]),
                            ]
                            if gate["lint"]:
                                context_parts.append(
                                    f"LINT issues: {gate['lint']}")
                            review_context = "\n".join(context_parts)
                            log_event("INFO", "objective_gate_passed",
                                      {"task_id": task.id,
                                       "applied": applied_files,
                                       "metrics": obj.get("metrics")})
                    elif task.requires_new_tests and real_tests_block(
                            gate["tests"], baseline_passed):
                        # Tests-first gate: the task's own tests must RUN.
                        # The diff DID apply -- keep the snapshot live so the
                        # CHANGES_REQUESTED/escalation rollback below removes
                        # the applied-but-unverified files.
                        patch_failed = True
                        active_snapshot = gate["snapshot"]
                        tests_now = gate["tests"]
                        # v2.9.1: the gate used to emit ONE message for both
                        # "the module does not exist yet" and "the module exists
                        # and imports a path that does not resolve", and its
                        # remediation text only described the first:
                        #
                        #   issue:       task tests did not run: passed 18 ==
                        #                baseline 18 (skipped 2); missing
                        #                modules: src.journal
                        #   remediation: implement the target module so its
                        #                pre-seeded test file executes
                        #
                        # The Builder obeys, writes the module again with the
                        # same wrong import, and produces an identical diff.
                        # That is a doom loop manufactured by the feedback, and
                        # it is how 3 of 7 tasks on a real authored board died:
                        # ``src.journal`` instead of ``src.ledger.journal``,
                        # ``ledger`` instead of ``src.ledger``.
                        #
                        # So an unresolvable import gets its own diagnosis and
                        # its own remediation, naming the paths.
                        missing = [str(m) for m in
                                   (tests_now.get("missing_modules") or [])]
                        invented = [(str(n), str(m)) for n, m in
                                    (tests_now.get("missing_names") or [])]
                        applied_paths = [str(p) for p in (applied_files or [])]
                        if invented:
                            # The module exists; the *name* does not.  Only
                            # ModuleNotFoundError used to be extracted, so this
                            # arrived as a bare "tests did not run" and the
                            # remediation -- "implement the target module" --
                            # described work the Builder had already done.
                            # Measured cost: nine attempts on one task whose
                            # only defect was importing EntryType from a module
                            # that defines Side, Posting and JournalEntry.
                            pairs = ", ".join(f"{n!r} from {m!r}"
                                              for n, m in invented[:6])
                            issue = (
                                f"task tests did not run: passed "
                                f"{tests_now['passed']} == baseline "
                                f"{baseline_passed} (skipped "
                                f"{tests_now.get('skipped', 0)}). "
                                f"INVENTED IMPORT NAMES: {pairs}")
                            remediation = (
                                "the module exists, but you imported name(s) "
                                "it does not define. Import only the names "
                                "the task's API spec and its test actually "
                                f"list. Remove or replace: {pairs}. Do not "
                                "invent extra imports, enums or types.")
                        elif missing:
                            # Name the exact replacement, not an example.  A
                            # remediation that only shows the *shape* of the fix
                            # invites the model to guess: told
                            # ``from src.journal import X`` was wrong, the
                            # Builder replied ``from src.ledger.journal.entry
                            # import X`` -- right package, invented submodule --
                            # and burned another attempt.  Naming the file that
                            # does exist ends the guessing.
                            fixes: List[str] = []
                            for mod in missing[:6]:
                                hint = spec_critic.nearest_project_module(
                                    mod, self.project_root)
                                if hint:
                                    dotted = hint[:-3].replace("/", ".") \
                                        if hint.endswith(".py") \
                                        else hint.replace("/", ".")
                                    fixes.append(f"{mod} -> {dotted}")
                                else:
                                    fixes.append(f"{mod} -> (no matching file; "
                                                 f"create it or drop the import)")
                            issue = (
                                f"task tests did not run: passed "
                                f"{tests_now['passed']} == baseline "
                                f"{baseline_passed} (skipped "
                                f"{tests_now.get('skipped', 0)}). "
                                f"UNRESOLVABLE IMPORTS: "
                                f"{', '.join(missing[:6])}")
                            remediation = (
                                "your files applied, so the target module "
                                "exists -- but an import you wrote names a "
                                "module that does not resolve, so the test "
                                "still skips and nothing ran. Correct these "
                                "import paths to the modules that actually "
                                f"exist: {'; '.join(fixes)}. Use the exact "
                                "dotted paths above; do not invent "
                                "submodules.")
                        else:
                            issue = (
                                f"task tests did not run: passed "
                                f"{tests_now['passed']} == baseline "
                                f"{baseline_passed} (skipped "
                                f"{tests_now.get('skipped', 0)})"
                                + (f"; failing: "
                                   f"{', '.join(tests_now.get('failing_tests', [])[:5])}"
                                   if tests_now.get("failing_tests") else ""))
                            remediation = (
                                f"implement the target module ({', '.join(applied_paths) or 'declared target_files'})"
                                " so its pre-seeded test file executes without "
                                "skips")
                        verdict = ReviewVerdict(
                            task_id=task.id, verdict="CHANGES_REQUESTED",
                            automated_tests=AutomatedTests(
                                passed=tests_now["passed"],
                                failed=tests_now["failed"]),
                            feedback=[ReviewFeedbackItem(
                                file="(gate)", line=0, issue=issue,
                                remediation=remediation)],
                            reviewer_notes="real-execution: new-tests gate")
                        log_event("WARNING", "new_tests_gate_blocked",
                                  {"task_id": task.id,
                                   "passed": tests_now["passed"],
                                   "baseline": baseline_passed,
                                   "skipped": tests_now.get("skipped", 0),
                                   "failing_tests": tests_now.get("failing_tests", []),
                                   "missing_modules": tests_now.get("missing_modules", []),
                                   "test_detail": tests_now.get("detail", "")[:300],
                                   # Include the diff head so a broken
                                   # conftest/module is debuggable after the
                                   # rollback has removed it from disk.
                                   "diff_head": build.diff[:600]})
                    elif task.requires_new_tests and (
                            _test_set_reason := test_set_gate_block(
                                baseline_test_ids,
                                executors.collect_test_ids(self.project_root),
                                required=True)):
                        # Node-ID level tests-first gate (v2.8): the suite must
                        # have *grown*, and nothing pre-existing may have been
                        # removed.  The count rule above cannot see either.
                        patch_failed = True
                        active_snapshot = gate["snapshot"]
                        tests_now = gate["tests"]
                        verdict = ReviewVerdict(
                            task_id=task.id, verdict="CHANGES_REQUESTED",
                            automated_tests=AutomatedTests(
                                passed=tests_now["passed"],
                                failed=tests_now["failed"]),
                            feedback=[ReviewFeedbackItem(
                                file="(gate)", line=0, issue=str(_test_set_reason),
                                remediation=("keep every existing test intact "
                                             "and add a new test covering "
                                             "the behaviour you implemented"))],
                            reviewer_notes="real-execution: test-set gate")
                        log_event("WARNING", "test_set_gate_blocked",
                                  {"task_id": task.id,
                                   "reason": str(_test_set_reason),
                                   "diff_head": build.diff[:300]})
                    else:
                        active_snapshot = gate["snapshot"]
                        applied_files = gate["applied"]
                        files_applied_count = len(applied_files)
                        real_tests = gate["tests"]
                        real_tests["baseline_passed"] = baseline_passed
                        context_parts = [
                            f"REAL EXECUTION: applied files: {applied_files}",
                            f"REAL TESTS: passed={real_tests['passed']} "
                            f"failed={real_tests['failed']} "
                            f"skipped={real_tests.get('skipped', 0)}"]
                        if task.requires_new_tests:
                            context_parts.append(
                                "TESTS-FIRST: the pre-seeded tests for this "
                                "task must run and pass (no skips for this "
                                f"task's file). Baseline was "
                                f"{baseline_passed} passed.")
                        # v2.8: name the failing tests. The Reviewer was given
                        # only a 400-char tail of pytest output, so with several
                        # failures it reasoned blind -- it argued with the
                        # Builder across four escalated attempts about code it
                        # could not see, and invented a test name that does not
                        # exist. The node IDs were already collected; they were
                        # simply never forwarded. Evidence beats interpretation.
                        failing = real_tests.get("failing_tests") or []
                        if failing:
                            context_parts.append(
                                "FAILING TESTS (these are the authoritative "
                                "failures; judge the diff against them): "
                                + ", ".join(str(t) for t in failing[:10])
                                + (f" (+{len(failing) - 10} more)"
                                   if len(failing) > 10 else ""))
                        skipped_here = real_tests.get("skipped") or 0
                        if skipped_here and task.requires_new_tests:
                            context_parts.append(
                                f"NOTE: {skipped_here} test(s) SKIPPED. "
                                "Tests-first tasks require this task's tests "
                                "to actually run.")
                        if real_tests.get("missing_modules"):
                            context_parts.append(
                                "MISSING MODULES (environment problem, not a "
                                "code problem): "
                                + ", ".join(
                                    str(m) for m in
                                    real_tests["missing_modules"][:5]))
                        if real_tests.get("detail"):
                            # 400 chars showed at most one failure of several;
                            # the Reviewer filled the gap with invention.
                            context_parts.append(
                                "test output tail:\n"
                                + str(real_tests["detail"])[:2000])
                        if gate["lint"]:
                            context_parts.append(f"LINT issues: {gate['lint']}")
                        review_context = "\n".join(context_parts)
                        log_event("INFO", "real_gate_passed",
                                  {"task_id": task.id,
                                   "applied": applied_files,
                                   "tests": real_tests})

                # 4. Review (skipped only when the patch itself failed).
                if not patch_failed:
                    verdict = await self.reviewer.run(
                        task, build.diff, plan,
                        context=review_context,
                        test_results=real_tests,
                        objective_passed=bool(
                            objective_result and objective_result.passed))
                if objective_result is not None:
                    # Objective evidence is authoritative: attach it and
                    # normalize the test counts so pattern promotion sees
                    # the same green-build contract as pytest tasks.
                    verdict.objective_gate = objective_result
                    verdict.automated_tests = AutomatedTests(
                        passed=max(1, verdict.automated_tests.passed),
                        failed=0)
                self._append_review_history({
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "task_id": task.id,
                    "attempt": attempt,
                    "verdict": verdict.verdict,
                    "failed_tests": verdict.automated_tests.failed,
                    "real_execution": self._real_execution(),
                    "applied_files": files_applied_count,
                    # Persist the reviewer's actual findings -- the board's
                    # last_feedback gets overwritten at escalation, so this
                    # is the durable record of WHY a cycle was rejected.
                    "feedback": [fb.model_dump() for fb in verdict.feedback][:3],
                })

                # 5. APPROVED -> complete (keep applied files).
                if verdict.verdict == "APPROVED":
                    patcher.discard_snapshot(active_snapshot)
                    active_snapshot = None
                    model_router.record_outcome(builder_usage, True,
                                                self.workspace_root)
                    duration_s = round(time.monotonic() - started, 2)
                    self._set_task_status(task.id, status="done",
                                          retry_count=attempt - 1,
                                          completed_at=time.strftime(
                                              "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                          last_feedback=None)
                    # v2.9: tell the line what this task established, so the
                    # next one starts from the settled API instead of
                    # re-deriving it.
                    self.builder.note_task_complete(
                        task, applied_files, attempt)
                    # v2.8: no pattern promotion on success.  The v2.6 record
                    # stored a copy of task.objective plus a diff hash; nothing
                    # read it back into any prompt, and no lesson was ever
                    # extracted from the failure.  Success is now recorded as a
                    # steering datapoint instead, which is what actually clears
                    # a doom-loop counter.
                    self.steering.record_attempt(self.project_name,
                        steering.AttemptRecord(
                            task_id=task.id, outcome="APPROVED", attempt=attempt,
                            diff_hash=steering.diff_fingerprint(build.diff),
                            duration_s=duration_s,
                            tokens=_usage_tokens(builder_usage)))
                    summary = CycleSummary(
                        task_id=task.id, attempts=attempt,
                        verdict="APPROVED", model_used=model_used,
                        duration_s=duration_s,
                        files_applied=files_applied_count)
                    self._write_cycle_summary(summary)
                    self._notify_integration(task, build.diff,
                                             files_changed=applied_files or None)
                    log_event("INFO", "task_completed",
                              {"project": self.project_name, "task_id": task.id,
                               "attempts": attempt, "duration_s": duration_s,
                               "model_used": model_used,
                               "files_applied": files_applied_count})
                    await self._notify_domain_pack(task, objective_result)
                    return

                # 6. CHANGES_REQUESTED -> rollback, then retry or escalate.
                _rollback()
                model_router.record_outcome(builder_usage, False,
                                            self.workspace_root)
                task.retry_count = attempt
                last_feedback = "; ".join(
                    f"{item.file}:{item.line} {item.issue} -> {item.remediation}"
                    for item in verdict.feedback) or "changes requested"
                self._set_task_status(task.id, status="pending",
                                      retry_count=attempt,
                                      defer_until=None,
                                      last_feedback=last_feedback[:500])
                summary = CycleSummary(
                    task_id=task.id, attempts=attempt,
                    verdict="CHANGES_REQUESTED", model_used=model_used,
                    duration_s=round(time.monotonic() - started, 2),
                    files_applied=0)
                self._write_cycle_summary(summary)
                self.steering.record_attempt(self.project_name,
                    steering.AttemptRecord(
                        task_id=task.id, outcome="CHANGES_REQUESTED",
                        attempt=attempt,
                        signature=task.error_signature,
                        diff_hash=steering.diff_fingerprint(build.diff),
                        duration_s=summary.duration_s,
                        tokens=_usage_tokens(builder_usage),
                        error_summary=last_feedback[:200]))
                log_event("WARNING", "task_changes_requested",
                          {"project": self.project_name, "task_id": task.id,
                           "attempt": attempt,
                           "remaining_attempts": config.MAX_ATTEMPTS - attempt})

            # 7. Retries exhausted -> escalate (tree already rolled back).
            await self._escalate(task, verdict, model_used,
                                 round(time.monotonic() - started, 2))
        except LLMUnavailableError as exc:
            # v2.6 capacity exhaustion is NOT a task failure: keep the task
            # pending (attempt budget untouched) and defer it to the
            # earliest provider quota reset.  Escalating would burn the
            # retry budget on an infrastructure condition.
            _rollback()
            defer_until = model_router.next_reset_iso(self.workspace_root)
            log_event("WARNING", "task_deferred_llm_unavailable",
                      {"project": self.project_name, "task_id": task.id,
                       "error": str(exc), "defer_until": defer_until})
            self._set_task_status(task.id, status="pending",
                                  defer_until=defer_until,
                                  last_feedback=(
                                      f"deferred (LLM capacity): {exc}")[:500])
        except Exception as exc:  # noqa: BLE001 -- per-task isolation
            _rollback()  # never leave a broken tree behind
            log_event("ERROR", "task_processing_exception",
                      {"project": self.project_name, "task_id": task.id,
                       "error": str(exc), "traceback": traceback.format_exc()})
            await self._escalate(task, None, "n/a",
                                 round(time.monotonic() - started, 2),
                                 reason=f"exception: {exc}")
        finally:
            self.current_task_id = None
            self.status["current_task_id"] = None

    # ------------------------------------------------------------------
    # v2.6 divergence: plan lenses + candidate strategies + blind judge
    # ------------------------------------------------------------------
    async def _plan_with_lenses(self, task: TaskSpec) -> Plan:
        """Divergent planning: one Planner run per lens, merged."""
        plans: List[Plan] = []
        for name, lens in divergence.PLAN_LENSES:
            plans.append(await self.planner.run(
                task, context=(f"Planning lens: {name} -- {lens}. "
                               "Optimize THIS plan for that lens.")))
        merged = divergence.merge_plans(task.id, plans)
        log_event("INFO", "plan_divergence_merged",
                  {"task_id": task.id, "lenses": len(plans),
                   "steps": len(merged.steps)})
        return merged

    async def _judge_candidates(self, task: TaskSpec,
                                records: List[divergence.CandidateRecord]
                                ) -> Optional[int]:
        """Blind near-tie judge; returns a winner index or None."""
        label = await self.judge.run(task, divergence.blind_payload(records))
        if not label:
            return None
        for index, record in enumerate(records):
            if record.label == label:
                log_event("INFO", "judge_selected_candidate",
                          {"task_id": task.id, "label": label})
                return index
        return None

    async def _run_divergence(self, task: TaskSpec, plan: Plan
                              ) -> Optional[Tuple[BuilderOutput, Dict[str, Any],
                                                  Optional[Dict[str, Any]]]]:
        """Fan out candidate strategies, gate each, converge on a winner.

        Every candidate is built and gated against a clean tree (the
        snapshot is restored after scoring), then the winner is re-applied
        so the normal review path sees it on disk.  Returns
        ``(winner_build, gate, usage)`` or ``None``.
        """
        strategies = divergence.load_strategies(task, self.project_root)
        if not config.DEMO_MODE:
            available = get_governor(self.workspace_root) \
                .total_available_requests()
            budget_cap = max(1, available // 2)  # ~2 calls per candidate
            if len(strategies) > budget_cap:
                log_event("WARNING", "divergence_capped_by_quota",
                          {"task_id": task.id,
                           "requested": len(strategies),
                           "allowed": budget_cap, "available": available})
                strategies = strategies[:budget_cap]
        log_event("INFO", "divergence_started",
                  {"task_id": task.id, "candidates": len(strategies),
                   "strategies": [s.name for s in strategies]})

        records: List[divergence.CandidateRecord] = []
        for index, strategy in enumerate(strategies):
            label = chr(ord("A") + index)
            task_copy = task.model_copy(deep=True)
            task_copy.objective = (
                task.objective
                + f"\n\nSTRATEGY ({strategy.name}): "
                + (strategy.approach or "no extra guidance"))
            self.builder.reset()
            self.builder.current_attempt = 1
            self.builder.is_retry = False
            self.builder.strategy_model_role = strategy.model_role or None
            build = await self.builder.run(
                task_copy, plan, resolve_required_tools(task, "builder"))
            usage = self.llm.usage_snapshot()

            gate: Optional[Dict[str, Any]] = None
            blocked = False
            if self._real_execution():
                if task.metric is not None and task.run_command:
                    gate = await self._apply_and_objective_gate(task_copy,
                                                                build.diff)
                    blocked, _ = objective_gate_block(
                        task.metric, gate.get("objective") or {},
                        task.metric.baseline)
                else:
                    gate = await self._apply_and_test(task_copy, build.diff)
                    tests = gate.get("tests") or {}
                    blocked = bool(
                        not gate.get("ok")
                        or tests.get("failed", 1) > 0
                        or (task.requires_new_tests
                            and tests.get("passed", 0) <= 0))
            else:
                # Demo/simulated gate: deterministic evidence without disk.
                gate = self._simulated_gate(task)

            record = divergence.record_from_gate(
                label, strategy, build.diff, gate,
                self.builder.model_role or "", usage, blocked, task)
            records.append(record)
            if gate is not None and gate.get("snapshot") is not None:
                patcher.restore_snapshot(self.project_root, gate["snapshot"])
            divergence.store_candidate(self.project_root, task.id, record)
            log_event("INFO", "candidate_scored",
                      {"task_id": task.id, "label": label,
                       "strategy": strategy.name, "gate_ok": record.gate_ok,
                       "score": round(record.score(), 4),
                       "tests_failed": record.tests_failed,
                       "metric": record.metric})

        if not records:
            return None
        winner_index = divergence.select_winner(records)
        if config.DIVERGENCE_JUDGE_ENABLED \
                and divergence.is_near_tie(records, winner_index):
            judged = await self._judge_candidates(task, records)
            if judged is not None and \
                    records[judged].gate_ok >= records[winner_index].gate_ok:
                winner_index = judged
            elif judged is not None:
                # Invariant: the blind judge never promotes a gate-failing
                # candidate over a gate-passing one.
                log_event("WARNING", "judge_override_rejected",
                          {"task_id": task.id,
                           "judge_pick": records[judged].label,
                           "deterministic": records[winner_index].label})
        # Merge round (v2.6): combine gate-passing near-tie candidates.
        if config.DIVERGENCE_MERGE_ENABLED \
                and sum(1 for r in records if r.gate_ok) >= 2 \
                and divergence.is_near_tie(records, winner_index):
            merged = await self._merge_candidates(task, records, winner_index)
            if merged is not None and merged.score() >= records[winner_index].score():
                log_event("INFO", "merge_accepted",
                          {"task_id": task.id,
                           "label": merged.label,
                           "score": round(merged.score(), 4)})
                records.append(merged)
                winner_index = len(records) - 1
            elif merged is not None:
                log_event("INFO", "merge_rejected",
                          {"task_id": task.id,
                           "merged_score": round(merged.score(), 4),
                           "best_score": round(records[winner_index].score(), 4)})
        divergence.log_selection(task.id, records, winner_index)
        winner = records[winner_index]

        # Re-apply the winning diff so the tree reflects it and the normal
        # review path receives a live snapshot (never in DEMO_MODE).
        #
        # This path previously computed ``touched`` for the snapshot but never
        # ran the scope guard or the test-integrity guard.  The winning diff is
        # byte-identical to one already gated by ``_apply_and_test`` with the
        # same ``target_files``, so in practice it was clean -- but that was an
        # assumption, not an invariant, and a merged candidate can arrive here
        # having been built from a different diff.  Both guards now run.
        gate = dict(winner.gate or {})
        if winner.diff and self._real_execution():
            parsed = patcher.parse_diff(winner.diff)
            touched = [fp.new_path or fp.old_path for fp in parsed
                       if (fp.new_path or fp.old_path)]
            out_of_scope = self._out_of_scope(touched, task)
            tampered = self._integrity_violations(touched, task)
            if out_of_scope or tampered:
                log_event("WARNING", "divergence_winner_reapply_refused",
                          {"task_id": task.id,
                           "label": winner.label,
                           "out_of_scope": sorted(out_of_scope)[:10],
                           "test_integrity": sorted(tampered)[:10]})
                gate["ok"] = False
                gate["snapshot"] = None
                gate["applied"] = []
                self.builder.strategy_model_role = None
                self.builder.reset()
                refused = BuilderOutput(
                    task_id=task.id, diff="", self_check_passed=False,
                    notes=("winning candidate touched files outside task scope "
                           "or pre-existing tests; refused"))
                return refused, gate, winner.usage
            snapshot = patcher.snapshot_files(self.project_root, touched)
            applied = patcher.apply_diff(self.project_root, winner.diff,
                                         guard=self._test_guard(task))
            gate["snapshot"] = snapshot if applied.ok else None
            gate["applied"] = applied.applied
            gate["ok"] = bool(gate.get("ok")) and bool(applied.ok)
        self.builder.strategy_model_role = None
        self.builder.reset()
        winner_build = BuilderOutput(
            task_id=task.id, diff=winner.diff, self_check_passed=True,
            notes=f"divergence winner {winner.label} ({winner.strategy.name})")
        return winner_build, gate, winner.usage

    @staticmethod
    def _simulated_gate(task: TaskSpec) -> Dict[str, Any]:
        """Deterministic demo/simulated gate evidence (no disk access)."""
        if task.metric is not None:
            return {"ok": True, "snapshot": None, "applied": [],
                    "objective": {"ok": True, "primary": 0.5,
                                  "runtime_s": 0.0}, "lint": []}
        return {"ok": True, "snapshot": None, "applied": [],
                "tests": {"passed": 3 + len(task.target_files),
                          "failed": 0, "skipped": 0}, "lint": []}

    async def _merge_candidates(
            self, task: TaskSpec,
            records: List[divergence.CandidateRecord],
            winner_index: int) -> Optional[divergence.CandidateRecord]:
        """LLM merge of gate-passing near-tie candidates; must re-pass."""
        payload = divergence.merge_payload(records)
        if len(payload) < 2:
            return None
        try:
            text = await self.merger.run(task, payload)
        except LLMUnavailableError:
            return None
        self.builder.strategy_model_role = None
        merged_diff, issues = self.builder._content_to_diff(text or "", task)
        if issues or not merged_diff:
            log_event("WARNING", "merge_invalid",
                      {"task_id": task.id, "issues": issues[:5]})
            return None
        blocked = False
        if self._real_execution():
            if task.metric is not None and task.run_command:
                gate = await self._apply_and_objective_gate(task, merged_diff)
                blocked, _ = objective_gate_block(
                    task.metric, gate.get("objective") or {},
                    task.metric.baseline)
            else:
                gate = await self._apply_and_test(task, merged_diff)
                tests = gate.get("tests") or {}
                blocked = bool(not gate.get("ok")
                               or tests.get("failed", 1) > 0
                               or (task.requires_new_tests
                                   and tests.get("passed", 0) <= 0))
        else:
            gate = self._simulated_gate(task)
        record = divergence.record_from_gate(
            "M",
            StrategySpec(name="merged",
                         approach="LLM merge of gate-passing candidates"),
            merged_diff, gate, "merger", self.llm.usage_snapshot(),
            blocked, task)
        if gate.get("snapshot") is not None:
            patcher.restore_snapshot(self.project_root, gate["snapshot"])
        divergence.store_candidate(self.project_root, task.id, record)
        return record

    # ------------------------------------------------------------------
    # v2.6 domain packs: task proposals at starvation + completion hooks
    # ------------------------------------------------------------------
    async def _notify_domain_pack(self, task: TaskSpec,
                                  objective_result: Optional[ObjectiveGateResult]
                                  ) -> None:
        """Let the active pack react to a completed task (best-effort)."""
        pack = domain_registry.active_pack(self.project_root)
        if pack is None:
            return
        try:
            await pack.on_task_completed(self, task, objective_result, None)
        except Exception as exc:  # noqa: BLE001 -- a pack must not break the core
            log_event("WARNING", "domain_pack_hook_failed",
                      {"project": self.project_name, "task_id": task.id,
                       "error": str(exc)})

    async def _try_domain_proposals(self) -> bool:
        """Ask the active pack for new tasks when the board starves.

        Returns True when at least one new task was appended (the loop then
        continues into a fresh scheduling tick).
        """
        pack = domain_registry.active_pack(self.project_root)
        if pack is None:
            return False
        board = self._load_board()
        if board is None:
            # Appending proposals onto an unread board would persist the
            # proposals *instead of* the real backlog.  Skip this round.
            log_event("WARNING", "domain_proposal_skipped_unreadable_board",
                      {"project": self.project_name})
            return False
        try:
            proposals = await pack.propose_tasks(self.project_root, board)
        except Exception as exc:  # noqa: BLE001
            log_event("WARNING", "domain_proposal_failed",
                      {"project": self.project_name, "error": str(exc)})
            return False
        if not proposals:
            return False
        existing = {str(t.get("id")) for t in board.get("tasks", [])}
        added = [t for t in proposals if str(t.get("id")) not in existing]
        if not added:
            return False
        # Keep pack-internal bookkeeping markers for the accept hook, but
        # never persist them into the board.
        accepted_meta = [dict(t) for t in added]
        for task in added:
            task.pop("_hypothesis_index", None)
        board.setdefault("tasks", []).extend(added)
        self._save_board(board)
        log_event("INFO", "domain_tasks_proposed",
                  {"project": self.project_name, "count": len(added),
                   "ids": [t.get("id") for t in added]})
        notifier = getattr(pack, "tasks_accepted", None)
        if notifier is not None:
            try:
                await notifier(self.project_root, accepted_meta)
            except Exception as exc:  # noqa: BLE001
                log_event("WARNING", "domain_accept_hook_failed",
                          {"project": self.project_name, "error": str(exc)})
        return True

    def _author_state_path(self) -> Path:
        return self.agent_dir / "spec_author_state.json"

    def _load_author_state(self) -> Dict[str, Any]:
        """Read the persisted authoring budget, or ``{}`` when absent/unreadable."""
        try:
            payload = json.loads(
                self._author_state_path().read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_author_state(self, payload: Dict[str, Any]) -> None:
        redaction.write_json_atomic(self._author_state_path(), payload)

    async def _try_author_from_goal(self) -> bool:
        """Fill an empty board from the project's stated goal (v2.9).

        This is the autonomy path: ``--new-project NAME --goal "..."`` produces
        a project the system can spec, build and test without further human
        input.  It is attempted only when the board is empty, because its whole
        purpose is replacing the missing specification rather than competing
        with an existing one.

        Ordering is deliberate and is the safety property of the whole design:

        1. author the plan (model proposes);
        2. write the tests to disk -- **before** the tasks exist, so
           ``TestIntegrityGuard`` baselines them when each task starts and the
           Builder is thereafter refused any edit to the contract it is graded
           against (verified by probe);
        3. run the Spec Critic's deterministic tier over each task, dropping
           any that disagree with their own tests;
        4. only then append the survivors to the board, through the normal
           validated path.

        v2.9.1 -- the attempt budget.  This used to retry unconditionally: a
        goal the author could not satisfy produced the same "authored zero
        tasks" note and the same cached reply every couple of seconds until
        ``--once``'s budget expired -- ten minutes of wall-clock to learn one
        thing once.  Now the budget is bounded (``SPEC_AUTHOR_MAX_ATTEMPTS``),
        each retry carries the previous attempt's rejection notes so it cannot
        replay the same answer, attempts back off, and exhausting the budget
        writes a terminal ``gave_up`` state the operator can read and delete.

        Returns True when at least one task was added.
        """
        if not config.SPEC_AUTHOR_ENABLED:
            return False
        board = self._load_board()
        if board is None:
            # Never author onto a board we could not read: writing would
            # replace the real backlog with a generated one.
            return False
        if board.get("tasks"):
            return False  # there is already work; authoring is not the job here
        goal = self._project_goal()
        if not goal:
            return False

        state = self._load_author_state()
        attempt = int(state.get("attempts", 0) or 0) + 1
        if attempt > config.SPEC_AUTHOR_MAX_ATTEMPTS:
            return False  # already gave up; operator action required

        if attempt > 1:
            backoff = float(config.SPEC_AUTHOR_RETRY_BACKOFF_S) * (attempt - 1)
            log_event("INFO", "spec_author_retry_wait",
                      {"project": self.project_name, "attempt": attempt,
                       "backoff_s": backoff,
                       "previous_notes": (state.get("last_notes") or [])[:3]})
            await asyncio.sleep(min(backoff, 60.0))

        existing = {str(t.get("id")) for t in board.get("tasks", [])}
        context = self._read_context()
        try:
            plan = await self.spec_author.run(
                goal, existing_ids=sorted(existing), context=context,
                max_tasks=config.SPEC_AUTHOR_MAX_TASKS, attempt=attempt,
                repair=[str(n) for n in (state.get("last_notes") or [])])
        except Exception as exc:  # noqa: BLE001 -- never break the loop
            log_event("WARNING", "spec_author_orchestrator_failed",
                      {"project": self.project_name, "error": str(exc)[:200],
                       "attempt": attempt})
            self._save_author_state({"attempts": attempt,
                                     "last_notes": [f"author raised: {exc}"[:200]],
                                     "goal": goal[:400]})
            return False
        if not plan.usable:
            gave_up = attempt >= config.SPEC_AUTHOR_MAX_ATTEMPTS
            self._save_author_state({
                "attempts": attempt,
                "last_notes": plan.notes[:6],
                "gave_up": gave_up,
                "goal": goal[:400],
                "at": datetime.now(timezone.utc).isoformat(),
                "remedy": "edit goal.txt or .agent/context.md, or delete this "
                          "file to let the author try again",
            })
            log_event("INFO" if not gave_up else "ERROR",
                      "spec_author_no_tasks",
                      {"project": self.project_name,
                       "attempt": attempt,
                       "gave_up": gave_up,
                       "notes": plan.notes[:4]})
            return False

        # 2. Tests first, so they are frozen before any Builder sees them.
        written = spec_author.write_tests(
            plan, self.project_root,
            dry_run=not config.SPEC_AUTHOR_WRITE_TESTS)
        usable = [t for t in plan.usable if t.usable]
        if not usable:
            return False

        # 3. The Critic gates activation.  A machine-authored test that
        #    contradicts its own objective would otherwise become the contract
        #    with nothing between it and the Builder.
        #
        #    v2.9.1: the *full* Critic runs here, not just the free
        #    deterministic tier.  Authored tasks used to get the mechanical
        #    checks only, on the reasoning that the author and the Critic share
        #    a blind spot -- but the author is the *weaker* party here, and the
        #    semantic tier is the only check that can see a test asserting
        #    ``entry.account`` when the objective it was written from declares
        #    ``JournalEntry(date, memo, postings)``.  Measured on a real
        #    7-task authored board: 5 tasks delivered, and both failures were
        #    exactly that contradiction -- a test importing modules no task
        #    creates, and a test asserting a shape its own objective does not
        #    have.  Neither is visible to ``ast``.
        #
        #    Cost is one call per authored task, spent before any Builder
        #    attempt rather than after three of them.
        accepted: List[Any] = []
        for authored in usable:
            report = await self._critique_authored(authored.task, plan)
            if report is not None and report.blocking:
                log_event("ERROR", "spec_author_task_blocked",
                          {"project": self.project_name,
                           "task_id": authored.task.id,
                           "findings": [f.to_dict() for f in report.blocking]})
                continue
            accepted.append(authored)

        if not accepted:
            # v2.9.1: persist why, so the next attempt gets it as repair context.
            # This used to return False silently, which discarded exactly the
            # information the retry needed -- the difference between a loop that
            # converges and one that re-proposes the same unbuildable board.
            # The LAST notes are kept (not the first): the critic's blocking
            # reasons are appended after the author's own notes, so a head
            # slice of a note-heavy plan drops exactly the context the retry
            # needs.  And like the sibling ``not plan.usable`` path, exhausting
            # the budget writes a terminal ``gave_up`` state: the next call
            # would return at the attempts check regardless, so leaving
            # ``gave_up: False`` would send the operator to wait forever.
            gave_up = attempt >= config.SPEC_AUTHOR_MAX_ATTEMPTS
            if gave_up:
                remedy = ("the author budget is exhausted: delete this file "
                          "or fix the blocked objectives, then let the author "
                          "try again")
            else:
                remedy = "every task was held by the Spec Critic; the notes " \
                         "above say which rule, and the next attempt should " \
                         "author around it"
            self._save_author_state({
                "attempts": attempt,
                "last_notes": plan.notes[-6:],
                "gave_up": gave_up,
                "goal": goal[:400],
                "at": datetime.now(timezone.utc).isoformat(),
                "remedy": remedy,
            })
            log_event("WARNING" if not gave_up else "ERROR",
                      "spec_author_all_tasks_blocked",
                      {"project": self.project_name, "attempt": attempt,
                       "tasks": len(usable), "accepted": 0,
                       "gave_up": gave_up,
                       "notes": plan.notes[-4:]})
            return False
        keep = {t.task.id for t in accepted}
        plan.tasks = [t for t in plan.tasks if t.usable or t.task is None]
        for task in plan.tasks:
            if task.task is not None and task.task.id not in keep:
                task.rejected = "spec critic blocked activation"
        # v2.9.1: delete the tests of the tasks we just dropped.
        #
        # Stage 2 writes every planned task's test file *before* the Critic runs,
        # because the Critic needs them on disk to check them. So a task the
        # Critic blocks leaves its test behind with no task to satisfy it.
        #
        # Measured on the slatepress board: 6 test files for 4 tasks, two of
        # them (`test_markdown.py`, `test_slatepress.py`) testing modules no task
        # owned. Those orphans are not inert -- they break collection for the
        # whole suite and inflate the gate's baseline, so the suite reads as
        # broken for reasons no board entry explains. A contract with no
        # implementation task is not a contract.
        removed_tests = self._remove_orphan_authored_tests(plan, keep)
        if removed_tests:
            written = [p for p in written if p not in removed_tests]

        entries = spec_author.board_entries(plan)
        if not entries:
            return False
        board.setdefault("tasks", []).extend(entries)
        self._save_board(board)
        self._save_author_state({
            "attempts": attempt,
            "last_notes": plan.notes[:6],
            "gave_up": False,
            "goal": goal[:400],
            "at": datetime.now(timezone.utc).isoformat(),
            "tasks_written": len(entries),
        })
        log_event("INFO", "spec_author_board_filled",
                  {"project": self.project_name, "goal": goal[:120],
                   "tasks": len(entries), "tests_written": len(written),
                   "attempt": attempt,
                   "chains": plan.summary()["chains"],
                   "notes": plan.notes[:4]})

        # A root conftest keeps the authored tests runnable from any working
        # directory.  Without one, ``pytest`` from anywhere but the project root
        # makes every module skip and report "N skipped" while verifying
        # nothing -- which is exactly what logtriage did until it was fixed.
        if config.SPEC_AUTHOR_WRITE_TESTS and \
                spec_author.needs_conftest(plan, self.project_root):
            self._write_root_conftest()
        return True

    def _remove_orphan_authored_tests(self, plan: Any,
                                     keep: Set[str]) -> List[str]:
        """Delete test files written for tasks that were not activated.

        Only ever removes a file this authoring pass itself created: the path
        must be inside the project's ``tests/`` directory and must have been
        recorded in ``plan`` as a written test. A pre-existing test the operator
        wrote is never a candidate, and the guard is checked before the unlink
        rather than trusted.
        """
        removed: List[str] = []
        tests_dir = (self.project_root / "tests").resolve()
        for authored in plan.tasks:
            if authored.usable or not authored.test_path:
                continue
            task_id = authored.task.id if authored.task is not None else ""
            if task_id in keep:
                continue
            candidate = (self.project_root / authored.test_path).resolve()
            try:
                candidate.relative_to(tests_dir)
            except ValueError:
                continue  # outside tests/ -- never ours to remove
            if not candidate.is_file():
                continue
            try:
                candidate.unlink()
            except OSError as exc:
                log_event("WARNING", "spec_author_orphan_test_kept",
                          {"project": self.project_name,
                           "test": authored.test_path,
                           "error": str(exc)[:150]})
                continue
            removed.append(authored.test_path)
            log_event("INFO", "spec_author_orphan_test_removed",
                      {"project": self.project_name, "task_id": task_id,
                       "test": authored.test_path,
                       "note": "the Critic blocked this task, so its test has "
                               "no implementation to grade"})
        return removed

    def _refresh_lines(self) -> None:
        """Recompute and publish the line partition (v2.9).

        Cheap enough to run every tick: it is arithmetic over the board, not a
        model call, and the result is memoised on the board's shape so an idle
        project does no work at all.

        Recomputed rather than computed once because the board changes as tasks
        complete and as the Spec Author fills an empty one.  A partition computed
        at startup would go stale exactly when it matters -- when new tasks
        arrive that could have run in parallel with work already finished.
        """
        board = self._load_board_or_empty()
        tasks = board.get("tasks") or []
        if not tasks:
            self.status["lines"] = None
            return
        shape = (len(tasks), tuple(str(t.get("id")) for t in tasks))
        if getattr(self, "_lines_shape", None) == shape:
            return
        partition = lines_mod.partition(tasks, max_lines=config.LINE_COUNT)
        problems = lines_mod.validate(partition, tasks)
        if problems:
            # Refuse to publish a partition we cannot prove is safe.  A wrong
            # partition is worse than none: it looks authoritative.
            log_event("ERROR", "line_partition_invalid",
                      {"project": self.project_name,
                       "problems": problems[:4]})
            self.status["lines"] = None
            return
        self._lines_shape = shape
        self.status["lines"] = {
            "count": partition.count,
            "sizes": [len(line) for line in partition.lines],
            "serialised": len(partition.serialised),
            "chunks": [line.task_ids for line in partition.lines],
            "notes": partition.notes[:3],
        }
        log_event("INFO", "line_partition_ready",
                  {"project": self.project_name, **self.status["lines"]})

    def _project_goal(self) -> str:
        """The project's one-paragraph goal, if it recorded one."""
        for name in ("goal.txt", "goal.md"):
            path = self.project_root / name
            try:
                text = path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                return text[:4000]
        return ""

    def _read_context(self) -> str:
        try:
            return (self.agent_dir / "context.md").read_text(encoding="utf-8")
        except OSError:
            return ""

    def _write_root_conftest(self) -> None:
        """Make the project root importable regardless of pytest's cwd."""
        path = self.project_root / "conftest.py"
        if path.exists():
            return
        try:
            path.write_text(
                '"""Put the project root on sys.path.\n\n'
                "The tests guard themselves tests-first with a find_spec probe "
                "and a\nmodule-level skip, which only fires correctly when the "
                "project root is\nimportable. ``python -m pytest`` happens to "
                "work because Python puts the CWD on\nsys.path; a bare "
                "``pytest``, or any invocation from elsewhere, does not -- and "
                "then every\nmodule skips and the suite reports \"N skipped\" "
                "with exit code 0, which\nreads exactly like a healthy run "
                "while verifying nothing.\n\"\"\"\n\n"
                "from __future__ import annotations\n\n"
                "import sys\n"
                "from pathlib import Path\n\n"
                "ROOT = Path(__file__).resolve().parent\n"
                "if str(ROOT) not in sys.path:\n"
                "    sys.path.insert(0, str(ROOT))\n",
                encoding="utf-8")
            log_event("INFO", "spec_author_conftest_written",
                      {"project": self.project_name})
        except OSError as exc:
            log_event("WARNING", "spec_author_conftest_failed",
                      {"project": self.project_name, "error": str(exc)[:150]})

    def _escalation_feedback(self, task: TaskSpec, reason: str,
                             diag: Optional[diagnosis.Diagnosis]) -> str:
        """What a human reads on the board when a task escalates.

        Three questions, answered in the order they get asked:

        1. *which bucket* -- the diagnosis, when one reached a conclusion.  When
           it did not, say so plainly instead of printing a cause word with
           zero confidence behind it; observed two escalations reporting
           ``missing_module`` at ``confidence 0.0`` with no evidence, which
           reads as a finding and is not one.
        2. *what happened* -- the escalation reason.
        3. *what the gate said* -- the reviewer's findings from the review
           history, which is the only place the actual pytest output survives.

        Assembled from records the loop already wrote, so this costs no call.
        """
        lines: List[str] = []
        if diag is not None and diag.confidence > 0.0 and (
                diag.evidence or diag.applied):
            lines.append(diag.feedback())
        elif diag is not None:
            lines.append(
                f"DIAGNOSIS: no conclusion (cause suggested "
                f"{diag.cause!r} at confidence {diag.confidence:.2f}, no "
                f"supporting evidence) -- read the gate findings below")
        else:
            lines.append("DIAGNOSIS: unavailable (diagnosis disabled, no model "
                         "capacity, or the diagnosis call failed) -- read the "
                         "gate findings below")
        if reason:
            lines.append(f"REASON: {reason}")

        attempts = self._review_history_for(task.id)
        findings: List[str] = []
        for entry in attempts[-config.MAX_ATTEMPTS:]:
            # Only the Reviewer's own findings are evidence.  DIAGNOSED and
            # ESCALATED rows carry the diagnosis's cause word, and echoing it
            # back as a gate finding made the board read
            #   - task tests did not run: ... missing modules: src.ledger.models
            #   - task tests did not run: ... missing modules: src.ledger.models
            #   - missing_module
            # which reads as three independent observations rather than one
            # failure plus the label we put on it.
            if str(entry.get("verdict", "")).upper() in ("DIAGNOSED",
                                                         "ESCALATED"):
                continue
            feedback = entry.get("feedback")
            if isinstance(feedback, str):
                findings.append(feedback.strip())
            elif isinstance(feedback, list):
                for item in feedback:
                    if isinstance(item, dict):
                        issue = str(item.get("issue", "")).strip()
                        if issue:
                            findings.append(issue)
                    elif str(item).strip():
                        findings.append(str(item).strip())
            elif not feedback:
                reason_text = str(entry.get("reason", "")).strip()
                if reason_text:
                    findings.append(reason_text)
        for item in findings[-6:]:
            lines.append(f"  - {item[:300]}")
        return "\n".join(lines).strip() or "escalated with no recorded evidence"

    def _doom_loop_signals(self, task_id: str) -> List[Any]:
        """Steering signals that mean another identical attempt cannot help.

        Read through :meth:`SteeringMonitor.evaluate` rather than duplicating
        its thresholds, so the thing that stops the loop and the thing that
        reports it can never disagree.

        Attribution is the subtle part, and getting it wrong is why the first
        version of this did nothing.  ``steering`` emits **two** doom-loop
        shapes:

        * a **diff** repeat, which carries the task id it came from; and
        * a **signature** repeat, which deliberately carries ``task_id=None``
          because it is one project-level signal per recurring *token* -- the
          same token can be shared by several tasks, and emitting one row per
          task was the noise this design removed.

        A naive ``s.task_id == task_id`` filter therefore matches the first
        shape and silently discards the second, which is the shape that actually
        fired on a real board (token ``LEDGER``, 5 consecutive failures).  So a
        token-level signal is matched against *this task's own*
        ``error_signature``: attribution the board already carries, and precise
        enough that one stuck task cannot cancel work on any other.
        """
        try:
            board = self._load_board_or_empty()
            signals = self.steering.evaluate(self.project_name,
                                             board.get("tasks"))
        except Exception:  # noqa: BLE001 -- advisory only, never blocks a build
            return []

        mine_tokens: Set[str] = set()
        for raw in board.get("tasks") or []:
            if isinstance(raw, dict) and str(raw.get("id")) == task_id:
                mine_tokens = set(steering.signature_tokens(
                    str(raw.get("error_signature") or "")))
                break

        out: List[Any] = []
        for signal in signals:
            if signal.kind != steering.DOOM_LOOP:
                continue
            if signal.task_id == task_id:
                out.append(signal)
            elif signal.task_id is None:
                token = str((signal.evidence or {}).get("token", ""))
                if token and token in mine_tokens:
                    out.append(signal)
        if not out and any(s.kind == steering.DOOM_LOOP for s in signals):
            # A loop is running and nothing on the board claims it, which is
            # precisely the state the docs warn makes a doom loop invisible.
            log_event("WARNING", "doom_loop_unattributed",
                      {"project": self.project_name, "task_id": task_id,
                       "my_signature_tokens": sorted(mine_tokens),
                       "signals": [s.detail for s in signals
                                   if s.kind == steering.DOOM_LOOP][:2],
                       "note": "a doom loop is running but this task's "
                               "error_signature claims none of its tokens; "
                               "set error_signature or steering cannot "
                               "attribute the loop to a task"})
        return out

    async def _escalate(self, task: TaskSpec, verdict: Optional[ReviewVerdict],
                        model_used: str, duration_s: float,
                        reason: str = "max retries exhausted") -> None:
            """Escalation after the retry budget is spent.

            v2.8: the pattern-confidence decay here was dead code --
            ``workspace_patterns_referenced`` was empty at every construction site
            in the system, so zero decrements were ever recorded.  Anti-pattern
            promotion is retained, but as an *audit* record only: its description
            text is no longer injected into any Builder prompt.  Systemic looping
            is now detected from counters by :mod:`steering`.

            v2.9: async, because escalation now *diagnoses* rather than merely
            recording.  An escalated task used to arrive at a human as a stack
            trace, and ``_try_auto_requeue`` then handed the identical objective
            back to the Builder -- observed repeating itself seven times.  The
            reason is structural: the escalator recorded *that* a task failed and
            nothing about *why*, so the retry had no new information.  One call
            here, on data the loop already collected, supplies the missing half.
            """
            diag: Optional[diagnosis.Diagnosis] = None
            if config.DIAGNOSIS_ENABLED:
                diag = await self._diagnose_escalation(
                    task, reason=reason, verdict=verdict)
            log_event("ERROR", "task_escalated",
                      {"project": self.project_name, "task_id": task.id,
                       "reason": reason,
                       "diagnosis": diag.to_dict() if diag else None})
            self._write_cycle_summary(CycleSummary(
                task_id=task.id, attempts=config.MAX_ATTEMPTS,
                verdict="ESCALATED", model_used=model_used,
                duration_s=duration_s, escalated=True, reason=reason))
            # v2.9.1: ``last_feedback`` is the only thing a human reads on the
            # board, and it used to carry the literal string "max retries
            # exhausted" -- the reason the escalation happened, and no evidence
            # of why the task could not be built.  ``Diagnosis.feedback()`` was
            # written for exactly this and never called, so the whole diagnosis
            # round trip ended in the log and died there.
            #
            # It is now the first line, followed by the escalation reason and
            # the reviewer's own findings, so the three questions answer in
            # order: what bucket, what happened, what the gate said.
            feedback = self._escalation_feedback(task, reason, diag)
            self._set_task_status(task.id, status="failed",
                                  last_feedback=feedback[:2000],
                                  failed_at=datetime.now(timezone.utc).isoformat())
            self._append_review_history({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_id": task.id, "attempt": config.MAX_ATTEMPTS,
                "verdict": "ESCALATED", "reason": reason,
            })
            signature = task.error_signature or f"ERR_{task.id.replace('-', '_')}"
            self.steering.record_attempt(self.project_name,
                steering.AttemptRecord(
                    task_id=task.id, outcome="ESCALATED",
                    attempt=config.MAX_ATTEMPTS,
                    signature=signature, duration_s=duration_s,
                    error_summary=reason[:200]))
            self.engine.add_anti_pattern(
                signature=signature,
                description=(f"max build-review cycles failed for {task.id} "
                             f"({task.title}): {reason}"),
                remediation="require planner redesign + manual review before retry",
                severity="high")
            if diag is not None:
                # The cause string is the point of the whole exercise: the
                # steering monitor clusters escalations by it, so three
                # spec_ambiguous failures surface as one systemic block instead
                # of three independent-looking incidents.
                self.steering.record_attempt(
                    self.project_name,
                    steering.AttemptRecord(
                        task_id=task.id, outcome=diag.cause,
                        attempt=config.MAX_ATTEMPTS,
                        signature=signature, duration_s=duration_s,
                        error_summary="; ".join(diag.evidence[:2])[:200]))

    def _task_test_sources(self, task: TaskSpec) -> List[Tuple[str, str]]:
        """The frozen test contract for *task*, as ``(path, source)`` pairs.

        Read from disk, not from the board, because under tests-first the test
        file exists before the task does and stays there.  A diagnoser that
        cannot see the contract cannot tell "the Builder is wrong" from "the
        contract is impossible", and those two need opposite remedies.
        """
        out: List[Tuple[str, str]] = []
        for rel in spec_critic._test_files_for(task, self.project_root):
            try:
                out.append((rel.relative_to(self.project_root).as_posix(),
                            rel.read_text(encoding="utf-8", errors="replace")))
            except (OSError, ValueError):
                continue
        return out

    async def _diagnose_escalation(self, task: TaskSpec, reason: str,
                                   verdict: Optional[ReviewVerdict]
                                   ) -> Optional[diagnosis.Diagnosis]:
        """One model call: why did this task fail, and what should happen?

        Never raises and never blocks escalation.  A diagnosis is an
        improvement to a failure report, so the failure is reported either way.
        """
        try:
            attempts = self._review_history_for(task.id)
            evidence = diagnosis.summarize_evidence(
                task, attempts, reason=reason,
                test_sources=self._task_test_sources(task))
            model_role = config.resolve_role("diagnosis")
            user = (
                "This task exhausted its retry budget. Diagnose the cause and "
                "recommend a remedy. Respond with one JSON object.\n"
                "Allowed causes: " + ", ".join(diagnosis.CAUSES) + "\n"
                "Allowed remedies: " + ", ".join(diagnosis.REMEDIES) + "\n"
                "Provide proposed_objective only for narrow_objective.\n"
                "Report the cause you can actually support from the evidence. "
                "If the evidence does not support any cause, say "
                "cause='unknown' and remedy='needs_human' rather than guessing "
                "-- and always state your confidence.\n"
                f"<<<EVIDENCE>>>\n{evidence}"
            )
            response = await self.llm.complete(
                system=diagnosis.SYSTEM, user=user, tools=[],
                model_role=model_role)
            data = _extract_json(response)
            self.llm.record_parse(data is not None)
            diag = diagnosis.coerce(data, task.id)
            if diag is None:
                log_event("WARNING", "diagnosis_unparseable",
                          {"task_id": task.id})
                return None
            diag = diagnosis.apply(diag, task, self._set_task_status)
            self._append_review_history({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_id": task.id, "attempt": config.MAX_ATTEMPTS,
                "verdict": "DIAGNOSED",
                "reason": diag.cause, "remedy": diag.remedy,
                "applied": diag.applied,
            })
            conclusive = bool(diag.evidence or diag.applied)
            log_event("INFO" if conclusive else "WARNING",
                      "escalation_diagnosed" if conclusive
                      else "escalation_diagnosis_inconclusive",
                      {"project": self.project_name, "task_id": task.id,
                       "cause": diag.cause, "remedy": diag.remedy,
                       "confidence": diag.confidence, "applied": diag.applied,
                       "evidence_items": len(diag.evidence),
                       "note": None if conclusive else
                       "no evidence and no applied remedy; treat the cause as "
                       "a guess and read the gate findings on the board"})
            return diag
        except LLMUnavailableError as exc:
            # Capacity exhaustion is not a diagnosis failure; the escalation
            # stands and the task stays failed.
            log_event("INFO", "diagnosis_skipped_no_capacity",
                      {"task_id": task.id, "error": str(exc)[:150]})
            return None
        except Exception as exc:  # noqa: BLE001
            log_event("WARNING", "diagnosis_failed",
                      {"task_id": task.id, "error": str(exc)[:200]})
            return None

    # ------------------------------------------------------------------
    # v2.5.5: board observability + bounded auto-requeue
    # ------------------------------------------------------------------
    def _update_board_stats(self) -> None:
        """Refresh ``status["board"]`` for the manifest writer (cheap: the
        board is a small local file re-read on every loop tick anyway)."""
        board = self._load_board_or_empty()
        self.status["board"] = compute_board_stats(board.get("tasks", []))

    def _try_auto_requeue(self) -> Optional[str]:
        """Requeue one cooldown-expired failed task (bounded; v2.5.5).

        Fires only when the loop has nothing eligible (starvation time) --
        it never preempts healthy work.  Budget: at most
        ``config.FAILED_AUTO_REQUEUES`` requeues per task; a missing or
        unparseable ``failed_at`` is treated as eligible (the cap still
        bounds the behavior).  Returns the requeued task id or None.
        """
        if config.FAILED_AUTO_REQUEUES <= 0:
            return None
        board = self._load_board()
        if board is None:
            return None
        now = datetime.now(timezone.utc)
        for task in board.get("tasks", []):
            if task.get("status") != "failed":
                continue
            requeues = int(task.get("auto_requeues", 0) or 0)
            if requeues >= config.FAILED_AUTO_REQUEUES:
                continue
            try:
                elapsed = (now - datetime.fromisoformat(
                    str(task.get("failed_at")))).total_seconds()
            except (TypeError, ValueError):
                elapsed = float(config.FAILED_RETRY_COOLDOWN_S) + 1.0
            if elapsed < config.FAILED_RETRY_COOLDOWN_S:
                continue
            task_id = str(task.get("id", "?"))
            task["status"] = "pending"
            task["retry_count"] = 0
            task["last_feedback"] = None
            task["failed_at"] = None
            task["defer_until"] = None
            task["auto_requeues"] = requeues + 1
            self._save_board(board)
            log_event("INFO", "task_auto_requeued",
                      {"project": self.project_name, "task_id": task_id,
                       "auto_requeues": task["auto_requeues"],
                       "cooldown_s": config.FAILED_RETRY_COOLDOWN_S})
            return task_id
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def outstanding_work(self) -> bool:
        """True when this loop still has something it could act on.

        ``--once`` used to answer "is there work left?" by grepping the board
        for ``status == "pending"``.  That is wrong twice over:

        * an **empty** board on a goal-only project is not *finished* -- the
          Spec Author simply has not run yet.  The poller read the two as
          identical, so the pass returned in seconds and the documented
          ``--new-project NAME --goal ...`` -> ``--once`` flow authored
          nothing at all (observed: empty board, 3.5 s exit, zero tasks).
        * a project paused on purpose looks exactly like one that is done.

        So the loop reports what *it* can do, and the supervisor waits on that
        report instead of inferring it from a file.  Deliberately bounded: a
        failed task counts only while it still has auto-requeue budget left
        (mirroring :meth:`_try_auto_requeue`), and a project whose Spec Author
        has exhausted its budget counts as nothing (there is genuinely nothing
        left to try -- the operator has to act).  So an escalated board awaiting
        a human, and a goal the author cannot satisfy, do not hold a ``--once``
        pass open until the budget expires.
        """
        board = self._load_board()
        if board is None:
            return False  # unreadable: nothing this loop may safely act on
        tasks = board.get("tasks") or []
        if any(t.get("status") == "pending" for t in tasks):
            return True
        if config.FAILED_AUTO_REQUEUES > 0 and any(
                t.get("status") == "failed"
                and int(t.get("auto_requeues", 0) or 0)
                < config.FAILED_AUTO_REQUEUES
                for t in tasks):
            return True
        # No tasks at all: the backlog is *unwritten*, not finished -- unless
        # the author has already tried and given up.
        if tasks or not config.SPEC_AUTHOR_ENABLED or not self._project_goal():
            return False
        state = self._load_author_state()
        if state.get("gave_up"):
            return False
        return int(state.get("attempts", 0) or 0) < config.SPEC_AUTHOR_MAX_ATTEMPTS

    def _paused(self) -> bool:
        return (self.workspace_root / "pause.flag").exists()

    async def run(self) -> None:
        log_event("INFO", "project_orchestrator_started",
                  {"project": self.project_name})
        # v2.9.1: announce that this loop has not yet decided anything.  The
        # supervisor needs a real "settled" signal to tell a goal-only project
        # that still owes a specification from one that is genuinely finished.
        self.status["loop_state"] = "starting"
        self.status["work_available"] = False
        while True:
            try:
                if self._paused():
                    self.status["status"] = "paused"
                    self.status["loop_state"] = "settled"
                    self.status["work_available"] = False
                    await asyncio.sleep(self.poll_interval_s)
                    continue
                if self.status.get("status") == "paused":
                    log_event("INFO", "project_orchestrator_resumed",
                              {"project": self.project_name})
                prev_status = self.status.get("status")
                self.status["status"] = "running"

                self._update_board_stats()
                # v2.9: publish the line partition so the manifest, the log and
                # the operator can see how the board would divide.  Computing it
                # is free and deterministic (graph colouring over shared files,
                # see ``lines.py``), so it is refreshed whenever the board
                # changes shape rather than only at startup.
                self._refresh_lines()
                next_task = self.get_next_pending_task()
                if next_task is None:
                    # Starvation path: give failed tasks one bounded second
                    # chance, then ask the active domain pack for new work,
                    # before settling into idle.
                    if self._try_auto_requeue() is not None:
                        self.status["loop_state"] = "working"
                        continue
                    if await self._try_domain_proposals():
                        self.status["loop_state"] = "working"
                        continue
                    # v2.9: before asking a domain pack for work, try the
                    # project's own goal.  This is the path that makes the
                    # system self-sufficient: a project created with a goal and
                    # no board fills its own backlog instead of waiting for a
                    # human to write one.  Placed last so it never pre-empts
                    # work that already exists.
                    if await self._try_author_from_goal():
                        self.status["loop_state"] = "working"
                        continue
                    # Nothing left to try.  Report what this loop can still do
                    # (for the supervisor's `--once` pass and the manifest) and
                    # settle.
                    self.status["work_available"] = self.outstanding_work()
                    self.status["loop_state"] = "settled"
                    if prev_status != "idle":
                        stats = self.status.get("board") or {}
                        blocked = int(stats.get("blocked", 0) or 0)
                        failed_ids = stats.get("failed_ids", []) or []
                        # v2.9.1: an idle log line that always said "blocked"
                        # read as an incident on every completed project
                        # (``blocked=0 failed_ids=[]`` on all four).  Only
                        # claim a block when something is actually blocked or
                        # escalated; otherwise say the project is simply idle
                        # and why.
                        if blocked or failed_ids:
                            log_event("WARNING", "project_idle_blocked",
                                      {"project": self.project_name,
                                       "blocked": blocked,
                                       "failed_ids": failed_ids})
                        else:
                            log_event("INFO", "project_idle",
                                      {"project": self.project_name,
                                       "done": stats.get("done", 0),
                                       "pending": stats.get("pending", 0),
                                       "can_author_backlog":
                                           self.status["work_available"],
                                       "note": "nothing scheduled and "
                                               "nothing blocked"})
                    # v2.8: an idle loop is the most likely place to be stuck
                    # in a doom loop, so this is where progress is measured.
                    # The signals are surfaced in the manifest and drive the breaker; the
                    # loop itself keeps its existing bounded self-healing
                    # (auto-requeue, then domain proposals).
                    board_now = self._load_board_or_empty()
                    self.steering.note_board(self.project_name,
                                             board_now.get("tasks", []))
                    signals = self.steering.evaluate(self.project_name,
                                                     board_now.get("tasks"))
                    if signals:
                        steering.emit(signals)
                        self.steering.report_emitted(
                            self.project_name, [s.kind for s in signals], signals)
                        self.status["steering"] = [s.to_dict() for s in signals]
                    self.status["status"] = "idle"
                    await asyncio.sleep(self.idle_interval_s)
                    continue
                self.status.pop("steering", None)
                self.status["work_available"] = True
                task, _board_entry = next_task
                await self.process_task(task)
            except asyncio.CancelledError:
                log_event("INFO", "project_orchestrator_stopped",
                          {"project": self.project_name})
                return
            except Exception as exc:  # noqa: BLE001 -- supervisor will restart
                log_event("ERROR", "project_orchestrator_loop_crashed",
                          {"project": self.project_name, "error": str(exc),
                           "traceback": traceback.format_exc()})
                self.status["status"] = "crashed"
                raise  # let the SuperOrchestrator's supervisor see the crash
