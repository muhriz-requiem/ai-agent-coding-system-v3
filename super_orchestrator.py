"""SuperOrchestrator: top tier of the 3-tier hierarchy.

Responsibilities (and non-responsibilities):
* discovers projects and spawns one ProjectOrchestrator asyncio task each,
* optionally starts ONE IntegrationAgent with an asyncio notification queue,
* runs the deterministic circuit breaker, which pauses a project that has
  spent quota without completing anything (v2.9; this replaced the Council),
* monitors everything: pause flag, crash restarts (5s backoff), manifest,
* consumes only HIGH-LEVEL status summaries -- it never loads code or runs
  tests.

v2.9 removed the Council.  It was the most expensive prompt in the system and,
over a full six-module build, it produced 47 arbiter failures, 9 decisions
held for a human, and not one intervention that actually repaired anything.
The one behaviour worth keeping -- stop a doomed run -- needs no model at all,
so it now lives in :meth:`_check_breaker` as arithmetic over the board and the
token window.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import steering
from agent_roles import IntegrationAgent, LLMClient
from logger import log_event
from project_orchestrator import ProjectOrchestrator
from shared_workspace_engine import SharedWorkspaceEngine

#: Delay before restarting a crashed ProjectOrchestrator task.
RESTART_BACKOFF_S = 5.0
#: Consecutive-restart threshold that triggers a CRITICAL + cooldown.
MAX_CONSECUTIVE_RESTARTS = 5


class SuperOrchestrator:
    """Tier-1 supervisor and manifest writer."""

    def __init__(self, workspace_root: Optional[Path] = None) -> None:
        self.workspace_root = Path(workspace_root) if workspace_root \
            else config.WORKSPACE_ROOT
        self.system_dir = self.workspace_root / ".system"
        self.manifest_path = self.system_dir / "manifest.json"
        self.engine = SharedWorkspaceEngine(self.workspace_root)
        self.llm = LLMClient(self.workspace_root)
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
        self.restarts: Dict[str, int] = {}
        self.consecutive_restarts: Dict[str, int] = {}
        self.integration_agent: Optional[IntegrationAgent] = None
        self.integration_task: Optional[asyncio.Task] = None
        self.integration_queue: Optional[asyncio.Queue] = None
        self._paused_logged = False
        self._stop = asyncio.Event()

    # ------------------------------------------------------------------
    # Discovery + spawning
    # ------------------------------------------------------------------
    def _discover_projects(self) -> list:
        projects_dir = self.workspace_root / "projects"
        if not projects_dir.is_dir():
            return []
        return sorted(d for d in projects_dir.iterdir() if d.is_dir())

    def _spawn_project(self, name: str) -> None:
        status: Dict[str, Any] = self.projects.setdefault(name, {
            "status": "starting", "current_task_id": None,
            "restarts": 0, "last_started": None,
        })
        orchestrator = ProjectOrchestrator(
            workspace_root=self.workspace_root,
            project_name=name,
            engine=self.engine,
            llm=self.llm,
            integration_queue=self.integration_queue,
            status=status,
        )
        # NOTE: pass the orchestrator OBJECT (not a pre-created coroutine) so
        # cancelling a task before its first step never leaks an un-awaited
        # coroutine.
        task = asyncio.create_task(self._supervise(name, orchestrator))
        self.tasks[name] = task
        self.restarts.setdefault(name, 0)
        status["last_started"] = datetime.now(timezone.utc).isoformat()
        status["status"] = "running"
        log_event("INFO", "project_orchestrator_spawned", {"project": name})

    async def _supervise(self, name: str, orchestrator: ProjectOrchestrator) -> None:
        """Run one project loop; restart it after 5s backoff if it crashes.
        Crash handling covers the whole lifecycle including re-spawns."""
        while True:
            try:
                await orchestrator.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- self-healing path
                log_event("ERROR", "project_orchestrator_crashed",
                          {"project": name, "error": str(exc),
                           "traceback": traceback.format_exc()})
                self.projects[name]["status"] = "crashed"
                self.consecutive_restarts[name] = \
                    self.consecutive_restarts.get(name, 0) + 1
                count = self.consecutive_restarts[name]
                if count > MAX_CONSECUTIVE_RESTARTS:
                    log_event("CRITICAL", "project_restart_loop_detected",
                              {"project": name,
                               "consecutive_restarts": count,
                               "action": "backing_off_and_abandoning"})
                    self.projects[name]["status"] = "abandoned"
                    return
                await asyncio.sleep(RESTART_BACKOFF_S)  # 5s self-healing backoff
                self.restarts[name] = self.restarts.get(name, 0) + 1
                self.projects[name]["restarts"] = self.restarts[name]
                log_event("WARNING", "project_orchestrator_restarting",
                          {"project": name,
                           "restart_number": self.restarts[name]})
                if self._paused():
                    continue  # wait out the pause instead of spawning now
                # Fresh object: a crashed run() coroutine cannot be reused.
                status: Dict[str, Any] = self.projects.setdefault(name, {})
                status["status"] = "running"
                orchestrator = ProjectOrchestrator(
                    workspace_root=self.workspace_root,
                    project_name=name,
                    engine=self.engine,
                    llm=self.llm,
                    integration_queue=self.integration_queue,
                    status=status,
                )
                continue
            # Healthy run completed (only via cancellation) -- reset counter.
            self.consecutive_restarts[name] = 0
            return

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    def _write_manifest(self) -> None:
        manifest: Dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "paused": self._paused(),
            "integration_agent": {
                "enabled": self.integration_agent is not None,
                "status": self._integration_status(),
            },
            "circuit_breaker": {
                "enabled": config.BREAKER_ENABLED,
                "tokens_per_window": config.BREAKER_TOKENS_PER_WINDOW,
            },
            "projects": {
                name: {
                    "status": state.get("status"),
                    "current_task_id": state.get("current_task_id"),
                    "restarts": state.get("restarts", 0),
                    # v2.9.1: has this loop made a scheduling decision yet, and
                    # can it still act?  Distinguishes "finished" from "has not
                    # been given a turn" (e.g. a goal-only project awaiting its
                    # specification) for both the operator and ``--once``.
                    "loop_state": state.get("loop_state", "starting"),
                    "work_available": bool(state.get("work_available", False)),
                    # v2.9: the deterministic breaker's verdict, if it fired.
                    "breaker": state.get("breaker"),
                    # v2.9: how the board divides into conflict-free lines.
                    "lines": state.get("lines"),
                    "steering": state.get("steering"),
                    # v2.5.5: per-project board summary (done/failed/pending/
                    # blocked/failed_ids) written by the ProjectOrchestrator;
                    # absent (None) before its first loop tick.
                    "board": state.get("board"),
                }
                for name, state in self.projects.items()
            },
        }
        # Unique tmp per write: two writers sharing one tmp path collide
        # ("file in use") before the replace even happens. Retries absorb
        # transient reader locks on the target file (dashboard, operators).
        payload = json.dumps(manifest, indent=2) + "\n"
        for attempt in range(4):
            tmp = self.manifest_path.with_name(
                f"manifest.json.{os.getpid()}.{attempt}.tmp")
            try:
                tmp.write_text(payload, encoding="utf-8")
                tmp.replace(self.manifest_path)
                return
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                if attempt == 3:
                    log_event("ERROR", "manifest_write_failed",
                              {"error": str(exc)})
                else:
                    time.sleep(0.25 * (attempt + 1))

    def _integration_status(self) -> str:
        if self.integration_task is None:
            return "disabled"
        if self.integration_task.done():
            return "stopped"
        if self._stop.is_set():
            return "stopping"
        return "running"

    def _paused(self) -> bool:
        return (self.workspace_root / "pause.flag").exists()

    # ------------------------------------------------------------------
    # Integration agent
    # ------------------------------------------------------------------
    def _start_integration_agent(self) -> None:
        if not config.ENABLE_INTEGRATION_AGENT or self.integration_agent:
            return
        self.integration_agent = IntegrationAgent(self.llm, self.workspace_root)
        self.integration_queue = self.integration_agent.queue
        # Pass the AGENT object; its run() is invoked inside the task body.
        self.integration_task = asyncio.create_task(
            self._run_integration_agent(self.integration_agent))
        log_event("INFO", "integration_agent_spawned", {})

    async def _run_integration_agent(self, agent: IntegrationAgent) -> None:
        await agent.run()

    # ------------------------------------------------------------------
    # Circuit breaker (v2.9) -- replaced the Council's one useful reflex
    # ------------------------------------------------------------------
    def _check_breaker(self) -> None:
        """Pause a project that is burning quota without finishing anything.

        The Council's single genuinely useful behaviour was *stop a doomed
        run*: during the logtriage build it was woken by a doom loop and had
        the authority to ``pause_project``.  Everything else it did was
        cost -- 47 arbiter failures, 9 decisions held for a human, and not one
        intervention that actually repaired anything -- so v2.9 deletes it.

        The reflex is worth keeping, so this keeps it with no model call at
        all.  The trigger is deliberately blunt and countable: a project that
        has completed zero tasks while spending more than
        ``BREAKER_TOKENS_PER_WINDOW`` tokens is not learning, it is looping.

        Why not reuse the Council's richer condition (a doom loop, a
        structural block)?  Both are already visible in the steering signals
        that feed the manifest, and a breaker that needs interpretation is a
        breaker that needs a model.  This one needs arithmetic.
        """
        if self._paused() or not config.BREAKER_ENABLED:
            return
        tokens_window = max(0, int(config.BREAKER_TOKENS_PER_WINDOW))
        if tokens_window <= 0:
            return
        for name, status in list(self.projects.items()):
            stats = status.get("board") or {}
            done = int(stats.get("done", 0) or 0)
            if done > 0:
                status["breaker"] = None
                continue
            spent = int((status.get("tokens") or {}).get("window", 0) or 0)
            if spent < tokens_window:
                continue
            evidence = {
                "tokens_in_window": spent,
                "done": done,
                "failed": stats.get("failed", 0),
                "blocked": stats.get("blocked", 0),
                "signals": [s.kind for s in
                            steering.monitor_for(self.workspace_root)
                            .evaluate(name) if s.actionable],
            }
            (self.agent_dir(name) / "pause.flag").write_text(
                "v2.9 circuit breaker: zero completions after "
                f"{spent} tokens in the window. Evidence: {evidence}\n",
                encoding="utf-8")
            status["breaker"] = {"at": "tokens_in_window", **evidence}
            log_event("WARNING", "breaker_tripped",
                      {"project": name, **evidence})

    def agent_dir(self, name: str) -> Path:
        return self.workspace_root / "projects" / name / ".agent"

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self, once: bool = False) -> None:
        log_event("INFO", "super_orchestrator_started",
                  {"workspace_root": str(self.workspace_root),
                   "demo_mode": config.DEMO_MODE,
                   "integration_agent": config.ENABLE_INTEGRATION_AGENT,
                   "breaker": config.BREAKER_ENABLED})
        self._start_integration_agent()
        for project_dir in self._discover_projects():
            self._spawn_project(project_dir.name)
        self._write_manifest()

        if once:
            # Single orchestration pass (useful for tests / CI): give the
            # projects one run window, wait for tasks to drain, then stop.
            await self._run_once()
            await self.shutdown()
            return

        try:
            while not self._stop.is_set():
                await asyncio.sleep(2.0)
                self._write_manifest()
                self._check_breaker()
                self._log_pause_transitions()
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()

    def _log_pause_transitions(self) -> None:
        """Log *transitions*, not every poll.

        The old loop emitted one INFO line every 2 seconds for the whole time
        the workspace was paused: 31,468 lines, 86% of the entire JSONL and
        4.84 MB, against 858 real LLM exchanges.  The pause state is already
        visible in the manifest; the log should record only the edges.
        """
        paused = self._paused()
        if paused and not self._paused_logged:
            self._paused_logged = True
            log_event("INFO", "super_orchestrator_paused",
                      {"note": "state change, not a poll"})
        elif not paused and self._paused_logged:
            self._paused_logged = False
            log_event("INFO", "super_orchestrator_resumed",
                      {"note": "state change, not a poll"})

    async def _run_once(self) -> None:
        """Wait (bounded) until every project has settled and none can act.

        Live LLM cycles are slow (planner + builder + reviewer calls can take
        minutes), so the single pass gets a generous budget
        (``config.ONCE_BUDGET_S``, default 10 minutes, env-overridable).

        v2.9.1 -- this used to answer "is there anything to do?" by grepping the
        boards for ``status == "pending"`` and breaking out on the first poll.
        Two things went wrong, both visible in the first three seconds of a run:

        * a goal-only project's board is *empty*, which is indistinguishable
          from a finished one, so ``--new-project NAME --goal ...`` followed by
          the documented ``--once`` returned in 3.5 s having authored nothing;
        * the pass could return before a single project loop had run its first
          scheduling step, so ``super_orchestrator_stopping`` was logged ahead
          of the project it was stopping and ``logtriage`` never emitted a
          start event at all.

        Now each loop reports ``loop_state``/``work_available`` when it makes a
        scheduling decision, and the pass waits for those reports.  ``ONCE_BOOT``
        bounds the wait for a loop that never reports (a wedged or starved
        task), so a bad workspace degrades to "proceed with what we have"
        instead of hanging until the budget expires.
        """
        deadline = time.monotonic() + float(config.ONCE_BUDGET_S)
        names = [d.name for d in self._discover_projects()]
        boot_deadline = time.monotonic() + float(config.ONCE_BOOT_S)
        outstanding = True
        # once_boot_timeout is edge-triggered, not level-triggered: log when
        # the unsettled set changes, or as a reminder at most every 60 s.
        # Every 0.5 s poll re-logging the same warning produced 100+ lines
        # per second against one wedged project (observed 2026-10-16).
        last_boot_warn: tuple = ()
        last_boot_warn_at = 0.0
        while time.monotonic() < deadline and not self._stop.is_set():
            unsettled = [n for n in names
                         if self._loop_unsettled(n)
                         and self.projects.get(n, {}).get("status")
                         not in ("crashed", "abandoned")]
            if not unsettled and not self._any_outstanding(names):
                outstanding = False
                break
            if unsettled and time.monotonic() > boot_deadline:
                now_s = time.monotonic()
                warn_signature = tuple(unsettled)
                if warn_signature != last_boot_warn \
                        or now_s - last_boot_warn_at >= 60.0:
                    last_boot_warn = warn_signature
                    last_boot_warn_at = now_s
                    log_event("WARNING", "once_boot_timeout",
                              {"unsettled": unsettled,
                               "boot_s": config.ONCE_BOOT_S,
                               "note": "a project loop never reported a "
                                       "scheduling decision; continuing with "
                                       "the rest"})
            # Let the integration agent drain queued events too.
            if self.integration_queue is not None:
                try:
                    await asyncio.wait_for(self.integration_queue.join(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
            await asyncio.sleep(0.5)
        # Drain integration events one last time.
        if self.integration_queue is not None:
            try:
                await asyncio.wait_for(self.integration_queue.join(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        if outstanding:
            still = [n for n in names if self.projects.get(n, {}).get("work_available")]
            log_event("WARNING", "once_budget_exhausted",
                      {"budget_s": config.ONCE_BUDGET_S,
                       "outstanding": still,
                       "note": "work remains; run the default loop for long "
                               "builds"})

    def _loop_unsettled(self, name: str) -> bool:
        """True until this project's loop has made its first decision."""
        state = self.projects.get(name) or {}
        return state.get("loop_state", "starting") == "starting"

    def _any_outstanding(self, names: List[str]) -> bool:
        return any(bool((self.projects.get(n) or {}).get("work_available"))
                   for n in names)

    async def shutdown(self) -> None:
        self._stop.set()
        log_event("INFO", "super_orchestrator_stopping", {})
        for name, task in list(self.tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        if self.integration_task and not self.integration_task.done():
            self.integration_task.cancel()
            try:
                await self.integration_task
            except asyncio.CancelledError:
                pass
        self._write_manifest()
