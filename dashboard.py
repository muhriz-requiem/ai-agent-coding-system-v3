"""Dashboard windows: a watch-only feed and a dedicated command console.

The old single-TUI (Rich Live + input in one console) fought itself for
the terminal, so the two concerns are now separate processes/windows:

* ``run_feed()``    (``python main.py --feed``)  -- watch-only live view:
  per-project panels + event stream.  Rich Live is safe here because
  nothing else writes to this console and there is no input prompt.
* ``run_command()`` (``python main.py --cmd``)   -- a plain command REPL
  (prompt_toolkit).  Nothing else writes to this console, so every typed
  character and every response is fully visible.

Both operate on the same file-based control plane the orchestrator polls
(``pause.flag``, task boards, ``context.md``), so no IPC is needed -- they
work even attached to an orchestrator started in another window.

``python main.py --dashboard`` starts the orchestrator headless and spawns
both windows as separate console processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config
from agent_roles import LLMClient
from logger import log_event

_WIDTH = 110
_HELP = ("commands: pause | resume | status | skip <task_id> | "
         "retry <task_id> | note <project> <text> | ask <question> | exit")


# ---------------------------------------------------------------------------
# Shared helpers (file-based control plane -- no orchestrator IPC needed)
# ---------------------------------------------------------------------------
def _workspace() -> Path:
    return config.WORKSPACE_ROOT


def _read_manifest() -> Dict[str, Any]:
    try:
        return json.loads(
            (_workspace() / ".system" / "manifest.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _count_board(payload: Dict[str, Any]) -> Counter:
    """Pure board summary (v2.5.5): status counts + ``blocked`` (pending
    tasks with unmet dependencies) + ``failed_ids`` -- the starvation
    picture at a glance."""
    counts: Counter = Counter()
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    done_ids = {t.get("id") for t in tasks if t.get("status") == "done"}
    for task in tasks:
        status = task.get("status", "?")
        counts[status] += 1
        if status == "pending":
            deps = task.get("dependencies") or []
            if any(dep not in done_ids for dep in deps):
                counts["blocked"] += 1
        if status == "failed":
            counts.setdefault("failed_ids", [])
            counts["failed_ids"].append(str(task.get("id")))
    return counts


def _board_counts(project: str) -> Counter:
    board = _workspace() / "projects" / project / ".agent" / "task_board.json"
    try:
        payload = json.loads(board.read_text(encoding="utf-8"))
        return _count_board(payload)
    except (json.JSONDecodeError, OSError):
        return Counter()


def _project_names() -> list:
    projects_dir = _workspace() / "projects"
    if not projects_dir.is_dir():
        return []
    return sorted(d.name for d in projects_dir.iterdir() if d.is_dir())


def _edit_task(task_id: str, status: str) -> str:
    """Set a task's status on whichever board holds it."""
    for name in _project_names():
        board = _workspace() / "projects" / name / ".agent" / "task_board.json"
        try:
            payload = json.loads(board.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for task in payload.get("tasks", []):
            if task.get("id") == task_id:
                task["status"] = status
                if status == "pending":
                    # Operator retry = a fresh budget: the manual reset also
                    # clears the bounded auto-requeue bookkeeping (v2.5.5).
                    task["retry_count"] = 0
                    task["last_feedback"] = None
                    task["failed_at"] = None
                    task["auto_requeues"] = 0
                fd, tmp = tempfile.mkstemp(dir=str(board.parent))
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2)
                    handle.write("\n")
                os.replace(tmp, board)
                log_event("INFO", "dashboard_task_edited",
                          {"task_id": task_id, "status": status})
                return f"{task_id} -> {status} (on {name})"
    return f"task {task_id} not found"


def _paused() -> bool:
    return (_workspace() / "pause.flag").exists()


def _poll_log_tail(offset: int, limit: int) -> tuple:
    """Read new JSONL entries from ``offset``; returns (events, new_offset)."""
    log_path = _workspace() / "logs" / "agent_log.jsonl"
    events: Deque[Dict[str, Any]] = deque(maxlen=limit)
    try:
        size = log_path.stat().st_size
        if size < offset:
            offset = 0
        if size > offset:
            with open(log_path, "r", encoding="utf-8") as handle:
                handle.seek(offset)
                for line in handle:
                    try:
                        entry = json.loads(line)
                        events.append({
                            "ts": entry.get("ts", "")[11:19],
                            "level": entry.get("level", "INFO"),
                            "message": entry.get("message", "")[:90],
                            "task": entry.get("data", {}).get("task_id", ""),
                        })
                    except json.JSONDecodeError:
                        continue
                offset = handle.tell()
    except OSError:
        pass
    return events, offset


# ---------------------------------------------------------------------------
# Feed window (watch-only)
# ---------------------------------------------------------------------------
def run_feed() -> None:
    """Watch-only live view. Ctrl+C to close."""
    console = Console()
    log_offset = 0
    paused_notified = None

    console.print("[bold cyan]Agent event feed attached "
                  "(Ctrl+C to close).[/bold cyan]")
    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                manifest = _read_manifest()
                paused = bool(manifest.get("paused"))
                if paused != paused_notified:
                    paused_notified = paused

                header = Table.grid(padding=(0, 2))
                header.add_column()
                header.add_column()
                mode = "DEMO" if config.DEMO_MODE else "LIVE"
                header.add_row(
                    Text(f"mode={mode}  paused={'YES' if paused else 'no'}",
                         style="bold cyan"),
                    Text(f"updated={datetime.now(timezone.utc).strftime('%H:%M:%S')}",
                         style="bright_black"))

                project_table = Table(expand=True)
                project_table.add_column("project", style="bold")
                project_table.add_column("status")
                project_table.add_column("current task")
                project_table.add_column("done")
                project_table.add_column("failed")
                project_table.add_column("pending")
                project_table.add_column("blocked")
                projects = manifest.get("projects", {})
                for name in _project_names():
                    state = projects.get(name, {})
                    counts = _board_counts(name)
                    color = {"running": "green", "idle": "cyan",
                             "paused": "yellow", "crashed": "red",
                             "abandoned": "red"}.get(state.get("status"),
                                                     "white")
                    project_table.add_row(
                        name,
                        Text(state.get("status", "?"), style=color),
                        state.get("current_task_id") or "-",
                        str(counts.get("done", 0)),
                        Text(str(counts.get("failed", 0)),
                             style="red" if counts.get("failed") else "white"),
                        str(counts.get("pending", 0)),
                        Text(str(counts.get("blocked", 0)),
                             style="yellow" if counts.get("blocked") else "white"))
                integration = manifest.get("integration_agent", {})
                project_table.add_row(
                    "integration_agent", integration.get("status", "off"),
                    "-", "-", "-", "-", "-")

                events, log_offset = _poll_log_tail(log_offset, 10)
                feed = Text()
                for event in events:
                    color = {"INFO": "white", "WARNING": "yellow",
                             "ERROR": "red",
                             "CRITICAL": "bold red"}.get(event["level"],
                                                         "white")
                    feed.append(f"{event['ts']} ", style="bright_black")
                    feed.append(event["message"], style=color)
                    if event["task"]:
                        feed.append(f" [{event['task']}]",
                                    style="bright_black")
                    feed.append("\n")

                live.update(Group(
                    Panel(header, title="[bold]Agent Activity Feed[/bold]",
                          width=_WIDTH),
                    Panel(project_table, title="projects", width=_WIDTH),
                    Panel(feed, title="event feed (newest last)",
                          width=_WIDTH)))
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    console.print("[bright_black]feed closed[/bright_black]")


# ---------------------------------------------------------------------------
# Command console (dedicated REPL)
# ---------------------------------------------------------------------------
class CommandConsole:
    """Plain command REPL: every typed line and response fully visible."""

    def __init__(self) -> None:
        self.llm = LLMClient(_workspace())

    # -- command implementations -------------------------------------------
    def _workspace_context(self) -> str:
        manifest = _read_manifest()
        projects = manifest.get("projects", {})
        parts = []
        for name in _project_names():
            state = projects.get(name, {})
            counts = _board_counts(name)
            parts.append(f"{name}: status={state.get('status', '?')} "
                         f"current={state.get('current_task_id', '-')} "
                         f"done={counts.get('done', 0)} "
                         f"failed={counts.get('failed', 0)} "
                         f"pending={counts.get('pending', 0)}")
        return "Projects:\n" + "\n".join(parts)

    def _status(self) -> str:
        lines = [self._workspace_context(),
                 f"paused={_paused()}"]
        return "\n".join(lines)

    def execute(self, line: str) -> str:
        parts = line.strip().split(maxsplit=2)
        if not parts:
            return "(empty command)"
        command = parts[0].lower()
        if command == "pause":
            (_workspace() / "pause.flag").write_text("", encoding="utf-8")
            log_event("INFO", "dashboard_pause", {})
            return "pause.flag created -- all loops pausing"
        if command == "resume":
            (_workspace() / "pause.flag").unlink(missing_ok=True)
            log_event("INFO", "dashboard_resume", {})
            return "pause.flag removed -- loops resume"
        if command == "status":
            return self._status()
        if command in ("skip", "retry") and len(parts) >= 2:
            return _edit_task(parts[1],
                              "blocked" if command == "skip" else "pending")
        if command == "note" and len(parts) >= 3:
            project, text = parts[1], parts[2]
            context_path = (_workspace() / "projects" / project / ".agent"
                            / "context.md")
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            with open(context_path, "a", encoding="utf-8") as handle:
                handle.write(f"\n- [operator note {stamp}] {text}\n")
            return f"note appended to {project}/context.md"
        if command == "ask":
            return "use: ask <question>  (this line is handled async)"
        return f"unknown command: {command}\n{_HELP}"

    async def _ask(self, question: str) -> str:
        system = ("You are the SuperOrchestrator assistant of a multi-agent "
                  "coding system. Answer questions about workspace state and "
                  "the build in progress concisely using the provided live "
                  "context. If asked to change something, explain which "
                  "command does it (pause/resume/skip/retry/note).")
        user = (f"Live workspace context:\n{self._workspace_context()}\n\n"
                f"Question: {question}")
        try:
            return await self.llm.complete(system=system, user=user,
                                           model_role="super_orchestrator")
        except Exception as exc:  # noqa: BLE001 -- never crash the console
            return f"(orchestrator unavailable: {exc})"

    # -- REPL ----------------------------------------------------------------
    async def run_command(self) -> None:
        try:
            from prompt_toolkit import PromptSession
            session = PromptSession()
        except Exception as exc:  # noqa: BLE001 -- no console (redirected IO)
            log_event("WARNING", "command_console_no_tty",
                      {"error": str(exc)})
            return self._run_plain_repl()

        console = Console()
        console.print("[bold cyan]Command console attached. "
                      "Type 'exit' to close.[/bold cyan]")
        console.print(f"[bright_black]{_HELP}[/bright_black]")
        while True:
            try:
                line = await session.prompt_async("command> ")
            except (EOFError, KeyboardInterrupt):
                break
            line = (line or "").strip()
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            if line.lower().startswith("ask "):
                question = line[4:].strip()
                console.print("[bright_black]thinking...[/bright_black]")
                answer = await self._ask(question)
                console.print(f"[green]{answer}[/green]\n")
            else:
                console.print(f"[yellow]{self.execute(line)}[/yellow]\n")
        console.print("[bright_black]command console closed[/bright_black]")

    def _run_plain_repl(self) -> None:
        """Fallback REPL for non-console contexts (plain input())."""
        print(f"Command console (plain mode). {_HELP}")
        while True:
            try:
                line = input("command> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("exit", "quit"):
                break
            print(self.execute(line))


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def run_command() -> None:
    try:
        asyncio.run(CommandConsole().run_command())
    except KeyboardInterrupt:
        pass


def spawn_dashboard_windows() -> None:
    """Open the feed + command windows as separate console processes."""
    import subprocess
    import sys

    # dashboard.py sits beside main.py in the workspace root.
    entry = Path(__file__).resolve().parent / "main.py"
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    for window_args in (["--feed"], ["--cmd"]):
        subprocess.Popen(
            [sys.executable, str(entry), *window_args],
            cwd=str(_workspace()), creationflags=flags)
    log_event("INFO", "dashboard_windows_spawned",
              {"windows": ["feed", "cmd"]})
