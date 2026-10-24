# -*- coding: utf-8 -*-
"""Portfolio registry (Phase 8b, read side): the parent's one view.

The "company" roadmap needs a single place where the operator can see every
subsidiary's state without opening nine boards.  This module is deliberately
READ-SIDE ONLY: every input is a file the orchestrator already maintains
(boards, goal files), every output is derived on demand, and nothing here
writes into any project.  The orchestrator-side *write* hook (escalations
pushed into an inbox at escalation time) touches operator-WIP retry logic
and therefore waits for the operator lane; until then the DIANGOSIS-prefixed
``last_feedback`` lines on drained boards are derived here instead.

Admission rule (kills a documented drift class): a folder counts as a
subsidiary only when it has BOTH ``.agent/task_board.json`` AND ``goal.txt``.
Board-without-goal and boardless folders are reported as OTHER/RESEARCH with
the reason, so half-formed children are visible instead of being recurring
``task_board_unreadable`` noise.

Also writes a derived snapshot to ``.system/portfolio.json`` (tmp+replace)
for the planned workspace-MCP lane to serve read-only.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SNAPSHOT_SCHEMA = "portfolio/1.0"

KIND_SUBSIDIARY = "subsidiary"
KIND_OTHER = "other"
KIND_RESEARCH = "research"

ESCALATION_PREFIX = "DIAGNOSIS:"


def parse_iso(value: Optional[str]) -> Optional[float]:
    """Unix timestamp for a board ISO stamp (Z or offset), tolerant, else None.

    A malformed ``defer_until`` must not crash the registry or silently park
    forever: report it as unknown instead (the caller decides severity).
    """
    if not value or not isinstance(value, str):
        return None
    stamp = value.strip()
    if stamp.endswith("Z"):
        stamp = stamp[:-1] + "+00:00"
    try:
        from datetime import datetime
        return datetime.fromisoformat(stamp).timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def _load_json(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable ({exc})"


def classify(project_dir: Path) -> Dict[str, Any]:
    """One project's derived registry row; never raises on data problems."""
    name = project_dir.name
    board, board_error = _load_json(
        project_dir / ".agent" / "task_board.json")
    goal = (project_dir / "goal.txt").is_file()
    row: Dict[str, Any] = {
        "name": name,
        "goal": goal,
        "board_error": board_error,
        "tasks": 0,
        "pending": 0,
        "done": 0,
        "failed": 0,
        "retries": 0,
        "parked": [],
        "escalations": [],
    }
    if board is None:
        if board_error == "missing":
            row["kind"] = KIND_RESEARCH
            row["note"] = "no task board"
        else:
            row["kind"] = KIND_OTHER
            row["note"] = board_error
        return row
    task_rows = board.get("tasks") or []
    if not isinstance(task_rows, list):
        row["kind"] = KIND_OTHER
        row["note"] = "board has no task list"
        return row
    pending_now = time.time()
    for task in task_rows:
        if not isinstance(task, dict):
            continue
        row["tasks"] += 1
        status = str(task.get("status") or "").lower()
        if status == "done":
            row["done"] += 1
        elif status == "failed":
            row["failed"] += 1
        else:
            row["pending"] += 1
        try:
            row["retries"] += int(task.get("retry_count") or 0)
        except (TypeError, ValueError):
            pass
        li_defer = task.get("defer_until")
        defer_at = parse_iso(li_defer if isinstance(li_defer, str) else None)
        if defer_at is not None and status != "done" and defer_at > pending_now:
            row["parked"].append({"task": str(task.get("id")),
                                  "defer_until": li_defer})
        feedback = task.get("last_feedback")
        if isinstance(feedback, str) and \
                feedback.lstrip().startswith(ESCALATION_PREFIX):
            row["escalations"].append(
                {"task": str(task.get("id")),
                 "diagnosis": feedback.lstrip()[len(ESCALATION_PREFIX):
                                               ].strip().splitlines()[0][:160]})
        elif status == "failed":
            row["escalations"].append(
                {"task": str(task.get("id")),
                 "diagnosis": "(failed status, no DIAGNOSIS feedback)"})
    if goal:
        row["kind"] = KIND_SUBSIDIARY
        row["note"] = ""
    else:
        row["kind"] = KIND_OTHER
        row["note"] = ("board present but goal.txt missing; not admissible "
                       "as a subsidiary until its goal is written")
    return row


def is_drained(row: Dict[str, Any]) -> bool:
    return (row["kind"] == KIND_SUBSIDIARY and row["tasks"] > 0
            and row["done"] == row["tasks"] and not row["parked"])


def build_report(workspace: Path) -> Dict[str, Any]:
    """Full registry snapshot over every folder under ``projects/``."""
    projects_root = Path(workspace) / "projects"
    rows: List[Dict[str, Any]] = []
    if projects_root.is_dir():
        for child in sorted(projects_root.iterdir()):
            if not child.is_dir() or child.name.startswith(
                    (".", "__", "_")):
                continue
            # Admission is a classification OUTCOME, not a pre-filter: a
            # bare or half-built folder must appear with its reason, not
            # vanish into recurring noise.
            rows.append(classify(child))
    subsidiaries = [r for r in rows if r["kind"] == KIND_SUBSIDIARY]
    report = {
        "schema": SNAPSHOT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(workspace),
        "projects": rows,
        "counts": {"subsidiaries": len(subsidiaries),
                   "drained": sum(1 for r in subsidiaries if is_drained(r)),
                   "other": sum(1 for r in rows if r["kind"] == KIND_OTHER),
                   "research": sum(1 for r in rows
                                   if r["kind"] == KIND_RESEARCH),
                   "escalations": sum(len(r["escalations"]) for r in rows),
                   "parked_tasks": sum(len(r["parked"]) for r in rows)},
        "_note": ("read-side derivative of board files; orchestrator-side "
                  "escalation inbox write awaits operator lane merge"),
    }
    return report


def render_text(report: Dict[str, Any]) -> List[str]:
    """Operator-readable lines; escalation inbox first, then the roster."""
    counts = report.get("counts") or {}
    lines = [
        f"company roster ({counts.get('subsidiaries', 0)} subsidiary(ies), "
        f"{counts.get('drained', 0)} drained, "
        f"{counts.get('escalations', 0)} escalation(s), "
        f"{counts.get('parked_tasks', 0)} parked task(s))",
    ]
    escalations = []
    parked = []
    for row in report.get("projects") or []:
        for esc in row.get("escalations") or []:
            escalations.append(f"  {row['name']}/{esc['task']}: "
                               f"{esc['diagnosis']}")
        for park in row.get("parked") or []:
            parked.append(f"  {row['name']}/{park['task']} "
                          f"until {park['defer_until']}")
    if escalations or parked:
        lines.append("escalation inbox:")
        lines.extend(escalations)
        lines.extend(parked)
    else:
        lines.append("escalation inbox: empty")
    lines.append("projects:")
    for row in report.get("projects") or []:
        kind = row["kind"]
        if kind == KIND_SUBSIDIARY:
            shape = (f"tasks={row['tasks']} done={row['done']} "
                     f"pending={row['pending']} failed={row['failed']} "
                     f"retries={row['retries']}"
                     + (" [DRAINED]" if is_drained(row) else ""))
        else:
            shape = row.get("note") or ""
        lines.append(f"  {row['name']}: {kind}"
                     + (f" -- {shape}" if shape else ""))
    return lines


def write_snapshot(workspace: Path, report: Dict[str, Any]) -> Path:
    """Derived-data snapshot under ``.system/`` (tmp + atomic replace)."""
    root = Path(workspace) / ".system"
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / "portfolio.json.tmp"
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    target = root / "portfolio.json"
    tmp.replace(target)
    return target
