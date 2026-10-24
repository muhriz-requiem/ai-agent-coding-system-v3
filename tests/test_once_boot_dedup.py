"""once_boot_timeout is edge-triggered, not level-triggered -- 2026-10-17.

Fix extracted from the modinstall payoff build: one wedged project loop made
the ``--once`` pass re-log the same warning on every 0.5 s poll (100+ lines
per second observed live, ~1.1k lines in a 3 minute window). These pins hold
the file to its own "transitions, not polls" convention, as already applied
to pause spam in ``_log_pause_transitions``.
"""

import asyncio
from pathlib import Path

import config
import super_orchestrator


def _orch_with_project(tmp_path: Path, loop_state: str) -> "super_orchestrator.SuperOrchestrator":
    """A SuperOrchestrator with one synthetic project in its status table."""
    orch = super_orchestrator.SuperOrchestrator(tmp_path)
    name = "demo_project"
    (tmp_path / "projects" / name / ".agent").mkdir(parents=True)
    orch.projects[name] = {
        "status": "running",
        "board": {"done": 0, "failed": 0, "blocked": 0},
        "tokens": {"window": 0},
        "loop_state": loop_state,
    }
    return orch


def test_once_boot_timeout_edge_trigger_logs_once(tmp_path, monkeypatch):
    """A wedged loop surfaces ONCE per stable unsettled set, not per poll."""
    orch = _orch_with_project(tmp_path, "starting")
    monkeypatch.setattr(config, "ONCE_BUDGET_S", 1.6)
    monkeypatch.setattr(config, "ONCE_BOOT_S", 0.0)
    seen = []
    monkeypatch.setattr(
        super_orchestrator, "log_event",
        lambda lvl, msg, data=None: seen.append(msg))

    asyncio.run(orch._run_once())

    warns = [m for m in seen if m == "once_boot_timeout"]
    assert warns, seen  # the wedged loop must surface at all
    # The budget yields at least 3 polls inside the 1.6 s window; the
    # edge trigger keeps the identical situation to a single line.
    assert len(warns) == 1, seen


def test_once_run_settles_without_boot_warning(tmp_path, monkeypatch):
    """A workspace with nothing unsettled and nothing outstanding stops
    immediately and does not log boot warnings at all."""
    orch = _orch_with_project(tmp_path, "decided")
    monkeypatch.setattr(config, "ONCE_BUDGET_S", 1.0)
    monkeypatch.setattr(config, "ONCE_BOOT_S", 0.0)
    seen = []
    monkeypatch.setattr(
        super_orchestrator, "log_event",
        lambda lvl, msg, data=None: seen.append(msg))

    asyncio.run(orch._run_once())

    assert not [m for m in seen if m == "once_boot_timeout"], seen
