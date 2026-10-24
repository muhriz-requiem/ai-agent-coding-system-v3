"""Objective gate: pure rule, executor, and the apply+run rail."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from project_orchestrator import (ProjectOrchestrator, objective_gate_block,
                                  _primary_value)
from validation import MetricSpec, TaskSpec

_WRITE_METRICS = '''\
import json
import pathlib
import sys
import time

payload = {}
for arg in sys.argv[1:]:
    key, _, value = arg.partition("=")
    try:
        payload[key] = float(value) if "." in value else int(value)
    except ValueError:
        payload[key] = value

delay = float(payload.pop("delay", 0))
if delay:
    time.sleep(delay)
exit_code = int(payload.pop("exit_code", 0))
pathlib.Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
sys.exit(exit_code)
'''


def _metric(**kwargs) -> MetricSpec:
    base = {"name": "auc", "direction": "maximize"}
    base.update(kwargs)
    return MetricSpec(**base)


def test_block_rule_shapes():
    blocked, _ = objective_gate_block(None, {}, None)
    assert not blocked

    blocked, reason = objective_gate_block(_metric(), {"ok": False}, None)
    assert blocked and "failed" in reason

    blocked, _ = objective_gate_block(
        _metric(target=0.9), {"ok": True, "primary": 0.8}, None)
    assert blocked

    blocked, _ = objective_gate_block(
        _metric(target=0.9), {"ok": True, "primary": 0.95}, None)
    assert not blocked

    blocked, _ = objective_gate_block(
        _metric(direction="minimize", target=0.1),
        {"ok": True, "primary": 0.05}, None)
    assert not blocked

    blocked, _ = objective_gate_block(
        _metric(min_improvement=0.05), {"ok": True, "primary": 0.84}, 0.80)
    assert blocked
    blocked, _ = objective_gate_block(
        _metric(min_improvement=0.05), {"ok": True, "primary": 0.86}, 0.80)
    assert not blocked

    # Scores from different splits are never comparable.
    blocked, reason = objective_gate_block(
        _metric(split_key="fold-v1"),
        {"ok": True, "primary": 0.9, "split_hash": "fold-v2"}, None)
    assert blocked and "split_hash" in reason


def test_primary_value_extraction():
    assert _primary_value({"auc": 0.7}, "auc") == 0.7
    assert _primary_value({"primary": 1}, "other") == 1.0
    assert _primary_value({"auc": "0.7"}, "auc") is None
    assert _primary_value({}, "auc") is None


def _command(tmp_path: Path, *args: str) -> str:
    (tmp_path / "write_metrics.py").write_text(_WRITE_METRICS, encoding="utf-8")
    return f'"{sys.executable}" write_metrics.py ' + " ".join(args)


def test_run_objective_happy_path(tmp_path):
    import executors
    result = executors.run_objective(
        tmp_path, _command(tmp_path, "auc=0.81", "split_hash=s1"),
        required_fields=["auc"])
    assert result["ok"], result["detail"]
    assert result["metrics"]["auc"] == 0.81
    assert result["split_hash"] == "s1"


def test_run_objective_fail_closed(tmp_path):
    import executors
    # Missing metrics file -> fail closed.
    (tmp_path / "noop.py").write_text("print('nothing')", encoding="utf-8")
    result = executors.run_objective(tmp_path, f'"{sys.executable}" noop.py')
    assert not result["ok"]

    # Non-zero exit with metrics present -> fail closed.
    result = executors.run_objective(
        tmp_path, _command(tmp_path, "auc=0.9", "exit_code=2"))
    assert not result["ok"]

    # Missing required field -> fail closed.
    result = executors.run_objective(
        tmp_path, _command(tmp_path, "loss=0.1"), required_fields=["auc"])
    assert not result["ok"] and "required fields" in result["detail"]

    # Stale evidence: a pre-existing metrics file must never satisfy the
    # gate for a command that does not write one this run.
    stale = tmp_path / "stale"
    (stale / "reports").mkdir(parents=True)
    (stale / "reports" / "metrics.json").write_text(
        '{"auc": 0.99}', encoding="utf-8")
    (stale / "noop.py").write_text("print('no write')", encoding="utf-8")
    result = executors.run_objective(
        stale, f'"{sys.executable}" noop.py', required_fields=["auc"])
    assert not result["ok"]


def test_run_objective_timeout_kills_tree(tmp_path):
    import executors
    result = executors.run_objective(
        tmp_path, _command(tmp_path, "auc=1.0", "delay=5"), timeout=1)
    assert not result["ok"] and "timed out" in result["detail"]


_NEW_FILE_DIFF = """--- /dev/null
+++ b/train.py
@@ -0,0 +1,3 @@
+import json, pathlib
+pathlib.Path("reports").mkdir(exist_ok=True)
+json.dump({"auc": 0.9, "split_hash": "s1"}, open("reports/metrics.json", "w"))
"""


def test_apply_and_objective_gate_rail(tmp_path):
    project = tmp_path / "projects" / "toy"
    (project / "src").mkdir(parents=True)
    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(
        id="T-EXP", title="toy", objective="write train.py",
        target_files=["train.py"],
        metric=_metric(split_key="s1", target=0.85),
        run_command=f'"{sys.executable}" train.py')
    gate = asyncio.run(
        orchestrator._apply_and_objective_gate(task, _NEW_FILE_DIFF))
    assert gate["ok"], gate["feedback"]
    assert gate["objective"]["ok"]
    assert gate["objective"]["primary"] == 0.9
    assert "train.py" in gate["applied"]
    # Metrics file was produced inside the project root.
    assert (project / "reports" / "metrics.json").exists()
