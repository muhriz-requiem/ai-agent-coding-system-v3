"""Divergence engine: policies, scoring, artifacts, orchestrator round."""

from __future__ import annotations

import asyncio
import json

import config
import divergence
from project_orchestrator import ProjectOrchestrator
from validation import Plan, StrategySpec, TaskSpec


def _task(**kwargs) -> TaskSpec:
    base = {"id": "T1", "title": "t", "objective": "o",
            "target_files": ["src/mod.py"]}
    base.update(kwargs)
    return TaskSpec(**base)


def test_default_strategies_by_kind():
    assert [s.name for s in divergence.default_strategies(_task())] == \
        ["minimal-diff", "robust-edges", "performance-first"]
    experiment = _task(task_type="experiment")
    assert "baseline-first" in [s.name for s in
                                divergence.default_strategies(experiment)]


def test_load_strategies_truncates_and_pads(tmp_path):
    task = _task(n_candidates=2)
    assert len(divergence.load_strategies(task, tmp_path)) == 2

    named = [StrategySpec(name="a", approach="x"),
             StrategySpec(name="b", approach="y"),
             StrategySpec(name="c", approach="z")]
    task = _task(strategies=named)
    assert [s.name for s in divergence.load_strategies(task, tmp_path)] == \
        ["a", "b", "c"]

    task = _task(n_candidates=4)
    assert len(divergence.load_strategies(task, tmp_path)) == 4


def test_enabled_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DIVERGENCE_ENABLED", False)
    assert not divergence.enabled(_task(), tmp_path)
    assert divergence.enabled(_task(divergent=True), tmp_path)
    assert not divergence.enabled(_task(divergent=False), tmp_path)

    agent_dir = tmp_path / ".agent"
    agent_dir.mkdir()
    (agent_dir / "divergence.json").write_text(
        json.dumps({"enabled": True, "n_candidates": 3}), encoding="utf-8")
    assert divergence.enabled(_task(), tmp_path)
    assert divergence.candidate_count(_task(), tmp_path, []) == 3

    # A policy that only tunes candidates falls through to the global flag.
    (agent_dir / "divergence.json").write_text(
        json.dumps({"n_candidates": 2}), encoding="utf-8")
    assert not divergence.enabled(_task(), tmp_path)
    monkeypatch.setattr(config, "DIVERGENCE_ENABLED", True)
    assert divergence.enabled(_task(), tmp_path)


def _record(label, strategy=None, **kwargs):
    record = divergence.CandidateRecord(
        label=label, strategy=strategy or StrategySpec(name="s"))
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def test_select_winner_gate_first():
    records = [
        _record("A", gate_ok=False, metric=0.99),
        _record("B", gate_ok=True, metric=0.50),
    ]
    assert divergence.select_winner(records) == 1


def test_select_winner_metric_direction():
    maximize = [_record("A", gate_ok=True, metric=0.7, direction="maximize"),
                _record("B", gate_ok=True, metric=0.8, direction="maximize")]
    assert divergence.select_winner(maximize) == 1
    minimize = [_record("A", gate_ok=True, metric=0.7, direction="minimize"),
                _record("B", gate_ok=True, metric=0.8, direction="minimize")]
    assert divergence.select_winner(minimize) == 0


def test_select_winner_tests_then_lint_then_size():
    records = [
        _record("A", gate_ok=True, tests_passed=5, tests_failed=0,
                lint_count=3, diff="x" * 100),
        _record("B", gate_ok=True, tests_passed=5, tests_failed=0,
                lint_count=0, diff="x" * 10),
    ]
    assert divergence.select_winner(records) == 1


def test_near_tie():
    records = [
        _record("A", gate_ok=True, metric=0.800),
        _record("B", gate_ok=True, metric=0.795),
    ]
    assert divergence.is_near_tie(records)
    records[1].metric = 0.500
    assert not divergence.is_near_tie(records)


def test_merge_plans_dedupes_and_orders():
    plans = [
        Plan(task_id="T", steps=["a", "b"], edge_cases=["empty"]),
        Plan(task_id="T", steps=["b", "c"], edge_cases=["empty", "null"]),
    ]
    merged = divergence.merge_plans("T", plans)
    assert merged.steps == ["a", "b", "c"]
    assert merged.edge_cases == ["empty", "null"]


def test_record_from_gate_and_store(tmp_path):
    task = _task()
    gate = {"ok": True, "tests": {"passed": 7, "failed": 0},
            "lint": ["E501"], "snapshot": None}
    record = divergence.record_from_gate(
        "A", StrategySpec(name="s", approach="x"), "diff-text", gate,
        "builder_initial", {"provider": "groq"}, False, task)
    assert record.gate_ok and record.tests_passed == 7

    path = divergence.store_candidate(tmp_path, "T1", record)
    assert path is not None and path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["label"] == "A" and payload["gate_ok"] is True


def test_record_from_gate_objective():
    task = _task(task_type="experiment", metric={"name": "auc"})
    gate = {"ok": True, "objective": {"ok": True, "primary": 0.9,
                                      "runtime_s": 12.0}}
    record = divergence.record_from_gate(
        "B", StrategySpec(name="s"), "diff", gate, "model", None, False, task)
    assert record.metric == 0.9 and record.gate_ok


def _demo_task_dict(task_id: str = "T-DIV") -> dict:
    return {
        "id": task_id, "title": "divergent demo", "objective": "o",
        "target_files": ["src/mod.py"], "dependencies": [], "priority": "high",
        "acceptance_criteria": [], "workspace_patterns_referenced": [],
        "retry_count": 0, "status": "pending", "last_feedback": None,
        "error_signature": None, "task_type": "feature", "should_pass": True,
        "requires_new_tests": False, "divergent": True, "n_candidates": 2,
    }


def test_divergence_round_demo_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DIVERGENCE_ENABLED", True)
    project = tmp_path / "projects" / "toy"
    (project / "src").mkdir(parents=True)
    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    task = TaskSpec(**_demo_task_dict())
    result = asyncio.run(orchestrator._run_divergence(task, Plan(task_id=task.id)))
    assert result is not None
    build, gate, _usage = result
    assert "divergence winner" in build.notes
    assert gate["ok"]
    artifacts = list((project / ".agent" / "candidates" / "T-DIV").glob("*.json"))
    assert len(artifacts) == 2
    # DEMO_MODE never touches project source files.
    assert not (project / "src" / "mod.py").exists()


def test_process_task_divergent_demo_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DIVERGENCE_ENABLED", True)
    project = tmp_path / "projects" / "toy"
    agent = project / ".agent"
    agent.mkdir(parents=True)
    board = {"tasks": [_demo_task_dict("T-E2E")]}
    (agent / "task_board.json").write_text(json.dumps(board), encoding="utf-8")
    orchestrator = ProjectOrchestrator(tmp_path, "toy")
    asyncio.run(orchestrator.process_task(TaskSpec(**_demo_task_dict("T-E2E"))))
    updated = json.loads((agent / "task_board.json").read_text(encoding="utf-8"))
    assert updated["tasks"][0]["status"] == "done"


def test_merge_payload_filters_non_passing():
    rec1 = _record("A", gate_ok=True, diff="def a(): pass")
    rec2 = _record("B", gate_ok=True, diff="def b(): pass")
    rec3 = _record("C", gate_ok=False, diff="def c(): pass")
    payload = divergence.merge_payload([rec1, rec2, rec3])
    assert len(payload) == 2
    assert all(p["gate_ok"] for p in payload)
    assert payload[0]["label"] in ("A", "B")


def test_merge_round_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DIVERGENCE_ENABLED", True)
    monkeypatch.setattr(config, "DIVERGENCE_MERGE_ENABLED", True)
    project = tmp_path / "projects" / "toy"
    (project / "src").mkdir(parents=True)
    orc = ProjectOrchestrator(tmp_path, "toy")

    # Monkeypatch the merger to return a valid content block.
    async def fake_merge(task, candidates, context=None):
        return "```python path=src/mod.py\nVALUE = 2\n```"
    monkeypatch.setattr(orc.merger, "run", fake_merge)

    task = TaskSpec(**_demo_task_dict("T-MERGE"))
    plan = Plan(task_id="T-MERGE")
    result = asyncio.run(orc._run_divergence(task, plan))
    assert result is not None
    build, _gate, _usage = result
    # The merged candidate ("M") should be the winner.
    assert "M" in build.notes
    artifacts = list((project / ".agent" / "candidates" / "T-MERGE").glob("*.json"))
    assert len(artifacts) == 3  # A, B, M


def test_merge_skipped_when_invalid(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DEMO_MODE", True)
    monkeypatch.setattr(config, "DIVERGENCE_ENABLED", True)
    monkeypatch.setattr(config, "DIVERGENCE_MERGE_ENABLED", True)
    project = tmp_path / "projects" / "toy"
    (project / "src").mkdir(parents=True)
    orc = ProjectOrchestrator(tmp_path, "toy")

    async def fake_bad(task, candidates, context=None):
        return "no fences here"  # invalid -> merger returns None
    monkeypatch.setattr(orc.merger, "run", fake_bad)

    task = TaskSpec(**_demo_task_dict("T-REJ"))
    plan = Plan(task_id="T-REJ")
    result = asyncio.run(orc._run_divergence(task, plan))
    assert result is not None
    # Winner must be one of the original candidates (A or B), not M.
    assert "M" not in result[0].notes
