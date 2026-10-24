"""Kaggle domain pack: scaffold, campaign director, submission gate."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import config
import domain_registry
import executors
from domains import kaggle_pack
from validation import ObjectiveGateResult


def test_scaffold_layout_and_refuses_clobber(tmp_path):
    project = kaggle_pack.scaffold("titanic_agent", "titanic", "accuracy",
                                   "maximize", 0.85, tmp_path)
    for rel in ("data/raw", "experiments", "reports", "submissions",
                ".agent/context.md", ".agent/domain.json",
                ".agent/campaign.json", "requirements.txt"):
        assert (project / rel).exists(), rel

    domain = json.loads((project / ".agent" / "domain.json").read_text("utf-8"))
    assert domain["pack"] == "kaggle" and domain["metric"] == "accuracy"
    campaign = kaggle_pack.load_campaign(project)
    assert campaign["pending_hypotheses"]

    # A non-empty board is never clobbered.
    board = project / ".agent" / "task_board.json"
    board.write_text(json.dumps({"tasks": [{"id": "X"}]}), encoding="utf-8")
    try:
        kaggle_pack.scaffold("titanic_agent", "titanic", root=tmp_path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected FileExistsError")


def test_registry_active_pack(tmp_path):
    project = tmp_path / "projects" / "toy"
    agent = project / ".agent"
    agent.mkdir(parents=True)
    (agent / "domain.json").write_text(json.dumps(
        {"pack": "kaggle", "competition": "titanic", "metric": "auc",
         "direction": "maximize"}), encoding="utf-8")
    pack = domain_registry.active_pack(project)
    assert pack is not None and pack.name == "kaggle"
    assert pack.competition == "titanic" and pack.metric == "auc"

    (agent / "domain.json").write_text(json.dumps({"pack": "nope"}),
                                       encoding="utf-8")
    assert domain_registry.active_pack(project) is None


def test_propose_tasks_queue_and_metric_contract(tmp_path, monkeypatch):
    project = kaggle_pack.scaffold("toycamp", "titanic", "auc", "maximize",
                                   None, tmp_path)
    monkeypatch.setattr(config, "CAMPAIGN_AUTO_GENERATE", False)
    pack = kaggle_pack.PACK
    board = {"tasks": []}
    assert asyncio.run(pack.propose_tasks(project, board)) == []

    monkeypatch.setattr(config, "CAMPAIGN_AUTO_GENERATE", True)
    monkeypatch.setattr(config, "CAMPAIGN_MAX_NEW_TASKS", 2)
    tasks = asyncio.run(pack.propose_tasks(project, board))
    assert len(tasks) == 2
    first = tasks[0]
    assert first["task_type"] == "experiment"
    assert first["metric"]["name"] == "auc"
    assert first["metric"]["required_fields"] == ["split_hash", "auc"]
    assert first["run_command"].startswith("python experiments/")
    # The queue is committed only once the core accepts the tasks.
    campaign = kaggle_pack.load_campaign(project)
    assert len(campaign["pending_hypotheses"]) == 2
    asyncio.run(pack.tasks_accepted(project, tasks))
    campaign = kaggle_pack.load_campaign(project)
    assert len(campaign["pending_hypotheses"]) == 0


def test_active_pack_is_isolated_per_project(tmp_path):
    for name, competition in (("a", "titanic"), ("b", "house-prices")):
        agent = tmp_path / "projects" / name / ".agent"
        agent.mkdir(parents=True)
        (agent / "domain.json").write_text(json.dumps(
            {"pack": "kaggle", "competition": competition}),
            encoding="utf-8")
    pack_a = domain_registry.active_pack(tmp_path / "projects" / "a")
    pack_b = domain_registry.active_pack(tmp_path / "projects" / "b")
    assert pack_a is not pack_b
    assert pack_a.competition == "titanic"
    assert pack_b.competition == "house-prices"


def _project_with_submission(tmp_path, sample, submission):
    project = kaggle_pack.scaffold("subcamp", "titanic", "auc", "maximize",
                                   None, tmp_path)
    raw = project / "data" / "raw"
    (raw / "sample_submission.csv").write_text(sample, encoding="utf-8")
    (project / "submissions" / "submission.csv").write_text(
        submission, encoding="utf-8")
    kaggle_pack.save_campaign(project, {
        "competition": "titanic", "metric": "auc", "direction": "maximize",
        "best_cv": None, "min_improvement": 0.0, "pending_hypotheses": [],
        "experiments": [], "submissions": [],
    })
    return project


def test_submission_validation(tmp_path):
    good = "id,target\n1,0.9\n2,0.8\n"
    ok, reason = executors.validate_submission(
        _project_with_submission(tmp_path, good, good),
        "submissions/submission.csv")
    assert ok, reason

    bad = "id,target\n1,0.9\n"  # row count mismatch
    ok, reason = executors.validate_submission(
        _project_with_submission(tmp_path / "b", good, bad),
        "submissions/submission.csv")
    assert not ok and "row count" in reason

    missing_cell = "id,target\n1,\n2,0.8\n"
    ok, reason = executors.validate_submission(
        _project_with_submission(tmp_path / "c", good, missing_cell),
        "submissions/submission.csv")
    assert not ok

    wrong_ids = "id,target\n1,0.9\n3,0.8\n"
    ok, reason = executors.validate_submission(
        _project_with_submission(tmp_path / "d", good, wrong_ids),
        "submissions/submission.csv")
    assert not ok and "differ" in reason


def test_on_task_completed_cv_gate_records_without_submitting(
        tmp_path, monkeypatch):
    good = "id,target\n1,0.9\n2,0.8\n"
    project = _project_with_submission(tmp_path, good, good)
    monkeypatch.setattr(config, "KAGGLE_AUTO_SUBMIT", False)
    orchestrator = SimpleNamespace(project_root=project)
    task = SimpleNamespace(id="EXP-0001")
    result = ObjectiveGateResult(task_id="EXP-0001", passed=True,
                                 primary=0.91, split_hash="split-1")
    asyncio.run(kaggle_pack.PACK.on_task_completed(orchestrator, task, result))
    campaign = kaggle_pack.load_campaign(project)
    assert campaign["best_cv"] == 0.91
    assert campaign["split_key"] == "split-1"
    assert len(campaign["experiments"]) == 1
    # Human gate: nothing was submitted.
    assert campaign["submissions"] == []


def test_improvement_rule():
    assert kaggle_pack.is_improvement(0.9, None, "maximize", 0.0)
    assert kaggle_pack.is_improvement(0.91, 0.9, "maximize", 0.005)
    assert not kaggle_pack.is_improvement(0.901, 0.9, "maximize", 0.005)
    assert kaggle_pack.is_improvement(0.09, 0.1, "minimize", 0.005)
    assert not kaggle_pack.is_improvement(None, 0.1, "minimize", 0.0)


def _campaign_scaffold(tmp_path, name="camp"):
    return kaggle_pack.scaffold(name, "titanic", "auc", "maximize", None,
                                tmp_path)


def test_campaign_director_agent_with_fake_llm(tmp_path):
    from agent_roles import CampaignDirectorAgent

    class FakeLLM:
        async def complete(self, system, user, tools=None, model_role=None):
            return ('[{"title": "XGB baseline", "objective": "train xgb.cv", '
                    '"target_files": ["experiments/xgb.py"], '
                    '"run_command": "python experiments/xgb.py", '
                    '"error_signature": "ERR_KAGGLE_XGB"}]')
        def usage_snapshot(self): return None
        def record_parse(self, ok): pass

    llm = FakeLLM()
    agent = CampaignDirectorAgent(llm, tmp_path)
    hypo = asyncio.run(agent.run({"best_cv": None, "metric": "auc"}, [], []))
    assert hypo is not None
    assert len(hypo) == 1
    assert hypo[0]["title"] == "XGB baseline"


def test_on_task_completed_triggers_director(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CAMPAIGN_DIRECTOR_ENABLED", True)
    project = _campaign_scaffold(tmp_path)

    async def fake_director(self, *args, **kwargs):
        return [{"title": "Next exp", "objective": "try lightgbm",
                 "target_files": ["experiments/lgbm.py"],
                 "run_command": "python experiments/lgbm.py",
                 "error_signature": "ERR_KAGGLE_NEXT"}]

    import agent_roles
    monkeypatch.setattr(agent_roles.CampaignDirectorAgent, "run",
                        fake_director)

    # Clear the queue so the director fires (starter hypotheses consumed).
    campaign = kaggle_pack.load_campaign(project)
    campaign["pending_hypotheses"] = []
    kaggle_pack.save_campaign(project, campaign)

    result = ObjectiveGateResult(task_id="EXP-0001", passed=True,
                                 primary=0.91, split_hash="s1")
    task = SimpleNamespace(id="EXP-0001")
    orc = SimpleNamespace(project_root=project, llm=object(),
                          workspace_root=tmp_path)
    asyncio.run(kaggle_pack.PACK.on_task_completed(orc, task, result))
    campaign = kaggle_pack.load_campaign(project)
    assert len(campaign.get("pending_hypotheses", [])) == 1
    assert campaign["pending_hypotheses"][0]["title"] == "Next exp"
