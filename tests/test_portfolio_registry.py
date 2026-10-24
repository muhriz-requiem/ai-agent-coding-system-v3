"""portfolio registry (Phase 8b, read side) -- the parent's one view.

Pins for the admission rule and the derived escalation inbox:

* a folder is a subsidiary only with BOTH board and goal.txt (kills the
  board-without-goal drift class that made every cycle emit
  task_board_unreadable noise around research-only folders);
* DIAGNOSIS-prefixed last_feedback lines are the escalation inbox until the
  orchestrator-side write hook merges;
* defer_until in the future parks a task -- and a malformed stamp must not
  crash the registry nor silently park forever;
* the snapshot is derived data only; nothing writes into any project.
"""

import json
from datetime import datetime, timedelta, timezone

import portfolio


def _board(project_dir, tasks):
    agent = project_dir / ".agent"
    agent.mkdir(parents=True, exist_ok=True)
    (agent / "task_board.json").write_text(
        json.dumps({"tasks": tasks}), encoding="utf-8")


def _task(id, status="done", feedback=None, defer_until=None,
          retry_count=0):
    return {"id": id, "status": status,
            "last_feedback": feedback,
            "defer_until": defer_until,
            "retry_count": retry_count}


def test_admission_requires_board_and_goal(tmp_path):
    pdir = tmp_path / "projects" / "good"
    _board(pdir, [_task("T-1")])
    (pdir / "goal.txt").write_text("g\n", encoding="utf-8")
    row = portfolio.classify(pdir)
    assert row["kind"] == portfolio.KIND_SUBSIDIARY
    assert row["done"] == 1 and row["tasks"] == 1


def test_board_without_goal_is_other(tmp_path):
    pdir = tmp_path / "projects" / "halfbuilt"
    _board(pdir, [_task("T-1")])
    row = portfolio.classify(pdir)
    assert row["kind"] == portfolio.KIND_OTHER
    assert "goal.txt" in row["note"]


def test_boardless_folder_is_research(tmp_path):
    pdir = tmp_path / "projects" / "study"
    pdir.mkdir(parents=True)
    (pdir / "SPEC.md").write_text("notes\n", encoding="utf-8")
    row = portfolio.classify(pdir)
    assert row["kind"] == portfolio.KIND_RESEARCH


def test_diagnosis_feedback_becomes_escalation_inbox(tmp_path):
    pdir = tmp_path / "projects" / "p"
    _board(pdir, [_task("T-7", feedback="DIAGNOSIS: spec_ambiguous -> "
                                      "narrow_objective\nmore detail")])
    row = portfolio.classify(pdir)
    assert row["escalations"] == [
        {"task": "T-7",
         "diagnosis": "spec_ambiguous -> narrow_objective"}]


def test_failed_without_diagnosis_is_still_escalation(tmp_path):
    pdir = tmp_path / "projects" / "p"
    _board(pdir, [_task("T-9", status="failed")])
    row = portfolio.classify(pdir)
    assert row["failed"] == 1
    assert "(failed status" in row["escalations"][0]["diagnosis"]


def test_future_defer_until_parks_never_done_parks(tmp_path):
    future = (datetime.now(timezone.utc)
              + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    pdir = tmp_path / "projects" / "p"
    _board(pdir, [
        _task("T-1", status="pending", defer_until=future),
        _task("T-2", status="done", defer_until=future),
    ])
    row = portfolio.classify(pdir)
    assert [p["task"] for p in row["parked"]] == ["T-1"]
    assert not portfolio.is_drained(row)


def test_past_defer_until_is_not_parking(tmp_path):
    past = "2020-01-01T00:00:00Z"
    pdir = tmp_path / "projects" / "p"
    _board(pdir, [_task("T-1", status="pending", defer_until=past)])
    row = portfolio.classify(pdir)
    assert row["parked"] == []


def test_malformed_defer_until_is_reported_not_fatal(tmp_path):
    pdir = tmp_path / "projects" / "p"
    _board(pdir, [_task("T-1", status="pending",
                        defer_until="not-a-clock")])
    (pdir / "goal.txt").write_text("g\n", encoding="utf-8")
    row = portfolio.classify(pdir)
    assert row["parked"] == []            # never silently parked forever
    assert row["kind"] == portfolio.KIND_SUBSIDIARY


def test_retries_sum_report(tmp_path):
    pdir = tmp_path / "projects" / "p"
    _board(pdir, [_task("T-1", retry_count=2), _task("T-2", retry_count=4)])
    (pdir / "goal.txt").write_text("g\n", encoding="utf-8")
    row = portfolio.classify(pdir)
    assert row["retries"] == 6


def test_report_counts_and_snapshot(tmp_path):
    good = tmp_path / "projects" / "good"
    _board(good, [_task("T-1"), _task("T-2")])
    (good / "goal.txt").write_text("g\n", encoding="utf-8")
    half = tmp_path / "projects" / "halfbuilt"
    _board(half, [_task("T-9")])
    (tmp_path / "projects" / "study").mkdir(parents=True)
    report = portfolio.build_report(tmp_path)
    assert report["counts"]["subsidiaries"] == 1
    assert report["counts"]["drained"] == 1
    assert report["counts"]["other"] == 1
    assert report["counts"]["research"] == 1
    snapshot = portfolio.write_snapshot(tmp_path, report)
    loaded = json.loads(snapshot.read_text("utf-8"))
    assert loaded["schema"] == portfolio.SNAPSHOT_SCHEMA
    assert loaded["counts"] == report["counts"]
    assert not list(snapshot.parent.glob("portfolio.json.tmp"))


def test_render_text_leads_with_inbox_and_marks_drained(tmp_path):
    good = tmp_path / "projects" / "good"
    _board(good, [
        _task("T-1"),
        _task("T-2", status="pending",
              feedback="DIAGNOSIS: contradicted pin"),
    ])
    (good / "goal.txt").write_text("g\n", encoding="utf-8")
    report = portfolio.build_report(tmp_path)
    lines = portfolio.render_text(report)
    assert any("escalation inbox:" in line for line in lines)
    assert any("good/T-2: contradicted pin" in line for line in lines)
    assert any("good:" in line and "pending=1" in line for line in lines)
    assert not any("[DRAINED]" in line for line in lines)
