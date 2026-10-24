"""workspace MCP (Phase 8f) -- read-only structured state for host sessions.

Contract pins for the server the opencode harness registers as a local
stdio MCP server:

* gate semantics: a disabled server refuses to start with an actionable
  message (research-lane precedent: enabling is an explicit config choice,
  config.MCP_WORKSPACE_ENABLED default False);
* pure tool layer runs regardless of the gate so unit tests can pin every
  tool's structured return without the transport;
* path safety: project-name validation refuses separators, traversal, and
  empty names -- a structured refusal, never an exception;
* all reads bounded: log tail caps count and per-line width; stale or
  unreadable state files produce structured errors, not crashes;
* cache stats expose counts only, never entry contents.
"""

import json

import pytest

import workspace_mcp


# ---------------------------------------------------------------------------
# Gate: disabled means refuse to serve, with how to enable
# ---------------------------------------------------------------------------

def test_serve_refuses_when_gate_off(ws_with_project):
    lines, out = [], []
    code = workspace_mcp.serve(ws_with_project, enabled=False,
                               lines=iter(lines), out=out.append)
    assert code == 1
    assert any("MCP_WORKSPACE_ENABLED" in line for line in out)


def test_initialize_handshake_echoes_legacy_protocol(ws_with_project):
    lines = [json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-11-25"}}),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
             json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
             ]
    _, out = _serve_collect(ws_with_project, lines)
    init, listed = json.loads(out[0]), json.loads(out[1])
    assert init["result"]["protocolVersion"] == "2025-11-25"
    assert init["result"]["capabilities"] == {"tools": {}}
    assert init["result"]["serverInfo"]["name"] == "workspace"
    names = {t["name"] for t in listed["result"]["tools"]}
    assert {"company_roster", "project_board", "log_tail", "steering_state",
            "export_staging", "cache_stats"} <= names


def _serve_collect(workspace, lines, enabled=True):
    out: list = []
    workspace_mcp.serve(workspace, enabled=enabled, lines=iter(lines),
                        out=out.append)
    return out, out


# ---------------------------------------------------------------------------
# Tool layer (pure functions, structured returns)
# ---------------------------------------------------------------------------

def _make_sub(workspace, name, board_tasks, goal=True):
    pdir = workspace / "projects" / name
    pdir.mkdir(parents=True)
    agent = pdir / ".agent"
    agent.mkdir()
    (agent / "task_board.json").write_text(
        json.dumps({"tasks": board_tasks}), encoding="utf-8")
    if goal:
        (pdir / "goal.txt").write_text("build a thing\n", encoding="utf-8")


def _task(id, status="done", retries=0, sig=None):
    return {"id": id, "status": status, "retry_count": retries,
            "error_signature": sig}


def test_company_roster_counts_and_inbox(ws_with_project):
    _make_sub(ws_with_project, "alpha",
              [_task("T-1"), _task("T-2", status="pending", retries=2,
                                   sig="ERR_X")])
    data = workspace_mcp.call_tool("company_roster", {},
                                   workspace=ws_with_project)
    assert data["ok"]
    counts = data["data"]["counts"]
    assert counts["subsidiaries"] == 1
    assert counts["escalations"] == 0
    rows = {r["name"]: r for r in data["data"]["projects"]}
    assert rows["alpha"]["pending"] == 1


def test_project_board_returns_task_rows(ws_with_project):
    _make_sub(ws_with_project, "alpha",
              [_task("T-9", status="failed", retries=3, sig="ERR_TWICE")])
    data = workspace_mcp.call_tool("project_board", {"project": "alpha"},
                                   workspace=ws_with_project)
    assert data["ok"]
    board = data["data"]
    assert board["tasks"][0]["id"] == "T-9"
    assert board["tasks"][0]["status"] == "failed"
    assert board["tasks"][0]["error_signature"] == "ERR_TWICE"


def test_project_board_truncates_feedback(ws_with_project):
    _make_sub(ws_with_project, "alpha", [
        {"id": "T-1", "status": "pending", "retry_count": 0,
         "error_signature": None,
         "last_feedback": f"DIAGNOSIS: {'y' * 500}"}])
    data = workspace_mcp.call_tool("project_board", {"project": "alpha"},
                                   workspace=ws_with_project)
    preview = data["data"]["tasks"][0]["feedback"]
    assert len(preview) <= workspace_mcp.FEEDBACK_PREVIEW + 3
    assert preview.endswith("...")


def test_project_name_path_safety(ws_with_project):
    leaves = ["../boardscan", "a/b", "", "a\\b", ".x", "..", "x;drop"]
    for bad in leaves:
        data = workspace_mcp.call_tool("project_board", {"project": bad},
                                       workspace=ws_with_project)
        assert data["ok"] is False, bad
        assert "unsafe" in data["error"]


def test_unknown_project_is_structured_error(ws_with_project):
    data = workspace_mcp.call_tool("project_board", {"project": "ghost"},
                                   workspace=ws_with_project)
    assert data["ok"] is False
    assert "no such project" in data["error"]


def test_unknown_tool_refused(ws_with_project):
    data = workspace_mcp.call_tool("rm_rf_slash", {}, workspace=ws_with_project)
    assert data["ok"] is False
    assert "unknown tool" in data["error"]


def test_log_tail_bounded(ws_with_project):
    logdir = ws_with_project / "logs"
    logdir.mkdir(parents=True)
    rows = [json.dumps({"ts": f"t{i}", "level": "INFO", "event": "e",
                        "project": "qa", "msg": "z" * 600})
            for i in range(50)]
    (logdir / "agent_log.jsonl").write_text("\n".join(rows) + "\n",
                                            encoding="utf-8")
    data = workspace_mcp.call_tool("log_tail", {"n": 3},
                                   workspace=ws_with_project)
    assert data["ok"]
    assert len(data["data"]["lines"]) == 3
    line = data["data"]["lines"][-1]
    assert len(line) <= workspace_mcp.LINE_CAP + 1
    assert line.endswith("[truncated]")


def test_log_tail_missing_file_is_structured(ws_with_project):
    data = workspace_mcp.call_tool("log_tail", {"n": 5},
                                   workspace=ws_with_project)
    assert data["ok"] is False
    assert "no agent log" in data["error"]


def test_steering_state_summary(ws_with_project):
    sysdir = ws_with_project / ".system"
    sysdir.mkdir(parents=True)
    (sysdir / "steering_state.json").write_text(json.dumps({
        "signature_runs": {"TWICE": 3, "OTHER": 1},
        "diff_runs": {"task|hash": 2},
    }), encoding="utf-8")
    data = workspace_mcp.call_tool("steering_state", {},
                                   workspace=ws_with_project)
    assert data["ok"]
    assert data["data"]["top_signatures"] == [["TWICE", 3], ["OTHER", 1]]


def test_cache_stats_counts_only(ws_with_project):
    cache = ws_with_project / ".system" / "llm_cache" / "ab"
    cache.mkdir(parents=True)
    (cache / f"{'a' * 64}.json").write_text(json.dumps({"response": "x"}),
                                            encoding="utf-8")
    data = workspace_mcp.call_tool("cache_stats", {}, workspace=ws_with_project)
    assert data["ok"]
    assert data["data"]["entries"] == 1
    assert "response" not in json.dumps(data["data"])


def test_export_staging_lists(ws_with_project):
    (ws_with_project / ".system" / "export_staging"
     / "alpha-20260101T000000Z-beef01").mkdir(parents=True)
    (ws_with_project / ".system" / "export_staging"
     / "alpha-20260101T000000Z-beef01" / "manifest.json").write_text(
        json.dumps({"project": "alpha", "approval": None,
                    "tests": {"ok": True}, "artifacts": [{}],
                    "destination": {"url": ""}}), encoding="utf-8")
    data = workspace_mcp.call_tool("export_staging", {},
                                   workspace=ws_with_project)
    assert data["ok"]
    assert any("alpha-20260101T000000Z-beef01" in line
               for line in data["data"]["staging"])


# ---------------------------------------------------------------------------
# JSON-RPC layer behaviors
# ---------------------------------------------------------------------------

def test_jsonrpc_call_tool_pipes_to_result(ws_with_project):
    req = {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
           "params": {"name": "company_roster", "arguments": {}}}
    resp = workspace_mcp.handle_request(req, ws_with_project)
    assert resp["id"] == 7
    assert resp["result"]["isError"] is False


def test_jsonrpc_unknown_method(ws_with_project):
    resp = workspace_mcp.handle_request(
        {"jsonrpc": "2.0", "id": 8, "method": "bogus"}, ws_with_project)
    assert resp["error"]["code"] == -32601


def test_jsonrpc_init_missing_params_defaults(ws_with_project):
    resp = workspace_mcp.handle_request(
        {"jsonrpc": "2.0", "id": 9, "method": "initialize"}, ws_with_project)
    assert resp["result"]["protocolVersion"] == "2025-11-25"


@pytest.fixture
def ws_with_project(tmp_path):
    ws = tmp_path
    (ws / "projects").mkdir()
    return ws
