# -*- coding: utf-8 -*-
"""workspace MCP (Phase 8f): read-only structured company state for hosts.

A local stdio MCP server exposing exactly one thing to any host session:
the workspace's own derived state (roster, boards, log tail, steering
counters, export staging, cache statistics), in structured form, with
bounded reads.  Design invariants:

* READ-ONLY, structurally.  No tool mutates anything; there is no write
  surface.  Mutations stay on the CLI path where the single-orchestrator
  lock and the export gate enforce what "autonomy" means.
* The pure tool layer :func:`call_tool` is transport-independent so it is
  unit-testable; :func:`serve` is the newline-delimited JSON-RPC 2.0
  stdio loop (MCP's classic handshake) around it.
* Gate: ``config.MCP_WORKSPACE_ENABLED`` (default False) decides whether
  the server may start at all; enabling is an explicit config choice, not
  a prompt-time one (research-lane precedent).
* Bounded reads: line counts and widths are capped; secret-shaped files
  (env, cache payloads) are never read; failures surface as structured
  ``{"ok": false, "error": ...}`` payloads, never exceptions across the
  transport.

Registered against the opencode harness as a project-local server:
``opencode mcp add workspace -- python <abs>\\workspace_mcp.py``.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROTOCOL_VERSION = "2025-11-25"
LINE_CAP = 400
LOG_TAIL_MAX = 200
FEEDBACK_PREVIEW = 160
PROJECT_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _project_ok(project: str) -> bool:
    return (bool(PROJECT_RE.match(project)) and project[0].isalnum()
            and project not in (".", ".."))

_TOOL_NAMES = ("company_roster", "project_board", "log_tail",
               "steering_state", "export_staging", "cache_stats")


# ---------------------------------------------------------------------------
# Tool layer (pure, transport-independent)
# ---------------------------------------------------------------------------

def _ok(data: Any) -> Dict[str, Any]:
    return {"ok": True, "data": data}


def _bad(error: str) -> Dict[str, Any]:
    return {"ok": False, "error": error}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clip(text: str, cap: int = LINE_CAP) -> str:
    """Never emit more than *cap* characters, suffix included."""
    if len(text) <= cap:
        return text
    return text[:cap - 11] + "[truncated]"


def _check_project(project: Any) -> Optional[str]:
    """Structured refusal reason for an unsafe project name."""
    if not isinstance(project, str) or not _project_ok(project):
        return f"unsafe project name: {project!r}"
    return None


def list_tools() -> List[Dict[str, Any]]:
    """Static tool catalog; each description doubles as documentation."""
    schema_project = {"type": "object",
                      "properties": {"project": {"type": "string"}},
                      "required": ["project"]}
    schema_n = {"type": "object",
                "properties": {"n": {"type": "integer",
                                     "default": 40, "minimum": 1,
                                     "maximum": LOG_TAIL_MAX}}}
    return [
        {"name": "company_roster",
         "description": "portfolio registry: subsidiaries/other/research "
                        "kinds, drained flags, escalation inbox",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "project_board",
         "description": "one project's task rows with status, retries, "
                        "error_signature, defer_until, feedback preview",
         "inputSchema": schema_project},
        {"name": "log_tail",
         "description": f"last N (default 40, max {LOG_TAIL_MAX}) lines of "
                        f"logs/agent_log.jsonl, each clipped to {LINE_CAP}",
         "inputSchema": schema_n},
        {"name": "steering_state",
         "description": "doom-loop counters: top failure-signature tokens "
                        "and repeated-diff counts",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "export_staging",
         "description": "export-gate staging state: manifests awaiting "
                        "operator approval",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "cache_stats",
         "description": "LLM cache statistics (entries, bytes) -- counts "
                        "only, entry contents are never exposed",
         "inputSchema": {"type": "object", "properties": {}}},
    ]


def call_tool(name: str, arguments: Dict[str, Any],
              workspace: Path) -> Dict[str, Any]:
    """One tool call against *workspace*, structured result, never raises.
    The tool layer runs regardless of the start gate: it is the server
    start that is a config choice, these functions are library code."""
    workspace = Path(workspace)
    try:
        return _dispatch(name, dict(arguments or {}), workspace)
    except Exception as exc:  # noqa: BLE001 - the transport must not tear
        return _bad(f"internal: {type(exc).__name__}: {exc}")


def _dispatch(name: str, args: Dict[str, Any],
              workspace: Path) -> Dict[str, Any]:
    if name == "company_roster":
        import portfolio
        report = portfolio.build_report(workspace)
        return _ok({"counts": report["counts"],
                    "projects": report["projects"]})
    if name == "project_board":
        reason = _check_project(args.get("project"))
        if reason:
            return _bad(reason)
        return _ok(_board(workspace, str(args["project"])))
    if name == "log_tail":
        return _ok(_log_tail(workspace, _bounded_n(args.get("n"))))
    if name == "steering_state":
        return _ok(_steering(workspace))
    if name == "export_staging":
        import export_gate
        return _ok({"staging": export_gate.list_staging(workspace)})
    if name == "cache_stats":
        return _ok(_cache_stats(workspace))
    return _bad(f"unknown tool: {name}")


def _board(workspace: Path, project: str) -> Dict[str, Any]:
    import portfolio
    pdir = workspace / "projects" / project
    if not pdir.is_dir():
        raise ValueError(f"no such project: {project}")
    row = portfolio.classify(pdir)
    board, board_error = _load_json_safe(
        pdir / ".agent" / "task_board.json")
    tasks: List[Dict[str, Any]] = []
    if isinstance(board, dict):
        for task in (board.get("tasks") or []):
            if not isinstance(task, dict):
                continue
            feedback = task.get("last_feedback")
            if not isinstance(feedback, str):
                feedback = ""
            tasks.append({
                "id": str(task.get("id")),
                "status": str(task.get("status")),
                "retry_count": task.get("retry_count"),
                "error_signature": task.get("error_signature"),
                "defer_until": task.get("defer_until"),
                "feedback": feedback.replace("\n", " ") + ""
                if len(feedback.replace("\n", " ")) <= FEEDBACK_PREVIEW
                else feedback.replace("\n", " ")[:FEEDBACK_PREVIEW] + "...",
            })
    return {"project": project, "kind": row["kind"],
            "goal": row["goal"], "board_error": board_error,
            "tasks": tasks}


def _load_json_safe(path: Path):
    try:
        return _load_json(path), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"unreadable ({exc})"


def _bounded_n(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 40
    return max(1, min(LOG_TAIL_MAX, n))


def _log_tail(workspace: Path, n: int) -> Dict[str, Any]:
    path = workspace / "logs" / "agent_log.jsonl"
    if not path.is_file():
        raise ValueError("no agent log at logs/agent_log.jsonl")
    lines: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line:
                lines.append(_clip(line))
    return {"lines": lines[-n:], "total": len(lines),
            "note": f"tail of logs/agent_log.jsonl, lines clipped to "
                    f"{LINE_CAP} chars"}


def _steering(workspace: Path) -> Dict[str, Any]:
    path = workspace / ".system" / "steering_state.json"
    state = _load_json(path)
    signatures = state.get("signature_runs") or {}
    diffs = state.get("diff_runs") or {}
    top = sorted(((k, int(v)) for k, v in signatures.items()),
                 key=lambda kv: kv[1], reverse=True)[:5]
    return {"top_signatures": [list(kv) for kv in top],
            "signature_tokens": len(signatures),
            "repeated_diffs": len(diffs)}


def _cache_stats(workspace: Path) -> Dict[str, Any]:
    root = workspace / ".system" / "llm_cache"
    entries = 0
    total_bytes = 0
    if root.is_dir():
        for path in root.rglob("*.json"):
            entries += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    return {"entries": entries, "total_bytes": total_bytes,
            "note": "statistics only; cache contents are never exposed"}


# ---------------------------------------------------------------------------
# JSON-RPC layer
# ---------------------------------------------------------------------------

def handle_request(msg: Dict[str, Any],
                   workspace: Path) -> Optional[Dict[str, Any]]:
    """One JSON-RPC message in, response dict out (None for notifications)."""
    method = str(msg.get("method") or "")
    msg_id = msg.get("id")
    params = msg.get("params") or {}
    try:
        result = _result_for(method, params, workspace)
    except KeyError as exc:
        return _rpc_error(msg_id, -32601, f"unknown method: {exc}")
    except ValueError as exc:
        return _rpc_error(msg_id, -32602, str(exc))
    if result is None:
        return None  # notification: no response
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _result_for(method: str, params: Dict[str, Any],
                workspace: Path) -> Optional[Dict[str, Any]]:
    if method == "initialize":
        requested = str(params.get("protocolVersion") or "")
        return {
            "protocolVersion": (requested
                                if requested in ("2025-11-25", "2025-06-18",
                                                 "2025-03-26", "2024-11-05")
                                else PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "workspace", "version": "8f"},
        }
    if method == "notifications/initialized" or method.startswith("notif"):
        return None
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": list_tools()}
    if method == "tools/call":
        name = str(params.get("name") or "")
        outcome = call_tool(name, params.get("arguments") or {}, workspace)
        text = json.dumps(outcome, ensure_ascii=False)
        return {"content": [{"type": "text", "text": text}],
                "isError": not outcome.get("ok", False)}
    raise KeyError(method)


def _rpc_error(msg_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------

def gate_reason(enabled: Optional[bool] = None) -> Optional[str]:
    """Reason the server must not start now, or None when it may."""
    if enabled is None or enabled is True:
        return None
    return ("MCP_WORKSPACE_ENABLED is false; the workspace server will not "
            "start (set MCP_WORKSPACE_ENABLED=true in .env to enable -- "
            "explicit config choice, never a prompt-time one)")


def serve(workspace: Optional[Path] = None,
          enabled: Optional[bool] = None,
          lines: Optional[Iterable[str]] = None,
          out: Optional[Any] = None) -> int:
    """Newline-delimited JSON-RPC 2.0 loop until stdin EOF.  Returns an
    exit code: 0 after a clean shutdown, 1 when the gate refused."""
    reason = gate_reason(enabled)
    if reason:
        message = f"workspace-mcp: {reason}"
        if out is not None:
            out(message)
        else:
            print(message, file=sys.stderr)
        return 1
    workspace = Path(workspace) if workspace else Path.cwd()
    emit = out if out is not None else (lambda text: print(text,
                                                           flush=True))
    stream = lines if lines is not None \
        else (line for line in sys.stdin)
    for raw in stream:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            emit(json.dumps(_rpc_error(None, -32768,
                                       f"parse error: {exc}")))
            continue
        response = handle_request(msg, workspace)
        if response is not None:
            emit(json.dumps(response, ensure_ascii=False))
    return 0


def main(workspace: Optional[Path] = None) -> int:
    """Process entrypoint: workspace from argv or cwd, gate honored."""
    if workspace is None:
        workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    try:
        import config
        enabled = config.MCP_WORKSPACE_ENABLED
    except Exception:  # config absence must not crash the host transport
        enabled = False
    return serve(workspace=workspace, enabled=enabled)


if __name__ == "__main__":
    raise SystemExit(main())

