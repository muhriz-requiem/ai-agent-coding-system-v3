"""Entry point for the Enterprise Multi-Agent Coding System.

Usage:
    python main.py          # run the full orchestration loop (Ctrl+C to stop)
    python main.py --once   # single orchestration pass, then exit (testing)
    python main.py --help   # CLI help (the only place raw prints are allowed)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import config
import providers
from logger import log_event
from super_orchestrator import SuperOrchestrator

try:  # OS-level locks: the OS releases them automatically on process death.
    import msvcrt
except ImportError:  # pragma: no cover -- non-Windows
    msvcrt = None
try:
    import fcntl
except ImportError:  # pragma: no cover -- Windows
    fcntl = None


@contextmanager
def _orchestrator_lock(workspace: Path) -> Iterator[None]:
    """Refuse to start a second orchestrator loop on the same workspace.

    Two orchestrator suites processing one workspace race on the task
    boards, snapshots and manifest -- the losing loop's rollback then
    resurrects the winning loop's files (observed in production as a
    rolled-back module that kept coming back). The lock is advisory but
    OS-owned: it is released automatically when the holder dies, so no
    stale-lock cleanup is needed.
    """
    lock_path = workspace / ".system" / "orchestrator.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+")
    locked = False
    try:
        # msvcrt locks the byte range at the CURRENT file position; "a+"
        # opens at EOF, which is non-zero once a previous holder wrote its
        # pid.  A second instance would then lock a different byte and
        # succeed.  Always lock the fixed range starting at byte 0.
        handle.seek(0)
        if msvcrt is not None:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                locked = True
            except OSError:
                locked = False
        elif fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except OSError:
                locked = False
        else:  # pragma: no cover -- no lock API available
            locked = True
        if not locked:
            print(
                "error: another orchestrator instance is already running on "
                f"this workspace (lock: {lock_path}). Stop it first, e.g. "
                "via its dashboard console or Task Manager.",
                file=sys.stderr)
            raise SystemExit(2)
        handle.seek(0)
        handle.write(f"{os.getpid()}\n")
        handle.truncate()
        handle.flush()
        log_event("INFO", "orchestrator_lock_acquired",
                  {"pid": os.getpid()})
        yield
    finally:
        if locked:
            try:
                if msvcrt is not None:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                elif fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        handle.close()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=f"Enterprise Multi-Agent Coding System {providers.VERSION} "
                    "(3-tier hierarchical orchestration, free multi-provider "
                    "routing, offline DEMO_MODE).")
    parser.add_argument(
        "--once", action="store_true",
        help="run a single orchestration pass (process all currently pending "
             "tasks once) and then exit -- useful for testing and CI")
    parser.add_argument(
        "--new-project", metavar="NAME",
        help="scaffold projects/NAME (task board, src/, tests/) and exit -- "
             "then edit .agent/task_board.json to describe your work, or pair "
             "it with --goal to have the system author the board itself")
    parser.add_argument(
        "--goal", metavar="TEXT",
        help="the project's one-paragraph goal, written to projects/NAME/"
             "goal.txt. With --new-project this becomes the board's origin: "
             "the Spec Author drafts the tests-first breakdown on first run. "
             "Without --new-project it is recorded for an existing project.")
    parser.add_argument(
        "--author-plan", action="store_true",
        help="with --new-project and --goal: author the breakdown now and "
             "print it, instead of waiting for the first orchestration run. "
             "Does not start the orchestrator")
    parser.add_argument(
        "--feed", action="store_true",
        help="open the watch-only live event feed window (Ctrl+C closes it)")
    parser.add_argument(
        "--cmd", action="store_true",
        help="open the dedicated command console (pause/resume/status/skip/"
             "retry/note/ask/exit)")
    parser.add_argument(
        "--dashboard", action="store_true",
        help="run the orchestrator headless and spawn BOTH dashboard windows "
             "(event feed + command console) as separate terminals")
    parser.add_argument(
        "--models", metavar="ACTION",
        choices=["status", "verify", "refresh", "probe", "cache-clear"],
        help="free-model capacity layer: status (rankings + quota), verify "
             "(one live call per provider), refresh (OpenRouter catalog), "
             "probe (canary battery), cache-clear")
    parser.add_argument(
        "--new-kaggle", metavar="NAME",
        help="scaffold projects/NAME as a Kaggle campaign (requires "
             "--competition; see also --metric/--metric-direction)")
    parser.add_argument(
        "--competition", metavar="SLUG", default="",
        help="Kaggle competition slug for --new-kaggle (e.g. titanic)")
    parser.add_argument(
        "--metric", metavar="NAME", default="accuracy",
        help="primary competition metric for --new-kaggle (default: accuracy)")
    parser.add_argument(
        "--metric-direction", choices=["maximize", "minimize"],
        default="maximize", help="metric direction for --new-kaggle")
    parser.add_argument(
        "--metric-target", type=float, default=None,
        help="optional metric target for --new-kaggle")
    parser.add_argument(
        "--campaign-status", metavar="PROJECT",
        help="print the Kaggle campaign state for a scaffolded project")
    parser.add_argument(
        "--preflight", metavar="ACTION",
        choices=["scan", "paths", "scrub"],
        help="secret hygiene: scan (walk the workspace for credential-shaped "
             "content before you commit or push), paths (list every file "
             "redaction considers a live secret), scrub (dry-run remediation of "
             "shared memory; add --apply to write)")
    parser.add_argument(
        "--apply", action="store_true",
        help="commit the changes --preflight scrub would make; without it "
             "scrub only reports. Requires --preflight scrub")
    parser.add_argument(
        "--committed", action="store_true",
        help="with --preflight scan: check only the files git tracks, i.e. "
             "what a commit could actually pick up. The whole-tree scan is "
             "always red (a live .env, and the secret-scanner test fixtures, "
             "match by design); this answers the question you actually have "
             "before a push. Exits 2 rather than scanning everything when git "
             "cannot tell us what is tracked.")
    parser.add_argument(
        "--export", metavar="ACTION",
        choices=["scan", "approve", "verify", "list"],
        help="export gate (Phase 8a): scan (run the project suite fresh and "
             "stage a signed-off-intent manifest with checksums), approve "
             "(record operator approval after every gate check holds), verify "
             "(re-hash + re-check the staged copy), list. The gate never "
             "pushes: the operator performs the physical export by hand")
    parser.add_argument(
        "--project", metavar="NAME",
        help="with --export: project under projects/ to gate")
    parser.add_argument(
        "--dest", metavar="URL", default="",
        help="with --export scan/approve: destination being offered; must be "
             "on the EXPORT_DESTINATION_ALLOWLIST and never on the blacklist")
    parser.add_argument(
        "--yes", action="store_true",
        help="with --export approve: skip the interactive confirm")
    parser.add_argument(
        "--allow-injection-hits", action="store_true",
        help="with --export approve: operator has inspected the injection "
             "scan hits and accepts them; choice is recorded in the manifest")
    parser.add_argument(
        "--portfolio", action="store_true",
        help="company registry (Phase 8b, read-only): classify every folder "
             "under projects/, summarize drained/pending/failed state, "
             "derive the escalation inbox from DIAGNOSIS feedback, and "
             "write a derived snapshot to .system/portfolio.json")
    parser.add_argument(
        "--mcp-serve", action="store_true",
        help="workspace MCP server (Phase 8f, read-only): serve roster, "
             "boards, log tail, steering counters, export staging, and "
             "cache stats over the MCP stdio transport for host sessions. "
             "Requires MCP_WORKSPACE_ENABLED=true; there is no write "
             "surface -- mutations stay on the CLI path behind the "
             "orchestrator lock")
    return parser.parse_args(argv)


def _report_scan(findings: List[Dict[str, Any]], scope: str) -> int:
    """Print a scan result and return the process exit code.

    Values are never printed, only rule names and sha256 hints, so this is safe
    to run on a shared screen.  The findings are typed loosely because
    ``redaction`` is imported locally (see :func:`_preflight_command`) and a
    module-level annotation would need the import moved to startup.
    """
    if not findings:
        print(f"preflight scan: clean, no credential-shaped content in {scope}")
        return 0
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for finding in findings:
        by_file.setdefault(str(finding.get("path", "?")), []).append(finding)
    print(f"preflight scan: {len(findings)} finding(s) in "
          f"{len(by_file)} file(s) [{scope}]")
    for path in sorted(by_file):
        rows = by_file[path]
        rules = sorted({str(r.get("rule")) for r in rows})
        hints = sorted({str(r.get("hint")) for r in rows})
        print(f"  {path}")
        print(f"      rules={','.join(rules)} hints={','.join(hints[:4])}")
    print("\nValues are never printed. Confirm every hit is a false "
          "positive, or remove the secret and re-run.")
    return 1


def _preflight_command(action: str, workspace: Path, apply: bool = False,
                       committed: bool = False) -> int:
    """Handle ``--preflight`` actions without taking the orchestrator lock."""
    import redaction

    if not config.REDACTION_ENABLED and not config.SECRET_SCAN_ENABLED:
        print("secret hygiene is disabled "
              "(REDACTION_ENABLED=false, SECRET_SCAN_ENABLED=false)")
        return 0

    if action == "paths":
        values = redaction.known_secret_values(workspace, refresh=True)
        names = sorted(redaction._env_candidates(workspace))
        print(f"{len(values)} live secret values discovered from "
              f"{len(names)} credential-shaped variable name(s):")
        for name in names:
            print(f"  {name}")
        print("values are never printed; use --preflight scrub to remove them "
              "from files")
        return 0

    if action == "scan":
        if committed:
            # v2.9.1: the question that actually matters before a push is "could
            # a secret reach a commit", not "does one exist anywhere here".  The
            # full-tree scan can never pass -- a live .env and the
            # secret-scanner test fixtures both match by design -- and a check
            # that is always red is a check nobody reads.
            paths, note = redaction.tracked_files(workspace)
            if not paths:
                # Never degrade silently: a fallback to "scan everything" would
                # report on a scope the caller did not ask for.
                print(f"preflight scan --committed: {note}", file=sys.stderr)
                print("Nothing was scanned. Initialise git and commit, or "
                      "drop --committed to scan the whole tree.",
                      file=sys.stderr)
                return 2
            allowed = redaction.scan_allowlist(workspace)
            findings = redaction.scan_paths(paths)
            suppressed: List[str] = []
            if allowed:
                keep: List[Dict[str, Any]] = []
                for finding in findings:
                    hit = str(finding.get("path", ""))
                    try:
                        rel = str(Path(hit).resolve().relative_to(
                            Path(workspace).resolve()).as_posix())
                    except (ValueError, OSError):
                        rel = hit
                    if redaction.is_scan_allowed(rel, allowed):
                        if rel not in suppressed:
                            suppressed.append(rel)
                    else:
                        keep.append(finding)
                findings = keep
            code = _report_scan(findings, scope=note)
            if suppressed:
                # Never silent about an exemption: the count and the paths are
                # part of the answer.
                print(f"\n{len(suppressed)} file(s) exempt under "
                      f".secret-scan-allow ({len(allowed)} pattern(s)): "
                      + ", ".join(suppressed[:8]))
            return code
        return _report_scan(redaction.scan_tree(workspace),
                            scope="whole tree")

    # action == "scrub"
    targets = [t.strip() for t in
               str(config.SECRET_SCRUB_TARGETS or "").split(",")
               if t.strip()]
    if not targets:
        print("no SECRET_SCRUB_TARGETS configured", file=sys.stderr)
        return 2

    total = 0
    touched: List[str] = []
    for rel in targets:
        path = Path(workspace) / rel
        if not path.exists():
            print(f"  {rel}: not present, skipped")
            continue
        if apply:
            count, backup, findings = redaction.apply_scrub_json_file(path)
            if backup is not None:
                rules = sorted({str(f.get("rule")) for f in findings})
                print(f"  {rel}: {count} replacement(s) "
                      f"[{','.join(rules)}] backup={backup.name}")
        else:
            count, findings = redaction.scrub_json_file(path)
            if count:
                rules = sorted({str(f.get("rule")) for f in findings})
                print(f"  {rel}: {count} would be replaced "
                      f"[{','.join(rules)}]  (dry run -- nothing written)")
        total += count
        if count:
            touched.append(rel)

    if not apply:
        print(f"\nDRY RUN: {total} replacement(s) available across "
              f"{len(touched)} file(s). Re-run with --apply to write "
              f"(a timestamped .bak- backup is taken first).")
        return 1 if total else 0

    log_event("INFO", "secrets_scrub_completed",
              {"files": len(touched), "replacements": total})
    print(f"\napplied: {total} replacement(s) across {len(touched)} file(s)")
    return 1 if total else 0


def _export_command(action: str, workspace: Path, project: str = "",
                    dest: str = "", yes: bool = False,
                    allow_injection_hits: bool = False) -> int:
    """Handle ``--export`` actions without taking the orchestrator lock.

    Mirrors ``--preflight``: read-mostly, bounded, no network.  The gate
    stages and refuses; it never pushes.  Exit codes: 0 exportable/approved,
    1 refusal (blocked, with the reason), 2 CLI error.
    """
    import export_gate

    if not config.EXPORT_GATE_ENABLED:
        print("export gate is disabled (EXPORT_GATE_ENABLED=false); nothing "
              "can be exported from this workspace", file=sys.stderr)
        return 1

    if action == "list":
        for line in export_gate.list_staging(workspace):
            print(line)
        return 0

    if not project:
        print(f"error: --export {action} requires --project NAME",
              file=sys.stderr)
        return 2
    project_dir = workspace / "projects" / project
    if not project_dir.is_dir():
        print(f"error: no such project: {project}", file=sys.stderr)
        return 2

    if action == "scan":
        if dest:
            reason = export_gate.validate_destination(
                dest,
                allowlist=config.EXPORT_DESTINATION_ALLOWLIST,
                forbidden_extra=config.EXPORT_FORBIDDEN_DESTS)
            if reason:
                print(f"refusing: {reason}", file=sys.stderr)
                return 1
        files, note = export_gate.project_tracked_files(workspace, project)
        if not files:
            print(f"error: cannot stage export: {note}. "
                  "Commit the project first (milestone-commit workflow: "
                  "the gate exports what git tracks, nothing else).",
                  file=sys.stderr)
            return 2
        print(f"collecting: {note}")
        banned = export_gate.forbidden_paths(files, workspace)
        if banned:
            for rel, reason in banned:
                print(f"  banned: {rel} -- {reason}", file=sys.stderr)
            print("VERDICT: project is not exportable; it contains "
                  "permanently blacklisted paths", file=sys.stderr)
            return 1
        tests = export_gate.run_project_tests(
            project_dir, timeout=config.EXPORT_TEST_TIMEOUT_S)
        print(f"tests: ok={tests.get('ok')} passed={tests.get('passed')} "
              f"failed={tests.get('failed')} ({tests.get('note')})")
        injection = export_gate.scan_injection_files(files)
        hits = injection.get("hits") or []
        print(f"injection scan: {injection.get('files_scanned')} file(s), "
              f"{len(hits)} hit(s)"
              + ("" if not hits else
                 f" [{', '.join(h.get('rule', '?') for h in hits[:4])}]"))
        manifest = export_gate.build_manifest(
            workspace, project, files, tests, injection, dest=dest)
        staging = export_gate.stage_manifest(workspace, project, manifest,
                                             files)
        print(f"staged: {staging} ({len(manifest['artifacts'])} artifact(s))")
        if not tests.get("ok"):
            print("VERDICT: test suite is red; nothing is exportable until "
                  "it is green", file=sys.stderr)
            return 1
        next_hint = ("export_gate exists only to stage and refuse; the "
                     "operator performs the physical export by hand")
        print(f"next: --export approve --project {project} then --export "
              f"verify --project {project}. {next_hint}.")
        return 0
    if action == "approve":
        code, message = export_gate.approve_manifest(
            workspace, project,
            allowlist=config.EXPORT_DESTINATION_ALLOWLIST,
            forbidden_extra=config.EXPORT_FORBIDDEN_DESTS, dest=dest,
            allow_hits=allow_injection_hits,
            confirm=(lambda: True) if yes else None)
        print(message, file=(sys.stdout if code == 0 else sys.stderr))
        return code
    # action == "verify"
    code, lines = export_gate.verify_manifest(
        workspace, project,
        allowlist=config.EXPORT_DESTINATION_ALLOWLIST,
        forbidden_extra=config.EXPORT_FORBIDDEN_DESTS,
        timeout=config.EXPORT_TEST_TIMEOUT_S)
    for line in lines:
        print(line)
    return code


def _portfolio_command(workspace: Path) -> int:
    """Handle ``--portfolio``: the parent's read-only one-view report."""
    import portfolio as portfolio_registry

    report = portfolio_registry.build_report(workspace)
    for line in portfolio_registry.render_text(report):
        print(line)
    snapshot = portfolio_registry.write_snapshot(workspace, report)
    counts = report["counts"]
    print(f"snapshot: {snapshot} "
          f"(subsidiaries={counts['subsidiaries']}, "
          f"escalations={counts['escalations']})")
    return 0


def _mcp_serve_command(workspace: Path) -> int:
    """Handle ``--mcp-serve``: host-managed, read-only, no orchestrator lock."""
    import workspace_mcp

    if not config.MCP_WORKSPACE_ENABLED:
        print("workspace-mcp: MCP_WORKSPACE_ENABLED is false; the server "
              "will not start (set MCP_WORKSPACE_ENABLED=true in .env -- "
              "explicit operator config choice, never a prompt-time one)",
              file=sys.stderr)
        return 1
    print("workspace-mcp: serving read-only tools; shutdown on stdin EOF "
          "(host-managed: opencode mcp list / /mcps)",
          file=sys.stderr)
    return workspace_mcp.serve(workspace=workspace)


def _models_command(action: str, workspace: Path) -> int:
    """Handle ``--models`` actions without taking the orchestrator lock."""
    import llm_cache
    import model_router

    if action == "refresh":
        result = model_router.refresh_catalog(workspace_root=workspace)
        if result.get("ok"):
            # v2.9.1: report the diff, because a roster change is the entire
            # reason to run this. It used to print only a model count that was
            # hardcoded to 0 by a missing return key, while the real diff
            # (+610 / -7 on the first live run) went only to the JSONL log.
            added = int(result.get("added_count", 0) or 0)
            removed = int(result.get("removed_count", 0) or 0)
            print(f"catalog refreshed: {result.get('count', 0)} model(s) "
                  f"across {result.get('providers_ok', 0)} provider(s)")
            print(f"  live catalog vs seed roster: +{added} present live but "
                  f"not seeded, -{removed} seeded but no longer listed")
            for key, row in (result.get("providers") or {}).items():
                if row.get("ok"):
                    continue
                why = row.get("skipped") or row.get("error") or "unknown"
                print(f"  {key}: NOT refreshed ({str(why)[:90]})")
            if added or removed:
                print("  Removals are reported, never applied to the seed "
                      "roster: absence from a provider's list is not proof a "
                      "model is dead.")
            return 0
        print(f"catalog refresh failed: {result.get('error')}", file=sys.stderr)
        return 1
    if action == "probe":
        from agent_roles import LLMClient
        llm = LLMClient(workspace)
        results = asyncio.run(model_router.probe_models(
            llm, workspace_root=workspace))
        for row in results:
            print(json.dumps(row, ensure_ascii=False))
        return 0
    if action == "verify":
        from agent_roles import LLMClient
        llm = LLMClient(workspace)
        results = asyncio.run(model_router.verify_providers(
            llm, workspace_root=workspace))
        for row in results:
            print(json.dumps(row, ensure_ascii=False))
        return 0 if results and all(r.get("ok") for r in results) else 1
    if action == "cache-clear":
        removed = llm_cache.clear(workspace)
        print(f"cache cleared: {removed} entries removed")
        return 0
    print(json.dumps(model_router.capacity_summary(workspace),
                     indent=2, ensure_ascii=False))
    return 0


def _author_plan_command(project_dir: Path, goal: str, workspace: Path) -> int:
    """Author a tests-first breakdown now and print it, without orchestrating.

    This is the readable half of Phase E.  The autonomous path runs on the first
    orchestration tick; this one exists so the generated specification can be
    *read* before it is trusted, which is the right way to meet a system that
    has just started writing its own contract.
    """
    import asyncio

    from agent_roles import LLMClient, SpecAuthorAgent

    agent = SpecAuthorAgent(LLMClient(workspace), project_dir, workspace)
    existing: List[str] = []
    board_path = project_dir / ".agent" / "task_board.json"
    try:
        board = json.loads(board_path.read_text(encoding="utf-8"))
        existing = [str(t.get("id")) for t in board.get("tasks", [])]
    except (OSError, json.JSONDecodeError, AttributeError):
        pass

    context = ""
    try:
        context = (project_dir / ".agent" / "context.md").read_text(
            encoding="utf-8")
    except OSError:
        pass

    plan = asyncio.run(agent.run(goal, existing_ids=existing, context=context,
                                 max_tasks=config.SPEC_AUTHOR_MAX_TASKS))
    if not plan.usable:
        print("the author produced no usable tasks:")
        for note in plan.notes:
            print(f"  - {note}")
        for task in plan.rejected:
            print(f"  - rejected {task.test_path or '(unnamed)'}: {task.rejected}")
        return 1

    print(f"\n{len(plan.usable)} task(s) across "
          f"{len(plan.chains())} chain(s):\n")
    for chain, group in plan.chains().items():
        print(f"  chain {chain}:")
        for authored in group:
            test_path = authored.test_path or "(no test)"
            print(f"    {authored.task.id}  {authored.task.title}")
            print(f"       target : "
                  f"{', '.join(authored.task.target_files or [])}")
            print(f"       test   : {test_path}")
    if plan.rejected:
        print(f"\n{len(plan.rejected)} task(s) rejected:")
        for task in plan.rejected:
            print(f"  - {task.test_path or '(unnamed)'}: {task.rejected}")
    if plan.notes:
        print("\nnotes:")
        for note in plan.notes:
            print(f"  - {note}")
    print("\nThis plan has NOT been written to the board. Review it, then run "
          "the\norchestrator to have the system author and build it.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    workspace = config.ensure_workspace()  # idempotent bootstrap

    if args.goal and not (args.new_project or args.new_kaggle
                          or args.campaign_status):
        # A goal on its own is meaningless: there is no project to attach it to.
        print("error: --goal requires --new-project (or --new-kaggle)",
              file=sys.stderr)
        return 2

    if args.new_project:
        try:
            project_dir = config.scaffold_new_project(args.new_project, workspace)
        except (ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)  # CLI error, not logging
            return 2
        board = project_dir / ".agent" / "task_board.json"
        print(f"scaffolded {project_dir}")  # CLI output, not logging
        if args.goal:
            goal_file = project_dir / "goal.txt"
            goal_file.write_text(args.goal.strip() + "\n", encoding="utf-8")
            print(f"goal recorded in {goal_file}")
            if args.author_plan:
                return _author_plan_command(project_dir, args.goal, workspace)
            print("next: python main.py --once   "
                  "(the system will author the board from the goal)")
            return 0
        print(f"next: describe tasks in {board} then run: python main.py")
        return 0

    if args.goal and args.campaign_status:
        project_dir = workspace / "projects" / args.campaign_status
        if not project_dir.exists():
            print(f"error: no such project: {project_dir}", file=sys.stderr)
            return 2
        (project_dir / "goal.txt").write_text(args.goal.strip() + "\n",
                                              encoding="utf-8")
        print(f"goal recorded in {project_dir / 'goal.txt'}")
        return 0

    if args.models:
        return _models_command(args.models, workspace)

    if args.preflight:
        if args.apply and args.preflight != "scrub":
            print("error: --apply is only valid with --preflight scrub",
                  file=sys.stderr)
            return 2
        if args.committed and args.preflight != "scan":
            print("error: --committed is only valid with --preflight scan",
                  file=sys.stderr)
            return 2
        return _preflight_command(args.preflight, workspace, apply=args.apply,
                                  committed=args.committed)

    if args.export:
        return _export_command(args.export, workspace, project=args.project,
                               dest=args.dest, yes=args.yes,
                               allow_injection_hits=args.allow_injection_hits)

    if args.portfolio:
        return _portfolio_command(workspace)

    if args.mcp_serve:
        return _mcp_serve_command(workspace)

    if args.apply:
        print("error: --apply requires --preflight scrub", file=sys.stderr)
        return 2

    if args.committed:
        print("error: --committed requires --preflight scan", file=sys.stderr)
        return 2

    if args.new_kaggle:
        if not args.competition:
            print("error: --new-kaggle requires --competition SLUG",
                  file=sys.stderr)
            return 2
        from domains import kaggle_pack
        try:
            project_dir = kaggle_pack.scaffold(
                args.new_kaggle, args.competition, args.metric,
                args.metric_direction, args.metric_target, workspace)
        except (ValueError, FileExistsError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"scaffolded Kaggle campaign {project_dir}")
        print("next: put competition data in "
              f"{project_dir / 'data' / 'raw'}, then run: python main.py")
        return 0

    if args.campaign_status:
        from domains import kaggle_pack
        project_dir = workspace / "projects" / args.campaign_status
        if not project_dir.exists():
            print(f"error: no such project: {project_dir}", file=sys.stderr)
            return 2
        print(json.dumps(kaggle_pack.load_campaign(project_dir),
                         indent=2, ensure_ascii=False))
        return 0

    orchestrator = SuperOrchestrator(workspace)

    if args.once:
        with _orchestrator_lock(workspace):
            asyncio.run(orchestrator.run(once=True))
        return 0

    if args.feed:
        from dashboard import run_feed
        run_feed()
        return 0

    if args.cmd:
        from dashboard import run_command
        asyncio.run(run_command())
        return 0

    if args.dashboard:
        from dashboard import spawn_dashboard_windows
        with _orchestrator_lock(workspace):
            spawn_dashboard_windows()  # feed + command consoles on the desktop
            try:
                asyncio.run(orchestrator.run())  # headless build loop
            except KeyboardInterrupt:
                log_event("INFO", "dashboard_interrupt_stop", {})
        return 0

    try:
        with _orchestrator_lock(workspace):
            asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        log_event("INFO", "keyboard_interrupt_stop", {})
    return 0


if __name__ == "__main__":
    sys.exit(main())
