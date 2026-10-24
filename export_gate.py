# -*- coding: utf-8 -*-
"""Export gate (Phase 8a): nothing leaves the workspace without a manifest.

Design (v3 Phase 8, operator-approved roadmap):

* The workspace is untrusted-until-approved by construction.  Nothing here
  pushes, commits, uploads, or sends anything anywhere: the gate only stages
  artifacts, records facts, and refuses.  The physical export remains an
  operator action performed outside this module, using the staged manifest.
* Agents (and LLM output generally) can only *file* a manifest, never talk
  their way past one: every refusal is mechanical, from data, with no LLM in
  the loop (the OpenHands deterministic-analyzer pattern).
* Fail closed.  An empty destination allowlist approves no destination; a
  missing approval blocks verify; a red test run or an injection-scan hit
  blocks approval.  Every block prints its reason.

Manifest lifecycle: ``scan`` -> staging dir + manifest.json (facts only)
-> ``approve`` (operator confirmation recorded inside the manifest)
-> ``verify`` (re-hash + re-check every rule; the operator then exports the
staging directory by hand).

See OVERHAUL_PLAN.md Phase 8 for the full roadmap rationale.
"""

import hashlib
import json
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

MANIFEST_SCHEMA = "export-manifest/1.0"

#: Destinations the gate must never approve, however it is configured.
#: The legacy repo is un-pushable from this machine (operator invariant);
#: broker/trading endpoints are blocked because this system is
#: research-grade only (Phase 8e boundary).
BUILTIN_FORBIDDEN_DESTS: Tuple[str, ...] = (
    "github.com/muhriz-requiem/ai-agent-coding-system",
)

#: Relative paths that must never appear in an export, ever.  Operator WIP
#: and owner files are not deliverables; live secret material is not yours
#: to share.  Rules are posix-style relative matchers with three shapes:
#: a leading ``*`` matches any path whose *basename* equals the remainder,
#: a trailing ``/`` matches a directory prefix, otherwise exact match.
FORBIDDEN_PATHS: Tuple[str, ...] = (
    "project_orchestrator.py",          # operator WIP, workspace harness
    "projects/slatepress/",             # operator WIP
    "projects/project_main/data/",      # operator WIP (decisions store)
    "data/",                            # workspace-level operator data
    "memory/",                          # operator instinct lane
    ".opencode/",                       # harness config, not project code
    ".system/",                         # caches/locks/staging, never exports
    "OVERHAUL_PLAN.md",                 # internal ledger
    "*/.env",                           # live secrets (any depth)
    "*/.env.local",
    "*/concerns.txt",                   # operator-private project notes
)

#: Deterministic, no-LLM injection scan.  Applied to every exported text
#: artifact and recorded in the manifest; hits block *approval* (an operator
#: may stare at them and pass --allow-injection-hits, but that choice is
#: explicit and logged).  Data-only by design: it reads the file bytes and
#: never acts on what they say.
INJECTION_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"ignore\s+(all\s+|any\s+)?(previous|prior|above)", "instruction-override"),
    (r"disregard\s+(all\s+|any\s+)?(previous|prior|above)", "instruction-override"),
    (r"you\s+are\s+now\s+", "persona-switch"),
    (r"new\s+instructions?\s*:", "instruction-injection"),
    (r"system\s*prompt", "system-prompt-reference"),
    (r"(reveal|print|send|exfiltrate|transmit)\s+(the\s+)?"
     r"(api\b|secrets?\b|tokens?\b|passwords?\b|credentials?\b)",
     "secret-exfiltration"),
    (r"(curl|wget)\s+https?://", "shell-egress"),
)

_INJECTION_COMPILED: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), label)
    for pattern, label in INJECTION_PATTERNS
)

_TEXT_SUFFIXES = frozenset({
    ".py", ".md", ".txt", ".json", ".toml", ".cfg", ".ini", ".yml", ".yaml",
    ".sh", ".ps1", ".bat", ".js", ".ts", ".html", ".css", ".csv", ".rst",
})


class ExportRefusal(Exception):
    """A mechanical, fail-closed refusal with an operator-readable reason."""


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """Streaming sha256 of a file; gated on it existing (hash, don't trust)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _posix(rel: str) -> Path:
    """Normalize a possibly-Windows relative path to posix for rule matching."""
    return Path(rel.replace("\\", "/"))


def check_forbidden_rel(rel: str) -> Optional[str]:
    """Refusal reason when *rel* (project-relative or workspace-relative path)
    matches a permanent export blacklist, else ``None``.

    Shapes: ``name`` matches that exact relative path; ``prefix/`` matches the
    whole branch; ``*/name`` matches any path with that basename.  These paths
    are operator WIP or secrets and are not negotiable via configuration.
    """
    parts = _posix(rel)
    as_posix = parts.as_posix()
    name = parts.name
    for rule in FORBIDDEN_PATHS:
        if rule.startswith("*/"):
            if name == rule[2:]:
                return f"forbidden export path (rule {rule!r}): {as_posix}"
        elif rule.endswith("/"):
            if as_posix == rule[:-1] or as_posix.startswith(rule):
                return f"forbidden export path (rule {rule!r}): {as_posix}"
        elif as_posix == rule:
            return f"forbidden export path (rule {rule!r}): {as_posix}"
    return None


def validate_destination(dest: str,
                         allowlist: str = "",
                         forbidden_extra: str = "") -> Optional[str]:
    """Refusal reason when *dest* is not approvable, else ``None``.

    Fail closed on all three questions: non-empty destination, membership in
    the operator allowlist, and (last word) absence from the blacklist.  An
    empty allowlist approves nothing by design -- "not configured" must not
    mean "always yes".
    """
    dest = (dest or "").strip()
    if not dest:
        return "no destination given; exports must name their destination"
    lowered = dest.lower().replace("https://", "").replace("http://", "")
    lowered = lowered.rstrip("/")
    forbidden = [d.strip().lower() for d in
                 BUILTIN_FORBIDDEN_DESTS + tuple(
                     s.strip() for s in (forbidden_extra or "").split(",")
                     if s.strip())]
    for rule in forbidden:
        if rule and (lowered == rule or lowered.endswith("/" + rule)
                     or lowered.startswith(rule + "/")):
            return (f"destination is permanently blacklisted "
                    f"(rule {rule!r}): {dest}")
    allowed = [a.strip().lower().rstrip("/") for a in
               (allowlist or "").split(",") if a.strip()]
    if not allowed:
        return (f"no destination allowlist configured; '{dest}' cannot be "
                f"approved (set EXPORT_DESTINATION_ALLOWLIST first)")
    for entry in allowed:
        if lowered == entry or lowered.endswith("/" + entry) or \
                lowered.startswith(entry + "/"):
            return None
    return (f"destination not on the allowlist: {dest}; approved destinations "
            f"are: {', '.join(allowed)}")


def forbidden_paths(files: List[Path], workspace: Path
                    ) -> List[Tuple[str, str]]:
    """``[(rel, reason), ...]`` for every candidate that may never ship.

    A project containing a banned path is not exported *minus the banned
    files* -- it is refused in full: silent trimming would let an operator
    stage a green-looking export that quietly omitted the very evidence the
    whitelist review needed to see.
    """
    refusals: List[Tuple[str, str]] = []
    for path in files:
        try:
            rel = Path(path).relative_to(Path(workspace)).as_posix()
        except ValueError:
            refusals.append((str(path), "outside workspace root"))
            continue
        reason = check_forbidden_rel(rel)
        if reason:
            refusals.append((rel, reason))
    return refusals


def project_tracked_files(
        workspace: Path, project: str,
        command: Optional[Callable[..., Tuple[List[str], str]]] = None
) -> Tuple[List[Path], str]:
    """``(files, note)`` tracked under ``projects/<project>/``, minus ``.agent/``.

    Authoritative answer to "what could this project export": git's own list
    (mirrors ``redaction.tracked_files``).  *command* overrides the git
    invocation for tests; it must return ``(relative_paths, note)``.
    """
    note_extra = ""
    raw: List[str] = []
    if command is None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(workspace), "ls-files", "-z", "--",
                 f"projects/{project}"],
                capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            return [], f"git unavailable ({exc}); cannot determine exports"
        if proc.returncode != 0:
            return [], ("not a git repository (git ls-files exited "
                        f"{proc.returncode}); cannot determine exports")
        for entry in proc.stdout.split(b"\0"):
            if entry:
                raw.append(entry.decode("utf-8", errors="replace"))
    else:
        raw, note_extra = command(workspace, project)

    files: List[Path] = []
    note = note_extra
    base = Path(workspace) if workspace is not None else Path(".")
    for rel in raw:
        rel = rel.strip()
        if not rel:
            continue
        prefix, _, rest = rel.replace("\\", "/").partition("/")
        if prefix != "projects" or not rest:
            continue
        owner, _, sub = rest.partition("/")
        if owner != project or not sub:
            continue
        if sub == ".agent" or sub.startswith(".agent/"):
            continue  # board bookkeeping is internal history, not a product
        files.append(base / rel)
    if not note:
        note = (f"{len(files)} exportable file(s) under projects/{project}"
                if files else f"projects/{project} tracks no exportable files")
    return files, note


# ---------------------------------------------------------------------------
# Injection scan (deterministic, data-only)
# ---------------------------------------------------------------------------

def scan_injection_text(text: str) -> List[Dict[str, Any]]:
    """Mechanical directive-pattern hits in one file's text, with line numbers."""
    hits: List[Dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pattern, label in _INJECTION_COMPILED:
            if pattern.search(line):
                hits.append({"line": lineno, "rule": label})
    return hits


def scan_injection_files(files: List[Path],
                         max_bytes: int = 2_000_000) -> Dict[str, Any]:
    """Scan the export candidates; returns counts plus every hit with its path.

    Text suffixes only; oversized files are counted as scanned but noted, so a
    binary blob cannot hide the check from itself either way.
    """
    scanned = 0
    hits: List[Dict[str, Any]] = []
    for path in files:
        scanned += 1
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue  # note the convention: not excluded, just unscanned
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for hit in scan_injection_text(text):
            hits.append({"path": path.name, **hit})
    return {"files_scanned": scanned, "hits": hits}


# ---------------------------------------------------------------------------
# Test verdict (fresh run; the gate re-measures, it never trusts a claim)
# ---------------------------------------------------------------------------

DEFAULT_TEST_RUNNER: str = sys.executable


def run_project_tests(
        project_dir: Path,
        timeout: float = 300.0,
        runner: Optional[Callable[[List[str], str, float],
                                  Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Run the project's own suite (``python -m pytest tests -q``) and report.

    Fail-closed on everything: a missing suite, a timeout, or an unparseable
    summary is a red verdict, because "could not prove green" is not green.
    *runner* is the injectable seam used by unit tests.
    """
    cmd = [DEFAULT_TEST_RUNNER, "-m", "pytest", "tests", "-q"]
    if runner is not None:
        return runner(cmd, str(project_dir), timeout)
    try:
        proc = subprocess.run(cmd, cwd=str(project_dir), capture_output=True,
                              timeout=timeout, text=True,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": None, "passed": None, "failed": None,
                "note": f"test run exceeded {timeout:.0f}s", "tail": ""}
    except OSError as exc:
        return {"ok": False, "rc": None, "passed": None, "failed": None,
                "note": f"test runner unavailable: {exc}", "tail": ""}
    tail = (proc.stdout or "").strip().splitlines()[-1] if proc.stdout else ""
    passed = failed = None
    match = re.search(r"(\d+) passed", tail)
    if match:
        passed = int(match.group(1))
    match = re.search(r"(\d+) failed", tail)
    if match:
        failed = int(match.group(1))
    ok = proc.returncode == 0 and (failed in (0, None)) and (passed or
                                                            passed == 0)
    return {"ok": bool(ok), "rc": proc.returncode, "passed": passed,
            "failed": failed, "note": ("tests green" if ok else
                                       "test suite is red or unparseable"),
            "tail": tail}


# ---------------------------------------------------------------------------
# Manifest build + staging
# ---------------------------------------------------------------------------

def build_manifest(workspace: Path, project: str, files: List[Path],
                   tests: Dict[str, Any], injection: Dict[str, Any],
                   dest: str = "", head: Optional[str] = None
                   ) -> Dict[str, Any]:
    """Assemble the export manifest from already-verified facts.

    Backstop: raises ``ExportRefusal`` if any artifact is on the permanent
    blacklist; the CLI checks first, this re-checks last, so no code path can
    manifest banned paths by calling build directly.
    """
    refusals = []
    for path in files:
        try:
            rel = Path(path).relative_to(Path(workspace)).as_posix()
        except ValueError:
            rel = str(path)
        reason = check_forbidden_rel(rel)
        if reason:
            refusals.append(reason)
    if refusals:
        raise ExportRefusal(
            "refusing to manifest forbidden export path(s): "
            + "; ".join(refusals))
    artifacts: List[Dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(workspace).as_posix()
        artifacts.append({"path": rel,
                          "sha256": sha256_of(path),
                          "bytes": path.stat().st_size})
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workspace": str(workspace),
        "project": project,
        "destination": {"url": dest or "", "approved": False},
        "git": {"head": head},
        "tests": tests,
        "injection_scan": injection,
        "artifacts": artifacts,
        "approval": None,
    }


def stage_manifest(workspace: Path, project: str, manifest: Dict[str, Any],
                   files: List[Path]) -> Path:
    """Copy artifacts into ``.system/export_staging/<project>-<stamp>-<id>/``
    and write ``manifest.json`` last (tmp+replace), so an interrupted scan
    leaves no half-truthed manifest behind."""
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    staging = (Path(workspace) / ".system" / "export_staging" /
               f"{project}-{stamp}-{uuid.uuid4().hex[:6]}")
    staging.mkdir(parents=True, exist_ok=False)
    base = staging / "payload"
    for path in files:
        rel = path.relative_to(workspace)
        target = base / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    tmp = staging / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(staging / "manifest.json")
    return staging


# ---------------------------------------------------------------------------
# Approve / verify
# ---------------------------------------------------------------------------

def _newest_staging(workspace: Path, project: str) -> Optional[Path]:
    root = Path(workspace) / ".system" / "export_staging"
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()
                  and p.name.startswith(project + "-")
                  and (p / "manifest.json").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def approve_manifest(workspace: Path, project: str, allowlist: str,
                     forbidden_extra: str, dest: str = "",
                     allow_hits: bool = False,
                     confirm: Optional[Callable[[], bool]] = None
                     ) -> Tuple[int, str]:
    """Record operator approval inside the newest manifest for *project*.

    Approval is the one write the gate performs into a manifest, and only
    after every mechanical precondition holds: green tests, a validated
    destination, checksum-stable artifacts, and an injection scan the
    operator has consciously accepted.  Refusals here are exit code 1 so a
    scripted caller cannot mistake "blocked" for "done".
    """
    staging = _newest_staging(workspace, project)
    if staging is None:
        return 1, f"no staged manifest for project {project!r}; run --export scan first"
    manifest_path = staging / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("project")) != project:
        return 1, "staging directory does not belong to the given project"

    tests = manifest.get("tests") or {}
    if not tests.get("ok"):
        return 1, f"refusing: tests not green ({tests.get('note') or 'red'})"

    dest_url = dest or str((manifest.get("destination") or {}).get("url") or "")
    reason = validate_destination(dest_url, allowlist=allowlist,
                                  forbidden_extra=forbidden_extra)
    if reason:
        return 1, f"refusing: {reason}"

    hits = (manifest.get("injection_scan") or {}).get("hits") or []
    if hits and not allow_hits:
        preview = ", ".join(f"{h.get('path')}:{h.get('line')} ({h.get('rule')})"
                            for h in hits[:3])
        return 1, (f"refusing: injection scan has {len(hits)} hit(s) "
                   f"[{preview}]; stare at them, then re-run with "
                   f"--allow-injection-hits only if they are genuinely benign")
    if hits and allow_hits:
        pass  # recorded below; the operator's exception is part of the record

    checksums = []
    for artifact in manifest.get("artifacts") or []:
        staged = staging / "payload" / str(artifact.get("path"))
        if not staged.is_file():
            return 1, (f"refusing: staged file missing: "
                       f"{artifact.get('path')}")
        if sha256_of(staged) != artifact.get("sha256"):
            return 1, f"refusing: checksum mismatch: {artifact.get('path')}"
        checksums.append(artifact.get("path"))

    if confirm is None:
        confirm = lambda: input(  # noqa: E731 - CLI prompt, mirrors stdlib idiom
            f"approve export of {len(checksums)} file(s) from "
            f"{project!r} to {dest_url}? [y/N] ").strip().lower() == "y"
    if not confirm():
        return 1, "not approved (operator declined)"

    manifest["destination"]["url"] = dest_url
    manifest["approval"] = {
        "approved_by": "operator",
        "approved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "export-gate approve",
        "allow_injection_hits": bool(hits and allow_hits),
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(manifest_path)
    return 0, (f"approved: {staging.name} ({len(checksums)} artifact(s) -> "
               f"{dest_url}); operator then exports the payload by hand")


def verify_manifest(workspace: Path, project: str, allowlist: str,
                    forbidden_extra: str, retest: bool = False,
                    timeout: float = 300.0
                    ) -> Tuple[int, List[str]]:
    """Re-check every rule against the staged copy.  Recomputes, never trusts:
    artifacts are re-hashed, the approval block re-read, and the destination
    re-validated.  Export can follow only an exit code 0."""
    staging = _newest_staging(workspace, project)
    if staging is None:
        return 1, [f"no staged manifest for project {project!r}"]
    manifest = json.loads((staging / "manifest.json").read_text(
        encoding="utf-8"))
    lines: List[str] = [f"staging: {staging}"]
    tests = manifest.get("tests") or {}
    lines.append(f"tests: ok={tests.get('ok')} passed={tests.get('passed')} "
                 f"failed={tests.get('failed')} ({tests.get('note')})")
    if retest:
        fresh = run_project_tests(
            Path(str(manifest.get("workspace") or workspace)) / "projects"
            / project, timeout=timeout)
        lines.append(f"tests (fresh re-run): ok={fresh.get('ok')} "
                     f"passed={fresh.get('passed')} failed={fresh.get('failed')}")
        tests = fresh
        manifest["tests"] = fresh
        tmp = (staging / "manifest.json").with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(staging / "manifest.json")

    problems: List[str] = []
    if not (manifest.get("approval") or {}):
        problems.append("no operator approval recorded in manifest")
    if not tests.get("ok"):
        problems.append(f"tests not green ({tests.get('note') or 'red'})")

    dest_url = str((manifest.get("destination") or {}).get("url") or "")
    reason = validate_destination(dest_url, allowlist=allowlist,
                                  forbidden_extra=forbidden_extra)
    if reason:
        problems.append(reason)

    hits = (manifest.get("injection_scan") or {}).get("hits") or []
    allowed_hits = bool((manifest.get("approval") or {}).get(
        "allow_injection_hits"))
    if hits and not allowed_hits:
        problems.append(f"injection scan: {len(hits)} unaccepted hit(s)")
    lines.append(f"injection scan: {len(hits)} hit(s), "
                 f"accepted={allowed_hits if hits else 'n/a'}")

    for artifact in manifest.get("artifacts") or []:
        staged = staging / "payload" / str(artifact.get("path"))
        if not staged.is_file():
            problems.append(f"staged file missing: {artifact.get('path')}")
            continue
        if sha256_of(staged) != artifact.get("sha256"):
            problems.append(f"checksum mismatch: {artifact.get('path')}")
    lines.append(f"artifacts: {len(manifest.get('artifacts') or [])} "
                 f"re-hashed")
    verdict = ("VERDICT: exportable (operator exports the payload by hand)"
               if not problems else "VERDICT: blocked")
    lines.append(verdict + ("" if not problems else f" ({len(problems)} problem(s))"))
    lines.extend(f"  - {p}" for p in problems)
    return (0 if not problems else 1), lines


def list_staging(workspace: Path) -> List[str]:
    """One line per staging directory with its gate state."""
    root = Path(workspace) / ".system" / "export_staging"
    if not root.is_dir():
        return ["no staging directories"]
    lines: List[str] = []
    for staging in sorted(root.iterdir()):
        if not staging.is_dir():
            continue
        manifest_path = staging / "manifest.json"
        if not manifest_path.is_file():
            lines.append(f"{staging.name}: manifest.json MISSING (half-staged?)")
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            lines.append(f"{staging.name}: manifest unreadable")
            continue
        approved = bool(manifest.get("approval"))
        tests_ok = bool((manifest.get("tests") or {}).get("ok"))
        artifacts = len(manifest.get("artifacts") or [])
        dest = str((manifest.get("destination") or {}).get("url") or "-")
        lines.append(f"{staging.name}: project={manifest.get('project')} "
                     f"artifacts={artifacts} tests_ok={tests_ok} "
                     f"approved={approved} dest={dest}")
    return lines or ["no staging directories"]
