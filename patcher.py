"""Real unified-diff application with safety guards and rollback.

Turns the Builder's diff output into actual file changes inside a project:

* :func:`parse_diff`      -- tolerant unified-diff parser (prose/fences ok),
* :func:`apply_diff`      -- guarded, atomic per-file application,
* :func:`snapshot_files`  -- byte-exact pre-apply backup,
* :func:`restore_snapshot`-- rollback (restore edits, delete created files).

Safety model: only *project-relative* paths are writable; absolute paths,
``..`` traversal, backslashes and sensitive directories (``.agent``,
``.system``, ...) are rejected outright.  Every write is atomic
(tmp-file + replace) and every apply is preceded by a snapshot so the
orchestrator can guarantee "never leave a broken tree".

v2.8 adds the **test-integrity** guard (:mod:`test_integrity`): a diff may not
modify or delete a pre-existing test, ``conftest.py`` or pytest config,
because those are the specification the Builder is graded against.  New test
files stay writable so tests-first keeps working.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from logger import log_event
from test_integrity import TestIntegrityGuard
from validation import PatchFailure, PatchResult

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FENCE_RE = re.compile(r"```(?:diff|patch)?\s*\n(.*?)```", re.DOTALL)
_FIRST_MARKER_RE = re.compile(r"(?m)^(--- |\+\+\+ |diff --git|@@ )")

#: Directories a diff may never touch (relative-path components).
FORBIDDEN_PARTS = {".agent", ".shared_workspace", ".system", "logs", ".git",
                   "__pycache__", ".pytest_cache"}

#: Exact filenames a diff may never write.  ``FORBIDDEN_PARTS`` only covers
#: whole directories, so credential-bearing *files* sitting inside a project
#: were writable: an agent could have committed a ``.env`` or overwritten a
#: ``kaggle.json``.  Matching is case-insensitive on the basename.
FORBIDDEN_NAMES = {
    ".env", "kaggle.json", "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    "credentials", "netrc", "npmrc", "pypirc", "known_hosts",
    "authorized_keys", ".npmrc", ".pypirc", ".netrc", ".htpasswd",
    "service-account.json",
}

#: File suffixes a diff may never write (private keys and keystores).
FORBIDDEN_SUFFIXES = {
    ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".ppk", ".asc",
    ".gpg", ".kdbx", ".jceks",
}

#: Extensions exempt from the credential-name heuristic: these are source or
#: documentation, where a name like ``secrets_manager.py`` is a module that
#: *handles* credentials rather than a store of them.  Deliberately excludes
#: data/config formats (.yaml, .json, .ini, .toml, .csv, .env): a file called
#: ``prod-secret.yaml`` or ``aws-credentials.json`` is exactly the credential
#: store this rule exists to refuse.
_SOURCE_SUFFIXES = {".py", ".pyi", ".js", ".mjs", ".ts", ".tsx", ".jsx",
                    ".md", ".rst"}

#: Filename fragments that mark a credential store.  Matched with word
#: boundaries rather than as bare substrings, and only for non-source files:
#: ``aws-credentials.csv`` and ``prod_secret.yaml`` are denied, while the
#: legitimate ``secrets_manager.py`` is not -- refusing to let the Builder
#: create a module that *manages* secrets would block real work.
_SECRET_NAME_RE = re.compile(
    r"(?:^|[^a-z])(?:secret|credential|creds|password|passwd)(?:s)?(?:[^a-z]|$)",
    re.IGNORECASE)

#: How far a hunk may drift from its declared position and still apply.
_SEARCH_RADIUS = 200


@dataclass
class Hunk:
    """One ``@@`` hunk: header positions + body lines (op, text)."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class FilePatch:
    """All hunks for one target file (unified-diff file section)."""

    old_path: Optional[str] = None
    new_path: Optional[str] = None
    is_new_file: bool = False
    is_delete: bool = False
    hunks: List[Hunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def clean_diff_text(text: str) -> str:
    """Strip code fences and any prose before the first diff marker."""
    text = (text or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced and ("---" in fenced.group(1) or "+++" in fenced.group(1)
                   or "@@" in fenced.group(1)):
        text = fenced.group(1).strip()
    marker = _FIRST_MARKER_RE.search(text)
    if marker and marker.start() > 0:
        text = text[marker.start():]
    return text


def parse_diff(diff_text: str) -> List[FilePatch]:
    """Parse unified diff text into per-file patches (tolerant, no raises)."""
    return _parse_diff_single_pass(clean_diff_text(diff_text or ""))


def _parse_diff_single_pass(text: str) -> List[FilePatch]:
    """Single-pass parser: sections close when the next ``--- `` appears."""
    patches: List[FilePatch] = []
    current: Optional[FilePatch] = None
    hunk: Optional[Hunk] = None

    def close() -> None:
        nonlocal current
        if current is not None and current.hunks:
            patches.append(current)
        elif current is not None and current.is_delete:
            patches.append(current)  # pure deletion, no hunks needed
        current = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        hunk_match = _HUNK_RE.match(line)
        if hunk_match:
            if current is None:
                current = FilePatch()
            hunk = Hunk(
                old_start=int(hunk_match.group(1)),
                old_count=int(hunk_match.group(2) or "1"),
                new_start=int(hunk_match.group(3)),
                new_count=int(hunk_match.group(4) or "1"),
            )
            current.hunks.append(hunk)
            continue
        if line.startswith("--- "):
            close()
            current = FilePatch(old_path=_strip_path(line[4:]))
            hunk = None
            continue
        if line.startswith("+++ "):
            if current is None:
                current = FilePatch()
            current.new_path = _strip_path(line[4:])
            current.is_new_file = current.old_path is None
            current.is_delete = current.new_path is None
            continue
        if line.startswith("@@"):
            close()
            current = FilePatch()
            hunk = None
            continue
        if current is not None and hunk is not None:
            if line.startswith("+"):
                hunk.lines.append(("+", line[1:]))
            elif line.startswith("-"):
                hunk.lines.append(("-", line[1:]))
            elif line.startswith(" ") or line == "":
                hunk.lines.append((" ", line[1:] if line else ""))
    close()
    return patches


def _strip_path(raw: str) -> Optional[str]:
    """Normalize a ---/+++ path; ``/dev/null`` -> None (new/delete file)."""
    path = raw.strip().split("\t")[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path or None


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------
def path_is_denied(rel_path: str) -> bool:
    """True when *rel_path* names credential material a diff may not touch.

    Applied by :func:`resolve_target` so the refusal happens before any disk
    write, and reusable by the domain packs when validating model-proposed
    target files.
    """
    pure = PurePosixPath(str(rel_path).replace("\\", "/").strip())
    name = pure.name.lower()
    if not name:
        return True
    if name in FORBIDDEN_NAMES:
        return True
    if pure.suffix.lower() in FORBIDDEN_SUFFIXES:
        return True
    # A *dotfile* variant (.env.local, .env.production) and any credential-store
    # name are denied too.
    if name.startswith(".env"):
        return True
    if pure.suffix.lower() in _SOURCE_SUFFIXES:
        return False
    return bool(_SECRET_NAME_RE.search(pure.name))


def resolve_target(project_root: Path, raw_path: Optional[str]) -> Optional[Path]:
    """Resolve a diff path to a safe project-relative target (or None).

    Rejects: missing path, absolute paths, ``..`` traversal, forbidden
    directories, credential-bearing filenames, and self-references.
    """
    if not raw_path:
        return None
    path = raw_path.strip().replace("\\", "/")
    if not path or path.startswith("/") or ":" in path.split("/")[0]:
        return None
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in ("..", "") for part in pure.parts):
        return None
    if any(part in FORBIDDEN_PARTS for part in pure.parts):
        return None
    if path_is_denied(path):
        return None
    resolved = (Path(project_root) / Path(*pure.parts)).resolve()
    root_resolved = Path(project_root).resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        return None
    return resolved


def scaffolding_allowed(rel_path: str, target_files) -> bool:
    """True when *rel_path* is an ``__init__.py`` that legitimate package
    scaffolding may write: the file must be a declared target itself, sit
    beside a declared target module, or be a package ancestor ``__init__``
    of one (e.g. ``src/quant/signals.py`` permits ``src/__init__.py`` and
    ``src/quant/__init__.py``).  Arbitrary ``__init__.py`` writes elsewhere
    in the project are rejected by the scope guard.
    """
    path = str(rel_path).replace("\\", "/")
    if PurePosixPath(path).name != "__init__.py":
        return False
    for raw in target_files or []:
        target = str(raw).replace("\\", "/")
        if path == target:
            return True
        parts = PurePosixPath(target).parent.parts
        for i in range(len(parts) + 1):
            ancestor = "/".join(parts[:i])
            candidate = f"{ancestor}/__init__.py" if ancestor else "__init__.py"
            if path == candidate:
                return True
    return False


# ---------------------------------------------------------------------------
# Hunk application
# ---------------------------------------------------------------------------
def _find_sequence(haystack: List[str], needle: List[str],
                   expected: int) -> Tuple[Optional[int], str]:
    """Locate ``needle`` in ``haystack``; returns (index, failure_reason).

    Anchor policy:

    * exact match at ``expected`` always wins;
    * otherwise the nearest match within ``_SEARCH_RADIUS`` is used;
    * a multi-line context matching at *several* positions (none of them
      ``expected``) is rejected as ambiguous -- silently editing the wrong
      occurrence of repeated code is worse than failing the hunk, and the
      structured failure feeds back to the Builder as review feedback.
    """
    if not needle:
        return max(0, min(expected, len(haystack))), ""
    span = len(haystack) - len(needle)
    matches = [i for i in range(span + 1)
               if haystack[i:i + len(needle)] == needle]
    if not matches:
        return None, "context not found"
    if expected in matches:
        return expected, ""
    if len(needle) >= 2 and len(matches) > 1:
        return None, (f"ambiguous context ({len(matches)} matches; "
                      f"refine the hunk context lines)")
    nearest = min(matches, key=lambda i: abs(i - expected))
    if abs(nearest - expected) > _SEARCH_RADIUS:
        return None, "context not found within search radius"
    return nearest, ""


def _apply_hunks(content_lines: List[str], hunks: List[Hunk],
                 display_name: str) -> Tuple[Optional[List[str]], str]:
    """Apply hunks to content lines; returns (new_lines, error_reason)."""
    result = list(content_lines)
    offset = 0  # accumulated drift from earlier hunks
    for hunk in hunks:
        old_seq = [text for op, text in hunk.lines if op in (" ", "-")]
        new_seq = [text for op, text in hunk.lines if op in (" ", "+")]
        expected = hunk.old_start - 1 + offset
        idx, anchor_reason = _find_sequence(result, old_seq, expected)
        if idx is None:
            return None, (f"hunk @@ -{hunk.old_start},{hunk.old_count} "
                          f"{anchor_reason} in {display_name}")
        result[idx:idx + len(old_seq)] = new_seq
        offset += len(new_seq) - len(old_seq)
    return result, ""


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".patchtmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------
def repair_future_imports(path: str, content: str) -> str:
    """Move misplaced ``from __future__`` imports to the top of a .py file.

    Models repeatedly emit ``from __future__ import annotations`` *after*
    other statements (a hard SyntaxError on import).  The repair is purely
    mechanical: relocate those lines to just after the module docstring (or
    the very top when there is none).  Already-legal files pass through
    untouched (AST-verified position).  Non-.py files are returned as-is.
    """
    if not path.endswith(".py") or "from __future__" not in content:
        return content

    import ast

    try:
        body = ast.parse(content).body
    except SyntaxError:
        body = None
    if body is not None:
        last_future = max(
            (i for i, s in enumerate(body)
             if isinstance(s, ast.ImportFrom) and s.module == "__future__"),
            default=-1)
        if last_future == -1:
            return content  # no __future__ at statement level
        positioned = True
        for i, stmt in enumerate(body[:last_future]):
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                continue
            if i == 0 and isinstance(stmt, ast.Expr) and \
                    isinstance(stmt.value, ast.Constant) and \
                    isinstance(stmt.value.value, str):
                continue  # module docstring
            positioned = False
            break
        if positioned:
            return content  # already legal -- leave untouched

    # Line-based rebuild: extract __future__ lines, re-insert after the
    # module docstring (or at the very top).
    lines = content.splitlines()
    future = [ln for ln in lines
              if re.match(r"\s*from __future__ import ", ln)]
    if not future:
        return content
    rest = [ln for ln in lines
            if not re.match(r"\s*from __future__ import ", ln)]
    insert_at = 0
    if rest and rest[0].lstrip().startswith(('"""', "'''")):
        quote = rest[0].lstrip()[:3]
        if rest[0].strip().endswith(quote) and len(rest[0].strip()) > 3:
            insert_at = 1  # single-line docstring
        else:
            for i in range(1, len(rest)):
                if quote in rest[i]:
                    insert_at = i + 1
                    break
    rebuilt = rest[:insert_at] + future + [""] + rest[insert_at:]
    return "\n".join(rebuilt) + "\n"


def apply_diff(project_root: Path, diff_text: str,
               guard: Optional["TestIntegrityGuard"] = None) -> PatchResult:
    """Apply a parsed diff inside *project_root*; never raises.

    Each file section is applied independently: a failing hunk marks that
    file as failed (with a concise reason) without aborting the others.

    *guard* is an optional :class:`test_integrity.TestIntegrityGuard`.  When
    supplied, a diff that modifies or deletes a **pre-existing** test,
    ``conftest.py`` or pytest config is refused before the file is touched.
    This is the only layer that also covers the divergence winner re-apply,
    which snapshots the touched paths but never runs the scope guard.
    """
    project_root = Path(project_root)
    result = PatchResult()
    try:
        patches = parse_diff(diff_text)
    except Exception as exc:  # noqa: BLE001 -- parser must never crash the loop
        result.failed.append(PatchFailure(file="(diff)", reason=f"unparseable diff: {exc}"))
        return result
    if not patches:
        result.failed.append(PatchFailure(
            file="(diff)", reason="no applicable file sections in diff"))
        return result

    denied: Dict[str, str] = {}
    if guard is not None:
        candidates = [p.new_path if not p.is_delete else p.old_path
                      for p in patches]
        denied = {v["path"]: v["reason"] for v in guard.violations(
            [c for c in candidates if c])}
        if denied:
            guard.log_refusal(denied)

    for patch in patches:
        target_raw = patch.new_path if not patch.is_delete else patch.old_path
        if target_raw is None:
            result.failed.append(PatchFailure(
                file="(unknown)", reason="file section without a target path"))
            continue
        target = resolve_target(project_root, target_raw)
        if target is None:
            result.failed.append(PatchFailure(
                file=target_raw, reason="unsafe or forbidden path"))
            continue
        rel = str(target.relative_to(project_root)).replace("\\", "/")
        if rel in denied:
            result.failed.append(PatchFailure(
                file=rel,
                reason=f"test-integrity: {denied[rel]}"))
            continue

        if patch.is_delete:
            if target.exists():
                target.unlink()
                result.applied.append(rel)
            else:
                result.failed.append(PatchFailure(
                    file=rel, reason="cannot delete: file does not exist"))
            continue

        exists = target.exists()
        try:
            content_lines = target.read_text(encoding="utf-8").splitlines() \
                if exists else []
        except (OSError, UnicodeDecodeError) as exc:
            result.failed.append(PatchFailure(file=rel, reason=f"unreadable: {exc}"))
            continue

        if patch.is_new_file and exists and all(
                h.old_count == 0 for h in patch.hunks):
            # New-file hunk over an existing file (content-mode regeneration
            # of a file that already exists): replace the whole file with
            # the additions instead of prepending to the old content.
            new_lines = [text for h in patch.hunks
                         for op, text in h.lines if op == "+"]
            err = ""
        else:
            new_lines, err = _apply_hunks(content_lines, patch.hunks, rel)
        if err:
            result.failed.append(PatchFailure(file=rel, reason=err))
            continue
        # Mechanical repair: misplaced ``from __future__`` imports are a
        # recurring model failure; relocate them before the file lands.
        joined = "\n".join(new_lines)
        if rel.endswith(".py"):
            joined = repair_future_imports(rel, joined)
            new_lines = joined.splitlines()
        try:
            _write_atomic(target, "\n".join(new_lines) + ("\n" if new_lines else ""))
        except OSError as exc:
            result.failed.append(PatchFailure(file=rel, reason=f"write failed: {exc}"))
            continue
        result.applied.append(rel)

    log_event("INFO", "diff_applied",
              {"project": project_root.name,
               "applied": result.applied,
               "failed": [f.model_dump() for f in result.failed]})
    return result


# ---------------------------------------------------------------------------
# Snapshot / rollback
# ---------------------------------------------------------------------------
def snapshot_files(project_root: Path, rel_paths: List[str]) -> Optional[Path]:
    """Byte-exact backup of *rel_paths* before applying a diff.

    Returns the backup directory (under ``.agent/attempt_backup/``) or None
    when there is nothing to back up.  The manifest records which files
    existed pre-apply so rollback can delete created files.
    """
    project_root = Path(project_root)
    entries: Dict[str, Dict[str, object]] = {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    backup_dir = project_root / ".agent" / "attempt_backup" / f"{stamp}_{uuid.uuid4().hex[:6]}"
    try:
        for rel in rel_paths:
            target = resolve_target(project_root, rel)
            if target is None or not target.exists():
                entries[rel] = {"existed": False}
                continue
            backup_copy = backup_dir / rel
            backup_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_copy)
            entries[rel] = {"existed": True}
        backup_dir.mkdir(parents=True, exist_ok=True)
        (backup_dir / "manifest.json").write_text(
            json.dumps(entries, indent=2), encoding="utf-8")
        return backup_dir
    except OSError as exc:
        log_event("ERROR", "snapshot_failed", {"error": str(exc)})
        return None


def restore_snapshot(project_root: Path, backup_dir: Optional[Path]) -> int:
    """Roll back to a snapshot: restore edits, delete created files.

    Returns the number of restored/deleted entries.  Never raises.
    """
    if backup_dir is None:
        return 0
    project_root = Path(project_root)
    backup_dir = Path(backup_dir)
    restored = 0
    try:
        manifest = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log_event("ERROR", "restore_manifest_unreadable", {"error": str(exc)})
        return 0
    for rel, info in manifest.items():
        target = resolve_target(project_root, rel)
        if target is None:
            continue
        try:
            if info.get("existed"):
                backup_copy = backup_dir / rel
                if backup_copy.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_copy, target)
                    restored += 1
            elif target.exists():
                target.unlink()
                restored += 1
                # prune now-empty directories created by the patch
                parent = target.parent
                while parent != project_root and parent.is_dir() \
                        and not any(parent.iterdir()):
                    parent.rmdir()
                    parent = parent.parent
        except OSError as exc:
            log_event("ERROR", "restore_entry_failed",
                      {"file": rel, "error": str(exc)})
    try:
        shutil.rmtree(backup_dir, ignore_errors=True)
    except OSError:
        pass
    log_event("INFO", "snapshot_restored",
              {"project": project_root.name, "entries": restored})
    return restored


def discard_snapshot(backup_dir: Optional[Path]) -> None:
    """Delete a snapshot without restoring it (task APPROVED: keep changes)."""
    if backup_dir is None:
        return
    try:
        shutil.rmtree(Path(backup_dir), ignore_errors=True)
    except OSError:
        pass
