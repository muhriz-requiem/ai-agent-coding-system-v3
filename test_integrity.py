"""Test-integrity guard: the Builder may not edit the tests that grade it.

The harness's one real strength is that verification is harness-owned: real
pytest, fail-closed parsing, byte-exact rollback.  That guarantee has a
single hole, and it is a large one -- nothing stops a diff from rewriting the
tests, ``conftest.py`` or pytest configuration it is about to be graded
against.  Because ``target_files`` is board-supplied *data*, a board entry
listing ``tests/test_config.py`` silently turns the gate into a suggestion.

The policy is a **baseline freeze**, not a blanket ban:

* Capture the protected tree at task start (:meth:`TestIntegrityGuard.capture`).
* A protected path that was **already present** may not be modified or
  deleted, unless the task explicitly sets ``allow_test_edits``.
* A **new** file under a protected directory is allowed.  Tests-first is the
  system's headline workflow; authoring new tests is not the threat.

The threat is *weakening the grader* -- deleting an assertion, adding a
``pytest_collection_modifyitems`` hook that deselects the real tests, or
smuggling ``addopts`` in through ``pyproject.toml`` -- not writing tests.

Why the check lives in :mod:`patcher` and not only in the orchestrator: the
divergence winner re-apply (``project_orchestrator`` ``_run_divergence``)
computes the touched paths and snapshots them but never runs the scope guard,
so an orchestrator-only rule would not fire there.  ``patcher.apply_diff`` is
the single funnel every write passes through.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Sequence

from logger import log_event

#: Directories whose entire subtree is graded, not authored-over.
PROTECTED_DIRS: Sequence[str] = ("tests", "test")

#: Config files that can change how pytest collects or runs (``addopts``,
#: ``testpaths``, plugin registration, console entry points).
PROTECTED_NAMES: Sequence[str] = (
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
)

#: Directories never walked when capturing a baseline.  Matches the
#: patcher's own forbidden set so the two policies cannot disagree.
_SKIP_DIRS = frozenset({
    ".agent", ".shared_workspace", ".system", "logs", ".git",
    "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv",
})


def _norm(rel: str) -> str:
    """Normalize a diff path to a comparable project-relative POSIX path."""
    return str(rel or "").replace("\\", "/").strip().lstrip("./")


def is_protected_path(rel: str) -> bool:
    """True when *rel* names part of the grading apparatus.

    Decided on the path shape alone, with no filesystem or task context, so
    it can also be used to filter *proposed* targets before a task is created.
    """
    pure = PurePosixPath(_norm(rel))
    if not pure.parts:
        return False
    if pure.name.lower() in PROTECTED_NAMES:
        return True
    return any(part.lower() in PROTECTED_DIRS for part in pure.parts[:-1]) or \
        any(part.lower() in PROTECTED_DIRS for part in pure.parts)


class TestIntegrityGuard:
    """A frozen snapshot of the tests/config a task is graded against."""

    # Not a pytest test class -- it is named ``Test*`` because it *guards* the
    # tests.  Without this, pytest tries to collect it, warns, and (since it
    # takes constructor arguments) reports a collection error.
    __test__ = False

    __slots__ = ("project_root", "_hashes", "allow_test_edits")

    def __init__(self, project_root: Path,
                 hashes: Optional[Dict[str, str]] = None,
                 allow_test_edits: bool = False) -> None:
        self.project_root = Path(project_root)
        #: rel_path -> sha256 of the content present at capture time.
        self._hashes: Dict[str, str] = dict(hashes or {})
        #: Per-task opt-out for the rare task that must legitimately rewrite
        #: its own tests (e.g. the task that *creates* the harness).
        self.allow_test_edits = bool(allow_test_edits)

    # -- construction ----------------------------------------------------
    @classmethod
    def capture(cls, project_root: Path, allow_test_edits: bool = False
                ) -> "TestIntegrityGuard":
        """Freeze every protected file currently under *project_root*."""
        root = Path(project_root)
        hashes: Dict[str, str] = {}
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(root)).replace("\\", "/")
                if any(part in _SKIP_DIRS for part in rel.split("/")):
                    continue
                if not is_protected_path(rel):
                    continue
                try:
                    data = path.read_bytes()
                except OSError:
                    continue
                hashes[rel] = hashlib.sha256(data).hexdigest()
        return cls(root, hashes, allow_test_edits)

    @classmethod
    def for_task(cls, project_root: Path, task: Any) -> "TestIntegrityGuard":
        """Capture a baseline honouring the task's ``allow_test_edits`` flag."""
        return cls.capture(project_root,
                           bool(getattr(task, "allow_test_edits", False)))

    # -- queries ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._hashes)

    def baseline_paths(self) -> List[str]:
        return sorted(self._hashes)

    def is_protected(self, rel: str) -> bool:
        return is_protected_path(rel)

    def is_baselined(self, rel: str) -> bool:
        """True when this exact path existed when the task started."""
        return _norm(rel) in self._hashes

    def unchanged(self, rel: str) -> bool:
        """True when the on-disk content still matches the captured hash."""
        rel = _norm(rel)
        expected = self._hashes.get(rel)
        if expected is None:
            return True
        target = self.project_root / Path(*PurePosixPath(rel).parts)
        try:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError:
            return False
        return actual == expected

    def violations(self, paths: Iterable[str]) -> List[Dict[str, str]]:
        """Protected-and-baselined paths in *paths* this task may not touch."""
        if self.allow_test_edits:
            return []
        out: List[Dict[str, str]] = []
        for raw in paths or ():
            rel = _norm(raw)
            if not rel or not is_protected_path(rel):
                continue
            if rel in self._hashes:
                out.append({"path": rel, "reason": "pre-existing test/config file"})
        return out

    def check(self, paths: Iterable[str]) -> bool:
        """True when every path in *paths* is permitted."""
        return not self.violations(paths)

    def log_refusal(self, paths: Iterable[str], task_id: str = "") -> None:
        bad = self.violations(paths)
        if bad:
            log_event("WARNING", "test_integrity_violation",
                      {"task_id": task_id,
                       "project_root": str(self.project_root),
                       "rejected": [v["path"] for v in bad][:10],
                       "rejected_count": len(bad),
                       "rule": "pre-existing tests/config are the fixed "
                               "specification; set allow_test_edits to override"})


def filter_allowed_targets(paths: Sequence[str], project_root: Path,
                           task: Any = None) -> List[str]:
    """Drop baselined test/config paths from a proposed ``target_files`` list.

    Used where targets are *proposed* rather than applied -- the Kaggle
    domain pack and the IntegrationAgent -- so an unwritable target never
    reaches the board in the first place.
    """
    if task is not None and getattr(task, "allow_test_edits", False):
        return [str(p) for p in paths]
    guard = TestIntegrityGuard.capture(project_root)
    bad = {v["path"] for v in guard.violations(paths)}
    return [str(p) for p in paths if _norm(p) not in bad]


def protected_candidates(paths: Sequence[str]) -> List[str]:
    """Just the protected entries of *paths* (diagnostics / docs)."""
    return [str(p) for p in paths if is_protected_path(p)]