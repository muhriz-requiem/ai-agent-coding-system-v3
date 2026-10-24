"""v2.9 Phase C: the Spec Critic -- proof that a spec and its tests agree.

The dogfood build made the case for this module.  All 12 task escalations lived
at the boundary where a human handed over an objective and a test file; none was
an engineering defect.  Six of them were contradictions *within* the handover,
and the Reviewer found them only after the Builder had already spent up to
three attempts each.  So the contradiction was real, it was visible at zero
cost, and it was discovered at maximum cost.

That is a placement problem, not a capability problem.  Verification belongs
*before* the Builder, not after a failure.

Two tiers, deliberately split:

* :func:`deterministic_checks` -- free, instant, no model call.  Catches the
  mechanical failures that are objectively decidable, including the
  ``NameError``-shaped one (a test importing a name its objective never
  defined) that cost a real escalation.
* :class:`SpecCriticAgent` -- one call per task with the *full* objective and
  the *full* test source.  Better posed than the Reviewer, which gets a diff
  and 2000 characters of pytest output and must judge from fragments.
"""

from __future__ import annotations

import ast
import builtins
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from validation import (TaskSpec, board_keys_for, is_executable_remedy,
                         validate_or_error)

#: Severity at or above which a task is not scheduled.  Lower severities are
#: annotated onto the objective and the build proceeds, because a Critic that
#: blocks on every nit is a Critic that stops the workspace.
BLOCK_SEVERITY = "high"
_SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class SpecFinding:
    """One objective-vs-tests problem, from either tier."""

    kind: str
    detail: str
    severity: str = "high"
    #: v2.9.1: an allow-listed remedy this finding implies, or "".  Only
    #: remedies present in ``validation.BOARD_WRITE_ALLOWLIST`` are ever
    #: executable, so naming one here cannot widen what automation may write.
    remedy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "detail": self.detail,
                "severity": self.severity, "remedy": self.remedy}


@dataclass
class SpecReport:
    """Everything the Critic found, deterministic tier first."""

    task_id: str = ""
    findings: List[SpecFinding] = field(default_factory=list)
    #: Populated only when the LLM tier ran.
    critique: Optional[Dict[str, Any]] = None
    #: Which tier produced the verdict, for the audit log.
    source: str = "deterministic"

    @property
    def blocking(self) -> List[SpecFinding]:
        return [f for f in self.findings
                if _SEVERITY_ORDER.get(f.severity, 2) >= _SEVERITY_ORDER[BLOCK_SEVERITY]]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def summary(self) -> str:
        if not self.findings:
            return "spec consistent with its tests"
        parts = [f"[{f.severity}] {f.kind}: {f.detail}" for f in self.findings]
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "source": self.source,
                "ok": self.ok, "findings": [f.to_dict() for f in self.findings],
                "critique": self.critique}


# ---------------------------------------------------------------------------
# Tier 1: deterministic.  No model, no quota, no latency.
# ---------------------------------------------------------------------------
def _is_test_path(rel: str) -> bool:
    """Whether a project-relative path names a test file.

    Decided by *convention*, not existence.  Tests-first means the Builder
    creates the test file, so the most common case by far is a path that does
    not exist yet -- and an existence-based test would classify those as
    implementation files, which then breaks every downstream check.
    """
    parts = Path(rel).as_posix().split("/")
    if "tests" in parts[:-1] or parts[0] == "tests":
        return True
    name = parts[-1]
    return name.startswith("test_") and name.endswith(".py")


def _conventional_tests_for(task: TaskSpec, project_root: Path) -> List[Path]:
    """Test files for a task that does not name them, by naming convention.

    This is the real shape of every board in the workspace: 11 of 12 tasks name
    only their implementation file and set ``requires_new_tests``, because under
    tests-first the test is *produced* by the task rather than listed as a
    target.  So the Critic cannot find the tests by looking at ``target_files``
    -- it has to derive them.

    The mapping is the ordinary pytest one: ``src/triage/reader.py`` is graded
    by ``tests/test_reader.py``, and the module basename is the join.  Both the
    flat (``tests/test_x.py``) and nested (``tests/triage/test_x.py``) layouts
    are checked, because projects use both.
    """
    root = Path(project_root)
    found: List[Path] = []
    for rel in (task.target_files or []):
        if _is_test_path(rel):
            continue
        stem = Path(rel).stem
        if stem == "__init__":
            stem = Path(rel).parent.name
        if not stem:
            continue
        parts = Path(rel).as_posix().split("/")
        package = "/".join(parts[:-1]) if len(parts) > 1 else ""
        candidates = [f"tests/test_{stem}.py"]
        if package:
            candidates.append(f"tests/{package.split('/')[-1]}/test_{stem}.py")
        for candidate in candidates:
            path = root / candidate
            if path.exists():
                found.append(path)
                break
    return found


def _test_files_for(task: TaskSpec, project_root: Path) -> List[Path]:
    """Test files this task is graded against.

    Two sources, in order: paths the task names explicitly, and paths derived by
    convention from its implementation targets.  The second is not a fallback --
    it is what almost every task actually needs, since real boards name only
    the implementation file.

    The whole tree is never globbed: that would attribute every unrelated
    failing test in the project to this task's specification.

    Existence is *not* required for explicitly named paths (see
    :func:`_is_test_path`), because tests-first means the file may not exist
    yet.
    """
    root = Path(project_root)
    named = [root / rel for rel in (task.target_files or [])
             if Path(rel).suffix == ".py" and _is_test_path(rel)]
    # 2026-10-17: an authored task's test path of record outranks both the
    # conventions and the hints, and does not require existence (same
    # tests-first semantics as explicitly named paths).  Without it, a test
    # written to a disambiguated name was invisible here and the task was
    # blocked forever at scheduling time.
    authored_path = getattr(task, "authored_test_path", None)
    if authored_path:
        authored = root / authored_path
        if authored not in named:
            named.insert(0, authored)
    derived = _conventional_tests_for(task, root)
    seen = {p.as_posix() for p in named}
    return named + [p for p in derived if p.as_posix() not in seen]


def _module_names(path: Path) -> Set[str]:
    """Names a module defines at top level."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _imported_names(path: Path) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
    return names


def _referenced_names(path: Path) -> Set[str]:
    """Every bare identifier the file reads.

    Local bindings are collected too and subtracted by the caller.  Without
    that, any test that assigns a local -- ``values = sorted(...)`` -- is
    reported as reading an undefined name, which is the check crying wolf on
    the most ordinary line of Python there is.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name):
                names.add(base.id)
    return names


def _looks_internal(dotted: str, project_root: Path) -> bool:
    """True when *dotted* is meant to be one of this project's own modules.

    The distinction that matters, and that a plain "does it resolve" test cannot
    make: ``import pandas`` failing is a missing dependency (the environment
    self-heals it from ``requirements.txt``), while ``from src.journal import
    X`` failing is a *wrong path* -- the module exists, at ``src/ledger/journal.py``,
    and no amount of installing will fix it.

    The test is whether the import's first segment names something that exists in
    the project root -- a source directory or the package the project is building
    -- which is what a project-internal import always does and a third-party one
    never does.
    """
    parts = dotted.split(".")
    if not parts or not parts[0]:
        return False
    root = Path(project_root)
    first = parts[0]
    if (root / first).is_dir():
        return True
    if (root / f"{first}.py").exists():
        return True
    # A project whose tests import the package bare (src/__init__.py layout).
    return False


def _nearest_match(dotted: str, project_root: Path) -> str:
    """The project's file whose stem matches the last segment, for a hint."""
    stem = dotted.split(".")[-1]
    root = Path(project_root)
    for path in sorted(root.rglob(f"{stem}.py")):
        if "__pycache__" in path.parts:
            continue
        return path.relative_to(root).as_posix()
    return ""


def _resolvable_project_module(dotted: str, project_root: Path) -> Optional[Path]:
    """Map ``src.triage.reader`` to its file, if it is one of the project's."""
    parts = dotted.split(".")
    if not parts:
        return None
    root = Path(project_root)
    candidate = root.joinpath(*parts).with_suffix(".py")
    if candidate.exists():
        return candidate
    package_init = root.joinpath(*parts) / "__init__.py"
    if package_init.exists():
        return package_init
    return None


def _imported_modules(test_file: Path) -> Set[str]:
    """Dotted module paths a file imports (``import x.y`` / ``from x.y import``).

    Distinct from :func:`_imported_names`, which returns *symbols*.  The board
    structure check needs to know which modules a test reaches for, so it can
    ask whether some *other* task is the one that creates them.
    """
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    modules: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and not node.level:
            if node.module:
                modules.add(node.module)
    return modules


def _resolvable_or_namespace(dotted: str, project_root: Path) -> Optional[Path]:
    """Like :func:`_resolvable_project_module`, but PEP 420 aware.

    A directory chain under the project root is importable even with no
    ``__init__.py``: Python 3 namespace packages make ``import src.data`` work
    against a bare ``src/data/``.  ``_resolvable_project_module`` only accepts a
    ``.py`` file or a directory *containing* ``__init__.py``, so it reports a
    bare directory import as unresolvable -- which is wrong, and was wrong on 6
    of the 20 completed tasks on the reference board, every one of them a
    ``test_package_dirs_importable`` smoke test that passes.

    Kept separate from :func:`_resolvable_project_module` on purpose. That
    function feeds the name-availability checks, and widening it there would let
    ``from src.data import missing_thing`` resolve to a directory and then be
    reported as importing an undefined name from an "empty" module. Only the
    "does this path exist at all" question wants the namespace rule.
    """
    hit = _resolvable_project_module(dotted, project_root)
    if hit is not None:
        return hit
    parts = dotted.split(".")
    if parts and all(parts):
        directory = Path(project_root).joinpath(*parts)
        if directory.is_dir():
            return directory
    return None


def _module_level_imports(tree: ast.AST) -> List[Tuple[str, bool]]:
    """``(dotted, guarded)`` for every import in *tree*.

    ``guarded`` is True when the import sits inside a ``try``/``except`` (or a
    ``try``/``finally``) block.  A guarded import is a *deliberate* optional
    dependency -- the tests-first guard itself is
    ``try: from src.x import y / except ImportError: pytest.skip(...)`` -- so it
    is not evidence that anyone got a path wrong.

    Getting this wrong is how the first version of this check flagged 9 of the
    20 completed tasks on the reference board, all of them correct: every hit
    was an ``import src.data`` inside a ``try`` in a shared helper.
    """
    guarded_nodes: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for child in ast.walk(node):
                guarded_nodes.add(id(child))

    out: List[Tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            if node.module:
                out.append((node.module, id(node) in guarded_nodes))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, id(node) in guarded_nodes))
    return out


def _unresolvable_internal_imports(test_file: Path, project_root: Path,
                                   impl_files: List[Path],
                                   declared_targets: Optional[Sequence[str]] = None
                                   ) -> List[str]:
    """Imports into the project's own namespace that resolve to nothing.

    v2.9.1.  Every other import check in this module starts from
    ``_resolvable_project_module``, so an import that resolves to *nothing* was
    invisible -- indistinguishable from a third-party package that happens not
    to be installed.  That distinction cost three tasks on a real authored
    board: the Builder wrote ``from src.journal import JournalEntry`` when the
    module is at ``src/ledger/journal.py``, and ``import ledger`` instead of
    ``from src.ledger...``.  The suite then reported ``passed == baseline`` with
    skipped modules, which reads exactly like "the Builder wrote nothing" --
    and the gate's remediation said precisely that, so the Builder wrote the
    same wrong import three times.

    Two things this deliberately does NOT report, because both are correct:

    * **the task's own target module.**  Under tests-first a test imports
      something that does not exist yet; that is the whole design.  ``impl_files``
      is the task's own targets, so those are excluded.
    * **a guarded import.**  ``try: import x / except ImportError: skip`` is how
      a test handles an optional or not-yet-written module, and the tests-first
      guard is exactly that shape.

    Note the division of labour this leaves: the *test* side of a wrong path is
    caught here, before a Builder attempt is spent.  A wrong path in the
    *Builder's own* implementation cannot be caught here at all -- the file does
    not exist yet -- so that case is the real-execution gate's job, and its
    remediation text is what has to name it.
    """
    root = Path(project_root)
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    # Declared targets, NOT files-with-content: under tests-first the task's own
    # module does not exist yet, which is the entire design.  Using
    # ``impls_with_content`` here meant a not-yet-built task excluded nothing
    # and every correctly tests-first task was reported as broken.
    declared = [Path(str(rel)) for rel in (declared_targets or [])]
    own = {p.as_posix() for p in declared}
    own |= {p.with_suffix("").as_posix().replace("\\", "/") for p in declared}
    own_stems = {p.stem for p in declared}

    problems: List[str] = []
    seen: Set[str] = set()
    for dotted, guarded in _module_level_imports(tree):
        if guarded:
            continue
        if _resolvable_or_namespace(dotted, root) is not None:
            continue
        if not _looks_internal(dotted, root):
            continue  # third party: an environment problem, not a spec defect
        normalised = dotted.replace(".", "/")
        if normalised in own or f"{normalised}.py" in own:
            continue
        # This task is going to create it.  That is tests-first, not a defect.
        if dotted.split(".")[-1] in own_stems:
            continue
        if dotted in seen:
            continue
        seen.add(dotted)
        problems.append(_internal_import_problem(dotted, root))
    return problems


def is_internal_module(dotted: str, project_root: Path) -> bool:
    """Public form of :func:`_looks_internal`.

    Exists because the distinction is needed in two places that are easy to let
    drift apart: the Critic (which reports a wrong in-project import path as a
    spec defect) and the environment self-heal (which must *not* try to
    ``pip install`` one).  Sharing the predicate is the point -- the auto-heal
    bug existed precisely because the Critic had the distinction and the
    auto-heal did not.
    """
    return _looks_internal(dotted, project_root)


def nearest_project_module(dotted: str, project_root: Path) -> str:
    """The project's module that *dotted* was probably meant to be.

    Scores every ``.py`` file under the project by how much of the requested
    path it shares, then returns the winner's project-relative path (without
    ``.py``), ready to be pasted into an ``import`` statement.

    Built because a gate that reports "src.journal does not resolve" without
    saying what does is only half a diagnosis.  Observed on a real build: the
    Builder was told ``from src.journal import X`` was wrong and replied with
    ``from src.ledger.journal.entry import X`` -- right package, invented
    submodule.  Naming the actual file ends the guessing:

        src.journal            -> src/ledger/journal.py
        src.ledger.journal.entry -> src/ledger/journal.py
        src.backtest           -> src/pairs/backtest.py
    """
    parts = [p for p in dotted.split(".") if p]
    if not parts:
        return ""
    root = Path(project_root)
    best: Tuple[int, str] = (0, "")
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).with_suffix("").parts
        if any(p == "__pycache__" for p in rel_parts):
            continue
        # Score: matched leading segments, then a matched final segment.
        score = 0
        for requested, actual in zip(parts, rel_parts):
            if requested == actual:
                score += 2
            elif requested.lower() == actual.lower():
                score += 1
        if len(rel_parts) == len(parts):
            score += 1
        if rel_parts and parts[-1].lower() == rel_parts[-1].lower():
            score += 4
        elif rel_parts and parts[-1].lower() in rel_parts[-1].lower():
            score += 2
        if score > best[0]:
            best = (score, "/".join(rel_parts))
    return best[1] if best[0] >= 4 else ""


def _internal_import_problem(dotted: str, root: Path) -> str:
    hint = _nearest_match(dotted, root)
    tail = ""
    if hint:
        stem = dotted.split(".")[-1]
        tail = (f"; the project has {hint} -- import it by its full dotted path "
                f"(e.g. 'from {hint[:-3].replace('/', '.')} import ...', not "
                f"'from {dotted.rsplit('.', 1)[0]}.{stem} import ...')")
    return (f"imports {dotted!r}, which is inside this project's own namespace "
            f"(it starts with a top-level name that exists in the project root) "
            f"but resolves to no file or package{tail}")


def _unresolvable_imports(test_file: Path, project_root: Path,
                          impl_files: List[Path],
                          objective: str = "") -> List[str]:
    """Names a test imports from a project module that the module lacks.

    This is the accurate form of the check that matters most, and it is
    different from "does this name exist anywhere".  A test doing
    ``from src.mod import _analyse_file_text`` *does* have that name bound in
    its own namespace -- so a naive undefined-name scan sees nothing wrong --
    yet the import raises ``ImportError`` at collection and the whole module
    fails.  Nothing about it is visible without resolving the module and asking
    what it actually defines.

    The dogfood build hit exactly this: the objective said
    ``analyse_file_text``, the test imported ``_analyse_file_text``, and the
    disagreement was only discoverable by reading both files.
    """
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return []
    problems: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue  # relative imports need package context we do not have
        module_path = _resolvable_project_module(node.module or "", project_root)
        if module_path is None:
            continue
        available = _module_names(module_path) | _imported_names(module_path)
        for alias in node.names:
            if alias.name == "*":
                continue
            if alias.name in available:
                continue
            # `from m import x as y` binds y regardless of whether m defines x,
            # so an aliased import is not evidence of anything. Same for a name
            # the module re-exports through its own imports -- both are covered
            # by the `available` check above, which unions definitions and
            # imports of the target module.
            if alias.asname:
                continue
            # A tests-first task's target module grows task by task: a
            # symbol the *objective itself names* is this task's work to
            # do, so its absence must not block scheduling (observed live
            # 2026-10-17: T-RENDERER was blocked forever because
            # T-DATALOADER's module already existed without the renderer
            # symbols).  The dogfood typo case is preserved: there the
            # objective named ``analyse_file_text`` while the test
            # imported ``_analyse_file_text``, a form the objective never
            # names -- the word-boundary match keeps those apart.
            if objective and re.search(
                    rf"\b{re.escape(alias.name)}\b", objective):
                continue
            problems.append(
                f"{test_file.relative_to(Path(project_root)).as_posix()} "
                f"imports {alias.name!r} from {module_path.relative_to(Path(project_root)).as_posix()}, "
                f"which does not define it")
    return problems


def _bound_names(path: Path) -> Set[str]:
    """Names the file binds itself: locals, parameters, comprehensions.

    A missing *global* is an error; a missing local almost never is, because the
    binding is in the same file.  ``ast`` can distinguish them, so this check
    does too -- otherwise it reports every local variable as undefined.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    bound: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg in (list(args.args) + list(args.posonlyargs)
                        + list(args.kwonlyargs)):
                bound.add(arg.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= _imported_names_of(node)
    return bound


def _imported_names_of(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            names.add((alias.asname or alias.name).split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name != "*":
                names.add(alias.asname or alias.name)
    return names


#: Names always available without an import, so their absence from a module's
#: definitions proves nothing.  ``dir(__builtins__)`` does not work here: inside
#: a module ``__builtins__`` is a *dict*, and ``dir()`` of that dict yields the
#: dict's own methods (``__class__``, ``__delitem__``, ...) rather than the
#: builtin function names.  ``builtins.__dict__`` is the correct source.
_BUILTIN_NAMES = frozenset(vars(builtins)) | {
    "self", "cls", "__name__", "__file__", "__doc__", "__package__",
    # Standard-library modules a project test may plausibly use unimported in
    # a fragment, plus the harness's own conventions.  Kept deliberately
    # small: every name added here is a name that can never be reported, so
    # generosity here is blindness.
    "pytest", "json", "os", "sys", "re", "math", "pathlib", "datetime",
    "itertools", "functools", "collections", "typing", "dataclasses", "enum",
    "abc", "tempfile", "subprocess", "shutil", "textwrap", "importlib",
    "inspect", "Path", "Any", "Dict", "List", "Optional", "Tuple", "Set",
}

#: English words that can never be a symbol an objective asserts exists.
#: Guards the prose-adjacent heuristics in this module: a regex that matches
#: "must expose the methods" will happily hand back ``the``, and demanding that
#: a symbol named ``the`` already exist blocks a perfectly good task.
_PROSE_WORDS = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "its", "their",
    "all", "any", "each", "every", "no", "not", "and", "or", "but", "so",
    "it", "he", "she", "they", "we", "you", "i", "is", "are", "was", "were",
    "be", "been", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "from", "into", "new", "same", "other", "own", "which", "who", "what",
    "when", "where", "how", "if", "then", "than", "also", "only", "just",
})


def _undefined_names(test_file: Path, impl_files: List[Path]) -> List[str]:
    """Names a test reads but nothing provides.

    This is the check that would have caught the ``_analyse_file_text`` vs
    ``analyse_file_text`` NameError: the objective specified one spelling, the
    test imported the other, and neither was wrong on its own -- they only
    disagreed.  The Builder then spent an attempt and a Reviewer rejection
    discovering a typo that was visible by reading two files.
    """
    defined: Set[str] = set(_BUILTIN_NAMES)
    defined |= _imported_names(test_file)
    for impl in impl_files:
        defined |= _module_names(impl)
    # Locals, parameters and comprehensions the test binds itself.
    defined |= _bound_names(test_file)
    referenced = _referenced_names(test_file)
    return sorted(referenced - defined - _deferred_definitions(test_file))


def _deferred_definitions(path: Path) -> Set[str]:
    """Names defined *inside* a function that the function's own scope covers.

    Python resolves a name against the enclosing function scope before the
    module, so ``def _impl_ready(): ...`` followed by ``if not _impl_ready():``
    at module level is perfectly valid code -- the call is not inside the
    function, it is after it.  A flat name-set has no notion of that scope and
    would report the callee as undefined.

    Rather than reimplement Python's scoping rules, this collects every name
    bound anywhere in the file and subtracts it.  That over-approximates (a
    name bound only inside one function will be forgiven elsewhere), which is
    the correct direction for this check: the failure mode we care about is a
    *missed* NameError, not a false alarm that trains the operator to ignore the
    Critic.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return set()
    defined: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            args = getattr(node, "args", None)
            if args is not None:
                for arg in (list(args.args) + list(args.posonlyargs)
                            + list(args.kwonlyargs)):
                    defined.add(arg.arg)
                if args.vararg:
                    defined.add(args.vararg.arg)
                if args.kwarg:
                    defined.add(args.kwarg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            defined |= _imported_names_of(node)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            target = node.target
            if isinstance(target, ast.Name):
                defined.add(target.id)
            elif isinstance(target, ast.Tuple):
                for element in target.elts:
                    if isinstance(element, ast.Name):
                        defined.add(element.id)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None and \
                        isinstance(item.optional_vars, ast.Name):
                    defined.add(item.optional_vars.id)
        elif isinstance(node, ast.Lambda):
            args = node.args
            for arg in (list(args.args) + list(args.kwonlyargs)):
                defined.add(arg.arg)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            defined |= set(node.names)
    return defined


#: Container types an objective may declare as a return value, and the AST
#: subscript form that is correct for each.  Used to catch the highest-value
#: contradiction class, which is invisible to any name-based check because both
#: sides reference the *same* symbol correctly -- they just disagree about what
#: it returns.  In the dogfood build this was the ``outliers() -> dict[str,float]``
#: vs ``outliers(...)[0]`` pair, and it cost an escalation.
_RETURN_CONTAINERS = {
    "dict": {"subscript_allowed": True, "index_by_int_is_suspicious": True},
    "list": {"subscript_allowed": True, "index_by_int_is_suspicious": False},
    "tuple": {"subscript_allowed": True, "index_by_int_is_suspicious": False},
    "set": {"subscript_allowed": True, "index_by_int_is_suspicious": True},
    "str": {"subscript_allowed": True, "index_by_int_is_suspicious": False},
    "int": {"subscript_allowed": False, "index_by_int_is_suspicious": True},
    "float": {"subscript_allowed": False, "index_by_int_is_suspicious": True},
    "bool": {"subscript_allowed": False, "index_by_int_is_suspicious": True},
}

#: ``name(...) -> dict[str, float]`` / ``name(...) -> Optional[list[str]]``
_RETURN_DECL = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*[-=]>\s*"
    r"(?:Optional\[)?([A-Za-z_][A-Za-z0-9_]*)")


def _declared_returns(task: TaskSpec) -> Dict[str, str]:
    """``{function_name: declared_container}`` parsed from the objective.

    Only the ``-> Type`` spelling counts.  It is the convention the whole
    workspace uses to pin an API -- 27 of 28 completed objectives use it -- so
    reading it is reliable, and anything not written that way is simply not
    checkable here.
    """
    return {name: container
            for name, container in _RETURN_DECL.findall(task.objective or "")}


def _return_type_conflicts(task: TaskSpec, existing_tests: List[Path]
                           ) -> List[str]:
    """Declared return type versus how the tests actually use the result.

    The check that a name-based scan cannot make.  When the objective declares
    ``outliers() -> dict`` and the test writes ``outliers(...)[0]``, every name
    resolves correctly and the module imports cleanly -- the disagreement only
    becomes a ``TypeError`` at runtime, which is the most expensive possible
    moment to discover it.
    """
    declared = _declared_returns(task)
    if not declared:
        return []
    problems: List[str] = []
    for test_file in existing_tests:
        try:
            tree = ast.parse(test_file.read_text(encoding="utf-8",
                                                errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            # Only direct calls: `f(...)[0]`, not `f(...).keys()`.
            if not isinstance(node, ast.Subscript):
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None)
            if not name or name not in declared:
                continue
            container = declared[name]
            rules = _RETURN_CONTAINERS.get(container.lower())
            if rules is None:
                continue
            key = node.slice
            is_int_key = isinstance(key, ast.Constant) and isinstance(
                key.value, int)
            if is_int_key and rules["index_by_int_is_suspicious"]:
                problems.append(
                    f"{test_file.relative_to(test_file.parents[2]).as_posix()} "
                    f"indexes {name}(...)[{key.value}] but the objective "
                    f"declares {name}() -> {container}")
            elif not rules["subscript_allowed"] and not isinstance(
                    key, ast.Constant):
                problems.append(
                    f"{test_file.relative_to(test_file.parents[2]).as_posix()} "
                    f"subscripts {name}(...) but the objective declares "
                    f"{name}() -> {container}, which is not subscriptable")
    return problems


def _objective_symbols_missing(task: TaskSpec, impl_files: List[Path],
                               test_files: List[Path]) -> List[str]:
    """Names the objective asserts exist but which nothing provides.

    The useful signal is not "the objective names a symbol no file defines" --
    that is a feature task, and flagging those would block essentially
    everything.  It is narrower and much more specific: the objective says
    something is *already there* (it exists, it is extended, it currently
    returns) and it is not.  That is a spec describing an API the Builder has
    no way to reach, and it is not fixable by writing better code.

    The undefined-name check already covers the inverse case (tests referencing
    something absent), so the two together catch spelling disagreements from
    both directions without flagging ordinary new work.
    """
    provided: Set[str] = set()
    for impl in impl_files:
        provided |= _module_names(impl)
        provided |= _imported_names(impl)
    for test_file in test_files:
        provided |= _imported_names(test_file)

    text = task.objective or ""
    # Existence-asserting prose, and it has to be *specific*.
    #
    # This check went through three wrong versions against real boards before it
    # was right, and each failure is worth recording because they are all
    # obvious in hindsight:
    #
    #  1. "any symbol no file defines" flagged ordinary feature tasks -- every
    #     one names symbols it wants *created*.
    #  2. "any symbol, if the objective contains 'extend'/'already'" was just as
    #     bad: T-003 says "Create failures.py" *and* mentions an existing
    #     helper in the same breath, so a single document-level cue widened the
    #     check to symbols that had nothing to do with the claim.
    #  3. Extracting the symbol *after* the cue missed the real cases, where
    #     the subject precedes it: "src/config.py already exposes reset_settings".
    #
    # What works is the narrow shape: an *existence-asserting* cue, then a
    # symbol, close together.  Real hits look like "src/config.py already
    # exposes reset_settings_cache".  Everything else is left alone.
    #
    # 4. v2.9.1: the alternation also carried a bare ``\bexposes?``, which is
    #    not an existence claim at all -- "the Ledger class must expose the
    #    methods add_entry, balance_of" is ordinary feature prose, and the
    #    check demanded that a symbol named ``the`` already exist.  It blocked
    #    a correct task on a real board.  The bare form is gone; "must expose"
    #    is a *demand* and "already exposes" is a *claim*, and only the claim is
    #    evidence of anything.
    for match in re.finditer(
            r"(?:(?P<subject>[\w./]+\.py)\s+)?"
            r"(?P<cue>already\s+(?:exposes?|defines?|provides?|returns?|"
            r"accepts?|takes?|has|contains?|includes?)|"
            r"\bcurrently\s+(?:returns?|accepts?|takes?|exposes?))\s+"
            r"`?(?P<symbol>[A-Za-z_][A-Za-z0-9_]*)`?", text, re.IGNORECASE):
        symbol = match.group("symbol")
        # Defence in depth: even with the cue narrowed, never treat English as a
        # symbol.  A blocked task is far more expensive than a missed one.
        if symbol.lower() in _PROSE_WORDS:
            continue
        if symbol not in provided:
            subject = match.group("subject")
            where = f"{subject} " if subject else ""
            return [f"{where}{match.group('cue')} {symbol}"]
    return []


def deterministic_checks(task: TaskSpec, project_root: Path,
                         collected_ids: Optional[List[str]] = None
                         ) -> SpecReport:
    """Objective-vs-tests consistency, decided without a model.

    Every check here is one a machine can settle definitively.  Anything
    requiring judgement belongs in the LLM tier, and keeping the tiers apart is
    what stops the Critic from spending quota to re-derive what ``ast`` already
    knows.
    """
    root = Path(project_root)
    report = SpecReport(task_id=task.id)
    tests = _test_files_for(task, root)
    test_rels = {p.as_posix() for p in tests}
    impls = [root / rel for rel in (task.target_files or [])
             if Path(rel).suffix == ".py" and rel not in test_rels]
    # Only files that exist can be read for symbols; a not-yet-created
    # implementation target is simply not evidence of anything.
    impls_with_content = [p for p in impls if p.exists()]
    existing_tests = [p for p in tests if p.exists()]

    # 1. A task that promises new tests but has none cannot be verified.
    #
    #    The objective often *names* the test file even though target_files does
    #    not -- "implement src/cli.py so tests/test_cli.py passes" -- so the
    #    objective is searched too.  Real boards do this in 3 of 28 cases, and
    #    blocking those would be wrong: the task is perfectly well specified,
    #    the test just lives somewhere the basename convention misses because
    #    several modules share one test file.
    #
    #    Note this asks whether any test file was *found*, not whether one is
    #    named: under tests-first the Builder writes the test, so requiring a
    #    named test file would block every correctly-specified task.  What makes
    #    a task verifiable is that a conventional path exists or will exist.
    if task.requires_new_tests and not tests and not task.allow_test_edits:
        named_in_objective = re.findall(
            r"\b(tests?/[\w./-]+\.py)\b", task.objective or "")
        if named_in_objective:
            report.findings.append(SpecFinding(
                "tests_named_in_objective_only",
                f"objective refers to {', '.join(named_in_objective[:3])} but "
                f"target_files lists only {task.target_files or '[]'}; the "
                f"test must exist for this task to be graded",
                severity="low"))
        else:
            report.findings.append(SpecFinding(
                "no_tests_resolvable",
                "requires_new_tests is set but no test file was named in "
                f"target_files ({task.target_files or '[]'}), none matches the "
                f"conventional path for these targets, and the objective "
                f"names none"))

    # 2. A test file that collects nothing verifies nothing.
    if collected_ids is not None and existing_tests and not collected_ids:
        report.findings.append(SpecFinding(
            "tests_collect_nothing",
            "pytest collected 0 node IDs from this task's test files"))

    # 3. Names the test needs and nothing provides.  The NameError class.
    #
    #    Only readable files are checked: an unreadable test is not evidence of
    #    an undefined name, it is just absent.  #1 covers the case where *no*
    #    readable test exists at all.
    for test_file in existing_tests:
        for name in _undefined_names(test_file, impls_with_content):
            report.findings.append(SpecFinding(
                "undefined_in_tests",
                f"{test_file.relative_to(root).as_posix()} references "
                f"{name!r}, which neither its own imports nor any target file "
                f"defines"))
        for problem in _unresolvable_imports(test_file, root,
                                            impls_with_content,
                                            task.objective or ""):
            report.findings.append(SpecFinding(
                "unresolvable_import", problem))
        for problem in _unresolvable_internal_imports(
                test_file, root, impls_with_content, task.target_files):
            report.findings.append(SpecFinding(
                "unresolved_project_import", problem))

    # 4. API the objective asserts exists but which nothing provides.
    missing = _objective_symbols_missing(task, impls_with_content, existing_tests)
    if missing:
        report.findings.append(SpecFinding(
            "objective_asserts_absent_symbols",
            f"objective states {', '.join(missing[:6])} already exist(s), but "
            f"no target file defines it and it is not in scope to create"))

    # 4b. Declared return type versus how the tests consume it.  The class of
    #     contradiction no name-based check can see, because both sides spell
    #     the symbol correctly and the module imports cleanly -- it only fails at
    #     runtime, which is the most expensive moment to find out.
    for problem in _return_type_conflicts(task, existing_tests):
        report.findings.append(SpecFinding(
            "return_type_conflict", problem))

    # 5. A protected path in target_files.  The Builder would be refused, so
    #    the task cannot succeed as written regardless of the tests.
    #
    #    New test files are the exception and the common case: tests-first means
    #    the Builder *creates* ``tests/test_x.py``, which is protected by path
    #    but not yet baselined, so ``TestIntegrityGuard`` permits writing it.
    #    Flagging those would block every well-formed task in the system.
    from test_integrity import is_protected_path
    for rel in (task.target_files or []):
        if not is_protected_path(Path(rel)):
            continue
        if task.allow_test_edits:
            continue  # the task has explicitly opted in
        candidate = root / rel
        if candidate.exists():
            # Pre-existing: genuinely unmodifiable without allow_test_edits.
            report.findings.append(SpecFinding(
                "target_is_protected",
                f"{rel} already exists and is protected by test-integrity; the "
                f"Builder may not modify it (set allow_test_edits to override)"))
        else:
            report.findings.append(SpecFinding(
                "target_test_missing",
                f"{rel} is a test file the task depends on but which does not "
                f"exist yet; under tests-first the Builder must create it",
                severity="low"))

    # 6. v2.9.1 -- structural checks that need the whole board, not one task.
    #
    #    Both of these were found by reading a real ``--author-plan`` output as
    #    an operator would: a 7-task plan for a bookkeeping library, where task 5
    #    declared two implementation modules under a single test file and task 7
    #    did the same.  Both are violations of the system's own hard-won rule
    #    ("the proven unit of work is ONE implementation file per task", and
    #    "multi-file vague tasks escalate") -- but the rule lived only in the
    #    docs, so nothing enforced it.  A rule that is not checked is a
    #    suggestion.
    for finding in project_structure_findings(task, root):
        report.findings.append(finding)

    return report


#: How many implementation modules one task may declare.
MAX_TARGET_FILES_PER_TASK = 1


def _impl_targets(task: TaskSpec) -> List[str]:
    """Declared targets that are implementation files, not tests."""
    out: List[str] = []
    for rel in (task.target_files or []):
        rel = str(rel)
        if _is_test_path(rel):
            continue
        if Path(rel).name == "__init__.py":
            continue
        out.append(rel)
    return out


def _unowned_package_imports(task: TaskSpec, root: Path,
                             board_tasks: Sequence[Dict[str, Any]]
                             ) -> List[SpecFinding]:
    """Tests import from a package whose ``__init__.py`` no task creates.

    The PEP 420 namespace-package rule hides this from
    :func:`_unresolvable_internal_imports`: ``src/pkg/`` exists as a directory,
    so ``from src.pkg import thing`` looks resolvable -- and it is, as a
    *package*. What does not exist is ``__init__.py``, so the *names* are
    missing and every such test dies at collection with an ImportError.

    Measured on a real authored board: the Spec Author wrote a test doing
    ``from src.slatepress import (...)`` for a goal that also asked for
    ``src/slatepress/__init__.py``, and produced no task for it. The result was
    four tasks, three of which failed on the same ImportError, each with a
    *different* diagnosis -- ``spec_ambiguous``, ``env_broken`` -- and none of
    them mentioning that the package initializer simply did not exist.

    Scans **every** test file on disk, not only this task's own. That matters:
    the offending test belonged to a task the Critic had dropped, so a
    per-task scan could not see it -- which is precisely how an orphan test
    manages to break the whole suite while every task looks individually fine.
    """
    owned: Set[str] = set()
    for raw in board_tasks:
        if not isinstance(raw, dict):
            continue
        for rel in raw.get("target_files") or []:
            owned.add(str(rel).replace("\\", "/"))

    tests_dir = root / "tests"
    if not tests_dir.is_dir():
        return []
    findings: List[SpecFinding] = []
    seen: Set[str] = set()
    for path in sorted(tests_dir.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level:
                continue
            module = node.module or ""
            if not module or not _looks_internal(module, root):
                continue
            pkg_dir = root / module.replace(".", "/")
            init = f"{module.replace('.', '/')}/__init__.py"
            if not pkg_dir.is_dir() or (root / init).exists() or init in owned:
                continue  # has an initializer, or a task will write one
            # PEP 420 still satisfies this import when the names are submodules
            # of the package: `from src.data import provider` resolves to
            # src/data/provider.py with no __init__.py anywhere. Only a name the
            # package does not contain needs the initializer to define it --
            # which is the case that actually breaks collection.
            absent = sorted(
                alias.name for alias in node.names
                if alias.name != "*"
                and not (pkg_dir / f"{alias.name}.py").exists()
                and not (pkg_dir / alias.name).is_dir()
                and alias.name != "__init__")
            if not absent:
                continue
            if module in seen:
                continue
            seen.add(module)
            findings.append(SpecFinding(
                "package_init_unowned",
                f"{path.name} does 'from {module} import {', '.join(absent)}', "
                f"but {module} has no __init__.py, no task creates one, and "
                f"those names are not submodules of it. The import fails at "
                f"collection. Either a task must own "
                f"{module.replace('.', '/')}/__init__.py and re-export them, "
                f"or the test must import the submodule that defines each one.",
                severity="high"))
    return findings


def project_structure_findings(task: TaskSpec,
                               project_root: Path) -> List[SpecFinding]:
    """Board-level structural checks, from one task's point of view.

    These answer questions no single-task check can.  All three were found by
    reading a real ``--author-plan`` output the way an operator does: a 7-task
    plan for a bookkeeping library in which one task declared two implementation
    modules under a single test file.  Both are violations of the system's own
    hard-won rules -- "the proven unit of work is ONE implementation file per
    task" and "multi-file vague tasks escalate" -- but the rules lived only in
    the docs.  A rule that is not checked is a suggestion.

    Calibrated against the reference board (``project_main``, 20 tasks, all
    delivered), which is why two of these are ``medium`` rather than ``high``:
    reporting a real defect must not block a board that demonstrably worked.
    """
    findings: List[SpecFinding] = []
    impl_targets = _impl_targets(task)

    # A bootstrap task legitimately owns the harness: its conftest, its config
    # module, the exception that lets it write tests at all.  Exempting it is
    # not a loophole -- ``allow_test_edits`` is already the single most
    # privileged flag on a task, and the reference board sets it on exactly one.
    if len(impl_targets) > MAX_TARGET_FILES_PER_TASK \
            and not task.allow_test_edits:
        findings.append(SpecFinding(
            "multi_file_task",
            f"declares {len(impl_targets)} implementation modules "
            f"({', '.join(impl_targets[:4])}); one implementation file per task "
            f"is the strongest predictor of first-try completion in this system",
            severity="medium", remedy="split_task"))

    board = load_board(project_root)
    if board is None or not impl_targets:
        return findings
    # Whole-board properties first, before the "are there other tasks" early
    # return below: a single-task board still has to be checked for them. Putting
    # them after meant the first version silently found nothing on a one-task
    # board, which is exactly the shape of the fixture that caught it.
    for finding in _duplicate_symbol_findings(task, Path(project_root),
                                             board.get("tasks") or []):
        findings.append(finding)
    for finding in _unowned_package_imports(task, Path(project_root),
                                            board.get("tasks") or []):
        findings.append(finding)

    board_tasks = [t for t in (board.get("tasks") or [])
                   if isinstance(t, dict) and str(t.get("id")) != task.id]
    others = board_tasks
    if not others:
        return findings

    owned = {str(t.get("id")) for t in others
             if str(t.get("id")) in (task.dependencies or [])}
    mine = {Path(rel).as_posix() for rel in impl_targets}

    # (a) Duplicate ownership: two tasks writing one file.  Under LINE_COUNT=1
    #     the second silently overwrites the first; under a real line partition
    #     they race.  Reported, not blocking -- the reference board has two such
    #     pairs and still delivered, which is exactly why it is worth surfacing
    #     rather than treating as fatal.
    for rel in sorted(mine):
        rivals = sorted(str(t.get("id")) for t in others
                        if rel in {Path(str(r)).as_posix()
                                   for r in (t.get("target_files") or [])})
        if rivals:
            findings.append(SpecFinding(
                "duplicate_target_file",
                f"{rel} is also a target of {', '.join(rivals)}; two tasks "
                f"writing one file overwrite each other sequentially and race "
                f"under a line partition",
                severity="medium", remedy="split_task"))

    # (b) A missing dependency edge, checked precisely: names the task's own
    #     test imports that none of its own targets provide, where another task
    #     does provide them and is not a dependency.  Matching on *paths* and
    #     excluding the task's own targets is what keeps this from firing on
    #     every task that imports the module it is building -- an earlier,
    #     stem-only version of this check flagged 6 of 20 healthy tasks.
    unsatisfied: Set[str] = set()
    for test_file in _test_files_for(task, Path(project_root)):
        for dotted in _imported_modules(test_file):
            resolved = _resolvable_project_module(dotted, Path(project_root))
            if resolved is None:
                continue  # stdlib or third party: not this board's problem
            stem = Path(str(resolved)).stem
            if stem and stem != "__init__":
                unsatisfied.add(stem)
    if not unsatisfied:
        return findings
    provided: Dict[str, List[str]] = {}
    for other in others:
        for rel in other.get("target_files") or []:
            rel = str(rel)
            if _is_test_path(rel):
                continue
            stem = Path(rel).stem
            if stem:
                provided.setdefault(stem, []).append(str(other.get("id")))
    orphan = sorted(
        name for name in unsatisfied
        if name in provided and not set(provided[name]) & owned
        and not any(Path(rel).stem == name for rel in impl_targets))
    if orphan:
        detail = "; ".join(f"{n} is created by {', '.join(provided[n])}"
                           for n in orphan[:4])
        findings.append(SpecFinding(
            "cross_task_dependency_missing",
            f"this task's test imports {detail}, but names none of them in "
            f"dependencies, so it can be scheduled before they exist and the "
            f"test fails at collection with a confusing passed==baseline block",
            severity="medium", remedy="narrow_objective"))
    return findings


def _board_test_imports(board_tasks: Sequence[Dict[str, Any]],
                        root: Path) -> Dict[str, List[Tuple[str, str]]]:
    """``symbol -> [(task_id, module), ...]`` demanded by the board's tests.

    Reads the *tests*, not the objectives, because the tests are the contract:
    if two test files each ask a different module for the same name, then two
    implementations have to define it independently, and the second will drift
    from the first.

    Test paths come from :func:`_test_files_for`, not from ``target_files`` --
    under tests-first an authored task deliberately declares only its
    *implementation* file (the test is the frozen contract, written before the
    task exists), so looking in ``target_files`` finds nothing at all.  That was
    the first version, and it silently detected zero.
    """
    wanted: Dict[str, List[Tuple[str, str]]] = {}
    for raw in board_tasks:
        if not isinstance(raw, dict):
            continue
        try:
            spec, err = validate_or_error(TaskSpec, raw)
        except Exception:  # noqa: BLE001 -- a malformed entry is not evidence
            continue
        if err or spec is None:
            continue
        task_id = str(raw.get("id"))
        for path in _test_files_for(spec, root):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8",
                                                errors="replace"))
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level:
                    continue
                module = node.module or ""
                if not module or not _looks_internal(module, root):
                    continue
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    wanted.setdefault(alias.name, []).append((task_id, module))
    return wanted


def _duplicate_symbol_findings(task: TaskSpec, root: Path,
                               board_tasks: Sequence[Dict[str, Any]]
                               ) -> List[SpecFinding]:
    """Two tasks asked to define the same public name.

    The failure this prevents, observed on a real authored board: the
    ``ledger`` task's test asked for ``JournalEntry`` from ``src.ledger.ledger``
    while the ``journal`` task had already defined it in
    ``src.ledger.journal``.  The Builder satisfied the frozen test by
    *duplicating* the class rather than importing it, the gate passed -- the
    gate only knows what the test checks -- and the delivered library has two
    incompatible ``JournalEntry`` types.

    ``__init__`` modules are excluded because re-exporting a name is the
    opposite of duplicating it, which is what makes ``from src.pkg import main``
    in one test and ``from src.pkg.cli import main`` in another legitimate.
    """
    wanted = _board_test_imports(board_tasks, root)
    mine = {str(t.get("id")) for t in board_tasks
            if isinstance(t, dict) and str(t.get("id")) == task.id}
    findings: List[SpecFinding] = []
    for symbol, sources in sorted(wanted.items()):
        mine_here = {m for tid, m in sources if tid in mine}
        others = {m for tid, m in sources if tid not in mine}
        # A package's __init__ is a re-export surface, not a definition site.
        mine_here = {m for m in mine_here if not m.endswith(".__init__")}
        others = {m for m in others if not m.endswith(".__init__")}
        clash = {m for m in mine_here if m not in others}
        if clash and others:
            findings.append(SpecFinding(
                "duplicate_public_symbol",
                f"{symbol!r} is demanded by this task's test from "
                f"{', '.join(sorted(mine_here))} and by another task's test "
                f"from {', '.join(sorted(others))}; both modules would have to "
                f"define it independently and the copies will drift. One task "
                f"must own it and the other must import it.",
                severity="medium"))
    return findings


def load_board(project_root: Path) -> Optional[Dict[str, Any]]:
    """The project's board, or None when absent/unreadable."""
    path = Path(project_root) / ".agent" / "task_board.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# Tier 2: one model call, with the whole spec and the whole test.
# ---------------------------------------------------------------------------
_CRITIC_SYSTEM = (
    "You audit a task specification against the tests that will grade it. "
    "You are looking ONLY for ways the two disagree -- not for style, not for "
    "completeness of prose, not for whether the implementation is a good idea.\n"
    "Report as a single JSON object:\n"
    '{"consistent": bool, "severity": "low|medium|high", "confidence": float,\n'
    ' "contradictions": ["objective says X, test asserts Y"],\n'
    ' "unsatisfied": ["acceptance criterion with no test covering it"]}\n'
    "Quote both sides of every contradiction. If the spec and tests agree, say "
    "so plainly with an empty contradictions list -- do not invent findings to "
    "look thorough."
)

_CRITIC_USER = (
    "Audit this task specification against its tests.\n"
    "<<<TASK_JSON>>>\n{task}\n"
    "<<<TEST_SOURCE>>>\n{source}\n"
    "Respond with one JSON object."
)


def build_critic_prompt(task: TaskSpec, project_root: Path,
                        max_chars: int = 24000) -> Tuple[str, str]:
    """The Critic's prompt: full objective, full test source.

    The Reviewer works from 2000 characters of pytest output and a diff.  This
    gets the two documents whose agreement is actually in question, in full.
    Truncation is applied to the *test source* tail only, and only past a
    generous budget -- a half-read test file invites a confident wrong verdict.
    """
    root = Path(project_root)
    chunks: List[str] = []
    budget = max_chars
    for test_file in _test_files_for(task, root):
        try:
            body = test_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = test_file.relative_to(root)
        if len(body) > budget:
            body = (body[:budget]
                    + f"\n... [truncated, {len(body) - budget} more chars]\n")
            budget = 0
        # POSIX separators: the board, the docs and every log row use them, and
        # on Windows ``relative_to`` yields backslashes, so a finding would
        # otherwise cite a path that appears nowhere else in the project.
        chunks.append(f"--- {rel.as_posix()} ---\n{body}")
    source = "\n\n".join(chunks) or "(no test source available)"
    return task.objective or "", source


def coerce_critique(data: Any, task_id: str = "") -> Optional[Dict[str, Any]]:
    """Tolerate real-model shapes for the critique payload."""
    if not isinstance(data, dict):
        return None
    severity = str(data.get("severity") or "medium").strip().lower()
    if severity not in _SEVERITY_ORDER:
        severity = "medium"
    contradictions = data.get("contradictions")
    if isinstance(contradictions, str):
        contradictions = [contradictions]
    unsatisfied = data.get("unsatisfied")
    if isinstance(unsatisfied, str):
        unsatisfied = [unsatisfied]
    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    # Derive consistency from the contradictions list rather than trusting the
    # boolean.  A model that writes ``consistent: true`` and then lists three
    # contradictions has not read its own output, and the boolean is the part
    # a lazy template defaults -- the list is the part it actually composed.
    # Trusting the boolean would let exactly that shape through.
    consistent = not contradictions
    return {
        "task_id": task_id,
        "consistent": consistent,
        "severity": severity,
        "confidence": max(0.0, min(1.0, confidence)),
        "contradictions": [str(c)[:400] for c in (contradictions or [])][:8],
        "unsatisfied": [str(u)[:400] for u in (unsatisfied or [])][:8],
    }


def merge_critique(report: SpecReport, critique: Dict[str, Any]) -> SpecReport:
    """Fold the model's findings into a deterministic report.

    Severity is combined by taking the worse of the two, and the model can only
    *add* findings -- it cannot clear a mechanical one.  A model that decides
    "actually this is fine" must not be able to un-flag a test importing a name
    nothing defines.
    """
    merged = SpecReport(task_id=report.task_id, source="deterministic+llm")
    merged.findings.extend(report.findings)
    for text in critique.get("contradictions") or []:
        merged.findings.append(SpecFinding(
            "spec_test_contradiction", text,
            severity=critique.get("severity", "medium")))
    for text in critique.get("unsatisfied") or []:
        # An acceptance criterion no test covers is a gap in the grade, not a
        # contradiction, and it should not block on its own.
        merged.findings.append(SpecFinding(
            "acceptance_criterion_untested", text, severity="low"))
    merged.critique = critique
    return merged


def annotate_objective(objective: str, report: SpecReport) -> str:
    """Prefix non-blocking findings onto an objective.

    Blocking findings are never annotated in -- the task is held instead, and
    handing a Builder a contradiction it has been told about produces worse code
    than not running the attempt at all.
    """
    notes = [f.to_dict() for f in report.findings
             if f not in report.blocking]
    if not notes:
        return objective
    lines = "\n".join(f"- [{n['severity']}] {n['kind']}: {n['detail']}"
                      for n in notes)
    return (f"{objective}\n\n<<<SPEC_CRITIC_NOTES>>>\n{lines}\n"
            f"<<<END_SPEC_CRITIC_NOTES>>>")


def remedy_is_allowed(remedy: str) -> bool:
    """Re-exported so callers need not import :mod:`validation` directly."""
    return is_executable_remedy(remedy)


def remedy_board_keys(remedy: str) -> Tuple[str, ...]:
    return board_keys_for(remedy)
