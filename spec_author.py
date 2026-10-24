"""v2.9 Phase E: the system authors its own specification.

Today the loop is *human writes tests -> human writes spec -> system
implements*.  The logtriage build measured exactly what that costs: 12
escalations, all of them at the handoff boundary, none of them an engineering
defect, and 6 contradictions between an objective and a test file that were each
individually reasonable.  Every one of those was writing work the harness could
do.

So this module does the writing, from one paragraph.

Two design points carry most of the weight, and both were verified empirically
before being relied on:

**Tests are written before the task is baselined, which makes them a frozen
contract.**  ``TestIntegrityGuard`` snapshots protected files when a task
*starts*.  A test authored here is therefore protected from the Builder that
must satisfy it.  Probed directly: the Builder's rewrite was refused, the
node-ID gate still fired, and the suite went green.  That is what makes
autonomous authoring safe -- the system writes the contract *and* is forbidden
from quietly editing it.

**Tasks are emitted as independent chains, not a flat list.**  Chain shape is
what makes Phase G's line partition free: each chain is a line, and chains that
share no target file can run concurrently.  Nothing else in the design derives
parallelism this cheaply.

The cost is stated plainly: the system now writes the contract its own Builder is
graded against, and the Spec Critic is the only independent check between them.
That is why Phase C gates this module rather than following it.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from validation import TaskSpec, validate_or_error

#: Cap on authored tasks per goal.  Enough for a real module decomposition,
#: small enough that one bad generation cannot flood a board.
MAX_TASKS = 8

#: Marker wrapped around untrusted model output in every prompt below.
_UNTRUSTED_OPEN = "<<<UNTRUSTED_MODEL_OUTPUT>>>"
_UNTRUSTED_CLOSE = "<<<END_UNTRUSTED_MODEL_OUTPUT>>>"


def untrusted(text: str) -> str:
    """Fence text that must never be read as instructions.

    Same defence the Builder and Reviewer prompts use.  An authored objective
    ends up concatenated into a prompt, so an objective containing "ignore
    previous instructions" is a prompt-injection vector unless it is fenced.
    """
    return f"{_UNTRUSTED_OPEN}\n{text}\n{_UNTRUSTED_CLOSE}"


@dataclass
class AuthoredTask:
    """One proposed task plus the test file that will grade it."""

    task: TaskSpec
    #: Project-relative path the test will be written to.
    test_path: str = ""
    #: Complete file content for that path.
    test_source: str = ""
    #: Index within its chain, 0-based.
    chain_index: int = 0
    #: Chain label, so independent chains can be told apart downstream.
    chain: str = ""
    #: Set when the task failed validation and was dropped.
    rejected: str = ""
    #: True when the model named test_path itself (operator-grade contract);
    #: False when coerce_task derived it from the target module stem.  Only
    #: derived paths may be silently disambiguated among siblings.
    test_path_explicit: bool = False

    @property
    def usable(self) -> bool:
        return not self.rejected


@dataclass
class AuthoredPlan:
    """A full proposed board: chains of tasks, each with its test."""

    goal: str = ""
    tasks: List[AuthoredTask] = field(default_factory=list)
    #: Non-fatal problems worth surfacing to a human.
    notes: List[str] = field(default_factory=list)

    @property
    def usable(self) -> List[AuthoredTask]:
        return [t for t in self.tasks if t.usable]

    @property
    def rejected(self) -> List[AuthoredTask]:
        return [t for t in self.tasks if t.rejected]

    def chains(self) -> Dict[str, List[AuthoredTask]]:
        """Tasks grouped by chain label, each in dependency order."""
        out: Dict[str, List[AuthoredTask]] = {}
        for task in self.usable:
            out.setdefault(task.chain or "default", []).append(task)
        for group in out.values():
            group.sort(key=lambda t: t.chain_index)
        return out

    def summary(self) -> Dict[str, Any]:
        return {"goal": self.goal[:200], "tasks": len(self.tasks),
                "usable": len(self.usable), "rejected": len(self.rejected),
                "chains": len(self.chains()), "notes": self.notes[:6]}


# ---------------------------------------------------------------------------
# Coercion: a model's task drafts into validated TaskSpec objects
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return []


def _next_task_id(existing: Sequence[str], index: int) -> str:
    """First unused ``T-00N`` id, so a re-run never collides with the backlog."""
    taken = {str(e) for e in existing}
    for candidate in range(index + 1, index + 1 + MAX_TASKS * 4):
        proposed = f"T-{candidate:03d}"
        if proposed not in taken:
            return proposed
    return f"T-{index + 1:03d}"


def _slug(text: str, limit: int = 40) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return out[:limit] or "task"


def coerce_task(draft: Any, chain: str, index: int,
                existing_ids: Sequence[str],
                project_root: Optional[Path] = None,
                require_tests: bool = True) -> AuthoredTask:
    """Turn one model draft into a validated :class:`AuthoredTask`.

    Always returns an :class:`AuthoredTask`; a draft that cannot be used comes
    back with ``task=None`` and ``rejected`` explaining which rule it broke.

    v2.9.1: the non-dict and empty-objective cases used to return ``None``, and
    the caller substituted the generic note "task #N was not a usable draft".
    That is the *only* thing an operator saw when a model answered with a list
    of plain English strings instead of task objects -- which is what
    ``cloudflare:@cf/openai/gpt-oss-120b`` did three times running against the
    same goal.  A rejected draft now names its own failure, so the log reads
    "draft #1 is a string, not a task object" and the difference between a
    schema mistake and a missing objective is legible.

    ``require_tests=False`` is the *breakdown* stage of two-stage authoring: the
    task is accepted on the strength of its API spec alone, and the test file is
    written by a second, separate call.  See :func:`build_test_prompt`.
    """
    def _rejected(reason: str) -> AuthoredTask:
        return AuthoredTask(task=None, chain=chain, chain_index=index,
                            rejected=reason[:300])

    if not isinstance(draft, dict):
        kind = type(draft).__name__
        detail = f"a {kind}, not a task object"
        if isinstance(draft, str) and draft.strip():
            detail += f": {draft.strip()[:80]!r}"
        return _rejected(detail)
    objective = str(draft.get("objective") or "").strip()
    if not objective:
        missing = sorted(str(k) for k in
                         ("objective", "title", "target_files", "test_path",
                          "test_source") if not draft.get(k))
        return _rejected("no 'objective'; empty keys: "
                         f"{', '.join(missing) or '(all blank)'}")

    target_files = _as_list(draft.get("target_files"))
    test_source = str(draft.get("test_source") or draft.get("test_content") or "")
    explicit_path = str(draft.get("test_path") or "").strip()
    test_path = explicit_path

    # The conventional location for a task's test, used when the model did not
    # say: tests/test_<module stem>.py beside the first implementation target.
    test_path_explicit = bool(test_path)
    if not test_path:
        if target_files:
            impl = Path(target_files[0]).stem
            if impl == "__init__":
                impl = Path(target_files[0]).parent.name
            test_path = f"tests/test_{_slug(impl)}.py"
        else:
            test_path = f"tests/test_{_slug(objective)}.py"

    task_id = str(draft.get("id") or "").strip() or _next_task_id(existing_ids,
                                                                  index)
    raw: Dict[str, Any] = {
        "id": task_id,
        "title": str(draft.get("title") or objective[:80]),
        "objective": objective,
        "target_files": target_files,
        "dependencies": _as_list(draft.get("dependencies")),
        "acceptance_criteria": _as_list(draft.get("acceptance_criteria")),
        # v2.9.1: derived when the model does not supply one.  ``steering``
        # clusters escalations on this key, so a task without one is invisible
        # to systemic-failure detection -- and the fallback the escalator used
        # ("TASK:<id>") parsed down to the bare token "TASK", so every
        # escalated task looked like the same signature as every other one.
        # Deriving it from the module under test is what makes three failures in
        # one module cluster as one design problem.
        "error_signature": str(draft.get("error_signature") or "").strip()
        or _derive_signature(target_files, task_id),
        "requires_new_tests": True,
        # 2026-10-17: the test path of record rides with the task so the
        # critic can resolve tests whose name is not the module stem
        # (authored boards may disambiguate paths among siblings).
        "authored_test_path": test_path or None,
        # Implementation targets only.  The test is written here, before the
        # task starts, precisely so the Builder is graded against it and
        # forbidden from changing it.  Putting it in target_files as well would
        # invite the Builder to rewrite the contract it is held to.
        "priority": str(draft.get("priority") or "medium"),
        "task_type": str(draft.get("task_type") or "feature"),
    }
    task, err = validate_or_error(TaskSpec, raw)
    if err:
        return _rejected(err["error"])

    if test_source and project_root is not None:
        problem = _validate_test_source(test_source, test_path)
        if problem:
            return _rejected(f"test_source: {problem}")
    if not test_source and require_tests:
        return _rejected("authored no test_source; a task without tests cannot "
                         "be graded")

    return AuthoredTask(task=task, test_path=test_path, test_source=test_source,
                        chain=chain, chain_index=index,
                        test_path_explicit=test_path_explicit)


def _validate_test_source(source: str, path: str) -> str:
    """Reject test source that will not even parse.

    Cheap and worth doing here: a task whose test cannot be imported fails at
    collection, which reads identically to "the Builder wrote nothing", and the
    real cause is only visible in this generated string.
    """
    return validate_test_source(source, path)


#: Public alias. Stage 2 of the two-stage author validates each generated test
#: on its own, after the breakdown has already been accepted.
def validate_test_source(source: str, path: str = "") -> str:
    """Return "" when *source* is acceptable Python, else a short reason."""
    if len(source) > 200_000:
        return "test source exceeds 200k characters"
    try:
        ast.parse(source)
    except SyntaxError as exc:
        return f"syntax error at line {exc.lineno}: {exc.msg}"
    return ""


def _derive_signature(target_files: Sequence[str], task_id: str) -> str:
    """A stable ``ERR_<DOMAIN>_<NAME>`` key when the model omits one.

    ``steering.py`` clusters escalations on this string, so three tasks dying
    the same way in the same module surface as one design problem instead of
    three unrelated-looking incidents.  Derived from the module under test
    rather than the task id, which is what gives that clustering anything to
    work with.
    """
    for rel in target_files or ():
        stem = Path(str(rel)).stem
        if stem and stem != "__init__":
            domain = "".join(ch for ch in stem.upper() if ch.isalnum()) or "X"
            return f"ERR_{domain[:20]}"
    return f"ERR_{''.join(c for c in str(task_id).upper() if c.isalnum())[:20]}"


def _looks_like_task_draft(value: Any) -> bool:
    """True when a dict is itself one task rather than a container of tasks.

    Cheap models routinely collapse the envelope and answer with the bare task
    they were thinking about.  Observed in production:
    ``{"task_id": "task-001", "steps": [...]}`` -- a *planner*-shaped answer to
    an authoring prompt.  That is still not a usable task (no objective, no
    target files, no test), but recognising it turns a silent
    "authored zero tasks" into a legible per-draft rejection, which is the
    difference between an operator reading a diagnosis and guessing.
    """
    if not isinstance(value, dict):
        return False
    return any(k in value for k in
               ("objective", "title", "target_files", "test_path",
                "test_source", "acceptance_criteria", "steps"))


def coerce_plan(payload: Any, goal: str, existing_ids: Sequence[str],
                project_root: Optional[Path] = None,
                require_tests: bool = True) -> AuthoredPlan:
    """Coerce a model response into an :class:`AuthoredPlan`.

    Tolerant of every shape a real model produces -- a bare list, a dict with
    ``tasks``, a single bare task object, a JSON string -- and strict about the
    individual tasks, which go through ``TaskSpec`` validation.

    ``require_tests=False`` accepts a task whose test has not been written yet,
    which is how the two-stage author works: the breakdown call returns API
    specs only, and one focused call per task writes the test afterwards.
    """
    plan = AuthoredPlan(goal=goal)
    data = payload
    if isinstance(data, str):
        import json
        try:
            data = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            plan.notes.append("response was not valid JSON")
            return plan
    drafts: List[Tuple[str, Any]] = []
    if isinstance(data, dict):
        raw_tasks = data.get("tasks") or data.get("plan")
        if raw_tasks is None and _looks_like_task_draft(data):
            # Envelope collapsed into a single task.  Take it as one draft so it
            # is validated (and rejected, with a reason) rather than vanishing.
            raw_tasks = [data]
        if isinstance(raw_tasks, list):
            for draft in raw_tasks:
                drafts.append((str(draft.get("chain") or "") if isinstance(draft, dict) else "", draft))
        elif raw_tasks is None:
            plan.notes.append(
                "no 'tasks' array in response (keys: "
                f"{sorted(str(k) for k in data)[:6]})")
        else:
            plan.notes.append(
                f"'tasks' was {type(raw_tasks).__name__}, not a list")
    elif isinstance(data, list):
        drafts = [("", d) for d in data]
    else:
        plan.notes.append(f"unusable response type {type(data).__name__}")
        return plan

    if not drafts:
        plan.notes.append("authored zero tasks"
                          + ("" if plan.notes else " (empty tasks array)"))
        return plan

    # Chains: explicit when the model supplied one, otherwise each task is its
    # own chain.  Independent chains are what Phase G partitions into lines, so
    # defaulting to "one task per chain" is the *safe* default: it can never
    # serialise work that could have run in parallel.
    for index, (chain, draft) in enumerate(drafts[:MAX_TASKS]):
        label = chain or f"chain-{index + 1}"
        authored = coerce_task(draft, label, index, existing_ids, project_root,
                               require_tests=require_tests)
        plan.tasks.append(authored)
        if not authored.usable:
            # v2.9.1: name the rule each dropped draft broke.  "task #3 was not
            # a usable draft" told an operator nothing about which of the five
            # authoring rules the model had missed.
            plan.notes.append(f"draft #{index + 1} rejected -- "
                              f"{authored.rejected}")
    if len(drafts) > MAX_TASKS:
        plan.notes.append(f"dropped {len(drafts) - MAX_TASKS} tasks over the "
                          f"{MAX_TASKS}-task cap")
    # Chain indices restart per chain so downstream ordering is per-line.
    per_chain: Dict[str, int] = {}
    for task in plan.tasks:
        task.chain_index = per_chain.get(task.chain, 0)
        per_chain[task.chain] = task.chain_index + 1
    _disambiguate_fallback_test_paths(plan)
    return plan


def _disambiguate_fallback_test_paths(plan: AuthoredPlan) -> None:
    """Give every *derived* test path a distinct name among plan siblings.

    The module-stem fallback names a test after the implementation target, so
    a plan with several tasks building ONE module emits the same path for all
    of them.  ``write_tests`` refuses to overwrite, so tasks 2..N were
    held ("test_path already exists") and silently lost at board fill --
    observed 2026-10-17 on a three-task boardscan plan where every task
    targeted ``tests/test_boardscan.py`` (tasks 2 and 3 never reached the
    board).  First claimant keeps its path; later derived tasks get a
    title-suffixed name (numeral when even that collides).  A path the model
    named explicitly is never rewritten: an explicit duplicate keeps the
    original hard-stop behaviour.
    """
    used: set = set()
    for authored in plan.usable:
        if not authored.test_path:
            continue
        if authored.test_path_explicit:
            used.add(authored.test_path)
            continue
        if authored.test_path in used:
            base = Path(authored.test_path)
            candidate = base.parent / \
                f"{base.stem}_{_slug(authored.task.title)}{base.suffix}"
            if candidate.as_posix() in used:
                counter = 2
                while True:
                    candidate = candidate.with_name(
                        f"{candidate.stem}_{counter}{base.suffix}")
                    if candidate.as_posix() not in used:
                        break
                    counter += 1
            authored.test_path = candidate.as_posix()
            # Keep the task's test path of record in sync with the
            # disambiguated name -- it is this field the critic resolves.
            authored.task.authored_test_path = authored.test_path
            used.add(authored.test_path)
            continue
        used.add(authored.test_path)


# ---------------------------------------------------------------------------
# Static properties of a plan, checked before anything is written
# ---------------------------------------------------------------------------
def chain_file_conflicts(plan: AuthoredPlan) -> List[Tuple[str, str, str]]:
    """``(chain, task_id, path)`` where two chains both claim one file.

    A conflict means two lines could write the same file concurrently, which
    races the snapshot chain.  Reported rather than fixed: the caller decides
    whether to re-partition or to serialise.
    """
    owner: Dict[str, Tuple[str, str]] = {}
    conflicts: List[Tuple[str, str, str]] = []
    for task in plan.usable:
        if task.task is None:
            continue
        for path in (task.task.target_files or []):
            previous = owner.get(path)
            if previous and previous[0] != task.chain:
                conflicts.append((task.chain, task.task.id, path))
            owner.setdefault(path, (task.chain, task.task.id))
    return conflicts


def dependency_cycles(tasks: Sequence[AuthoredTask]) -> List[List[str]]:
    """Cycles among authored task ids. Empty means the plan is schedulable."""
    ids = {t.task.id for t in tasks if t.task is not None}
    graph: Dict[str, List[str]] = {}
    for task in tasks:
        if task.task is None:
            continue
        graph[task.task.id] = [d for d in (task.task.dependencies or [])
                               if d in ids]
    cycles: List[List[str]] = []
    state: Dict[str, int] = {}

    def visit(node: str, path: List[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            if node in path:
                cycles.append(path[path.index(node):] + [node])
            return
        state[node] = 1
        for nxt in graph.get(node, []):
            visit(nxt, path + [node])
        state[node] = 2

    for node in graph:
        visit(node, [])
    return cycles


def dangling_dependencies(tasks: Sequence[AuthoredTask]) -> List[Tuple[str, str]]:
    """``(task_id, dep)`` for dependencies on ids that do not exist.

    A dangling dependency silently blocks a task forever -- it never becomes
    eligible and nothing reports why, which is the starvation shape the
    circuit breaker has to catch.  Better to surface it at authoring time.
    """
    ids = {t.task.id for t in tasks if t.task is not None}
    out: List[Tuple[str, str]] = []
    for task in tasks:
        if task.task is None:
            continue
        for dep in (task.task.dependencies or []):
            if dep not in ids:
                out.append((task.task.id, dep))
    return out


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------
AUTHOR_SYSTEM = (
    "You turn a one-paragraph product goal into a task breakdown. "
    "You do not write implementation code, and you do not write the test "
    "files -- a separate step does that from your API spec.\n\n"
    "Rules that matter more than they look:\n"
    "1. ONE implementation module per task. Never two tasks that both write the "
    "same file -- concurrent lines would race.\n"
    "2. Group tasks that build on each other into a chain, and give independent "
    "chains different labels. Independent chains are what allows the work to run "
    "in parallel.\n"
    "3. Specify the API EXACTLY: every function name, parameter list, return "
    "shape and edge-case rule, spelled exactly. The test is generated from this "
    "spec, so a disagreement here is the single most expensive mistake in this "
    "system.\n"
    "4. The first task in a chain must depend on nothing. Later tasks name the "
    "ids of tasks they build on.\n\n"
    "Respond with ONE JSON object and nothing else:\n"
    '{"tasks": [{"id": "T-001", "chain": "core", "title": "short label",\n'
    '  "objective": "exact API and behaviour required",\n'
    '  "target_files": ["src/pkg/module.py"],\n'
    '  "dependencies": [],\n'
    '  "acceptance_criteria": ["concrete, checkable"]}]}\n\n'
    "Worked example of the exact shape -- copy this structure:\n"
    '{"tasks": [{"id": "T-001", "chain": "core",\n'
    '  "title": "Sliding-window mean",\n'
    '  "objective": "Implement rolling_mean(values: list[float], window: int) '
    '-> float in src/stats/rolling.py. Returns the arithmetic mean of the last '
    'window values. Raises ValueError when window <= 0 or window > len(values).",\n'
    '  "target_files": ["src/stats/rolling.py"],\n'
    '  "dependencies": [],\n'
    '  "acceptance_criteria": ["tests/test_rolling.py is collected and green",\n'
    '   "ValueError raised for window <= 0"]}]}'
)

AUTHOR_USER = (
    "Break this goal into tasks.\n"
    "<<<GOAL>>>\n{goal}\n"
    "Respond with one JSON object containing a 'tasks' array. No commentary."
)

# ---------------------------------------------------------------------------
# Stage 2: the test file, one focused call per task (v2.9.1)
# ---------------------------------------------------------------------------
#: Why the author is split in two.  Asking one call for a multi-task breakdown
#: *with every test file inlined* is the same structural mistake content-mode
#: fixed for Builders: it demands a large, nested, escaped-code payload out of a
#: small model in one shot.  Measured on the observed free roster, every
#: candidate failed it -- one returned a list of English sentences, one returned
#: ``{"tasks": [{}]}``, one returned test bodies with the function names
#: stripped (``def:``), and two held the call for the full 360 s budget.
#:
#: The Builder's own hard-won lesson applies verbatim: small, focused requests
#: are what make cheap models viable.  So stage 1 asks only for API specs (a
#: small JSON object, which small models do follow), and stage 2 makes one
#: plain code request per task.  Each test is then independently cached,
#: independently retryable, and independently droppable -- so one bad test no
#: longer costs the whole plan.
TEST_SYSTEM = (
    "You write ONE pytest file for ONE Python module that does not exist yet. "
    "You write tests only; never implementation.\n\n"
    "Rules:\n"
    "1. Start by guarding for the missing module, exactly like this:\n"
    "   import importlib.util, pytest\n"
    "   if importlib.util.find_spec('src.PKG.MODULE') is None:\n"
    "       pytest.skip('implementation not written yet', "
    "allow_module_level=True)\n"
    "   from src.PKG.MODULE import NAME1, NAME2\n"
    "2. Every test function must be named: def test_something(). Never write a "
    "bare `def:` with no name -- that is a syntax error and it is the single "
    "most common way a generated test file fails.\n"
    "3. Import exactly the names the API spec lists. Do not invent extras.\n"
    "4. Cover the happy path, each stated edge case, and each typed error.\n"
    "5. No network. No fixtures. Plain asserts.\n\n"
    "Reply with ONE fenced block and nothing else:\n"
    "```python path=tests/test_module.py\n<the complete file>\n```"
)


def build_test_prompt(task: TaskSpec, test_path: str) -> str:
    """Stage-2 prompt: the complete test file for one already-specified task."""
    targets = ", ".join(task.target_files) or "(unspecified)"
    criteria = "\n".join(f"- {c}" for c in (task.acceptance_criteria or [])) \
        or "- the objective above"
    return (
        "Write the test file for this task.\n"
        f"<<<TASK_ID>>>\n{untrusted(task.id)}\n"
        f"<<<IMPLEMENTATION_MODULE>>>\n{untrusted(targets)}\n"
        f"<<<TEST_PATH>>>\n{untrusted(test_path)}\n"
        f"<<<API_SPEC>>>\n{untrusted(task.objective)}\n"
        f"<<<ACCEPTANCE_CRITERIA>>>\n{untrusted(criteria)}\n"
        f"<<<DEPENDENCIES>>>\n"
        f"{untrusted(', '.join(task.dependencies) or '(none)')}\n"
        "Emit the one fenced python block now."
    )


def extract_test_source(text: str, path: str) -> str:
    """Pull the complete file out of a stage-2 reply.

    Accepts a ```` ```python path=... ```` fenced block (the same shape the
    Builder's content-mode consumes), a bare fenced block, or a reply that is
    already just source.  Returns "" when nothing usable is present, which the
    caller reports as that task's rejection rather than failing the plan.
    """
    import re as _re
    blocks = _re.findall(r"```(?:python|py)?[^\n`]*\n(.*?)```",
                         text, _re.DOTALL)
    if blocks:
        # Prefer a block whose path header matches the requested file.
        for block in blocks:
            if path and path.split("/")[-1] in block[:400]:
                return block.strip("\n")
        return blocks[0].strip("\n")
    stripped = text.strip()
    if not stripped:
        return ""
    # No fence at all: accept it only if it actually looks like Python.
    if _re.search(r"^\s*(import |from |def test_|@pytest)", stripped,
                  _re.MULTILINE):
        return stripped
    return ""

#: v2.9.1 -- appended verbatim on a retry so a second attempt is not a replay.
#: Without it the retry prompt is byte-identical to the first, the response
#: cache returns the *same* unusable answer, and the loop burns its whole
#: budget re-reading one failure (observed: 10 minutes, one cached reply).
AUTHOR_REPAIR_BLOCK = (
    "Your previous answer was rejected. Do not repeat it.\n"
    "<<<PREVIOUS_ATTEMPT_NOTES>>>\n{notes}\n"
    "Re-read the required JSON shape in the system prompt and answer again "
    "with exactly that object."
)


def build_author_prompt(goal: str, existing_ids: Sequence[str],
                        context: Optional[str] = None,
                        max_tasks: int = MAX_TASKS,
                        repair: Optional[Sequence[str]] = None) -> str:
    """The authoring prompt, with the existing backlog so ids do not collide."""
    taken = ", ".join(str(i) for i in existing_ids) or "(none)"
    blocks = [AUTHOR_USER.format(goal=untrusted(goal))]
    if context:
        blocks.append(f"<<<PROJECT_CONTEXT>>>\n{untrusted(context[:6000])}")
    notes = [str(n) for n in (repair or ()) if str(n).strip()]
    if notes:
        blocks.append(AUTHOR_REPAIR_BLOCK.format(
            notes=untrusted("\n".join(f"- {n}" for n in notes[:6]))))
    blocks.append(f"Existing task ids (do not reuse): {taken}")
    blocks.append(f"Produce at most {max_tasks} tasks.")
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Writing the plan to disk
# ---------------------------------------------------------------------------
def write_tests(plan: AuthoredPlan, project_root: Path,
                dry_run: bool = True) -> List[str]:
    """Write each task's test file. Returns the paths written.

    Written *before* the tasks become eligible, which is the whole mechanism:
    ``TestIntegrityGuard`` snapshots protected paths when a task starts, so a
    test that already exists at that moment is frozen against the Builder.

    Never overwrites.  A refusal here is a hard stop rather than a warning --
    silently replacing an authored test would reintroduce exactly the
    contract-drift this design exists to prevent.
    """
    root = Path(project_root)
    written: List[str] = []
    for task in plan.usable:
        if not task.test_path:
            continue
        path = root / task.test_path
        if path.exists():
            task.rejected = (f"test_path already exists: {task.test_path}; "
                             f"refusing to overwrite an authored test")
            continue
        if dry_run:
            written.append(task.test_path)
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(task.test_source, encoding="utf-8")
            written.append(task.test_path)
        except OSError as exc:
            task.rejected = f"could not write {task.test_path}: {exc}"
    return written


def board_entries(plan: AuthoredPlan) -> List[Dict[str, Any]]:
    """The board-ready dicts for the usable tasks, dependencies resolved.

    Dependencies are rewritten to reference *authored* ids: the model may have
    named a task that does not exist (a dangling dependency blocks a task
    forever and reports nothing), so cross-chain references are dropped and
    intra-chain ordering is re-derived from the chain index.
    """
    by_chain: Dict[str, List[AuthoredTask]] = plan.chains()
    entries: List[Dict[str, Any]] = []
    for chain, group in by_chain.items():
        ids = [t.task.id for t in group if t.task is not None]
        for position, task in enumerate(group):
            if task.task is None:
                continue
            entry = task.task.model_dump()
            # Position within the chain is the dependency we trust; anything
            # the model asserted about ids outside this chain is discarded.
            entry["dependencies"] = ids[:position]
            entries.append(entry)
    return entries


def needs_conftest(plan: AuthoredPlan, project_root: Path) -> bool:
    """Whether the project needs a root conftest for imports to resolve.

    Without one, ``python -m pytest`` works only because the CWD lands on
    ``sys.path``; a bare ``pytest``, or any invocation from elsewhere, makes
    every module skip its tests and report ``6 skipped`` while verifying
    nothing.  That is not hypothetical -- it is exactly what ``logtriage`` did
    until it was fixed.
    """
    return not (Path(project_root) / "conftest.py").exists()
