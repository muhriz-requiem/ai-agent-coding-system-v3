"""v2.9 Phase G: partition a board into independent lines.

Line assignment is **graph colouring**, not a model decision.  Two tasks may sit
on the same line only if they cannot conflict, and the two ways they can conflict
are both knowable from the board alone:

* **shared target files** -- two lines writing one file would race the snapshot
  chain, and the loser's diff would be applied on top of a tree it never saw;
* **dependency edges** -- a line must not start work whose input does not exist
  yet.

Both are decided here, deterministically, for free.  Spending a model call to
work out a partition that arithmetic already determines would be spending the
scarce resource on a question with an exact answer.

The measured justification, from the two real boards in this workspace:

* ``logtriage`` -- 6 tasks, 6 distinct files, 0 shared. Fully parallel. A 2x
  wall-clock win with no serialisation at all.
* ``project_main`` -- 20 tasks, 21 files, **3 shared files**
  (``src/config.py``, ``src/cli.py``, ``src/pipeline/smoke.py``). Those pairs
  must serialise, and colour placement is what enforces it rather than luck.

RPM is the binding constraint on this account (217 genuine 429s in one
reference build), so line count defaults to 1: today's behaviour exactly, with
the capability present and off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple



@dataclass
class Line:
    """One chain of tasks that may run in sequence without conflicting."""

    index: int
    task_ids: List[str] = field(default_factory=list)
    #: ``task_id -> ids this line must finish first``.
    internal_deps: Dict[str, List[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.task_ids)


@dataclass
class Partition:
    """A board split into lines, with the reasoning that produced it."""

    lines: List[Line] = field(default_factory=list)
    #: Tasks deliberately placed on one line because they share a file.
    serialised: List[Tuple[str, str, str]] = field(default_factory=list)
    #: Human-readable notes; never silently swallowed.
    notes: List[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.lines)

    def line_for(self, task_id: str) -> Optional[int]:
        for line in self.lines:
            if task_id in line.task_ids:
                return line.index
        return None

    def summary(self) -> Dict[str, Any]:
        return {"lines": self.count,
                "sizes": [len(line) for line in self.lines],
                "serialised": len(self.serialised),
                "notes": self.notes[:4]}


def _target_files(task: Dict[str, Any]) -> Set[str]:
    """Files a task will write.

    Test files are excluded.  Two tasks may both have tests without conflicting,
    because each authored test is written once at authoring time and then frozen
    -- test *content* is never written concurrently by two Builders.
    """
    out: Set[str] = set()
    for raw in (task.get("target_files") or []):
        rel = str(raw).strip()
        if not rel:
            continue
        parts = Path(rel).as_posix().split("/")
        if parts[0] == "tests" or any(p == "tests" for p in parts[:-1]):
            continue
        out.add(rel)
    return out


def partition(board_tasks: Sequence[Dict[str, Any]],
              max_lines: int = 1) -> Partition:
    """Colour *board_tasks* into at most *max_lines* conflict-free lines.

    Deterministic and order-stable: given the same board it always produces the
    same partition, because a partition that changed between runs would make
    every log comparison useless.

    The algorithm, and why each step is there:

    1. **Union shared files.**  Two tasks writing the same file must land in the
       same connected component.  Transitive closure, so a chain of tasks
       sharing files is held together even when its ends do not overlap.
    2. **Topologically order each component**, breaking ties by board order so
       the result is stable and reproducible.
    3. **Distribute components across lines** largest-first, respecting
       ``max_lines``.

    Components -- not individual tasks -- are the unit of placement.  A component
    is conflict-free by construction, so *any* assignment of components to lines
    is conflict-free too, and nothing downstream has to re-check for conflicts.

    **Dependencies do not constrain placement, only ordering.**  A dependency
    edge may cross lines; the second task simply waits, which the scheduler's
    existing dependency gate already enforces.  Treating a dependency as a
    placement constraint is a category error, and it is what made the first
    version of this function collapse a fully parallel board into one line.
    """
    result = Partition()
    if not board_tasks:
        result.notes.append("no tasks to partition")
        return result
    if max_lines < 1:
        max_lines = 1

    index_by_id: Dict[str, int] = {}
    ids: List[str] = []
    for position, task in enumerate(board_tasks):
        task_id = str(task.get("id") or f"__idx{position}")
        index_by_id[task_id] = position
        ids.append(task_id)

    parent: Dict[str, str] = {task_id: task_id for task_id in ids}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 1. Shared files.
    file_owner: Dict[str, str] = {}
    for task_id, task in zip(ids, board_tasks):
        for path in sorted(_target_files(task)):
            previous = file_owner.get(path)
            if previous is None:
                file_owner[path] = task_id
            else:
                result.serialised.append((task_id, previous, path))
                union(task_id, previous)

    # 2. Dependencies do NOT union.
    #
    #    The first version of this function did, and it collapsed the
    #    ``logtriage`` board -- 6 tasks, 6 distinct files, *zero* shared -- into
    #    a single line.  That is the whole opportunity thrown away, and the
    #    reason is a category error: a dependency constrains *ordering*, not
    #    *placement*.  Two tasks joined by an edge may sit on different lines;
    #    the second simply waits, which the scheduler's existing dependency gate
    #    already does via ``get_next_pending_task``.
    #
    #    Only a shared file is a true conflict, because it is the only case where
    #    two *running* Builders touch the same bytes.
    #
    #    Unioning on dependencies would also make parallelism impossible on any
    #    realistically-ordered board, since a chain of N tasks would always
    #    collapse to one line.

    # 3. Components, ordered by board position.
    components: Dict[str, List[str]] = {}
    for task_id in ids:
        components.setdefault(find(task_id), []).append(task_id)

    def order_component(members: List[str]) -> List[str]:
        member_set = set(members)
        by_id = {task_id: board_tasks[index_by_id[task_id]]
                 for task_id in members}
        ordered: List[str] = []
        placed: Set[str] = set()
        pending = sorted(members, key=lambda t: index_by_id[t])
        # Kahn's algorithm over the *internal* dependency edges only; a
        # dependency outside the component is another line's business.
        remaining = dict(by_id)
        while pending:
            progressed = False
            for task_id in list(pending):
                deps = {str(d) for d in (remaining[task_id].get("dependencies")
                                         or [])}
                if deps & member_set - placed:
                    continue
                ordered.append(task_id)
                placed.add(task_id)
                pending.remove(task_id)
                progressed = True
            if not progressed:
                # A cycle inside the component.  Emit in board order rather than
                # looping forever; the scheduler will refuse the cycle anyway.
                ordered.extend(pending)
                result.notes.append(
                    f"dependency cycle inside a line: {pending[:3]}")
                break
        return ordered

    ordered_components = [order_component(members)
                          for members in components.values()]
    # Largest first: better balance, and deterministic because the component
    # order comes from board order.
    ordered_components.sort(key=lambda members: (-len(members),
                                                 index_by_id[members[0]]))

    lines = [Line(index=i) for i in range(max_lines)]
    for members in ordered_components:
        target = min(lines, key=lambda line: (len(line), line.index))
        target.task_ids.extend(members)

    # Re-order each line as a whole.
    #
    # Components are ordered internally, but a dependency can point *across*
    # components, and when several components share a line those edges are
    # between them.  Ordering component-by-component therefore produces an
    # invalid sequence whenever such an edge points backwards -- which it did on
    # the real project_main board, where TASK-7011 (an early component) depends
    # on TASK-7010B (a late one).  Ordering the finished line topologically
    # catches every intra-line edge, cross-component or not.
    for line in lines:
        line.task_ids = order_component(line.task_ids)
        members = set(line.task_ids)
        for task_id in line.task_ids:
            task = board_tasks[index_by_id[task_id]]
            internal = sorted({str(d) for d in (task.get("dependencies") or [])}
                              & members)
            if internal:
                line.internal_deps[task_id] = internal

    result.lines = [line for line in lines if line.task_ids]
    if max_lines == 1:
        result.notes.append(
            "LINE_COUNT=1: single sequential line (pre-v2.9 behaviour)")
    if len(ordered_components) > max_lines:
        result.notes.append(
            f"{len(ordered_components)} independent components placed over "
            f"{max_lines} line(s)")
    return result


def validate(partition_result: Partition,
             board_tasks: Sequence[Dict[str, Any]]) -> List[str]:
    """Prove the partition is conflict-free. Returns problems, empty if sound.

    Deliberately re-derives everything from the board rather than trusting the
    partitioner's own bookkeeping.  A partition that is wrong and also
    self-consistent is exactly the failure this check exists to catch, and it is
    cheap enough to run every time.
    """
    problems: List[str] = []
    by_id = {str(t.get("id")): t for t in board_tasks}
    line_of: Dict[str, int] = {}
    for line in partition_result.lines:
        for task_id in line.task_ids:
            if task_id in line_of:
                problems.append(f"{task_id} appears on two lines")
            line_of[task_id] = line.index

    missing = set(by_id) - set(line_of)
    if missing:
        problems.append(f"tasks not placed: {sorted(missing)[:5]}")

    # No two tasks on different lines may share a file.
    owner: Dict[str, Tuple[int, str]] = {}
    for task_id, line_index in sorted(line_of.items()):
        task = by_id.get(task_id)
        if task is None:
            continue
        for path in sorted(_target_files(task)):
            previous = owner.get(path)
            if previous and previous[0] != line_index:
                problems.append(
                    f"{path} is written by {previous[1]} (line "
                    f"{previous[0]}) and {task_id} (line {line_index})")
            owner.setdefault(path, (line_index, task_id))

    # Within a line, a task must not precede its own dependency.
    #
    # A dependency *across* lines is legitimate and expected -- that is what the
    # scheduler's dependency gate is for -- so it is not a problem.  Only the
    # intra-line ordering is the partitioner's responsibility, and getting it
    # wrong means a line tries to build on something it has not produced yet.
    for line in partition_result.lines:
        order = {task_id: position
                 for position, task_id in enumerate(line.task_ids)}
        for task_id in line.task_ids:
            task = by_id.get(task_id)
            if task is None:
                continue
            for dep in (task.get("dependencies") or []):
                dep_id = str(dep)
                if dep_id not in by_id or line_of.get(dep_id) != line.index:
                    continue
                if order.get(dep_id, 0) >= order.get(task_id, 0):
                    problems.append(
                        f"{task_id} precedes its dependency {dep_id} on line "
                        f"{line.index}")
    return problems


def describe(partition_result: Partition) -> str:
    """A compact human-readable rendering, for the manifest and the log."""
    lines = []
    for line in partition_result.lines:
        chain = " -> ".join(line.task_ids)
        lines.append(f"line {line.index} ({len(line)}): {chain}")
    if partition_result.serialised:
        lines.append(
            f"serialised for shared files: {len(partition_result.serialised)} pair(s)")
    for note in partition_result.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)
