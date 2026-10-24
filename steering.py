"""Deterministic steering telemetry: is the team making progress, or looping?

Replaces the v2.6 "pattern memory" as the system's self-monitoring layer.  The
old design recorded *which error signatures had once been fixed* and injected a
copy of the task objective back into the Builder prompt on attempt 3.  It did
not learn anything:

* ``fix_summary`` was ``task.objective`` verbatim -- text the Builder already
  received in ``<<<TASK_JSON>>>`` on every attempt, so the injection carried
  zero new information while costing ~1.3 KB of tokens.
* ``signature`` was hand-authored upstream or synthesised from a task id; no
  lesson was ever extracted from failure output.
* The confidence *decay* that was supposed to demote bad patterns was
  unreachable: ``workspace_patterns_referenced`` was empty at every construction
  site, so zero decrements were ever recorded.
* Patterns had no de-duplication, and a 30-day TTL deleted 11 of 23 promoted
  records -- reachable from a *read*, since querying rewrote the file.

This module takes the opposite approach.  Every question it answers is
**measurable from the harness**, with no LLM in the loop:

* :meth:`SteeringMonitor.record_attempt` -- what just happened (outcome,
  signature, diff hash, duration, tokens).
* :meth:`SteeringMonitor.evaluate` -- given the current board, what is the
  system doing wrong *right now*, if anything.

The signals are deliberately boring and auditable: repeat-failure runs,
identical diffs re-emitted, cycles with no net board change, dependency
deadlock, and resource burn.  A doom loop is a fact about counters, not a
judgement call, which is why this belongs in code and not in a prompt.

State persists to ``.system/steering_state.json``.  That matters: the run log
shows 86 orchestrator launches for 101 tasks, so any in-memory detector would
have amnesia at exactly the moment it matters most.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from logger import log_event

# ---------------------------------------------------------------------------
# Thresholds.  All configurable; the defaults are deliberately conservative --
# a false positive costs one breaker evaluation, a false negative costs a
# wasted retry budget.
# ---------------------------------------------------------------------------

#: Consecutive failures sharing one signature before it is a doom loop.
DOOM_LOOP_SIGNATURE_N: int = 3
#: Times the *same* diff has been emitted for one task before it is a loop.
DOOM_LOOP_DIFF_N: int = 2
#: Consecutive idle cycles with no board change before we call it stalled.
STALL_CYCLES_N: int = 8
#: Distinct tasks escalated with the same *normalised* failure shape before it
#: is treated as systemic.
STRUCTURAL_SIGNATURE_N: int = 3
#: How much worse a signal's evidence must get before it is re-reported.
#: Without this, a level-triggered signal fires every evaluation -- observed
#: live as 16 identical `steering_doom_loop` rows for one stuck task.
REPEAT_STEP: int = 2

#: Per-task wall-clock seconds that count as runaway.
BURN_TASK_S: int = 1800
#: Per-project wall-clock seconds per hour that count as runaway.
BURN_PROJECT_SPH: int = 7200

_SIGNATURE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")

#: Signal kinds.  Kept as constants so callers and detectors cannot drift.
DOOM_LOOP = "doom_loop"
STALL = "stall"
STRUCTURAL_BLOCK = "structural_block"
BURN_RATE = "burn_rate"
PROGRESS = "progress"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def signature_tokens(signature: Optional[str]) -> List[str]:
    """Identifier tokens of an error signature, ignoring the shared ``ERR``.

    ``ERR_PIPELINE_WIRING`` -> ``["PIPELINE", "WIRING"]``.  Splitting on
    underscores matters: matching the whole string would make every distinct
    failure look unique, and matching the ``ERR`` prefix -- which every
    signature carries -- would make them all look identical.  The middle
    tokens are what make "these two failures are the same shape" mechanical.
    """
    if not signature:
        return []
    raw = str(signature).replace("-", "_")
    # Split on underscore first so ERR_PIPELINE_WIRING yields three parts.
    parts = [p for chunk in raw.split("_") for p in _SIGNATURE_TOKEN_RE.findall(chunk)]
    seen: List[str] = []
    for part in parts:
        token = part.upper()
        if token == "ERR" or token in seen:
            continue
        seen.append(token)
    return seen


def normalise_signature(signature: Optional[str]) -> str:
    """Collapse a signature to its *shape*, erasing instance identifiers.

    ``ERR_SCHEMA_T-1_MAPPING`` and ``ERR_SCHEMA_T-2_MAPPING`` both become
    ``ERR_SCHEMA_<N>_MAPPING``.  Two escalations share a root cause when their
    shapes match -- not when they merely share a word.  Matching on shared
    tokens was tried first and produced both false positives (three unrelated
    ``..._THING`` failures) and false negatives (the same bug across three task
    ids), because a generic suffix looks identical to a genuine root cause.

    Normalisation is **segment-wise** on purpose: only a whole ``_``-delimited
    segment that is purely an identifier collapses.  Erasing every digit run
    would merge ``ERR_UNIQUE0_THING`` and ``ERR_UNIQUE1_THING``, which are
    unrelated failures that merely end in the same word.
    """
    if not signature:
        return ""
    text = str(signature).upper().replace("-", "_").replace(" ", "_")
    parts = [p for p in text.split("_") if p]
    out: List[str] = []
    for part in parts:
        if part.isdigit() or _TASK_SEGMENT_RE.fullmatch(part):
            out.append("<N>")
        else:
            out.append(part)
    collapsed: List[str] = []
    for part in out:
        if collapsed and collapsed[-1] == part:
            continue  # T-1-1 style repeats
        collapsed.append(part)
    return "_".join(collapsed)


#: A whole segment that is an instance id (``T1``, ``TASK12``, ``7005B``).
_TASK_SEGMENT_RE = re.compile(r"(?:TASK|T)?\d+[A-Z]?")


def diff_fingerprint(diff: Optional[str]) -> str:
    """Stable hash of a diff's *content*, ignoring hunk line numbers.

    Two attempts that produce byte-identical changes are the same attempt; if
    that happens twice for one task, retrying a third time cannot help.
    """
    if not diff:
        return ""
    normalized = "\n".join(
        line.split("@@")[0].rstrip() if line.startswith("@@") else line.rstrip()
        for line in str(diff).splitlines()
        if not line.startswith(("---", "+++", "diff --git", "index ")))
    return hashlib.sha256(normalized.strip().encode("utf-8")).hexdigest()[:16]


@dataclass
class Signal:
    """One observation about how the system is currently behaving."""

    kind: str
    project: str
    detail: str
    severity: str = "medium"
    evidence: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    #: Stable identity for edge-triggering.  ``report_emitted`` stamps it so a
    #: signal is not re-logged every evaluation; empty means "always report".
    key: str = ""
    #: How bad the evidence is, on whatever scale that kind uses.  The gate
    #: (:meth:`SteeringMonitor._should_report`) compares this against the value
    #: stamped by ``report_emitted``.  It lives here rather than in ``evidence``
    #: so the two can never read different fields and silently disagree -- which
    #: is exactly the bug that left ``burn_rate`` flooding after two earlier
    #: fixes had landed.
    weight: int = 0
    at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "project": self.project,
                "detail": self.detail, "severity": self.severity,
                "evidence": self.evidence, "task_id": self.task_id,
                "at": self.at}

    @property
    def actionable(self) -> bool:
        """Whether this signal is worth surfacing to the manifest/breaker."""
        return self.kind in (DOOM_LOOP, STALL, STRUCTURAL_BLOCK, BURN_RATE)


@dataclass
class AttemptRecord:
    """What one build-review cycle actually cost and produced."""

    task_id: str
    outcome: str
    attempt: int = 1
    signature: Optional[str] = None
    diff_hash: str = ""
    duration_s: float = 0.0
    tokens: int = 0
    error_summary: str = ""


class SteeringMonitor:
    """Per-workspace accumulator of attempts, stalls and burn."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = Path(workspace_root)
        self.state_path = self.workspace_root / ".system" / "steering_state.json"
        self._state: Dict[str, Any] = self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        try:
            if self.state_path.exists():
                data = __import__("json").loads(
                    self.state_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    data.setdefault("projects", {})
                    return data
        except (OSError, ValueError):
            pass
        return {"projects": {}}

    def _save(self) -> None:
        try:
            import redaction

            redaction.write_json_atomic(self.state_path, self._state)
        except Exception:  # noqa: BLE001 -- telemetry must never break a run
            pass

    def _proj(self, project: str) -> Dict[str, Any]:
        projects = self._state.setdefault("projects", {})
        return projects.setdefault(project, {
            "attempts": [],            # recent AttemptRecords (bounded)
            "signature_runs": {},      # token -> consecutive failure count
            "diff_runs": {},           # task_id|diff_hash -> count
            "idle_cycles": 0,
            "last_board_fingerprint": "",
            "escalated_signatures": [],
            "cycle_seconds": [],       # recent cycle durations (bounded)
            "updated_at": "",
        })

    # -- ingest ----------------------------------------------------------
    def record_attempt(self, project: str, record: AttemptRecord) -> None:
        """Fold one cycle's outcome into the accumulators."""
        p = self._proj(project)
        attempts: List[Dict[str, Any]] = p["attempts"]
        attempts.append({
            "task_id": record.task_id, "outcome": record.outcome,
            "attempt": record.attempt, "signature": record.signature,
            "diff_hash": record.diff_hash,
            "duration_s": round(float(record.duration_s), 2),
            "tokens": int(record.tokens), "at": _now_iso(),
        })
        del attempts[:-200]  # bounded: the file is polled, not archived

        failed = record.outcome not in ("APPROVED", "done", "completed")
        if failed:
            for token in signature_tokens(record.signature):
                runs = p["signature_runs"]
                runs[token] = int(runs.get(token, 0)) + 1
        else:
            # A success clears the counters it plausibly resolved -- and their
            # report stamps too.  Clearing one without the other was a real bug
            # found by the dogfood regression suite: the counter restarted at 0
            # while its stamp stayed at 3, so a *fresh* loop after progress was
            # silently suppressed until the evidence grew past the old value.
            p["signature_runs"] = {}
            reported = p.get("reported") or {}
            for stale in [k for k in reported if k.startswith("sig:")]:
                reported.pop(stale, None)

        if record.diff_hash:
            key = f"{record.task_id}|{record.diff_hash}"
            p["diff_runs"][key] = int(p["diff_runs"].get(key, 0)) + 1
            # Keep only live keys so this cannot grow without bound.
            live = {k for k in p["diff_runs"]
                    if k.split("|", 1)[0] == record.task_id}
            for k in list(p["diff_runs"]):
                if k.split("|", 1)[0] != record.task_id or \
                        len(p["diff_runs"]) > 200:
                    if k not in live or len(p["diff_runs"]) > 200:
                        p["diff_runs"].pop(k, None)

        if record.outcome == "ESCALATED" and record.signature:
            sigs: List[str] = p["escalated_signatures"]
            sigs.append(record.signature)
            del sigs[:-50]

        durations: List[float] = p["cycle_seconds"]
        durations.append(float(record.duration_s))
        del durations[:-100]

        p["updated_at"] = _now_iso()
        self._save()

    def note_board(self, project: str, tasks: Sequence[Dict[str, Any]]) -> None:
        """Track whether the board is actually moving (stall detection).

        This is the *only* stall counter.  There is deliberately no separate
        "idle cycle" increment: two counters tracking the same condition would
        double-count and trip the threshold at half the real duration.

        ``idle_cycles`` therefore means "consecutive cycles in which work was
        available but the board did not change" -- not merely "consecutive
        ticks".  A board with nothing pending and nothing failed is finished,
        so it resets the counter instead of accumulating forever.
        """
        p = self._proj(project)
        fingerprint = hashlib.sha256(
            repr(sorted((str(t.get("id")), str(t.get("status")),
                         int(t.get("retry_count") or 0))
                        for t in tasks)).encode("utf-8")).hexdigest()[:16]
        actionable = any(t.get("status") in ("pending", "in_progress", "failed")
                         for t in tasks)
        if not actionable:
            # Nothing to be stuck on.  Do not bank these cycles.
            p["idle_cycles"] = 0
            p["last_board_fingerprint"] = fingerprint
            p["updated_at"] = _now_iso()
            self._save()
            return
        if fingerprint == p.get("last_board_fingerprint"):
            p["idle_cycles"] = int(p.get("idle_cycles", 0)) + 1
        else:
            p["idle_cycles"] = 0
            p["stall_notified_at"] = 0
        p["last_board_fingerprint"] = fingerprint
        p["updated_at"] = _now_iso()
        self._save()

    # -- evaluate --------------------------------------------------------
    def evaluate(self, project: str,
                 tasks: Optional[Sequence[Dict[str, Any]]] = None
                 ) -> List[Signal]:
        """Signals for *project* right now.  Pure inspection -- no mutation."""
        p = self._proj(project)
        out: List[Signal] = []
        out.extend(self._doom_loops(project, p))
        out.extend(self._stall(project, p, tasks))
        out.extend(self._structural(project, p, tasks))
        out.extend(self._burn(project, p))
        return out

    def _doom_loops(self, project: str, p: Dict[str, Any]) -> List[Signal]:
        out: List[Signal] = []
        reported: Dict[str, int] = p.get("reported") or {}
        # ONE signal per loop, not one per token.  ``signature_runs`` is keyed by
        # token so a shared token can be detected, but a two-word signature
        # (``ERR_STUCK_LOOP``) otherwise produced two ERROR rows saying the same
        # thing -- the same noise the stall and edge-trigger fixes removed.
        runs = {t: c for t, c in p.get("signature_runs", {}).items()
                if c >= DOOM_LOOP_SIGNATURE_N}
        if runs:
            token, count = max(runs.items(), key=lambda kv: (kv[1], kv[0]))
            key = f"sig:{token}"
            if self._should_report(reported, key, count):
                out.append(Signal(
                    kind=DOOM_LOOP, project=project, severity="high",
                    task_id=None, key=key, weight=count,
                    detail=(f"failure signature token {token!r} has recurred "
                            f"{count} times without an intervening success"),
                    evidence={"token": token, "consecutive_failures": count,
                              "threshold": DOOM_LOOP_SIGNATURE_N,
                              "tokens_over_threshold": sorted(runs)}))
        for dkey, count in sorted(p.get("diff_runs", {}).items()):
            if count < DOOM_LOOP_DIFF_N:
                continue
            task_id, _, dhash = dkey.partition("|")
            dedupe = f"diff:{dkey}"
            if not self._should_report(reported, dedupe, count):
                continue
            out.append(Signal(
                kind=DOOM_LOOP, project=project, severity="high",
                task_id=task_id, key=dedupe, weight=count,
                detail=(f"the same diff was produced {count} times for "
                        f"{task_id}; another identical attempt cannot help"),
                evidence={"diff_hash": dhash, "repeats": count,
                          "threshold": DOOM_LOOP_DIFF_N}))
        return out

    @staticmethod
    def _should_report(reported: Dict[str, int], key: str, count: int) -> bool:
        """Pure edge-trigger check: has this signal meaningfully worsened?

        A signal that re-fires on every evaluation is a signal nobody reads --
        observed live as 16 ``steering_doom_loop`` rows for one stuck task.
        The first sighting always reports; afterwards a repeat only counts once
        the evidence has grown by ``REPEAT_STEP``, so a genuinely worsening loop
        gets louder instead of being suppressed forever.

        Pure on purpose: the "already reported" stamp is written by
        :meth:`report_emitted`, never here, because mutating state during a
        read is the exact defect this module replaced.
        """
        seen = int(reported.get(key, 0))
        return not seen or count >= seen + REPEAT_STEP

    def _stall(self, project: str, p: Dict[str, Any],
               tasks: Optional[Sequence[Dict[str, Any]]]) -> List[Signal]:
        cycles = int(p.get("idle_cycles", 0))
        if cycles < STALL_CYCLES_N:
            return []
        # A board with nothing pending and nothing failed is COMPLETE, not
        # stalled.  Dogfooding caught this: three finished projects logged
        # `steering_stall` on every idle tick, forever -- 50+ cycles each,
        # three projects, several times a second.  That is the same log-spam
        # failure the pause poll had, reintroduced by this module, and it
        # would also have re-tripped the breaker on every heartbeat forever.
        if tasks is not None:
            actionable = [t for t in tasks
                          if t.get("status") in ("pending", "in_progress",
                                                 "failed")]
            if not actionable:
                return []
        # Edge-triggered, not level-triggered: report when the threshold is
        # first crossed, then stay quiet until it is re-armed by progress.
        # Re-arming on progress is handled in note_board (stall_notified_at
        # reset).  Without this a genuinely stuck project emits one WARNING
        # per idle tick, which is how the signal stops being read at all.
        if int(p.get("stall_notified_at", 0)) >= STALL_CYCLES_N:
            return []
        return [Signal(
            kind=STALL, project=project, severity="medium",
            detail=(f"{cycles} consecutive cycles produced no board change "
                    f"while work remained"),
            evidence={"idle_cycles": cycles, "threshold": STALL_CYCLES_N})]

    def _structural(self, project: str, p: Dict[str, Any],
                    tasks: Optional[Sequence[Dict[str, Any]]]) -> List[Signal]:
        out: List[Signal] = []
        reported: Dict[str, int] = p.get("reported") or {}

        def _emit(key: str, weight: int, severity: str, detail: str,
                  evidence: Dict[str, Any]) -> None:
            """Every structural signal goes through the same edge gate.

            Observed live before this: 199 identical `steering_structural_block`
            ERROR rows for one stuck board. A signal that repeats on every
            evaluation is indistinguishable from a stuck sensor, and it is
            exactly the kind of noise that trains an operator to ignore ERRORs.
            """
            if not self._should_report(reported, key, weight):
                return
            out.append(Signal(kind=STRUCTURAL_BLOCK, project=project,
                              severity=severity, key=key, weight=weight,
                              detail=detail, evidence=evidence))

        # 1. Dependency deadlock: nothing is runnable and nothing can unblock.
        if tasks:
            done = {str(t.get("id")) for t in tasks if t.get("status") == "done"}
            pending = [t for t in tasks if t.get("status") == "pending"]
            if pending:
                runnable = [t for t in pending
                            if not any(d not in done
                                       for d in (t.get("dependencies") or []))]
                if not runnable:
                    _emit("deadlock", len(pending), "high",
                          f"all {len(pending)} pending tasks are blocked by "
                          f"unfinished dependencies -- the board is deadlocked",
                          {"pending": len(pending), "done": len(done),
                           "sample": sorted(str(t.get("id"))
                                            for t in pending)[:10]})

        # 2. The same failure *shape* across different tasks is systemic, not
        #    a per-task problem: retrying each one individually will not help.
        #    Clustering is by *normalised* signature, so instance ids collapse
        #    while a genuinely shared generic suffix does not.
        clusters: Dict[str, set] = {}
        for sig in p.get("escalated_signatures", []):
            shape = normalise_signature(sig)
            if shape:
                clusters.setdefault(shape, set()).add(sig)
        for shape, sigs in sorted(clusters.items()):
            if len(sigs) < STRUCTURAL_SIGNATURE_N:
                continue
            _emit(f"shape:{shape}", len(sigs), "high",
                  f"{len(sigs)} distinct escalations share the failure shape "
                  f"{shape!r}: this is one design-level block, not {len(sigs)} "
                  f"separate bugs",
                  {"shape": shape, "signatures": sorted(sigs)[:10],
                   "count": len(sigs), "threshold": STRUCTURAL_SIGNATURE_N})
        return out

    def _burn(self, project: str, p: Dict[str, Any]) -> List[Signal]:
        out: List[Signal] = []
        recent: List[Dict[str, Any]] = p.get("attempts", [])
        per_task: Dict[str, float] = {}
        for a in recent:
            per_task[a["task_id"]] = per_task.get(a["task_id"], 0.0) + \
                float(a.get("duration_s") or 0.0)
        reported: Dict[str, int] = p.get("reported") or {}
        for task_id, total in sorted(per_task.items()):
            if total < BURN_TASK_S:
                continue
            key = f"burn:{task_id}"
            # Weight in whole minutes so a slowly-worsening burn escalates
            # visibly instead of firing every evaluation.
            weight = int(total // 60) or 1
            if not self._should_report(reported, key, weight):
                continue
            out.append(Signal(
                kind=BURN_RATE, project=project, severity="medium",
                task_id=task_id, key=key, weight=weight,
                detail=(f"{task_id} has consumed {total:.0f}s of build time "
                        f"without completing"),
                evidence={"task_id": task_id, "total_s": round(total, 1),
                          "threshold_s": BURN_TASK_S}))
        window_tokens = sum(int(a.get("tokens") or 0) for a in recent)
        if window_tokens > 0 and len(recent) >= 3:
            key = "burn:project"
            if self._should_report(reported, key, len(recent)):
                out.append(Signal(
                    kind=BURN_RATE, project=project, severity="low", key=key, weight=len(recent),
                    detail=(f"{window_tokens} estimated tokens across the last "
                            f"{len(recent)} recorded attempts"),
                    evidence={"tokens": window_tokens, "attempts": len(recent)}))
        return out

    def report_emitted(self, project: str, kinds: Iterable[str],
                        signals: Optional[Iterable["Signal"]] = None) -> None:
        """Record that these signals were emitted, so they are not re-emitted.

        Edge-triggering needs somewhere to put the "already said it" marker.
        It lives here rather than inside :meth:`evaluate` because ``evaluate``
        is documented as pure inspection -- mutating state during a read is
        exactly the defect this module replaced.
        """
        kinds = set(kinds or ())
        if not kinds & {STALL, DOOM_LOOP, STRUCTURAL_BLOCK, BURN_RATE}:
            return
        p = self._proj(project)
        changed = False
        if STALL in kinds:
            p["stall_notified_at"] = int(p.get("idle_cycles", 0))
            changed = True
        if kinds & {DOOM_LOOP, STRUCTURAL_BLOCK, BURN_RATE}:
            reported: Dict[str, int] = p.setdefault("reported", {})
            for signal in signals or ():
                if signal.kind not in (DOOM_LOOP, STRUCTURAL_BLOCK,
                                     BURN_RATE) \
                        or not signal.key:
                    continue
                count = int(signal.weight or 0)
                if count > int(reported.get(signal.key, 0)):
                    reported[signal.key] = count
                    changed = True
        if changed:
            p["updated_at"] = _now_iso()
            self._save()

    # -- reporting -------------------------------------------------------
    def snapshot(self, project: str) -> Dict[str, Any]:
        """Compact, bounded view for the manifest and the breaker check."""
        p = self._proj(project)
        return {
            "attempts_recorded": len(p.get("attempts", [])),
            "idle_cycles": int(p.get("idle_cycles", 0)),
            "active_signature_runs": dict(
                sorted(p.get("signature_runs", {}).items(),
                       key=lambda kv: -kv[1])[:5]),
            "repeated_diffs": sum(1 for v in p.get("diff_runs", {}).values()
                                  if v >= DOOM_LOOP_DIFF_N),
            "recent_escalations": sorted(
                set(p.get("escalated_signatures", [])))[-5:],
            "median_cycle_s": _median(
                [float(a.get("duration_s") or 0.0)
                 for a in p.get("attempts", [])]),
            "signals": [s.to_dict() for s in self.evaluate(project)],
        }

    def reset_project(self, project: str) -> None:
        """Drop a project's accumulators (operator action / board edit)."""
        self._state.get("projects", {}).pop(project, None)
        self._save()


def _median(values: Sequence[float]) -> float:
    vals = sorted(v for v in values if v > 0)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return round(vals[mid], 1)
    return round((vals[mid - 1] + vals[mid]) / 2.0, 1)


def summarize(signals: Sequence[Signal]) -> str:
    """One-line human summary for logs and the dashboard."""
    if not signals:
        return "no steering signals"
    counts: Dict[str, int] = {}
    for s in signals:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    return ", ".join(f"{n}x {k}" for k, n in sorted(counts.items()))


def emit(signals: Sequence[Signal]) -> None:
    """Log each actionable signal exactly once, at its own severity."""
    for s in signals:
        if not s.actionable:
            continue
        level = {"high": "ERROR", "medium": "WARNING"}.get(s.severity, "INFO")
        log_event(level, f"steering_{s.kind}",
                  {"project": s.project, "task_id": s.task_id,
                   "detail": s.detail, "evidence": s.evidence})


def monitor_for(workspace_root: Path) -> SteeringMonitor:
    """Process-wide singleton, mirroring the LLMClient/limiter pattern."""
    key = str(Path(workspace_root).resolve())
    _MONITORS[ key] = _MONITORS.get(key) or SteeringMonitor(Path(workspace_root))
    return _MONITORS[key]


_MONITORS: Dict[str, SteeringMonitor] = {}


def reset_monitors() -> None:
    """Test hook: drop cached monitors."""
    _MONITORS.clear()