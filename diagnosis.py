"""v2.9 Phase D: diagnosis for escalated tasks.

Escalation used to be the end of the line.  A task that spent its retry budget
became a human problem carrying a stack trace, and ``_try_auto_requeue`` then
handed the identical objective back to the Builder -- observed doing exactly
that seven times in the logtriage build.

The reason is structural rather than a missing feature: the escalator recorded
*that* a task failed and nothing about *why*, so the retry had no new
information to act on.  This module supplies the missing half.  One call per
escalation, on data the loop already collected, producing a cause and a remedy.

The interesting constraint is authority.  A diagnosis is produced by a model
that also had a hand in the work that failed, so its remedies are confined to
the same board-key whitelist the Council's actions were bound by
(:data:`validation.BOARD_WRITE_ALLOWLIST`) and anything outside it is inert.
And the ``cause`` string is fed to the steering monitor, which is what turns
three separate ``spec_ambiguous`` escalations into one visible systemic block
instead of three silent failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
from validation import TaskSpec, board_keys_for, is_executable_remedy

#: The diagnosis prompt.  Deliberately narrow: it names the two closed
#: vocabularies it must choose from, because a free-form cause cannot be
#: clustered by the steering monitor and a free-form remedy cannot be checked
#: against the board-write whitelist.
SYSTEM = (
    "You diagnose why an automated coding task failed, using only the evidence "
    "provided. You did not write this code and you cannot run anything.\n"
    "Separate the two failure classes that matter most:\n"
    "  spec_ambiguous / spec_wrong -- the objective or its tests disagree, or "
    "admit more than one reading. This was the dominant cause in the reference "
    "build: the implementation was never the problem.\n"
    "  missing_module / env_broken -- imports, paths, or tooling.\n"
    "  scope_too_large -- the task asks for more than one pass.\n"
    "  reviewer_noise -- the deterministic gates passed and the objection is "
    "not supported by evidence.\n"
    "Cite specific evidence for every claim. If the evidence does not support "
    "a confident conclusion, say cause='unknown' and remedy='needs_human' "
    "rather than guessing.\n\n"
    # v2.9.1: the response shape was never stated.  The prompt described the two
    # vocabularies in prose and then said "respond with one JSON object", so
    # models returned whatever the taxonomy suggested -- almost never a
    # ``confidence`` or ``evidence`` field.  ``coerce`` defaults those to 0.0
    # and [], which made every diagnosis self-report as zero-confidence and
    # evidence-free, so no remedy was ever auto-applied and every escalation
    # logged as inconclusive.  The tier was inert, and looked like a model
    # limitation rather than a missing schema.
    "Respond with ONE JSON object and nothing else:\n"
    '{"cause": "<one of the allowed causes>",\n'
    ' "confidence": <0.0-1.0 float, how well the evidence supports the cause>,\n'
    ' "evidence": ["short quoted fact from the evidence block", ...],\n'
    ' "remedy": "<one of the allowed remedies>",\n'
    ' "proposed_objective": "<only for narrow_objective: the rewritten, '
    'unambiguous objective>",\n'
    ' "proposed_split": [{"title": "...", "objective": "...", '
    '"target_files": ["..."], "dependencies": []}]}\n\n'
    "Always fill confidence and evidence. A confidence you did not state is "
    "recorded as 0.0, which marks the whole diagnosis inconclusive and applies "
    "no remedy."
)

#: Cause taxonomy.  Fixed vocabulary on purpose: these strings are clustered by
#: the steering monitor, so a free-form cause would defeat the whole point --
#: three escalations caused by the same thing have to *look* the same.
CAUSES = (
    "spec_ambiguous",      # the objective permits more than one reading
    "spec_wrong",          # the objective contradicts itself or its tests
    "missing_module",      # import/path/environment, not logic
    "env_broken",          # dependency or tooling problem
    "scope_too_large",     # the task is bigger than one Builder pass
    "reviewer_noise",      # the gates disagree and the reviewer is wrong
    "model_capability",    # the model could not do this task
    "unknown",
)

#: Remedy vocabulary, aligned with the board-write whitelist.  A remedy not in
#: that whitelist cannot modify the board even if a model returns it.
REMEDIES = (
    "no_action",           # annotate only
    "narrow_objective",    # rewrite the objective to remove the ambiguity
    "split_task",          # one blocked task becomes two smaller ones
    "requeue_task",        # give it another bounded attempt
    "defer_task",          # wait for a prerequisite (quota, upstream)
    "fix_env",             # operator action; no board write
    "reroute_model",       # operator action; no board write
    "needs_human",         # stop; a human decides
)

#: Remedies this module will apply to the board without a human.  Deliberately
#: two, and both reversible.  ``fix_env``/``reroute_model``/``needs_human``
#: are advice, not edits: acting on a model's guess that the environment is
#: broken can turn a recoverable task into a stalled one.
AUTO_REMEDIES = ("narrow_objective", "split_task")


@dataclass
class Diagnosis:
    """One structured explanation of an escalation."""

    task_id: str = ""
    cause: str = "unknown"
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    remedy: str = "needs_human"
    #: Populated when remedy is ``narrow_objective``.
    proposed_objective: Optional[str] = None
    #: Populated when remedy is ``split_task``; each item is a task draft.
    proposed_split: List[Dict[str, Any]] = field(default_factory=list)
    #: True when the remedy was actually written to the board.
    applied: bool = False

    @property
    def actionable(self) -> bool:
        """Whether this remedy may touch the board right now.

        Three independent gates, all of which must pass: the remedy is one of
        the two reversible ones, it is on the board-write whitelist, and the
        model was confident enough.  The confidence gate matters most in
        practice -- the diagnosing model also participated in the work that
        failed, so its opinion is not independent evidence.
        """
        return (self.remedy in AUTO_REMEDIES
                and is_executable_remedy(self.remedy)
                and self.confidence >= config.DIAGNOSIS_MIN_CONFIDENCE)

    def to_dict(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "cause": self.cause,
                "confidence": self.confidence, "evidence": self.evidence,
                "remedy": self.remedy, "applied": self.applied,
                "proposed_objective": self.proposed_objective,
                "proposed_split": self.proposed_split}

    def feedback(self) -> str:
        """The ``last_feedback`` prefix a human reads on the board.

        The point is that escalation stops delivering a stack trace and starts
        delivering a conclusion with evidence behind it.
        """
        head = f"DIAGNOSIS: {self.cause} (confidence {self.confidence:.2f}) -> " \
               f"{self.remedy}"
        if self.applied:
            head += " [applied]"
        lines = [head]
        for item in self.evidence[:5]:
            lines.append(f"  - {item}")
        if self.proposed_objective:
            lines.append(f"  narrowed objective: {self.proposed_objective[:200]}")
        if self.proposed_split:
            lines.append(f"  proposed split into {len(self.proposed_split)} task(s)")
        return "\n".join(lines)


def coerce(data: Any, task_id: str = "") -> Optional[Diagnosis]:
    """Tolerate real-model shapes; reject anything unrecognised.

    Free-form cause/remedy strings are normalised into the fixed vocabularies
    rather than trusted, because a cause the steering monitor cannot cluster is
    a cause that does not contribute to systemic detection.
    """
    if not isinstance(data, dict):
        return None
    cause = str(data.get("cause") or "unknown").strip().lower()
    if cause not in CAUSES:
        cause = "unknown"
    remedy = str(data.get("remedy") or "needs_human").strip().lower()
    if remedy not in REMEDIES:
        remedy = "needs_human"
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    evidence = data.get("evidence")
    if isinstance(evidence, str):
        evidence = [evidence]
    proposed = data.get("proposed_objective")
    if proposed is not None and not isinstance(proposed, str):
        proposed = None
    split = data.get("proposed_split")
    if not isinstance(split, list):
        split = []
    clean_split: List[Dict[str, Any]] = []
    for item in split[:4]:
        # A split draft is an untrusted task payload; only the descriptive
        # fields survive, and they still go through TaskSpec validation before
        # anything is written.
        if isinstance(item, dict) and str(item.get("objective") or "").strip():
            clean_split.append({
                "objective": str(item["objective"])[:4000],
                "title": str(item.get("title") or "")[:200],
                "target_files": [str(f) for f in (item.get("target_files") or [])][:8],
                "dependencies": [str(d) for d in (item.get("dependencies") or [])][:8],
            })
    return Diagnosis(task_id=task_id, cause=cause,
                     confidence=max(0.0, min(1.0, confidence)),
                     evidence=[str(e)[:300] for e in (evidence or [])][:6],
                     remedy=remedy, proposed_objective=proposed,
                     proposed_split=clean_split)


def summarize_evidence(task: TaskSpec, attempts: List[Dict[str, Any]],
                       reason: str = "", failing_tests: Optional[List[str]] = None,
                       missing_modules: Optional[List[str]] = None,
                       test_sources: Optional[List[Tuple[str, str]]] = None
                       ) -> str:
    """Build the diagnosis prompt's evidence block from what the loop recorded.

    Everything here already exists by the time a task escalates -- which is the
    argument for doing this at all.  No new instrumentation, no new run.

    ``test_sources`` is the addition that makes escalation diagnosis actually
    able to reach a conclusion (v2.9.1).  Without the test file, the model sees
    "max retries exhausted" plus a gate line and cannot distinguish the two
    classes that matter most: *the Builder is wrong* and *the contract is
    wrong*.  Those need opposite remedies, and picking wrong is what produces
    seven identical retries.  Observed on a real build: a task whose authored
    test called ``JournalEntry()`` while the authored objective for
    ``JournalEntry`` raised ``ValueError`` on fewer than two postings.  The test
    could never pass and the Builder could never be at fault -- and every
    escalation of it came back ``confidence 0.00`` with no evidence, because
    nobody had shown the diagnoser the contract.
    """
    lines = [f"task_id: {task.id}", f"title: {task.title}",
             f"task_type: {task.task_type}",
             f"objective: {(task.objective or '')[:1200]}",
             f"target_files: {task.target_files}",
             f"acceptance_criteria: {task.acceptance_criteria}",
             f"escalation_reason: {reason[:400]}",
             f"attempts: {len(attempts)}"]
    if failing_tests:
        lines.append("failing_tests: " + ", ".join(failing_tests[:10]))
    if missing_modules:
        lines.append("missing_modules: " + ", ".join(missing_modules[:10]))
    for index, attempt in enumerate(attempts[-config.MAX_ATTEMPTS:], 1):
        lines.append(f"--- attempt {index} ---")
        lines.append(f"verdict: {attempt.get('verdict')}")
        lines.append(f"feedback: {str(attempt.get('feedback'))[:800]}")
        tests = attempt.get("tests")
        if tests:
            lines.append(f"tests: {tests}")
    for path, source in (test_sources or [])[:3]:
        lines.append(f"--- test contract: {path} ---")
        lines.append(source[:4000])
    if test_sources:
        lines.append(
            "IMPORTANT: the test above is a FROZEN contract -- the Builder is "
            "forbidden from editing it. If the test contradicts the objective "
            "(calls a constructor the objective says raises, asserts a shape "
            "the objective does not have, or imports a path the objective's "
            "modules do not provide), then the SPECIFICATION is wrong, not the "
            "Builder: report cause='spec_wrong'.")
    return "\n".join(lines)


def config_max_attempts() -> int:
    """The orchestrator's retry budget, for prompt budgeting."""
    return int(config.MAX_ATTEMPTS)


# ---------------------------------------------------------------------------
# Board application -- bounded by the whitelist, always reversible
# ---------------------------------------------------------------------------
def apply(diag: Diagnosis, task: TaskSpec,
          set_status: Any) -> Diagnosis:
    """Apply an auto-remedy through the injected board writer.

    ``set_status`` is the orchestrator's own ``_set_task_status``, injected so
    this module cannot reach the board except through the caller's validated
    path -- the same reason the Council's actions were whitelist-bound.
    """
    if not diag.actionable:
        return diag
    allowed = set(board_keys_for(diag.remedy))
    updates: Dict[str, Any] = {}
    if diag.remedy == "narrow_objective" and diag.proposed_objective:
        if "objective" in allowed:
            updates["objective"] = diag.proposed_objective[:4000]
    elif diag.remedy == "split_task" and diag.proposed_split:
        # Splitting creates tasks, which is outside any single-task whitelist;
        # it is recorded as advice for a human rather than executed here.
        diag.remedy = "needs_human"
        return diag
    if not updates:
        return diag
    try:
        set_status(task.id, **updates)
        diag.applied = True
    except Exception:  # noqa: BLE001 -- advice must never break escalation
        diag.applied = False
    return diag
