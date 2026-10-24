"""Divergent/convergent candidate engine (v2.6).

Two levels of divergent thinking feed one convergent selection:

* **Plan lenses** (``merge_plans``): the Planner runs once per lens
  (correctness / performance / simplicity) and the plans are merged
  deterministically.
* **Candidate strategies** (:func:`load_strategies`): each strategy brief
  drives one Builder pass; every candidate is scored by the authoritative
  deterministic gate first (tests or the objective metric), then by
  lint/diff-size/runtime.

Convergence is deterministic (:func:`select_winner`); an LLM judge is only
consulted for near-ties (:func:`is_near_tie`), sees blind labels only, and
can never override a gate-passing candidate with a gate-failing one.

This module is pure: no LLM calls, no network, trivially testable.  The
orchestrator owns the apply/rollback loop and calls :func:`store_candidate`
for the audit trail.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config
import redaction
from logger import log_event
from validation import Plan, StrategySpec, TaskSpec


# ---------------------------------------------------------------------------
# Strategy pools
# ---------------------------------------------------------------------------
DEFAULT_CODE_STRATEGIES: List[StrategySpec] = [
    StrategySpec(
        name="minimal-diff",
        approach=("smallest correct change: reuse existing helpers and "
                  "patterns, touch only what the task requires, keep the "
                  "public API byte-for-byte as pinned")),
    StrategySpec(
        name="robust-edges",
        approach=("correctness-first: explicitly handle every edge case "
                  "named in the objective plus empty/malformed inputs; "
                  "prefer explicitness over cleverness")),
    StrategySpec(
        name="performance-first",
        approach=("professional implementation optimizing clarity and "
                  "runtime: avoid redundant work/allocations while keeping "
                  "the exact pinned signatures")),
]

DEFAULT_EXPERIMENT_STRATEGIES: List[StrategySpec] = [
    StrategySpec(
        name="baseline-first",
        approach=("start from a simple, strongly regularized baseline; "
                  "measure on the pinned split and report the primary "
                  "metric honestly")),
    StrategySpec(
        name="feature-rich",
        approach=("engineer additional informative features from the "
                  "provided data; keep the split identical so scores remain "
                  "comparable")),
    StrategySpec(
        name="model-diverse",
        approach=("use a different model family/algorithm than the "
                  "baseline; same split, same metric, pinned "
                  "hyperparameters")),
]


def default_strategies(task: TaskSpec) -> List[StrategySpec]:
    if task.metric is not None or task.task_type == "experiment":
        return list(DEFAULT_EXPERIMENT_STRATEGIES)
    return list(DEFAULT_CODE_STRATEGIES)


# ---------------------------------------------------------------------------
# Activation policy
# ---------------------------------------------------------------------------
def board_policy(project_root: Path) -> Dict[str, Any]:
    """Per-project divergence opt-in (``.agent/divergence.json``)."""
    try:
        payload = json.loads(
            (Path(project_root) / ".agent" / "divergence.json")
            .read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def enabled(task: TaskSpec, project_root: Path) -> bool:
    """Task flag wins; else the board default; else the global toggle.

    A board file only overrides the global toggle when it actually states
    ``enabled``; a file carrying only tuning (e.g. ``n_candidates``) falls
    through to ``config.DIVERGENCE_ENABLED``.
    """
    if task.divergent is not None:
        return bool(task.divergent)
    policy = board_policy(project_root)
    if "enabled" in policy:
        return bool(policy.get("enabled"))
    return bool(config.DIVERGENCE_ENABLED)


def candidate_count(task: TaskSpec, project_root: Path,
                    strategies: Sequence[StrategySpec]) -> int:
    if task.n_candidates > 0:
        n = task.n_candidates
    elif task.strategies:
        n = len(task.strategies)
    else:
        n = int(board_policy(project_root).get("n_candidates")
                or config.DIVERGENCE_DEFAULT_CANDIDATES)
    return max(1, min(n, int(config.DIVERGENCE_MAX_CANDIDATES)))


def load_strategies(task: TaskSpec, project_root: Path) -> List[StrategySpec]:
    """Task strategies, else the default pool, padded/truncated to N."""
    named = list(task.strategies) or default_strategies(task)
    n = candidate_count(task, project_root, named)
    out: List[StrategySpec] = []
    rounds = 0
    while len(out) < n:
        for base in named:
            if len(out) >= n:
                break
            label = base.name if rounds == 0 else f"{base.name}-r{rounds + 1}"
            out.append(base.model_copy(update={"name": label}))
        rounds += 1
    return out


# ---------------------------------------------------------------------------
# Candidate records + scoring
# ---------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    """One gated candidate: evidence + deterministic score."""

    label: str
    strategy: StrategySpec
    diff: str = ""
    gate_ok: bool = False
    tests_passed: Optional[int] = None
    tests_failed: Optional[int] = None
    lint_count: int = 0
    metric: Optional[float] = None
    direction: str = "maximize"
    runtime_s: float = 0.0
    model_role: str = ""
    usage: Optional[Dict[str, Any]] = None
    feedback: str = ""
    gate: Dict[str, Any] = field(default_factory=dict)

    def score(self) -> float:
        """Gate-passing always beats gate-failing; then metric/quality."""
        gate_bonus = 1_000_000.0 if self.gate_ok else 0.0
        if self.metric is not None:
            value = self.metric if self.direction == "maximize" else -self.metric
            runtime_penalty = min(60.0, max(0.0, self.runtime_s)) * 0.01
            return gate_bonus + value - runtime_penalty
        passed = float(self.tests_passed or 0)
        failed = float(self.tests_failed or 0)
        return (gate_bonus + passed - 10.0 * failed
                - 0.5 * float(self.lint_count)
                - 0.001 * float(len(self.diff)))


def record_from_gate(label: str, strategy: StrategySpec, diff: str,
                     gate: Optional[Dict[str, Any]], model_role: str,
                     usage: Optional[Dict[str, Any]], blocked: bool,
                     task: TaskSpec) -> CandidateRecord:
    """Build a CandidateRecord from a gate result (tests or objective)."""
    record = CandidateRecord(label=label, strategy=strategy, diff=diff or "",
                             model_role=model_role, usage=usage,
                             gate=dict(gate or {}))
    gate = gate or {}
    if task.metric is not None:
        objective = gate.get("objective") or {}
        record.metric = objective.get("primary")
        record.direction = (task.metric.direction or "maximize").lower()
        record.runtime_s = float(objective.get("runtime_s", 0.0) or 0.0)
        record.gate_ok = bool(objective.get("ok")) and not blocked
    else:
        tests = gate.get("tests") or {}
        record.tests_passed = tests.get("passed")
        record.tests_failed = tests.get("failed")
        record.lint_count = len(gate.get("lint") or [])
        record.gate_ok = (tests.get("failed", 1) == 0
                          and not blocked
                          and bool(gate.get("ok", True)))
    return record


def select_winner(records: Sequence[CandidateRecord]) -> int:
    """Index of the deterministic winner (gate first, then score)."""
    if not records:
        raise ValueError("no candidates to select from")
    return max(range(len(records)), key=lambda i: records[i].score())


def is_near_tie(records: Sequence[CandidateRecord],
                winner_index: Optional[int] = None) -> bool:
    """True when the top two scores are within ``DIVERGENCE_TIE_EPSILON``."""
    if len(records) < 2:
        return False
    winner_index = select_winner(records) if winner_index is None \
        else winner_index
    scores = sorted((r.score() for i, r in enumerate(records)
                     if i != winner_index), reverse=True)
    return abs(records[winner_index].score() - scores[0]) <= \
        float(config.DIVERGENCE_TIE_EPSILON)


# ---------------------------------------------------------------------------
# Plan lenses (divergent planning -> deterministic merge)
# ---------------------------------------------------------------------------
PLAN_LENSES = (
    ("correctness", "optimize the plan for correctness and edge-case coverage"),
    ("performance", "optimize the plan for runtime and memory efficiency"),
    ("simplicity", "optimize the plan for the smallest clear implementation"),
)


def merge_plans(task_id: str, plans: Sequence[Plan]) -> Plan:
    """Deterministic union: first-seen order, de-duplicated, capped."""
    merged = Plan(task_id=task_id)
    for field_name, cap in (("steps", 12), ("data_structures", 8),
                            ("edge_cases", 10), ("test_approach", 8)):
        seen: set = set()
        values: List[str] = []
        for plan in plans:
            for item in getattr(plan, field_name, []) or []:
                text = str(item).strip()
                if text and text not in seen:
                    seen.add(text)
                    values.append(text)
        setattr(merged, field_name, values[:cap])
    return merged


# ---------------------------------------------------------------------------
# Candidate artifacts (audit trail; never raises)
# ---------------------------------------------------------------------------
def _safe_label(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value)
    return (safe or "cand")[:40]


def store_candidate(project_root: Path, task_id: str,
                    record: CandidateRecord) -> Optional[Path]:
    """Persist one candidate under ``.agent/candidates/<task>/``."""
    try:
        directory = (Path(project_root) / ".agent" / "candidates"
                     / _safe_label(task_id))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{_safe_label(record.label)}-{_safe_label(record.strategy.name)}.json"
        payload = {
            "label": record.label,
            "strategy": record.strategy.model_dump(),
            "model_role": record.model_role,
            "usage": record.usage,
            "gate_ok": record.gate_ok,
            "tests_passed": record.tests_passed,
            "tests_failed": record.tests_failed,
            "lint_count": record.lint_count,
            "metric": record.metric,
            "direction": record.direction,
            "runtime_s": record.runtime_s,
            "score": record.score(),
            # Archived verbatim model output; redacted so a credential the model
            # was shown cannot be persisted into a candidate record.
            "diff": redaction.redact_known_secrets(record.diff[:20000]),
            "feedback": redaction.redact_known_secrets(record.feedback[:1000]),
        }
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        _prune(directory)
        return path
    except OSError:
        return None


def _prune(directory: Path, keep: int = 24) -> None:
    try:
        files = sorted(directory.glob("*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            stale.unlink(missing_ok=True)
    except OSError:
        pass


def blind_payload(records: Sequence[CandidateRecord]) -> List[Dict[str, Any]]:
    """Judge input: blind labels, gate evidence, diff heads (no models)."""
    payload: List[Dict[str, Any]] = []
    for record in records:
        payload.append({
            "label": record.label,
            "strategy_hint": record.strategy.approach[:200],
            "gate_ok": record.gate_ok,
            "tests_passed": record.tests_passed,
            "tests_failed": record.tests_failed,
            "lint_count": record.lint_count,
            "metric": record.metric,
            "runtime_s": record.runtime_s,
            "diff_head": record.diff[:1500],
        })
    return payload


#: Max chars of each candidate diff included in the merge prompt.
MERGE_DIFF_CAP = 8000


def merge_payload(records: Sequence[CandidateRecord]) -> List[Dict[str, Any]]:
    """Merger input: full gate-passing candidate diffs + evidence."""
    payload: List[Dict[str, Any]] = []
    for record in records:
        if not record.gate_ok:
            continue
        payload.append({
            "label": record.label,
            "strategy": record.strategy.approach[:200],
            "gate_ok": record.gate_ok,
            "tests_passed": record.tests_passed,
            "tests_failed": record.tests_failed,
            "metric": record.metric,
            "diff": record.diff[:MERGE_DIFF_CAP],
        })
    return payload


def log_selection(task_id: str, records: Sequence[CandidateRecord],
                  winner_index: int) -> None:
    log_event("INFO", "divergence_winner_selected",
              {"task_id": task_id,
               "winner": records[winner_index].label,
               "strategy": records[winner_index].strategy.name,
               "candidates": len(records),
               "scores": {r.label: round(r.score(), 4) for r in records}})
