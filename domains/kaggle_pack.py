"""Kaggle domain pack (v2.6) -- an optional vertical of the coding core.

The pack is inert unless a project opts in with ``.agent/domain.json``::

    {"pack": "kaggle", "competition": "titanic", "metric": "accuracy",
     "direction": "maximize", "target": 0.85}

It contributes three things, all through generic core hooks:

* a **scaffold** (``--new-kaggle``): data/src/experiments/reports/
  submissions layout, Kaggle requirements, context template, campaign
  state;
* a **campaign director** (``propose_tasks``): when the board starves and
  ``CAMPAIGN_AUTO_GENERATE=true``, queued hypotheses become new
  ``experiment`` tasks carrying the pinned metric + split-hash contract;
  optionally an **LLM campaign director** (``CAMPAIGN_DIRECTOR_ENABLED``)
  generates new hypotheses from experiment results automatically;
* a **submission gate** (``on_task_completed``): a format-validated
  ``submissions/submission.csv`` is submitted only when out-of-fold CV
  improves, under a daily cap, and only with ``KAGGLE_AUTO_SUBMIT=true``.
  It additionally passes a fail-closed secret gate: the submission file is
  written by a model-authored script, so nothing credential-shaped may leave
  the machine.

Nothing Kaggle-specific exists outside this module.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import executors
import patcher
import test_integrity
from domain_registry import DomainPack
from logger import log_event

#: Shell metacharacters that let a command chain, substitute or redirect away
#: from the project.  Rejected by :func:`validate_run_command`.
_SHELL_METACHARS = (";", "|", "&", "`", "$(", ">", "<", "\n", "\r")

#: Interpreters an experiment command may invoke.
_ALLOWED_RUNNERS = ("python", "python3", "py")


# ---------------------------------------------------------------------------
# Campaign state
# ---------------------------------------------------------------------------
def campaign_path(project_root: Path) -> Path:
    return Path(project_root) / ".agent" / "campaign.json"


def load_campaign(project_root: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(campaign_path(project_root).read_text("utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_campaign(project_root: Path, campaign: Dict[str, Any]) -> None:
    try:
        path = campaign_path(project_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        campaign["updated_at"] = datetime.now(timezone.utc).isoformat()
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(campaign, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def is_improvement(value: Optional[float], best: Optional[float],
                   direction: str, min_improvement: float) -> bool:
    if value is None:
        return False
    if best is None:
        return True
    delta = (value - best) if (direction or "maximize") == "maximize" \
        else (best - value)
    return delta >= float(min_improvement)


def _submissions_today(campaign: Dict[str, Any]) -> int:
    today = _today()
    return sum(1 for record in campaign.get("submissions", [])
               if str(record.get("day") or "") == today)


def _next_task_id(board: Dict[str, Any], extra: List[Dict[str, Any]]) -> str:
    used = {str(t.get("id")) for t in board.get("tasks", [])}
    used |= {str(t.get("id")) for t in extra}
    index = 1
    while f"EXP-{index:04d}" in used:
        index += 1
    return f"EXP-{index:04d}"


# ---------------------------------------------------------------------------
# Untrusted-input validation (LLM-authored hypotheses)
# ---------------------------------------------------------------------------
def validate_target_files(project_root: Path, raw: Any,
                          default: Optional[List[str]] = None) -> List[str]:
    """Keep only project-relative, in-scope, writable, non-secret targets.

    A campaign hypothesis is model output, and ``target_files`` decides what the
    Builder is permitted to write.  Passed through unvalidated, a hypothesis
    naming ``.env`` would make the orchestrator's scope guard *permit* the write
    -- the guard checks membership in this very list.  Every entry must survive
    three filters:

    1. ``patcher.resolve_target`` -- traversal, absolute paths and the
       secret-bearing names in ``patcher.FORBIDDEN_NAMES``.
    2. the test-integrity baseline (v2.8) -- a hypothesis may not claim a
       pre-existing test or pytest config as writable, or the scope guard would
       permit the Builder to rewrite the grader.
    3. de-duplication.
    """
    candidates = raw if isinstance(raw, (list, tuple)) else []
    kept: List[str] = []
    guard: Optional[test_integrity.TestIntegrityGuard] = None
    for entry in candidates:
        rel = str(entry).replace("\\", "/").strip()
        if not rel:
            continue
        if patcher.resolve_target(project_root, rel) is None:
            log_event("WARNING", "campaign_target_file_rejected",
                      {"project": Path(project_root).name, "path": rel,
                       "reason": "outside project scope or a denied name"})
            continue
        if test_integrity.is_protected_path(rel):
            if guard is None:
                guard = test_integrity.TestIntegrityGuard.capture(project_root)
            if guard.is_baselined(rel):
                log_event("WARNING", "campaign_target_file_rejected",
                          {"project": Path(project_root).name, "path": rel,
                           "reason": "pre-existing test or pytest config is "
                                     "the fixed specification, not a "
                                     "writable target"})
                continue
        if rel not in kept:
            kept.append(rel)
    return kept or list(default or [])


def validate_run_command(command: Any) -> Optional[str]:
    """Return a safe experiment command, or None to refuse it.

    ``run_command`` is executed with ``shell=True`` and inherits the full
    environment (``executors.run_objective``).  A hypothesis such as
    ``env > reports/metrics.json`` satisfies the objective gate -- it exits 0
    and leaves a parseable metrics file -- while dumping every configured
    credential into a file that is then submitted to Kaggle.  So by default the
    command must be a plain interpreter invocation with no chaining,
    substitution or redirection.

    Set ``KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND=true`` to restore the previous
    unrestricted behaviour.
    """
    if not isinstance(command, str) or not command.strip():
        return None
    text = command.strip()
    if getattr(config, "KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND", False):
        return text
    for meta in _SHELL_METACHARS:
        if meta in text:
            log_event("WARNING", "campaign_run_command_rejected",
                      {"reason": f"shell metacharacter {meta!r}"})
            return None
    try:
        argv = shlex.split(text, posix=False)
    except ValueError:
        return None
    argv = [part.strip('"').strip("'") for part in argv if part.strip()]
    if not argv:
        return None
    runner = Path(argv[0]).name.lower()
    if runner not in _ALLOWED_RUNNERS:
        log_event("WARNING", "campaign_run_command_rejected",
                  {"reason": f"runner {argv[0]!r} is not an allowed interpreter"})
        return None
    # Re-serialize so the executed string is exactly what was validated.
    return " ".join(shlex.quote(part) for part in argv)


# ---------------------------------------------------------------------------
# Scaffold
# ---------------------------------------------------------------------------
_CONTEXT_TEMPLATE = """# {name} -- Kaggle competition brief ({competition})

Primary metric: **{metric}** ({direction}). Target: {target}.

## Hard rules

1. **One split only.** Every experiment must use the split pinned by the
   task (`split_hash` in reports/metrics.json must equal the value in the
   task's metric spec). Scores from different splits are never comparable
   and the gate rejects them.
2. **Metric contract.** `reports/metrics.json` must contain at least
   `{{"split_hash": "...", "{metric}": <number>}}`; the primary metric is
   read from that file after the command exits 0.
3. **Data is read-only.** `data/raw/` is never modified; write derived
   data to `data/processed/`.
4. **Submission format.** `submissions/submission.csv` must match
   `data/raw/sample_submission.csv` (columns, row count, id set, no NaN).
5. **CV before leaderboard.** Model selection is decided by out-of-fold
   {metric} only; leaderboard scores are recorded for drift tracking.
6. **No leakage.** Target-derived features, train/test contamination, and
   label leakage are hard failures; validate with adversarial validation.

## Layout

- `experiments/` -- one experiment script per task (declared target file)
- `reports/metrics.json` -- written by the experiment command
- `submissions/submission.csv` -- produced by the best validated pipeline
- `.agent/campaign.json` -- campaign state (best CV, submissions, queue)

## Workflow

Tokens are free-tier and scarce: one focused experiment per task, small
outputs, deterministic seeds, no network in scripts beyond the local data.
Divergence: `.agent/divergence.json` sets `{{"enabled": true}}` so hard
tasks fan out candidate strategies and converge on the CV winner.
"""

_REQUIREMENTS = """\
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
optuna
kaggle
"""

_STARTER_HYPOTHESES = [
    {
        "title": "Baseline model",
        "objective": (
            "Tests-first is OFF for this experiment task. Write a single "
            "self-contained script at experiments/exp_baseline.py that: "
            "loads the competition training data from data/raw, builds the "
            "SAME deterministic split every experiment must use (pinned "
            "seed), trains a simple regularized baseline, evaluates the "
            "primary metric out-of-fold, and writes reports/metrics.json "
            "with {metric, split_hash} where split_hash is a stable hash of "
            "the split definition. Print the metric. No network."),
        "target_files": ["experiments/exp_baseline.py"],
        "run_command": "python experiments/exp_baseline.py",
        "error_signature": "ERR_KAGGLE_BASELINE",
    },
    {
        "title": "Feature engineering pass",
        "objective": (
            "Write experiments/exp_features.py: same split as the baseline "
            "(identical split_hash), engineered features derived only from "
            "training folds (no leakage), gradient-boosted model, OOF metric "
            "written to reports/metrics.json. Must improve on the baseline "
            "CV or report honestly."),
        "target_files": ["experiments/exp_features.py"],
        "run_command": "python experiments/exp_features.py",
        "error_signature": "ERR_KAGGLE_FEATURES",
    },
]


def scaffold(name: str, competition: str, metric: str = "accuracy",
             direction: str = "maximize", target: Optional[float] = None,
             root: Optional[Path] = None) -> Path:
    """Create ``projects/<name>/`` as a Kaggle campaign (idempotent)."""
    import re as _re
    if not _re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,63}", name or ""):
        raise ValueError(
            "project name must be 2-64 chars: letters, digits, '_' or '-'")
    base = Path(root) if root is not None else config.WORKSPACE_ROOT
    config.ensure_workspace(base)
    project = base / "projects" / name
    for directory in ("data/raw", "data/processed", "src", "experiments",
                      "reports", "submissions", "tests", ".agent"):
        (project / directory).mkdir(parents=True, exist_ok=True)

    board_path = project / ".agent" / "task_board.json"
    if board_path.exists():
        try:
            existing = json.loads(board_path.read_text("utf-8"))
            if existing.get("tasks"):
                raise FileExistsError(
                    f"refusing to clobber non-empty board: {board_path}")
        except json.JSONDecodeError as exc:
            raise FileExistsError(
                f"refusing to clobber unreadable board: {exc}") from exc

    def _write(rel: str, content: str) -> None:
        path = project / rel
        if not path.exists():
            path.write_text(content, encoding="utf-8")

    _write(".agent/task_board.json", json.dumps({"tasks": []}, indent=2) + "\n")
    _write(".agent/context.md", _CONTEXT_TEMPLATE.format(
        name=name, competition=competition, metric=metric,
        direction=direction, target=target))
    _write(".agent/domain.json", json.dumps({
        "pack": "kaggle", "competition": competition, "metric": metric,
        "direction": direction, "target": target,
        "min_improvement": 0.0, "split_key": None,
    }, indent=2) + "\n")
    _write("requirements.txt", _REQUIREMENTS)
    _write("experiments/.gitkeep", "")
    _write("reports/.gitkeep", "")
    _write("submissions/.gitkeep", "")
    _write(".agent/divergence.json", json.dumps({
        "enabled": False, "n_candidates": 3,
    }, indent=2) + "\n")
    if not campaign_path(project).exists():
        save_campaign(project, {
            "competition": competition, "metric": metric,
            "direction": direction, "target": target,
            "best_cv": None, "split_key": None,
            "pending_hypotheses": list(_STARTER_HYPOTHESES),
            "experiments": [], "submissions": [],
        })
    log_event("INFO", "kaggle_scaffold_created",
              {"project": name, "competition": competition})
    return project


# ---------------------------------------------------------------------------
# The pack
# ---------------------------------------------------------------------------
class KagglePack(DomainPack):
    """Campaign director + CV-gated submission flow for Kaggle projects."""

    name = "kaggle"
    #: Class-level defaults so an unconfigured instance is still safe to
    #: use (per-project ``configure`` overwrites them on a copy).
    competition = ""
    metric = "accuracy"
    direction = "maximize"
    target = None
    min_improvement = 0.0
    split_key = None

    def configure(self, cfg: Dict[str, Any]) -> None:
        self.competition = str(cfg.get("competition") or "")
        self.metric = str(cfg.get("metric") or "accuracy")
        self.direction = str(cfg.get("direction") or "maximize")
        self.target = cfg.get("target")
        self.min_improvement = float(cfg.get("min_improvement") or 0.0)
        self.split_key = cfg.get("split_key")

    # -- hook 1: campaign director -------------------------------------
    async def propose_tasks(self, project_root: Path,
                            board: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not config.CAMPAIGN_AUTO_GENERATE:
            return []
        campaign = load_campaign(project_root)
        pending = list(campaign.get("pending_hypotheses") or [])
        if not pending:
            return []
        metric_name = campaign.get("metric") or self.metric
        direction = campaign.get("direction") or self.direction
        baseline = campaign.get("best_cv")
        split_key = campaign.get("split_key") or self.split_key
        required = ["split_hash", metric_name]
        out: List[Dict[str, Any]] = []
        for index, hypothesis in enumerate(
                pending[:config.CAMPAIGN_MAX_NEW_TASKS]):
            task_id = _next_task_id(board, out)
            out.append({
                "id": task_id,
                # Internal marker consumed by ``tasks_accepted``; stripped
                # from the board by the core before persisting.
                "_hypothesis_index": index,
                "title": hypothesis.get("title", f"Experiment {task_id}"),
                "objective": hypothesis.get("objective", ""),
                # Both fields are model output: ``target_files`` decides what
                # the Builder may write (the scope guard checks membership in
                # this list) and ``run_command`` is executed with shell=True in
                # a fully-populated environment.
                "target_files": validate_target_files(
                    project_root, hypothesis.get("target_files"),
                    default=[f"experiments/{task_id.lower()}.py"]),
                "dependencies": hypothesis.get("dependencies", []),
                "priority": hypothesis.get("priority", "high"),
                "acceptance_criteria": [
                    "reports/metrics.json carries the primary metric and a "
                    "split_hash matching the campaign split",
                    "the command exits 0 and the primary metric is reported "
                    "honestly (no fabricated numbers)",
                ],
                "workspace_patterns_referenced": [],
                "retry_count": 0,
                "status": "pending",
                "last_feedback": None,
                "error_signature": hypothesis.get(
                    "error_signature", f"ERR_KAGGLE_{task_id}"),
                "task_type": "experiment",
                "should_pass": True,
                "requires_new_tests": False,
                "metric": {
                    "name": metric_name, "direction": direction,
                    "target": campaign.get("target"),
                    "min_improvement": float(
                        campaign.get("min_improvement")
                        or self.min_improvement),
                    "baseline": baseline,
                    "split_key": split_key,
                    "required_fields": required,
                },
                "run_command": validate_run_command(hypothesis.get("run_command")),
            })
        log_event("INFO", "campaign_tasks_proposed",
                  {"project": project_root.name, "count": len(out),
                   "ids": [t["id"] for t in out]})
        return out

    async def tasks_accepted(self, project_root: Path,
                             tasks: List[Dict[str, Any]]) -> None:
        """Commit the queue only for hypotheses the core actually accepted."""
        indexes = sorted(
            {int(t.get("_hypothesis_index")) for t in tasks
             if isinstance(t.get("_hypothesis_index"), int)},
            reverse=True)
        if not indexes:
            return
        campaign = load_campaign(project_root)
        pending = list(campaign.get("pending_hypotheses") or [])
        removed = 0
        for index in indexes:
            if 0 <= index < len(pending):
                pending.pop(index)
                removed += 1
        campaign["pending_hypotheses"] = pending
        save_campaign(project_root, campaign)
        log_event("INFO", "campaign_queue_committed",
                  {"project": project_root.name, "consumed": removed,
                   "remaining": len(pending)})

    # -- hook 2: post-approval reaction --------------------------------
    async def on_task_completed(self, orchestrator: Any, task: Any,
                                objective_result: Any,
                                applied_files: Any = None) -> None:
        """Record the experiment; CV-gate + validate + (optionally) submit."""
        if objective_result is None or not getattr(objective_result, "passed", False):
            return
        project_root: Path = orchestrator.project_root
        campaign = load_campaign(project_root)
        if not campaign:
            return
        direction = campaign.get("direction") or self.direction
        primary = objective_result.primary
        best = campaign.get("best_cv")
        min_improvement = float(campaign.get("min_improvement")
                                or self.min_improvement
                                or config.KAGGLE_MIN_CV_IMPROVEMENT)
        campaign.setdefault("experiments", []).append({
            "task_id": task.id,
            "primary": primary,
            "split_hash": objective_result.split_hash,
            "runtime_s": objective_result.runtime_s,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        improved = is_improvement(primary, best, direction, min_improvement)
        if improved:
            campaign["best_cv"] = primary
            if objective_result.split_hash:
                campaign["split_key"] = objective_result.split_hash
        log_event("INFO", "campaign_experiment_recorded",
                  {"project": project_root.name, "task_id": task.id,
                   "primary": primary, "best_cv": campaign.get("best_cv"),
                   "improved": improved})

        # Persist the experiment record before branching into submission.
        save_campaign(project_root, campaign)

        # --- CV-gated submission (only on improvement) -----------------
        if improved:
            submission_rel = "submissions/submission.csv"
            valid, reason = executors.validate_submission(
                project_root, submission_rel,
                "data/raw/sample_submission.csv")
            # Soft egress gate, ahead of the hard one in run_kaggle_submit:
            # report *why* a submission was withheld at the point of decision,
            # so an operator watching the campaign ledger sees the cause rather
            # than a silently missing entry.
            secrets = executors.submission_secret_findings(
                project_root, submission_rel) if valid else []
            if valid and secrets and config.KAGGLE_AUTO_SUBMIT:
                log_event("WARNING", "kaggle_submission_blocked_secrets",
                          {"project": project_root.name,
                           "task_id": task.id,
                           "rules": sorted({f["rule"] for f in secrets}),
                           "findings": len(secrets),
                           "note": "submission withheld; rebuild it without "
                                   "credential-shaped content"})
            elif valid and config.KAGGLE_AUTO_SUBMIT:
                cap = int(config.KAGGLE_DAILY_SUBMISSION_CAP)
                if cap <= 0 or _submissions_today(campaign) < cap:
                    competition = (campaign.get("competition")
                                   or self.competition)
                    submit_result = await asyncio.to_thread(
                        executors.run_kaggle_submit, project_root,
                        competition, submission_rel,
                        f"CV {primary:.6g} ({task.id})")
                    campaign.setdefault("submissions", []).append({
                        "day": _today(),
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "task_id": task.id, "cv": primary,
                        "ref": submit_result.get("ref"),
                        "score": submit_result.get("score"),
                        "status": ("submitted"
                                   if submit_result.get("ok")
                                   else "failed"),
                        "detail": str(submit_result.get("detail", ""))[:300],
                    })
                    save_campaign(project_root, campaign)
            elif not valid:
                log_event("WARNING", "kaggle_submission_invalid",
                          {"project": project_root.name, "reason": reason})
            elif secrets:
                log_event("WARNING", "kaggle_submission_blocked_secrets",
                          {"project": project_root.name, "task_id": task.id,
                           "rules": sorted({f["rule"] for f in secrets}),
                           "findings": len(secrets),
                           "note": "KAGGLE_AUTO_SUBMIT=false (human gate) and "
                                   "the file would not pass the secret gate"})
            else:
                log_event("INFO", "kaggle_submission_ready",
                          {"project": project_root.name, "reason": reason,
                           "note": "KAGGLE_AUTO_SUBMIT=false (human gate)"})

        # --- LLM campaign director: generate next hypotheses from results -
        if config.CAMPAIGN_DIRECTOR_ENABLED \
                and not campaign.get("pending_hypotheses"):
            import agent_roles
            ctx = {
                "competition": campaign.get("competition"),
                "metric": campaign.get("metric"),
                "direction": campaign.get("direction"),
                "best_cv": campaign.get("best_cv"),
                "target": campaign.get("target"),
                "experiments_count": len(campaign.get("experiments", [])),
                "recent_experiments": campaign.get("experiments", [])[-5:],
            }
            director = agent_roles.CampaignDirectorAgent(
                orchestrator.llm, orchestrator.workspace_root)
            hypotheses = await director.run(ctx, [], [])
            if hypotheses and isinstance(hypotheses, list):
                campaign.setdefault("pending_hypotheses",
                                    []).extend(hypotheses)
                save_campaign(project_root, campaign)
                log_event("INFO", "campaign_hypotheses_generated",
                          {"project": project_root.name,
                           "count": len(hypotheses),
                           "titles": [h.get("title") for h in hypotheses]})


PACK = KagglePack()
