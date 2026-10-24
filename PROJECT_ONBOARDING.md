# PROJECT_ONBOARDING.md — Operator's Playbook for the Coding Agent

**Who this is for:** you, the coding agent (OpenCode, Claude Code, Cursor, or
a plain chat model) operating this repository. **The human provides intent,
decisions, and credentials. You perform every mechanical step:** scaffolding,
spec decomposition, tests-first authoring, configuration, running the system,
monitoring, and recovery. The human should never need to hand-edit a task
board or write a test file.

Read `README.md` first for architecture and run instructions. This document
is the *procedure* for onboarding a new project and operating the system.
`projects/project_main/` is the worked reference implementation.

---

## 1. Division of labor

| Human (intent + decisions) | You (everything mechanical) |
|---|---|
| Describe what to build (a rough document is enough) | Distill it into `context.md` (digest, hard rules, v1/v2 scope cut) |
| Approve model routing and cost | Verify/propose routing; configure `.env` |
| Provide API keys (broker, LLM) | Write `requirements.txt`, install, validate the board |
| Refine acceptance criteria if asked | Decompose the spec into single-file tasks + tests-first |
| Go / no-go on runs | Start, monitor, recover from escalations |

## 2. Pre-flight checklist (before onboarding anything)

1. `pip install -r requirements.txt` from the workspace root (includes
   `pytest` for the real-execution gate and `rich` for the dashboard).
2. `.env` exists at the workspace root: `OPENROUTER_API_KEY` set ⇒ live mode;
   empty ⇒ `DEMO_MODE` (offline canned runs — useful for a dry run of the
   plumbing, but it cannot build real projects).
   **Never commit `.env`.** The repo ships a `.gitignore` that excludes it and
   every `projects/*/.env`, but verify before your first push:
   ```bash
   python main.py --preflight scan     # exit 0 = clean, non-zero = findings
   ```
   Put real keys only in `.env` / `projects/<name>/.env`, never in
   `.env.example`, a task `objective`, or a `context.md` rule — those three are
   read into model prompts.
3. **Model routing sanity** — this is the single most important operational
   lesson: the *builder* model is the bottleneck. Verify with:

   ```bash
   python -c "import config; print(config.select_model('builder', attempt=1))"
   ```

   Routing is **cheap-first with bounded escalation**:

   | Attempt | Role | Model class |
   |---|---|---|
   | 1 | `builder_initial` | strong-but-affordable codegen (`deepseek/deepseek-chat-v3.1` class) |
   | 2 | `builder_patch` | same tier (patches from compact feedback) |
   | 3 | `builder_escalate` | premium (`anthropic/claude-sonnet-5`) — fires only when both cheap attempts failed |

   Cheap/fast chat models (z-ai/glm flash class) are excellent as
   `reviewer_*` — evidence: they caught real code defects all session — but
   as *builders* they failed 5/5 tasks even after content-mode (they emit
   tool-call hallucinations). Set overrides via `MODEL_OVERRIDES`.
4. If the project needs libraries (pandas, httpx, ...), write
   `projects/<name>/requirements.txt` and install them **before** starting
   the run — the real-test gate fails closed on missing imports.
   **v2.5.6:** the gate now self-heals missing imports by running
   `pip install -r projects/<name>/requirements.txt` once per task and
   re-running — so keep that file accurate; it is the authority for what
   the system may install. Imports no listed package provides still
   escalate, with the exact module named.

## 3. Onboarding a new project (the procedure)

### Step 1 — Scaffold

```bash
python main.py --new-project NAME     # creates projects/NAME with an empty board
```

### Step 1b — (v2.9) Or let the system write the specification

Steps 2–5 below are still the right way to onboard *important* work. But the
reference build made the cost of doing them by hand concrete: **all 12 task
escalations sat at this boundary and none was an engineering defect.** Six were
contradictions between an objective and a test file that were each individually
reasonable. So the system can now do this part:

```bash
# read the plan it would build, without committing to it
python main.py --new-project NAME --goal "one paragraph" --author-plan

# or: record the goal and let the first run author the board
python main.py --new-project NAME --goal "one paragraph"
```

What that produces, and how it stays safe:

* one implementation file per task, grouped into **chains**;
* each task's **test written by the system**, to disk *before* the task exists.
  That ordering is the whole safety mechanism — `TestIntegrityGuard` snapshots
  protected files when a task *starts*, so a test already on disk at that moment
  is frozen, and the Builder is refused any edit to the contract it is graded
  against;
* the Spec Critic then checks each objective against its own test before the
  task is scheduled, and holds any that disagree.

**Start with `SPEC_AUTHOR_WRITE_TESTS=false`** and read the plan first. You are
reviewing a specification the machine wrote, and the honest framing is that this
is a first draft — the Critic is a check, not a guarantee.

Three things to know about the plan you just read:

* **Authoring is two calls, not one.** The breakdown (ids, chains, exact API
  specs, dependencies) comes from one call; each task's test file comes from its
  own focused call. That is deliberate — asking one call for a breakdown *with
  every test file inlined* reliably failed on the free tier (a list of English
  sentences, `{"tasks": [{}]}`, test bodies with the function names stripped).
* **Read the `objective` against the `test`.** This is the one thing no check
  can do for you. An authored test that contradicts its own objective becomes
  the contract anyway, and no amount of retrying fixes it. On the reference
  build, 5 of 7 tasks delivered and both failures were exactly this: a test
  asserting a shape its objective did not have.
* **Every task carries an `error_signature`**, derived from the module under
  test if the model omitted one. Keep it: `steering.py` clusters escalations on
  it, and a task without one cannot be attributed to a failure pattern.

If the author cannot satisfy the goal it stops after `SPEC_AUTHOR_MAX_ATTEMPTS`
(default 3) and writes
`projects/NAME/.agent/spec_author_state.json` with `gave_up: true` and a
`remedy`. Fix `goal.txt` or `.agent/context.md` and delete that file to let it
try again. Do not delete it hoping for a different answer: each attempt carries
the previous one's rejection notes, so the retry is a different request.

### Step 2 — Write `projects/NAME/.agent/context.md`

A digest the inner agents will read on every task: one-paragraph goal, the
pipeline/architecture, hard rules (numbered, imperative), explicit v1 vs v2
scope cut, and a "build order" summary. See
`projects/project_main/.agent/context.md` as the template.

### Step 3 — Decompose into single-file tasks

**The proven unit of work is ONE implementation file per task.** Vague or
multi-file tasks are the leading cause of escalation, even with strong
models. Rules:

* `objective` states the **exact API**: full signatures, class names,
  return shapes, edge-case behavior, and which module-level seams exist for
  tests to monkeypatch. The tests pin these precisely — do not deviate.
* 2–4 `acceptance_criteria`, each objectively checkable (ideally "its test
  file is collected and green").
* Chain tasks with `dependencies`; keep independent tasks dependency-free so
  the orchestrator runs them in parallel.
* Give each task an `error_signature` (`ERR_<DOMAIN>_<NAME>`) — since v2.8
  this is what `steering.py` clusters on to detect a *systemic* failure (three
  escalations sharing a normalised shape are one design problem, not three
  bugs), and it is the key the Council reads when deciding whether to
  requeue, redesign or escalate. Give **every** task one; a missing signature
  is the fastest way to make a doom loop invisible.
* Set `"requires_new_tests": true` on every task that should add working
  tests (that is nearly all of them — see Step 4). The gate now diffs
  collected test **node IDs**, so a suite that shrank, or that did not grow,
  is rejected — not just one whose `passed` count failed to increase.
* `workspace_patterns_referenced` is **retired in v2.8 and ignored**. Leave it
  out of new boards.

### Step 4 — Write the tests FIRST (this is the whole game)

For every task, write `projects/NAME/tests/test_<module>.py` **before the
implementation exists**:

```python
"""Tests for src/<module>.py. No network anywhere in the suite."""


def _impl_ready() -> bool:
    """Tests-first gate: True once the implementation module exists."""
    import importlib.util
    modules = ("src.<package>.<module>",)   # all modules this file imports
    try:
        return all(importlib.util.find_spec(m) is not None for m in modules)
    except Exception:
        return False


if not _impl_ready():
    import pytest
    pytest.skip("implementation module(s) not present yet (tests-first)",
                allow_module_level=True)

from src.<package>.<module> import <api>  # noqa: E402
```

The self-skip guard keeps the whole suite green while modules are missing
and activates each file the moment its implementation lands — the
orchestrator's real-test gate then runs it for real. Test-writing rules:

* Compute expected values **independently** (reference implementation in a
  scratch script), assert with `pytest.approx` — never copy the
  implementation's logic into the test.
* **Never write `param or default` for list/set parameters in test
  helpers** — `[] or default` evaluates to `default`, silently replacing
  the explicit-empty edge case with a valid value (cost TASK-7007B two
  false escalations: the builder was correct, the helper was not). Use
  `param if param is not None else default`.
* All I/O behind module-level seams (`_download`, `_raw_info`, `_post`, ...)
  that tests `monkeypatch`. **Zero network in the suite** — this is what
  makes the gate deterministic.
* Cover the happy path, each acceptance-criterion edge case, and the typed
  error paths.

### Step 4b — Content-mode (how the Builder writes new modules)

For tests-first tasks the Builder does **not** emit unified diffs. It emits
each complete file in a fenced block with a path header
(```python path=src/... ```) — the system syntax-checks (`compile`), checks
coverage against `target_files`, rejects module-level `sys.exit` /
`raise SystemExit` (kills the pytest runner), auto-repairs misplaced
`from __future__` imports, and converts blocks to new-file diffs so every
safety rail still applies. Multi-file tasks generate one focused request
per file (small outputs — the sweet spot of cheap models). This is what
made cheap builders viable: with diff output they failed every task; with
content-mode they succeed.

### Step 5 — Validate the board

```bash
python -c "import json, sys; sys.path.insert(0, '.'); \
from validation import TaskSpec, validate_or_error; \
b = json.load(open(r'projects/NAME/.agent/task_board.json', encoding='utf-8')); \
print([t['id'] for t in b['tasks'] if validate_or_error(TaskSpec, t)[1]] or 'all valid')"
```

   Also confirm every `dependencies` entry references an id present on the
   board. Invalid fields surface later as
   `{"error": "Missing field: target_files"}` — catch them here instead.

**Or run the free gate (v2.9).** The check above validates *shape* only. The
Spec Critic validates *substance* — that the objective and the tests it will be
graded against actually agree — and it costs no model call for the mechanical
classes:

```bash
python -c "import sys; sys.path.insert(0, '.'); \
import json, pathlib; \
from validation import TaskSpec, validate_or_error; \
from spec_critic import deterministic_checks; \
root = pathlib.Path(r'projects/NAME'); \
board = json.loads((root / '.agent/task_board.json').read_text(encoding='utf-8')); \
[print(t.id, deterministic_checks(t, root).summary()) \
 for t in (validate_or_error(TaskSpec, r)[0] for r in board['tasks'])]"
```

This is worth running before every orchestration start on a hand-written board.
It is what catches the two failure classes that cost the most in the reference
build, and neither is visible by reading the board:

- a test importing a name the implementation module does not define — an
  `ImportError` at collection, which looks identical to "the Builder wrote
  nothing";
- a declared `-> dict` return that the test indexes as `[0]` — both sides spell
  everything correctly, the module imports cleanly, and it fails as a
  `TypeError` at runtime.

### Step 5b - Know which four switches matter (v2.9)

Not every knob needs setting. These three change behaviour materially:

| switch | default | when to touch it |
|---|---|---|
| `SPEC_AUTHOR_WRITE_TESTS` | `false` in `.env` | Leave false until you have read one `--author-plan` output. This is what lets the system write the contract its own Builder is graded against. |
| `SPEC_CRITIC_LLM_ENABLED` | `true` | The deterministic tier needs no model and cannot be collapsed. This switch turns off only the *judgement* pass — the right lever under quota pressure, and it costs you the semantic half of the check. |
| `LINE_COUNT` | `1` | `1` is exactly the pre-v2.9 sequential behaviour. The conflict-free partition is computed, validated and published in the manifest at any value, so raise this only after the learned RPM limits have settled — RPM is the binding constraint on a free tier. |
| `ONCE_BOOT_S` | `120` | How long a `--once` pass waits for every project loop to report its first scheduling decision. A pass waits for the loops to *settle*, not for the boards to look empty — a goal-only project has an empty board until the Spec Author writes one, and treating that as "finished" made `--once` return in seconds having authored nothing. Raise it only if a loop is genuinely slow to start; on expiry it logs `once_boot_timeout` and continues. |

### Step 5c - Check the board's shape, not just its validity

The gate in step 5 catches a malformed spec. It does not catch a board that is
*well-formed and structurally wrong*. Four things to look at by eye, each of
which is now a deterministic (non-blocking) Spec Critic finding:

```bash
python -c "import sys, json, pathlib; sys.path.insert(0, '.'); \
from validation import TaskSpec, validate_or_error; \
import spec_critic; \
root = pathlib.Path(r'projects/NAME'); \
board = json.loads((root / '.agent/task_board.json').read_text(encoding='utf-8')); \
[print(t.id, [f.kind for f in spec_critic.deterministic_checks(t, root).findings]) \
 for t in (validate_or_error(TaskSpec, r)[0] for r in board['tasks'])]"
```

* `multi_file_task` — more than one implementation module. The proven unit of
  work is ONE file per task; this is the leading cause of escalation. Split it.
* `duplicate_target_file` — two tasks claim one file. Under `LINE_COUNT=1` the
  second silently overwrites the first; under a real partition they race. The
  reference board has two such pairs and still delivered, which is why this is
  reported rather than fatal.
* `cross_task_dependency_missing` — this task's test imports a module another
  task creates, with no dependency edge. Both become eligible at once and the
  gate reports the confusing `passed == baseline` instead of the real cause.
* `unresolved_project_import` — an import into the project's own namespace that
  resolves to nothing. This is a *wrong path* (`from src.journal import X` when
  the module is `src/ledger/journal.py`), not a missing dependency: no
  `pip install` fixes it, and the message names the file that does exist.

One class no check can catch for you: **two tasks defining the same public
name.** On the reference build the `ledger` task was told to re-export
`JournalEntry` from a module another task already owned, and the Builder
satisfied the frozen test by *duplicating* the class instead of importing it.
The gate passed, because the gate only knows what the test checks. Read the
objectives for repeated class and function names.

### Step 5d - Three others act on you rather than being set by you

- **the circuit breaker** (`BREAKER_TOKENS_PER_WINDOW`) pauses a project that
  completes zero tasks while spending quota, and writes the evidence into
  `pause.flag`. If a project you expected to build is idle, read that file
  before assuming the orchestrator is broken.
- **auditing independence** (`AUDITING_ROLES` in `agent_roles.py`) keeps every
  role that judges Builder output off the Builder's model *and* its family. It
  is best-effort: on a free tier only one family is often healthy, so a
  collapse is possible and is logged as `reviewer_independence_unavailable`
  with `spec_critic_done.judgement_independent` recording whether that call
  actually ran degraded.
- **the doom-loop halt.** `steering.py` detects a failure signature recurring
  three times or a byte-identical diff twice, and that now **stops** the task
  rather than only reporting it: the attempt loop halts and escalates with a
  diagnosis. If you see `doom_loop_halts_attempts`, the task will not be retried
  again — read its `last_feedback` and fix the cause. `doom_loop_unattributed`
  means a loop is running that no task's `error_signature` claims, which is the
  state that makes a loop invisible; set the signature.

### Step 6 - Start and operate

```bash
python main.py                # headless loop
python main.py --dashboard    # orchestrator + TWO separate windows:
                              #   feed window -- per-project panels + event
                              #                 stream (watch-only)
                              #   command console -- pause/resume/status/
                              #                 skip/retry/note/ask/exit;
                              #                 'ask' answers questions via
                              #                 the super_orchestrator role
```

Watch `logs/agent_log.jsonl`, `.system/manifest.json`, and
`projects/NAME/.agent/cycle_summary.json` (the dashboard shows all of it).
Expect ~2–3 minutes per task in live mode. `pause.flag` in the workspace
root pauses everything; delete it to resume (the dashboard's
`pause`/`resume` commands do this for you).

### Step 7 — Handle escalations

When a task logs `task_escalated` (3 failed cycles): read
`last_feedback` in the task board. Typical remedies, in order:

1. Environment problem (missing dep, broken test) → fix, reset task to
   `pending`, `retry_count: 0`.
2. Task too big / objective ambiguous → split it smaller, sharpen the API
   spec, update its test file, reset to `pending`.
3. Repeated reviewer rejections of a green diff → usually a real quality
   issue in the code; re-scope or strengthen the objective.

**v2.9:** `last_feedback` now opens with a **diagnosis** rather than a stack
trace — `DIAGNOSIS: <cause> (confidence N.NN) -> <remedy>`, the evidence
behind it, and `[applied]` if a remedy was written. The cause vocabulary is
fixed (`spec_ambiguous`, `spec_wrong`, `missing_module`, `env_broken`,
`scope_too_large`, `reviewer_noise`, `model_capability`, `unknown`) precisely so
repeated causes cluster: three tasks diagnosed `spec_ambiguous` surface as one
systemic block instead of three unrelated-looking incidents.

So the first question is no longer "what broke" but "which bucket":

* `spec_ambiguous` / `spec_wrong` → the specification is the problem. Split it,
  sharpen the API, or pin the return shape with `-> Type`. This was the
  dominant cause in the reference build; the implementation was never at fault.
* `missing_module` / `env_broken` → an import, path or dependency problem.
* `scope_too_large` → one task is asking for more than a single pass.
* `reviewer_noise` → the deterministic gates passed and the objection is not
  supported by evidence; treat the rejection as the suspect.
* `model_capability` → the model could not do this; consider a narrower task.
* `unknown` → the evidence did not support a conclusion. Read it yourself.

Only `narrow_objective` and `split_task` can be applied automatically, only
within `validation.BOARD_WRITE_ALLOWLIST`, and only above
`DIAGNOSIS_MIN_CONFIDENCE`. Everything else is advice.

**v2.5.5:** an escalated task is auto-requeued to `pending` once after
`FAILED_RETRY_COOLDOWN_S` (default 600 s) before it stays `failed` for
operator action — the task's `auto_requeues` field tracks that budget
(`FAILED_AUTO_REQUEUES`, default 1). A dashboard/operator `retry` grants
a fresh budget. While a failure starves its dependents, the manifest's
per-project `board` block shows `blocked` counts and `failed_ids`, and
the dashboard feed has a `blocked` column — an idle loop means "starved,
operator needed", not "dead".

Never leave a failed task's files in the tree — escalations roll back
automatically, but if you edited anything by hand, restore it before
resetting.

## 4. TaskSpec schema reference

Source of truth: `validation.py`. Board file:
`projects/<name>/.agent/task_board.json` → `{"tasks": [ ... ]}`.

| Field | Type / Default | Meaning |
|---|---|---|
| `id` | str, required | Unique, stable; referenced by `dependencies` |
| `title` | str, required | Short human label |
| `objective` | str, required | Exact API + behavior spec (the contract) |
| `target_files` | list[str], **required** | Files the diff may touch — enforced by the scope guard (checked before *and* after apply). Credential-named entries (`.env*`, `*.pem`, `id_rsa`, `kaggle.json`, `*-secret.yaml`) are refused by `patcher.path_is_denied` |
| `dependencies` | list[str] | Ids that must be `done` first |
| `priority` | `high` \| `medium` \| `low` | Scheduling order within eligible tasks |
| `acceptance_criteria` | list[str] | Checkable conditions the Reviewer sees |
| `workspace_patterns_referenced` | list[str] | **Retired in v2.8** — ignored; omit it |
| `retry_count` | int = 0 | Managed by the orchestrator |
| `status` | `pending` \| `in_progress` \| `done` \| `failed` \| `blocked` | Managed by the orchestrator |
| `last_feedback` | str \| null | Compact failure feedback from the last cycle |
| `error_signature` | str \| null | Failure-shape key (`ERR_<DOMAIN>_<NAME>`): drives steering doom-loop + systemic-block detection |
| `task_type` | `feature` \| `bugfix` | Routes the Reviewer model; bugfix verdicts demand a regression test + stated root cause |
| `should_pass` | bool = true | DEMO_MODE simulation switch only |
| `requires_new_tests` | bool = false | Tests-first gate: final passed-count must exceed the pre-task baseline |
| `completed_at` | str \| null | Set on completion |

Complete example (from `project_main`):

```json
{
  "id": "TASK-7002",
  "title": "Quant node: z-score + momentum signals",
  "objective": "Tests-first: implement src/quant/signals.py so the pre-seeded tests/test_signals.py passes. Exact API: compute_z_score(prices: pd.Series, window: int = 20) -> float ... [full signatures, rules, edge cases]",
  "target_files": ["src/quant/signals.py"],
  "dependencies": ["TASK-7001"],
  "priority": "high",
  "acceptance_criteria": ["tests/test_signals.py collected and fully green (no skips)", "entire pytest suite green", "no network access"],
  "retry_count": 0,
  "status": "pending",
  "last_feedback": null,
  "error_signature": "ERR_QUANT_SIGNAL_MATH",
  "task_type": "feature",
  "should_pass": true,
  "requires_new_tests": true,
  "completed_at": null
}
```

> **Kaggle campaigns only:** `task_type: "experiment"` tasks also carry a
> `metric` block and a `run_command`. Since v2.7 the domain pack validates
> both, because they are model-authored and `run_command` executes with
> `shell=True` in a fully-populated environment: `target_files` entries are
> filtered through the patcher's path rules, and `run_command` must be a plain
> `python`/`py` invocation with no chaining, substitution or redirection
> (`env > reports/metrics.json` would pass the objective gate and then be
> submitted to Kaggle with every credential in it). Refusals log
> `campaign_target_file_rejected` / `campaign_run_command_rejected`; set
> `KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND=true` to opt out. See `KAGGLE_SETUP.md`.

## 5. Hard-won lessons (do not relearn them)

1. **Builder model quality is the bottleneck.** Weak builders produce
   tool-call markup, prose, or ghost diffs. Route a strong codegen model to
   `builder_initial`; keep cheap models for review/patch.
2. **Single-file tasks with pinned APIs and pre-written tests complete
   first-try.** Multi-file vague tasks escalate. Decompose ruthlessly.
3. **The suite must never touch the network** — determinism of the gate
   depends on it.
4. **Approved modules are sacred.** The scope guard and rollback protect
   them; do not hand-edit files in `src/` between runs.
5. **The Reviewer is your quality signal — but calibrate it.** It must
   default to APPROVED when real tests pass; taste-based rejections of
   green code blocked convergence until calibrated.
6. **Content-mode is what makes cheap builders viable** — diff-format
   output was the structural barrier, not intelligence.
7. **NEVER delete `projects/` during a demo/regression pass.** User
   projects live there. Reset only the sample boards + shared state.
8. **Watch out for models writing `sys.exit()` / `raise SystemExit` at
   module level (especially in conftest.py)** — it silently kills the
   pytest runner (exit 4, empty output). Auto-rejected by content checks.
9. **ONE orchestrator per workspace — enforced.** Two suites (e.g. a
   headless run plus a `--dashboard` run) race on task boards, snapshots
   and the manifest; the losing loop's rollback then *resurrects* the
   winning loop's applied files (observed: a rolled-back module that kept
   coming back after every `snapshot_restored`). `main.py` now holds an
   OS-level `orchestrator.lock`; a second orchestrator-bearing start
   refuses with exit code 2. Watch-only `--feed`/`--cmd` windows are
   exempt and safe to open anytime.
10. **JSON caches must receive JSON-safe payloads.** A storage recipe like
   `frame.reset_index().to_dict(orient='list')` produces `pd.Timestamp`
   values that `json.dumps` cannot serialize — `Cache.set` raises
   `TypeError`, the task's own cache tests fail, and every attempt burns.
    Pin the recipe: stringify datetime index columns to ISO-8601
    (`.isoformat()`) before `cache.set`; rebuild with `pd.DataFrame` +
    `pd.to_datetime` + `set_index` on hits. (Cost three failed cycles
    of TASK-7003B before being diagnosed.)
11. **Pin exact import paths for cross-task types.** When a task uses a
    type built by a dependency (e.g. `OrderPlan` from TASK-7005A), state
    the import verbatim (`from src.agents.models import OrderPlan`) — a
    guessed wrong path (`src.broker.models`) makes the test file fail
    COLLECTION, which the new-tests gate reports as a confusing
    `passed == baseline` block.
12. **Read pytest output carefully: `ERROR tests/test_x.py` is a
    collection failure (import-time), `FAILED` is an assertion.** With
    `--continue-on-collection-errors` both fold into the failure count,
    but they need opposite remedies: sharpen imports/objective for
    errors, fix logic for failures. A gate block with
    `passed == baseline` almost always means the task's own test file
    never activated — check for collection errors first.
13. **Escalation is a signal to re-read the SPEC, not to swap models.**
    All three first-run escalations in the reference project's final
    chain were fixed by sharpening the objective (recipe, import path,
    prompt contract) — after which each task passed on the first try
    with the same cheap builder model.
14. **Never put a real credential in a place the system reads back.**
    `task.objective`, `acceptance_criteria`, `context.md`, and any
    comment inside a `target_files` file are all concatenated into model
    prompts and written to the JSONL log. Live values are blanked
    automatically (`redaction.py`), but only values the process already
    holds — a key you invent for the exercise is not one of them. Use
    `os.getenv("MY_KEY")` in specs and fixtures; keep the real value in
    `.env` only.
15. **A `target_files` entry decides what the Builder may write** — it is
    the scope guard's allow-list. For campaign/task specs, keep it to the
    single source file under test. Anything credential-named
    (`.env`, `*.pem`, `id_rsa`, `kaggle.json`, `*-secret.yaml`) is refused
    outright by `patcher.path_is_denied`, and the domain pack drops such
    entries from model-proposed hypotheses
    (`campaign_target_file_rejected`).
16. **Pre-existing tests are a frozen baseline (v2.8).** Listing a test file
    in `target_files` no longer buys permission to rewrite it:
    `test_integrity.py` refuses to modify or delete any `tests/**`,
    `conftest.py`, `pytest.ini`, `pyproject.toml`, `tox.ini` or `setup.cfg`
    that existed when the task started — which is what made the real-pytest
    gate advisory before. **New** test files are still fine, so tests-first
    works normally. The single exception is the bootstrap task that *creates*
    the harness: set `"allow_test_edits": true` on it, and nowhere else.
    `TASK-7001` is the one board entry that carries the flag today.
17. **Give every task an `error_signature`.** Since v2.8 it is not
    bookkeeping — it is how `steering.py` notices that three tasks are dying
    the same way. Without it, a doom loop is invisible: the system retries
    three unrelated-looking failures until the budget runs out.
18. **Before any commit, run `python main.py --preflight scan`** (exit 1
    means findings). It reports rule names and `sha256` hints, never
    values, so triage is safe to do on a shared screen.
19. **If the Council pauses a project, read `pause.flag`'s companion log line
    first.** `council_pause_project` carries the deterministic signal that
    triggered it (a doom loop, a deadlock, a burn-rate breach). Remove
    `pause.flag` to resume. The Council has *process* authority only — it can
    never approve or reject your code, so a pause always means "the loop is
    stuck", not "the work is wrong".

## 6. Reference implementation

`projects/project_main/` — a production-grade multi-agent AI stock trading
system built with this exact procedure: 20 tasks (TASK-7001..7013B; a
two-file integration task escalated three cycles and only converged after
being split into two single-file tasks), 19 pre-written test files (113
tests activating progressively), context.md as the standing brief. Read
its board, tests, and context before onboarding your first project.
