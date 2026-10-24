# Kaggle Competition Setup — Quickstart

This file captures the state of the Kaggle domain pack (v2.6) so future coding-agent sessions can pick it up without re-reading the full codebase.

## Prerequisites

| Item | Where | How |
|---|---|---|
| Kaggle API token | kaggle.com → Settings → API → Create New Token | `C:\Users\muhri\.kaggle\kaggle.json` |
| Kaggle CLI | `pip install kaggle` | Already installed (added to `requirements.txt`) |
| Competition rules | kaggle.com — accept on the competition page | Required before submit works |
| Competition data | `kaggle competitions download -c <slug>` → `projects/<name>/data/raw/` | Or manual download |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | In `D:\workspace\.env` as commented placeholders | Uncomment and fill if you prefer env vars over the file |

## Scaffold a campaign

```
python main.py --new-kaggle <project_name> --competition <slug> --metric <name> --metric-direction <maximize|minimize>
```

Example:
```
python main.py --new-kaggle titanic_campaign --competition titanic --metric accuracy --metric-direction maximize
```

Creates `projects/<project_name>/` with:
- `data/raw/`, `data/processed/`, `experiments/`, `reports/`, `submissions/`, `tests/`
- `.agent/domain.json` (pack configuration)
- `.agent/campaign.json` (state: best CV, experiment log, hypothesis queue)
- `.agent/context.md` (competition brief, CV/split rules)
- `.agent/divergence.json` (opt-in divergence, default off)
- `requirements.txt` (pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna, kaggle)

## Data

Place raw data in `data/raw/`. The pack expects:
- `data/raw/sample_submission.csv` — used by the submission validator (row count, header, id set, NaN checks)
- Training/test files named however the competition describes them (referred to in experiment scripts)

## Fill the task board

The scaffold seeds 3 starter hypotheses in `campaign.json` (`pending_hypotheses`):

1. **Baseline** — `experiments/exp_baseline.py`: load data, build a deterministic split, train a regularized baseline, write OOF metric to `reports/metrics.json`
2. **Feature engineering** — `experiments/exp_features.py`: same split hash, engineered features, gradient-boosted model

Each hypothesis becomes a `TaskSpec` with `task_type: "experiment"``, a `metric` block (name, direction, target, split_key, required_fields), and a `run_command`.

You can push additional hypotheses into `campaign.json["pending_hypotheses"]` manually, or flip the LLM campaign director (see below).

## Run the campaign

```
python main.py                     # headless loop (Ctrl+C to stop)
python main.py --dashboard         # orchestrator + event feed + command console
python main.py --once              # single pass (useful for one experiment)
```

When a task completes, the campaign records the metric and (if improved) attempts submission under `KAGGLE_AUTO_SUBMIT`.

### Auto-generation modes

| Env var | Effect |
|---|---|
| `CAMPAIGN_AUTO_GENERATE=true` | When the board starves, packs propose the next hypothesis from the static queue (`pending_hypotheses`). The starter queue has 2 entries. Add more hypotheses manually to continue. |
| `CAMPAIGN_DIRECTOR_ENABLED=true` | After every experiment result, an LLM agent reads the campaign state (best CV, experiment log) and generates 1–3 new hypotheses. The campaign becomes fully autonomous — no human queue editing needed. |

Both are off by default. Start with one or the other.

## Submission flow

1. An experiment completes with a CV improvement
2. Format validation: `submissions/submission.csv` vs `data/raw/sample_submission.csv` (columns, row count, id set, no NaN)
3. Cap check: `KAGGLE_DAILY_SUBMISSION_CAP` (default 5/day)
4. `kaggle competitions submit -c <slug> -f submissions/submission.csv --wait <N>`
5. Submissions logged to `campaign.json["submissions"]` for CV-vs-LB drift tracking

Set `KAGGLE_AUTO_SUBMIT=true` only after a human has verified one submission pipeline end-to-end.

## Known env knobs relevant to Kaggle

| Env var | Default | Purpose |
|---|---|---|
| `KAGGLE_AUTO_SUBMIT` | `false` | Enable auto-submission when CV improves |
| `KAGGLE_DAILY_SUBMISSION_CAP` | `5` | Max submissions per UTC day |
| `KAGGLE_MIN_CV_IMPROVEMENT` | `0.0` | Minimum metric delta to trigger a submission |
| `KAGGLE_SUBMIT_WAIT_S` | `600` | `--wait` timeout for scoring |
| `CAMPAIGN_AUTO_GENERATE` | `false` | PoP hypothesis queue when board starves |
| `CAMPAIGN_DIRECTOR_ENABLED` | `false` | LLM generates next hypotheses from results |
| `CAMPAIGN_MAX_NEW_TASKS` | `3` | Max tasks per proposal batch |
| `DIVERGENCE_ENABLED` | `false` | Per-experiment candidate fan-out (opt-in per task) |
| `KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND` | `false` | Allow an arbitrary shell string as `run_command` |

## Secrets and submissions (read before enabling auto-submit)

Two gates protect the Kaggle egress path. Both are **fail-closed**.

1. **Submission secret gate** — `submissions/submission.csv` is written by a
   model-authored experiment script, so its contents are model-determined.
   Before `kaggle competitions submit`, the file is scanned for configured
   secret values and credential-shaped patterns; any hit refuses the
   submission and logs `kaggle_submission_blocked_secrets`. Rule names and a
   `sha256` hint are reported — never the value.
2. **`run_command` validation** — each hypothesis's `run_command` is executed
   in a fully-populated environment. By default it must be a plain
   `python`/`py` invocation: chaining (`;`, `&&`, `|`), substitution
   (`` ` ``, `$()`) and redirection (`>`, `<`) are refused, and the runner must
   be an allowed interpreter. Without this, a hypothesis such as
   `env > reports/metrics.json` exits 0, leaves a parseable
   `reports/metrics.json`, **passes the objective gate** — and then uploads
   every credential in the environment to the competition.

   A rejected command is stored as `null`, which fails the objective gate
   closed with `task declares no run_command`. Refusals log
   `campaign_run_command_rejected` with the reason.

   **v2.8 closes the same hole on the generic path.** `executors.run_objective`
   used to pass a board-authored string straight to `shell=True`, so a
   non-Kaggle task's `run_command` had no metacharacter check at all. It now
   parses to an argv and runs with `shell=False`, so shell syntax has no
   meaning; anything containing a metacharacter is refused outright
   (`objective_run_command_refused`).

   If a campaign legitimately needs shell features, set
   `KAGGLE_ALLOW_ARBITRARY_RUN_COMMAND=true` — but understand what you are
   re-enabling.

   Similarly, a hypothesis's `target_files` list (which decides what the Builder
   may write) is filtered twice: through the patcher's path rules (entries
   naming `.env`, `*.pem`, `id_rsa`, `kaggle.json` or credential-named files
   are dropped, anything outside the project is refused) **and** through the
   v2.8 test-integrity baseline, so a hypothesis cannot claim a pre-existing
   test or pytest config as writable. Refusals log
   `campaign_target_file_rejected`.

3. **Self-healing installs plain packages only (v2.8)** — when the suite fails
   on a missing import, `env_autoheal` runs `pip install -r requirements.txt`
   with the harness's live environment. A `requirements.txt` is
   Builder-influenced, so editable (`-e .`), VCS and direct-URL requirements
   are refused before pip runs (`env_autoheal_requirement_refused`) — each of
   those executes arbitrary code at install time.

Operator check before any commit or submission:

```powershell
python main.py --preflight scan      # exit 1 = findings; values are never printed
```

See "Secrets and egress" in `README.md` for the full boundary map.

## How free LLM routing works for Kaggle

The system routes every experiment-planning and code-generation call through the free multi-provider capacity layer. The router picks models by role capability + observed health + quota headroom. For experiment scripts (long-form content-mode generation) it avoids Groq (8k TPM too small) and prefers Gemini, Cloudflare, Z.ai, NVIDIA or OpenRouter models with large output windows.

The LLM campaign director uses the same free routing. It produces JSON (array of hypothesis objects), so the router filters for models with `structured_outputs` or `response_format` support — satisfied by qwen3.8-27b:free, nemotron-3-super:free, apodex-mini:free, etc.

## Monitoring

```
python main.py --models status         # provider credentials, quota headroom, top builders
python main.py --models verify         # one live call per provider (key/endpoint check)
python main.py --models probe          # bounded canaries: JSON adherence, fence emission
python main.py --campaign-status <proj> # campaign state: best CV, experiment log, submissions
tail -f logs/agent_log.jsonl           # live event stream (candidate_scored, objective_gate_*, kaggle_submit_*)
```

## First campaign checklist

```
1.  mkdir -p ${USERPROFILE}\.kaggle
    # paste the API token into C:\Users\muhri\.kaggle\kaggle.json
2.  Accept the competition rules on kaggle.com
3.  python main.py --new-kaggle my_comp --competition <slug> --metric <name> ...
4.  Download data: kaggle competitions download -c <slug> -p projects/my_comp/data/raw/
    # (or manually place CSV files)
5.  # Decide: static queue or LLM director? Uncomment one:
    #   CAMPAIGN_AUTO_GENERATE=true
    #   CAMPAIGN_DIRECTOR_ENABLED=true
6.  python main.py --once                    # run the first experiment batch
7.  python main.py --campaign-status my_comp  # see results
8.  # Once the pipeline works, set KAGGLE_AUTO_SUBMIT=true and run full loop
```