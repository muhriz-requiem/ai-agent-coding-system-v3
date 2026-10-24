# Project: logtriage

## Goal

A diagnostic library that reads the orchestration harness's own JSONL event log
(`logs/agent_log.jsonl`) and answers the questions that are expensive to answer
by hand: which roles are slow, which failures repeat, which providers are
unreliable, and where the tokens went. Built to be run against the *real*
reference log, whose 36k+ events are the ground truth for every expected value.

## Architecture

Pure-Python, no third-party runtime dependencies beyond the standard library.
`statistics` is avoided in favour of hand-rolled percentiles so the numbers are
deterministic and the interpolation rule is explicit.

```
src/triage/
  __init__.py     package marker only
  reader.py       JSONL ingestion + tolerant normalisation
  latency.py      per-role / per-model duration percentiles and outliers
  failures.py     failure clustering by normalised signature
  providers.py    provider reliability, cooldowns, fallback chains
  cost.py         token accounting from llm_usage + llm_exchange
  report.py       assembles a report dict from the four analyses
```

Data flows one way: `reader` normalises raw rows into `Event` dicts, the four
analysis modules each take `list[Event]` and return a plain dict, and `report`
composes them. No analysis module imports another.

## Hard rules

1. **Zero network, zero clock, zero filesystem in the analysis functions.** Every
   analysis module takes data in and returns a dict. `reader` is the only module
   that touches the filesystem; tests monkeypatch nothing because there is
   nothing to patch.
2. **Deterministic output.** Given the same input list, every function returns
   byte-identical output. No set iteration order, no dict ordering dependence on
   insertion luck, no `random`. Sort everything explicitly.
3. **Never raise on malformed input.** The reference log contains at least one
   row with a non-ISO timestamp (`{"ts": "probe"}`) and rows written by an
   older schema version. A diagnostic tool that crashes on the log it is meant
   to diagnose is useless. Malformed rows are *counted*, not fatal.
4. **Empty input is a valid input.** Every function returns a well-formed empty
   result for `[]` — never `None`, never an exception, never a division by zero.
5. **Percentiles use nearest-rank, not interpolation.** For a sorted list `xs`,
   `p(xs, q)` is `xs[ceil(q/100 * len(xs)) - 1]` clamped to the ends. This is
   stated because the tests pin it exactly and a different convention would
   fail them.
6. **No floating-point equality in comparisons.** Use exact int counts for
   anything that gates a decision; floats are only ever *reported*, and tests
   assert them with `pytest.approx`.

## v1 scope (build this, in this order)

1. `reader` — load and normalise.
2. `latency` — percentiles and outliers.
3. `failures` — clustering.
4. `providers` — reliability.
5. `cost` — tokens.
6. `report` — composition.

## Explicitly out of scope for v1

- Writing any file. `report.py` returns a dict; it does not print or save. A
  `--json` CLI flag belongs to v2, and even then it lives outside `src/`.
- Streaming / incremental parsing of a growing log. v1 reads the file once.
- Any pandas/numpy dependency.
- Cross-run comparison ("is this run worse than last week") — needs storage we
  do not have.

## Build order

`reader` has no dependencies and everything else needs it, so it is first.
`report` needs the other four, so it is last. `latency`, `failures`,
`providers` and `cost` are independent of each other and may be built in any
order after `reader`.

## The real log is the acceptance criterion

`projects/logtriage/tests/` must stay green as modules land (the self-skip
guard). Once `report.py` exists, `python -c "from src.triage.report import
analyse; ..."` run against the workspace's own `logs/agent_log.jsonl` must
produce a report whose `events.total` matches the real line count and whose
`failures.top_shapes` is non-empty. That end-to-end check is the point of the
project; unit tests alone do not count as done.