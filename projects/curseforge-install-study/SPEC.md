# SPEC DRAFT (evidence-backed) — package installer with SDK-pattern service API

Written 2026-10-16 from the static REA study in `research/FINDINGS.md`
(Evidence: app analysis `ev_e9842d9a…7797`, traces `ev_00593811…51b1` /
`ev_4029f904…eaaa`, bundle `research/evidence-bundle.json`). This is the
operator-reviewed input for `--new-project`; the machine-authored spec by
`spec_author` remains the binding contract.

## Build command (next session, after operator approval)

```
python main.py --new-project modinstall --goal "Build a package installer:
resolve a dependency manifest, download artifacts with SHA-256 verification,
place files transactionally with rollback, and expose a renderer-side service
API in the SDK pattern observed in CurseForge (template `${Service}.{method}`
channels, `renderer:on:` handler registry)" --author-plan
```

Then READ the authored plan before building (AGENTS.md rule: machine-authored
specs are a first draft, not a contract you trust).

## Requirements, each pinned to evidence

R1 — **Renderer-side service API (SDK pattern, from the `install` trace).**
Channels follow `` `${Service}.${method}` `` with service methods at minimum:
`installVersion`, `getVersion`, `getVersions`, `getLatest`, `getConfiguration`,
`addListener`; update-notification methods (`onShowUpdateNotification`,
`shouldShowUpdateNotification`, `performUpdate`). Handlers subscribe through a
`renderer:on:` registry step.
*Evidence pin: trace `ev_00593811…` seed match (ipc-channel
`${this._nameOfService}.installVersion`) + handler identity keys
`renderer:on:` (2 nodes, byte-range identity).*

R2 — **DESIGN IMPROVEMENT over the observed app: pairable channels.**
CurseForge's fully-dynamic channel names are why 12 dynamic IPC operations
paired with 0 static handled channels and `native_handoffs: 0` — the app is
statically unreadable at the seam. Our spec REQUIRES literal channel names
(or a registry table derivable by static analysis). Rationale: our own
tooling (Phase-4 research lane) grades builders against static evidence; a
project whose IPC seam is dynamic would grade itself dark.

R3 — **Integrity policy = record-and-continue, not fail.**
Per-artifact sha256 at download; on mismatch: quarantine the artifact,
record a structured finding (declared vs observed digest), and continue the
pipeline for OTHER artifacts (mirrors REA's posture which recovered CurseForge's
JS layer after `fail` aborting the whole run on 2 native EXEs — our research
lived through it: `ev_d49ca868…47688` empty envelope first, success second).

R4 — **Transactional install with rollback.** Resolve → download → verify →
place with a pre-install snapshot restore on failure (mirrors the workspace
orchestrator's `patcher.restore_snapshot`, `objective_gate_block` semantics).

R5 — **Explicit third-party call allowlist.** The observed app carries a
hard dependency on `widget.freshworks.com` from its renderer. We require an
explicit, config-listed allowlist for any external call the renderer may
make; unknown hosts are declined (egress gate discipline, `executors.py:572`
convention).

R6 — **Verification against this evidence.** The authored tests must pin:
channel-name shape (R1), quarantine-not-abort (R3), rollback (R4), allowlist
blocking (R5). Cite `research/FINDINGS.md` section names in the tests' doc
comments so review probes tie assertions to the study.

## Test seeds (new-file names for the builder)

- `tests/test_service_channels.py` — R1/R2 shape + registry.
- `tests/test_integrity_quarantine.py` — R3 record-and-continue.
- `tests/test_transactional_install.py` — R4 rollback on failure mid-place.
- `tests/test_call_allowlist.py` — R5 decline unknown host.

## Non-goals

- No CurseForge internals replication beyond the documented channel surface
  (native agent behavior stays out of scope — evidence `FINDINGS.md`
  "What static analysis bounds here").
- No Minecraft coupling: generic "package installer" semantics.
