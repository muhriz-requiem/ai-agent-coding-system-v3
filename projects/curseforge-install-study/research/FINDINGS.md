# CurseForge Windows — mod-install static study (Phase 5, v3)

Written 2026-10-16 by the v3-research-lane session. Target the operator picked:
the installed **CurseForge Windows** desktop app (Electron). Feature under study:
the **mod-install pipeline**, renderer-side surface. All claims below are backed
by REA Evidence IDs recorded in the session ledger and exported to
`evidence-bundle.json` (36.7 MB, 10 records, 0 residual unknowns) sitting next
to this file.

## Subject and toolchain

- Subject: `resources/app.asar`, sha256 `88337583254547f158f8817248306006035904f0fc0c238d82c736fca0cc2bb8`.
- Retained in-session application analysis: `ev_e9842d9a5b25bbc51c881d2fbcb439ee3218585611400a70a3ac76517c2f7797`
  (26 module identities paged; statistics + Electron summary cited below).
- CLI envelope (repro checkbox): `ev_d49ca8686710757d016842d997801c63455225c06f45b5affa764ab395747688`.
- Feature traces: `ev_00593811abd1bf5bb624c7dca72802543e39153c…` (seed `install`,
  unique IPC-channel hit) [full id `ev_00593811abd1bf5bb624c7dca72802543e391556c3bd35b03c4abd3371fc51b1`]
  and `ev_4029f9046622d2977f9e7cf46eec0992ede4293f1079b035ebe232035fae9eaa`
  (seed `add-content`, unique route-endpoint hit). Both sweeps saturate the same
  210 nodes / 256 edges closure — the static layer has no more to give without
  a runtime or native pass.

## Integrity findings (verbatim per REA limitations)

`plugins/curse/win/Curse.Agent.Host.exe` and `plugins/curse/win/owver64.exe`
have artifact-integrity metadata that contradicts their observed bytes; those
observed bytes are recorded as **untrusted** and were excluded from JS analysis.
Their internals were **not** studied (native engines unavailable on this
Windows host; that lane remains open).

## Application map (static, observed)

- 37 relevant files; 5 parsed JS bundles; 0 parse failures; 1,165,302 AST
  nodes visited; 164 findings.
- **electron-main entrypoint: `dist/background/background.js`**
  (identity key `electron-main:package-entrypoint:dist/background/background.js`).
- Renderers: `dist/desktop/desktop.html` (+ +widget callout to
  `https://widget.freshworks.com/widgets/9000000433.js` — observed third-party
  network dependency), `dist/background/background.html`,
  `dist/game-dashboard/game-dashboard.html`.
- `dist/preload/preload.js`: 22 lines CommonJS, one import — a deliberately
  thin preload (static IPC pairing therefore stays 1:1 for receiver-invocation
  only; splash `main_handlers: 0` and `context_bridge_apis: 0` in summary are
  consistent with the dynamic-命名 pattern below, not with an absent pipeline).
- Webpack organization: dynamic hub `import(e.module)` in `desktop.js`
  (chunks load by runtime key: `__webpack_id__` exports in `dist/514/514.js`,
  bootstrap@587), explaining why literal channel pairing is 0 while dynamic
  channel operations = 12.
- Endpoints: 63 (route endpoints incl. `projects`, ` discovers`, `index`,
  **`add-content/:modpackId`**; network candidates like `call:e.delete` are
  labeled `candidate`), storages: 22 (`local-storage`, `session-storage`).

## Install-feature surface (renderer side; traces cited)

The renderer speaks an SDK-patterned IPC service whose channels are **template
literal** names — this is the load-bearing finding for a spec:

```
`${this._nameOfService}.installVersion`     <- unique `install` seed match (ipc-channel)
`${this._nameOfService}.getVersion`
`${this._nameOfService}.getVersions`
`${this._nameOfService}.getLatest`
`${this._nameOfService}.getConfiguration`
`${this._nameOfService}.addListener`
`${m}.addListener`
`${m}.onShowUpdateNotification`
`${m}.shouldShowUpdateNotification`
`${m}.performUpdate`
```

Handler registrations (`ipc-handler` identity keys): `renderer:on:` +
the same `${this._nameOfService}.${e}` shapes at
`21:35438–21:35530` and `21:24018–21:24076` (byte ranges in whichever module
instantiates them — REA records them as artifact-local keys).

`installVersion` is the install transmission on the renderer side; the single
statically resolved invocation edge into it comes from the renderer asset node
(`preload.js` asset, relation `invokes`, confidence per trace edges).

## What static analysis bounds here (honest limits, verbatim-quoted themes)

- Dynamic IPC channel expressions remain unresolved and are never paired by
  textual similarity → **main-side handlers behind the dynamic channel names
  are unknown**; `native_handoffs: 0` in both traces.
- Runtime reachability/registration/enforcement are unproven; "unavailable
  facts remain explicit and must not be read as absence".
- Native agent EXEs (the likely download/host workhorse) untouched per
  integrity finding above.

## The next layers (operator decision)

1. **Dynamic layer (recommended next)**: operator launches the app with
   `--remote-debugging-port`, REA `inspect_electron_page` /
   `observe_javascript_runtime` passively records actual channel names,
   handler responses, and the install call sequence. No code execution by REA
   (passive CDP only) — still evidence-grade, and it closes the dynamic-命名
   gap the static layer cannot.
2. **Spec from static layer (possible now)**: the renderer-side service shape
   is already buildable — install/update subscription API with
   `${Service}.{method}` channel naming, signed-integrity downloads targeted to
   a `plugins/<game>/win/` layout with quarantined-but-recorded mismatches
   (the app's own integrity posture, evidenced above).
3. **Native lane**: needs Ghidra headless (marked experimental P0 on Windows)
   — defer until the dynamic pass creates specific questions about
   `Curse.Agent.Host.exe`.
