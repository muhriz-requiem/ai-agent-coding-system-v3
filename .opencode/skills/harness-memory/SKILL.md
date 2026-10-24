---
name: harness-memory
description: Distill durable lessons from this session into the workspace instinct file, or read instincts before starting work. Use at session end, after a debugging victory, or when starting a new session in D:\workspace.
---

Harness-side memory for sessions working in D:\workspace. This is the layer ABOVE
the orchestrator's build-side pattern stores (`.shared_workspace/patterns.json` and
`anti_patterns.json`) — do not write project-code patterns here, and do not write
session instincts there.

## Appending (session end / lesson learned)

Append to `D:\workspace\memory\instincts.md`, respecting the file's format:

- One bullet per lesson, newest at the top under `## Instincts`.
- Format: `- YYYY-MM-DD — <lesson in one line> (source: <file:line or event>)`.
- Include only what generalizes to future sessions: environment traps (e.g. what
  breaks npm/exe files on this machine), repo conventions that surprised you,
  workflows that measurably helped.
- Never store secrets, tokens, or credential-shaped strings. Never store
  machine-specific values that are already in config (paths in `.env`).
- The file is append-only — do not rewrite or "clean up" prior entries. If a
  lesson became wrong, append its correction referencing the date; never edit
  history.

## Reading (session start)

Read the last ~30 instincts before planning anything nontrivial in this
workspace. An instinct contradicted by current reality outranks your prior —
verify, then append the correction.
