---
name: resume
description: Start-of-session check-in for the watchdog repo. Reads PROGRESS.md, recent commits and working-tree state, runs the tests, and states the exact next step.
disable-model-invocation: true
---

Start-of-session routine. Do these in order and do not start any new work.

1. Read `CLAUDE.md` and `PROGRESS.md` in full.
2. Run `git status --short` and `git log --oneline -5`.
3. If `pyproject.toml` exists, run `uv run pytest -q` and `uv run ruff check .`. If it doesn't exist,
   say the project is before SP0 and skip this step.
4. If there are uncommitted changes, list them and ask whether to keep them (continue the work) or discard
   them. Never discard anything without an explicit "discard".

Then reply in at most five lines:
- Save point and its status
- Tests: pass / fail (which) / not set up yet
- Uncommitted changes: none, or a short summary
- Pending or failed teach-back, if any
- The exact next step

Rules for choosing the next step:
- A pending or failed teach-back is always the next step (CLAUDE.md hard rule).
- If the next step is the first step of a new save point and PROGRESS.md has no prompt or spec for it,
  say: "Get the SP spec from the Claude Project first" and stop.
- Otherwise take the "Next concrete step" from PROGRESS.md.
