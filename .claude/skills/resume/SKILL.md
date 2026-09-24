---
name: resume
description: Resume work on the watchdog project from the last save point. Use at the start of every session.
disable-model-invocation: true
---

Resume the project. Do these in order and do not start any new work:

1. Read `PROGRESS.md` and `CLAUDE.md`.
2. Run `git status` and `git log --oneline -5`. If there are uncommitted changes, list them and ask whether to keep or discard them. Do not discard anything yourself.
3. If `pyproject.toml` exists, run `uv run pytest -q` and report pass/fail counts. If tests fail, say so first.
4. If any entry under "Teach-backs" in `PROGRESS.md` is "pending" or "fail", the next step is that teach-back, before anything else.
5. Reply in at most five lines:
   - current save point and its status
   - what the last session finished
   - test status
   - the exact next step (a command or a prompt)
   - whether the next step needs a spec from the chat Project first (true for the first step of a new save point)
