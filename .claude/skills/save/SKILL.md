---
name: save
description: Save progress on the watchdog project — run tests, update PROGRESS.md, commit, and tag a finished save point. Use at the end of every session.
disable-model-invocation: true
---

Save the session. Argument (optional): a finished save point id such as `SP3`. Given: $ARGUMENTS

1. Run `uv run pytest -q` and `uv run ruff check`. Report the result. If either fails, still save, but mark the session log entry "TESTS FAILING" and do not push or tag.
2. Update `PROGRESS.md`:
   - "Current": save point, today's date, and the next concrete step (a command or prompt someone could run cold).
   - Prepend one line to "Session log": date · SP · what changed · tests pass/fail · next step.
   - If a core piece (splits, feature definitions, calibration/threshold, metrics) was written or changed this session and has no passed teach-back, add it under "Teach-backs" as "pending".
   - Add any unresolved question to "Open questions".
3. Show the `git diff --stat` and the proposed commit message, then commit all tracked changes plus new files under src/, tests/, scripts/, configs/, docs/, results/. Never add .env or data files.
4. If an argument like `SP3` was given AND tests pass AND no teach-back for that save point is "pending" or "fail": set that row's status to "done" in the save point table, create the tag `spNN-<short-name>` (ask for the short name if unclear), and set "Current" to the next save point. If a teach-back is pending or failed, do not close the save point: say which one, and offer to run it now.
5. Push commits (and the tag, if created) only if tests pass.
6. End with a two-line summary: what was saved, and the next step. If a save point finished, remind the owner to re-upload `PROGRESS.md` to the chat Project.
