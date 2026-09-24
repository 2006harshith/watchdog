---
name: save
description: End-of-session save for the watchdog repo. Runs tests, updates the PROGRESS.md session log, commits, pushes if green; with an SP argument (e.g. /save SP3) also closes and tags that save point.
disable-model-invocation: true
argument-hint: "[SPnn to close a save point]"
---

End-of-session save. Arguments: "$ARGUMENTS" (empty = ordinary session save; e.g. "SP3" = close SP3).

1. Run `uv run pytest -q` and `uv run ruff check .` (skip only if `pyproject.toml` does not exist yet).
   Record the result.
2. Append to "Session log" in `PROGRESS.md`, one entry:
   `YYYY-MM-DD — done: … — next concrete step: … — open questions: …`
   Make the next step concrete enough that `/resume` can act on it without guessing.
   Update "Current position" to match.
3. Before staging, run `git status --short` and check: no `.env`, no data files (parquet, jsonl, csv
   over ~1 MB), nothing from `mlruns/` or `.venv/`. If anything like that shows up, stop and ask.
4. Commit with a message that says what changed. If tests failed, prefix the message with `[wip]`.
5. Push only if tests and ruff passed. If they failed, don't push; say why.

If an SP argument was given, before step 4 also:
a. Check the save point's "done when" criteria (PROGRESS.md table and the SP prompt/spec). List each
   criterion with met / not met.
b. If the save point has a core piece (splits, features, calibration/threshold, metrics), the
   "Teach-backs" table must show a PASS for it. If not, refuse to close: the next step is the teach-back.
c. Only if a and b are fully met: set the SP status to "done" in the table, then after the commit create
   the tag `spNN-<short-name>` (e.g. `sp03-e1`) and push the tag with the commit.
d. Remind the owner: re-upload PROGRESS.md to the Claude Project, plus any results table for a gate.

Finish with a three-line summary: what was saved, pushed or not, and the next step.
