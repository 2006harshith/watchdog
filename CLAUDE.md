# watchdog — instructions for Claude Code

## What this repo is
A supervisor agent for LLM agents. Its core is a small trained model that reads a partial agent run
and outputs a calibrated probability that the run has already failed, using model-independent step
signals (tool-call structure, state changes, repetition, verifier results). It must keep working
after the underlying LLM is swapped, recalibrating from a few healthy runs without failure labels.

Current position: always read `PROGRESS.md` first. The plan (save points SP0–SP11) lives in the
"Watchdog — Project Blueprint" doc; the save point table is summarised in `PROGRESS.md`.

## Who writes what
You write all the code. The owner (Harshith) is practising AI-directed development: he specs,
reviews every diff, and must be able to defend the core ML in an interview.

Core pieces — these need a TEACH-BACK before their save point can close:
- train/test split logic (held-out model family, task-disjoint)
- feature definitions (which signals are model-independent)
- calibration and recalibration threshold logic
- metrics (AUROC, AUPRC, detection at fixed false-alarm rate, ECE) — always tested against scikit-learn

For a core piece, in this order:
1. Write it (following the save point's spec) and its tests.
2. Run the tests and the relevant script until everything passes with no errors. Do not start the
   teach-back on broken code.
3. Walk through the lines that matter (why, not what), including the design choice and the rejected alternative.
4. Ask three interview-style questions about the mechanism. He answers in his own words.
5. Grade honestly: say exactly where he was vague or reciting a label. Pass only if he could
   defend it to a skeptical interviewer. If not passed, explain the gap and re-ask.
6. Log questions, a one-line summary of his answers, and pass/fail under "Teach-backs" in `PROGRESS.md`.

HARD RULE: do not start any new work while a teach-back is pending or failed. The next step is
always the teach-back.

Everything else (loaders, config, CLI, runners, plotting, CI, demo, API harness): write it, keep the
diff small, and point out anything non-obvious in it.

## How to work
- Plan before code for anything non-trivial: inputs/outputs, files touched, tests, failure modes. Wait for "go".
- Small diffs, one concern each. He must be able to read every diff.
- Never claim something works until it has run. If no test covers a change, write the test first.
- When you use a non-obvious NumPy / pandas / scikit-learn / PyTorch idiom, name it and say in one line what it does.
- Comment the why, never the what.
- No new dependency without saying what it replaces and why. Add dependencies with `uv add`, never pip.
- After anything non-trivial outside the core pieces, offer (not force) three interview-style questions.
- Be direct. If his approach is wrong, say so first: "I disagree because X. Instead: Y. Risk of yours: Z."
- Do not guess library APIs or versions; check installed code or docs, and say when unverified.

## Environment
- Windows 10, native (Git Bash available). GPU: GTX 1650 4 GB — not needed before SP7.
- Python 3.11 via uv. Run everything as `uv run <cmd>` (e.g. `uv run pytest`, `uv run python -m watchdog_agent.cli ...`).
- GPU jobs (SP7) run on Colab/Kaggle by cloning this repo; notebooks contain no logic, only calls into the package.
- Dataset: `sunnydubey1111/agent-trajectory-sentinel` on Hugging Face (repo_type="dataset").
  Download with `huggingface_hub` into the default HF cache; never copy data into the repo, never commit data.
  Download only the files a script needs (e.g. `data/episodes.parquet`), not the ~5k raw trace files.
- Secrets (e.g. `HF_TOKEN`) live in `.env`, which is gitignored.

## Naming
- Repo name: `watchdog` (placeholder). Python import package: `watchdog_agent`.
  Not `watchdog`: that name is taken by a PyPI package that Streamlit (SP9) installs on Windows/Linux,
  and the two would clash on import.

## Layout (grows by save point)
```
src/watchdog_agent/  package code
tests/               pytest; one test file per module
scripts/             one-off scripts (e.g. e0_data_audit.py)
configs/             YAML experiment configs
results/             small JSON/CSV result files (committed); MLflow store in mlruns/ (gitignored)
docs/                data_card.md, decisions.md, per-SP notes
```

## Git
- Commit only through `/save` or when asked. Never force-push. Never commit secrets (.env is gitignored).
- Tags: `spNN-<short-name>` when a save point completes.
