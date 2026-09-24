# PROGRESS — watchdog

The one file that says where the project is. `/resume` reads it; `/save` updates it.
Re-upload it to the Claude Project "WatchDog" at every save point.

## Current position
- Save point: **SP1 — not started**
- Next concrete step: get the SP1 prompt confirmed / go ahead with the SP1 prompt below (E0 data audit port).
- Pending teach-back: none

## Save points

| SP | What | Core piece needing teach-back | Gate | Status | Tag |
| --- | --- | --- | --- | --- | --- |
| SP0 | Setup: uv env, package skeleton, pytest + ruff, CI green | — | — | done | sp00-setup |
| SP1 | E0 data audit script in repo + docs/data_card.md | — | — | not started (a one-off Colab audit exists; not in repo) | |
| SP2 | Data layer: one row per step (prefix), split module | splits | — | | |
| SP3 | E1: reproduce the cross-family collapse; ESN monitor + eval harness; MLflow | metrics | **yes** | | |
| SP4 | Feature sets: model-independent vs model-specific | features | — | | |
| SP5 | E2 ablation: XGBoost (+GRU) per feature set, held-out families | — | **yes** | | |
| SP6 | E3 recalibration curve: detection @5% FAR vs 0–50 healthy runs | calibration / threshold | — | | |
| SP7 | E4 cost/latency vs LLM judges (local, Groq, LoRA on Colab) | — | **yes** | | |
| SP8 | E5 organic runs on AgentDojo, 3 families | — | **yes** | | |
| SP9 | Supervisor (continue/retry/halt/escalate) + Streamlit demo on HF Spaces | — | — | | |
| SP10 | Write-up: arXiv report, README, blog | — | — | | |
| SP11 | Phase 2 — E6 budgeted human review | — | — | only after SP8 | |

A save point is closed when: tests pass, work is committed, git tag `spNN-<name>` exists, this file is updated.

---

## SP0 prompt (paste into Claude Code)

```text
SP0 — setup. Read CLAUDE.md and PROGRESS.md first.
Goal: an empty but working Python project with tests and CI.
1. `uv init` as a package with a src layout; Python import package name `watchdog_agent`
   (src/watchdog_agent/__init__.py with __version__ = "0.0.1"). Pin Python 3.11 (`uv python pin 3.11`).
2. Dev dependencies via `uv add --dev pytest ruff`. Ruff config in pyproject.toml (line-length 100).
3. tests/test_smoke.py: imports watchdog_agent and asserts __version__ is a string.
4. .github/workflows/ci.yml: on push and pull_request, install uv with the official astral-sh setup-uv
   action (look up its current major version in its README; do not guess), then `uv sync --locked`,
   `uv run ruff check .`, `uv run pytest`.
5. Keep the existing .gitignore; add anything uv needs.
Done when: `uv run pytest` and `uv run ruff check .` pass locally, the commit is pushed,
and the GitHub Actions run is green.
Plan first (files you will create, commands you will run), then wait for my "go".
```

## SP1 prompt (after SP0 is tagged)

```text
SP1 — E0 data audit. Below is the audit code I ran once in Colab. Port it into
scripts/e0_data_audit.py, runnable as `uv run python scripts/e0_data_audit.py`.
- Get data with huggingface_hub.hf_hub_download(repo_id="sunnydubey1111/agent-trajectory-sentinel",
  repo_type="dataset", filename=...). HF cache only; nothing in the repo. Download only
  data/episodes.parquet plus the few label/manifest files the checks need, NOT all ~5k trace files.
- Read HF_TOKEN from .env if present (python-dotenv, or plain env var — say which and why).
- Deps: `uv add pandas pyarrow huggingface_hub`.
Then extend it with these checks and write results/e0_audit_v2.json:
1. corpus value counts; crosstab corpus x model x failure_class (including None).
2. Organic labels: for organic* corpora list all metadata keys and value counts of metadata.success,
   accepted_because and any label-like key; load traces/organic7b/organic_labels.csv, report columns
   and counts. How many failure_class==None rows are NOT verified healthy?
3. Duplicates/replays: rows per metadata.provenance.task_sha256; task_sha256 appearing under >1 model;
   episode_id in >1 corpus; duplicate trace_sha256.
4. Qwen: 3005 rows here vs 2,247 on the dataset card. Which corpora explain the gap?
5. tau: tau/T distribution per failure_class; share with tau==2; T distribution healthy vs failed per corpus.
6. has_logprobs x model crosstab. Per failure_class: mean latency_s, output_tokens, share of steps with error==True.
7. Injection provenance: share of failures with non-empty metadata.injection; requested_class vs
   failure_class agreement; size of rejected.json / landing_failures.json per corpus.
Inspect real key names; don't guess them. Plan first, wait for "go".
<paste the Colab audit code here>
```

After the script runs: paste results/e0_audit_v2.json into the Claude Project; the data card
(docs/data_card.md) is drafted there, then committed here.

---

## Open questions (from the one-off Colab audit, 2026-09-25)
- `failure_class == None` (2348 rows) may hide organic failures that are labelled post-hoc elsewhere.
- Failure onset tau is almost always 2 (median 2, max 6): an injection artefact; prefix labels must be
  "failed by step k" (k >= tau).
- `organic_demo7b_cold_retry` has ~16 replay variants of the same episodes → leakage risk; split by task.
- Model imbalance: Qwen 3005 / Llama 433 / Gemini 143 rows; only 3 real families.
- `has_logprobs`, `latency_s`, `output_tokens` likely identify the model → not model-independent.
- Source of the SP3 target "0.885 → ~0.53" is not on the dataset card. Owner to supply the citation.
- Licence: Gemini outputs (Google ToS: no developing competing models) → use Gemini for evaluation only.

## Decisions
- 2026-09-25: Python import package named `watchdog_agent`, not `watchdog` (PyPI `watchdog` is a
  Streamlit dependency on Windows/Linux; verified in streamlit 1.64.0 metadata).

## Teach-backs

| Date | SP | Core piece | Questions (short) | Answer summary | Result |
| --- | --- | --- | --- | --- | --- |

## Session log
<!-- /save appends: date — what got done — next concrete step — open questions -->
- 2026-09-25: SP0 closed. `uv init --package`, pinned Python 3.11, dev deps pytest+ruff (ruff
  line-length 100), tests/test_smoke.py, CI workflow (astral-sh/setup-uv pinned to v10.1.0 SHA).
  Local tests + ruff pass, pushed, GitHub Actions run green (run 36048770688). Tagged sp00-setup.
  Next: SP1 — E0 data audit script.
