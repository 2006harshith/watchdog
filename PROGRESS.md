# PROGRESS

## Current
- Save point: SP2 — data layer: per-step table + splits (teach-back: splits)
- Last session: 2026-09-25 — SP1 closed (tag sp01-audit): docs/data_card.md written from
  results/e0_audit_v2.json (families/labels, task-identity caveats, the ollama7b/ollama_llama8b
  matched pair for SP3, shortcut risks D4/D5/D6).
- Next concrete step: plan SP2 (per-step table + task-disjoint, model-family-held-out splits) before
  writing any code; wait for "go".

## SP0 prompt (paste into Claude Code)
> Set up SP0 for this repo. Plan first, wait for my "go". Target state: `pyproject.toml` managed by uv
> (Python 3.11, package `watchdog_agent` under `src/`), dev dependencies pytest and ruff only, one trivial
> test in `tests/test_smoke.py`, `.gitignore` (Python, .env, mlruns/, data caches), and a GitHub
> Actions workflow that runs `uv run pytest` and `uv run ruff check` on every push. Move
> `scripts/e0_data_audit.py` in as-is. Done when `uv run pytest` passes locally and CI is green.

## Save points
| SP | What | Status | Tag |
|---|---|---|---|
| SP0 | Setup: uv env, CLAUDE.md, CI with one test | done | sp00-setup |
| SP1 | E0 data audit → docs/data_card.md | done | sp01-audit |
| SP2 | Data layer: per-step table + splits (teach-back: splits) | not started | |
| SP3 | E1 reproduce collapse — GATE (teach-back: metrics). Pass: on the same held-out llama3.1:8b episodes, the qwen-fitted monitor is >= 0.15 AUROC below the llama-fitted one, with non-overlapping episode-bootstrap 95% CIs. Reference: 0.527 vs 0.885, arXiv 2608.02464 §5 (transferred vs refitted, not before/after) | not started | |
| SP4 | Feature sets (teach-back: feature definitions) | not started | |
| SP5 | E2 ablation — GATE | not started | |
| SP6 | E3 recalibration curve (teach-back: threshold logic) | not started | |
| SP7 | E4 cost/latency vs LLM judges — GATE | not started | |
| SP8 | E5 organic runs on AgentDojo — GATE | not started | |
| SP9 | Supervisor agent + Streamlit demo on HF Spaces | not started | |
| SP10 | Write-up: arXiv report, README, blog | not started | |
| SP11 | Phase 2: budgeted human review | not started | |

## Open questions
- (resolved in docs/data_card.md, SP1) Gemini licence clause → D6: eval-only, never training.
- (resolved in docs/data_card.md, SP1) Organic labels live only in traces/organic7b/organic_labels.csv;
  organic rows excluded from SP2–SP6 (D1).
- task_sha256 vs task-text-hash disagree on episode grouping (229 vs 311 groups on the same 1,989
  rows) — SP2 must pick one and say why before building splits on it.

## Decisions (after HANDOFF.md)
- 2026-09-25: Python import package is `watchdog_agent` (see CLAUDE.md "Naming").

## Teach-backs
<!-- core piece · SP · date · 3 questions · one-line summary of answers · pass/fail/pending -->
(none yet)

## Session log
<!-- /save appends here, newest first: date · SP · what changed · tests · next step -->
- 2026-09-25 · SP1 (close) · Wrote docs/data_card.md from results/e0_audit_v2.json: licence terms
  per model family (Gemini D6 eval-only rule), size/schema, families & labels table, task-identity
  caveats (task_sha256 vs derived-task-id disagreement), the ollama7b/ollama_llama8b matched pair for
  SP3, and shortcut risks (D4 API-class split, D5 all-healthy corpora as recal pools, D6 above) ·
  tests: pass (pytest + ruff) · next: plan SP2 (per-step table + task-disjoint, family-held-out
  splits), wait for "go". Tag sp01-audit.
- 2026-09-25 · SP1 · Added 3 checks to scripts/e0_data_audit.py: organic manifest/collection_meta
  label scan (no organic* corpus outside organic7b carries a per-episode label beyond the existing
  null failure_class), derived_task_id (sha256 of normalised steps[0].task) vs task_sha256 grouping
  agreement (disagree — 1,989 rows with both, groupings differ), and ollama7b vs ollama_llama8b tool
  roster + failure_class side by side (same single tool roster, llama8b has proportionally more
  failures in every class). Results appended to results/e0_audit_v2.json · tests: pass (pytest +
  ruff) · next: paste results/e0_audit_v2.json into the chat Project, draft docs/data_card.md,
  `/save SP1` to close.
- 2026-09-25 · SP1 (housekeeping) · Fixed PROGRESS.md status (SP0/SP1 had reverted to "not
  started" from a stale starter-kit paste; corrected to reflect the real repo state), rewrote
  SP3's gate wording, added the HANDOFF.md pointer to CLAUDE.md, restored scripts/e0_data_audit.py
  after it was overwritten back to the pre-SP1 Colab stub by the same stale paste · tests: pass ·
  next: paste results/e0_audit_v2.json into the Claude chat Project, draft docs/data_card.md, then
  `/save SP1` to close SP1.
