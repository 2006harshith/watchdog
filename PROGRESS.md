# PROGRESS

## Current
- Save point: SP1 — E0 data audit (in progress: script + results done, docs/data_card.md not written yet)
- Last session: 2026-09-25 — SP0 closed (tag sp00-setup); SP1 script run for real against the live
  HF dataset and results/e0_audit_v2.json committed and pushed.
- Next concrete step: paste results/e0_audit_v2.json into the Claude chat Project, draft
  docs/data_card.md there, bring it back to commit, then `/save SP1` to tag sp01-audit.

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
| SP1 | E0 data audit → docs/data_card.md | in progress | |
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
- Gemini episodes in agent-trajectory-sentinel carry a licence clause against building competing models — check before training on them (SP1).

- Colab E0 audit v1 (2026-09-25), to be re-run from the repo in SP1:
  - parquet 3,581 rows vs card 2,823; the extra 758 are all qwen2.5:7b.
  - failure_class None = 2,348 = rows with tau NaN. The card lists organic failures labelled post hoc
    (hallucinated, incomplete, arithmetic error); those may be inside None. Treating None as healthy
    could mislabel them. Find where organic labels live before SP2.
  - tau (0-indexed) is 2 for most failures (median 2, 75% 3, max 6): an injection artefact. Prefix
    label must be 1 only for step_idx >= tau.
  - has_logprobs False on 999 rows; latency_s and output_tokens per step: likely model identifiers.
  - rate_limit / timeout classes may be detectable from latency/error flags alone: report separately.

## Decisions (after HANDOFF.md)
- 2026-09-25: Python import package is `watchdog_agent` (see CLAUDE.md "Naming").

## Teach-backs
<!-- core piece · SP · date · 3 questions · one-line summary of answers · pass/fail/pending -->
(none yet)

## Session log
<!-- /save appends here, newest first: date · SP · what changed · tests · next step -->
- 2026-09-25 · SP1 (housekeeping) · Fixed PROGRESS.md status (SP0/SP1 had reverted to "not
  started" from a stale starter-kit paste; corrected to reflect the real repo state), rewrote
  SP3's gate wording, added the HANDOFF.md pointer to CLAUDE.md, restored scripts/e0_data_audit.py
  after it was overwritten back to the pre-SP1 Colab stub by the same stale paste · tests: pass ·
  next: paste results/e0_audit_v2.json into the Claude chat Project, draft docs/data_card.md, then
  `/save SP1` to close SP1.
