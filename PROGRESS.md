# PROGRESS

## Current
- Save point: SP3 — E1 reproduce the collapse (in progress). SP2 stays open (code done, untagged):
  splits teach-back DEFERRED by owner override on 2026-09-25; it must pass before `/save SP2` tags it.
- Last session: 2026-09-25 — SP3 Step 0 report done (author code: Apache-2.0, esn_cusum_max is the
  0.885/0.527 source, author protocol not task-disjoint, 16 healthy test runs; logprobs 100%; llama
  fold 1 = 108 runs because one task has 83 runs). Owner said "go" on all Step 0 recommendations.
  Step 1 done: src/watchdog_agent/metrics.py + tests/test_metrics.py; `uv add numpy`,
  `uv add --dev scikit-learn` (1.9.1).
- Step 2 done (metrics teach-back DEFERRED by owner override): author's ESN vendored verbatim into
  src/watchdog_agent/baselines/esn/_vendor (Apache-2.0, 1b3e07f; only import paths + headers
  changed; ruff-excluded), wrapper ESNBaseline in baselines/esn/monitor.py, 7 tests.
  scripts/reproduce_author_transfer.py -> results/e1_author_repro.json: in-domain reproduces
  EXACTLY (AUROC 0.8847, AUPRC 0.9552, det 0.7672, FA 0.125); transfer does NOT (AUROC 0.6805 vs
  0.5275). Not the channel set; parquet == released trace files (sha256 match). Released
  code+data do not reproduce the paper's 0.527.
- Step 3 done: src/watchdog_agent/experiments/e1.py, scripts/run_e1.py, configs/e1.yaml, 8 tests;
  `uv add mlflow pyyaml` (mlflow 3.16.1 refuses the ./mlruns file store -> sqlite at
  mlruns/mlflow.db, artifacts in mlruns/artifacts). Run is deterministic (2 identical runs).
  RESULT (results/e1_collapse.json): GATE FAIL, no collapse under the task-disjoint protocol.
  AUROC A (qwen-fit) 0.644 [0.554, 0.730], B (llama-refit) 0.634 [0.547, 0.717]; gap -0.009,
  paired CI [-0.046, 0.025]. Sanity flag: B < 0.70. Sensitivity agrees (cluster bootstrap gap
  -0.009; fold seeds 1-3 gaps -0.08/-0.13/-0.02; leave-one-task-group-out 0.647 vs 0.655).
  TPR at 5% FPR ~3-4% for both; FA at default threshold 0.36 for both.
- 2026-09-25 (saved): owner approved SP3 diagnostics D-1..D-4 (twin audit, split ablation for B and
  A, clean-commit rerun of E1 -> results/e1_diagnostics.json); plan in chat, decisions 1-4 accepted.
- Next concrete step: implement SP3 diagnostics (experiments/author_protocol.py,
  experiments/e1_diagnostics.py, scripts/run_e1_diagnostics.py, tests), run D-4 from this commit in
  a temporary git worktree, then paste results/e1_collapse.json, e1_author_repro.json and
  e1_diagnostics.json into the chat Project to decide what the SP3 gate FAIL means for SP4+.

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
| SP2 | Data layer: per-step table + splits (teach-back: splits) | code done; teach-back deferred | |
| SP3 | E1 reproduce collapse — GATE (teach-back: metrics). Pass: on the same held-out llama3.1:8b episodes, the qwen-fitted monitor is >= 0.15 AUROC below the llama-fitted one, with non-overlapping episode-bootstrap 95% CIs. Reference: 0.527 vs 0.885, arXiv 2608.02464 §5 (transferred vs refitted, not before/after) | E1 run: gate FAIL (no collapse, task-disjoint); diagnostics next; teach-back deferred | |
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
- (resolved in SP2, D3) task_sha256 vs task-text-hash disagreement: task_group is now the union-find
  connected component over rows sharing either key (OR, not a pick-one), so both signals contribute.
- SP2 finding: leave_one_family_out(test_family="qwen") leaves only 34 non-overlapping train runs
  from llama+gemini (302/336 dropped for task_group overlap with qwen). SP3 needs a design call on
  whether/how a qwen-held-out arm is even meaningful with that little train data, or whether SP3 only
  ever holds out llama/gemini (matches the paper's llama-held-out setting anyway).
- SP3: E1 gate FAIL — under task-disjoint folds B (llama-refit, 0.634) is no better than A
  (qwen-fit, 0.644), and B trips the < 0.70 sanity flag. Is the paper's "collapse" a task-overlap
  effect rather than a model-swap effect? D-2/D-3 diagnostics test this; then the chat Project
  decides what SP4+ should target (task shift vs model shift).
- SP3: the paper's transfer AUROC 0.527 does not reproduce from released code+data (0.6805). Ask the
  author which ollama7b data produced it? (owner's call; public GitHub issue)

## Decisions (after HANDOFF.md)
- 2026-09-25: Python import package is `watchdog_agent` (see CLAUDE.md "Naming").
- 2026-09-25: Owner override of the CLAUDE.md teach-back HARD RULE for SP2: splits teach-back
  deferred so SP3 can start. SP2 is not tagged until it passes. Risk accepted: SP3 builds on splits
  the owner cannot yet defend.
- 2026-09-25: SP3 baseline accepted as faithful on the exact in-domain reproduction (0.8847, all 4
  metrics). The paper's transfer AUROC 0.527 does not reproduce from released code+data @1b3e07f
  (0.6805); E1 reports both as references. Option: owner opens a GitHub issue with the author.
- 2026-09-25: SP3 Step 0 recommendations accepted: vendor author code; episode score = max of the
  channel-max CUSUM stream; author defaults for unspecified ESN settings; per-fold threshold/
  standardisation from a 25% task-group-disjoint healthy validation slice of each fit pool (not
  the fit runs, which are in-sample for the ESN); k=5 grouped folds primary + leave-one-task-
  group-out sensitivity.

## Teach-backs
<!-- core piece · SP · date · 3 questions · one-line summary of answers · pass/fail/pending -->
- 2026-09-25 · SP2 · splits (src/watchdog_agent/splits.py + task_group in data.py) · Q1: why drop
  overlapping train rows by task_group (not task_sha256 alone) in leave_one_family_out — what does
  task_group catch that task_sha256 alone misses? Q2: why fold-by-group (not fold-by-row) in
  grouped_kfold, and what would break in an SP3 concatenated-out-of-fold AUROC if it were fold-by-row?
  Q3: walk through what breaks in leave_one_family_out for real_research7b (291/291 orphan rows) if
  assign_task_groups used the literal "unknown:&lt;corpus&gt;" one-bucket-per-corpus fallback instead
  of singleton-per-row, and that corpus needed to be inside a training set · answers: not yet given ·
  DEFERRED (owner override 2026-09-25) — must pass before SP2 is tagged.
- 2026-09-25 · SP3 · metrics (src/watchdog_agent/metrics.py) · code + 164 tests pass (sklearn
  parity incl. ties, hand cases, unit-whole bootstrap, paired diff) · questions not yet asked ·
  DEFERRED (owner override 2026-09-25) — required before `/save SP3`.
- 2026-09-25 · SP3 · split logic in E1 (src/watchdog_agent/experiments/e1.py: plan_grouped_folds,
  plan_leave_one_group_out, split_by_task_group — task-disjoint A/B fit pools and the inner
  task-group validation split) · questions not yet asked · PENDING — required before `/save SP3`.

## Session log
<!-- /save appends here, newest first: date · SP · what changed · tests · next step -->
- 2026-09-25 · SP3 · Step 0 report (author code Apache-2.0; esn_cusum_max is the 0.885/0.527
  source; author protocol not task-disjoint). metrics.py + 164 tests (core, teach-back deferred).
  Vendored author ESN (baselines/esn/_vendor @1b3e07f) + ESNBaseline wrapper; author-protocol
  reproduction: in-domain exact (0.8847), transfer 0.6805 vs paper 0.527. E1 (experiments/e1.py,
  scripts/run_e1.py, configs/e1.yaml, MLflow sqlite in mlruns/): GATE FAIL, A 0.644 vs B 0.634,
  gap -0.009 [-0.046, 0.025]. Deps: numpy, pyyaml, mlflow; dev scikit-learn. Owner deferred SP2
  splits + SP3 metrics teach-backs · tests: pass (190, ruff clean) · next: SP3 diagnostics D-1..D-4.
- 2026-09-25 · SP2 · Added src/watchdog_agent/data.py (load_episodes: drops organic* per D1, adds
  family via model_to_family, adds task_group/task_known via assign_task_groups's union-find over
  task_sha256 OR normalised-task-text-hash, orphan rows get singleton groups per owner's call;
  to_step_table for the per-step frame) and src/watchdog_agent/splits.py (leave_one_family_out,
  grouped_kfold, healthy_subset, all seeded). scripts/sp2_summary.py →
  results/sp2_split_summary.json. tests/test_data.py + tests/test_splits.py (11 tests, incl. a
  real-data smoke test and the union-find transitivity case) · tests: pass (pytest + ruff) · next:
  splits teach-back is PENDING (3 questions asked, unanswered) — grade before anything else, then
  `/save SP2` to close.
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
