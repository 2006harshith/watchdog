# PROGRESS

## Current
- Save point: SP4 — next, but its spec (and SP5's) is being rewritten in the chat Project after SP3's
  reframe (docs/decisions.md). No SP4 code until that spec is pasted. Date: 2026-09-27.
- SP3 closed 2026-09-27 (tag sp03-e1): gate FAIL. Task-disjoint E1: AUROC A 0.644, B 0.634, gap
  -0.009 [-0.046, 0.025]; the paper's in-domain 0.885 is mostly exact-twin leakage (D-2: median
  0.896 random vs 0.683 twin-aware). Write-up: docs/sp3_e1_result.md. GitHub issue drafted, not
  posted: docs/drafts/github_issue_author.md.
- SP2 closed 2026-09-27 (tag sp02-data; code from 64622a4, tag on the closing commit).
- Teach-back answers are owed before deploy (SP9): see "Teach-backs" below.
- Next concrete step: in the chat Project, send "We're starting SP4 (rewritten after SP3). Read
  PROGRESS.md, docs/decisions.md and docs/sp3_e1_result.md. Write the SP4 spec paste-ready for Claude
  Code, including D7 (drop the 787 no-tool-call runs) and task-group cluster CIs as primary." Paste
  the spec into Claude Code; plan, wait for "go".

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
| SP2 | Data layer: per-step table + splits (teach-back: splits) | done (teach-back owner-marked passed) | sp02-data |
| SP3 | E1 reproduce collapse — GATE (teach-back: metrics). Pass: on the same held-out llama3.1:8b episodes, the qwen-fitted monitor is >= 0.15 AUROC below the llama-fitted one, with non-overlapping episode-bootstrap 95% CIs. Reference: 0.527 vs 0.885, arXiv 2608.02464 §5 (transferred vs refitted, not before/after) | done — gate FAIL (no collapse task-disjoint; docs/sp3_e1_result.md); teach-back owner-marked passed | sp03-e1 |
| SP4 | Feature sets (teach-back: feature definitions) (spec to be rewritten after SP3) | not started | |
| SP5 | E2 ablation — GATE (spec to be rewritten after SP3) | not started | |
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
- (resolved 2026-09-27: reframe, see docs/decisions.md) SP3: E1 gate FAIL — under task-disjoint folds B (llama-refit, 0.634) is no better than A
  (qwen-fit, 0.644), and B trips the < 0.70 sanity flag. Is the paper's "collapse" a task-overlap
  effect rather than a model-swap effect? D-2/D-3 diagnostics test this; then the chat Project
  decides what SP4+ should target (task shift vs model shift). Diagnostics say yes: most of B's
  0.885 comes from exact-twin healthy runs in test (twin-aware 0.678, task-disjoint 0.623).
- SP3/D-1: 10 context_corruption "failures" in llama have a tool-call sequence identical to a
  healthy run — is the injection visible to any model-independent tool-level signal? (SP4 input)
- SP3: the paper's transfer AUROC 0.527 does not reproduce from released code+data (0.6805). Issue
  drafted in docs/drafts/github_issue_author.md; owner to post it and record the author's answer.
- SP4 spec: the rewritten SP4/SP5 specs and the SP5 gate are decided in the chat Project.

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
- 2026-09-27: Owner marked all four open teach-backs (SP2 splits, SP3 metrics, SP3 E1 split logic,
  SP3 diagnostic split logic) PASSED without answering them, for time. The questions stay logged
  below; answering all of them is a pre-deploy requirement (before SP9). They no longer block tags.
- 2026-09-27 (SP3 close-out, full record in docs/decisions.md):
  - SP3 gate FAIL under rule (a) and rule (b): AUROC A 0.644 [0.554, 0.730], B 0.634 [0.547, 0.717],
    gap -0.009, paired CI [-0.046, 0.025] (results/e1_collapse.json: monitors.*.auroc, gate.*).
  - Direction: reframe to unseen tasks + families. Task-disjoint, the llama- and qwen-fitted monitors
    are equal, so the problem is unseen tasks/families, not the model swap as such. Recalibration
    after a swap is about thresholds, not ranking. SP4/SP5 specs to be rewritten in the Project;
    SP5 gate decided there.
  - D7: drop the 787 runs with no tool calls (7 corpora; = SP2's task-orphan rows). Implemented in
    the SP4 data change.
  - SP5 baseline to beat: the author's ESN on the same task-disjoint splits.
  - From SP4 on: task-group cluster bootstrap is the primary CI; every headline reports n_healthy.

## Teach-backs
<!-- core piece · SP · date · 3 questions · one-line summary of answers · pass/fail/pending -->
- 2026-09-25 · SP2 · splits (src/watchdog_agent/splits.py + task_group in data.py) · Q1: why drop
  overlapping train rows by task_group (not task_sha256 alone) in leave_one_family_out — what does
  task_group catch that task_sha256 alone misses? Q2: why fold-by-group (not fold-by-row) in
  grouped_kfold, and what would break in an SP3 concatenated-out-of-fold AUROC if it were fold-by-row?
  Q3: walk through what breaks in leave_one_family_out for real_research7b (291/291 orphan rows) if
  assign_task_groups used the literal "unknown:&lt;corpus&gt;" one-bucket-per-corpus fallback instead
  of singleton-per-row, and that corpus needed to be inside a training set · answers: not yet given ·
  PASSED (owner-marked 2026-09-27, not answered; owner will answer all before deploy).
- 2026-09-25 · SP3 · metrics (src/watchdog_agent/metrics.py) · Q1: AUROC is computed from ranks;
  why does a tied (failed, healthy) pair count 1/2, and what AUROC does a monitor that gives every
  run the same score get? Q2: in E1 the individual AUROC CIs of A [0.554, 0.730] and B [0.547, 0.717]
  overlap almost completely, yet the paired CI of B-A is only [-0.046, 0.025]; why is the paired
  interval so much narrower, and why is it the right one for the gate? Q3: why resample runs or task
  groups rather than steps, and why is the task-group cluster CI of the gap ([-0.19, 0.07]) wider
  than the run-level one? · answers: not yet given · PASSED (owner-marked 2026-09-27, not answered;
  owner will answer all before deploy).
- 2026-09-25 · SP3 · split logic in E1 (src/watchdog_agent/experiments/e1.py: plan_grouped_folds,
  plan_leave_one_group_out, split_by_task_group) · Q1: A_f excludes qwen runs whose task_group is
  in fold f; what would A's score measure if it did not? Q2: why are the alarm threshold and the
  pooled-score standardisation set on a task-group-disjoint validation slice, not on the fit runs?
  Q3: fold 1 holds 108/193 runs because one task has 83 runs; what does that do to pooled vs mean
  per-fold AUROC, and why is pooled-after-standardisation the primary? · answers: not yet given ·
  PASSED (owner-marked 2026-09-27, not answered; owner will answer all before deploy).
- 2026-09-27 · SP3 · diagnostic split logic (src/watchdog_agent/experiments/e1_diagnostics.py:
  trajectory_hash, twin_aware_split, task_disjoint_split, source_pools) · Q1: twins are defined on
  tool calls (name, args, result) only; name one way two "twins" could still differ to the ESN and one
  way two non-twins could look identical to it. Q2: in the twin-aware split, why are trimmed twins
  dropped rather than sent to test? Q3: task-disjoint D-2 skipped 10 of 30 seeds; how could that skip
  rule bias the reported 0.623? · answers: not yet given · PASSED (owner-marked 2026-09-27, not
  answered; owner will answer all before deploy).

## Session log
<!-- /save appends here, newest first: date · SP · what changed · tests · next step -->
- 2026-09-27 · SP2 (close) · No code change; SP2 marked done (splits teach-back owner-marked
  passed, answers owed before deploy). Tag sp02-data · tests: pass (216, ruff clean) · next: get the
  rewritten SP4 spec from the chat Project.
- 2026-09-27 · SP3 (close) · Close-out: teach-backs owner-marked passed (questions logged, answers
  owed before deploy); twin_audit gained runs_in_mixing_clusters (+ test); diagnostics JSON
  regenerated from clean commit cf6d42c (pre-existing keys identical); docs/decisions.md (gate FAIL,
  reframe, D7, SP5 baseline, cluster CIs primary); docs/sp3_e1_result.md (negative result, unpaired
  D-2 comparison: median 0.896 vs 0.683, 12/20 below min random, P = 0.875);
  docs/drafts/github_issue_author.md (not posted). Tag sp03-e1 · tests: pass (216, ruff clean) ·
  next: get the rewritten SP4 spec from the chat Project.
- 2026-09-27 · SP3 · Diagnostics D-1..D-4 (results/e1_diagnostics.json): exact tool-call twins
  common, 10/16 healthy test runs in the author's in-domain split are twins of fit runs; B AUROC
  random 0.884 / twin-aware 0.678 / task-disjoint 0.623; A gains nothing from task overlap
  (0.558 vs 0.591); E1 identical from clean commit 1a3787a (results/e1_collapse.json now carries
  the clean git block). Moved load_corpus/author_split into experiments/author_protocol.py ·
  tests: pass (215, ruff clean) · next: paste the three SP3 result JSONs into the chat Project for
  review and the SP4+ direction decision.
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
