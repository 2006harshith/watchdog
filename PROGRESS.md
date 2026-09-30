# PROGRESS

## Current
- Save point: SP5 (E2 ablation, GATE) — in progress. Step 0 (pre-registration, ae18348), Step 1
  (recalibration.py, 4733739), Step 2 (experiments/e2.py, 860249b) done. Step 3 run 2026-09-30 from
  clean 860249b (results/e2_ablation.json, MLflow d3a8305364a64be29c2425b4a4449871): GATE FAIL.
  llama TL-pct − ESN_best (ESN_i) = −0.197 [−0.302, −0.057]; TL-pct 0.648 [0.555, 0.768], ESN_i
  0.846 [0.770, 0.896]. Both secondary expectations hold (TL-pct 0.648 > TL-raw 0.641; ≥ MS-pct
  0.552). Not gated: gemini TL-pct 0.716 vs ESN_i 0.481, ALL-pct 0.780.
- SP5 Step 3b diagnostics 2026-09-30 (results/e2_diagnostics.json, dirty tree; three rows reproduce
  0.846 / 0.858 / 0.646 exactly). The ESN 0.646 (SP4) -> 0.846 (E2) gap is the FIT POOL, not the
  class mix (SP4 headline-only 0.665) and not pooling or z-scoring (SP4 raw 0.661; per-fold mean
  0.602). On SP4's own folds, per-fold: E2's single model 0.838 vs SP4's fold models 0.602. Pool size
  (21 vs ~34 fit runs) and composition (E2 excludes every llama task group) are not separated.
  Surprisal is NOT the source: ESN_i without u scores 0.905 (context_corruption 1.000); MS-raw
  without surprisal 0.812, surprisal only 0.547. The ESN reads step-output content (e: text
  embedding; x: cos drift, task similarity) plus latency and output length (m), which TL excludes by
  design; context_corruption scrambles result text, so content catches it.
- SP4 closed 2026-09-29 (tag sp04-features): D7/D6 data changes, matched-step evaluation, 118
  features (TL/MS/ENV/TEXT; TL was MI until the close-out), feature audit (results/sp4_feature_audit.json, clean commit e9ef64f).
  Headline: TL (ex-MI) is NOT model-independent in practice (healthy-run family AUROC 0.95-0.96 in the matched
  booking pair via llama's invented calculator args; 0.92-0.96 in the research pair via gemini's live-API
  errors); ESN at matched steps A 0.646 [0.518, 0.736], B 0.694 [0.531, 0.825] (mean t=3,4) is the
  number SP5 must beat. SP4 split and metrics teach-backs owner-marked; feature-definitions
  teach-back NOT ANSWERED (overridden twice). Close-out 2026-09-29 (tag sp04-closeout): MI renamed
  TL, audit JSON regenerated from clean fffba52 (values identical); "Proposed for SP5" from the
  Project's specs/SP4-review.md NOT yet recorded (close-out item 4).
- SP3 closed 2026-09-27 (tag sp03-e1 at 3e061bb): gate FAIL. Task-disjoint E1: AUROC A 0.644, B
  0.634, gap -0.009 [-0.046, 0.025]; the paper's in-domain 0.885 is mostly exact-twin leakage (D-2:
  median 0.896 random vs 0.683 twin-aware). D-5 added after the tag: 30 paired ESN seeds, CLAIM HOLDS
  (results/e1_seed_sweep.json, clean commit 43cb2c2). Write-up: docs/sp3_e1_result.md. GitHub issue
  drafted, not posted: docs/drafts/github_issue_author.md.
- SP2 closed 2026-09-27 (tag sp02-data; code from 64622a4, tag on the closing commit).
- Teach-back answers are owed before deploy (SP9): see "Teach-backs" below.
- Next concrete step: /save, rerun scripts/run_e2_diagnostics.py from the clean commit, then paste
  results/e2_ablation.json + results/e2_diagnostics.json into the chat Project for the negative-result
  write-up spec (gate on_fail: SP6 becomes the main contribution).

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
| SP3 | E1 reproduce collapse — GATE (teach-back: metrics). Pass: on the same held-out llama3.1:8b episodes, the qwen-fitted monitor is >= 0.15 AUROC below the llama-fitted one, with non-overlapping episode-bootstrap 95% CIs. Reference: 0.527 vs 0.885, arXiv 2608.02464 §5 (transferred vs refitted, not before/after) | done — gate FAIL; diagnostics D-1..D-5 done (docs/sp3_e1_result.md); teach-back owner-marked passed | sp03-e1 |
| SP4 | Feature sets + matched-step evaluation (teach-back: feature definitions) | done — audit in results/sp4_feature_audit.json; split/metrics teach-backs owner-marked, feature-definitions teach-back not answered (owner override) | sp04-features, sp04-closeout |
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
- SP3/D-5: pooled AUROC shows no B advantage, but mean per-fold AUROC favours B in 30/30 seeds (median
  +0.074). Is that a real, smaller model-swap effect, or fold weighting (the 108-run fold counts the
  same as the 16-run one)? Relevant to how SP5 aggregates across folds.
- (resolved in SP4) SP3/D-5: the episode-max score tracks run length (Spearman 0.85/0.80). SP4 built
  matched-step evaluation; prefix length scores exactly 0.5 there, and the ESN keeps ~0.65 (A).
- (resolved in SP4, 4c) SP3/D-1 twins: at matched steps 10-20% of llama positives (all
  context_corruption) and 2-3% of qwen positives have a tool-level prefix identical to a healthy run
  of the same task; no tool-level monitor can separate them.
- SP4/4a: TL (ex-MI) features predict family on healthy runs (booking 0.95-0.96: llama calls the calculator
  with invented args {a, b, c, ..., op} and errs 38% of the time; research 0.92-0.96: gemini hit live
  APIs, 42% errors, median result 49 chars vs ~280). Does SP5 normalise features on the target
  family's healthy runs, and does the "model-independent" claim get rewritten?
- (resolved 2026-09-29, SP5 Decisions) tool_cascade, rate_limit, timeout inject is_error directly (4b):
  they form the error-visible group, reported separately; headline = looping, goal_drift,
  context_corruption (revises D4).
- (resolved 2026-09-29, SP5 Decisions) held-out qwen dropped for good (21 train runs after D6).
- (resolved 2026-09-29, SP5 Decisions "D6 reading") percentiles or thresholds from gemini healthy runs,
  used only to score gemini at evaluation time, are evaluation-time calibration, not training.
- SP4: results/sp4_feature_audit.json not yet pasted into the chat Project (a SP4 done criterion).
- (resolved 2026-09-29) SP4 close-out item 4: the SP5 spec's Decisions and GATE sections are recorded
  verbatim in docs/decisions.md and PROGRESS.md (SP5 Step 0).
- (resolved in SP4 close-out) Looping: loops are exact (name, args) repeats, one step after tau, not
  same-tool-new-args; exact repeat by t=3 0.11, t=4 0.32. A name-only same_tool_as_prev scores AUROC
  0.54-0.58 on qwen looping at t=3,4 (healthy runs repeat the tool 70% of the time), so not added.
- (resolved in SP4 close-out) context_corruption: the injector edits result content in place
  (appends a spurious value, shuffles words, scrambles text) and keeps corrupting later results
  (applied_count 1-13); it does not shrink results (median 1.45x the healthy size). Corrupting an
  error message flips is_error True -> False. Data property, no feature bug.
- SP5/3b: the ESN baseline swings 0.60 -> 0.84 (per-fold, same folds) with its healthy fit pool.
  Is that pool size (21 vs ~34 runs) or composition (E2's pool excludes every llama task group)? A
  fragile baseline is itself evidence for SP6 (few healthy runs); separating the two needs a
  size-matched subsample of SP4's pools.
- SP5/3b: the ESN wins on llama through step-output content (e, x) and latency/length (m), which TL
  excludes by design; ESN_i without u reaches context_corruption 1.000. Is context_corruption then a
  content-injection artefact (scrambled text is trivially off-distribution) rather than a realistic
  failure? Does the write-up add a content-aware arm, or state the TL design limit?
- SP5/E2: MS-pct is below chance on llama context_corruption (0.361) while MS-raw scores 0.948. The
  percentile transform may flip the MS signal's direction; not investigated.

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
- 2026-09-29: Owner override of the teach-back HARD RULE for the SP4 split-logic teach-back
  (D7 + D6): deferred so Step 2 can start. Answers owed before deploy. The feature-definition
  teach-back must still be answered, not owner-marked, before /save SP4 (SP4 spec).
- 2026-09-29: Same override for the SP4 matched-step metrics teach-back (evaluation.py), so Step 3
  can start. Six teach-backs now owed before deploy.
- 2026-09-29: Owner override of the SP4 spec's own done criterion ("feature-definition teach-back
  answered, not owner-marked") to close SP4. All three SP4 teach-backs marked PASSED (owner-marked,
  not answered). Claude disagreed: the feature definitions are the core ML to defend, and 4a showed
  MI is family-predictable in practice. Risk accepted: SP5 builds on a feature story the owner has
  not yet defended. Seven teach-backs owed before deploy (SP9).
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
- 2026-09-27: D-5 (30 paired ESN seeds, pre-registered rule): CLAIM HOLDS. Task-cluster upper bound of
  B-A < 0.15 in 30/30 seeds (max 0.126); pooled B-A median -0.005. Mean per-fold AUROC favours B in
  30/30 (median +0.074, max +0.114): a smaller model-swap effect may remain (results/e1_seed_sweep.json).
- Proposed for the SP4 spec, NOT decided: a length-only baseline in SP5 (length alone AUROC 0.592;
  ESN score vs length Spearman 0.85/0.80), and matched-step prefix evaluation instead of episode-max.

- 2026-09-29: SP5 Decisions and GATE, verbatim from the SP5 spec (same text in docs/decisions.md;
  encoded in configs/e2.yaml):

```text
Decisions (defaults chosen; override before pasting)
TL rename kept. Groups: TL (tool level), MS (model level: latency, output tokens, text length, surprisal), ENV, TEXT (the char-3-gram hash).
Headline classes: looping, goal_drift, context_corruption.
Error-visible group, reported separately: tool_cascade, rate_limit, timeout.
Also reported separately: malformed_json, wrong_document.
This revises D4, using qwen-only evidence (SP4 4b).
Held-out qwen is dropped for good (21 train runs after D6).
Model: sklearn HistGradientBoostingClassifier.
It is already a dependency, so nothing new is added.
It handles NaN natively and learns which way missing values go at each split.
Rejected: XGBoost (uv add xgboost). It is the same model family and gains nothing here.
Fixed hyperparameters, no tuning on any test family: max_depth 3, learning_rate 0.05, max_iter 300, l2_regularization 1.0, early_stopping off, random_state 0.
Sensitivity: two other fixed configs, reported but not gated.
Training rows:
one row per (uid, t) for t = 2..8 on the training families, with D8 labels (failed with tau <= t = 1; healthy = 0; failed with tau > t excluded);
sample weight 1 / (the run's number of rows), so long runs don't dominate;
classes: headline + error-visible + API all included as positives in training (the monitor must flag any failure). Evaluation filters by class.
Label-free re-expression (core piece: calibration logic).
Healthy-percentile transform per feature and checkpoint: x → the mid-rank fraction of that family's HEALTHY prefix values at the same t that are ≤ x.
It is bounded to [0,1] and can't divide by zero (z-scores explode when a feature is constant on healthy runs, e.g. is_error on qwen). The idiom is np.searchsorted on the sorted healthy values.
Source families: transformed by their own healthy runs, per corpus, because corpora differ in task and tools.
Target family, cross-fitted by 5 task-group folds: fold f is transformed with healthy runs from the other folds only. The runs used for the transform are never scored with it.
Sensitivity: z-score with a std floor.
D6 reading: calibration is not training.
Computing per-feature percentiles or a threshold from gemini healthy runs, used only to score gemini at evaluation time, counts as evaluation-time calibration, not training.
Nothing fitted on gemini is saved or reused.
This is a conservative reading of "evaluation only", not legal advice. Override → the gemini arms run raw only.
ESN baseline, the stronger of two variants per test family, decided on the test result (conservative toward us):
(i) qwen healthy runs from ollama7b only (E1's setting);
(ii) all training-pool healthy runs. It is scored at matched steps, so AUROC needs no standardisation.
GATE (pre-registered; written into configs/e2.yaml before the first run)
Test family: llama held out; train on the leave_one_family_out(llama) pool (1,337 qwen runs; D6, D7).
Primary metric: mean matched-step AUROC over t = 3,4, headline classes vs all healthy runs.
PASS if the paired task-group cluster-bootstrap 95% CI of (TL-pct − ESN_best) has lower bound > 0.
Not gated, all reported:
gemini (26–27 headline positives);
TPR@5%FPR (5% is 3–4 of 60–77 healthy runs);
every other arm.
Secondary, pre-registered expectations (the direction is stated so it can be wrong):
TL-pct > TL-raw on llama;
TL-pct ≥ MS-pct on llama.
FAIL → the negative result is written up; SP6 (thresholds from few healthy runs) becomes the main contribution.
```

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
- 2026-09-29 · SP4 · split logic, D7 + D6 (data.py: drop_runs_without_tool_calls; splits.py:
  eval_only_families in leave_one_family_out and grouped_kfold) · Q1: D7 drops 787 runs from
  load_episodes itself, not inside a split function; what would go wrong in the grouped folds and the
  cluster bootstrap if they were kept? Q2: why is removing gemini from train NOT counted in
  n_dropped, and why does grouped_kfold raise for gemini instead of returning folds? Q3: after D6,
  held-out qwen trains on 21 llama runs; why can't we fix that by relaxing the task-overlap drop for
  that one split? · answers: not yet given · PASSED (owner-marked 2026-09-29, not answered; owed
  before deploy).
- 2026-09-29 · SP4 · matched-step metrics (src/watchdog_agent/evaluation.py) · Q1: at a matched step
  every run has the same prefix length; why does that remove the length confound, and what
  length-related signal can still leak in through a per-step score? Q2: why is a failed run with
  tau > t excluded rather than counted as healthy at t? Q3: why one task-group draw per replicate
  shared by all checkpoints, and why does a one-class checkpoint skip the whole replicate? ·
  answers: not yet given · PASSED (owner-marked 2026-09-29, not answered; owed before deploy).
- 2026-09-29 · SP4 · feature definitions (src/watchdog_agent/features.py) · Q1: is_error is MI, but
  healthy runs err at 0.03 (qwen), 0.38 (llama), 0.32 (gemini); in what sense is it still
  "model-independent", and what will a qwen-fitted monitor do with it on healthy llama runs? Q2:
  why is the text hash its own TEXT group instead of MS, and what would the SP5 MS-vs-MI ablation
  wrongly conclude if it stayed in MS? Q3: surprisal is NaN on every gemini step; why is that a
  reason to keep it out of MI, and how does XGBoost turn missingness into a feature? · answers:
  not yet given · NOT ANSWERED, owner override twice (SP4 close; SP4 close-out, whose own spec said
  "Do NOT owner-mark it"). Not graded, so not recorded as passed. Owed before deploy (SP9). Group
  renamed MI -> TL in the close-out; Q1 reads "is_error is TL".

- 2026-09-29 · SP5 · recalibration: healthy-percentile re-expression
  (src/watchdog_agent/recalibration.py) · Q1: why a mid-rank percentile instead of a z-score, what
  happens to is_error count on qwen (constant 0 on healthy runs) under each, and what does the
  percentile throw away that a z-score keeps? Q2: why must a healthy target run never sit in its own
  reference, which way would in-sample bias push the AUROC, and why fold by task group rather than
  by run? Q3: the transform needs to know which target runs are healthy; in what sense is it still
  "label-free", what breaks if some reference runs are secretly failed, and what asymmetry does
  transforming the source corpora in-sample (not cross-fitted) create? · answers: not yet given ·
  NOT ANSWERED, owner override 2026-09-29 (Step 2 started); overridden again 2026-09-30 to run E2
  (Step 3). Claude disagreed: it is the calibration logic the gate result rests on. SP5 done criteria
  require it answered. Eight teach-backs owed before deploy (SP9).

## Session log
- 2026-09-30 · SP5 (Step 3, 3b) · Recalibration teach-back overridden again (still NOT ANSWERED).
  E2 run from clean 860249b: GATE FAIL, TL-pct − ESN_i = −0.197 [−0.302, −0.057]
  (results/e2_ablation.json, e2_prefix_scores.csv). run_e2.py crashed at the MLflow step on "+" in
  param keys: fixed, saved result logged to MLflow (d3a8305364a64be29c2425b4a4449871). Step 3b
  diagnostics (experiments/e2_diagnostics.py, scripts/run_e2_diagnostics.py, 8 tests): the ESN
  0.646 -> 0.846 gap is the fit pool, not class mix or pooling; surprisal is not the source
  (ESN_i without u 0.905). results/e2_diagnostics.json is from a dirty tree · tests: pass (425, ruff
  clean) · next: rerun `uv run python scripts/run_e2_diagnostics.py` from this commit, then paste
  both E2 JSONs into the chat Project for the write-up spec.
- 2026-09-29 · SP4 (close-out) · Looping and context_corruption checks (no feature added; findings
  in Open questions); feature group MI renamed TL everywhere (fffba52, 390 tests unchanged); audit
  JSON regenerated from clean fffba52, values identical after key mapping (e9f4179);
  feature-definitions teach-back asked, owner overrode all questions: recorded NOT ANSWERED; item 4
  (Proposed for SP5) not done, the source text is only in the Project. Tag sp04-closeout · tests:
  pass (390, ruff clean) · next: paste "Proposed for SP5" to record it, then get the SP5 spec.
- 2026-09-29 · SP4 (close) · Step 0 report (fields, rosters, checkpoints, injection artefacts; D8, D9,
  checkpoints 3/4 + 2, TEXT group, 4a matched pairs in docs/decisions.md); Step 1 D7 in load_episodes
  (787 dropped, exact) + D6 eval_only_families in splits, sp2 summary rerun (qwen groups 1,003 -> 216),
  data card licence Apache-2.0; Step 2 evaluation.py (matched-step AUROC, shared-draw task-group
  cluster bootstrap); Step 3 features.py (118 features, per-feature leakage tests, D9 checks); Step 4
  experiments/sp4_audit.py + scripts/sp4_feature_audit.py -> results/sp4_feature_audit.json from clean
  e9ef64f; scikit-learn moved to main deps. Tag sp04-features · tests: pass (390, ruff clean) · next:
  paste the audit JSON into the chat Project and get the SP5 spec.
<!-- /save appends here, newest first: date · SP · what changed · tests · next step -->
- 2026-09-27 · SP3 (D-5, after tag) · Paired ESN-seed sweep (seed_sweep_rows / decide_seed_sweep /
  dominant_task_and_length_checks in experiments/e1.py, scripts/run_e1_seed_sweep.py,
  configs/e1_seed_sweep.yaml with the pre-registered rule, 3 tests); results/e1_seed_sweep.json from
  clean commit 43cb2c2: CLAIM HOLDS (cluster upper bound < 0.15 in 30/30, pooled median -0.005;
  per-fold favours B 30/30, median +0.074); length confound (Spearman 0.85/0.80, length AUROC 0.592).
  docs/sp3_e1_result.md headline + D-5 + length limit; docs/decisions.md D-5 + SP4 proposals · tests:
  pass (219, ruff clean) · next: get the rewritten SP4 spec from the chat Project.
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
