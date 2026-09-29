# Decisions

Earlier data decisions D1–D6 are in `docs/data_card.md`. Numbers below are read from the committed
results files; the JSON key path is given with each.

## 2026-09-27 — SP3 close-out

### SP3 gate: FAIL under rule (a) and rule (b)
Source: `results/e1_collapse.json` (clean commit 1a3787a, `git.dirty` false). 193 llama3.1:8b runs
(`counts.healthy` 77, `counts.failed` 116), task-disjoint grouped 5-fold.

| | value | key |
|---|---|---|
| AUROC A (qwen-fitted) | 0.644 [0.554, 0.730] | `monitors.A.auroc` |
| AUROC B (llama-refitted) | 0.634 [0.547, 0.717] | `monitors.B.auroc` |
| gap B − A | −0.009 | `gate.gap_B_minus_A` |
| paired run-bootstrap 95% CI of B − A | [−0.046, 0.025] | `gate.rule_b.paired_ci_B_minus_A` |

- Rule (b) needs gap ≥ 0.15 and the paired CI's lower bound > 0. Gap −0.009, lower bound −0.046: FAIL
  (`gate.rule_b.pass` false).
- Rule (a) needs gap ≥ 0.15 and non-overlapping individual CIs. The CIs overlap: FAIL
  (`gate.rule_a.pass` false).
- Sanity flag: B < 0.70 (`flags.baseline_B_below_sanity_auroc` true).

### Direction: reframe
- On task-disjoint splits the llama-fitted and qwen-fitted monitors score the same (above), so the problem
  is unseen tasks and families, not the model swap as such.
- Recalibration after a swap is about thresholds, not ranking.
- The SP4/SP5 specs will be rewritten in the chat Project; the SP5 gate is decided there.

### D7: drop runs with no tool calls
787 runs in 7 corpora have no tool call at all (`results/e1_diagnostics.json`,
`d1_twins.per_corpus.<corpus>.runs_without_tool_calls`):

| corpus | runs without tool calls |
|---|---|
| demo7b | 113 |
| demo7b_scoped | 120 |
| demo_real | 48 |
| real_ollama7b | 61 |
| real_research3b | 82 |
| real_research7b | 291 |
| real_research7b_long | 72 |
| **total** | **787** |

These are the same 787 rows as SP2's task-orphan rows, same corpora and counts
(`results/sp2_split_summary.json`, `unknown_task_rows_per_corpus`). Implemented in the SP4 data
change, not now.

### SP5 baseline to beat
The author's ESN (`esn_cusum_max`, vendored at 1b3e07f), run on the same task-disjoint splits as the
new monitor.

### D-5: ESN-seed sweep outcome — CLAIM HOLDS
Source: `results/e1_seed_sweep.json`. E1 was repeated over 30 paired ESN seeds (1300–1329), under a rule
pre-registered in `configs/e1_seed_sweep.yaml`. The outcome is CLAIM HOLDS (`outcome`):
- the task-cluster 95% upper bound of B − A is below 0.15 in 30 of 30 seeds, max 0.126;
- pooled B − A has median −0.005 (`summary`).
Mean per-fold AUROC favours B in 30 of 30 seeds (median +0.074, max +0.114; `rows[*]`), so a smaller
model-swap effect may remain.

### Proposed for the SP4 spec (not decided)
- A length-only baseline in SP5. Run length alone scores AUROC 0.592 pooled, and the ESN score's Spearman
  correlation with length is 0.85 / 0.80 (`seed_1300_checks`).
- Matched-step prefix evaluation (score every run at the same step k) instead of the episode maximum, which
  rewards longer runs.

### Reporting from SP4 on
- The task-group cluster bootstrap is the primary CI; the run bootstrap is secondary.
- Every headline number reports n_healthy.

## 2026-09-29 — SP4 Step 0 decisions (owner "go" on the Step 0 report)

### D8: labels at a matched checkpoint t
A run is in the checkpoint-t set if it has a step t (T >= t + 1). Positive: failed with tau <= t.
Negative: healthy. Excluded: failed with tau > t (its failure has not happened yet at t).

### D9: fields no feature may read
`failure_class`, `tau`, `T` / `n_steps` (future length), all of `metadata.*`, `corpus`, `model`,
`family`, `uid` / `episode_id`, `task_group`, `has_logprobs`, `tool_events[].id`,
`tool_events[].source` (present only in the `*_real` corpora), `steps[].schema` (constant 5).
Features read step fields of steps 0..t only.

### Checkpoints
t = 3 and 4 primary, t = 2 secondary. At t >= 5 llama's healthy runs come from 2 (t = 5) and 1
(t = 6) task groups, too few for a task-group cluster CI.

### Feature groups: TEXT is its own group
On tool steps `text` is `[name(args) -> result]`: the tool result is in the text for 100% of
events, including injected strings. The char-3-gram hash (the ESN's input) is group TEXT, not MS.

### Step 4a uses matched corpus pairs
llama only runs booking tasks and gemini only research tasks, so a family-vs-family classifier on
all runs mostly detects the task domain. 4a compares ollama7b vs ollama_llama8b (same framework and
tools) and qwen research corpora vs real_gemini_long.

### Consequence of D6 for held-out qwen
With gemini out of every train set, `leave_one_family_out(test_family="qwen")` trains on 21 llama
runs (172 dropped for task overlap) (`results/sp2_split_summary.json`,
`leave_one_family_out_overlap_drops.qwen`). Holding out qwen is not a usable setting.

## 2026-09-29 — SP5 Decisions and GATE (verbatim from the SP5 spec; also closes SP4 close-out item 4)
Pre-registered: `configs/e2.yaml` encodes these and is committed before any E2 model is trained.

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
