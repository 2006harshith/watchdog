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

## 2026-10-26 — SP5 close (write-up: `docs/sp5_e2_result.md`)

### SP5 gate: FAIL, as pre-registered
Source: `results/e2_ablation.json` (clean commit 860249b). On llama, TL-pct − ESN_i = −0.197
[−0.302, −0.057] (`gate.difference`, `gate.outcome` FAIL). The gate is final. The close-out diagnostics
(D-6, D-7, attribution) are for interpretation only.

### Baselines are reported as a distribution over fit draws, from SP6 on
- Why: the ESN's score depends on its healthy fit set. 30 random 21-of-28 draws give median 0.770, range
  0.566–0.857, and E2's single split (0.846) is matched or beaten by only 1 of 30
  (`results/e2_closeout.json` `d6.subsets`). A single-split baseline can sit in the tail without anyone
  noticing.
- Rule: every baseline fitted on healthy runs (ESN, and any refitted comparator) is reported as the median
  and range over ≥ 30 random fit-set draws of the stated size, with the pre-registered split's percentile
  in that distribution. Gates that compare against a baseline name in the spec, before the run, which
  statistic they use (a single pre-registered split, or the median draw).
- Rejected: keep a single pre-registered split. It is honest but hides a ±0.1 swing that is larger than
  most of the effects we test.

### Next: SP6 (the pre-registered fail path)
Per the SP5 GATE `on_fail`: the negative result is written up, and SP6 (E3, thresholds from few healthy
runs) becomes the main contribution. D-6 is direct evidence for it: the ESN moves by ±0.1 with which few
healthy runs it sees. The SP6 spec comes from the chat Project.

## 2026-10-26 — SP6 Decisions and pre-registered expectations (verbatim from the SP6 spec)
Pre-registered: `configs/e3.yaml` encodes these and is committed before the first E3 run. SP6 has no gate.

```text
Why SP6 looks like this
SP5's TL-pct ranks gemini failures (AUROC 0.716) but catches none at 5% FPR. Ranking without a threshold is not a monitor.
The ESN's score moves ±0.1 with its fit set (D-6). Any baseline is a distribution over fit draws (decisions 2026-10-26).
The pitch under test is "swap the model, recalibrate from a few healthy runs". So the n healthy target runs must pay for everything the monitor needs on the new model: the percentile reference (TL-pct), the ESN refit (ESN-refit) and the threshold.
Scope: llama is the primary family. Gemini is descriptive only. Gemini already generated the SP5 hypothesis, so it cannot test it; that test is SP8's, on new data.
SP6 has no gate (HANDOFF §11). Instead it has pre-registered expectations, written into config before the run. Each one can come out wrong.
Decisions (defaults chosen; override before pasting)
Alarm rule. A run alarms by checkpoint t if its running max score over the scored checkpoints {2..t} exceeds θ. ESN: the running max of the CUSUM stream over steps 0..t, as in SP4/SP5.
FA(t) = share of held-out healthy runs with a step t that alarm by t.
TPR(t) = share of held-out headline positives (tau <= t) that alarm by t.
Early alarms (an alarm before tau in a failed run) are reported separately.
t = 3, 4; headline = mean over t = 3, 4.
Threshold rules from m calibration scores (healthy, held out from the fit):
Conformal: θ = the ⌈(m+1)(1−α)⌉-th smallest calibration score. If that index is greater than m, then θ = +∞ (no alarms, reported as such). This rule guarantees E[FA] ≤ α only if calibration and test runs are exchangeable. Across task groups they may not be, which is what E-1 tests.
Plug-in: θ = the empirical (1−α) quantile. It has no guarantee; it is shown as the naive comparator.
α ∈ {0.05, 0.10}. Primary α = 0.10. At α = 0.05 conformal needs m ≥ 19; at 0.10, m ≥ 9.
How the n target healthy runs are spent:
Arms that need nothing else from the target (TL-raw, MS-raw, ESN-src) use all n for calibration (m = n).
TL-pct and ALL-pct use n/2 for the percentile reference and n/2 for calibration (m = n/2). The calibration runs must not sit in their own reference.
ESN-refit fits on n/2 and calibrates on n/2.
Rejected: leave-one-out inside the n. It needs n ESN refits per draw, and using different rules per arm would make the comparison unfair.
So TL-pct pays for its reference in calibration data. That is the honest cost of the transform.
Budgets n ∈ {0, 10, 20, 40}.
n = 0: the source threshold, i.e. θ from the source family's own healthy validation runs. Only arms that need no target data (TL-raw, MS-raw, ESN-src).
n = 40 is skipped in a fold if fewer than 40 eligible healthy runs exist; the skips are counted.
Cross-fitting by task group:
target runs are split into 5 task-group folds (the SP5 target folds, seed 0);
for eval fold f, the n calibration and reference runs are drawn from healthy runs of the OTHER folds with a step t;
fold f's healthy runs are only ever test runs.
One draw = one calibration set per fold. FA and TPR pool the 5 folds' test runs.
Draws. R = 200 per (arm, n) for the HGB arms and ESN-src; R = 30 for ESN-refit (cost).
ESN-src: each draw also picks one of 30 random 21-of-28 source fit sets (D-6 pool). Fit-set variance and calibration variance are integrated, per the 2026-10-26 rule.
Paired draws: arms compared at the same n use the same calibration runs where their m matches. Where m differs, the n runs are the same and the split is by a fixed seed.
Arms:
TL-raw, TL-pct, MS-raw, ALL-pct: the SP5 HGB models, trained once on source with the E2 config, not retrained;
ESN-src: fitted on qwen, threshold recalibrated;
ESN-refit: fitted on the target's own n/2 healthy runs. It is the direct competitor: if refitting the ESN on 20 llama runs wins, "recalibrate a source-trained monitor" adds nothing.
Reporting:
for each (arm, n, rule, α, t): FA median, 10th and 90th percentile and mean over draws, plus P(FA > 2α); TPR median and 10th–90th percentile; early-alarm rate; skip counts;
for each pre-registered comparison: the per-draw paired difference (median, 10th–90th percentile, share of draws > 0).
The spread over draws is calibration-set variability. Test-set sampling is NOT in it; say so in every table.
Pre-registered expectations (written into configs/e3.yaml and committed before the first run)
E-1 validity (llama): for every arm and every n where conformal at α=0.10 is defined, mean FA over draws ≤ 0.12 (α + 0.02 Monte Carlo slack). If this fails, calibration across task groups is not exchangeable, and "a few healthy runs" cannot certify a false-alarm rate on unseen tasks.
E-2 (llama): at n=20, conformal α=0.10, the median paired TPR difference TL-pct − ESN-src is ≥ 0.
I expect this to FAIL: SP5's AUROC gap was −0.20.
It is pre-registered anyway so the main contribution is measured, not narrated.
E-3 (llama): at n=20, ESN-refit TPR median ≤ ESN-src TPR median. That is, refitting on 10 target runs does not beat recalibrating a source-fitted ESN.
Direction is open. SP3's B vs A gap was −0.009.
E-4 (llama): plug-in at n=10 gives mean FA > α for every arm (the small-sample optimism of quantiles).
Gemini: every table is reported, nothing is tested, and it is labelled descriptive.
```

Filled in at Step 0 (owner "go" on the Step 0 plan, 2026-10-26, all 12 recommendations accepted;
each is marked FILLED IN (k) in configs/e3.yaml):
1. One θ per checkpoint t. A calibration run's score at t is its running max over {2..t}; only runs
   with a step t count.
2. n=0 for the HGB arms: θ from out-of-fold source scores (5 task-group folds of the source pool).
3. n=0 for ESN-src: θ from the draw's 7 left-out pool runs. m=7 < 9, so conformal θ=+∞ at both α.
4. ESN-refit runs on llama only: a fit on gemini's healthy runs is a model fit on gemini (D6).
5. Gemini ESN-src fit draws: 30 random 42-of-70 draws (gemini's ESN_i pool); the other 28 are source
   validation.
6. E-1 excludes n=0 (a source threshold is not a target calibration); n=0 is reported separately.
7. Per draw, headline TPR = mean of TPR(3) and TPR(4). E-3 compares medians; ESN-refit's 30 draws
   pair with ESN-src's first 30 draws.
8. E-4 uses α=0.10 and every arm at n=10, including the m=5 arms.
9. Plug-in uses np.quantile(method="linear"), numpy's default. Rejected: "higher".
10. TPR(t) counts any alarm by t, including one before τ. The early-alarm rate is the share of those
    positives whose first alarm comes before τ.
11. The reference/calibration halves are split by run with a fixed seed. They may share task groups;
    E-1 measures the effect.
12. ESN-refit: the calibration half is passed as ESNBaseline's validation runs (they set only its
    unused native θ). A fit failure makes the cell UNDEFINED, counted.
A reading of the decision text, not a new choice: E-1 and E-4 use headline FA (mean of FA(3) and
FA(4)) per draw.

Step 0 counts (results/e3_step0_counts.json): n=40 is infeasible in exactly one cell, llama t=4 fold 1
(29 eligible; `families.llama.per_t.4.folds_skipped.40` = 1). Fold 1 holds 31 of llama's 60 healthy
runs with a step 4 (`families.llama.per_t.4.per_fold[1].test_healthy`), so llama FA(4) at n=40 pools
only the other 29 test healthy runs. The n=40 headline therefore mixes a 5-fold t=3 and a 4-fold t=4.
Gemini is feasible everywhere. ESN-src pools match the config (llama 28, gemini 70).
