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

## 2026-10-27 — SP6 close-out diagnostics D-8..D-12: design and expected outcomes (POST HOC)

Everything in this section is post hoc. E-1..E-4 above are the pre-registered verdicts and are
reported unedited. These expectations are written and committed before any D-8..D-12 number is
computed. When they were written, the author had seen the E3 headline (the E-1/E-4 cells, the
E-2/E-3 comparisons) and the Step 3 counts per fold and task group. Nothing else.

Input: results/e3_group_counts.parquet from the clean d0cd3e9 rerun. That file holds per (cell,
draw, fold, task group) counts: healthy, healthy alarms, positives, positive alarms by t, pre-tau
first alarms, and post-onset alarms (some score > θ at a step in tau..t). Its sums equal
results/e3_draws.csv exactly, which the runner checks. Config: configs/e3_diagnostics.yaml. Code:
experiments/e3_diagnostics.py and scripts/run_e3_diagnostics.py, which write
results/e3_diagnostics.json. Llama is primary; gemini gets the same tables, descriptive only.

The n=40 skip (Step 0): at llama t=4, fold 1 has 29 eligible calibration runs, fewer than 40, so
n=40 never tests fold 1 at t=4. Fold 1 holds 31 of the 60 healthy runs and 57 of the 79 headline
positives at t=4, and one of its task groups has 23 runs, 30% (t=3) and 38% (t=4) of all llama
healthy test runs. So E3's n=40 headline mixes a full t=3 with a t=4 that has a third of the runs.
No n-curve that crosses n=40 is read without D-8.

### D-8 Common test set
Design: the same test runs for every n ∈ {10, 20, 40}. That is all 5 folds at t=3 (77 healthy, 92
positives) and folds {0, 2, 3, 4} at t=4 (29 healthy, 22 positives). Re-pool FA and TPR from the
group counts. Headline = the mean of t=3 and t=4. Report the old and new E-1 mean FA, and the TPR
median, for every arm at conformal α=0.10.
Check (not an expectation): n=40 reproduces E3 exactly, because it is already this set.
Expected:
(a) TL-pct, ALL-pct and ESN-refit mean FA at n=10/20 moves by more than 0.02, downward if fold 1
    carries the excess (D-9).
(b) Those three arms still fail E-1's 0.12 at n=20, because t=3 keeps fold 1.
(c) MS-raw and ESN-src move by at most 0.02.
(d) On the common set, TL-pct/ALL-pct/ESN-refit mean FA at n=40 is not more than 0.03 below n=20:
    their FA does not converge as n grows.

### D-9 Fold decomposition
Design: FA(t) per fold for TL-pct, ALL-pct and ESN-refit at n=20 and n=40, conformal α=0.10. Each
fold's FA pools its test healthy runs over draws: total alarms / total healthy, which equals the mean
over draws because the denominator is fixed per fold. Reported: fold 1 against the other folds pooled,
plus fold 1's share of the excess alarms Σ(alarms − α·healthy). n=40 has fold 1 at t=3 only.
Reading, fixed now: a cell's excess is CONCENTRATED in fold 1 if fold 1's FA > 0.20 and the pooled
other folds' FA ≤ 0.12 at every t where fold 1 is tested. It is SPREAD if the other folds' FA > 0.12.
Expected (the spec's hypothesis): concentrated in fold 1 for all three arms at n=20 and n=40. If it is
spread, the task-shift story (one big task group drives E-1) is wrong, and the write-up says so.

### D-10 Exchangeable benchmark
Design: conformal α_eff(m) = 1 − k/(m+1) with k = ⌈(m+1)(1−α)⌉, and 0 when k > m (θ = +∞).
Plug-in E[FA](m) = 1 − (h+1)/(m+1) with h = (m−1)(1−α), for np.quantile(method="linear") under iid
Uniform scores. Every table gets these columns, m per arm (m = n for TL-raw, MS-raw and ESN-src; n/2
for TL-pct, ALL-pct and ESN-refit), and observed mean FA − expected.
Verification: 1e5 iid replicates per (rule, α, m), tolerance 0.005.
Caveat fixed now: conformal α_eff is distribution-free for any continuous scores. The plug-in formula
is exact for Uniform scores only, because linear interpolation between order statistics is not
invariant to the score distribution. It is verified on Uniform (tolerance 0.005), and Normal and
Exponential are simulated and reported, expected within 0.01 of the formula. Real scores tie (TL
percentiles, ESN washout zeros), and with the strict ">" rule ties push observed FA BELOW the
exchangeable value.
Expected:
(a) The simulation confirms both formulas within tolerance on Uniform.
(b) MS-raw and ESN-src are within ±0.03 of the exchangeable value at every n ≥ 10, for conformal and
    plug-in.
(c) TL-pct, ALL-pct and ESN-refit are more than 0.10 above it at n=20 and n=40 (conformal).
(d) E-4 holds mechanically: plug-in E[FA] at m=10 is 0.173, and 0.233 at m=5, both above α=0.10
    even under exchangeability.

### D-11 Post-onset TPR
Design: TPR_post(t) = positives with an alarm at a step in tau..t / positives. Headline = the mean
over t=3, 4 per draw. Post-onset alarms are a subset of alarms by t, so TPR_post ≤ TPR in every draw
(a check). E-2 and E-3 are restated with TPR_post, and the per-draw (FA, TPR) and (FA, TPR_post)
points are stored. E-3 is read both ways: over the 30 paired draws, and ESN-refit's 30-draw median
against ESN-src's 200-draw median. The 30 paired ESN-src draws' median TPR is placed among the medians
of 10,000 random 30-of-200 subsets (seeded), as a percentile.
Expected:
(a) TPR_post drops most for the high-FA arms (TL-pct, ALL-pct, ESN-refit), because part of their
    pre-tau alarms ride on the same low threshold that gives false alarms.
(b) E-2's median paired difference with TPR_post is below +0.115, and its share of draws > 0 falls
    below 0.80.
(c) E-3's ESN-refit − ESN-src stays > 0 under both readings: refit buys TPR with FA.
(d) The 30 paired ESN-src draws' median sits below the 10th percentile of 30-of-200 subset medians.

### D-12 Test-side uncertainty
Design: a task-group cluster bootstrap over each family's test task groups, B = 2000 with seed 0. The
calibration draws are held fixed: each replicate re-weights the per-group counts of every draw. The
groups are resampled with replacement over all test task groups. There is one resample per
replicate, shared across t, n, arms and draws, so paired comparisons stay paired (as in
evaluation.py). Each statistic is recomputed over all draws (mean FA, median TPR, median paired
difference), and the result is a percentile 95% CI. A replicate whose denominator is 0 for a
statistic is skipped for that statistic and counted. The CI covers every cell the write-up quotes.
Fold 1's 23-run group makes some intervals wide; they are reported anyway.
Expected:
(a) The FA CIs of TL-pct, ALL-pct and ESN-refit at n=20 are wider than 0.15.
(b) MS-raw's mean-FA CI at n=10/20 contains 0.10.
(c) The CI of E-2's median paired difference (any-alarm TPR) contains 0.

## 2026-10-28 — SP6 close-out part 2, D-13: design and expected outcomes (POST HOC)

Post hoc, as D-8..D-12. E-1..E-4 and the D-8..D-12 outcomes (including D-8d, D-9 and D-10a FAILS)
are reported unedited. Written and committed before any D-13 number is computed. When this was
written the author had seen E3's headline, results/e3_diagnostics.json (D-8..D-12) and the code of
e3.py, and nothing per fold for MS-raw, ESN-src or TL-raw.

Owner override, 2026-10-28: the SP6 threshold-logic teach-back is overridden a third time to start
this work. NOT ANSWERED, not owner-marked. Claude disagreed (the write-up's lead is this logic, and
tag sp06-e3 needs it answered under SP6's done criteria). Risk accepted.

Input: results/e3_group_counts.parquet (the clean 0d3fc92 rerun that reproduces 711a23e). Nothing is
refitted or rescored. Config: configs/e3_diagnostics.yaml, block d13. Output: new keys in
results/e3_diagnostics.json; the D-8..D-12 values must come out unchanged.

### D-13a Per-fold FA of the arms that spend no target runs on fitting or reference
Design: FA(t) per fold for MS-raw, ESN-src and TL-raw at n = 10, 20, 40, conformal α = 0.10, t = 3
and 4. A fold's FA pools its healthy test runs over draws (total alarms / total healthy), as D-9.
Each row has n_test_healthy. Per-fold FA carries test-sampling noise that pooling over draws does not
remove (the same few test runs in every draw).
Reading, fixed now, per (arm, n, t): a fold is rated only if it has ≥ 5 healthy test runs at t.
- HIGH if the pooled FA over all tested folds is > 0.12.
- NOMINAL if not HIGH and every rated fold's FA is in [0.03, 0.20].
- OFFSETTING if not HIGH and some rated fold's FA is outside [0.03, 0.20].
- UNRATED if no fold is rated.
Expected: OFFSETTING for at least one of the three arms at n = 20 at some t. The folds are
task-group disjoint, so exchangeability between other folds' healthy runs and a fold's test runs is
not expected to hold fold by fold. If every cell reads NOMINAL, the write-up says the near-nominal
pooled FA of these arms is near nominal per fold; if OFFSETTING, it says the pooled value hides
offsetting fold errors.

### D-13b Fold 1's healthy test runs by task group
Design: per task group in fold 1, its healthy test runs at t = 3 and t = 4, with fold 1's total and
the all-fold total. Report only. No reading from run counts alone.

### D-13c Definitions of early_alarm_rate and tpr_post (from e3.py and thresholds.py)
- early_alarm_rate (thresholds.fa_tpr, e3._Pool.frame): per (cell, t, draw), pooled over folds,
  n_early / n_pos_alarm. Positives: headline classes, tau ≤ t, a step t. n_pos_alarm: positives
  whose first alarm (first step with score > θ) is ≤ t. n_early: those whose first alarm is < tau.
  The denominator is ALARMED positives, not all positives. A draw with no alarmed positive at some
  t is NaN, and E3's headline (mean over t) leaves it out of the median: a selection.
- tpr_post (e3.group_counts): positives with SOME score > θ at a step in tau..t, over all positives.
  That is "above θ at some step ≥ tau", not "first crossing ≥ tau". ESN arms score a running max
  (e2_diagnostics._esn_running_max), so once above θ they stay above it: for ESN-src and ESN-refit,
  tpr_post = TPR by construction. HGB arms score each checkpoint separately and can drop back
  below θ. So D-11a's ESN drops of 0.0 are mechanical, and D-11's E-2/E-3 restatements compare an
  any-alarm ESN rate with a stricter HGB rate.
- Added: first-crossing TPR = (n_pos_alarm − n_pos_alarm_pre_tau) / n_pos, the positives whose
  first alarm is at a step in tau..t. Same meaning for every arm. Reported per cell (conformal
  α = 0.10), with E-2 (TL-pct − ESN-src, n = 20) restated as a paired median difference and a D-12
  bootstrap interval (same B, seed and resample).
Checks, expected to hold (mechanical, not findings): (i) first-crossing ≤ post-onset ≤ any-alarm in
every group row of every cell; (ii) post-onset = any-alarm in every group row of ESN-src and
ESN-refit; (iii) post-onset > first-crossing in at least one row of some HGB arm (otherwise the
distinction does not matter on these data, and the write-up says so).

### D-14 Half split by task group (needs a rerun)
Written and committed before the run, after D-13's results. The author had seen every D-8..D-13
number, the eligible pools' task-group sizes and two timings (an ESN refit takes about 1 s, a
percentile re-reference 0.06 s). No FA or TPR under a group split.

Question: does the by-run half split explain the split arms' excess FA? Under it, the calibration
half shares task groups with the reference half (TL-pct, ALL-pct) or the fit half (ESN-refit). The
test fold's groups are new to both halves. So calibration scores may look more normal than test
scores, and θ comes out too low.

Design (llama only: ESN-refit is llama-only and llama is primary; arms TL-pct, ALL-pct, ESN-refit;
n = 20 and 40; t = 3, 4; conformal and plug-in at α = 0.05 and 0.10):
- Per (t, fold f, n, draw r), split the eligible pool's task groups (E3's eligible runs: healthy,
  other folds, a step t) at random into sides A and B. Each group goes to A with probability 1/2,
  from default_rng([half_seed, family, t, f, n, r, try]). Retry with try = 0, 1, ... until both sides
  hold ≥ n/2 runs, up to 1000 tries. If no try works, the fold is skipped at that (n, t) and counted.
- The reference/fit half is n/2 runs from A; the calibration half is n/2 runs from B,
  default_rng([seed, family, t, f, n, r]). No task group is in both halves. m = n/2 exactly, as in E3.
- Everything else as E3: the same source models, test runs, folds, R (200; ESN-refit the first 30),
  seeds, θ rules, n=40 skips and task-group counts.
- Rejected: a task-group split of E3's own n picked runs. It keeps run-level pairing with E3, but m
  would vary per draw (one eligible group has 23 runs; the others have 1–4). That changes α_eff(m),
  and at m < 9 θ = +∞. The design above loses pairing with E3's draws. So the two splits are compared
  as distributions on the same test runs, not draw by draw.
- Code path check: the D-14 loop run with E3's own by-run draws must reproduce E3's group counts
  exactly for n=20, t=3, fold 0 (all three arms).

Comparison, per arm and n, conformal α = 0.10. FA_run is from E3's group counts and FA_group from
D-14's. Headline mean FA over t = 3, 4, as E-1.
- excess = FA_run − α_eff(n/2), where α_eff = 0.091 at m=10 and 0.095 at m=20.
- r = FA_run − FA_group. The CI is a task-group cluster bootstrap over the test groups (B = 2000,
  seed 0), with one resample shared by both splits, so r's interval is test-side only.
- FALLS if r ≥ 0.5 · excess and r's 95% CI excludes 0. PARTIAL if r > 0.02 and not FALLS.
  DOES NOT FALL if r ≤ 0.02.
Expected: FALLS for TL-pct, ALL-pct and ESN-refit at n = 20. If DOES NOT FALL, the write-up says the
by-run split is not the explanation for that arm.
Also reported (descriptive, no reading): mean FA, P(FA > 2α), median TPR and first-crossing TPR, and
per-fold FA (as D-9) under both splits. Per (n, t, fold): where the 23-run group sits (reference/fit
half, calibration half, or neither), FA by placement, the number of distinct partitions and the
largest try count.
Known confounds, stated now: a group half spans fewer task groups than a by-run half, so the
reference or fit set is narrower, and that alone can raise FA and cancel part of any gain. That is
the first alternative if the result is DOES NOT FALL. The placement of the 23-run group shifts half
composition. At n=40 t=4 the retry condition restricts the possible partitions.

## 2026-10-28 — SP6 write-up (docs/sp6_e3_result.md): amendments to the Step 8 spec
Owner-approved ("add the amendments to the write-up and go"), after D-13 and D-14:
1. E-2's lead reports first-crossing TPR (−0.012 [−0.220, 0.069]) next to the any-alarm +0.115 and
   post-onset +0.081, because post-onset = any-alarm by construction for ESN arms (D-13c).
2. "MS-raw and ESN-src are near nominal on the common test set" becomes "near nominal pooled, with
   offsetting per-fold errors" (D-13a). The common-set range is 0.108–0.122 over n = 10–40, not the
   spec's 0.109–0.120, which covered n = 10–20 only.
3. Added to "not allowed": comparing tpr_post between ESN and HGB arms.
4. D-14's status: the by-run split is not the explanation for TL-pct; it explains part of ESN-refit's
   and ALL-pct's (n = 40) excess; the narrower fit set is an open confound.
Tag sp06-e3 is NOT applied: SP6's done criteria require the threshold-logic teach-back, still NOT ANSWERED.
- 2026-10-28 (later): owner override of SP6's done criterion; tag sp06-e3 applied with the
  threshold-logic teach-back NOT ANSWERED (fourth override for SP6). Claude disagreed. Risk accepted.

## 2026-10-28 — SP7 spec: owner decisions before Step 0
- Override: SP7 starts with the SP6 threshold-logic teach-back NOT ANSWERED. Claude disagreed.
- Judge sampling: temperature 0, 1 sample per (run, t), plus a determinism subset (3 samples on a fixed
  subset, SD reported). Rejected: T=0 x 3 (near-identical samples at 3x the cost).
- ESN-src's 30 fit-draw scores are not on disk (E3 kept only counts). They are recomputed exactly as E3
  did, and must reproduce results/e2_closeout.json (D-6 AUROCs) and results/e3_recalibration.json
  (esn_src.headline_auroc_per_fit_draw) exactly. HGB arms are reused from results/e2_prefix_scores.csv.
- Best judge is re-selected inside every bootstrap resample, so the CI includes the selection.
- The power check states the monitor-judge score correlation it assumes (ρ = 0, 0.3, 0.6).
- Step 0d covers qwen traces as well as llama (prompt development sends qwen traces to J-api).
- 2026-10-28, after the Step 0 report (owner: "drop the gate, Haiku thinking off ×3, no local judge"):
  - NO GATE. Step 0c's power check: 80% power needs δ = 0.196 / 0.178 / 0.137 at monitor-judge
    correlation 0 / 0.3 / 0.6, so a non-inferiority gate at δ ≤ 0.15 is uninformative here. D (median
    over ESN-src's 30 fit draws of AUROC_ESN-src,draw − AUROC_judge, paired task-group cluster
    bootstrap, B = 2000, seed 0) is reported with its CI; J-1 and J-2 stay pre-registered.
  - One judge, J-api = claude-haiku-5-5, thinking disabled, default sampling (the API rejects
    temperature since 2026-10-21), 3 samples per (run, t), score = their mean, sample SD reported.
    This replaces "T=0 × 1": T=0 cannot be set on this model. "Best judge" becomes the one judge.
  - No J-local: no runtime or model is installed, and the owner declined to install one.
- 2026-10-28 (later): owner declined to buy API credit. The judge becomes J-local: a Microsoft Phi
  model (~3.8B) served by Ollama on this machine (GTX 1650, 4 GB), installed via winget. Consequences:
  no third-party API, so the Llama-licence question for sending traces does not arise; temperature 0 is
  available again, so sampling returns to T=0 x 1 sample plus a determinism subset (3 samples on a fixed
  subset, SD reported); the cost row is $0 plus measured local GPU time. Superseded: J-api =
  claude-haiku-5-5 (thinking off x3). Limit stated in advance: a small local judge, so J-1 is about this
  judge, not LLM judges in general.
- 2026-10-28 (Step 2, before any scored call): a 3-call smoke test on qwen dev prompts showed
  phi4-mini answering "10", a blank line, then an explanation, despite "Output only the integer";
  every reply would fail the strict parser. Decoding now stops at the first newline (Ollama option
  stop: ["\n"]); the template (v1) and the strict parser are unchanged, so a reply that starts with
  words is still a counted parse failure. The decoding options are hashed into the cache key's model
  field. Smoke calls were direct (not cached, not scored). First call 144 s (58 s model load), then
  about 3 s per call with the stop.
- 2026-10-28: Step 2 closed. v1 frozen by the pre-set rule (0/240 parse failures on qwen, AUROC 0.738).
  configs/e4.yaml (prompt sha256 4bc3bf11..., J-local phi4-mini@78fad5d1, no gate, D, J-1, J-2
  verbatim) committed at 15b31e8, before the first llama judge call.
- 2026-10-28, SP7 Step 3, POST HOC (after the llama judge calls, before any statistic was computed):
  one of 479 scoring replies did not parse (real-context_corruption-039, t=4: prose, no number; 0.21%,
  under the 2% stop). Owner chose option 1: that (run, t=4) pair is dropped from the t=4 AUROC set for
  EVERY scorer (J-local, all 30 ESN-src fit draws, the HGB arms), so D stays paired on identical runs;
  t=3 is untouched. Sensitivity, reported next to it: the missing score set to 0 (worst case for the
  judge on a failed run), with no pair dropped. J-2 needs no rule: the run is failed, so it is never a
  calibration run nor in FA, and a missing step cannot alarm. The monitor reproduction checks keep the
  full E2 / D-6 sets. A missing score on a healthy run would still stop the run (not covered here).
- 2026-10-28: SP7 write-up docs/sp7_e4_result.md. No gate; D +0.132 [+0.045, +0.206]; J-1 FAILS (expected
  HOLD), J-2 HOLDS, both unedited. Tag sp07-e4 not applied: the SP7 teach-back on D is NOT ANSWERED.
- 2026-10-28 (later): owner override; tag sp07-e4 applied with the SP7 teach-back on D NOT ANSWERED
  (second override for it). Claude disagreed. Risk accepted.

## 2026-10-28 — SP8: AgentDojo dropped; organic failures from the dataset's own organic llama8b runs
- AgentDojo pilots (5 set-aside tasks x 2 seeds, both drivers): utility 0/20, median 1 LLM call per run,
  almost no run with a step 3 or 4. SP8 as specified is infeasible with llama3.1:8b on this machine.
- Owner chose option 3 ("option 3, go"): label a subset of the existing organic llama8b corpora
  (organic_llama8b, organic_llama8b_cold), with a written label rule fixed before any monitor score.
  This is a spec change made without the chat Project; the SP8 gate, c, K and O-1..O-4 are re-derived
  for the new data in a new Step 0 report before configs/e5.yaml is written.
- AgentDojo pilot artefacts stay (envs/agentdojo, data/agentdojo_pilot); its 5 pilot tasks are irrelevant now.
- 2026-10-28, owner ("strict rule, cold-corpus gate, drop O-2, go"), before any SP8 monitor score:
  - Label (strict): a run is HEALTHY iff its final synthesis step's text states metadata.expected_total
    (a number within 0.5 of it); otherwise FAILED. Counts on the final texts: 11/240 healthy
    (organic_llama8b 1/120, organic_llama8b_cold 10/120).
  - Gate on organic_llama8b_cold only (10 healthy vs 110 failed), c = 0.5, stated as low-powered;
    organic_llama8b (T = 0.9) is descriptive only, because the label is confounded with the corpus.
  - O-2 dropped: fewer than 20 healthy runs exist, so E3's n = 20 operating point cannot be run.
