# SP3 / E1: the transfer collapse does not survive task-disjoint evaluation

Files: `results/e1_author_repro.json` (repro), `results/e1_collapse.json` (E1), `results/e1_diagnostics.json`
(diag). Each number is followed by its file and JSON key path. A = monitor fitted on healthy qwen2.5:7b
runs (ollama7b); B = same monitor type refitted on healthy llama3.1:8b runs (ollama_llama8b); both scored on
llama runs. Monitor: the author's `esn_cusum_max` (channel-max ESN-CUSUM), vendored verbatim at 1b3e07f.

## 1. Claim tested
arXiv 2608.02464 §5: on the same llama3.1:8b runs, A reaches episode AUROC 0.527 and B 0.885, i.e. a cheap
monitor collapses when the agent's LLM family changes. SP3's gate asked for B − A ≥ 0.15 with a paired
bootstrap CI above 0, under our task-disjoint protocol.

## 2. Reproduction at author commit 1b3e07f (author's own protocol)
Healthy runs split 60/20/20 at random, test = 16 healthy + 116 failed (repro `ours.in_domain.n_test_healthy`,
`.n_test_failed`).
- B, in-domain: 0.8847 (repro `ours.in_domain.auroc`) vs author 0.8847 (`author.in_domain.auroc`). AUPRC,
  detection and false-alarm rates match too.
- A, transfer: 0.6805 (repro `ours.transfer.auroc`), not 0.527 (`author.transfer.auroc` 0.5275).

## 3. Task-disjoint E1 (193 runs: 77 healthy, 116 failed; `counts.*`)
Grouped 5-fold over llama task groups; A_f and B_f never fit on a task in fold f. Run-bootstrap CI (E1
`monitors.<m>.<metric>`) and task-group cluster-bootstrap CI (E1 `sensitivity.task_group_cluster_bootstrap.<m>.<metric>`).

| | A: run CI | A: cluster CI | B: run CI | B: cluster CI |
|---|---|---|---|---|
| AUROC | 0.644 [0.554, 0.730] | [0.495, 0.794] | 0.634 [0.547, 0.717] | [0.490, 0.754] |
| AUPRC | 0.666 [0.579, 0.762] | [0.520, 0.825] | 0.674 [0.586, 0.770] | [0.469, 0.820] |
| TPR @ 5% FPR | 0.026 [0.000, 0.090] | [0.019, 0.375] | 0.043 [0.009, 0.132] | [0.023, 0.467] |

- B − A: −0.009, paired run CI [−0.046, 0.025] (E1 `gate.rule_b.paired_ci_B_minus_A`), cluster CI
  [−0.190, 0.071] (E1 `sensitivity.task_group_cluster_bootstrap.paired_auroc_B_minus_A`). Gate FAIL under rule
  (b) and rule (a) (E1 `gate.outcome`, `gate.rule_a.pass`). B < 0.70 sanity flag set (E1 `flags`).
  Both the run and task-cluster 95% CIs of B − A exclude 0.15 (upper bounds 0.025, 0.071); the cluster CI
  still allows A to be up to 0.19 better.
- Mean per-fold AUROC: A 0.587, B 0.595 (E1 `monitors.<m>.mean_per_fold_auroc`). A's pooled-vs-mean-fold
  divergence flag (> 0.05) is set, B's is not (E1 `monitors.<m>.pooled_vs_mean_fold_divergence_flag`).
  Fold 1 holds 108 of 193 runs (E1 `counts.per_fold[1].eval_runs`).
- False alarms on healthy runs at each monitor's default threshold: 0.364 for both, n_healthy 77
  (E1 `monitors.<m>.default_alarm.healthy_fa_rate`, `.n_healthy`).
- Per-class AUROC, run CI (E1 `monitors.<m>.per_class_auroc.<class>`):

| class (n failed) | A | B |
|---|---|---|
| context_corruption (32) | 0.618 [0.508, 0.723] | 0.605 [0.491, 0.718] |
| goal_drift (32) | 0.574 [0.461, 0.688] | 0.550 [0.434, 0.667] |
| looping (29) | 0.730 [0.627, 0.824] | 0.692 [0.587, 0.791] |
| tool_cascade (23) | 0.669 [0.557, 0.773] | 0.719 [0.604, 0.822] |

## 4. Why the paper's gap appears
**Twins (D-1).** A twin is a run whose per-step tool calls (name, args, result) are identical to another's.
118 of 193 llama runs sit in twin clusters (diag `d1_twins.per_corpus.ollama_llama8b.runs_in_twin_clusters`);
no llama run has a qwen twin (diag `d1_twins.cross_corpus.shared_trajectories` 0). In the author's in-domain
split, 20 of 132 test runs have an exact twin in B's fit set and 121 share a task with it (diag
`d1_twins.author_in_domain_split.vs_fit`).

**Split ablation for B (D-2).** Same n_fit 46 and n_val 15, ESN seed fixed; 20 usable seeds per split (diag
`d2_split_ablation_B.<split>`; n_healthy/n_failed from `.per_seed[*]`).

| split | mean AUROC | SD | min–max | skipped seeds | test n_healthy mean (min–max) | test n_failed mean |
|---|---|---|---|---|---|---|
| author random | 0.884 | 0.060 | 0.706–0.976 | 0 | 16 (16–16) | 116 |
| twins on one side | 0.678 | 0.167 | 0.346–0.973 | 0 | 14.6 (13–16) | 108.2 |
| task-disjoint | 0.623 | 0.198 | 0.231–1.000 | 10 of 30 drawn | 11.7 (3–16) | 13.2 |

Random seed 0 is the author's split and gives 0.8847. Seeds are not paired across splits, so random and
twin-aware are compared unpaired, from `per_seed[*].auroc`: median 0.896 vs 0.683; 12 of 20 twin-aware seeds
fall below the lowest random seed (0.706); across all 20 × 20 pairs, the random seed's AUROC is higher in
0.875 of them.

The task-disjoint test sets are a different population (mean 11.7 healthy runs, min 3; mean 13.2 failed), so
twin-aware is the like-for-like comparison with random.

**Source task overlap for A (D-3).** A fitted on 20 qwen runs of the test set's tasks: mean AUROC 0.558 (SD
0.091); fitted on 20 qwen runs of other tasks: 0.591 (SD 0.080), 20 seeds each (diag
`d3_source_overlap_A.overlap`, `.excluded`). Sharing tasks gives A no advantage, which fits the absence of
cross-model twins.

Reading: B's 0.885 is mostly explained by healthy test runs that are exact twins of fit runs. Keeping twins on
one side of the split lowers B's median from 0.896 to 0.683 at a comparable test size. Under task-disjoint E1,
A and B score 0.644 and 0.634 on the same 193 runs. The model swap is not what separates A from B in the
paper's setting.

## 5. Limits
- One corpus pair (ollama7b → ollama_llama8b) and 193 llama runs.
- The author's protocol scores only 16 healthy test runs; the task-disjoint D-2 test sets are smaller still.
- One task holds 83 of 193 llama runs; whichever fold or split gets it dominates.
- Failures are injected, not organic. 10 llama and 13 qwen failed runs, all context_corruption, have tool
  calls identical to a healthy run (diag `d1_twins.per_corpus.<corpus>.runs_in_mixing_clusters.by_failure_class`),
  so no tool-level signal can separate them.
- ESN seed fixed at 1300 throughout; all spread shown is split variation.
- Task-disjoint D-2 skipped 10 of 30 seeds drawn (8 left a one-class test set, 2 could not be filled with
  whole task groups; diag `d2_split_ablation_B.task_disjoint.skipped_seeds`); its 20 seeds are a selected subset.
- Twins are defined on tool calls only; text, timing and logprobs are ignored.
- D-3 fits A on 20 runs (the author used 42), capped by the 28-run excluded pool.

## 6. What changes for the project
1. The target becomes unseen tasks and model families, not the model swap as such.
2. Recalibration after a swap is treated as a threshold problem, not a ranking problem.
3. From SP4 on: task-disjoint splits, task-group cluster CIs as primary, n_healthy on every headline, and the
   ESN on the same splits as the baseline to beat.
