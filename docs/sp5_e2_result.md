# SP5 / E2: tool-level features do not beat the ESN on held-out llama

The pre-registered gate fails. On held-out llama3.1:8b, TL-pct trails the author's ESN by 0.197 AUROC.
The ESN's lead is fragile, though. Its score moves by about ±0.1 with the choice of 21 healthy fit runs,
and E2's pre-registered split sits near the top of that range. Its llama signal comes from model-reaction
channels (m: action, latency, output length, error flag; u: surprisal), not from step-output content. SP6
(thresholds from few healthy runs) becomes the main contribution, as pre-registered.

Files: ab = `results/e2_ablation.json` (clean 860249b), dg = `results/e2_diagnostics.json` (clean a472577),
co = `results/e2_closeout.json` (clean ecadcfb), at = `results/e2_channel_attribution.json` (clean e33db31).
Every number is followed by its file and key. "Derived" marks arithmetic on JSON values that is not stored
in a JSON file. AUROC = mean matched-step AUROC over t = 3, 4, with a task-group cluster-bootstrap 95% CI.

## 1. Gate: FAIL, as pre-registered
The pass condition: the lower bound of the paired cluster CI of (TL-pct − ESN_best) on llama must be > 0
(`docs/decisions.md`, 2026-09-29; ab `config.gate`).
- TL-pct − ESN_i = −0.197 [−0.302, −0.057] (ab `gate.difference`). Outcome FAIL (ab `gate.outcome`).
  ESN_best = ESN_i (ab `families.llama.esn.best`). Per checkpoint: t=3 −0.175 [−0.279, −0.051],
  t=4 −0.220 [−0.352, 0.048] (ab `families.llama.paired.TL-pct_vs_ESN_best.per_checkpoint`).
- Secondary expectations: both hold in direction, but neither CI excludes 0 (ab `secondary_expectations`,
  `families.llama.paired.*.mean`). TL-pct − TL-raw = +0.007 [−0.044, 0.137].
  TL-pct − MS-pct = +0.096 [−0.043, 0.313].

## 2. Arms, llama (gated) vs gemini (not gated)
AUROC = ab `families.<fam>.arms.<arm>.headline.mean_primary`. TPR = TPR@5%FPR at t=3 / t=4, from ab
`….headline.tpr_at_fpr`. Headline positives / healthy runs: llama 92/77 (t=3), 79/60 (t=4); gemini 22/77,
21/66 (ab `….headline.n`). Gemini has no goal_drift positives, so it has 22, not the 26–27 the spec expected.

| arm | llama AUROC | llama TPR | gemini AUROC | gemini TPR |
|---|---|---|---|---|
| TL-raw | 0.641 [0.546, 0.723] | 0.07 / 0.19 | 0.592 [0.523, 0.686] | 0.55 / 0.57 |
| **TL-pct** | **0.648 [0.555, 0.768]** | 0.21 / 0.10 | **0.716 [0.578, 0.851]** | **0.00 / 0.00** |
| MS-raw | 0.858 [0.807, 0.937] | 0.68 / 0.63 | 0.685 [0.583, 0.785] | 0.00 / 0.00 |
| MS-pct | 0.552 [0.422, 0.618] | 0.00 / 0.05 | 0.383 [0.276, 0.500] | 0.00 / 0.00 |
| ENV-raw / -pct | 0.667 / 0.526 | | 0.532 / 0.786 | |
| TEXT-raw / -pct | 0.656 / 0.671 | | 0.667 / 0.728 | |
| TL+MS-raw / -pct | 0.729 / 0.647 | | 0.623 / 0.713 | |
| ALL-raw / -pct | 0.742 / 0.675 | | 0.651 / 0.780 | |
| **ESN_i** | **0.846 [0.770, 0.896]** | 0.54 / 0.57 | 0.481 [0.350, 0.615] | 0.00 / 0.05 |
| ESN_ii | 0.756 [0.662, 0.948] | 0.19 / 0.25 | 0.341 [0.218, 0.481] | 0.00 / 0.10 |

The percentile transform leaves TL unchanged on llama. It cuts MS from 0.858 to 0.552. Why MS-pct falls was
not investigated.

## 3. Per class, llama
AUROC point estimates, from ab `families.llama.arms.<arm>.per_class.<class>.mean_primary.point`. n_pos at
t=3/4 comes from `….per_class.<class>.n`. tool_cascade is error-visible, so it is outside the headline.

| class (n_pos) | TL-raw | TL-pct | MS-raw | MS-pct | ESN_i | ESN_ii |
|---|---|---|---|---|---|---|
| looping (29/27) | 0.912 | 0.861 | 0.808 | 0.661 | 0.739 | 0.796 |
| goal_drift (32/25) | 0.461 | 0.492 | 0.813 | 0.639 | 0.812 | 0.591 |
| context_corruption (31/27) | 0.557 | 0.601 | 0.948 | 0.361 | 0.978 | 0.876 |
| tool_cascade (18/20) | 0.802 | 0.773 | 0.800 | 0.665 | 0.640 | 0.790 |

TL beats the ESN on looping and loses on goal_drift and context_corruption, where MS-raw and the ESN reach
0.81–0.98. The gate is lost on those two classes.

## 4. D-6: the ESN baseline depends on which 21 fit runs it gets
- ESN_i refitted on 30 random 21-of-28 fit subsets of its own pool (co `config.n_subsets`,
  `d6.subsets.pool_runs`, `.fit_size`): median 0.770, range 0.566–0.857 (co `d6.subsets.median`, `.min`, `.max`).
- E2's pre-registered split scores 0.846 (co `d6.subsets.e2_split_auroc`). Only 1 of 30 draws scores at or
  above it (share 0.033, co `d6.subsets.share_of_draws_at_or_above_e2_split`), so it sits at about the 97th
  percentile of the distribution.
- TL-pct (0.648) is below 24 of the 30 draws (derived: co `d6.subsets.aurocs` vs ab
  `families.llama.arms.TL-pct.headline.mean_primary.point`). Its margin vs the median draw is about −0.12
  (derived: 0.648 − 0.770). This is one point against a median: it is not paired and has no CI, and it does
  not change the gate.
- The fit structure adds nothing. SP4's per-fold structure with E2's pool also gives 0.846 (co
  `d6.a_fit_pool_only.sp4_structure_e2_pool_standardised`).
- Fit-size effect: SP4's fold pools cut to 21 fit runs give a per-fold mean with median 0.788, range
  0.664–0.877 (co `d6.b_size_matched.per_fold_mean`). The full fold pools give 0.602 (dg
  `rows.sp4_A_standardised.per_fold_headline.mean`), which is below every size-21 draw. More healthy fit runs
  scored worse. The mechanism is unknown and was not investigated.

## 5. D-7: what the ESN reads on llama
- Step latency includes tool latency. Step latency ≥ tool latency in 1.000 of llama tool steps, 0.842 of
  gemini's and 0.825 of qwen's. The step text contains the tool result in 1.000 of steps for all three
  families (co `d7.a_evidence.<fam>.share_step_latency_ge_tool_latency`, `.share_step_text_contains_tool_result`).
  So the m channel's latency mixes environment time with model time.
- Result-blind (D-7c: every tool result replaced by same-length filler, in all runs): ESN_i@llama goes
  0.846 → 0.846, diff exactly 0 (co `d7.c_result_blind.ESN_i@llama.headline`). Without u it goes
  0.905 → 0.906 (+0.0004 [−0.0006, 0.0047]) (at `esn.ESN_i_no_u@llama.auroc.headline`). The transform does
  remove content: TEXT-raw@llama goes 0.656 → 0.562 (co `d7.c_result_blind.TEXT-raw@llama.headline`).
- Attribution: we count which channel holds the running max of a failed llama headline run's score.
  For ESN_i, m holds it in 79 of 92 runs at t=3 and in 76 of 79 at t=4. u holds it in the rest. Neither e
  nor x holds it in any run, and the counts are identical after blinding (derived sums of at
  `esn.ESN_i@llama.attribution.{before,after_blind}.<t>.<class>`). Without u, m holds it in 91 of 92 at t=3
  and 79 of 79 at t=4 (at `esn.ESN_i_no_u@llama.attribution`). So m and u set the score.
- No single m input explains this (at `m_inputs.contrast.<input>.<at_t|max_to_t>`). For context_corruption,
  latency at t scores 0.720. For looping, errors so far score 0.792. goal_drift runs have lower output
  tokens (0.298) and fewer errors (0.292).
- Masking step τ (inputs at τ carried from τ−1) costs about 0.08. ESN_i@llama goes 0.846 → 0.761, a change
  of −0.085 [−0.147, −0.014], and context_corruption goes 0.978 → 0.831 (co
  `d7.b_masking.ESN_i@llama:carry`). MS-raw goes 0.858 → 0.781, −0.077 [−0.116, 0.004] (co
  `d7.b_masking.MS-raw@llama:carry.headline`).
- This corrects the Step 3b reading ("the ESN reads step-output content"), which was wrong for llama.

## 6. Gemini: exploratory, a pre-registered hypothesis for SP8, not a claim
Gemini is not gated. It has 22/21 positives, eval-only under D6, and no goal_drift. On gemini, TL-pct scores
0.716 vs ESN_i 0.481, a paired difference of +0.236 [0.075, 0.390] (ab
`families.gemini.paired.TL-pct_vs_ESN_best.mean`). ALL-pct scores 0.780 and ENV-pct 0.786. The ranking does
not turn into alarms: **TL-pct's TPR@5%FPR on gemini is 0 at t=3 and at t=4** (ab
`families.gemini.arms.TL-pct.headline.tpr_at_fpr`). Gemini's ESN score is mostly content shift. Its e channel
holds the running max in every gemini run (at `esn.ESN_i@gemini.attribution`). Blinding moves it by
+0.096 [−0.108, 0.281] (co `d7.c_result_blind.ESN_i@gemini.headline`).

Hypothesis for SP8, stated before any SP8 data is seen: on a held-out family whose ESN signal is
content-driven, TL-pct ranks failures above ESN_best, but it does not alarm at 5% FPR until its threshold is
recalibrated (SP6). Both halves can come out wrong. The SP8 spec fixes the test and threshold before the run.

## 7. Limits
- One gated family (llama, 193 runs, 23 task groups among healthy runs). Failures are injected.
- ESN_best was picked on the test result (conservative toward us). D-6 shows that this pick sits on one draw
  of a ±0.1 distribution, which is not part of the gate.
- MS-pct falls below chance on llama context_corruption (0.361 vs MS-raw 0.948). This was not investigated.
- The m signal behind context_corruption may be a size artefact: corrupted results are larger, and the
  blinded filler kept their size. This is untested.
