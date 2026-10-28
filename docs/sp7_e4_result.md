# SP7 / E4: on held-out llama, the cheap monitor out-ranks a small local LLM judge

SP7 has **no gate**. The power check showed that a non-inferiority margin this sample could resolve
would be 0.14–0.20 AUROC (§6), so the comparison is reported as an estimate. On held-out llama3.1:8b,
the source-fitted ESN monitor (ESN-src, median of its 30 fit draws) ranks failed runs above healthy ones
better than J-local, a 3.8B judge run locally:

- **D = +0.132 [+0.045, +0.206]** in mean matched-step AUROC.
- **J-1 FAILS.** I expected the judge to win.
- **J-2 HOLDS.** At E3's operating point the judge keeps its false-alarm rate under 0.12. At that same
  rate it catches 30% of failed runs, where ESN-src catches 50% and MS-raw 73%.

This is one small local judge, not "LLM judges".

Files: r = `results/e4_judges.json` (clean f86e0d4), pd = `results/e4_prompt_dev.json` (v1, clean 3dc1d39),
pw = `results/e4_step0_power.json` (clean d175ba2). Config: `configs/e4.yaml`, frozen at **15b31e8, before
the first llama judge call**. Judge responses: `results/e4_judge_cache.jsonl` (1618056). A rerun reads it
and makes no call. Decisions: `docs/decisions.md`, 2026-10-28 (SP7 entries). Every number below is followed
by its file and key.

**Every table:**
- llama only.
- AUROC is the E2 metric: the mean of the matched-step AUROCs at t = 3 and 4, headline classes (looping,
  goal_drift, context_corruption) vs all healthy runs.
- Test set: 92 positive / 77 healthy runs at t = 3 (25 task groups), 78 / 60 at t = 4 (17 task groups),
  after the one dropped pair (§5).
- Spread over ESN-src fit draws or E3 calibration draws is fit-set or calibration variability only.
- CIs are task-group cluster bootstraps (test-side only).

## 1. Reproduction: exact
Every monitor number is recomputed by the code that produced it before, and is checked against the
committed earlier result (r `checks`; all differences 0.0):
- HGB arms (TL-raw, TL-pct, MS-raw), from `results/e2_prefix_scores.csv`: headline AUROC equals E2's.
- ESN-src's 30 fit draws, refitted with E3's `_esn_src`: per-draw AUROC equals D-6's
  (`results/e2_closeout.json`) and E3's.
- The J-2 operating point rerun for MS-raw and ESN-src: every cell equals E3's at n = 20.

## 2. D: ESN-src minus the judge
D is the median over ESN-src's 30 fit draws of (AUROC_draw − AUROC_J-local). Its CI is a paired task-group
bootstrap with B = 2000 and seed 0, where each replicate recomputes all 30 draws and the judge.

| | value |
|---|---|
| D | **+0.132 [+0.045, +0.206]** (r `D`) |
| share of fit draws > 0 | 0.80 (24 of 30); per-draw range −0.073 to +0.221 (r `D.per_draw_diff`) |
| J-local headline AUROC | 0.636 [0.555, 0.744] (r `auroc.J-local`) |
| ESN-src headline AUROC, median of 30 draws | 0.768, range 0.566–0.857 (r `auroc.ESN-src`) |
| Sensitivity: missing score set to 0, nothing dropped | judge 0.632; D +0.138 [+0.045, +0.213] (r `sensitivity_missing_as_0`) |

The CI excludes 0. With no pre-registered margin, that is an estimate of the gap on these 25 task groups,
not a test. It covers test-side sampling only: the 30 fit draws are held fixed, and the fit-draw spread is
the per-draw range above.

## 3. J-1 (pre-registered): FAILS
"The best judge's headline AUROC > ESN-src's median AUROC. I expect this to HOLD."
0.636 vs 0.768: **FAILS** (r `J-1`). With one judge, "best judge" is J-local. The expectation is reported
unedited.

## 4. J-2 (pre-registered): HOLDS
"At conformal α=0.10, n=20, on E3's folds and draws, each judge's pooled mean FA ≤ 0.12."

Setup: E3's 5 task-group folds and 200 calibration draws, m = n = 20 healthy target runs, an alarm when
the running max over {2..t} exceeds θ, and α_eff(20) = 0.095. Sources: r `J-2.judge_headline` and
`J-2.context_at_same_point`. MS-raw and ESN-src were rerun here, and ESN-refit is reused from E3.

| arm | mean FA [p10, p90] | P(FA > 2α) | median TPR [p10, p90] | first-crossing TPR | calibration scores tied at θ |
|---|---|---|---|---|---|
| **J-local** | **0.097** [0.082, 0.132] | 0.00 | 0.296 [0.274, 0.356] | 0.296 | 0.22 |
| ESN-src | 0.105 [0.078, 0.132] | 0.00 | 0.502 [0.098, 0.611] | 0.502 | 0.08 |
| MS-raw (post hoc) | 0.088 [0.066, 0.117] | 0.00 | 0.729 [0.691, 0.779] | 0.721 | 0.05 |
| ESN-refit (E3) | 0.295 [0.199, 0.444] | 0.90 | 0.602 [0.414, 0.801] | not run | not run |

- **J-2 HOLDS**: 0.097 ≤ 0.12.
- The judge, ESN-src and MS-raw sit at nearly the same FA (0.088–0.105), so their TPRs are compared
  with both FAs stated: at about 0.1 FA the judge catches 30% of headline failures, ESN-src 50% and
  MS-raw 73%.
- ESN-refit buys its TPR at an FA of 0.295, so it is not comparable here.
- The judge's ties: 22% of its calibration scores sit exactly at θ. With the strict ">" rule they never
  alarm, which pulls its FA below α_eff, as the spec warned.
- First-crossing TPR equals any-alarm TPR for the judge, so its alarms are not pre-onset (D-13c
  definitions).

## 5. Per class (AUROC, vs all healthy)
r `auroc.<arm>.per_class`; for ESN-src, the median over the 30 draws.

| class | J-local | ESN-src | MS-raw | TL-pct | TL-raw |
|---|---|---|---|---|---|
| looping | 0.680 | 0.710 | 0.808 | 0.861 | 0.912 |
| goal_drift | 0.626 | 0.705 | 0.813 | 0.492 | 0.461 |
| context_corruption | 0.603 | 0.883 | 0.946 | 0.598 | 0.557 |

The judge is below ESN-src on every class. Its gap is largest on context_corruption, where a corrupted
tool result reads like any other tool result.

**The missing score.** One of 479 scoring replies did not parse: `real-context_corruption-039` at t = 4
got prose with no number (r `judge.parse_failures` = 1, rate 0.21%, under the 2% stop). Handling a
missing score was not pre-registered, so the run stopped. Post hoc, the owner chose to drop that
(run, t = 4) pair for every scorer (decisions.md 2026-10-28). The worst-case-0 sensitivity in §2 does not
move the result.

## 6. Cost and latency
These are measured properties, reported, not findings. Machine: Windows 10, GTX 1650 4 GB, Ollama
0.40.12. Measured 2026-10-28 (r `cost_latency`).

| | value |
|---|---|
| J-local, per call | wall-clock p50 3.03 s, p95 3.51 s; 334 tokens in, 2 out; model load up to 15.7 s |
| J-local, per run | 2.8 calls (one per scored checkpoint); $0, local GPU, no per-token price |
| ESN-src | 1.57 ms CPU per step |
| HGB arms | 1.80 ms CPU per step for features, plus 0.04 ms to predict (MS-raw) |

The judge numbers are wall-clock on a GPU; the monitor numbers are CPU process time. The two are
different clocks and are not compared as one ratio.

## 7. The judge, and how it was chosen
- **Prompt (v1).** Developed on 60 qwen source runs only; no llama trace was read before the freeze.
  0/240 parse failures, qwen headline AUROC 0.738 (pd `versions.v1`). The pre-set rule allowed a revision
  only for parse failures above 2% or off-format output, so v1 was frozen without revision.
- **Failure-mode line.** The prompt names the three headline failure modes in plain words
  (owner-approved).
- **Decoding.** phi4-mini answers "10", a blank line, then an explanation, despite "output only the
  integer". Decoding therefore stops at the first newline (decided on qwen, before the freeze). Settings:
  temperature 0, seed 0, `num_ctx` 4096.
- **Determinism.** 29/30 repeated qwen prompts and 26/30 repeated llama prompts gave identical text, even
  at temperature 0 with a fixed seed (r `judge.determinism`). The judge is not fully deterministic on this
  GPU.
- **Coarse scale.** 311 of 539 llama replies are `10` and 94 are `20` (r `judge.reply_counts`).
- **Why a local judge.** The spec's hosted judge (claude-haiku-5-5) was dropped: the owner declined to buy
  API credit, and Claude models reject `temperature` from API version 2026-10-21. So the judge is a local
  model, and no trace left the machine.
- **Licences.** Ollama and Phi-4-mini are both MIT.

## 8. Allowed and not allowed
Allowed:
- On held-out llama, ESN-src's median fit draw out-ranks this judge by +0.132 [+0.045, +0.206] AUROC,
  and 80% of its fit draws are above it.
- J-1 fails, against the expectation.
- At E3's operating point, the judge holds conformal FA (0.097). At about the same FA, ESN-src's TPR is
  higher (0.502 vs 0.296, FAs 0.105 vs 0.097).

Not allowed:
- Any claim about LLM judges in general. This is one 3.8B local judge on a 4 GB GPU; a larger hosted
  judge was not run.
- Any claim at unequal FA without both FAs.
- "Cheaper" as a finding. The cost table is a measured property.
- Extending the result to organic failures (SP8).
- Reading D as a non-inferiority or superiority test (no margin was pre-registered).

## 9. Limits
- **One small judge.** phi4-mini (3.8B; its quantisation is not stated on the Ollama page), with a coarse output scale and mild GPU
  nondeterminism. A stronger judge could reverse J-1.
- **Sample.** 25 task groups (17 with a step 4). Fold 1's 23-run booking group dominates, as in SP6.
- **Prompt.** Developed on qwen only; whether a llama-tuned prompt would score higher is untested, by
  design.
- **One post-hoc rule** for one missing score (§5), with a sensitivity that does not move the result.
- **The CI is test-side only.** D's CI holds the 30 fit draws fixed; the fit-draw spread is reported
  separately (−0.073 to +0.221).
- **MS-raw** is the best scorer on every metric. It is labelled post hoc, and its signal is still not
  understood (SP5, SP6).
- **Teach-backs.** The SP6 threshold-logic teach-back and the SP7 teach-back on D are NOT ANSWERED
  (owner overrides).
