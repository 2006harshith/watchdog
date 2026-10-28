# SP6 / E3: a few healthy runs do not certify a false-alarm rate on unseen llama tasks

E3 asked whether a monitor can move to a new model (llama3.1:8b) and get its alarm threshold from n
healthy runs of that model. Conformal α = 0.10 should keep mean false alarms at or below about 0.12. It
does for the arms that spend no target runs on anything but the threshold (MS-raw, ESN-src at n ≤ 20). It
does not for the arms that also spend target runs on a percentile reference or a refit (TL-pct, ALL-pct,
ESN-refit). Even where the pooled rate looks nominal, it hides fold errors that cancel. TL-pct's apparent
TPR lead over ESN-src comes from alarms that start before the failure; on a definition that means the same
for every arm, it is gone.

Files: rc = `results/e3_recalibration.json` and dr = `results/e3_draws.csv` (E3, clean 3ce8921;
reproduced exactly by the clean 0d3fc92 rerun, bad72c0), dg = `results/e3_diagnostics.json` (D-8..D-13,
clean 5042a7a), d14 = `results/e3_d14.json` (D-14, clean 59d8119), s0 = `results/e3_step0_counts.json`.
Designs: `docs/decisions.md` 2026-10-26 (E-1..E-4, pre-registered), 2026-10-27 (D-8..D-12), 2026-10-28
(D-13, D-14). **E-1..E-4 are pre-registered. Everything labelled D-8..D-14 is post hoc**, designed and
committed before its own run but after E3's headline was seen. The pre-registered verdicts and every D
outcome (including D-8c, D-9, D-10a, D-10b, D-10c, D-11a and D-14 FAILS) are reported unedited.

**Every table, unless it says otherwise:**
- llama, conformal α = 0.10, headline = the mean of t = 3 and t = 4 per draw.
- The spread over draws is calibration-set variability only. Test-set sampling is not in it.
- "CI" is D-12's task-group cluster bootstrap over test groups (B = 2000), with calibration draws held fixed.
- Healthy test runs: 77 at t = 3, 60 at t = 4. At n = 40 there are only 29 at t = 4: fold 1 is skipped,
  because it has 29 eligible runs and holds 31 of the 60 (s0 `families.llama.per_t.4`).
- m is the number of calibration runs. α_eff(m) = 1 − ⌈(m+1)(1−α)⌉/(m+1), the exact conformal FA under
  exchangeability, and 0 when θ = +∞.

## 1. Pre-registered expectations

**E-1 FAILS: 9 of 15 cells, 5 of them grossly** (mean FA > 2α = 0.20) (rc `expectations[E-1].cells`). The
gross cells are TL-pct n=20 (0.292), ALL-pct n=20/40 (0.396 / 0.291) and ESN-refit n=20/40 (0.295 / 0.328).
The four cells within 0.12–0.14 are ESN-src n=40 (0.122), TL-pct n=40 (0.128) and TL-raw n=10/40
(0.132 / 0.131).
Limit (D-12): the 0.02 slack in E-1 covers Monte Carlo error over calibration draws only. Test-side
intervals are wide. Every failing cell's CI contains 0.12 except ESN-refit n=40 [0.168, 0.477], and passing
cells reach above 0.12 too (MS-raw n=20 [0.039, 0.176]) (dg `families.llama.d12.stats.<arm>.n<n>.conformal.fa_mean`).
So E-1's pass/fail is a statement about these 25 task groups, not a rate estimated tightly enough to
generalise.

**E-2 HOLDS as pre-registered, but the lead is not like for like.** I expected E-2 to fail. TL-pct − ESN-src
median paired TPR at n=20 is +0.115, with 81.5% of draws > 0 (rc `expectations[E-2].comparison`) and
CI [−0.220, 0.284] (dg `…d12.stats.E-2.TL-pct_minus_ESN-src.n20.tpr.diff_median`). Post-onset TPR gives
+0.081 [−0.220, 0.232] (dg `…d11.e2_restated.tpr_post`, `…d12.stats…tpr_post.diff_median`). Neither is a
fair comparison:
- TL-pct's FA is 0.292 and ESN-src's is 0.105, so the two TPRs sit at unequal false-alarm rates.
- The post-onset definition is not the same quantity for both arms (D-13c, §5).

On first-crossing TPR, which counts a positive only if its first alarm falls at a step in τ..t and means
the same for every arm, the difference is **−0.012 [−0.220, 0.069]**, with 46% of draws > 0
(dg `…d13c.e2_tpr_first`). TL-pct's lead comes from alarms that begin before the failure.

**E-3 FAILS: ESN-refit's TPR median exceeds ESN-src's, but the margin is fragile and bought with FA.**
- The pre-registered paired reading over 30 draws gives +0.320 [0.114, 0.369] (rc `expectations[E-3]`;
  dg `…E-3…paired_diff_median`). Its 30 ESN-src draws have a median of 0.274, which sits at the 2.4th
  percentile of medians over random 30-of-200 subsets (dg `…d11.e3_restated.tpr.src_paired_median_among_subsets`).
- Against ESN-src's 200-draw median (0.502), the margin is **+0.100 [−0.083, 0.176]**
  (dg `…E-3…refit30_minus_src_all`).
- ESN-refit runs at FA 0.295 against ESN-src's 0.105.

**E-4 HOLDS.** Plug-in at n=10 gives mean FA > α for every arm, 0.158–0.476 (rc `expectations[E-4]`; table 2).

## 2. FA and TPR by arm and budget
Columns: FA = mean FA over draws; P(FA>2α) is over draws; TPR = median any-alarm TPR over draws; first-cross =
median first-crossing TPR. Sources: rc `families.llama.cells.<arm>.<n>.conformal.0.1.headline`, dg
`…d10_cells`, dg `…d12.stats`, dg `…d13c.cells`. R = 200 draws, ESN-refit 30.

| arm | n | m | α_eff(m) | FA [CI] | P(FA>2α) | TPR | first-cross | E-1 |
|---|---|---|---|---|---|---|---|---|
| TL-raw | 10 | 10 | 0.091 | 0.132 [0.000, 0.248] | 0.355 | 0.225 | 0.225 | fail |
| TL-raw | 20 | 20 | 0.095 | 0.114 [0.000, 0.241] | 0.265 | 0.225 | 0.225 | hold |
| TL-raw | 40 | 40 | 0.098 | 0.131 [0.000, 0.305] | 0.085 | 0.297 | 0.297 | fail |
| MS-raw | 10 | 10 | 0.091 | 0.090 [0.047, 0.168] | 0.010 | 0.661 | 0.657 | hold |
| MS-raw | 20 | 20 | 0.095 | 0.088 [0.039, 0.176] | 0.000 | 0.729 | 0.721 | hold |
| MS-raw | 40 | 40 | 0.098 | 0.108 [0.044, 0.192] | 0.000 | 0.710 | 0.710 | hold |
| ESN-src | 10 | 10 | 0.091 | 0.097 [0.052, 0.182] | 0.005 | 0.488 | 0.488 | hold |
| ESN-src | 20 | 20 | 0.095 | 0.105 [0.056, 0.194] | 0.000 | 0.502 | 0.502 | hold |
| ESN-src | 40 | 40 | 0.098 | 0.122 [0.048, 0.230] | 0.010 | 0.506 | 0.506 | fail |
| TL-pct | 20 | 10 | 0.091 | 0.292 [0.001, 0.547] | 0.920 | 0.595 | 0.471 | fail |
| TL-pct | 40 | 20 | 0.095 | 0.128 [0.000, 0.246] | 0.045 | 0.476 | 0.422 | fail |
| ALL-pct | 20 | 10 | 0.091 | 0.396 [0.059, 0.692] | 1.000 | 0.704 | 0.429 | fail |
| ALL-pct | 40 | 20 | 0.095 | 0.291 [0.100, 0.449] | 0.940 | 0.525 | 0.399 | fail |
| ESN-refit | 20 | 10 | 0.091 | 0.295 [0.101, 0.440] | 0.900 | 0.602 | 0.599 | fail |
| ESN-refit | 40 | 20 | 0.095 | 0.328 [0.168, 0.477] | 0.967 | 0.712 | 0.712 | fail |

- At n=10, TL-pct, ALL-pct and ESN-refit have m = 5. Conformal gives θ = +∞, so they raise no alarms
  (FA 0, TPR 0), and E-1 excludes them, as pre-registered.
- n = 0 (source thresholds, no target data) fails on llama:
  - MS-raw: FA 0.349 [0.266, 0.467].
  - TL-raw: FA 0.487 [0.101, 0.755].
  - ESN-src: m = 7, so conformal θ = +∞ (FA 0, TPR 0). Plug-in gives FA 0.862.
- Recalibration at n ≥ 10 brings MS-raw and ESN-src (n ≤ 20) under 0.12. It does not do this for TL-raw.
- `early_alarm_rate` = alarmed positives whose first alarm comes before τ ÷ all alarmed positives (D-13c).
  The denominator is not all positives, and draws with no alarmed positive at some t drop out. Its median
  at n=20 is 0.204 for TL-pct, 0.385 for ALL-pct, 0.007 for MS-raw and 0.0 for both ESN arms (rc `…headline.early_alarm_rate`).

Table 2. Plug-in at n=10 (E-4). The benchmark is the **uniform-score benchmark**: E[FA] of
np.quantile(linear) under iid Uniform scores (D-10). It is exact only for Uniform scores, and is off by up
to 0.025 for Normal or Exponential scores (dg `d10_simulation`).

| arm | m | mean FA | uniform-score benchmark |
|---|---|---|---|
| TL-raw / MS-raw / ESN-src | 10 | 0.256 / 0.166 / 0.158 | 0.173 |
| TL-pct / ALL-pct / ESN-refit | 5 | 0.380 / 0.476 / 0.385 | 0.233 |

Plug-in exceeds α at small m even under exchangeability. The split arms exceed even the benchmark.

## 3. Where the excess sits (D-8, D-9, D-13b)
- **D-8, common test set:** the same test runs at every n, which means dropping fold 1 at t=4.
  - On that set, TL-pct n=20 falls from 0.292 to 0.120, ALL-pct from 0.396 to 0.211, and ESN-refit from
    0.295 to 0.248.
  - MS-raw rises from 0.088 to 0.114 at n=20, more than the 0.02 expected, so D-8c FAILS. ESN-src goes
    from 0.105 to 0.120 (dg `…d8.rows`).
  - On the common set, the split arms' FA at n=40 is not below their FA at n=20 (D-8d).
- **No n-curve is read across n=40 without D-8.**
- **D-9, fold decomposition** (dg `…d9.rows`; D-9 FAILS as pre-registered, because not every cell is
  concentrated):
  - TL-pct at n=20 and n=40, and ALL-pct at n=20: the excess is concentrated in fold 1. Fold-1 FA is
    0.48–0.80, and the other folds pooled are at 0.025–0.08.
  - ALL-pct n=40 and ESN-refit at n=20 and n=40: the excess is spread. Fold 1 is high, and the other folds
    are at 0.13–0.28.
- **D-13b** (counts only): fold 1 holds 32 of the 77 healthy test runs at t=3 and 31 of the 60 at t=4. Of
  those, 23 come from one booking task (the item-1/3/6 total with 8% tax). Its other groups have 3, 3, 2 and
  1 runs (dg `…d13b.per_t`).
- Because fold 1's 23-run group is the test fold there, it is never in that fold's calibration pool. In
  every other fold it is in the pool.

## 4. Arms that do not split target runs: near nominal pooled, offsetting per fold (D-13a)
Per-fold FA, pooled over draws (dg `…d13a.rows`). Per-fold healthy test runs: 8–32. The reading rule is fixed
in decisions.md 2026-10-28.

| arm | pooled FA, t=3 (n=10 / 20 / 40) | fold 3 (13 runs at t=3, 8 at t=4) | fold 1 (32) / fold 4 (12) | reading |
|---|---|---|---|---|
| MS-raw | 0.099 / 0.104 / 0.110 | 0.36–0.44 (t=3) | 0.02–0.05 / ≈0.00 | OFFSETTING in 5 of 6 (n, t) |
| ESN-src | 0.088 / 0.096 / 0.107 | 0.19–0.22 (t=3), 0.42–0.45 (t=4) | 0.05–0.09 / 0.05–0.12 | NOMINAL at t=3 for n ≤ 20; OFFSETTING at t=4; HIGH at n=40 t=4 |
| TL-raw | 0.156 / 0.143 / 0.090 | 0.385 (t=3), 0.625 (t=4) | 0.00–0.22 / 0.000 | HIGH at n=10/20 t=3 and n=40 t=4; otherwise OFFSETTING |

D-13a HOLDS as expected. Pooled over folds, MS-raw and ESN-src sit near nominal, but that hides fold
errors that cancel: fold 3 alarms 2–6× too often and folds 1 and 4 too rarely. Per-fold FA carries
test-sampling noise that pooling over draws does not remove.

## 5. Post-onset vs first-crossing TPR (D-11, D-13c)
- `tpr_post` (D-11) counts a positive if its score is above θ at some step in τ..t. ESN arms score a
  running max, so once above θ they stay above it, and for them `tpr_post` = TPR by construction. This held
  in every group row (dg `…d13c.checks.esn_post_ne_alarm_entries` = 0).
- HGB arms score each checkpoint separately and can drop back below θ. Post-onset exceeded first-crossing
  in 11,334 of their (draw, task group) entries.
- So D-11a's ESN drop of 0.0 is mechanical, not a finding. D-11a FAILS; the drops were TL-pct 0.035,
  ALL-pct 0.032, MS-raw 0.001.
- **`tpr_post` is not compared across ESN and HGB arms anywhere in this write-up.**
- First-crossing TPR (`n_pos_alarm − n_pos_alarm_pre_tau` over positives) means the same for every arm, and
  table 1 reports it.

## 6. D-14: is the by-run half split the explanation? Not for TL-pct; partly for the others
The reference/fit half and the calibration half were redrawn from disjoint task groups, with m = n/2 kept
exactly. Everything else is E3's. The D-14 loop, run on E3's own draws, reproduces E3's counts exactly:
8,600 rows, 0 mismatches (d14 `code_path_check`). The two splits draw different runs, so they are compared
as distributions on the same test runs. r = FA_run − FA_group, and its CI is a task-group bootstrap with one
resample shared by both splits (d14 `comparison.cells`).

| arm | n | m | α_eff | FA by run → by group | r [CI] | reading |
|---|---|---|---|---|---|---|
| TL-pct | 20 | 10 | 0.091 | 0.292 → 0.303 | −0.010 [−0.092, 0.029] | DOES NOT FALL |
| TL-pct | 40 | 20 | 0.095 | 0.128 → 0.164 | −0.036 [−0.108, 0.013] | DOES NOT FALL |
| ALL-pct | 20 | 10 | 0.091 | 0.396 → 0.376 | +0.020 [−0.004, 0.044] | DOES NOT FALL |
| ALL-pct | 40 | 20 | 0.095 | 0.291 → 0.207 | +0.084 [0.028, 0.157] | PARTIAL |
| ESN-refit | 20 | 10 | 0.091 | 0.295 → 0.224 | +0.071 [0.043, 0.105] | PARTIAL |
| ESN-refit | 40 | 20 | 0.095 | 0.328 → 0.215 | +0.113 [0.060, 0.180] | PARTIAL |

D-14 FAILS: FALLS was expected for all three arms at n=20.
- **For TL-pct, the by-run split is not the explanation.**
- For ESN-refit, the split explains less than half the excess, and FA stays at about 0.22.
- Fold 1 keeps FA 0.48–0.80 by run and 0.46–0.73 by group for TL-pct and ALL-pct at n = 20 and 40
  (d14 `…per_fold`, `…fa_by_big_group_placement`).
- The descriptive placement table supports the same reading:
  - Where the 23-run group lands drives FA in the other folds. For TL-pct in fold 3, FA is 0.42–0.56 when
    the group is in the reference half and 0.03–0.09 when it is in the calibration half
    (d14 `…fa_by_big_group_placement`).
  - The excess follows a test task group that is new to the reference. It does not follow shared groups
    between the halves.
- Confound, stated before the run: a group half spans fewer task groups. ESN-refit's TPR falls from 0.602
  to 0.464 at n=20 under the group split. A narrower fit set may cancel part of any gain, and D-14 does
  not separate the two.

## 7. Allowed and not allowed
Allowed:
- The arms that spend no target runs on fitting or a reference (MS-raw, ESN-src) are near nominal pooled
  on the common test set: 0.108–0.122 over n = 10–40 (dg `…d8.rows`). That pooled value hides offsetting
  per-fold errors (D-13a). MS-raw's standing as the best-calibrated arm is observed, not pre-registered.
- The arms that split target runs (TL-pct, ALL-pct, ESN-refit) are not near nominal.
  - TL-pct's and ALL-pct's excess at n=20 is concentrated in fold 1 (D-9).
  - The by-run split does not explain TL-pct's excess, and explains only part of ESN-refit's and ALL-pct's
    at n=40 (D-14).
- n = 0 source thresholds fail on llama. Recalibration at n ≥ 10 fixes MS-raw and ESN-src only.
- E-2: TL-pct's any-alarm TPR lead over ESN-src does not survive first-crossing TPR: −0.012 [−0.220, 0.069].

Not allowed:
- "Recalibration restores validity."
- "TL-pct beats ESN-src."
- "ESN-refit beats ESN-src" without a comparison at equal FA.
- Any n-curve across n=40 without D-8.
- Any mechanism for ESN-refit's spread excess.
- Comparing `tpr_post` between ESN and HGB arms.
- Quoting `early_alarm_rate` without its definition (§2).

## 8. Gemini (descriptive only; nothing is tested)
Gemini healthy test runs: 77 at t=3, 66 at t=4 (rc `families.gemini.counts`). Conformal α=0.10, mean FA /
median TPR at n=20:

| arm | FA | TPR |
|---|---|---|
| TL-pct | 0.138 | 0.095 |
| ALL-pct | 0.140 | 0.284 |
| MS-raw | 0.098 | 0.071 |
| ESN-src | 0.108 | 0.000 |
| TL-raw | 0.000 | 0.558 |

- ESN-src's source fit is inverted on gemini: 28 of its 30 fit draws have AUROC < 0.5, with a median of
  0.227 [min 0.105, max 0.625] (rc `families.gemini.esn_src.headline_auroc_per_fit_draw`).
- E2's ESN_i 0.481 would sit at about the 93rd percentile of these draws, if the metric matches E2's. That
  match is unchecked.
- At n=0, ESN-src alarms on every gemini run (FA 1.0, TPR 1.0).
- Gemini TL-raw has zero variance over draws at every n: FA 0.000 and TPR 0.558 in every draw. This is
  unexplained.

## 9. Limits
- **25 task groups, and fold 1 is dominated by a few.** One booking group carries 23 of fold 1's 32
  healthy runs at t=3. Test-side intervals (D-12) are wide, and E-1 is a statement about these groups.
- **MS-raw's signal is not understood.** As in SP5, no single input explains it.
- **ESN-src's fit-draw distribution is bimodal on llama.** Its AUROCs cluster at 0.57–0.68 and 0.76–0.86
  (rc `families.llama.esn_src.headline_auroc_per_fit_draw`). ESN-src's TPR spread mixes the two modes.
- **Gemini TL-raw's zero variance is unexplained** (§8).
- **D-14 is done.** It rules out the by-run split as the explanation for TL-pct and leaves the narrower-fit
  confound open for ESN-refit.
- **The plug-in benchmark is exact for Uniform scores only.** D-10a FAILS because the pre-set 0.01
  tolerance for Normal and Exponential scores was too tight (up to 0.025). Conformal α_eff matched
  simulation exactly.
- **D-10b FAILS at α = 0.05, n = 40:** ESN-src exceeds α_eff by +0.039 (conformal) and +0.031 (plug-in),
  and MS-raw by +0.031 (conformal).
- **The threshold-logic teach-back for SP6 is NOT ANSWERED** (owner override ×3). The write-up's §1 rests
  on that logic.
