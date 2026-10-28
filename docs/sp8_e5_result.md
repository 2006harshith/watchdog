# SP8 / E5: on organic llama8b failures, the frozen source monitors are at chance

**The gate fails.** On organic_llama8b_cold, MS-raw's AUROC is **0.510 [0.355, 0.674]**, and the lower
bound does not clear c = 0.5. MS-raw is the frozen SP5 model and was the best scorer on injected failures
(E2: 0.858). None of the frozen monitors separates the 10 healthy runs from the 110 failed ones better
than chance. Removing latency raises MS-raw's score (O-3), so latency is not what MS-raw has left on
these runs. The gate was pre-registered as low-powered: with 10 healthy runs it could only pass at a
true AUROC of about 0.65 or more.

Files: r = `results/e5_organic.json` (clean 4eacb51), fp = `results/e5_nolat_fingerprint.json` (clean
138c75d). Pre-registration: `configs/e5.yaml`, committed at **51c7723**, before any SP8 monitor score and
before `experiments/e5.py` existed. Data and label: `src/watchdog_agent/organic.py`, and
`docs/data_card.md` (SP8 section). Decisions: `docs/decisions.md`, 2026-10-28 (SP8 entries). Every number
below is followed by its file and key.

**Estimand (every table).**
- The label is run-level: HEALTHY iff the run's last synthesis step states the correct grand total
  (strict rule).
- A failed run counts as a positive at every checkpoint it reaches (τ = 0). So the AUROC at t estimates
  "will this run end wrong", read from the prefix up to t.
- This is **not** E2's estimand. E2's failures were injected, with a known onset τ, and a run counted as
  positive only from τ on.
- AUROC = the mean of the matched-step AUROCs at t = 3 and 4.
- The CI is a run-level cluster bootstrap (each run is its own task group), B = 2000, seed 0.
- The ESN-src spread over fit draws is fit-set variability only.

## 1. Reproduction: exact (before any SP8 statistic)
From r `checks`:
- The frozen HGB arms (MS-raw, TL-raw, TL-pct), refitted from source, give E2's llama prefix scores to
  1.1e-16 (floating-point rounding).
- MS-raw-noLat, refitted, reproduces its committed fingerprint (`34838f50…`, fp).
- ESN-src's 30 fit draws reproduce D-6's AUROCs exactly (difference 0.0). The ESN channels are e, u, m, x,
  with logprobs on 100% of organic steps.

## 2. Gate: FAIL
"On organic_llama8b_cold, MS-raw's mean matched-step AUROC over t = 3, 4 (run-level labels) has a
run-level cluster-bootstrap 95% CI (B = 2000, seed 0) whose lower bound is > 0.5."

**0.510 [0.355, 0.674] → FAIL** (r `gate`). The negative result is written up, as pre-registered. SP9's
scope is for the chat Project.

## 3. Arms on the gate corpus: organic_llama8b_cold (10 healthy, 110 failed)
From r `per_corpus.organic_llama8b_cold`.

| arm | AUROC [95% CI] | t = 3 / t = 4 |
|---|---|---|
| **MS-raw** (gate) | **0.510 [0.355, 0.674]** | 0.490 / 0.531 |
| MS-raw-noLat (pre-registered sensitivity) | 0.615 [0.468, 0.756] | 0.560 / 0.669 |
| TL-pct (reference cross-fitted on the corpus's 10 healthy runs) | 0.526 [0.437, 0.607] | 0.505 / 0.548 |
| TL-raw | 0.498 [0.486, 0.510] | 0.505 / 0.491 |
| ESN-src, median of 30 fit draws (range 0.405–0.620) | 0.451; the median draw's CI is [0.280, 0.655] | — |

Every interval contains 0.5. TL-raw's very narrow interval says its scores barely vary across these runs:
it ranks almost nothing.

## 4. Pre-registered expectations
- **O-1 HOLDS.** MS-raw's organic AUROC (0.510) is below its synthetic llama AUROC (E2: 0.858)
  (r `O-1`). The two numbers have different estimands: run-level and organic here, onset-based and
  injected in E2. The comparison says only that the injected-failure result does not carry over.
- **O-2 dropped.** Only 11 healthy runs exist in total, so E3's n = 20 operating point cannot be run.
  There is therefore no FA or TPR in this write-up.
- **O-3 HOLDS, in the opposite direction from the concern.** MS-raw-noLat is 0.615, **+0.104 above**
  MS-raw (r `O-3`). Latency is not carrying MS-raw's organic score; removing it helps. One reading, not
  tested here: MS-raw learned latency on source, and the latency regime differs between source and these
  runs (§6).
- **O-4 untestable.** Its premise is a held-out family whose ESN signal is content-driven. These runs are
  llama, and SP5's close-out traced llama's ESN signal to the m channel. The hypothesis is quoted
  verbatim in `configs/e5.yaml`.

## 5. The T = 0.9 corpus: organic_llama8b (descriptive only; 1 healthy, 119 failed)
From r `per_corpus.organic_llama8b`: MS-raw 0.055, MS-raw-noLat 0.311, TL-raw 0.401, ESN-src median
0.464 (0.391–0.630).
- TL-pct is undefined: one fold's percentile reference has no healthy run.
- Every CI is undefined: 733 of 2000 resamples contain no healthy run.

With one healthy run, an AUROC is just the share of failed runs scored above that run. These numbers
describe one run and are not evidence. The corpus was excluded from the gate because the label is
confounded with it (1/120 vs 10/120 healthy).

## 6. The latency shift seen by the frozen MS arm
From r `latency_shift`: step latency and output tokens, q10 / q50 / q90.

| runs | latency (s) | share of steps < 50 ms | output tokens |
|---|---|---|---|
| organic cold, healthy (10) | 2.69 / 2.83 / 4.90 | 0% | 18 / 24 / 45 |
| organic cold, failed (110) | 2.71 / 2.83 / 4.40 | 0% | 18 / 24 / 46 |
| organic T = 0.9, all (120) | 2.70 / 2.88 / 7.24 | 0% | 23 / 24 / 255 |
| llama source, healthy (77) | 0.0003 / 0.0018 / 10.19 | 57% | 18 / 42 / 120 |
| llama source, failed (116) | 0.0003 / 0.0006 / 3.11 | 53% | 18 / 27 / 120 |
| qwen source (MS-raw's training data), healthy (900) | 0.34 / 2.55 / 5.62 | 5% | 21 / 25 / 83 |
| qwen source, failed (753) | 0.25 / 2.68 / 2.94 | 8% | 21 / 25 / 66 |

On the organic cold runs, healthy and failed runs have the same latency and output-token profile, so
neither carries signal there. Most of the llama source steps are faster than 50 ms, which cannot be
generation time. That is unexplained, and it means E2's llama latencies came from a different regime
than these.

## 7. Allowed and not allowed
Allowed:
- On organic_llama8b_cold, with the run-level strict label, no frozen source monitor separates healthy
  from failed runs better than chance (every 95% CI contains 0.5).
- The pre-registered gate fails.
- Dropping MS-raw's latency features raises its AUROC on these runs (0.615 vs 0.510).

Not allowed:
- Organic and synthetic AUROCs in one comparison without the estimand note.
- Any claim at unequal FA (no operating point was run).
- Generalising beyond llama3.1:8b, these two corpora of the booking environment, and the dataset's own
  collection.
- Reading the T = 0.9 corpus's numbers as evidence.
- "The monitors cannot detect organic failures" in general: this is one environment and 10 healthy runs.

## 8. Limits
- **10 healthy runs.** The gate was low-powered by design. A true AUROC of about 0.6 would most likely
  have failed it too.
- **The label is confounded with the corpus,** hence the cold-corpus gate. The strict rule is literal:
  any number in the final answer that matches the total counts as healthy.
- **The estimand is run-level.** Many runs fail only at the final arithmetic, so the prefix at t = 3, 4 may
  look the same as a healthy run's. That caps any monitor read early.
- **Not unseen tasks, and not this machine.** It is the same booking environment as the source pool, and
  the runs come from the dataset's own collection. The SP8 spec asked for runs collected here: the
  AgentDojo attempt (llama3.1:8b via Ollama, both drivers) scored 0/20 on its set-aside pilot tasks, with
  1–2 LLM calls per run, and was dropped (decisions.md 2026-10-28).
- **The old step schema.** Tool events were parsed from the step text (5 of 2,457 tool steps did not
  parse), and tool latency is absent, so ENV features are NaN.
- **Teach-backs.** The SP8 label/estimand teach-back is NOT ANSWERED (owner override), as are those for
  SP6 and SP7.
