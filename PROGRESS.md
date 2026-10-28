# PROGRESS

## Current
- Save point: SP6 (E3 recalibration curve, no gate) — in progress. Step 0 2026-10-26: Decisions +
  pre-registered expectations E-1..E-4 verbatim (docs/decisions.md, below), configs/e3.yaml (12 fill-ins
  accepted), results/e3_step0_counts.json: n=40 infeasible only at llama t=4 fold 1 (29 eligible; that
  fold holds 31 of 60 healthy test runs). Step 1 2026-10-26: src/watchdog_agent/thresholds.py + 38
  tests; threshold-logic teach-back NOT ANSWERED (owner override). Step 2: experiments/e3.py +
  scripts/run_e3.py (3ce8921); results/e3_recalibration.json + e3_draws.csv from clean 3ce8921, E2
  prefix scores reproduce (max diff 1e-16). Llama, conformal α=0.10: E-1 FAILS (mean FA TL-pct n=20
  0.292, ALL-pct 0.396, ESN-refit 0.295; MS-raw and ESN-src n≤20 hold); E-2 HOLDS against the
  expectation (TL-pct − ESN-src TPR median +0.115, 81.5% of draws > 0); E-3 FAILS (ESN-refit beats
  ESN-src, +0.320 over 30 paired draws); E-4 HOLDS. Caveat: E-2/E-3 compare TPR at unequal realised
  FA (TL-pct and ESN-refit ~0.29 vs ESN-src ~0.10). Teach-back re-asked 2026-10-27, overridden again.
  Step 3 (d0cd3e9, bad72c0): per-(draw, fold, task group) alarm counts incl. post-onset
  (results/e3_group_counts.parquet, gitignored); E3 rerun from clean 0d3fc92 reproduces 711a23e
  exactly (max diff 0; group counts sum to pooled). Step 4 (cbd5108 design before run, 0d3fc92 code,
  b90cab6 results/e3_diagnostics.json from clean bad72c0), POST HOC: D-8 a/b/d HOLD, D-8c FAILS;
  D-9 FAILS (TL-pct concentrated in fold 1 at n=20/40, ALL-pct n=40 and ESN-refit spread);
  D-10 a/b/c FAIL, d HOLDS; D-11a FAILS, b/c/d HOLD; D-12 HOLDS. Close-out part 2 spec received 2026-10-28
  (Steps 6-8: D-13, D-14, docs/sp6_e3_result.md, tag sp06-e3). Threshold-logic teach-back
  overridden a third time 2026-10-28. Step 6 2026-10-28 (design 49580f7 before run, code 5042a7a,
  results 70a1eef from clean 5042a7a; D-8..D-12 unchanged exactly), POST HOC: D-13a HOLDS (no-spend
  arms' pooled FA hides offsetting fold errors: fold 3 over-alarms for MS-raw, ESN-src and TL-raw,
  folds 1/4 under; TL-raw HIGH pooled at n=10/20 t=3). D-13b: fold 1 = 32 healthy at t=3 (31 at
  t=4), 23 from one booking task group. D-13c: tpr_post = TPR by construction for ESN arms
  (running max); first-crossing TPR added; E-2 restated with it: TL-pct − ESN-src median −0.012,
  46% of draws > 0, 95% CI [−0.220, 0.069]. Step 7 2026-10-28 (design 54325dd before run, code
  59d8119, results from clean 59d8119; code path check on E3's draws exact, 8600 rows), POST HOC:
  D-14 FAILS. Half split by task group, r = FA_run − FA_group (conformal α=0.10): TL-pct n=20
  −0.010 [−0.092, 0.029], n=40 −0.036 → DOES NOT FALL (by-run split is not the explanation);
  ALL-pct n=20 +0.020 DOES NOT FALL, n=40 +0.084 [0.028, 0.157] PARTIAL; ESN-refit n=20 +0.071
  [0.043, 0.105], n=40 +0.113 [0.060, 0.180] PARTIAL (FA still 0.22). Fold 1 (the 23-run group,
  never in a pool when it is tested) keeps FA 0.49–0.73 for TL-pct/ALL-pct under both splits.
  Step 8 2026-10-28: write-up docs/sp6_e3_result.md with the owner-approved amendments (decisions.md
  2026-10-28). SP6 closed 2026-10-28 (tag sp06-e3) with the threshold-logic teach-back NOT ANSWERED
  (owner override, fourth time for SP6). Next: SP7 needs a spec from the chat Project (paste
  docs/sp6_e3_result.md and this section). Ten teach-backs owed before deploy (SP9).
- Save point: SP7 (E4 cheap monitor vs LLM judges, GATE) — Step 0 (read-only report) started
  2026-10-28 under an owner override of the SP6 teach-back. Owner decisions on the SP7 spec: judge
  sampling T=0 x 1 sample plus a determinism subset (not T=0 x 3); ESN-src fit-draw scores recomputed
  with exact reproduction (they are not on disk); best judge re-selected inside each bootstrap
  resample; power check at stated monitor-judge correlations; Step 0d covers qwen traces too.
  Step 0 report 2026-10-28 (read-only, nothing spent): no local LLM runtime or model installed;
  J-api proposal claude-haiku-5-5 at $0.10/$0.50 per MTok (platform.claude.com/docs/en/about-claude/
  pricing, read 2026-10-28), which rejects temperature (API 2026-10-21) so T=0 is impossible there;
  479 scoped llama calls per sample (healthy + headline, t=2..4), prefix p50 556 chars; power check
  (synthetic, real llama groups): 80% power needs delta 0.196 / 0.178 / 0.137 at rho 0 / 0.3 / 0.6,
  so the gate is uninformative at delta <= 0.15 unless rho >= 0.6. STOPPED for owner decision.
  Owner decisions 2026-10-28: no gate (D reported with CI; J-1, J-2 kept), one judge claude-haiku-5-5
  thinking off x3 samples, no local judge. Step 1 done 2026-10-28 (judges.py d175ba2, power JSON
  0e85d4a); SP7 teach-back on D NOT ANSWERED (override). Step 2 harness committed fb6841d (template v1
  with the failure-mode examples line, owner-approved; judge caches committed). Owner declined API
  credit: judge is now J-local = phi4-mini:3.8b via Ollama, T=0 x 1 + 30-prompt determinism subset
  (5cd0237). No third-party API, so the Llama-licence question no longer arises. BLOCKED: Ollama not
  installed (winget absent; Claude's install was blocked by the permission classifier). Next: owner
  installs Ollama and runs `ollama pull phi4-mini:3.8b`, then `uv run python scripts/run_e4_prompt_dev.py v1`.
  Done 2026-10-28: Ollama 0.40.12, phi4-mini:3.8b digest 78fad5d1; stop-at-newline decoding (3dc1d39);
  v1 on 60 qwen dev runs (0d5be1b): 0/240 parse failures, qwen headline AUROC 0.738, 29/30 repeat calls
  identical, p50 3.0 s/call; v1 frozen by the pre-set rule. FREEZE configs/e4.yaml committed 15b31e8,
  before any llama judge call. Step 3 2026-10-28 (code 637e818..f86e0d4; judge cache 1618056, 539 local
  calls; results/e4_judges.json from clean f86e0d4): every reproduction check exact (HGB vs E2, ESN-src
  vs D-6/E3, E3 operating-point cells for MS-raw/ESN-src). 1/479 judge replies unparsed (0.21%):
  post-hoc option 1 (pair dropped for every scorer; worst-case-0 sensitivity). J-local headline AUROC
  0.636 [0.555, 0.744] vs ESN-src median 0.768: D = +0.132 [+0.045, +0.206] (80% of fit draws > 0);
  J-1 FAILS (expected HOLD); J-2 HOLDS (judge mean FA 0.097, TPR median 0.296, 22% of calibration
  scores tied at theta). Step 4 2026-10-28: write-up docs/sp7_e4_result.md. Not tagged: the SP7 teach-back
  on D (and SP6's) is NOT ANSWERED. SP7 closed 2026-10-28 (tag sp07-e4) under owner override of the
  teach-back on D (second override for it). Twelve teach-backs owed before deploy (SP9). Next: SP8 needs
  a spec from the chat Project (paste docs/sp7_e4_result.md and this section).
- Save point: SP8 (E5 frozen monitors on organic AgentDojo failures, GATE). Step 0 (read-only + pilot)
  started 2026-10-28 under owner override of the SP6 and SP7 teach-backs ("override go"), which also
  approves installing AgentDojo in its own environment and pulling llama3.1:8b in Ollama.
  Step 0 report 2026-10-28: AgentDojo 0.1.35 (MIT, own uv env envs/agentdojo, 0ad34a2); 97 user tasks
  (workspace 40, travel 20, banking 16, slack 21), utility = one boolean at run end; Ollama's OpenAI
  endpoint returns logprobs (u and surprisal producible); llama3.1:8b Q4_K_M 4.92 GB, 2.36/5.63 GB in
  VRAM, ~6.7 tok/s. Llama SOURCE step latencies: 53-57% < 50 ms (not generation time; unexplained).
  Pilot (10 runs, 5 set-aside tasks, d8fe0ff): utility 0/10, 1-2 LLM calls per run, 9/10 runs with <= 1
  tool call; failure modes: stops after the first tool result and writes prose, never calls a tool,
  invalid args ("null" strings), invented IDs. No run reaches t = 3 or 4; no healthy run exists, so AUROC
  and the operating point are undefined. STOPPED for owner decision (no monitor score computed).
  Option A re-pilot 2026-10-28 (AgentDojo LocalLLM, prompt-based tools; same 10 runs; 31ae4d5): utility
  0/10 again; median 1 LLM call per run (one run 9). New failure modes: closing the call with <function>
  instead of </function> (AgentDojo's parser then sees no call and the run ends; 4/10 runs), privacy
  refusals (workspace 16, both seeds), calling an unrelated tool. Across both drivers: 0/20 successes.
  STOPPED again: SP8 as specified is not feasible with llama3.1:8b on AgentDojo here; spec decision.
  Option 3 chosen 2026-10-28 (decisions.md): organic_llama8b + organic_llama8b_cold, 240 runs, T 5-20
  (median 8), logprobs 100%, real latencies (2.7-10 s), old step schema (no tool_events / task text;
  parseable from "[tool(args) -> result]"), metadata expected_total = the correct answer. Candidate
  label (no monitor score seen): healthy iff the final synthesis text states expected_total: 11/240
  healthy (1/120 at T=0.9, 10/120 in the T=0.2 'cold' corpus): label confounded with corpus; O-2
  (n=20 healthy) infeasible. STOPPED for a decision on the label rule and the gate.
  Owner: strict rule, cold-corpus gate (c = 0.5), O-2 dropped (66d72df). Step 1: organic.py + tests +
  data card (3ccd493); label/estimand teach-back NOT ANSWERED (override). Step 2: MS-raw-noLat fitted
  from clean 138c75d (MS-raw reproduces E2 to 1.1e-16; noLat fingerprint 34838f50); configs/e5.yaml
  (gate, O-1, O-3, O-4 untestable, O-2 dropped) committed BEFORE any SP8 score and before e5.py exists.
  Next: Step 4 plan (e5.py), wait for "go".
  SP5 closed 2026-10-26 (tag sp05-e2): gate FAIL as pre-registered, write-up docs/sp5_e2_result.md;
  recalibration teach-back NOT ANSWERED (owner override, fourth time).
- SP5 history (E2 ablation, GATE). Step 0 (pre-registration, ae18348), Step 1
  (recalibration.py, 4733739), Step 2 (experiments/e2.py, 860249b) done. Step 3 run 2026-09-30 from
  clean 860249b (results/e2_ablation.json, MLflow d3a8305364a64be29c2425b4a4449871): GATE FAIL.
  llama TL-pct − ESN_best (ESN_i) = −0.197 [−0.302, −0.057]; TL-pct 0.648 [0.555, 0.768], ESN_i
  0.846 [0.770, 0.896]. Both secondary expectations hold (TL-pct 0.648 > TL-raw 0.641; ≥ MS-pct
  0.552). Not gated: gemini TL-pct 0.716 vs ESN_i 0.481, ALL-pct 0.780.
- SP5 Step 3b diagnostics 2026-09-30 (results/e2_diagnostics.json from clean a472577, identical to the dirty run; three rows reproduce
  0.846 / 0.858 / 0.646 exactly). The ESN 0.646 (SP4) -> 0.846 (E2) gap is the FIT POOL, not the
  class mix (SP4 headline-only 0.665) and not pooling or z-scoring (SP4 raw 0.661; per-fold mean
  0.602). On SP4's own folds, per-fold: E2's single model 0.838 vs SP4's fold models 0.602. Pool size
  (21 vs ~34 fit runs) and composition (E2 excludes every llama task group) are not separated.
  Surprisal is NOT the source: ESN_i without u scores 0.905 (context_corruption 1.000); MS-raw
  without surprisal 0.812, surprisal only 0.547. CORRECTED 2026-10-24 by the close-out: the Step 3b
  reading "the ESN reads step-output content (e, x); content catches context_corruption" is WRONG on
  llama. The m channel (action, latency, output length, error flag) sets the score, not e or x.
- SP5 close-out 2026-10-24 (interpretation only; the gate is final). results/e2_closeout.json (clean
  ecadcfb) and results/e2_channel_attribution.json (clean e33db31); every re-run E2 number reproduces
  exactly. D-6: ESN_i on 30 random 21-of-28 fit subsets of its own pool: median 0.770 [min 0.566, max
  0.857]; E2's split (0.846) is beaten or matched by only 3% of draws. SP4 fold pools cut to 21 fit
  runs: per-fold mean median 0.788 [0.664, 0.877]. SP4's structure with E2's pool = 0.846 (structure
  adds nothing). D-7b, step tau carried from tau-1: ESN_i 0.846 -> 0.761 (−0.085 [−0.147, −0.014]),
  context_corruption 0.978 -> 0.831; MS-raw 0.858 -> 0.781. D-7c, every run's tool results replaced
  by same-length filler: ESN_i@llama 0.846 -> 0.846 (diff exactly 0), no-u 0.905 -> 0.906,
  context_corruption 1.000 -> 1.000; ESN_i@gemini 0.481 -> 0.577 (+0.096 [−0.108, 0.281]);
  TEXT-raw@llama 0.656 -> 0.562. Attribution (t=3,4): m holds the running max in nearly every failed
  llama run for both ESN variants, before and after blinding; e holds it in every gemini run. No
  single m input explains it: context_corruption latency-at-t AUROC 0.720, looping error-so-far
  0.792, goal_drift lower output tokens (0.298) and fewer errors (0.292).
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
- SP5 write-up 2026-10-26: docs/sp5_e2_result.md (gate FAIL, arm table llama vs gemini, per class,
  D-6, D-7, gemini as an SP8 hypothesis), SP3 limit added (fit-set variance not varied in SP3),
  docs/decisions.md 2026-10-26 (baselines reported as a fit-draw distribution from SP6 on; SP6 next).
- Next concrete step (2026-10-26): re-upload PROGRESS.md to the chat Project and paste
  docs/sp5_e2_result.md plus the docs/decisions.md 2026-10-26 section; ask for the SP6 spec
  (pre-registered fail path: SP6 becomes the main contribution; baselines as a fit-draw
  distribution). Plan SP6 before code and wait for "go". Recalibration teach-back still owed (SP9).

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
| SP5 | E2 ablation — GATE (spec to be rewritten after SP3) | done — gate FAIL (pre-registered fail path); write-up docs/sp5_e2_result.md; recalibration teach-back NOT ANSWERED (owner override) | sp05-e2 |
| SP6 | E3 recalibration curve (teach-back: threshold logic) | done — write-up docs/sp6_e3_result.md; threshold-logic teach-back NOT ANSWERED (owner override x4) | sp06-e3 |
| SP7 | E4 cheap monitor vs LLM judge (gate dropped: power) | done — docs/sp7_e4_result.md: D +0.132 [+0.045, +0.206], J-1 FAILS, J-2 HOLDS; teach-back on D NOT ANSWERED (owner override x2) | sp07-e4 |
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
- (partly resolved in the SP5 close-out, D-6) SP5/3b: the ESN baseline swings 0.60 -> 0.84 with its
  healthy fit pool. Neither size nor composition explains it: 21-run subsets of E2's pool give median
  0.770, of SP4's fold pools 0.788; the spread (0.566-0.857, 0.664-0.877) is the fit-set draw. Still
  open: SP4's FULL fold pools (34-49 fit runs) score 0.602, below every one of 30 size-21 subsamples
  (min 0.664), so more healthy fit runs scored worse. Not investigated. Either way the ESN is a
  fragile baseline, which is evidence for SP6 (few healthy runs).
- (resolved in the SP5 close-out, D-7c + attribution) SP5/3b: the ESN does NOT win on llama through
  step-output content. With result content blinded in every run, ESN_i is unchanged (0.846, diff 0)
  and the no-u variant too (0.905 -> 0.906, context_corruption 1.000); the m channel holds the
  running max in nearly every failed llama run. The write-up's point becomes model-side reaction
  signals (m, u) vs TL, not a content arm.
- SP5 close-out: what in m gives context_corruption 1.000? No single m input does (best:
  latency-at-t 0.720). Candidates: a combination or the step-to-step dynamics of m. Is it a size
  artefact? Corrupted results are ~1.45x the healthy size, D-7c kept sizes, and longer input
  plausibly slows llama's steps. Untested: needs a size-changing transform or per-input ablation of m
  inside the vendored ESN. Matters for whether context_corruption is a realistic failure.
- SP5 close-out: ESN_i@gemini moves +0.096 under blinding (CI [−0.108, 0.281]): its e channel sets
  every gemini score and reads gemini's live-API result text, which is off-distribution for a qwen
  pool. Gemini's ESN number is then mostly content shift, not failure detection.
- SP5/E2: MS-pct is below chance on llama context_corruption (0.361) while MS-raw scores 0.948. The
  percentile transform may flip the MS signal's direction; not investigated.
- SP6/E3: E-1 fails for exactly the arms that split the n runs (TL-pct, ALL-pct, ESN-refit: mean FA
  ~0.29-0.40 at α=0.10), while MS-raw and ESN-src (m = n) mostly hold. Is it non-exchangeability
  across task groups, the run-level (not task-group) split into reference/calibration halves, or the
  smaller m? Not separated.
- SP6/E3: E-2 and E-3 compare TPR at unequal realised FA (TL-pct, ESN-refit ~0.29 vs ESN-src ~0.10),
  so their "wins" are partly bought with false alarms. Needs TPR at matched realised FA (or within-α
  arms only) before either is read as a monitor comparison.
- SP6/E3: ESN-src median TPR is 0.274 on the 30 draws paired with ESN-refit vs 0.502 over all 200
  draws; E-3's +0.320 gap partly reflects which 30 draws were paired.

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
- 2026-10-26 (SP5 close, full record in docs/decisions.md): from SP6 on, every baseline fitted on
  healthy runs is reported as a distribution over ≥ 30 fit-set draws (median, range, and the
  pre-registered split's percentile), because D-6 put E2's ESN split at about the 97th percentile of a
  0.566–0.857 range. SP6 is next, on the pre-registered fail path.
- 2026-10-26: Owner override of the teach-back HARD RULE and the /save rule to close SP5 with the
  recalibration teach-back NOT ANSWERED (fourth override; not owner-marked). Claude disagreed: SP6
  is threshold logic built directly on this transform, so SP6 starts on calibration logic the owner
  has not yet defended. Risk accepted. Eight teach-backs owed before deploy (SP9).
- 2026-10-26: Owner override of the teach-back HARD RULE for the SP6 threshold-logic teach-back, so
  SP6 Step 2 can start. NOT ANSWERED, not owner-marked. Claude disagreed: E3's results, and E-1's
  validity claim above all, rest on this logic. Risk accepted. Nine teach-backs owed before deploy.
- 2026-10-27: Second owner override of the SP6 threshold-logic teach-back, after Step 2's run showed
  E-1 FAILS. Claude disagreed: the E-1 failure is the mechanism Q2 asks about. Risk accepted.
- 2026-10-28: Third owner override of the SP6 threshold-logic teach-back, to start the close-out
  part 2 (Steps 6-8) ("override it, i'll do it later go"). NOT ANSWERED, not owner-marked. Claude
  disagreed: the write-up's lead (E-1, alpha_eff(m), D-12) is this logic, and tag sp06-e3 needs the
  teach-back answered under SP6's done criteria, so tagging will need a further override. Risk accepted.
- 2026-10-28: Fourth owner override of the SP6 threshold-logic teach-back, to close SP6 and tag
  sp06-e3 ("override it and tag sp06-e3"). NOT ANSWERED, not owner-marked. Claude disagreed: the tag
  marks SP6 done while the logic behind its lead (conformal alpha_eff, exchangeability, D-12) is
  undefended. Risk accepted.
- 2026-10-28: Owner override of the teach-back HARD RULE to start SP7 Step 0 with the SP6
  threshold-logic teach-back NOT ANSWERED ("override, go"). Claude disagreed (same reason). Risk accepted.
- 2026-10-28: Owner override again ("override, go") to start SP7 Step 1 (judges.py) with the SP6
  threshold-logic teach-back NOT ANSWERED. SP7's Step 1 teach-back moves from the dropped gate
  statistic to D (paired bootstrap of a median over fit draws). Risk accepted.
- 2026-10-28: Owner override of the SP7 teach-back on D ("override it, go to step 2"). NOT ANSWERED.
  Claude disagreed: D is the headline number of SP7. Risk accepted.
- 2026-10-28: Second owner override of the SP7 teach-back on D, to close SP7 and tag sp07-e4 ("override
  it and tag sp07-e4"). NOT ANSWERED. Claude disagreed: the tag marks SP7 done while its core statistic
  is undefended. Risk accepted.
- 2026-10-28: Owner override of the teach-back HARD RULE to start SP8 with the SP6 and SP7 teach-backs
  NOT ANSWERED ("override go"). Claude disagreed: SP8's gate is an AUROC cluster CI and O-2 is E3's
  threshold logic. Risk accepted.
- 2026-10-28: Owner override of the SP8 label/estimand teach-back ("override it, go to step 2"). NOT
  ANSWERED. Claude disagreed: the gate's meaning rests on the tau = 0 estimand. Risk accepted.
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

- 2026-10-26: SP6 Decisions and pre-registered expectations, verbatim from the SP6 spec (same text,
  the 12 Step 0 fill-ins and the Step 0 counts in docs/decisions.md; encoded in configs/e3.yaml):

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
  require it answered. Eight teach-backs owed before deploy (SP9). Re-asked 2026-10-24 after the
  close-out run; overridden a third time ("I'll do them at last"). Re-asked 2026-10-26 at the SP5
  write-up; overridden a fourth time to close SP5 ("override and move on to the next step i will
  answer them later"). NOT ANSWERED, not owner-marked, not graded. The owner's spec cites 4 questions
  in specs/SP5-spec.md; that file is not in the repo, so Q4 is not recorded here (paste it from the
  chat Project). Owed before deploy (SP9).

- 2026-10-26 · SP6 · threshold logic (src/watchdog_agent/thresholds.py) · Q1: why does
  k = ceil((m+1)(1-alpha)) give E[FA] <= alpha, where does the +1 come from, what does "a guarantee
  over calibration sets, not over the deployed theta" mean in practice, and what does P(FA > 2alpha)
  show that mean FA hides? Q2: E3 calibrates on other task groups and splits TL-pct's
  reference/calibration halves by run; one concrete way each breaks exchangeability and which way FA
  moves; why does plug-in overshoot alpha at m=10 even under exchangeability? Q3: TL-pct at n=10 has
  m=5: what theta does conformal give at alpha=0.10 and why is that honest; why "score > theta" and
  not ">=" (ESN example); why is a calibration score the running max up to t, not the score at t? ·
  answers: not yet given · NOT ANSWERED, owner override 2026-10-26 ("override the teach-back and go
  to step 2"). Not owner-marked, not graded. SP6 done criteria require it answered. Claude disagreed:
  every E3 number, and E-1 in particular, rests on this logic. Nine teach-backs owed before deploy
  (SP9). Re-asked 2026-10-27 at resume after Step 2 (E-1 FAILS); overridden a second time ("override
  it ill do it later"). Overridden a third time 2026-10-28 for the close-out part 2 ("override it,
  i'll do it later go"). Overridden a fourth time 2026-10-28 to close SP6 and tag sp06-e3 ("override
  it and tag sp06-e3"). Still NOT ANSWERED, not owner-marked, not graded.

- 2026-10-28 · SP7 · statistic D (paired task-group bootstrap of a median over 30 fit draws;
  evaluation._cluster_bootstrap, scripts/run_e4_step0_power.py; e4.py's D lands in Step 3) · Q1: why
  resample task groups, not runs, and why one shared resample for monitor and judge and for t=3,4;
  what each mistake does to the CI width? Q2: what uncertainty does D's CI include and leave out when
  the 30 fit draws are recomputed inside each replicate; why the median, not the mean, given ESN-src's
  bimodal fit-draw AUROCs? Q3: why picking the better of two judges on the test data biases D against
  the monitor and why re-select per replicate; with no delta, what does a CI of [-0.15, +0.05] allow
  you to say? · answers: not yet given · NOT ANSWERED, owner override 2026-10-28 ("override it, go
  to step 2"). Overridden again 2026-10-28 to tag sp07-e4. Not owner-marked, not graded.

- 2026-10-28 · SP8 · run-level label and estimand (src/watchdog_agent/organic.py) · Q1: why encode the
  run-level label as tau = 0, what an AUROC at t = 3 then estimates, and why it is not like-for-like with
  E2's 0.858? Q2: one way the strict rule mislabels in each direction, how you would notice, and why a
  programmatic rule over a human or LLM judgement? Q3: how pooling both corpora could give a high AUROC
  for the wrong reason, why the gate is cold-only, and what "task group = run" assumes? · answers: not
  yet given · NOT ANSWERED, owner override 2026-10-28 ("override it, go to step 2"). Not owner-marked.

## Session log
- 2026-10-28 · SP7 (close) · Write-up committed f3b6405; teach-back on D overridden a second time (NOT
  ANSWERED). Tag sp07-e4 · tests: pass (588, ruff clean) · next: SP8 spec from the chat Project.
- 2026-10-28 · SP7 (Steps 0-3) · No API credit: J-local phi4-mini via Ollama. Prompt v1 frozen on qwen
  (0/240 failures, AUROC 0.738); configs/e4.yaml frozen 15b31e8 before any llama call. E4 run exact on
  all reproduction checks; D +0.132 [+0.045, +0.206], J-1 FAILS, J-2 HOLDS. Bugs found and fixed on the
  way: digest prefix (stopped before any call), CPU timer resolution; f3ec448 was committed with a
  failing test (fixed in f86e0d4) · tests: pass (588, ruff clean) · next: Step 4 write-up.
- 2026-10-28 · SP6 (close) · Threshold-logic teach-back overridden a fourth time (NOT ANSWERED). Tag
  sp06-e3 · tests: pass (535, ruff clean) · next: SP7 spec from the chat Project.
- 2026-10-28 · SP6 (Step 8) · docs/sp6_e3_result.md: E-1..E-4 unedited, D-8..D-14, the amended
  allowed/not-allowed lists, gemini descriptive, limits. Amendments recorded in decisions.md. Not
  tagged (teach-back owed) · tests: pass (535, ruff clean) · next: threshold-logic teach-back.
- 2026-10-28 · SP6 (Step 7) · D-14 design committed before run (54325dd); group_split_draws +
  experiments/e3_d14.py (59d8119); run from clean 59d8119 in 19 min, code path check exact. D-14
  FAILS (TL-pct DOES NOT FALL; ALL-pct, ESN-refit partial) · tests: pass (535, ruff clean) · next:
  Step 8 write-up.
- 2026-10-28 · SP6 (close-out part 2, Step 6) · Third threshold-logic teach-back override logged.
  D-13 design committed before run (49580f7); D-13a/b/c in e3_diagnostics.py (5042a7a), results from
  clean 5042a7a (70a1eef), D-8..D-12 values unchanged. D-13a HOLDS, D-13c i/ii/iii HOLD · tests: pass
  (524, ruff clean) · next: Step 7 D-14 cost plan.
- 2026-10-27 · SP6 (Step 2 recorded) · No code change. PROGRESS.md brought up to date with Step 2
  (3ce8921, results from clean 3ce8921, 711a23e): E-1 FAILS, E-2 HOLDS (expected fail), E-3 FAILS,
  E-4 HOLDS; SP6 row set to in progress. Threshold-logic teach-back re-asked, owner override a second
  time (NOT ANSWERED) · tests: pass (496, ruff clean) · next: paste results/e3_recalibration.json +
  "Current" into the chat Project for the SP6 next-step spec.
- 2026-10-26 · SP5 (close) · Write-up docs/sp5_e2_result.md (gate FAIL, llama vs gemini arm table,
  per class, D-6, D-7, gemini as an SP8 hypothesis; numbers with file + key); SP3 limit added (ESN
  fit-set variance not varied in SP3, D-5 varied only the reservoir seed); docs/decisions.md
  2026-10-26 (baselines as a fit-draw distribution from SP6 on; SP6 next). Recalibration teach-back
  re-asked, owner override a fourth time (NOT ANSWERED). Tag sp05-e2 · tests: pass (445, ruff clean)
  · next: get the SP6 spec from the chat Project.
- 2026-10-24 · SP5 (close-out) · D-6/D-7 code committed (ecadcfb) plus D-7c result-blind transform
  (blind_results/blind_episodes: every tool result -> same-length filler in tool_events and step
  text, label-free, all runs); results/e2_closeout.json from clean ecadcfb. Channel attribution
  (ESNBaseline.channel_streams, channel_attribution, m_input_contrast;
  scripts/run_e2_channel_attribution.py) -> results/e2_channel_attribution.json from clean e33db31.
  All re-run E2 numbers reproduce exactly. Findings: D-6 median 0.770 [0.566, 0.857], E2's split in
  the top 3%; blinding leaves ESN_i@llama unchanged; m sets llama ESN scores; Step 3b content
  reading corrected. Recalibration teach-back re-asked, overridden a third time (NOT ANSWERED).
  Background-task notices in this session were false (runs reported done while still running; one
  duplicate pair stopped) · tests: pass (445, ruff clean) · next: paste the four E2 JSONs +
  PROGRESS.md into the chat Project for the write-up spec; answer the recalibration teach-back
  before SP5 closes.
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
