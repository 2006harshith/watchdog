# Data card — agent-trajectory-sentinel (as used by watchdog)

Source: Hugging Face `sunnydubey1111/agent-trajectory-sentinel` (repo_type="dataset"), file
`data/episodes.parquet`. Companion paper: arXiv 2608.02464 (S. Dubey, Aug 2026).
All numbers below come from `results/e0_audit.json` (E0 audit, SP1, 2026-09-25) unless marked.

## Licence
- Code and trace format: MIT.
- Qwen episodes: Apache-2.0 (model outputs, no redistribution condition).
- llama3.1:8b episodes: Llama 3.1 Community License + Acceptable Use Policy.
- gemini-2.5-flash episodes: Google terms forbid using the Services "to develop models that compete
  with the Services". Rule (proposed, D6): Gemini episodes are used for evaluation only, never training.
- Tool results: Open-Meteo (CC BY 4.0), Wikipedia (CC BY-SA 4.0).

## Size
- 3,581 episodes (rows) in the parquet; the dataset card says 2,823. All 758 extra rows are
  qwen2.5:7b (2,648 vs 1,890 on the card). Which corpora were added after the card is not resolved;
  irrelevant for us because we define our own inclusion list.
- 33 corpora. Step count T per episode: 2–24, bimodal around 5–7 (mock booking) and 11–12 (long research).

## Schema
Episode level: `uid` (unique), `episode_id` (NOT unique: 733 ids appear in more than one corpus),
`corpus`, `model`, `failure_class` (None = no injected failure), `tau` (onset step, 0-indexed,
NaN when failure_class is None), `T`, `has_logprobs`, `n_steps` (= T), `steps`, `metadata`.

Step level (`steps[i]`): `text`, `token_logprobs`, `logprobs_available`, `action`, `latency_s`,
`output_tokens`, `error`, `task`, `tool_events[]` (`name`, `args`, `result`, `result_chars`,
`result_truncated`, `is_error`, `latency_s`, `id`), `schema`.

Metadata: `provenance` (collector, backend, model, episode_seed, task_name, task_sha256, tools,
tool_roster_sha256, requested_class, requested_tau, injector_seed), `injection`, `trace_sha256`,
`collected_at`, `framework`, `accepted_because`, `success` (always None).

Which step fields are model-specific is decided in SP4. Evidence from E0 already: `has_logprobs`
identifies the family (gemini 0/143, llama 433/433, qwen mixed); healthy steps are slower
(mean latency 3.85 s vs 1.5–3.2 s) and longer (48 vs 36–43 output tokens) than failure-class steps,
which is corpus/model confounding, not failure signal.

## Families and labels

| Family | Models | All rows | Organic (unlabelled) | Modelling rows | Healthy | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | qwen2.5:7b (2,648), qwen2.5:3b (357) | 3,005 | 565 | 2,440 | 1,389 | 1,051 |
| llama | llama3.1:8b | 433 | 240 | 193 | 77 | 116 |
| gemini | gemini-2.5-flash | 143 | 0 | 143 | 77 | 66 |
| **Total** | | **3,581** | **805** | **2,776** | **1,543** | **1,233** |

Failure classes (injected): looping 269, context_corruption 266, goal_drift 235, tool_cascade 197,
wrong_document 72, malformed_json 70, rate_limit 64, timeout 60.
- llama has only 4 classes (goal_drift 32, context_corruption 32, looping 29, tool_cascade 23), all in
  one corpus, `ollama_llama8b`.
- gemini is mostly API-level classes (malformed_json 11, wrong_document 10, timeout 9, rate_limit 8)
  plus looping 12, context_corruption 10, tool_cascade 6; no goal_drift.

Label semantics and caveats:
- "Healthy" means "no failure was injected". It is never verified: `success` is None on every row.
- Failures are injected; injections that did not land were dropped (`rejected.json` /
  `landing_failures.json` per corpus). Kept failures may be easier than average.
- 24% of failures (298) carry no injection metadata; they are exactly the failures in the four
  empty-metadata corpora (real_research7b, real_research3b, real_research7b_long, real_ollama7b)
  and still have failure_class and tau. requested_class matches failure_class on all 935 rows that have it.
- tau == 2 for 71% of failures; median tau/T by class 0.18–0.43. Step label rule: a step is
  "failed" only if the run failed and step_idx >= tau.

Organic corpora (organic7b, organic_demo7b, _cold, _ext, _holdout, _provoked, organic_llama8b,
organic_llama8b_cold; 805 rows): failure_class is None on all of them, and neither metadata nor the
per-corpus manifests carry a label. Only organic7b has labels, in `traces/organic7b/organic_labels.csv`
(30 rows: 19 healthy, 11 failed; modes aborted 7, fabricated_count 3, ungrounded_retrieval_blend 1;
with onset_step). Decision D1: organic rows are excluded from SP2–SP6; organic7b is kept aside.

## Task identity (needed for task-disjoint splits)
- `task_sha256` present on 1,989 rows: 311 tasks, median 2 rows per task, max 182; 122 tasks appear
  under more than one model.
- 1,592 rows have neither task_sha256 nor task text in the steps. After removing the 805 organic rows,
  about 787 qwen rows remain without a task id (inferred from counts: the 614 empty-metadata rows plus
  demo7b/demo7b_scoped; SP2's loader reports the exact list).
- Hashing the task text gives 229 groups vs 311 task_sha256 groups on the same rows: the two do not
  group identically. SP2 resolves this (see SP2 spec).

## The matched pair for SP3
`ollama7b` (qwen2.5:7b: 70 healthy, 85 failed) and `ollama_llama8b` (llama3.1:8b: 77 healthy,
116 failed): same framework, one identical 5-tool roster (calculator, get_weather, lookup_flight,
lookup_hotel, search_catalog), same 4 failure classes. They share 11 of their ~24–25 task groups
(task-text hash). Most likely the setting of the paper's §5 transfer result (0.527 transferred vs
0.885 refitted); the exact episodes used there are not published.

## Shortcut risks to design around
- Corpus predicts the label: demo7b, demo7b_scoped, demo_real (281 rows) are 100% healthy;
  real_research7b_long_drift is exactly 120/120. Proposed (D5): all-healthy corpora serve as healthy
  pools for recalibration, not as AUROC evaluation data; always report per-corpus results too.
- Error flag: share of steps with error=True is ~0.48 for rate_limit and timeout, 0.39 looping,
  0.36 tool_cascade, 0.03 healthy, 0.002 for malformed_json and wrong_document. Proposed (D4): headline
  transfer on the four behaviour classes; API-level classes reported separately.
- Family ↔ class mix: families don't share failure classes, so held-out-family results are reported
  per class.
