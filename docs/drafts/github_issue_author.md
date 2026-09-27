<!-- DRAFT for github.com/sunnydubey1111/agent-trajectory-sentinel. Not posted; the owner posts it. -->

**Title:** Question about reproducing the §5 cross-family transfer result (0.527 vs 0.885)

Hi, and thanks for releasing the code, traces and results tables. It made a reproduction possible. I'm
trying to reproduce the qwen2.5:7b → llama3.1:8b transfer result from arXiv 2608.02464 §5
(`results/tables/model_transfer_family.csv`, monitor `esn_cusum_max`). One number matches exactly and one
doesn't, and I'd like to understand the difference before I cite either.

**What I ran**
- Your code at commit `1b3e07fee53ae13407173c3ea932adb4a43e8230`: `ChannelMaxESNMonitor` (K=8, cusum=True,
  seed=1300), the telemetry adapter with `extended=True` (the 51-dim view), `Standardizer`, and
  `pick_threshold` with a 5% budget. The files are vendored unmodified apart from import paths.
- The data are the `ollama7b` and `ollama_llama8b` traces from the HF dataset. The trace files match your
  manifests' `trace_sha256`.
- The protocol follows `run_hybrid_study.load_real` and `run_model_transfer.py`:
  - runs with T ≥ 4, in manifest order;
  - healthy runs split 60/20/20 with `rng_for(0, "real-split")`;
  - test = held-out healthy runs + all injected runs (16 healthy + 116 injected for llama);
  - episode score = max over steps, AUROC as in `episode_auc`.

**Results**

| arm | your table | my run |
|---|---|---|
| in-domain (ollama_llama8b) | 0.8847 | 0.8847 |
| transfer (qwen7b → llama8b) | 0.5275 | 0.6805 |

The in-domain row matches exactly, including AUPRC 0.9552, detection 0.7672 and healthy false-alarm
rate 0.125. The transfer row doesn't. I checked the channel rule (all four channel groups apply, since both
corpora have logprobs). No channel choice I tried gives 0.53.

**Question 1:** which `ollama7b` episodes, seed and config produced 0.5275? For example, was the table
generated from an earlier version of the `ollama7b` corpus, or with different settings, before the release
commit?

**Question 2, about the in-domain split.** I compared runs by their per-step tool calls (name, args, result).
In the in-domain split described above:
- 20 of the 132 test runs have an exact tool-call twin among the 46 healthy fit runs;
- 121 of the 132 share a task with the fit runs.

Holding task groups out, or keeping twins on one side of the split, lowers the in-domain AUROC in my runs.
Was the overlap between fit and test intended, as a within-task setting? Or would you consider a
task-disjoint split the intended protocol?

**Minimal reproduction** (Python 3.11, [uv](https://docs.astral.sh/uv/)):
```
git clone https://github.com/2006harshith/watchdog
cd watchdog
git checkout cf6d42c
uv sync
uv run python scripts/reproduce_author_transfer.py      # prints both arms, writes results/e1_author_repro.json
uv run python scripts/run_e1_diagnostics.py             # twin / task-overlap counts (~5 min)
```
If it's easier, running `py -m derail.experiments.run_model_transfer --source ollama7b --target
ollama_llama8b --label "qwen7b->llama8b" --out model_transfer_family` on the current main would show whether
the difference is on my side. I haven't run it in your environment.

Thanks for any pointers. Happy to share more detail.
