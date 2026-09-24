# WATCHDOG — COMPLETE HANDOFF

Everything decided, verified and planned in the idea-discovery chat (24–25 Sep 2026), in one file.
Upload this to the Claude chat Project "watchdog" as knowledge. From now on the Project chat is the
place for reasoning; if something here turns out wrong, correct this file there.

Related documents (same content in more detail; this file is the superset):
- Idea-discovery playbook (Stages 0–5): https://claude.ai/code/artifact/f5e0a0ab-882e-47cf-9a6c-da2380e4c48d
- Watchdog blueprint (execution plan): https://claude.ai/code/artifact/f2794d6c-0020-45d2-b2bb-aa9dd7a72e0e
- Starter kit (CLAUDE.md, PROGRESS.md, /resume and /save skills, scripts/e0_data_audit.py): watchdog-starter.zip

Contents
1. What we are building
2. Constraints and how you want AI to work
3. How the idea was chosen (short)
4. Research question, hypothesis, novelty
5. Evidence: verified facts vs unverified claims
6. Prior work and competitors
7. Datasets
8. Compute and free tiers
9. Spec
10. Decisions made
11. Save points SP0–SP11 in detail
12. Tech stack
13. Which AI does what
14. Local vs Colab, and one-time setup
15. Session protocol and teach-backs
16. Claude chat Project instructions
17. Prompt library
18. Colab template
19. Open questions and checks still to do
20. Corrections log
21. Honest assessment: what this project will and won't do for you

---

## 1. What we are building

A supervisor agent that watches another AI agent while it runs, predicts from the partial run that
it has already failed, and then retries from the last good step, halts, or escalates to a human.

The core is a small model trained on step signals that do not depend on which LLM is inside the
agent: tool-call structure, state changes, repetition, error flags, verifier results. After the
underlying LLM is swapped, it recalibrates from a handful of healthy runs with no failure labels.

Pitch: "Swap the model under your agent; the watchdog is recalibrated after a few healthy runs."

Phase 2 (only after the core is proven): the same calibrated risk scores decide which agent actions
a human must review, so a fixed review budget catches the most bad actions.

Name "watchdog" is a placeholder.

## 2. Constraints and how you want AI to work

From your answers:
- Priority: job hiring > top-college (German master's) admissions > monetization. Users only if needed.
- ML core: must be load-bearing; models you train yourself are required.
- Zero budget: free tiers only. Solo. Fully open-source (MIT). No deadline.
- No regulated domains (health, finance, legal). Desk research only (no user interviews).
- Hardware: HP laptop, Ryzen 5 3550H, GTX 1650 Mobile (4 GB), 16 GB RAM, Windows 10.
- Accounts: Claude Pro, Perplexity free, Grok (inside X), ChatGPT Go, Google AI Pro via Jio
  (includes Gemini Deep Research, NotebookLM, higher Gemini CLI limits), Groq API key.

How AI should work with you (also in CLAUDE.md):
- Blunt, no warm-up, uncomfortable truth first, no praise openers.
- Never present a guess as fact; verify what is checkable; say plainly what is unverified.
- Disagreement format: "I disagree because X. Here's what I'd do instead: Y. The risk in your approach is Z."
- Hold position under pushback unless there is new information.
- Plan before code; small diffs; never claim something works until it has run; test first.
- Name non-obvious NumPy/pandas/sklearn/PyTorch idioms in one line.
- No new dependency without saying what it replaces. Comment the why, not the what.
- Claude Code writes ALL code. For the four core pieces, code is written and run until it passes,
  THEN a teach-back; nothing moves on until you pass (section 15).

## 3. How the idea was chosen (short)

Pipeline: Stage 0 baseline → Stage 1 pain mining (Gemini Deep Research G1–G3, Grok X1–X3, Reddit
Answers R1–R15, Perplexity P1–P2) → Stage 2 clustering (9 clusters) → Stage 3 kill round (Perplexity
competitor checks P3, ChatGPT red-team C1) → Stage 4 deep dive (G4, NotebookLM N1, Grok X4, Reddit
R16–R20) → Stage 5 spec.

Kill criteria used: K1 ML is load-bearing; K2 survives a 2x better frontier LLM; K3 free data
exists; K4 runs free; K5 not already done (or a measurable wedge); K6 solo v1 possible.
Scoring weights: hiring 30%, evaluability 20%, pain evidence 15%, 5-year durability 15%,
research novelty 10%, money/adoption 10%.

Outcome:
- Stage 2 survivors: A online failure prediction, B learned approval gate, C trajectory security
  monitoring. Killed: cost routing (funded competitors, weak learned-router gains), coding-agent
  context (frontier labs absorb it), data reconciliation and manufacturing (no free data), SOC
  triage (single-platform evidence).
- Stage 3: C killed (sequence-level agent security already a product category: Reveal Security,
  Invariant, Lakera, plus research). A and B survived only reframed; both critics pointed to the
  same open problem: does a small failure-risk model keep working when the underlying model changes?
- Scores (my judgment): A' transferable failure monitor 3.90; B' near-miss review under a budget 3.65.
- Chosen: A' as the lead, B' as phase 2.

Known bias in the process: Stage 1 prompts over-sampled agent builders, so all survivors are agent
infrastructure. Stage 1 G2 answered "ML could help" on all rows because the prompt was leading.

## 4. Research question, hypothesis, novelty

Research question: Can a cheap monitor built on model-independent step features match the transfer
of 4–7B LLM auditors across LLM families, at a tiny fraction of the cost, and recalibrate on a new
model from a handful of healthy runs with no failure labels?

Hypothesis: cheap telemetry monitors fail to transfer because they rely on model-specific signals
(token surprisal, embeddings of model-written text, output length, latency). A monitor using only
model-independent signals should transfer; the remaining gap is threshold calibration, fixable from
healthy runs alone.

Why it's open:
- A cheap telemetry monitor collapses under a model swap (0.527 vs 0.885, section 5).
- 4–7B LLM auditors transfer reasonably (reported, not verified by me: AgentForesight-7B on Who&When;
  PolicyGuard-4B held-out domain 91.0% → 86.5%). They cost far more per step.
- So the question is the cost–transfer trade-off plus label-free recalibration.

Novelty is moderate and rests partly on one single-author preprint (arXiv 2608.02464). If one
simple normalization fixes transfer, the result is small but still a clean, publishable finding.

## 5. Evidence: verified facts vs unverified claims

"Verified" = opened the primary source during the chat.

Verified:
| Fact | Source |
|---|---|
| Monitor fitted on healthy qwen2.5:7b episodes, applied unchanged to llama3.1:8b: episode AUROC 0.527 (healthy false-alarm 0.75). Same monitor type refitted on llama's own healthy episodes: 0.885. Both on the same llama3.1:8b episodes, "matched task, tool and injector plan". NOT a before/after pair. | arXiv 2608.02464 §5 (Sunny Dubey, 3 Aug 2026) |
| That paper's monitor: echo-state-network ensemble with CUSUM alarms; features = 32-dim char-3-gram hash embeddings, 4 surprisal measures, action metadata (tool calls, latency, output length, error flags); ~200 µs/step; readout fit by ridge regression on healthy episodes | arXiv 2608.02464 |
| The repo mentions the transfer result only under "Future work"; no script, results file, episode counts or seeds for it; "7b → 3b falls below chance" | github.com/sunnydubey1111/agent-trajectory-sentinel |
| agent-trajectory-sentinel dataset: 2,823 episodes; qwen2.5:7b 1,890, llama3.1:8b 433, qwen2.5:3b 357, gemini-2.5-flash 143; injected failures (goal_drift, looping, tool_cascade, context_corruption, grounding_loss, ...); onset step `tau` (0-indexed, null for healthy); Parquet + JSONL; 59.6 MB; mixed licences (Qwen Apache-2.0, Llama 3.1 Community License, Gemini episodes restrict building competing models; code MIT) | HF dataset card |
| AgentForesight: AFTraj-2K = 2,276 trajectories (1,162 safe, 1,114 unsafe), HF `ZBox008003/AFTraj`, dataset CC BY 4.0, code MIT, 7B weights "released upon paper acceptance" (not yet); paper claims AgentForesight-7B beats GPT-4.1 and DeepSeek-V4-Pro by up to 19.9% | github.com/ZBox1005/AgentForesight, arXiv 2605.08715 |
| Who&When: failure logs from 127 multi-agent systems; best method 53.5% agent-level, 14.2% step-level accuracy; data at github.com/mingyin1/Agents_Failure_Attribution | arXiv 2505.00212 |
| MAST-Data: 1,642 traces, 7 frameworks (AG2, MetaGPT, ChatDev, Magentic, AppWorld, HyperAgent, OpenManus), 5 LLMs (GPT-4o, Claude, GPT-4o-mini, Qwen, CodeLlama), CC BY 4.0, labels from an LLM judge, trace-level (14 failure modes), no step onset; 19 human-labelled traces | HF mcemri/MAST-Data |
| CORA: guardian risk model + conformal calibration, execute/abstain for mobile GUI agents; Phone-Harm benchmark 150 harmful + 150 benign tasks; code released | cora-agent.github.io, arXiv 2604.09155 |
| Claude Code auto mode: a Sonnet 4.6 classifier gates higher-risk tool calls; "Claude Code users approve 93% of permission prompts" (25 Mar 2026) | anthropic.com/engineering/claude-code-auto-mode |
| ReDAct: defers from a small to a large model using token-uncertainty thresholds (not a trained model, not humans); ALFWorld, MiniGrid | arXiv 2604.07036 |
| "Oversight Has a Capacity" (Headroom) exists | arXiv 2606.08919 |
| Galileo Luna-2: 3B/8B proprietary evaluator models; Fiddler Centor: proprietary, page does not mention step-level agent failure prediction | docs.galileo.ai, fiddler.ai |
| Waxell approval workflows use static policies, not a learned model | waxell.ai blog, 25 Jun 2026 |
| Groq free tier (gpt-oss-120b, gpt-oss-20b, qwen3.8-27b): 30 RPM, 1K RPD, 8K TPM, 200K TPD each | console.groq.com/docs/rate-limits |
| Gemini CLI personal free tier: 60 requests/min, 1,000/day; Jio AI Pro offer includes higher Gemini CLI limits, Deep Research, NotebookLM | github.com/google-gemini/gemini-cli, jio.com |
| Claude Code needs Pro/Max/Team/Enterprise/Console (not free). Windows 10 1809+ supported natively; Git for Windows recommended | code.claude.com/docs/en/setup |
| Custom commands are now skills: `.claude/skills/<name>/SKILL.md` (old `.claude/commands/*.md` still work); CLAUDE.md loads at startup | code.claude.com/docs/en/slash-commands |
| YC S26: 235 companies, ~60% AI; agent-infra companies incl. Inkbox (identity), Archal (agent testing sandboxes), Glen and Egoist Machines (memory), Understudy Labs (cost routing), Agentcard (payments) | byteiota (third-party, Sep 2026) |
| Hugging Face is blocked only in Claude's cloud sandbox, not on your laptop | observed |

Reported by other AIs, NOT verified (treat as leads):
- ESN monitor from own testbed → AFTraj-2K: AUROC 0.872 → 0.745, detection at 5% false-alarm 71% → 4.8% (G4, N1).
- Removing token-logprob features changed ESN AUROC by +0.000; deterministic verifiers caught 110/110 failures after the swap (G4, N1).
- AgentForesight-7B on Who&When: 57.69% step accuracy, +19.6 points over GPT-4.1 (N1).
- PolicyGuard-4B: 91.0% in-domain → 86.5% held-out domain; 90.1% acc, 22.5 ms/example (N1).
- Near-miss ratios 8.6%–20% on τ²-verified Airlines; "8–17% of successful trajectories contain latent violations" (N1, C1-B).
- Headroom: 125 hand-labelled actions, κ = 0.52, inverted-U safety vs escalation rate (N1).
- One Human, N Agents (2607.28317): confidence-ranked audits can be worse than random for open-weight models (P3-B, N1).
- Gemini API free tier "100 RPD" (forum post, G4); ~10K tokens per AgentDojo episode and ~150K per τ-bench episode (G4 estimates).
- AgentDojo licence MIT (P3-C). Kaggle free GPU quota (unread).
- Kitaru by ZenML replays sessions against a new model (X4-13). Reveal Security "sequence-based detection for AI agents", Noma, Invariant flow rules (C1-C).
- "GPT-6 Astra system card" monitor-evasion claim (C1-C); Galileo acquired by Cisco (P3-A).
- arXiv 2604.23425 exists but is a single-author position paper asserting a sandbox escape, not an incident report (P3-C).

## 6. Prior work and competitors

Research (arXiv IDs):
- 2608.02464 Real-Time Detection and Repair of LLM Agent Failures — cheap telemetry monitor; the transfer-collapse number. Closest baseline.
- 2605.08715 AgentForesight — online auditing, 7B auditor, AFTraj-2K. Closest research competitor for "online failure prediction".
- 2505.00212 Who&When — failure attribution; step-level is hard.
- 2605.26563 TrajAudit — failure diagnosis on repo-level coding (RootSE).
- 2512.07850 SABER — failures concentrate at mutating steps.
- 2603.29665 Near-Miss — latent policy failures in "successful" runs.
- 2510.03485 PolicyGuard — 4B trajectory policy guard.
- 2604.09155 CORA — guardian + conformal risk control (closest to phase 2).
- 2604.07036 ReDAct — uncertainty-based deferral.
- 2606.29654 Budgeted Act-or-Defer / Certifiable Multi-Agent Debate.
- 2607.28317 One Human, N Agents — audit budgets.
- 2606.08919 Oversight Has a Capacity (Headroom) — reviewer fatigue.
- TRAIL (Patronus, 2505.08638) — 148 traces, human-annotated error localization (G4; not verified by me).

Products: Galileo Luna-2, Fiddler Centor, Patronus, Arize Phoenix, LangSmith, Langfuse, Braintrust,
AgentOps (observability/eval — tracing, not prefix prediction); Archal (YC S26, pre-deployment
sandboxes); Claude Code auto mode (LLM classifier gating actions); Waxell, HumanLayer, AgentGate
(static approval policies); Kitaru/ZenML (model-swap replay, unverified).

Your differentiation: small, cheap, open, calibrated, model-independent features, label-free
recalibration after a model swap, measured against both the cheap monitor and LLM auditors.

## 7. Datasets

| Dataset | Use in this project | Caveats |
|---|---|---|
| agent-trajectory-sentinel (HF sunnydubey1111/agent-trajectory-sentinel) | Main data for SP1–SP6; 3 families with step onset labels | Injected failures; 67% qwen2.5:7b; Gemini licence clause; exact episodes behind 0.527/0.885 unknown |
| AFTraj-2K (HF ZBox008003/AFTraj) | Framework-shift test; AgentForesight comparison | Reportedly single generator model (unverified) |
| Who&When | External test | Post-hoc attribution labels |
| MAST-Data | Second transfer test (5 families) | LLM-judge, trace-level labels only |
| Your own AgentDojo runs (SP8) | Organic failures, 3 families | Must generate; free-tier budgets |

## 8. Compute and free tiers

- Local CPU: enough for SP0–SP6 (60 MB data, XGBoost, small GRU, NumPy ESN).
- GTX 1650 4 GB: small GRU; 3B quantized models via Ollama (plausible, untested).
- Colab/Kaggle GPU: SP7 LoRA and 7B inference only.
- Groq free: ~200K tokens/day per model → at ~10K tokens/episode (unverified), ~20 AgentDojo episodes/day/model.
- Gemini free: ~41 AgentDojo episodes/day if 100 RPD and ~2.4 calls/episode (both unverified).
- Estimate: ~100 episodes/day across 3 families, ~3,000/month, IF the token estimate holds. τ-bench is not feasible on free tiers (~150K tokens/episode, unverified).

## 9. Spec

Input: an agent run as a sequence of steps. Per step: tool name, argument shape, result
status/size, error flags, state-change flags, timing, repetition counters, verifier outputs.
Ablation-only (model-specific): token surprisal, embeddings of model-written text, output length, latency.

Output: after each step, a calibrated probability that the run has already failed, plus an alarm
decision under a fixed false-alarm budget. Supervisor actions: continue, retry from last good
checkpoint, halt, escalate.

Splits: always hold out by model family AND by task (no task on both sides). Never random splits.

Metrics: episode AUROC, AUPRC, detection at 5% false-alarm rate, false interventions per run,
ECE, steps between alarm and failure onset (lead time), µs per step. Bootstrap 95% CIs resampled
by episode (not by step).

Experiments:
- E0 data audit (SP1)
- E1 reproduce the collapse (SP3) — GATE
- E2 feature ablation: model-independent vs model-specific vs both, held-out families (SP5) — GATE
- E3 recalibration from healthy runs only (SP6)
- E4 cost/latency vs LLM auditors (SP7) — GATE
- E5 organic runs (SP8) — GATE
- E6 budgeted human review (SP11, phase 2)

Known ways it fails: learns injection artefacts (E5 catches it); family imbalance (report per
family, reweight); slow goal drift evades per-step monitors (report per failure type); Gemini
licence clause (check before training on those episodes); in-sample optimism if calibration and
evaluation episodes overlap.

Out of scope: tracing UI (use Langfuse if needed), multi-agent blame attribution, security.

## 10. Decisions made

1. Data: sentinel's injected failures first (E1–E3), organic runs later (E5).
2. Monitor: gradient-boosted trees (XGBoost) or a small GRU on step features; ≤1.5B LM only as a comparator.
3. Recalibration: healthy runs only, no failure labels.
4. Phase 2 only after E5 passes.
5. Claude Code writes all code. Core pieces get a teach-back after the code passes; nothing moves on until you pass.
6. Almost everything runs locally; Colab only for SP7 GPU jobs.
7. SP3 gate redefined (the old "0.885 → ~0.53" wording was wrong; see SP3).

## 11. Save points SP0–SP11 in detail

Every save point ends with: tests pass, commit, tag `spNN-<name>`, PROGRESS.md updated, PROGRESS.md
re-uploaded to the chat Project. Before starting each save point, ask the chat Project for its spec
(prompt in section 17), paste it into Claude Code.

SP0 — Setup (tag sp00-setup)
- pyproject via uv (Python 3.11, package `watchdog_agent` in src/), dev deps pytest + ruff, tests/test_smoke.py,
  .gitignore (Python, .env, mlruns/, data caches), GitHub Actions running `uv run pytest` and `uv run ruff check`.
- Move scripts/e0_data_audit.py in. Done: tests green locally and on CI. Prompt is in PROGRESS.md.

SP1 — E0 data audit (tag sp01-audit)
- `uv add huggingface_hub pandas pyarrow`, then `uv run python scripts/e0_data_audit.py`.
- Paste output into the chat Project; write docs/data_card.md.
- Must answer: exact per-step fields; which are model-specific; is there a task id (needed for
  task-disjoint splits); do families share the same tasks ("matched plan"); how many healthy vs
  failed per family; tau distribution; Gemini licence terms; which episodes could be the §5 experiment.
- Optional: open a GitHub issue on agent-trajectory-sentinel asking for the episode IDs behind the §5 numbers.

SP2 — Data layer (tag sp02-data) — teach-back: splits
- Loader → one row per step: run_id, corpus, model, model_family, task_id, step_idx, is_failed_run,
  tau, step_label (1 if failed run and step_idx ≥ tau), raw step fields.
- Split module: leave-one-family-out, task-disjoint; healthy-only calibration subsets of size n.
- Tests: no task_id in both train and test; no run in two splits; family held out entirely.

SP3 — E1 reproduce the collapse (tag sp03-e1) — GATE — teach-back: metrics
- Baseline: reuse the paper's own ESN code from the repo (MIT) rather than reimplementing, if it
  runs; otherwise reimplement from the paper description and say so.
- Protocol: fit monitor A on healthy qwen2.5:7b episodes; fit monitor B on a held-out subset of
  healthy llama3.1:8b episodes; score both on the SAME llama3.1:8b test episodes (none used in fitting).
- Pass: A's AUROC at least 0.15 below B's, with non-overlapping episode-bootstrap 95% CIs.
  Reference only: 0.527 vs 0.885 (arXiv 2608.02464 §5). The 0.15 threshold is our choice.
- If no gap: that is a finding (transfer problem smaller than claimed); decide next step in chat before SP4.
- Also set up MLflow local logging and the metrics module (tested against sklearn roc_auc_score,
  average_precision_score).

SP4 — Feature sets (tag sp04-features) — teach-back: feature definitions
- Model-independent: tool identity, repeated identical calls, argument shape/schema validity,
  result status and error flags, result size bucket, state-change flags, step index, verifier outputs.
- Model-specific (ablation): surprisal, text embeddings, output length, latency.
  Latency and output length look structural but depend on the model — defend that in the teach-back.
- Prefix aggregation: counts, rates, last-k window stats per prefix.
- One unit test per feature.

SP5 — E2 ablation (tag sp05-e2) — GATE
- XGBoost on prefix features, per feature set; small GRU on the step sequence as a variant.
- Evaluate leave-one-family-out; per-family and per-failure-type results.
- Red-team with ChatGPT (template in section 17).
- Pass: model-independent features transfer clearly better than model-specific ones.
  Fail: write the negative result; SP6 becomes the main contribution.

SP6 — E3 recalibration curve (tag sp06-e3) — teach-back: threshold logic
- Fit on source families; on the target family, set the alarm threshold from n healthy runs only
  (n = 0, 5, 10, 25, 50), e.g. the (1−α) quantile of per-run max scores so the false-alarm rate is
  ≈ α; optionally also normalise features by target healthy-run statistics.
- Output: detection at 5% false-alarm vs n, with CIs from repeated resampling.

SP7 — E4 cost vs LLM auditors (tag sp07-e4) — GATE
- Judges: local 1.5–3B (Ollama), Groq gpt-oss-20b/120b and qwen3.8-27b, a LoRA-tuned ≤1.5B model on
  Colab, AgentForesight-7B if weights released.
- Same prefixes, same metrics, plus cost and latency per step.
- Pass: your monitor reaches ≥80% of the best judge's detection at a stated cost ratio.
- Refresh the landscape first with Gemini Deep Research (section 17).

SP8 — E5 organic runs (tag sp08-e5) — GATE
- Harness running AgentDojo tasks (no injections, utility tasks) with 3 families via Groq and
  Gemini free APIs; environment-checked success labels; ~500 runs per family target.
- First verify: AgentDojo licence, provider support, real tokens per episode, actual rate limits.
- Onset step unknown → evaluate episode-level AUROC and lead time relative to the end.
- Pass: the transfer result holds on organic failures.

SP9 — Supervisor agent + demo (tag sp09-demo)
- Wrapper around any agent's step loop; policy: alarm → retry once from last checkpoint → halt → escalate.
- Streamlit app on Hugging Face Spaces: replay episodes, per-step risk curve, "swap model +
  recalibrate with N healthy runs" walkthrough.
- README with results table, failure analysis section (hiring managers look for it), architecture diagram.
- Grok check of X reaction after launch.

SP10 — Write-up (tag sp10-writeup)
- Technical report / paper (LaTeX on Overleaf): problem, related work (NotebookLM), method, E1–E5,
  negative results, limitations. arXiv first-time submitters may need an endorsement — check.
- ChatGPT red-teams the draft; you write the final text yourself.
- Blog post + LinkedIn post linking repo and demo.

SP11 — Phase 2: budgeted human review (tag sp11-review)
- Same scores decide which actions go to a human under 5/10/20% budgets; near-miss labels by
  executing actions in a sandbox; compare with static rules, an LLM judge, random. Spec it in chat first.

## 12. Tech stack

| Choice | Replaces | When |
|---|---|---|
| Python 3.11 (same locally and on Colab) | — | SP0 |
| uv (pyproject + uv.lock) | venv + pip + requirements.txt | SP0 |
| pytest, ruff | flake8 + black | SP0 |
| GitHub Actions CI | — | SP0 |
| pandas, pyarrow, huggingface_hub | — | SP1 |
| scikit-learn, XGBoost | — | SP3–SP5 |
| NumPy (ESN), PyTorch (GRU) | — | SP3, SP5 |
| MLflow (local file store) | scattered CSVs | SP3 |
| transformers + peft (Colab) | — | SP7 |
| Groq + Gemini SDKs behind one wrapper | — | SP7–SP8 |
| AgentDojo | — | SP8 |
| Streamlit on HF Spaces | — | SP9 |
| LaTeX on Overleaf | — | SP10 |
No LangChain, no agent framework, no vector DB: the supervisor stays framework-agnostic.

## 13. Which AI does what

| Tool | Job | Not for |
|---|---|---|
| Claude Code | All repo code, tests, runs, git | Literature, independent critique |
| Claude chat Project | Specs per save point, reading results, decisions, prose | Repo code |
| Gemini CLI | Second-opinion review of big diffs; reading huge files/logs | Writing code |
| ChatGPT Go | Red-team each gate's result and the paper | Code |
| NotebookLM | Paper Q&A, related work (source-bound) | Anything outside uploads |
| Gemini Deep Research | Landscape refresh before SP7 and SP10 | Daily questions |
| Perplexity | Quick cited lookups (versions, limits) | Deep research |
| Grok | X reaction at demo launch | Everything else |
| Groq/Gemini APIs | Generating runs, LLM-judge baselines | — |
| Colab/Kaggle | GPU jobs only | Development |

## 14. Local vs Colab, and one-time setup

Local: SP0–SP6, SP7 API judges, SP8, SP9. Colab: SP7 LoRA / 7B inference.

Windows setup (PowerShell, not as Administrator):
1. Git for Windows: https://git-scm.com/downloads/win
2. uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. Claude Code: `irm https://claude.ai/install.ps1 | iex`, reopen terminal, `claude --version`
4. Node.js LTS (nodejs.org), then `npm install -g @google/gemini-cli`, run `gemini`, sign in with Google
5. Create public GitHub repo `watchdog` (MIT), `git clone https://github.com/2006harshith/watchdog.git`, `cd watchdog`
6. Copy the contents of the starter kit folder (CLAUDE.md, PROGRESS.md, .claude/, scripts/) into the repo root
7. `claude` → `/resume` → paste the SP0 prompt from PROGRESS.md
Optional: `uv python install 3.11` (uv also downloads Python automatically when needed).

## 15. Session protocol and teach-backs

Start: `git pull` → `claude` → `/resume` (reads PROGRESS.md, git log, runs tests, tells you the next
step; a pending teach-back always comes first). New save point → get its spec from the chat Project.

End: `/save` (tests, PROGRESS.md session log, commit, push if green). Finished save point:
`/save SP3` also tags it — refused if a teach-back is pending/failed. Re-upload PROGRESS.md to the Project.

Crash mid-session: `/resume` lists uncommitted changes and asks keep or discard.

Teach-back (core pieces: splits, feature definitions, calibration/threshold, metrics):
1. Claude Code writes the code and tests; runs until everything passes.
2. Walks you through the lines that matter, incl. the design choice and the rejected alternative.
3. Asks three interview-style questions; you answer in your own words.
4. Honest grade: where you were vague or reciting a label. Pass only if you could defend it to a skeptical interviewer.
5. Logged in PROGRESS.md "Teach-backs". Nothing new starts until you pass.

## 16. Claude chat Project instructions (paste into the Project)

```text
Project: watchdog — a supervisor agent whose trained core predicts LLM-agent failure from partial runs using model-independent signals, and recalibrates after a model swap from a few healthy runs.

This chat is for specs, decisions, reading results and writing. Code is written in Claude Code in the repo, never here. Before answering, check PROGRESS.md and HANDOFF.md in project knowledge for the current save point and prior decisions.

For each new save point, produce a spec before any code: goal, inputs and outputs, files to touch, tests that prove it works, failure modes, the decisions that could go either way, and which core pieces need a teach-back (splits, feature definitions, calibration and threshold logic, metrics). Claude Code writes all the code; I must be able to defend the core pieces. Keep it short enough to paste into Claude Code.

When I paste results: say plainly whether the gate passed, what could make the result wrong (leakage, a family imbalance, injection artefacts, calibration/eval overlap), and what to check next. Never call a result good because it looks good; ask for the baseline and the held-out split.

Check anything current (library versions, API limits, new papers) instead of answering from memory, and say what you couldn't verify. Be blunt; lead with the uncomfortable truth; use "I disagree because X. Instead: Y. Risk: Z."
```

Knowledge files: HANDOFF.md, the blueprint, PROGRESS.md (re-upload each save point),
docs/data_card.md (from SP1), each gate's results table.

## 17. Prompt library

Spec request (chat Project, start of each save point):
```text
We're starting SP<N> (<name>). Read PROGRESS.md and HANDOFF.md section 11 for SP<N>. Write the spec per the project instructions, paste-ready for Claude Code. Flag anything in HANDOFF.md that is unverified and affects this save point.
```

Gate red-team (ChatGPT Go, fresh chat, only numbers and claim):
```text
You are a skeptical ML reviewer. Claim: <one sentence>. Setup: <data, splits, model, baselines>. Results: <table>. Attack it: leakage between train and test, whether the held-out split really tests transfer, whether the baseline is fair, whether CIs support the claim, whether the effect could be an artefact of injected failures or family imbalance, and what single additional experiment would most likely overturn it. Verdict per point: survives / weak / fatal.
```

Gemini Deep Research refresh (before SP7 and SP10):
```text
Since <date>, list new papers, repos and products on: online failure prediction for LLM agents, transfer of agent monitors/evaluators across LLM families, label-free recalibration of classifiers under model swap, and budgeted human review of agent actions. For each: link, date, what it claims, whether it releases code/data. Cite only pages opened. Flag anything that makes "cheap model-independent monitor with label-free recalibration" non-novel.
```

NotebookLM (upload the section 6 papers):
```text
Using only the sources: which papers test a monitor/evaluator on a different generator model or framework than it was trained on? For each: train setup, test setup, metric before and after. Which release data and code? Where do they disagree? Say "not in sources" when absent.
```

Perplexity quick check:
```text
Current official free-tier limits for <provider/model> as of today, with the official page link. Only official documentation.
```

## 18. Colab template (SP7 only; module names get defined in SP7)

```python
!git clone https://github.com/2006harshith/watchdog.git
%cd watchdog
!pip install -q -e .
!python -m watchdog_agent.cli <sp7-command> --config configs/<sp7-config>.yaml
from google.colab import drive; drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/watchdog_results && cp results/sp7_*.json /content/drive/MyDrive/watchdog_results/
```
Then download the JSON and commit it locally via Claude Code. Notebooks hold no logic.

## 19. Open questions and checks still to do

- Exact episodes, counts, seeds and fit/eval split behind 0.527/0.885 (ask the author; SP1).
- Whether the sentinel data has task ids and matched tasks across families (SP1).
- Gemini episode licence terms (SP1).
- Whether AgentForesight-7B weights get released (check before SP7).
- AgentDojo licence, provider support, real tokens/episode; Gemini API official free limits; Kaggle GPU quota (before SP8/SP7).
- Kitaru/ZenML and Reveal Security capabilities (competitor check before SP10).
- arXiv endorsement requirement for your category (before SP10).
- All "reported, not verified" items in section 5.

## 20. Corrections log (mistakes caught during the chat)

- "Groq for X" → the X-reading AI is Grok (xAI); Groq is an inference provider.
- I first called G3's "AFTraj-2K" wrong; the repo confirms 2,276 trajectories — G3 was right.
- Weighted scores first written as 3.95/3.70; correct values 3.90/3.65.
- "0.885 → ~0.53" was framed as before/after; both numbers are on the same llama3.1:8b episodes
  (transferred vs refitted). SP3 gate rewritten.
- ChatGPT C1-B said ReDAct couldn't be verified; it exists (2604.07036).
- G4 said new-run generation is infeasible; true for τ-bench, not for short AgentDojo episodes given per-model Groq limits.
- G2's "38% parsing failures" figure was not in the cited paper's abstract.
- I promised HN/GitHub mining in Stage 1 and did not run it; it was only partly done later.
- (25 Sep, Project chat) Package name `watchdog` → `watchdog_agent`: PyPI `watchdog` is a Streamlit
  dependency on Windows/Linux (streamlit 1.64.0 metadata), which would clash at import in SP9.

## 21. Honest assessment

- Hiring: yes, if finished through SP8 with real results. It demonstrates evaluation, calibration,
  distribution shift, monitoring and reliability — the skills rising in 2026 postings.
- Admissions: small direct effect. TUM, Freiburg and Tübingen score mainly on grades and ECTS
  content; the project counts through CV, motivation letter, or a paper.
- Biggest risks: novelty rests partly on one preprint; injected-failure data; scope creep into a
  tracing product; results that only hold on the benchmark they were built on.
- A negative result at SP3 or SP5 is not failure — write it up; it still shows research judgement.
