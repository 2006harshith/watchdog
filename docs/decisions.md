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

### Reporting from SP4 on
- The task-group cluster bootstrap is the primary CI; the run bootstrap is secondary.
- Every headline number reports n_healthy.
