import math

import numpy as np
import pandas as pd
import pytest

from watchdog_agent.experiments.e5 import (
    ORGANIC_CFG,
    esn_block,
    evaluate,
    gate_verdict,
    latency_table,
)
from watchdog_agent.organic import FAILED


def _organic_runs(n_healthy=6, n_failed=30, seed=0):
    rows = []
    for i in range(n_healthy + n_failed):
        healthy = i < n_healthy
        rows.append({"uid": f"r{i}", "task_group": f"run:r{i}", "T": 6,
                     "failure_class": None if healthy else FAILED, "tau": math.nan if healthy else 0.0,
                     "healthy": healthy,
                     "steps_parsed": [{"latency_s": 2.8, "output_tokens": 23}] * 6})
    return pd.DataFrame(rows)


def _scores(runs, shift, seed):
    rng = np.random.default_rng(seed)
    return {u: rng.standard_normal(6) + shift * (not h) for u, h in zip(runs["uid"], runs["healthy"])}


def test_run_level_labels_count_every_failed_run_at_every_checkpoint():
    runs = _organic_runs()
    out = evaluate({"A": _scores(runs, 3.0, 0)}, runs, B=200, seed=0)["A"]
    assert out["headline"] > 0.9 and set(out["per_t"]) == {"3", "4"}
    assert out["ci"]["lo"] <= out["headline"] <= out["ci"]["hi"]


def test_gate_is_strictly_above_c():
    assert gate_verdict(0.51, 0.5) == "PASS" and gate_verdict(0.5, 0.5) == "FAIL" and gate_verdict(0.3, 0.5) == "FAIL"


def test_a_corpus_with_one_healthy_run_still_has_an_auroc():
    runs = _organic_runs(n_healthy=1, n_failed=30)
    out = evaluate({"A": _scores(runs, 1.0, 1)}, runs, B=50, seed=0)["A"]
    assert 0.0 <= out["headline"] <= 1.0  # it is the share of failed runs scored above the one healthy run
    assert out["ci"]["lo"] is None and "one-class" in out["ci"]["undefined"]  # no CI from one healthy run


def test_esn_block_reports_the_draw_distribution_and_an_actual_median_draw():
    runs = _organic_runs()
    draws = [_scores(runs, s, k) for k, s in enumerate((0.0, 1.0, 2.0, 3.0, 4.0))]
    b = esn_block(draws, runs, B=50, seed=0)
    assert len(b["headline_per_fit_draw"]) == 5 and b["headline_min"] <= b["headline_median"] <= b["headline_max"]
    assert b["headline_per_fit_draw"][b["median_draw"]] == pytest.approx(b["headline_median"])


def test_latency_table_quantiles_and_fast_share():
    runs = _organic_runs(2, 2)
    t = latency_table({"x": runs})["x"]
    assert t["steps"] == 24 and t["share_latency_below_0_05s"] == 0.0 and t["latency_s_q10_q50_q90"][1] == 2.8


def test_organic_cfg_targets_the_organic_failure_class():
    assert ORGANIC_CFG["classes"]["headline"] == [FAILED] and ORGANIC_CFG["checkpoints"]["report"] == [3, 4]
