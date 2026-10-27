import json

import numpy as np
import pandas as pd
import pytest
from test_e3 import GROUP_RECORD, e3_run  # noqa: F401  (e3_run is a fixture)

from watchdog_agent.experiments.e3 import group_frame
from watchdog_agent.experiments.e3_diagnostics import (
    FA,
    TPR,
    FamilyCounts,
    bootstrap_weights,
    conformal_alpha_eff,
    plugin_expected_fa,
    run_diagnostics,
    simulate_fa,
    vectorised_thresholds,
)
from watchdog_agent.thresholds import conformal_threshold, plugin_threshold


def test_closed_forms():
    assert conformal_alpha_eff(10, 0.1) == pytest.approx(1 - 10 / 11)  # k = ceil(9.9) = 10
    assert conformal_alpha_eff(20, 0.1) == pytest.approx(1 - 19 / 21)
    assert conformal_alpha_eff(5, 0.1) == 0.0  # k = 6 > m: theta = +inf, no alarms
    assert conformal_alpha_eff(8, 0.1) == 0.0 and conformal_alpha_eff(9, 0.1) == pytest.approx(0.1)
    assert plugin_expected_fa(10, 0.1) == pytest.approx(1 - 9.1 / 11)  # h = 8.1
    assert plugin_expected_fa(5, 0.1) == pytest.approx(1 - 4.6 / 6)


@pytest.mark.parametrize("m", [5, 9, 10, 23])
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_vectorised_thresholds_match_the_thresholds_module(m, alpha):
    cal = np.random.default_rng(m).standard_normal((50, m))
    for rule, fn in (("conformal", conformal_threshold), ("plugin", plugin_threshold)):
        np.testing.assert_array_equal(vectorised_thresholds(cal, rule, alpha), [fn(row, alpha) for row in cal])


def test_simulation_agrees_with_the_closed_forms_on_uniform_scores():
    rng = np.random.default_rng(0)
    for m in (5, 10, 20):
        assert simulate_fa("conformal", 0.1, m, 40000, "uniform", rng) == pytest.approx(conformal_alpha_eff(m, 0.1), abs=0.01)
        assert simulate_fa("plugin", 0.1, m, 40000, "uniform", rng) == pytest.approx(plugin_expected_fa(m, 0.1), abs=0.01)


def test_bootstrap_weights_resample_every_group_count_once_per_column():
    M = bootstrap_weights(7, 100, 0)
    assert M.shape == (7, 100) and (M.sum(axis=0) == 7).all()
    np.testing.assert_array_equal(M, bootstrap_weights(7, 100, 0))


def _diag_cfg():
    return {"alpha": 0.5, "primary_family": "llama",
            "d8": {"n": [4, 8, 14]},
            "d9": {"arms": ["TL-pct", "ALL-pct", "ESN-refit"], "n": [8, 14], "fold": 1,
                   "concentrated_if": {"fold_fa_gt": 0.2, "rest_fa_le": 0.12}, "spread_if": {"rest_fa_gt": 0.12}},
            "d10": {"reps": 2000, "seed": 0, "m": [3, 4], "alphas": [0.5], "distributions": ["uniform"],
                    "tolerance": 0.05, "other_distribution_tolerance": 0.05},
            "d11": {"n": 8, "subsets": 50, "subset_size": 2, "seed": 0},
            "d12": {"B": 40, "seed": 0}}


@pytest.fixture(scope="module")
def diag(e3_run):  # noqa: F811
    _, cfg, _, result, frame, _, _ = e3_run
    groups = group_frame(GROUP_RECORD)
    # Round-trip the result as the runner writes it (NaN -> None), and the frame through CSV.
    e3 = json.loads(json.dumps(result, default=lambda o: o.item() if hasattr(o, "item") else list(o)).replace("NaN", "null"))
    out = run_diagnostics(groups, frame, e3, _diag_cfg(), cfg, evaluate_expectations=False)
    return groups, frame, out


def test_group_counts_reproduce_every_e3_headline_cell(diag):
    _, _, out = diag
    for fam in out["families"].values():
        assert fam["reproduction"]["ok"], fam["reproduction"]
        assert fam["tpr_post_le_tpr_violations"] == 0


def test_common_test_set_is_e3_itself_at_the_largest_n(diag):
    _, _, out = diag
    d8 = out["families"]["llama"]["d8"]
    assert any(d8["dropped_folds"].values())  # the synthetic run skips folds at n = 14
    for r in d8["rows"]:
        if r["n"] == 14 and r["e3_fa_mean"] is not None:
            assert r["common_fa_mean"] == pytest.approx(r["e3_fa_mean"], abs=1e-12)
            assert r["common_n_test_healthy"] == r["e3_n_test_healthy"]


def test_bootstrap_point_equals_the_unweighted_statistic(diag):
    groups, frame, out = diag
    llama = FamilyCounts(groups[groups["family"] == "llama"], frame[frame["family"] == "llama"], [3, 4])
    c = llama.cell("TL-pct", 8, "conformal", 0.5)
    ones = llama.ones()
    as_column = {t: w[:, None] for t, w in ones.items()}
    for fields in (FA, TPR):
        np.testing.assert_array_equal(llama.headline(c, fields, as_column)[:, 0], llama.headline(c, fields, ones))
    st = out["families"]["llama"]["d12"]["stats"]["TL-pct.n8.conformal.fa_mean"]
    d8 = {(r["arm"], r["n"]): r for r in out["families"]["llama"]["d8"]["rows"]}
    assert st["point"] == pytest.approx(d8[("TL-pct", 8)]["e3_fa_mean"], abs=1e-12)


def test_fold_fa_pools_draws_and_folds_partition_the_test_set(diag):
    groups, frame, _ = diag
    llama = FamilyCounts(groups[groups["family"] == "llama"], frame[frame["family"] == "llama"], [3, 4])
    folds = sorted(set(llama.fold_of[llama.fold_of >= 0].tolist()))
    c = llama.cell("TL-pct", 8, "conformal", 0.5)
    ones = llama.ones()
    for t in (3, 4):
        whole = c.counts[t]["n_healthy"] @ ones[t]
        by_fold = sum(c.counts[t]["n_healthy"] @ llama.only_folds(ones, [f])[t] for f in folds)
        np.testing.assert_array_equal(whole, by_fold)


def test_a_group_in_two_folds_is_rejected():
    groups = pd.DataFrame({"family": "x", "arm": "A", "n": 4, "rule": "conformal", "alpha": 0.5, "t": 3, "draw": [0, 0],
                           "fold": [0, 1], "task_group": "g", "n_healthy": 1, "n_healthy_alarm": 0, "n_pos": 0,
                           "n_pos_alarm": 0, "n_pos_alarm_pre_tau": 0, "n_pos_alarm_post": 0})
    draws = pd.DataFrame({"family": "x", "arm": "A", "n": 4, "rule": "conformal", "alpha": 0.5, "t": [3], "draw": [0],
                          "undefined": [False], "m": [4]})
    with pytest.raises(ValueError, match="more than one fold"):
        FamilyCounts(groups, draws, [3])
