import math

import numpy as np
import pandas as pd
import pytest

from watchdog_agent.baselines.esn.monitor import RunScore
from watchdog_agent.evaluation import prefix_eval_set
from watchdog_agent.experiments.sp4_audit import (
    ceiling_counts,
    family_predictability_auroc,
    indistinguishable_positives,
    matched_step_esn_scores,
    standardised_running_max,
    tool_prefix_signature,
    univariate_auroc,
)


def step(name="a", args=None, result="ok", is_error=False, latency=1.0):
    return {
        "action": "tool_call",
        "latency_s": latency,
        "text": "whatever",
        "tool_events": [
            {"name": name, "args": {"x": 1} if args is None else args, "result": result, "is_error": is_error}
        ],
    }


def synth():
    return {"action": "synthesis", "latency_s": 1.0, "text": "done", "tool_events": []}


# ---- 4c ceiling ---------------------------------------------------------------------------------


def test_tool_prefix_signature_ignores_non_tool_fields_and_steps_after_t():
    a = [step(latency=1.0), step("b"), synth(), step("c")]
    b = [step(latency=9.0), step("b"), synth(), step("zzz")]
    assert tool_prefix_signature(a, 2) == tool_prefix_signature(b, 2)
    assert tool_prefix_signature(a, 3) != tool_prefix_signature(b, 3)
    assert tool_prefix_signature([step(result="x")], 0) != tool_prefix_signature([step(result="y")], 0)
    assert tool_prefix_signature([step(is_error=True)], 0) != tool_prefix_signature([step()], 0)


def _ceiling_runs():
    healthy = [step("a"), step("b"), step("c"), step("d")]
    rows = [
        ("h1", "g1", None, np.nan, healthy),
        ("f_twin", "g1", "context_corruption", 2, [step("a"), step("b"), step("c"), step("x")]),
        ("f_other_task", "g2", "context_corruption", 2, list(healthy)),  # same calls, other task
        ("f_diff", "g1", "looping", 2, [step("a"), step("b"), step("c", result="retry")]),
        ("f_late", "g1", "looping", 3, list(healthy)),  # tau > t: not a positive at t = 2
    ]
    return pd.DataFrame(
        [
            {"uid": u, "task_group": g, "failure_class": c, "tau": tau, "T": len(s), "steps_parsed": s}
            for u, g, c, tau, s in rows
        ]
    )


def test_indistinguishable_positives_returns_the_uids():
    assert indistinguishable_positives(_ceiling_runs(), t=2) == {"f_twin"}


def test_ceiling_counts_failed_positives_indistinguishable_from_a_healthy_run_of_the_same_task():
    out = ceiling_counts(_ceiling_runs(), t=2)
    assert out["n_pos"] == 3
    assert out["n_indistinguishable"] == 1  # f_twin only
    assert out["share"] == pytest.approx(1 / 3)
    assert out["by_class"]["context_corruption"] == {"n_pos": 2, "n_indistinguishable": 1}
    assert out["by_class"]["looping"] == {"n_pos": 1, "n_indistinguishable": 0}


# ---- 4b univariate --------------------------------------------------------------------------------


def test_univariate_auroc_ranks_nan_lowest_and_uses_the_row_at_t():
    runs = pd.DataFrame(
        {
            "uid": ["h1", "h2", "f1", "f2"],
            "T": [5, 5, 5, 5],
            "failure_class": [None, None, "looping", "looping"],
            "tau": [np.nan, np.nan, 2, 2],
            "task_group": ["g1", "g2", "g3", "g4"],
        }
    )
    idx = pd.MultiIndex.from_tuples([(u, t) for u in runs["uid"] for t in (3, 4)], names=["uid", "t"])
    table = pd.DataFrame({"feat": [0, 9, 1, 9, 5, 0, np.nan, 0]}, index=idx)
    cset = prefix_eval_set(runs, 3)
    # at t = 3: h1 0, h2 1, f1 5, f2 NaN (ranked lowest) -> pairs (f1 > h1, f1 > h2) = 2 of 4
    assert univariate_auroc(table, cset, "feat") == pytest.approx(0.5)
    table.loc[("f2", 3), "feat"] = 7
    assert univariate_auroc(table, cset, "feat") == pytest.approx(1.0)


# ---- 4a family predictability -------------------------------------------------------------------


def _family_frame(separable: bool, seed=0):
    rng = np.random.default_rng(seed)
    n = 200
    fam = np.repeat([0, 1], n // 2)
    x = rng.normal(size=(n, 3))
    if separable:
        x[:, 0] += 4 * fam
    x[rng.random(size=(n, 3)) < 0.1] = np.nan
    groups = np.array([f"g{i % 20}" for i in range(n)])
    return pd.DataFrame(x, columns=["f0", "f1", "f2"]), fam, groups


def test_family_predictability_high_when_a_feature_carries_the_family():
    X, fam, groups = _family_frame(separable=True)
    assert family_predictability_auroc(X, fam, groups, n_splits=5) > 0.9


def test_family_predictability_near_half_when_features_are_noise():
    X, fam, groups = _family_frame(separable=False)
    assert abs(family_predictability_auroc(X, fam, groups, n_splits=5) - 0.5) < 0.15


def test_family_predictability_keeps_task_groups_out_of_their_own_training_fold():
    # One-hot task identity, with family decided per group: a model that saw a group's own runs in
    # training predicts that group's family perfectly; with grouped folds it never has.
    n_groups = 20
    groups = np.repeat([f"g{i}" for i in range(n_groups)], 10)
    rng = np.random.default_rng(0)
    fam = np.repeat(rng.permutation(np.repeat([0, 1], n_groups // 2)), 10)
    X = pd.get_dummies(pd.Series(groups), dtype=float)
    assert abs(family_predictability_auroc(X, fam, groups, n_splits=5) - 0.5) < 0.25


# ---- 4d ESN matched-step scores -----------------------------------------------------------------


def test_standardised_running_max_uses_val_runs_with_a_step_t():
    stream = np.array([0.0, 0.0, 0.0, 2.0, 1.0, 5.0])
    val = [np.array([0.0, 0.0, 0.0, 1.0, 3.0]), np.array([0.0, 0.0, 0.0, 3.0, 3.0]), np.array([0.0, 0.0, 0.0])]
    out, flags = standardised_running_max(stream, val, checkpoints=[2, 3, 4])
    # t = 3: val running max {1, 3} -> mean 2, std 1; eval running max 2 -> 0
    assert out[3] == pytest.approx(0.0)
    # t = 4: val running max {3, 3} -> std 0: centred, not scaled, and flagged
    assert out[4] == pytest.approx(2.0 - 3.0)
    assert flags[4] == "constant_val_scores_centred_only"
    # t = 2: washout, every score is 0
    assert out[2] == 0.0
    assert math.isnan(out[0]) and math.isnan(out[5])


class _FakeMonitor:
    theta = 0.0

    def fit(self, fit_runs, val_runs):
        self.n_fit = len(fit_runs)

    def score(self, run):
        per_step = np.array([float(len(s["text"])) for s in run.steps])
        return RunScore(per_step, None, float(per_step.max()))


def test_matched_step_esn_scores_scores_every_eval_run_once_per_monitor():
    rows = []
    for g in range(6):
        for fam, corpus in (("qwen", "src"), ("llama", "tgt")):
            for i, cls in enumerate((None, None, "looping")):
                steps = [{"text": "x" * (k + g + i + 1), "tool_events": []} for k in range(5)]
                rows.append({"uid": f"{corpus}{g}{i}", "corpus": corpus, "family": fam, "task_group": f"g{g}",
                             "failure_class": cls, "tau": np.nan if cls is None else 2, "T": 5, "steps_parsed": steps})
    episodes = pd.DataFrame(rows)
    cfg = {"source_corpus": "src", "target_corpus": "tgt", "target_family": "llama", "k": 3,
           "fold_seed": 0, "val_fraction": 0.25}
    scores, info = matched_step_esn_scores(episodes, cfg, checkpoints=[3, 4], make_monitor=_FakeMonitor)
    target = set(episodes.loc[episodes["corpus"] == "tgt", "uid"])
    for name in ("A", "B"):
        assert set(scores[name]) == target
        assert all(np.isfinite(scores[name][u][3]) and np.isfinite(scores[name][u][4]) for u in target)
    assert len(info) == 3
