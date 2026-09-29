import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from watchdog_agent.evaluation import (
    matched_step_auroc,
    matched_step_cluster_ci,
    paired_matched_step_diff,
    prefix_eval_set,
)


def _runs(rows):
    """rows: (uid, T, failure_class, tau, task_group)."""
    return pd.DataFrame.from_records(rows, columns=["uid", "T", "failure_class", "tau", "task_group"])


def _mixed_runs():
    return _runs(
        [
            ("h1", 5, None, np.nan, "g1"),
            ("h2", 6, None, np.nan, "g2"),
            ("h3", 4, None, np.nan, "g3"),
            ("h4", 3, None, np.nan, "g4"),  # no step 3
            ("f1", 5, "looping", 2, "g1"),
            ("f2", 6, "goal_drift", 3, "g2"),
            ("f3", 6, "tool_cascade", 4, "g3"),  # tau > 3: excluded at t = 3
            ("f4", 4, "looping", 2, "g5"),
        ]
    )


def test_prefix_eval_set_labels_per_d8():
    s = prefix_eval_set(_mixed_runs(), t=3)
    assert s.t == 3
    assert dict(zip(s.uids, s.y.tolist())) == {"h1": 0, "h2": 0, "h3": 0, "f1": 1, "f2": 1, "f4": 1}
    assert (s.n_pos, s.n_neg, s.n_excluded, s.n_without_step) == (3, 3, 1, 1)


def test_failed_run_with_tau_after_t_is_excluded():
    runs = _mixed_runs()
    assert "f3" not in prefix_eval_set(runs, t=3).uids
    assert "f3" in prefix_eval_set(runs, t=4).uids  # tau == t counts as failed already


def test_run_shorter_than_t_plus_one_steps_is_excluded():
    runs = _mixed_runs()
    assert "h4" in prefix_eval_set(runs, t=2).uids  # T = 3 has steps 0, 1, 2
    assert "h4" not in prefix_eval_set(runs, t=3).uids


def test_score_equal_to_prefix_length_gives_exactly_half_at_every_checkpoint():
    runs = _mixed_runs()
    scores = {uid: [k + 1 for k in range(T)] for uid, T in zip(runs["uid"], runs["T"])}
    res = matched_step_auroc(scores, runs, checkpoints=[2, 3, 4])
    assert res.per_checkpoint == {2: 0.5, 3: 0.5, 4: 0.5}
    assert res.mean == 0.5


def test_uses_the_score_at_step_t_not_a_later_step():
    runs = _runs([("h", 5, None, np.nan, "g1"), ("f", 5, "looping", 2, "g2")])
    scores = {"h": [0, 0, 0, 5, 0], "f": [0, 0, 0, 0, 9]}
    res = matched_step_auroc(scores, runs, checkpoints=[3, 4])
    assert res.per_checkpoint == {3: 0.0, 4: 1.0}


def test_hand_computed_auroc_on_six_run_toy_set_matches_sklearn():
    runs = _runs(
        [
            ("p1", 4, "looping", 2, "g1"),
            ("p2", 4, "looping", 2, "g2"),
            ("p3", 4, "looping", 2, "g3"),
            ("n1", 4, None, np.nan, "g4"),
            ("n2", 4, None, np.nan, "g5"),
            ("n3", 4, None, np.nan, "g6"),
        ]
    )
    at_t = {"p1": 0.9, "p2": 0.4, "p3": 0.6, "n1": 0.5, "n2": 0.2, "n3": 0.6}
    scores = {uid: [0.0, 0.0, 0.0, s] for uid, s in at_t.items()}
    # 9 (failed, healthy) pairs: p1 beats all 3; p2 beats n2; p3 beats n1, n2 and ties n3.
    expected = (3 + 1 + 2 + 0.5) / 9
    res = matched_step_auroc(scores, runs, checkpoints=[3])
    assert res.per_checkpoint[3] == pytest.approx(expected)
    y = [1, 1, 1, 0, 0, 0]
    assert res.per_checkpoint[3] == pytest.approx(roc_auc_score(y, list(at_t.values())))


def test_missing_score_at_step_t_raises():
    runs = _runs([("h", 5, None, np.nan, "g1"), ("f", 5, "looping", 2, "g2")])
    with pytest.raises(ValueError, match=r"step 3 for runs \['h'\]"):
        matched_step_auroc({"h": [0, 0, 0], "f": [0, 0, 0, 0, 1]}, runs, checkpoints=[3])


def _bootstrap_runs():
    rows = []
    for g in range(12):
        rows.append((f"h{g}", 6, None, np.nan, f"g{g}"))
        rows.append((f"f{g}", 6, "looping", 2, f"g{g}"))
    return _runs(rows)


def _noisy_scores(runs, seed):
    rng = np.random.default_rng(seed)
    return {uid: rng.normal(size=6) + (1.0 if uid.startswith("f") else 0.0) for uid in runs["uid"]}


def test_paired_difference_of_identical_monitors_is_zero():
    runs = _bootstrap_runs()
    scores = _noisy_scores(runs, seed=0)
    res = paired_matched_step_diff(scores, scores, runs, checkpoints=[2, 3, 4], n=200, seed=1)
    for interval in [*res.per_checkpoint.values(), res.mean]:
        assert (interval.point, interval.lo, interval.hi) == (0.0, 0.0, 0.0)


def test_one_task_group_draw_is_shared_by_every_checkpoint():
    # Same runs and same scores at t = 2 and t = 3: with one shared draw per replicate the two
    # bootstrap distributions are identical; independent draws per checkpoint would differ.
    runs = _bootstrap_runs()
    scores = {uid: np.r_[s[:3], s[2], s[4:]] for uid, s in _noisy_scores(runs, seed=0).items()}
    res = matched_step_cluster_ci(scores, runs, checkpoints=[2, 3], n=200, seed=1)
    assert res.per_checkpoint[2] == res.per_checkpoint[3]
    assert res.per_checkpoint[2].lo < res.per_checkpoint[2].hi


def test_bootstrap_resamples_task_groups_not_runs():
    # One task group holds every run: each replicate redraws that one group, so the CI collapses
    # to the point estimate. A run-level bootstrap would give a non-degenerate interval.
    runs = _bootstrap_runs().assign(task_group="only")
    scores = _noisy_scores(runs, seed=0)
    res = matched_step_cluster_ci(scores, runs, checkpoints=[2, 3], n=50, seed=1)
    for interval in [*res.per_checkpoint.values(), res.mean]:
        assert interval.lo == interval.point == interval.hi


def test_cluster_ci_point_equals_matched_step_auroc_and_is_seeded():
    runs = _bootstrap_runs()
    scores = _noisy_scores(runs, seed=0)
    point = matched_step_auroc(scores, runs, checkpoints=[2, 3, 4])
    a = matched_step_cluster_ci(scores, runs, checkpoints=[2, 3, 4], n=200, seed=5)
    b = matched_step_cluster_ci(scores, runs, checkpoints=[2, 3, 4], n=200, seed=5)
    assert a == b
    assert {t: iv.point for t, iv in a.per_checkpoint.items()} == point.per_checkpoint
    assert a.mean.point == point.mean
    for interval in [*a.per_checkpoint.values(), a.mean]:
        assert interval.lo <= interval.point <= interval.hi
