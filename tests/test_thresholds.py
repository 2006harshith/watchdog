import math

import numpy as np
import pandas as pd
import pytest

from watchdog_agent.thresholds import (
    alarms_by_t,
    calibration_score,
    conformal_threshold,
    fa_tpr,
    first_alarm_steps,
    plugin_threshold,
    running_max,
)

HEADLINE = ["looping", "goal_drift", "context_corruption"]


def _shuffled(m, seed=0):
    # 1..m in random order: the k-th smallest is k, so every expected theta is readable by hand.
    return np.random.default_rng(seed).permutation(np.arange(1.0, m + 1))


# ---------------------------------------------------------------- conformal, hand cases


@pytest.mark.parametrize(
    "m, alpha, expected",
    [
        (5, 0.10, math.inf),   # ceil(6 * 0.90) = ceil(5.4) = 6 > 5
        (5, 0.05, math.inf),   # ceil(6 * 0.95) = ceil(5.7) = 6 > 5
        (9, 0.10, 9.0),        # ceil(10 * 0.90) = 9: the largest of 9
        (9, 0.05, math.inf),   # ceil(10 * 0.95) = ceil(9.5) = 10 > 9
        (19, 0.10, 18.0),      # ceil(20 * 0.90) = 18
        (19, 0.05, 19.0),      # ceil(20 * 0.95) = 19: the largest of 19
    ],
)
def test_conformal_hand_cases(m, alpha, expected):
    assert conformal_threshold(_shuffled(m), alpha) == expected


@pytest.mark.parametrize("alpha, min_m", [(0.10, 9), (0.05, 19)])
def test_conformal_is_inf_below_min_m_and_finite_from_it(alpha, min_m):
    for m in range(1, min_m):
        assert conformal_threshold(_shuffled(m), alpha) == math.inf, m
    assert math.isfinite(conformal_threshold(_shuffled(min_m), alpha))


def test_conformal_index_when_m_plus_1_times_1_minus_alpha_is_an_integer():
    # The boundary case of the ceil: k must equal the integer, not one above it (float 1 - 0.05 is
    # 0.94999...; checked for m < 5000 at the alphas used here, plain floats give the same k).
    for m, alpha in [(19, 0.05), (39, 0.05), (99, 0.05), (9, 0.10), (29, 0.10)]:
        k = round((m + 1) * (1 - alpha))
        assert conformal_threshold(_shuffled(m), alpha) == float(k)


def test_conformal_empty_calibration_is_inf():
    assert conformal_threshold(np.array([]), 0.10) == math.inf


@pytest.mark.parametrize("bad", [[1.0, np.nan], [1.0, np.inf]])
def test_thresholds_reject_non_finite_scores(bad):
    with pytest.raises(ValueError):
        conformal_threshold(np.array(bad), 0.10)
    with pytest.raises(ValueError):
        plugin_threshold(np.array(bad), 0.10)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_thresholds_reject_alpha_outside_unit_interval(alpha):
    with pytest.raises(ValueError):
        conformal_threshold(_shuffled(10), alpha)


# ---------------------------------------------------------------- plug-in, hand cases


@pytest.mark.parametrize(
    "m, alpha, expected",
    [
        # method="linear": position (m-1)q between the sorted values, linear between neighbours.
        (5, 0.10, 4.6),    # 4 * 0.90 = 3.6  -> 4 + 0.6 * (5 - 4)
        (9, 0.10, 8.2),    # 8 * 0.90 = 7.2  -> 8 + 0.2
        (19, 0.10, 17.2),  # 18 * 0.90 = 16.2 -> 17 + 0.2
        (19, 0.05, 18.1),  # 18 * 0.95 = 17.1 -> 18 + 0.1
    ],
)
def test_plugin_hand_cases(m, alpha, expected):
    assert plugin_threshold(_shuffled(m), alpha) == pytest.approx(expected, abs=1e-12)


def test_plugin_is_finite_where_conformal_is_inf():
    assert math.isfinite(plugin_threshold(_shuffled(5), 0.10))


def test_plugin_empty_calibration_raises():
    with pytest.raises(ValueError):
        plugin_threshold(np.array([]), 0.10)


# ---------------------------------------------------------------- simulation


def _mean_fa(rule, m, alpha, reps=2000, n_test=200, seed=0, sampler=None):
    rng = np.random.default_rng(seed)
    sampler = sampler or (lambda size: rng.uniform(size=size))
    fa = np.empty(reps)
    for r in range(reps):
        theta = rule(sampler(m), alpha)
        fa[r] = (sampler(n_test) > theta).mean()
    # Monte Carlo standard error of the mean over reps (covers theta's spread and test sampling).
    return fa.mean(), fa.std(ddof=1) / math.sqrt(reps)


@pytest.mark.parametrize("m, alpha", [(9, 0.10), (19, 0.10), (19, 0.05), (40, 0.10)])
def test_conformal_mean_fa_within_alpha_on_exchangeable_sets(m, alpha):
    mean, se = _mean_fa(conformal_threshold, m, alpha)
    assert mean <= alpha + 3 * se


def test_plugin_at_m10_exceeds_alpha():
    # Expected FA ~0.17 for uniforms (theta ~ E[U_(9)] + 0.1 (E[U_(10)] - E[U_(9)]) = 0.827).
    mean, se = _mean_fa(plugin_threshold, 10, 0.10)
    assert mean > 0.10 + 3 * se


# ---------------------------------------------------------------- ties


def test_tie_at_theta_does_not_alarm():
    theta = conformal_threshold(np.ones(19), 0.10)
    assert theta == 1.0
    assert alarms_by_t({"a": [1.0, 1.0, 1.0]}, theta, t=2) == {"a": False}
    assert alarms_by_t({"b": [1.0, 1.0 + 1e-9, 1.0]}, theta, t=2) == {"b": True}


def test_strict_rule_keeps_conformal_valid_with_ties_and_ge_would_not():
    # A discrete score (95% of healthy runs score exactly 0, like an ESN still in washout): theta is
    # usually 0. With ">" FA = P(score > 0) = 0.05; with ">=" every 0 would alarm, FA ~ 1.
    rng = np.random.default_rng(1)

    def sampler(size):
        return (rng.uniform(size=size) < 0.05).astype(float)

    mean, se = _mean_fa(conformal_threshold, 19, 0.10, sampler=sampler, seed=1)
    assert mean <= 0.10 + 3 * se
    theta = conformal_threshold(np.zeros(19), 0.10)
    assert (np.zeros(5) >= theta).mean() == 1.0  # what ">=" would do


# ---------------------------------------------------------------- running max, alarms


def test_running_max_skips_unscored_steps():
    # HGB arms score only checkpoints 2..4: NaN elsewhere must not poison the running max.
    assert running_max([np.nan, np.nan, 0.2, 0.1, 0.5]).tolist() == [-np.inf, -np.inf, 0.2, 0.2, 0.5]


def test_calibration_score_is_running_max_up_to_t():
    s = [np.nan, np.nan, 0.7, 0.1, 0.3]
    assert calibration_score(s, 3) == 0.7
    assert calibration_score(s, 4) == 0.7


def test_calibration_score_needs_a_score_at_t():
    with pytest.raises(ValueError):
        calibration_score([np.nan, np.nan, 0.7], 3)  # no step 3
    with pytest.raises(ValueError):
        calibration_score([0.1, 0.2, np.nan, 0.3], 2)  # step 2 present but unscored


def test_alarms_use_only_steps_up_to_t():
    scores = {"late": [np.nan, np.nan, 0.1, 0.1, 0.9], "early": [np.nan, np.nan, 0.9, 0.1, 0.1]}
    assert alarms_by_t(scores, 0.5, t=3) == {"late": False, "early": True}
    assert alarms_by_t(scores, 0.5, t=4) == {"late": True, "early": True}


def test_inf_theta_never_alarms():
    assert alarms_by_t({"a": [1e300, 1e308]}, math.inf, t=1) == {"a": False}


def test_first_alarm_steps():
    scores = {"a": [np.nan, np.nan, 0.1, 0.9, 0.9], "b": [0.0, 0.0, 0.1], "c": [0.0, 0.6]}
    assert first_alarm_steps(scores, 0.5) == {"a": 3, "b": None, "c": 1}


# ---------------------------------------------------------------- FA, TPR, early alarms


def _runs():
    return pd.DataFrame([
        # healthy: h1 alarms; h2 does not; h3 has no step 3 (T=3) and alarms -> not in FA(3)
        {"uid": "h1", "T": 5, "failure_class": None, "tau": np.nan},
        {"uid": "h2", "T": 5, "failure_class": None, "tau": np.nan},
        {"uid": "h3", "T": 3, "failure_class": None, "tau": np.nan},
        # headline positives at t=3: p1 alarms at tau, p2 alarms before tau (early), p3 never
        {"uid": "p1", "T": 6, "failure_class": "looping", "tau": 3},
        {"uid": "p2", "T": 6, "failure_class": "goal_drift", "tau": 3},
        {"uid": "p3", "T": 6, "failure_class": "context_corruption", "tau": 2},
        # not counted: tau > t (D8), and an error-visible class outside `classes`
        {"uid": "x1", "T": 6, "failure_class": "looping", "tau": 5},
        {"uid": "x2", "T": 6, "failure_class": "tool_cascade", "tau": 2},
    ])


FIRST_ALARM = {"h1": 2, "h2": None, "h3": 1, "p1": 3, "p2": 2, "p3": None, "x1": 2, "x2": 2}


def test_fa_tpr_counts():
    r = fa_tpr(FIRST_ALARM, _runs(), t=3, classes=HEADLINE)
    assert (r.n_healthy, r.n_healthy_alarm) == (2, 1)  # h3 excluded: no step 3
    assert r.fa == 0.5
    assert (r.n_pos, r.n_pos_alarm) == (3, 2)  # p1, p2 alarm; p3 does not; x1, x2 not positives
    assert r.tpr == pytest.approx(2 / 3)
    assert r.n_early == 1  # p2's first alarm (step 2) is before its tau (3)
    assert r.early_alarm_rate == 0.5  # 1 of the 2 alarming positives
    assert r.n_healthy_without_step == 1
    assert r.n_excluded_tau_after_t == 1


def test_fa_tpr_alarm_after_t_does_not_count():
    first = {**FIRST_ALARM, "h1": 4, "p1": 4}
    r = fa_tpr(first, _runs(), t=3, classes=HEADLINE)
    assert (r.n_healthy_alarm, r.n_pos_alarm) == (0, 1)


def test_fa_tpr_rates_are_nan_on_empty_sets():
    r = fa_tpr({"p1": None}, _runs()[_runs()["uid"] == "p1"], t=3, classes=HEADLINE)
    assert r.n_healthy == 0 and math.isnan(r.fa)
    assert r.n_pos_alarm == 0 and math.isnan(r.early_alarm_rate)


def test_fa_tpr_raises_on_a_run_without_an_alarm_entry():
    with pytest.raises(ValueError):
        fa_tpr({"h1": None}, _runs(), t=3, classes=HEADLINE)
