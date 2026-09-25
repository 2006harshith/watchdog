import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from watchdog_agent.metrics import (
    auprc,
    auroc,
    bootstrap_ci,
    ece,
    paired_bootstrap_diff,
    tpr_at_fpr,
)


def _random_case(seed: int, n: int, ties: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    y[0], y[1] = 0, 1  # guarantee both classes
    s = rng.normal(size=n) + 0.8 * y
    if ties:
        s = np.round(s, 1)  # coarse rounding creates many tied scores across classes
    return y, s


CASES = [(seed, n, ties) for seed in range(5) for n in (10, 57, 300) for ties in (False, True)]


@pytest.mark.parametrize("seed,n,ties", CASES)
def test_auroc_matches_sklearn(seed, n, ties):
    y, s = _random_case(seed, n, ties)
    assert auroc(y, s) == pytest.approx(roc_auc_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed,n,ties", CASES)
def test_auprc_matches_sklearn(seed, n, ties):
    y, s = _random_case(seed, n, ties)
    assert auprc(y, s) == pytest.approx(average_precision_score(y, s), abs=1e-12)


@pytest.mark.parametrize("seed,n,ties", CASES)
@pytest.mark.parametrize("alpha", [0.0, 0.05, 0.2])
def test_tpr_at_fpr_matches_sklearn_roc_points(seed, n, ties, alpha):
    y, s = _random_case(seed, n, ties)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    expected = tpr[fpr <= alpha].max()
    assert tpr_at_fpr(y, s, alpha) == pytest.approx(expected, abs=1e-12)


def test_auroc_hand_case():
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.7, 0.4, 0.6, 0.3, 0.2])
    assert auroc(y, s) == pytest.approx(8 / 9)


def test_auroc_ties_count_half():
    # one failed and one healthy run at the same score: the pair counts 1/2
    assert auroc(np.array([1, 0]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def test_tpr_at_fpr_hand_case():
    # healthy scores 0.6/0.3/0.2: FPR 0 allows threshold above 0.6 -> TPR 2/3 (0.9, 0.7)
    y = np.array([1, 1, 1, 0, 0, 0])
    s = np.array([0.9, 0.7, 0.4, 0.6, 0.3, 0.2])
    assert tpr_at_fpr(y, s, alpha=0.05) == pytest.approx(2 / 3)


def test_ece_hand_case():
    y = np.array([1] * 30 + [0] * 20 + [1] * 10 + [0] * 40)
    p = np.array([0.8] * 50 + [0.2] * 50)
    assert ece(y, p) == pytest.approx(0.10)


def test_ece_puts_p_equal_one_in_last_bin():
    assert ece(np.array([1, 1]), np.array([1.0, 1.0])) == pytest.approx(0.0)


def test_ece_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError):
        ece(np.array([0, 1]), np.array([0.5, 1.2]))


def test_single_class_raises():
    with pytest.raises(ValueError):
        auroc(np.array([1, 1]), np.array([0.1, 0.2]))


def test_bootstrap_same_seed_same_ci():
    y, s = _random_case(0, 80, ties=False)
    units = np.arange(len(y))
    a = bootstrap_ci(auroc, y, s, units, n=500, seed=7)
    b = bootstrap_ci(auroc, y, s, units, n=500, seed=7)
    assert a == b


def test_bootstrap_point_is_full_sample_metric_and_inside_ci():
    y, s = _random_case(1, 120, ties=False)
    ci = bootstrap_ci(auroc, y, s, np.arange(len(y)), n=1000, seed=0)
    assert ci.point == pytest.approx(auroc(y, s))
    assert ci.lo <= ci.point <= ci.hi


def test_bootstrap_resamples_whole_units():
    sizes = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6}
    unit_ids = np.concatenate([[u] * k for u, k in sizes.items()])
    y = np.concatenate([[0, 1] + [0] * (k - 2) for k in sizes.values()])  # every unit has both classes
    s = unit_ids.astype(float)  # score encodes the unit, so the metric can see which rows came together

    def whole_units_only(_y, s_resampled):
        values, counts = np.unique(s_resampled, return_counts=True)
        for v, c in zip(values, counts):
            assert c % sizes[int(v)] == 0, f"unit {int(v)} was split: {c} rows"
        return 0.0

    ci = bootstrap_ci(whole_units_only, y, s, unit_ids, n=300, seed=0)
    assert ci.n_skipped == 0


def test_bootstrap_raises_when_too_many_one_class_resamples():
    # two single-row units, one per class: half of all resamples have one class
    with pytest.raises(ValueError, match="one class"):
        bootstrap_ci(auroc, np.array([0, 1]), np.array([0.1, 0.9]), np.array([0, 1]), n=200, seed=0)


def test_paired_diff_identical_scores_is_exactly_zero():
    y, s = _random_case(2, 60, ties=True)
    d = paired_bootstrap_diff(auroc, y, s, s.copy(), np.arange(len(y)), n=500, seed=0)
    assert (d.point, d.lo, d.hi) == (0.0, 0.0, 0.0)


def test_paired_diff_sign_is_b_minus_a():
    y, s = _random_case(3, 200, ties=False)
    noise = np.random.default_rng(0).normal(size=len(y))
    d = paired_bootstrap_diff(auroc, y, noise, s, np.arange(len(y)), n=500, seed=0)
    assert d.point == pytest.approx(auroc(y, s) - auroc(y, noise))
    assert d.point > 0


def test_paired_diff_same_seed_same_result():
    y, s = _random_case(4, 90, ties=False)
    s_b = s + np.random.default_rng(1).normal(size=len(y))
    units = np.arange(len(y)) // 3  # 3-run clusters
    a = paired_bootstrap_diff(auprc, y, s, s_b, units, n=400, seed=11)
    b = paired_bootstrap_diff(auprc, y, s, s_b, units, n=400, seed=11)
    assert a == b
