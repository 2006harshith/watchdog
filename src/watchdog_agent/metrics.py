"""SP3 core piece: episode-level metrics and unit-level bootstrap CIs.

Convention everywhere: y = 1 for a failed run, 0 for a healthy run; higher
score = more likely failed. Each metric is checked against scikit-learn in
tests/test_metrics.py; scikit-learn is a test-only dependency.
"""

from collections.abc import Callable
from typing import NamedTuple

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]

# Above this share of one-class resamples the CI describes a different
# population than the point estimate, so it is refused rather than reported.
MAX_SKIPPED_FRACTION = 0.01


class Interval(NamedTuple):
    point: float
    lo: float
    hi: float
    n_skipped: int


def _check_binary(
    y: np.ndarray, s: np.ndarray, need_both_classes: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"y and s must be 1-D with equal length, got {y.shape} and {s.shape}")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y must contain only 0 and 1")
    if not np.isfinite(s).all():
        raise ValueError("scores must be finite")
    if need_both_classes and np.unique(y).size < 2:
        raise ValueError("y must contain both classes")
    return y.astype(int), s


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """P(score of a random failed run > score of a random healthy run), ties 1/2.

    Mann-Whitney form: sum the ranks of the failed runs, subtract the smallest
    possible rank sum, divide by the number of (failed, healthy) pairs.
    """
    y, s = _check_binary(y, s)
    # np.unique(..., return_inverse, return_counts): sorted distinct scores, each
    # row's position among them, and how many rows share each score. A tie group
    # occupying ranks (end-count+1 .. end) gets their mean, end - (count-1)/2,
    # which is exactly what makes a tied (failed, healthy) pair count 1/2.
    _, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2.0)[inverse]
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _counts_at_thresholds(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative true/false positives with the threshold at each distinct score,
    from the highest score down (the ROC and PR curves' operating points).
    """
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    # Tied scores cannot be separated by any threshold, so only the last row of
    # each tie group is an operating point.
    last_of_group = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tps = np.cumsum(y_sorted)[last_of_group]
    fps = (last_of_group + 1) - tps
    return tps, fps


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision: sum over operating points of (recall gain) x precision.

    Step-wise, not trapezoidal: linear interpolation between PR points is
    optimistic because precision is not linear in recall.
    """
    y, s = _check_binary(y, s)
    tps, fps = _counts_at_thresholds(y, s)
    recall = tps / tps[-1]
    precision = tps / (tps + fps)
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, alpha: float = 0.05) -> float:
    """Highest TPR among ROC operating points with FPR <= alpha, no interpolation.

    This reads the ROC curve of the evaluated runs themselves: the threshold is
    chosen with their labels, so it is an operating point, not a deployable
    threshold (SP6 sets thresholds from healthy runs only).
    """
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must be in [0, 1), got {alpha}")
    y, s = _check_binary(y, s)
    tps, fps = _counts_at_thresholds(y, s)
    tpr = np.r_[0.0, tps / tps[-1]]  # r_ prepends the "flag nothing" point (0, 0)
    fpr = np.r_[0.0, fps / fps[-1]]
    return float(tpr[fpr <= alpha].max())


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1], weighted by bin size.

    sum_b (n_b / N) * |mean(y_b) - mean(p_b)| simplifies to
    sum_b |sum(y_b) - sum(p_b)| / N, so no per-bin means are needed.
    """
    y, p = _check_binary(y, p, need_both_classes=False)
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("probabilities must be in [0, 1]")
    # p == 1.0 would land in bin n_bins; it belongs in the last bin.
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    # np.bincount(bins, weights=w): per-bin sum of w in one vectorised pass.
    sum_y = np.bincount(bins, weights=y, minlength=n_bins)
    sum_p = np.bincount(bins, weights=p, minlength=n_bins)
    return float(np.abs(sum_y - sum_p).sum() / y.size)


def _unit_rows(unit_ids: np.ndarray) -> list[np.ndarray]:
    """Row indices of each unit (run or task group), one array per unit."""
    _, inverse, counts = np.unique(unit_ids, return_inverse=True, return_counts=True)
    # Stable argsort of the unit index lists rows unit by unit; np.split at the
    # cumulative counts then cuts that list into one block per unit.
    order = np.argsort(inverse, kind="stable")
    return np.split(order, np.cumsum(counts)[:-1])


def _bootstrap(
    stat: Callable[[np.ndarray], float], y: np.ndarray, unit_ids: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, int]:
    """Resample whole units with replacement n times; return stat per usable resample."""
    if n < 1:
        raise ValueError("n must be >= 1")
    unit_ids = np.asarray(unit_ids)
    if unit_ids.shape != y.shape:
        raise ValueError("unit_ids must have one entry per row")
    groups = _unit_rows(unit_ids)
    # default_rng(seed): a local, seeded Generator, so the CI depends only on
    # `seed` and never on (or changes) global NumPy random state.
    rng = np.random.default_rng(seed)
    stats, skipped = [], 0
    for _ in range(n):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[k] for k in picked])
        # A resample with one class has no AUROC; skipping it (and counting it)
        # beats imputing a value that would narrow the interval.
        if np.unique(y[idx]).size < 2:
            skipped += 1
            continue
        stats.append(stat(idx))
    if skipped > MAX_SKIPPED_FRACTION * n:
        raise ValueError(
            f"{skipped}/{n} bootstrap resamples had only one class (> {MAX_SKIPPED_FRACTION:.0%}); "
            "too few units of one class for a unit-level CI"
        )
    return np.asarray(stats), skipped


def bootstrap_ci(
    metric: Metric, y: np.ndarray, s: np.ndarray, unit_ids: np.ndarray, n: int = 10_000, *, seed: int
) -> Interval:
    """Percentile 95% CI of metric(y, s), resampling whole units (runs or task groups).

    Resampling units rather than rows keeps correlated rows (steps of one run,
    runs of one task) together, so the CI reflects the number of independent
    units, not the number of rows.
    """
    y, s = _check_binary(y, s)
    stats, skipped = _bootstrap(lambda idx: metric(y[idx], s[idx]), y, unit_ids, n, seed)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return Interval(float(metric(y, s)), float(lo), float(hi), skipped)


def paired_bootstrap_diff(
    metric: Metric,
    y: np.ndarray,
    s_a: np.ndarray,
    s_b: np.ndarray,
    unit_ids: np.ndarray,
    n: int = 10_000,
    *,
    seed: int,
) -> Interval:
    """Point difference metric(B) - metric(A) and its percentile 95% CI.

    Both monitors are scored on the SAME resampled units each time, so the
    run-to-run difficulty they share cancels; two separate CIs would count it
    twice and look wider than the uncertainty in the difference.
    """
    y, s_a = _check_binary(y, s_a)
    _, s_b = _check_binary(y, s_b)
    stats, skipped = _bootstrap(
        lambda idx: metric(y[idx], s_b[idx]) - metric(y[idx], s_a[idx]), y, unit_ids, n, seed
    )
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return Interval(float(metric(y, s_b) - metric(y, s_a)), float(lo), float(hi), skipped)
