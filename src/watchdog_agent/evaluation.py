"""SP4 matched-step evaluation (metrics core piece).

Every run is scored at the same step t, with what the monitor outputs after seeing steps
0..t. An episode-level score (max over all steps) rewards long runs: in SP3 the ESN's episode
score tracked run length (Spearman 0.85), and length alone scored AUROC 0.59. At a matched
step every run in the set has the same prefix length, so length cannot rank them.

Labels per D8 (docs/decisions.md). AUROC comes from metrics.py; nothing is re-implemented.
`runs` is an episode-level frame with uid, T, failure_class, tau, task_group.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from watchdog_agent.metrics import MAX_SKIPPED_FRACTION, Interval, auroc

Scores = Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class CheckpointSet:
    t: int
    uids: list[str]
    y: np.ndarray  # 1 = failed with tau <= t, 0 = healthy
    task_groups: np.ndarray
    n_pos: int
    n_neg: int
    n_excluded: int  # failed with tau > t: the failure has not happened yet (D8)
    n_without_step: int  # fewer than t + 1 steps: no step t to score


@dataclass(frozen=True)
class MatchedStepAUROC:
    per_checkpoint: dict[int, float]
    mean: float
    sets: dict[int, CheckpointSet]


@dataclass(frozen=True)
class MatchedStepCI:
    per_checkpoint: dict[int, Interval]
    mean: Interval


def prefix_eval_set(runs: pd.DataFrame, t: int) -> CheckpointSet:
    has_step = runs["T"] >= t + 1
    failed = runs["failure_class"].notna()
    # tau is NaN on healthy runs and NaN compares False both ways, so each tau test is ANDed with
    # `failed`; a healthy run can never land in pos or excluded.
    pos = has_step & failed & (runs["tau"] <= t)
    neg = has_step & ~failed
    excluded = has_step & failed & (runs["tau"] > t)
    keep = pos | neg
    return CheckpointSet(
        t=t,
        uids=runs.loc[keep, "uid"].tolist(),
        y=pos[keep].to_numpy(dtype=int),
        task_groups=runs.loc[keep, "task_group"].to_numpy(),
        n_pos=int(pos.sum()),
        n_neg=int(neg.sum()),
        n_excluded=int(excluded.sum()),
        n_without_step=int((~has_step).sum()),
    )


def _scores_at(scores: Scores, cset: CheckpointSet) -> np.ndarray:
    short = [uid for uid in cset.uids if len(scores[uid]) <= cset.t]
    if short:
        # A run in the set has a step t by construction; a score stream without it means the
        # monitor and the data disagree about the run, which must not be papered over.
        raise ValueError(f"no score at step {cset.t} for runs {short[:5]}")
    return np.array([scores[uid][cset.t] for uid in cset.uids], dtype=float)


def matched_step_auroc(
    scores_by_run_step: Scores, runs: pd.DataFrame, checkpoints: Sequence[int]
) -> MatchedStepAUROC:
    """AUROC at each checkpoint t using score[uid][t], plus their unweighted mean."""
    sets = {t: prefix_eval_set(runs, t) for t in checkpoints}
    per = {t: auroc(s.y, _scores_at(scores_by_run_step, s)) for t, s in sets.items()}
    return MatchedStepAUROC(per, float(np.mean(list(per.values()))), sets)


def _rows_by_group(cset: CheckpointSet, group_index: dict[str, int]) -> list[np.ndarray]:
    """For each task group (global index), this checkpoint's row indices in that group."""
    rows: list[list[int]] = [[] for _ in group_index]
    for i, g in enumerate(cset.task_groups):
        rows[group_index[g]].append(i)
    return [np.asarray(r, dtype=int) for r in rows]


def _cluster_bootstrap(
    sets: dict[int, CheckpointSet],
    stat: Callable[[int, np.ndarray], float],
    n: int,
    seed: int,
) -> MatchedStepCI:
    """Percentile 95% CIs of stat(t, rows) per checkpoint and of their mean, resampling task groups.

    One draw of task groups per replicate, reused for every checkpoint. The checkpoint sets share
    runs (a run scored at t = 3 is also scored at t = 4), so their AUROCs are correlated; the mean
    over checkpoints is only as uncertain as that shared sample. Independent draws per checkpoint
    would treat the checkpoints as independent evidence and make the CI of the mean too narrow.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    groups = sorted({g for s in sets.values() for g in s.task_groups})
    group_index = {g: k for k, g in enumerate(groups)}
    rows = {t: _rows_by_group(s, group_index) for t, s in sets.items()}

    point = {t: stat(t, np.arange(len(s.uids))) for t, s in sets.items()}
    # default_rng(seed): a local, seeded Generator; the CI depends only on `seed`.
    rng = np.random.default_rng(seed)
    per_rep: dict[int, list[float]] = {t: [] for t in sets}
    means: list[float] = []
    skipped = 0
    for _ in range(n):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = {t: np.concatenate([rows[t][k] for k in picked]) for t in sets}
        # A one-class checkpoint has no AUROC, and then the mean over checkpoints is undefined
        # too, so the whole replicate is skipped (and counted), never imputed.
        if any(np.unique(sets[t].y[idx[t]]).size < 2 for t in sets):
            skipped += 1
            continue
        vals = {t: stat(t, idx[t]) for t in sets}
        for t, v in vals.items():
            per_rep[t].append(v)
        means.append(float(np.mean(list(vals.values()))))
    if skipped > MAX_SKIPPED_FRACTION * n:
        raise ValueError(
            f"{skipped}/{n} bootstrap replicates had a one-class checkpoint (> {MAX_SKIPPED_FRACTION:.0%}); "
            "too few task groups of one class for a cluster CI"
        )

    def interval(p: float, stats: list[float]) -> Interval:
        lo, hi = np.percentile(stats, [2.5, 97.5])
        return Interval(float(p), float(lo), float(hi), skipped)

    return MatchedStepCI(
        per_checkpoint={t: interval(point[t], per_rep[t]) for t in sets},
        mean=interval(float(np.mean(list(point.values()))), means),
    )


def matched_step_cluster_ci(
    scores_by_run_step: Scores, runs: pd.DataFrame, checkpoints: Sequence[int], n: int = 10_000, *, seed: int
) -> MatchedStepCI:
    """Task-group cluster-bootstrap CIs of the matched-step AUROCs and their mean."""
    sets = {t: prefix_eval_set(runs, t) for t in checkpoints}
    s = {t: _scores_at(scores_by_run_step, cs) for t, cs in sets.items()}
    return _cluster_bootstrap(sets, lambda t, idx: auroc(sets[t].y[idx], s[t][idx]), n, seed)


def paired_matched_step_diff(
    scores_a: Scores,
    scores_b: Scores,
    runs: pd.DataFrame,
    checkpoints: Sequence[int],
    n: int = 10_000,
    *,
    seed: int,
) -> MatchedStepCI:
    """AUROC(B) - AUROC(A) per checkpoint and for the mean, both monitors on the same resampled
    task groups each replicate, so the difficulty they share cancels (as metrics.paired_bootstrap_diff).
    """
    sets = {t: prefix_eval_set(runs, t) for t in checkpoints}
    sa = {t: _scores_at(scores_a, cs) for t, cs in sets.items()}
    sb = {t: _scores_at(scores_b, cs) for t, cs in sets.items()}

    def diff(t: int, idx: np.ndarray) -> float:
        y = sets[t].y[idx]
        return auroc(y, sb[t][idx]) - auroc(y, sa[t][idx])

    return _cluster_bootstrap(sets, diff, n, seed)
