"""SP6 core piece: alarm thresholds from m healthy calibration runs, and the alarm rates they give.

A monitor that only ranks runs (AUROC) is not yet a monitor: something has to decide when to
alarm. Here theta comes from healthy runs of the new model only, never from failure labels.

- Calibration score of a healthy run at checkpoint t: its running max over the scored steps <= t
  (the same statistic the alarm rule compares to theta, so FA(t) is what the rule controls).
- Conformal rule: the ceil((m+1)(1-alpha))-th smallest calibration score. If a new healthy run is
  exchangeable with the m calibration runs, its rank among the m+1 is uniform, so
  P(score > theta) <= 1 - k/(m+1) <= alpha. Exchangeable is the whole condition: calibration runs
  from other task groups need not be, which is what E-1 tests.
- Plug-in rule: the empirical (1-alpha) quantile. No guarantee; with few runs it sits too low
  (the top of a small sample underestimates the top of the distribution), so FA runs above alpha.

The alarm is strict (score > theta), so a tie at theta does not alarm. That is what keeps the
conformal bound when scores tie: with ">=", a score that is often exactly theta (an ESN still at
0 in washout, a constant TL percentile) would alarm on every such run.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

ScoreStream = Sequence[float] | np.ndarray  # per-step score, index = step; NaN = not scored there


def _check(cal_scores, alpha: float) -> np.ndarray:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.asarray(cal_scores, dtype=float).ravel()
    if not np.isfinite(s).all():
        # A NaN or inf calibration score means a run without a score at t got in, or a broken monitor.
        raise ValueError("calibration scores must be finite")
    return s


def conformal_threshold(cal_scores, alpha: float) -> float:
    """theta = the k-th smallest of m scores, k = ceil((m+1)(1-alpha)); +inf if k > m (no alarms).

    k > m happens when m < 1/alpha - 1 (alpha=0.10: m < 9; 0.05: m < 19): too few runs to certify
    the rate, and an infinite threshold says so instead of guessing.
    """
    s = _check(cal_scores, alpha)
    m = s.size
    k = math.ceil((m + 1) * (1 - alpha))
    if k > m:
        return math.inf
    # np.partition puts the k-th smallest at index k-1 without a full sort.
    return float(np.partition(s, k - 1)[k - 1])


def plugin_threshold(cal_scores, alpha: float) -> float:
    """theta = np.quantile(scores, 1 - alpha, method="linear"), numpy's default quantile: it
    interpolates linearly between the two sorted values around position (m-1)(1-alpha)."""
    s = _check(cal_scores, alpha)
    if s.size == 0:
        raise ValueError("plug-in threshold needs at least one calibration score")
    return float(np.quantile(s, 1 - alpha, method="linear"))


def running_max(scores: ScoreStream) -> np.ndarray:
    """Running max over the scored steps; unscored steps (NaN) are -inf, so they never win the
    max and never alarm. HGB arms score only checkpoints 2..4, the ESN every step."""
    s = np.asarray(scores, dtype=float)
    # np.maximum.accumulate: element i = max of elements 0..i (the running max). NaN would
    # propagate through every later step, hence -inf first.
    return np.maximum.accumulate(np.where(np.isnan(s), -np.inf, s))


def calibration_score(scores: ScoreStream, t: int) -> float:
    """A calibration run's score at checkpoint t. The run must have a scored step t: a run that
    stops earlier is not a healthy run "at t" and would deflate theta."""
    s = np.asarray(scores, dtype=float)
    if s.size <= t or np.isnan(s[t]):
        raise ValueError(f"no score at step {t}; only runs with a scored step t calibrate theta at t")
    return float(running_max(s)[t])


def first_alarm_steps(scores_by_uid_step: Mapping[str, ScoreStream], theta: float) -> dict[str, int | None]:
    """Per run, the first step whose score is > theta (strict), or None if it never alarms."""
    out: dict[str, int | None] = {}
    for uid, scores in scores_by_uid_step.items():
        s = np.asarray(scores, dtype=float)
        # NaN > theta is False, so unscored steps cannot alarm.
        hits = np.flatnonzero(s > theta)
        out[uid] = int(hits[0]) if hits.size else None
    return out


def alarms_by_t(scores_by_uid_step: Mapping[str, ScoreStream], theta: float, t: int) -> dict[str, bool]:
    """Alarm by t <=> running max over the scored steps <= t exceeds theta <=> first alarm <= t."""
    return {uid: step is not None and step <= t for uid, step in first_alarm_steps(scores_by_uid_step, theta).items()}


@dataclass(frozen=True)
class AlarmRates:
    t: int
    fa: float  # NaN when no healthy run has a step t
    tpr: float  # NaN when there is no positive at t
    early_alarm_rate: float  # of the positives that alarm by t, the share whose first alarm < tau
    n_healthy: int
    n_healthy_alarm: int
    n_pos: int
    n_pos_alarm: int
    n_early: int
    n_healthy_without_step: int  # healthy runs that end before step t: not in FA(t)
    n_excluded_tau_after_t: int  # failed runs of `classes` with tau > t: not failed yet (D8)


def _rate(k: int, n: int) -> float:
    return k / n if n else math.nan


def fa_tpr(
    first_alarm: Mapping[str, int | None], runs: pd.DataFrame, t: int, classes: Sequence[str]
) -> AlarmRates:
    """FA(t) over healthy runs with a step t; TPR(t) over runs of `classes` with tau <= t and a
    step t; an alarm counts if its first step is <= t, also before tau (early alarms are counted
    separately). `first_alarm` comes from first_alarm_steps; `runs` has uid, T, failure_class, tau.
    """
    missing = set(runs["uid"]) - set(first_alarm)
    if missing:
        raise ValueError(f"no alarm entry for runs {sorted(missing)[:5]}")
    has_step = runs["T"] >= t + 1
    healthy = runs["failure_class"].isna()
    in_classes = runs["failure_class"].isin(list(classes))
    # tau is NaN on healthy runs; NaN <= t is False, and in_classes is False for them anyway.
    pos = has_step & in_classes & (runs["tau"] <= t)
    neg = has_step & healthy
    step_of = runs["uid"].map(first_alarm)
    alarm = step_of.notna() & (step_of <= t)
    early = pos & alarm & (step_of < runs["tau"])
    n_h, n_ha = int(neg.sum()), int((neg & alarm).sum())
    n_p, n_pa = int(pos.sum()), int((pos & alarm).sum())
    return AlarmRates(
        t=t,
        fa=_rate(n_ha, n_h),
        tpr=_rate(n_pa, n_p),
        early_alarm_rate=_rate(int(early.sum()), n_pa),
        n_healthy=n_h,
        n_healthy_alarm=n_ha,
        n_pos=n_p,
        n_pos_alarm=n_pa,
        n_early=int(early.sum()),
        n_healthy_without_step=int((healthy & ~has_step).sum()),
        n_excluded_tau_after_t=int((has_step & in_classes & (runs["tau"] > t)).sum()),
    )
