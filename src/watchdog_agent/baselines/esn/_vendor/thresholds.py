# Vendored from github.com/sunnydubey1111/agent-trajectory-sentinel
# commit 1b3e07fee53ae13407173c3ea932adb4a43e8230, file derail/evaluation/metrics.py
# Copyright 2026 Sunny Dubey. Apache License 2.0: see LICENSE and NOTICE in this directory.
# Modified by watchdog (2026-09-25): EXCERPT. Only first_alarm, min_calibration_episodes and
# pick_threshold (original lines 31-129, verbatim); module imports reduced to numpy, since the
# rest of that module pulls in pandas/scikit-learn at import time.
"""Alarm rule and healthy-validation threshold of the ESN baseline (excerpt)."""

from __future__ import annotations

import numpy as np


def first_alarm(scores: np.ndarray, theta: float) -> int | None:
    """Return the first step index t with scores[t] > theta, or None.

    Strict inequality, matching the alarm rule tau_hat = min{t : s_t > theta}.

    A non-finite score is REFUSED rather than treated as quiet. `NaN > theta`
    is False, so a monitor that failed to score a step would otherwise be
    indistinguishable from one that scored it and saw nothing -- the failure
    would be counted as evidence of health. The monitors guard finiteness
    themselves, but those guards are asserts and `python -O` removes them, so
    the last gate before the alarm decision has to be a real check.
    """
    s = np.asarray(scores, dtype=float)
    if not np.all(np.isfinite(s)):
        bad = int(np.flatnonzero(~np.isfinite(s))[0])
        raise FloatingPointError(
            f"non-finite score at step {bad}; the monitor could not score "
            f"this episode, which is not the same as scoring it as quiet")
    idx = np.flatnonzero(s > theta)
    return int(idx[0]) if idx.size else None


def min_calibration_episodes(fa_budget: float) -> int:
    """Smallest healthy-episode count at which `fa_budget` is reachable.

    A threshold read off the maxima of n healthy episodes cannot deliver an
    expected false-alarm rate below 1/(n+1): a fresh healthy episode exceeds
    the maximum of n exchangeable ones with exactly that probability. So an
    empirical threshold needs n >= 1/fa_budget - 1, and below that the budget
    is unreachable no matter which quantile is taken.
    """
    if not 0.0 < fa_budget < 1.0:
        raise ValueError("fa_budget must be in (0, 1)")
    return int(np.ceil(1.0 / fa_budget - 1.0))


def pick_threshold(val_healthy_scores: list[np.ndarray],
                   fa_budget: float = 0.05,
                   method: str = "empirical",
                   warn_infeasible: bool = True) -> float:
    """Threshold theta from healthy validation score streams.

    ``method="empirical"`` (default, unchanged): the per-episode maximum of
    each stream, then the (1 - fa_budget) quantile of those maxima with
    np.quantile method="higher", so theta is an observed maximum and the
    realized healthy-val false-alarm rate (fraction of maxima strictly above
    theta) is <= fa_budget *on that sample*.

    That in-sample guarantee does not transfer. Measured on six real corpora
    at a 5% budget, the realized held-out rate is 8.2% (per-corpus 4.7-11.0%),
    for two separate reasons:

      - an ORDER-STATISTIC FLOOR of 1/(n+1) (see `min_calibration_episodes`);
        corpora calibrating on 12-15 episodes cannot reach 5% at all, and are
        measured sitting exactly on their floor;
      - HEAVY TAILS, which push some corpora well above even that floor
        (real_research7b: floor 4.0%, realized 11.0%).

    ``method="lognormal"`` fits a log-normal to the maxima and returns its
    analytic (1 - fa_budget) quantile. Because it extrapolates past the
    largest observed value it escapes the order-statistic floor legitimately,
    and it is less sensitive to a single extreme episode. Measured on the same
    six corpora: realized FA 6.7% (vs 8.2%) for 3.0 points of detection
    (50.8% -> 47.8%). It is available but is not the default anywhere, because
    on corpora where the empirical rule already lands near the budget the tail
    fit overshoots it instead.

    Neither method fabricates data: with few episodes a budget may simply be
    unreachable, and `warn_infeasible` says so rather than returning a
    threshold that quietly misses it.
    """
    if not val_healthy_scores:
        raise ValueError("val_healthy_scores must be non-empty")
    maxima = np.array([float(np.max(np.asarray(s, dtype=float)))
                       for s in val_healthy_scores])
    n_min = min_calibration_episodes(fa_budget)
    if warn_infeasible and len(maxima) < n_min and method == "empirical":
        import warnings
        warnings.warn(
            f"pick_threshold: {len(maxima)} healthy episodes cannot deliver a "
            f"{fa_budget:.0%} false-alarm budget by an empirical quantile "
            f"(order-statistic floor 1/(n+1) = {1/(len(maxima)+1):.1%}; "
            f"needs n >= {n_min}). Collect more healthy episodes, relax the "
            f'budget, or use method="lognormal" to extrapolate the tail.',
            RuntimeWarning, stacklevel=2)
    if method == "empirical":
        return float(np.quantile(maxima, 1.0 - fa_budget, method="higher"))
    if method == "lognormal":
        from scipy import stats as _sps
        pos = maxima[maxima > 0.0]
        if len(pos) < 3:            # too few to fit a tail; stay empirical
            return float(np.quantile(maxima, 1.0 - fa_budget, method="higher"))
        lg = np.log(pos)
        sd = float(lg.std(ddof=1))
        if not np.isfinite(sd) or sd == 0.0:
            return float(np.quantile(maxima, 1.0 - fa_budget, method="higher"))
        return float(np.exp(lg.mean() + sd * _sps.norm.ppf(1.0 - fa_budget)))
    raise ValueError(f"unknown method {method!r}; "
                     "expected 'empirical' or 'lognormal'")
