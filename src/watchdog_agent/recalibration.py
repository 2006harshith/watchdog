"""SP5 core piece: label-free re-expression of prefix features against healthy runs.

A feature value is re-expressed as where it falls among HEALTHY runs of the same group (family
or corpus) at the same checkpoint t. Unlike a raw value it means the same thing in every family:
"0.97" = higher than 97% of this family's healthy runs. SP4 4a showed raw TL values identify the
model family on healthy runs (llama errs 38% of the time when healthy, qwen 3%), so a threshold
learned on raw qwen values fires on healthy llama; the percentile removes that shift.

Label use: the reference needs to know which runs are healthy (the premise: a few healthy runs
of the new model) and how to group and fold them. Nothing else about a run is read, and a run's
own label never changes its own transformed value (cross-fitting, tests/test_recalibration.py).

D6 reading (docs/decisions.md, SP5): building a reference from gemini's healthy runs to score
gemini at evaluation time is calibration, not training.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from watchdog_agent.splits import grouped_kfold

GROUP_KEYS = ("family", "corpus")


@dataclass(frozen=True)
class HealthyReference:
    by: str  # "family" or "corpus"
    sorted_values: dict[tuple[str, int, str], np.ndarray]  # (group, t, feature) -> sorted, NaN removed
    n_healthy_rows: dict[tuple[str, int], int]  # (group, t) -> healthy prefix rows in the reference


def _row_keys(prefix_table: pd.DataFrame, runs: pd.DataFrame, by: str) -> tuple[np.ndarray, np.ndarray]:
    """The group (runs[by]) and checkpoint t of every row of the prefix table."""
    uids = prefix_table.index.get_level_values("uid")
    group_of = runs.set_index("uid")[by]
    missing = set(uids) - set(group_of.index)
    if missing:
        raise ValueError(f"prefix rows for runs not in `runs`: {sorted(missing)[:5]}")
    return group_of.loc[uids].to_numpy(), prefix_table.index.get_level_values("t").to_numpy()


def fit_healthy_reference(prefix_table: pd.DataFrame, runs: pd.DataFrame, by: str) -> HealthyReference:
    """Sorted healthy values per (group, t, feature). Only healthy runs' rows enter it."""
    if by not in GROUP_KEYS:
        raise ValueError(f"by must be one of {GROUP_KEYS}, got {by!r}")
    healthy = set(runs.loc[runs["failure_class"].isna(), "uid"])
    rows = prefix_table[prefix_table.index.get_level_values("uid").isin(healthy)]
    group, t = _row_keys(rows, runs, by)
    values: dict[tuple[str, int, str], np.ndarray] = {}
    counts: dict[tuple[str, int], int] = {}
    # groupby on two parallel arrays: one block of rows per (group, t) pair.
    for (g, tt), block in rows.groupby([group, t]):
        counts[(g, int(tt))] = len(block)
        for feature in rows.columns:
            v = block[feature].to_numpy(dtype=float)
            values[(g, int(tt), feature)] = np.sort(v[~np.isnan(v)])
    return HealthyReference(by, values, counts)


def _apply(
    prefix_table: pd.DataFrame,
    reference: HealthyReference,
    runs: pd.DataFrame,
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> pd.DataFrame:
    """fn(sorted healthy values, x) per (group, t, feature); NaN in -> NaN out; empty reference -> NaN."""
    group, t = _row_keys(prefix_table, runs, reference.by)
    x_all = prefix_table.to_numpy(dtype=float)
    out = np.full(x_all.shape, np.nan)
    for (g, tt), idx in pd.Series(range(len(prefix_table))).groupby([group, t]).groups.items():
        key = (g, int(tt))
        if key not in reference.n_healthy_rows:
            # A whole (group, t) without healthy runs cannot be re-expressed; silently leaving it
            # NaN would hand the classifier an all-missing block that looks like a signal.
            raise ValueError(f"no healthy reference for {reference.by}={g!r} at t={tt}")
        rows = np.asarray(idx)
        for j, feature in enumerate(prefix_table.columns):
            ref = reference.sorted_values[(g, int(tt), feature)]
            x = x_all[rows, j]
            if ref.size:
                y = fn(ref, x)
                out[rows, j] = np.where(np.isnan(x), np.nan, y)
    return pd.DataFrame(out, index=prefix_table.index, columns=prefix_table.columns)


def _mid_rank(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    # np.searchsorted on the sorted healthy values: side="left" counts healthy values < x,
    # side="right" counts values <= x; their mean is the mid-rank, so ties count half.
    below = np.searchsorted(ref, x, side="left")
    at_or_below = np.searchsorted(ref, x, side="right")
    return (below + at_or_below) / (2.0 * ref.size)


def percentile_transform(
    prefix_table: pd.DataFrame, reference: HealthyReference, runs: pd.DataFrame
) -> pd.DataFrame:
    """Each value -> mid-rank fraction of the healthy reference values of its (group, t, feature).

    In [0, 1]; a value equal to the healthy median maps to 0.5; a feature constant on healthy runs
    maps to 0 / 0.5 / 1 (below / equal / above), never inf. NaN stays NaN.
    """
    return _apply(prefix_table, reference, runs, _mid_rank)


def zscore_transform(
    prefix_table: pd.DataFrame, reference: HealthyReference, runs: pd.DataFrame, std_floor: float
) -> pd.DataFrame:
    """Sensitivity arm: (x - healthy mean) / max(healthy std, std_floor). The floor is what keeps
    a feature that is constant on healthy runs (std 0) from dividing by zero; the percentile needs none.
    """
    if std_floor <= 0:
        raise ValueError("std_floor must be > 0")
    return _apply(prefix_table, reference, runs, lambda ref, x: (x - ref.mean()) / max(ref.std(), std_floor))


def cross_fitted_target_transform(
    prefix_table: pd.DataFrame,
    runs: pd.DataFrame,
    k: int = 5,
    seed: int = 0,
    method: str = "percentile",
    std_floor: float | None = None,
) -> pd.DataFrame:
    """Re-express one target family's prefixes, fold by fold over k task-group folds: fold f is
    transformed with a reference built only from the healthy runs of the other folds.

    The runs that build a reference are never scored with it, so a healthy test run cannot sit
    inside its own reference (which would pull it towards the middle and make it look healthier
    than an unseen healthy run of the same model would).
    """
    families = runs["family"].unique()
    if len(families) != 1:
        raise ValueError(f"cross-fitting works on one family at a time, got {sorted(families)}")
    if method == "percentile":
        def transform(table, ref):
            return percentile_transform(table, ref, runs)
    elif method == "zscore":
        if std_floor is None:
            raise ValueError("method='zscore' needs std_floor")

        def transform(table, ref):
            return zscore_transform(table, ref, runs, std_floor)
    else:
        raise ValueError(f"unknown method {method!r}")

    uids = prefix_table.index.get_level_values("uid")
    # eval_only_families=(): folds of an eval-only family are allowed here because they only set a
    # calibration reference, never a trained model (D6 reading, docs/decisions.md, SP5).
    folds = grouped_kfold(runs, families[0], k, seed, eval_only_families=())
    parts = []
    for fit_uids, eval_uids in folds:
        in_fit = uids.isin(fit_uids)
        reference = fit_healthy_reference(prefix_table[in_fit], runs[runs["uid"].isin(fit_uids)], by="family")
        parts.append(transform(prefix_table[uids.isin(eval_uids)], reference))
    return pd.concat(parts).loc[prefix_table.index]
