"""SP2 core piece: task-disjoint, model-family-held-out splits.

All functions are deterministic given `seed` and operate on the episode-level
frame returned by data.load_episodes() (must have family, task_group, uid).
"""

import random

import pandas as pd


def leave_one_family_out(episodes: pd.DataFrame, test_family: str) -> tuple[list[str], list[str], int]:
    """Test = all runs of test_family. Train = other families minus every run
    whose task_group also appears in test (no task leaks across the split).

    Returns (train_uids, test_uids, n_dropped_for_overlap).
    """
    test_mask = episodes["family"] == test_family
    test_df = episodes[test_mask]
    train_df = episodes[~test_mask]

    test_groups = set(test_df["task_group"])
    overlap_mask = train_df["task_group"].isin(test_groups)

    train_uids = train_df.loc[~overlap_mask, "uid"].tolist()
    test_uids = test_df["uid"].tolist()
    n_dropped = int(overlap_mask.sum())
    return train_uids, test_uids, n_dropped


def grouped_kfold(episodes: pd.DataFrame, family: str, k: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """k folds over one family's task_groups, so every run in that family is
    scored out-of-fold exactly once (SP3 cross-fitting). Folds partition the
    family's uids; no task_group is split across two folds.
    """
    sub = episodes[episodes["family"] == family]
    groups = sorted(sub["task_group"].unique())
    rng = random.Random(seed)
    rng.shuffle(groups)

    fold_of_group = {group: i % k for i, group in enumerate(groups)}
    sub_fold = sub["task_group"].map(fold_of_group)

    folds = []
    for fold_idx in range(k):
        eval_uids = sub.loc[sub_fold == fold_idx, "uid"].tolist()
        fit_uids = sub.loc[sub_fold != fold_idx, "uid"].tolist()
        folds.append((fit_uids, eval_uids))
    return folds


def healthy_subset(
    episodes: pd.DataFrame, family: str, n: int, seed: int, exclude_task_groups: set[str]
) -> list[str]:
    """n healthy uids from `family` whose task_group is not in exclude_task_groups
    (SP6 recalibration pool: must not reuse task groups already scored/held out).
    """
    sub = episodes[
        (episodes["family"] == family)
        & episodes["failure_class"].isna()
        & (~episodes["task_group"].isin(exclude_task_groups))
    ]
    uids = sub["uid"].tolist()
    rng = random.Random(seed)
    rng.shuffle(uids)
    return uids[:n]
