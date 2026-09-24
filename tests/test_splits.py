import pandas as pd

from watchdog_agent.splits import grouped_kfold, healthy_subset, leave_one_family_out


def _frame(rows):
    return pd.DataFrame.from_records(
        rows, columns=["uid", "family", "task_group", "failure_class"]
    )


def test_leave_one_family_out_drops_overlapping_task_groups():
    df = _frame(
        [
            ("q1", "qwen", "g1", None),
            ("q2", "qwen", "g2", "looping"),  # g2 overlaps with test family -> dropped
            ("q3", "qwen", "g3", None),
            ("l1", "llama", "g2", None),
            ("l2", "llama", "g4", "looping"),
        ]
    )
    train_uids, test_uids, n_dropped = leave_one_family_out(df, test_family="llama")

    assert set(test_uids) == {"l1", "l2"}
    assert set(train_uids) == {"q1", "q3"}
    assert n_dropped == 1
    assert "llama" not in df.loc[df["uid"].isin(train_uids), "family"].tolist()


def test_grouped_kfold_partitions_and_never_splits_a_task_group():
    rows = []
    for i in range(12):
        rows.append((f"u{i}", "qwen", f"g{i // 2}", None))  # 6 groups, 2 uids each
    df = _frame(rows)

    folds = grouped_kfold(df, family="qwen", k=3, seed=0)

    assert len(folds) == 3
    all_eval_uids = [uid for _, eval_uids in folds for uid in eval_uids]
    assert sorted(all_eval_uids) == sorted(df["uid"])  # partition, no dup/missing

    for fit_uids, eval_uids in folds:
        eval_groups = set(df.loc[df["uid"].isin(eval_uids), "task_group"])
        fit_groups = set(df.loc[df["uid"].isin(fit_uids), "task_group"])
        assert eval_groups.isdisjoint(fit_groups)


def test_grouped_kfold_same_seed_same_split():
    rows = [(f"u{i}", "qwen", f"g{i}", None) for i in range(10)]
    df = _frame(rows)

    folds_a = grouped_kfold(df, family="qwen", k=4, seed=7)
    folds_b = grouped_kfold(df, family="qwen", k=4, seed=7)

    assert folds_a == folds_b


def test_healthy_subset_excludes_given_task_groups_and_failed_runs():
    rows = [
        ("h1", "qwen", "g1", None),
        ("h2", "qwen", "g2", None),
        ("h3", "qwen", "g3", None),  # excluded task_group
        ("f1", "qwen", "g4", "looping"),  # not healthy
    ]
    df = _frame(rows)

    uids = healthy_subset(df, family="qwen", n=10, seed=0, exclude_task_groups={"g3"})

    assert set(uids) == {"h1", "h2"}
