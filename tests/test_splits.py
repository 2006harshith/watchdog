import pandas as pd
import pytest

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


def _three_family_frame():
    return _frame(
        [
            ("q1", "qwen", "g1", None),
            ("q2", "qwen", "g2", "looping"),
            ("l1", "llama", "g3", None),
            ("l2", "llama", "g4", "looping"),
            ("m1", "gemini", "g5", None),
            ("m2", "gemini", "g6", "timeout"),
        ]
    )


def test_leave_one_family_out_never_trains_on_eval_only_family():
    df = _three_family_frame()
    for test_family in ("qwen", "llama", "gemini"):
        train_uids, test_uids, _ = leave_one_family_out(df, test_family=test_family)
        train_families = set(df.loc[df["uid"].isin(train_uids), "family"])
        assert "gemini" not in train_families
        assert test_family not in train_families
        assert set(df.loc[df["uid"].isin(test_uids), "family"]) == {test_family}

    # gemini as the held-out family is still allowed: evaluation-only, per D6.
    _, test_uids, _ = leave_one_family_out(df, test_family="gemini")
    assert set(test_uids) == {"m1", "m2"}


def test_leave_one_family_out_eval_only_exclusion_is_not_counted_as_overlap_drop():
    df = _three_family_frame()
    train_uids, _, n_dropped = leave_one_family_out(df, test_family="llama")
    assert set(train_uids) == {"q1", "q2"}
    assert n_dropped == 0


def test_leave_one_family_out_eval_only_families_is_configurable():
    df = _three_family_frame()
    train_uids, _, _ = leave_one_family_out(df, test_family="llama", eval_only_families=())
    assert set(train_uids) == {"q1", "q2", "m1", "m2"}


def test_grouped_kfold_refuses_to_build_fit_sets_from_eval_only_family():
    df = _three_family_frame()
    with pytest.raises(ValueError, match="gemini"):
        grouped_kfold(df, family="gemini", k=2, seed=0)


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
