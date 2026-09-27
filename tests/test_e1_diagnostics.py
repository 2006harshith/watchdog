import pandas as pd
import pytest

from watchdog_agent.baselines.esn import Run
from watchdog_agent.experiments.author_protocol import author_split
from watchdog_agent.experiments.e1_diagnostics import (
    random_split,
    source_pools,
    task_disjoint_split,
    trajectory_hash,
    twin_audit,
    twin_aware_split,
)


def _step(result: str, text: str = "t", latency: float = 1.0) -> dict:
    return {
        "text": text, "latency_s": latency, "token_logprobs": [-0.1, -0.2],
        "tool_events": [{"id": "x", "name": "calc", "args": {"a": 1, "b": 2}, "result": result}],
    }


def test_hash_ignores_text_latency_logprobs_and_ids():
    a = [_step("3", text="one", latency=1.0)]
    b = [_step("3", text="two", latency=9.0)]
    b[0]["token_logprobs"] = [-5.0]
    b[0]["tool_events"][0]["id"] = "y"
    assert trajectory_hash(a) == trajectory_hash(b)


def test_hash_changes_with_result_and_ignores_args_key_order():
    assert trajectory_hash([_step("3")]) != trajectory_hash([_step("4")])
    reordered = [_step("3")]
    reordered[0]["tool_events"][0]["args"] = {"b": 2, "a": 1}
    assert trajectory_hash(reordered) == trajectory_hash([_step("3")])


def test_twin_audit_counts_runs_in_mixing_clusters():
    # cluster "1": 2 healthy; cluster "2": healthy + looping + looping (mixing);
    # cluster "3": 2 goal_drift; plus one singleton that is no twin of anything.
    layout = [("1", None), ("1", None), ("2", None), ("2", "looping"), ("2", "looping"),
              ("3", "goal_drift"), ("3", "goal_drift"), ("solo", "looping")]
    episodes = pd.DataFrame({
        "corpus": "c",
        "task_group": "g",
        "failure_class": [fc for _, fc in layout],
        "steps_parsed": [[_step(result)] for result, _ in layout],
    })
    audit = twin_audit(episodes)["c"]
    assert audit["twin_clusters"] == 3
    assert audit["clusters_mixing_healthy_and_failed"] == 1
    assert audit["runs_in_mixing_clusters"] == {
        "total": 3,
        "by_label": {"healthy": 1, "failed": 2},
        "by_failure_class": {"looping": 2},
    }


def test_hash_keeps_step_boundaries():
    two_steps = [_step("3"), _step("3")]
    one_step = [{**_step("3"), "tool_events": _step("3")["tool_events"] * 2}]
    assert trajectory_hash(two_steps) != trajectory_hash(one_step)


# 10 task groups x (8 healthy + 4 failed); healthy runs come in twin pairs within a group.
RUNS = [
    Run(f"g{g}-{j}", [], None if j < 8 else "looping") for g in range(10) for j in range(12)
]
GROUP_OF = {r.uid: r.uid.split("-")[0] for r in RUNS}
HASH_OF = {r.uid: f"{GROUP_OF[r.uid]}-{int(r.uid.split('-')[1]) // 2}" for r in RUNS}
N_FIT, N_VAL = round(0.6 * 80), round(0.2 * 80)


def test_random_split_seed_0_is_the_author_split():
    fit, val, test, _ = random_split(RUNS, 0)
    a_fit, a_val, a_test = author_split(RUNS, 0)
    assert (fit, val, test) == ([r.uid for r in a_fit], [r.uid for r in a_val], [r.uid for r in a_test])


@pytest.mark.parametrize("seed", range(5))
def test_all_split_types_have_the_same_fit_and_val_size(seed):
    splits = [
        random_split(RUNS, seed),
        twin_aware_split(RUNS, HASH_OF, seed, N_FIT, N_VAL),
        task_disjoint_split(RUNS, GROUP_OF, seed, N_FIT, N_VAL),
    ]
    for split in splits:
        assert not isinstance(split, str)
        fit, val, _, _ = split
        assert (len(fit), len(val)) == (N_FIT, N_VAL)


@pytest.mark.parametrize("seed", range(5))
def test_twin_aware_split_never_splits_a_twin_cluster(seed):
    fit, val, test, _ = twin_aware_split(RUNS, HASH_OF, seed, N_FIT, N_VAL)
    side = {}
    for name, uids in (("fit", fit), ("val", val), ("test", test)):
        for u in uids:
            assert side.setdefault(HASH_OF[u], name) == name


@pytest.mark.parametrize("seed", range(5))
def test_twin_aware_split_trims_overflowing_clusters_instead_of_leaking_them(seed):
    hash_of_4 = {r.uid: f"{GROUP_OF[r.uid]}-{int(r.uid.split('-')[1]) // 4}" for r in RUNS}
    fit, val, test, extra = twin_aware_split(RUNS, hash_of_4, seed, 47, 15)
    assert (len(fit), len(val)) == (47, 15)
    assert extra["healthy_dropped_trimmed_twins"] > 0
    assert not {hash_of_4[u] for u in fit + val} & {hash_of_4[u] for u in test}


@pytest.mark.parametrize("seed", range(5))
def test_task_disjoint_split_shares_no_group_between_fit_val_and_test(seed):
    fit, val, test, _ = task_disjoint_split(RUNS, GROUP_OF, seed, N_FIT, N_VAL)
    test_groups = {GROUP_OF[u] for u in test}
    assert not {GROUP_OF[u] for u in fit + val} & test_groups
    failed = {r.uid for r in RUNS if r.failure_class is not None}
    assert not set(fit + val) & failed


def test_source_pools_split_healthy_source_runs_by_test_task_groups():
    # source: groups g0..g4 shared with the target, s0..s2 its own
    source = [Run(f"src-{g}-{j}", [], None if j < 3 else "looping")
              for g in ("g0", "g1", "g2", "g3", "g4", "s0", "s1", "s2") for j in range(4)]
    group_of = GROUP_OF | {r.uid: r.uid.split("-")[1] for r in source}
    test_uids, pools = source_pools(source, RUNS, group_of)
    test_groups = {group_of[u] for u in test_uids}
    assert pools["overlap"] and pools["excluded"]
    assert {group_of[u] for u in pools["overlap"]} <= test_groups
    assert not {group_of[u] for u in pools["excluded"]} & test_groups
    by_uid = {r.uid: r for r in source}
    assert all(by_uid[u].failure_class is None for u in pools["overlap"] + pools["excluded"])
