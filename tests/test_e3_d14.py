import numpy as np
import pytest
from test_e3 import (  # noqa: F401  (e3_run is a fixture)
    GROUP_RECORD,
    RecordingESN,
    RecordingHGB,
    e3_run,
)

from watchdog_agent.experiments.e3 import calibration_draws, group_frame, group_split_draws
from watchdog_agent.experiments.e3_d14 import code_path_check, compare, d14_reading, run_d14
from watchdog_agent.features import prefix_feature_table


def _pool(sizes):
    group_of = {f"g{g}-{i}": f"g{g}" for g, k in enumerate(sizes) for i in range(k)}
    return sorted(group_of), group_of


def test_group_split_halves_share_no_task_group_and_keep_m_at_n_over_2():
    eligible, group_of = _pool([6, 3, 3, 2, 2, 1, 1])
    draws, tries = group_split_draws(eligible, group_of, 8, 20, 0, 0, (0, 3, 1), 1000)
    assert len(draws) == 20 and all(k >= 1 for k in tries)
    for d in draws:
        assert len(d.half_a) == len(d.half_b) == 4
        assert not {group_of[u] for u in d.half_a} & {group_of[u] for u in d.half_b}
        assert set(d.uids) == set(d.half_a) | set(d.half_b)
    again, _ = group_split_draws(eligible, group_of, 8, 20, 0, 0, (0, 3, 1), 1000)
    assert again == draws  # seeded per (key, n, r, try): reproducible
    other, _ = group_split_draws(eligible, group_of, 8, 20, 0, 0, (0, 3, 2), 1000)
    assert other != draws


def test_group_split_is_none_when_no_partition_or_too_few_runs():
    eligible, group_of = _pool([10])  # one group cannot feed both halves
    assert group_split_draws(eligible, group_of, 4, 3, 0, 0, (0,), 50) is None
    eligible, group_of = _pool([2, 2])
    assert group_split_draws(eligible, group_of, 6, 3, 0, 0, (0,), 50) is None  # n > eligible: E3's skip


@pytest.mark.parametrize("r, excess, ci, reading", [
    (0.10, 0.20, {"lo": 0.01, "hi": 0.2}, "FALLS"),  # half the excess, CI above 0
    (0.10, 0.20, {"lo": -0.01, "hi": 0.2}, "PARTIAL"),  # enough drop, but the CI contains 0
    (0.09, 0.20, {"lo": 0.01, "hi": 0.2}, "PARTIAL"),  # CI above 0, under half the excess
    (0.02, 0.20, {"lo": 0.01, "hi": 0.03}, "DOES NOT FALL"),  # 0.02 is the inclusive no-fall cut
    (-0.05, 0.20, {"lo": -0.1, "hi": 0.0}, "DOES NOT FALL"),
    (0.10, 0.20, {"lo": float("nan"), "hi": float("nan")}, "UNDEFINED"),
])
def test_d14_reading_rule(r, excess, ci, reading):
    assert d14_reading(r, excess, ci, {"falls_share_of_excess": 0.5, "no_fall_le": 0.02}) == reading


def _d14_cfg():
    return {"family": "llama", "arms": ["TL-pct", "ALL-pct", "ESN-refit"], "n": [4, 8], "alpha": 0.5,
            "split": {"by": "task_group", "max_tries": 1000}, "big_group_min_runs": 3,
            "code_path_check": {"n": 8, "t": 3, "fold": 0},
            "reading": {"falls_share_of_excess": 0.5, "no_fall_le": 0.02}, "bootstrap": {"B": 30, "seed": 0}}


@pytest.fixture(scope="module")
def setup(e3_run):  # noqa: F811
    episodes, e3_cfg, e2_cfg, _, frame, _, _ = e3_run
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), [2, 3, 4, 5])
    return episodes, e3_cfg, e2_cfg, table, frame, group_frame(GROUP_RECORD)


def _run(setup, split, only=None):
    episodes, e3_cfg, e2_cfg, table, _, _ = setup
    return run_d14(table, episodes, _d14_cfg(), e3_cfg, e2_cfg, split=split, only=only,
                   make_esn=lambda ch: RecordingESN(), make_model=RecordingHGB)


def test_the_d14_loop_on_e3_draws_reproduces_e3_counts_exactly(setup):
    *_, e3_groups = setup
    _, groups, _ = _run(setup, "run", only=(8, 3, 0))
    check = code_path_check(groups, e3_groups, _d14_cfg())
    assert check["ok"], check
    assert set(groups["arm"]) == {"TL-pct", "ALL-pct", "ESN-refit"}


def test_group_split_run_records_disjoint_halves_and_compares_on_the_same_test_runs(setup):
    *_, e3_frame, e3_groups = setup
    frame, groups, extra = _run(setup, "task_group")
    feasible = [s for s in extra["splits"] if s["feasible"]]
    assert feasible and all(s["shared_groups_max"] == 0 for s in feasible)
    out = compare(e3_groups, e3_frame, groups, frame, extra, _d14_cfg(), setup[1])
    for arm in _d14_cfg()["arms"]:
        for n in ("4", "8"):
            c = out["cells"][arm][n]
            assert c["reading"] in {"FALLS", "PARTIAL", "DOES NOT FALL", "UNDEFINED"}
            # Same test runs under both splits wherever both test a fold.
            for v in c["per_fold"].values():
                for a, b in zip(v["run"], v["group"]):
                    if a["n_test_healthy"] and b["n_test_healthy"]:
                        assert a["n_test_healthy"] == b["n_test_healthy"]


def test_group_split_draws_differ_from_e3_draws_only_in_how_the_halves_are_formed():
    # The by-run draws of the same key are a different sample, so pairing with E3 is by distribution.
    eligible, group_of = _pool([6, 3, 3, 2, 2, 1, 1])
    by_run = calibration_draws(eligible, 8, 5, 0, 0, (0, 3, 1))
    by_group, _ = group_split_draws(eligible, group_of, 8, 5, 0, 0, (0, 3, 1), 1000)
    assert any({group_of[u] for u in d.half_a} & {group_of[u] for u in d.half_b} for d in by_run)
    assert [len(d.half_b) for d in by_run] == [len(d.half_b) for d in by_group]
    np.testing.assert_array_equal([len(d.uids) for d in by_run], [8] * 5)
