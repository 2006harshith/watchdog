from pathlib import Path

import pandas as pd
import yaml

from watchdog_agent.experiments.e3 import calibration_pool_counts


def _runs():
    # 6 task groups x 4 runs; T = 3..6 per group, so "has a step t" (T >= t + 1) thins out with t.
    # Within each group the T=6 run is failed: it has every step but must never count as calibration.
    rows = []
    for g in range(6):
        for i, T in enumerate([3, 4, 5, 6]):
            rows.append({
                "uid": f"g{g}-{i}", "family": "fam", "task_group": f"g{g}", "T": T,
                "failure_class": "looping" if T == 6 else None,
            })
    other = {"uid": "other-0", "family": "x", "task_group": "g0", "T": 9, "failure_class": None}
    return pd.DataFrame([*rows, other])


def test_counts_partition_and_exclude_failed_and_short_runs():
    out = calibration_pool_counts(_runs(), "fam", [2, 3, 4], k=3, seed=0, budgets=[0, 5, 10])
    # Healthy runs with a step t per group: t=2 -> T in {3,4,5} = 3; t=3 -> 2; t=4 -> 1 (T=5 only).
    for t, per_group in [(2, 3), (3, 2), (4, 1)]:
        block = out["per_t"][str(t)]
        assert block["healthy_with_step"] == 6 * per_group  # the other family's run never counts
        folds = block["per_fold"]
        assert sum(f["test_healthy"] for f in folds) == block["healthy_with_step"]
        for f in folds:
            # Eligible = everything healthy with a step t that is not in the eval fold itself.
            assert f["eligible"] == block["healthy_with_step"] - f["test_healthy"]
            assert f["feasible"] == {"0": True, "5": 5 <= f["eligible"], "10": 10 <= f["eligible"]}
        assert block["folds_skipped"]["10"] == sum(not f["feasible"]["10"] for f in folds)
    # k=3 over 6 groups: 2 groups per fold, so at t=4 each fold holds 2 test runs and 4 eligible.
    assert [f["eligible"] for f in out["per_t"]["4"]["per_fold"]] == [4, 4, 4]
    assert out["per_t"]["4"]["folds_skipped"] == {"0": 0, "5": 3, "10": 3}


def test_e3_config_parses_and_holds_the_four_expectations():
    cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    assert [e["id"] for e in cfg["expectations"]] == ["E-1", "E-2", "E-3", "E-4"]
    assert cfg["thresholds"]["primary_alpha"] == 0.10
    assert cfg["budgets"]["n"] == [0, 10, 20, 40]
    assert cfg["arms"]["ESN-refit"]["families"] == ["llama"]  # D6: no model fit on gemini
