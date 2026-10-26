"""SP6 E3 recalibration curve: alarms on a new model from n healthy target runs.

Pre-registered in configs/e3.yaml (docs/decisions.md, "SP6 Decisions and pre-registered
expectations"). No gate. Step 0 holds only the calibration-pool counts; the experiment is Step 2.
"""

from collections.abc import Sequence

import pandas as pd

from watchdog_agent.splits import grouped_kfold


def calibration_pool_counts(
    episodes: pd.DataFrame, family: str, checkpoints: Sequence[int], k: int, seed: int, budgets: Sequence[int]
) -> dict:
    """Per checkpoint t and target fold f: fold f's healthy runs with a step t (test only), and the
    eligible calibration runs, i.e. healthy runs with a step t in the OTHER folds. A budget n is
    feasible in fold f only if n <= eligible, so this says where n = 40 must be skipped."""
    runs = episodes[episodes["family"] == family]
    # eval_only_families=(): an eval-only family's folds only feed calibration, never a fit (D6 reading).
    folds = grouped_kfold(runs, family, k, seed, eval_only_families=())
    out: dict = {"family": family, "k": k, "seed": seed, "per_t": {}}
    for t in checkpoints:
        healthy_with_step = set(runs.loc[runs["failure_class"].isna() & (runs["T"] >= t + 1), "uid"])
        per_fold = []
        for f, (other_uids, eval_uids) in enumerate(folds):
            eligible = len(healthy_with_step & set(other_uids))
            per_fold.append({
                "fold": f,
                "runs": len(eval_uids),
                "test_healthy": len(healthy_with_step & set(eval_uids)),
                "eligible": eligible,
                "feasible": {str(n): n <= eligible for n in budgets},
            })
        out["per_t"][str(t)] = {
            "healthy_with_step": len(healthy_with_step),
            "per_fold": per_fold,
            "folds_skipped": {str(n): sum(not p["feasible"][str(n)] for p in per_fold) for n in budgets},
        }
    return out
