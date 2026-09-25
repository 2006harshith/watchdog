import json

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn import RunScore
from watchdog_agent.experiments.e1 import (
    CSV_COLUMNS,
    plan_grouped_folds,
    plan_leave_one_group_out,
    run_e1,
    score_plans,
    split_by_task_group,
    write_outputs,
)

T = 8
TAU = 4
TOOLS = ("lookup_flight", "lookup_hotel", "calculator")


def _step(rng: np.random.Generator, t: int, broken: bool) -> dict:
    if broken:
        return {
            "text": "ERROR ERROR timeout ### retrying ??? " + "x" * int(rng.integers(50, 80)),
            "token_logprobs": (-rng.exponential(5.0, size=300)).tolist(),
            "action": "tool_call", "latency_s": 40.0, "output_tokens": 400, "error": True,
        }
    tool = TOOLS[t % len(TOOLS)]
    return {
        "text": f'[{tool}({{"q": "item-{t}"}}) -> {{"price": {int(rng.integers(50, 500))}}}] ok.',
        "token_logprobs": (-rng.exponential(0.3, size=30)).tolist(),
        "action": "tool_call", "latency_s": float(rng.uniform(0.8, 1.5)),
        "output_tokens": int(rng.integers(25, 40)), "error": False,
    }


def _episodes() -> pd.DataFrame:
    """Target: 6 task groups x (5 healthy + 3 failed). Source: 3 groups shared with the
    target + 3 of its own, each 6 healthy + 2 failed."""
    rows, seed = [], 0
    layout = [("tgt", "fam_t", f"g{i}", 5, 3) for i in range(6)]
    layout += [("src", "fam_s", g, 6, 2) for g in ("g0", "g1", "g2", "s0", "s1", "s2")]
    for corpus, family, group, n_healthy, n_failed in layout:
        for j in range(n_healthy + n_failed):
            failed = j >= n_healthy
            rng = np.random.default_rng(seed)
            rows.append({
                "uid": f"{corpus}-{group}-{j}", "corpus": corpus, "family": family,
                "task_group": group, "T": T,
                "failure_class": "looping" if failed else None,
                "tau": float(TAU) if failed else np.nan,
                "steps_parsed": [_step(rng, t, failed and t >= TAU) for t in range(T)],
            })
            seed += 1
    return pd.DataFrame(rows)


EPISODES = _episodes()
PLAN_ARGS = {"source_corpus": "src", "target_corpus": "tgt", "target_family": "fam_t",
             "k": 3, "val_fraction": 0.25}


def _groups(uids: list[str]) -> set[str]:
    return set(EPISODES.loc[EPISODES["uid"].isin(uids), "task_group"])


def _all_plans():
    return plan_grouped_folds(EPISODES, fold_seed=0, **PLAN_ARGS) + plan_leave_one_group_out(
        EPISODES, source_corpus="src", target_corpus="tgt", val_fraction=0.25, seed=0
    )


def test_a_fit_and_val_share_no_task_group_with_eval_fold():
    for plan in _all_plans():
        eval_groups = _groups(plan.eval_uids)
        assert not _groups(plan.fit["A"] + plan.val["A"]) & eval_groups


def test_a_uses_only_healthy_source_runs():
    by_uid = EPISODES.set_index("uid")
    for plan in _all_plans():
        used = by_uid.loc[plan.fit["A"] + plan.val["A"]]
        assert (used["corpus"] == "src").all()
        assert used["failure_class"].isna().all()


def test_b_fits_only_on_healthy_target_runs_outside_the_eval_fold():
    by_uid = EPISODES.set_index("uid")
    for plan in _all_plans():
        used_uids = plan.fit["B"] + plan.val["B"]
        used = by_uid.loc[used_uids]
        assert (used["corpus"] == "tgt").all()
        assert used["failure_class"].isna().all()
        assert not set(used_uids) & set(plan.eval_uids)
        assert not _groups(used_uids) & _groups(plan.eval_uids)


def test_eval_folds_partition_the_target_runs():
    plans = plan_grouped_folds(EPISODES, fold_seed=0, **PLAN_ARGS)
    eval_uids = [u for p in plans for u in p.eval_uids]
    assert sorted(eval_uids) == sorted(EPISODES.loc[EPISODES["corpus"] == "tgt", "uid"])


def test_inner_fit_and_val_are_task_group_disjoint():
    for plan in _all_plans():
        for name in ("A", "B"):
            assert not _groups(plan.fit[name]) & _groups(plan.val[name])


def test_split_by_task_group_holds_out_at_least_the_fraction():
    pool = EPISODES[(EPISODES["corpus"] == "src") & EPISODES["failure_class"].isna()]
    fit, val = split_by_task_group(pool, 0.25, seed=3)
    assert len(val) >= 0.25 * len(pool)
    assert sorted(fit + val) == sorted(pool["uid"])


class _LatencyMonitor:
    """Cheap stand-in: episode score = total latency; checks it only ever fits healthy runs."""

    def __init__(self):
        self.theta = None

    def fit(self, fit_runs, val_runs):
        assert all(r.failure_class is None for r in fit_runs + val_runs)
        self.theta = 1e9

    def score(self, run):
        per_step = np.array([s["latency_s"] for s in run.steps])
        return RunScore(per_step, None, float(per_step.sum()))


def test_each_eval_run_scored_exactly_once_per_monitor():
    plans = plan_grouped_folds(EPISODES, fold_seed=0, **PLAN_ARGS)
    scores, fold_info = score_plans(EPISODES, plans, _LatencyMonitor)
    target_uids = sorted(EPISODES.loc[EPISODES["corpus"] == "tgt", "uid"])
    assert sorted(scores["uid"]) == target_uids  # one row per target run
    assert scores[["score_A", "score_B"]].notna().all().all()
    for plan in plans:
        fold_rows = scores[scores["fold"] == plan.fold]
        assert sorted(fold_rows["uid"]) == sorted(plan.eval_uids)
    assert sum(info["eval_runs"] for info in fold_info) == len(target_uids)


def test_end_to_end_synthetic_run_writes_json_with_all_keys(tmp_path):
    cfg = {
        **{k: v for k, v in PLAN_ARGS.items()},
        "fold_seed": 0,
        "esn": {"channels": ["e", "u", "m", "x"], "K": 8, "seed": 1300, "fa_budget": 0.05},
        "bootstrap": {"n": 200, "seed": 0},
        "tpr_at_fpr_alpha": 0.05,
        "gate": {"min_gap": 0.15},
        "sanity_min_auroc_b": 0.70,
        "divergence_flag": 0.05,
        "sensitivity": {"fold_seeds": [1], "leave_one_task_group_out": True, "cluster_bootstrap": True},
        "reference": {"paper": {"A": 0.527, "B": 0.885}},
    }
    result, scores = run_e1(EPISODES, cfg)
    json_path, csv_path = write_outputs(result, scores, tmp_path)
    out = json.loads(json_path.read_text())

    # < 19 validation runs cannot reach a 5% budget: recorded per fold, not raised
    assert all(f["threshold_budget_infeasible_A"] for f in out["counts"]["per_fold"])

    assert {"config", "git", "reference", "counts", "monitors", "difference_B_minus_A",
            "gate", "flags", "sensitivity", "episode_score", "standardisation"} <= set(out)
    for name in ("A", "B"):
        m = out["monitors"][name]
        for metric in ("auroc", "auprc", "tpr_at_fpr"):
            assert {"point", "lo", "hi", "n_skipped"} <= set(m[metric])
        assert {"healthy_fa_rate", "detection_rate", "median_detection_delay"} <= set(m["default_alarm"])
        assert set(m["per_class_auroc"]) == {"looping"}
        assert len(m["per_fold_auroc"]) == 3
    assert {"pass", "paired_ci_B_minus_A"} <= set(out["gate"]["rule_b"])
    assert "pass" in out["gate"]["rule_a"]
    assert out["gate"]["outcome"] in ("PASS", "FAIL")
    assert len(out["counts"]["per_fold"]) == 3
    assert {"task_group_cluster_bootstrap", "fold_seeds", "leave_one_task_group_out"} <= set(
        out["sensitivity"]
    )
    csv = pd.read_csv(csv_path)
    assert list(csv.columns) == CSV_COLUMNS
    assert len(csv) == 48
