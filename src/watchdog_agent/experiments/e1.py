"""SP3 E1: does the paper's ESN monitor collapse when the agent's LLM family changes?

Two monitors, scored on the same llama3.1:8b runs, fold by fold over llama task groups:
  A_f  fit on healthy qwen2.5:7b runs of the source corpus whose task_group is not in fold f;
  B_f  fit on healthy llama runs of the other folds.
Every llama run is scored exactly once by each monitor, and never by a model that saw its task.
Each fit pool is split by task group into fit runs (ESN + standardiser) and healthy validation
runs (alarm threshold + score standardisation), so neither uses a scored run or its task.

Gate (rule b): pooled AUROC_B - AUROC_A >= min_gap AND the paired run-bootstrap 95% CI of that
difference has lower bound > 0. Rule a (gap AND non-overlapping individual CIs) is reported too.
"""

import json
import random
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn import ESNBaseline, Run, RunScore
from watchdog_agent.metrics import (
    Interval,
    auprc,
    auroc,
    bootstrap_ci,
    paired_bootstrap_diff,
    tpr_at_fpr,
)
from watchdog_agent.splits import grouped_kfold

MONITORS = ("A", "B")
EPISODE_SCORE_DEFINITION = (
    "max over steps of the author's channel-max ESN-CUSUM stream (steps 0-2 are washout and "
    "score 0); = derail.evaluation.metrics.episode_auc's statistic @ 1b3e07f"
)
STANDARDISATION = (
    "pooled scores: each fold model's episode scores minus the mean, divided by the std, of that "
    "model's episode scores on its own healthy validation runs (no eval labels used)"
)
CSV_COLUMNS = [
    "uid", "fold", "label", "failure_class", "tau", "score_A", "score_B",
    "alarm_step_A", "alarm_step_B", "raw_score_A", "raw_score_B", "task_group",
]


class Monitor(Protocol):
    theta: float | None

    def fit(self, fit_runs: list[Run], val_runs: list[Run]) -> None: ...

    def score(self, run: Run) -> RunScore: ...


MonitorFactory = Callable[[], Monitor]


@dataclass(frozen=True)
class FoldPlan:
    fold: int
    eval_uids: list[str]
    fit: dict[str, list[str]]  # monitor name -> uids
    val: dict[str, list[str]]


# ---------------------------------------------------------------- planning


def _healthy(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["failure_class"].isna()]


def split_by_task_group(pool: pd.DataFrame, val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out whole task groups, in seeded random order, until >= val_fraction of the pool's
    runs are held out. Whole groups, so a threshold is never set on a task the ESN fitted on.
    """
    sizes = pool.groupby("task_group").size()
    groups = sorted(sizes.index)
    random.Random(seed).shuffle(groups)
    val_groups, n_val = set(), 0
    for group in groups:
        if n_val >= val_fraction * len(pool):
            break
        val_groups.add(group)
        n_val += int(sizes[group])
    in_val = pool["task_group"].isin(val_groups)
    fit, val = pool.loc[~in_val, "uid"].tolist(), pool.loc[in_val, "uid"].tolist()
    if len(fit) < 2 or len(val) < 2:
        raise ValueError(f"pool of {len(pool)} runs too small to split: fit={len(fit)} val={len(val)}")
    return fit, val


def _make_plan(
    fold: int,
    eval_uids: list[str],
    target_pool: pd.DataFrame,
    source_healthy: pd.DataFrame,
    eval_groups: set[str],
    val_fraction: float,
    seed: int,
) -> FoldPlan:
    pools = {
        "A": source_healthy[~source_healthy["task_group"].isin(eval_groups)],
        "B": _healthy(target_pool[~target_pool["task_group"].isin(eval_groups)]),
    }
    fit, val = {}, {}
    for name, pool in pools.items():
        fit[name], val[name] = split_by_task_group(pool, val_fraction, seed)
    return FoldPlan(fold, list(eval_uids), fit, val)


def plan_grouped_folds(
    episodes: pd.DataFrame,
    *,
    source_corpus: str,
    target_corpus: str,
    target_family: str,
    k: int,
    fold_seed: int,
    val_fraction: float,
) -> list[FoldPlan]:
    target = episodes[episodes["corpus"] == target_corpus]
    # grouped_kfold folds a whole family, so the target corpus must be that entire family.
    if set(episodes.loc[episodes["family"] == target_family, "uid"]) != set(target["uid"]):
        raise ValueError(f"{target_corpus} is not the whole {target_family} family")
    source_healthy = _healthy(episodes[episodes["corpus"] == source_corpus])
    plans = []
    for fold, (fit_uids, eval_uids) in enumerate(grouped_kfold(episodes, target_family, k, fold_seed)):
        eval_groups = set(target.loc[target["uid"].isin(eval_uids), "task_group"])
        target_pool = target[target["uid"].isin(fit_uids)]
        seed = 1000 * fold_seed + fold  # distinct, reproducible inner split per (fold_seed, fold)
        plans.append(
            _make_plan(fold, eval_uids, target_pool, source_healthy, eval_groups, val_fraction, seed)
        )
    return plans


def plan_leave_one_group_out(
    episodes: pd.DataFrame, *, source_corpus: str, target_corpus: str, val_fraction: float, seed: int
) -> list[FoldPlan]:
    target = episodes[episodes["corpus"] == target_corpus]
    source_healthy = _healthy(episodes[episodes["corpus"] == source_corpus])
    plans = []
    for fold, group in enumerate(sorted(target["task_group"].unique())):
        in_group = target["task_group"] == group
        plans.append(
            _make_plan(
                fold, target.loc[in_group, "uid"].tolist(), target[~in_group], source_healthy,
                {group}, val_fraction, seed + fold,
            )
        )
    return plans


# ---------------------------------------------------------------- scoring


def score_plans(
    episodes: pd.DataFrame, plans: list[FoldPlan], make_monitor: MonitorFactory
) -> tuple[pd.DataFrame, list[dict]]:
    frame = episodes.set_index("uid")

    def to_run(uid: str) -> Run:
        failure_class = frame.at[uid, "failure_class"]
        return Run(uid, frame.at[uid, "steps_parsed"], None if pd.isna(failure_class) else failure_class)

    rows, fold_info = [], []
    for plan in plans:
        eval_runs = [to_run(u) for u in plan.eval_uids]
        info = {"fold": plan.fold}
        scored = {}
        for name in MONITORS:
            monitor = make_monitor()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", RuntimeWarning)
                monitor.fit([to_run(u) for u in plan.fit[name]], [to_run(u) for u in plan.val[name]])
            val_scores = np.array([monitor.score(to_run(u)).episode_score for u in plan.val[name]])
            mean, std = float(val_scores.mean()), float(val_scores.std())
            if std == 0.0:
                raise ValueError(f"fold {plan.fold} monitor {name}: constant validation scores")
            scored[name] = ([monitor.score(r) for r in eval_runs], mean, std)
            info |= {
                f"n_fit_{name}": len(plan.fit[name]),
                f"n_val_{name}": len(plan.val[name]),
                f"theta_{name}": monitor.theta,
                f"val_mean_{name}": mean,
                f"val_std_{name}": std,
                f"threshold_budget_infeasible_{name}": any(
                    issubclass(w.category, RuntimeWarning) for w in caught
                ),
            }
        fold_rows = []
        for i, run in enumerate(eval_runs):
            row = {
                "uid": run.uid,
                "fold": plan.fold,
                "task_group": frame.at[run.uid, "task_group"],
                "label": int(run.failure_class is not None),
                "failure_class": run.failure_class,
                "tau": frame.at[run.uid, "tau"],
            }
            for name, (scores, mean, std) in scored.items():
                result = scores[i]
                row[f"raw_score_{name}"] = result.episode_score
                row[f"score_{name}"] = (result.episode_score - mean) / std
                row[f"alarm_step_{name}"] = np.nan if result.alarm_step is None else result.alarm_step
            fold_rows.append(row)
        labels = [r["label"] for r in fold_rows]
        info |= {
            "eval_runs": len(fold_rows),
            "healthy": labels.count(0),
            "failed": labels.count(1),
            "task_groups": len({r["task_group"] for r in fold_rows}),
        }
        rows.extend(fold_rows)
        fold_info.append(info)
    return pd.DataFrame(rows), fold_info


# ---------------------------------------------------------------- summary


def _ci(interval: Interval) -> dict:
    return interval._asdict()


def _default_alarm_stats(scores: pd.DataFrame, name: str) -> dict:
    alarm = scores[f"alarm_step_{name}"]
    healthy = scores["label"] == 0
    failed = ~healthy
    fired = alarm.notna()
    caught = failed & fired & (alarm >= scores["tau"])
    early = failed & fired & (alarm < scores["tau"])
    delays = (alarm - scores["tau"])[caught]
    return {
        "healthy_fa_rate": float(fired[healthy].mean()),
        "n_healthy": int(healthy.sum()),
        "detection_rate": float(caught.sum() / failed.sum()),
        "early_alarm_rate": float(early.sum() / failed.sum()),
        "median_detection_delay": float(delays.median()) if len(delays) else None,
        "n_caught": int(caught.sum()),
    }


def _gate(a: Interval, b: Interval, diff: Interval, min_gap: float) -> dict:
    gap = b.point - a.point
    rule_b = bool(gap >= min_gap and diff.lo > 0)
    rule_a = bool(gap >= min_gap and b.lo > a.hi)
    return {
        "gap_B_minus_A": gap,
        "min_gap": min_gap,
        "rule_b": {"pass": rule_b, "paired_ci_B_minus_A": _ci(diff)},
        "rule_a": {"pass": rule_a, "auroc_A_ci": [a.lo, a.hi], "auroc_B_ci": [b.lo, b.hi]},
        "outcome": "PASS" if rule_b else "FAIL",
    }


def _headline(scores: pd.DataFrame, unit_col: str, cfg: dict) -> dict:
    """AUROC/AUPRC/TPR@FPR CIs for A and B plus the paired AUROC difference, resampling unit_col."""
    y = scores["label"].to_numpy()
    units = scores[unit_col].to_numpy()
    n, seed, alpha = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"], cfg["tpr_at_fpr_alpha"]

    def tpr(y_, s_):
        return tpr_at_fpr(y_, s_, alpha)

    out = {}
    for name in MONITORS:
        s = scores[f"score_{name}"].to_numpy()
        out[name] = {
            "auroc": bootstrap_ci(auroc, y, s, units, n, seed=seed),
            "auprc": bootstrap_ci(auprc, y, s, units, n, seed=seed),
            "tpr_at_fpr": bootstrap_ci(tpr, y, s, units, n, seed=seed),
        }
    out["diff"] = paired_bootstrap_diff(
        auroc, y, scores["score_A"].to_numpy(), scores["score_B"].to_numpy(), units, n, seed=seed
    )
    return out


def summarise(scores: pd.DataFrame, cfg: dict) -> dict:
    head = _headline(scores, "uid", cfg)
    n, seed = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    healthy = scores[scores["label"] == 0]
    monitors = {}
    for name in MONITORS:
        per_fold = {
            int(fold): auroc(g["label"].to_numpy(), g[f"raw_score_{name}"].to_numpy())
            for fold, g in scores.groupby("fold")
            if g["label"].nunique() == 2
        }
        mean_fold = float(np.mean(list(per_fold.values())))
        pooled = head[name]["auroc"]
        per_class = {}
        for cls in sorted(scores["failure_class"].dropna().unique()):
            sub = pd.concat([healthy, scores[scores["failure_class"] == cls]])
            ci = bootstrap_ci(
                auroc, sub["label"].to_numpy(), sub[f"score_{name}"].to_numpy(),
                sub["uid"].to_numpy(), n, seed=seed,
            )
            per_class[cls] = _ci(ci) | {"n_failed": int((sub["label"] == 1).sum())}
        monitors[name] = {
            "auroc": _ci(pooled),
            "auprc": _ci(head[name]["auprc"]),
            "tpr_at_fpr": _ci(head[name]["tpr_at_fpr"]),
            "per_fold_auroc": per_fold,
            "mean_per_fold_auroc": mean_fold,
            "pooled_vs_mean_fold_divergence_flag": bool(
                abs(pooled.point - mean_fold) > cfg["divergence_flag"]
            ),
            "default_alarm": _default_alarm_stats(scores, name),
            "per_class_auroc": per_class,
        }
    gate = _gate(head["A"]["auroc"], head["B"]["auroc"], head["diff"], cfg["gate"]["min_gap"])
    return {
        "monitors": monitors,
        "difference_B_minus_A": {
            "pooled_auroc": _ci(head["diff"]),
            "mean_per_fold_auroc": monitors["B"]["mean_per_fold_auroc"]
            - monitors["A"]["mean_per_fold_auroc"],
        },
        "gate": gate,
        "flags": {
            "baseline_B_below_sanity_auroc": bool(head["B"]["auroc"].point < cfg["sanity_min_auroc_b"]),
            "sanity_min_auroc_b": cfg["sanity_min_auroc_b"],
        },
    }


def _pooled_aurocs(scores: pd.DataFrame) -> dict:
    y = scores["label"].to_numpy()
    a = auroc(y, scores["score_A"].to_numpy())
    b = auroc(y, scores["score_B"].to_numpy())
    return {"auroc_A": a, "auroc_B": b, "gap_B_minus_A": b - a}


# ---------------------------------------------------------------- orchestration


def default_monitor_factory(cfg: dict) -> MonitorFactory:
    esn = cfg["esn"]
    return lambda: ESNBaseline(
        channels=tuple(esn["channels"]), K=esn["K"], seed=esn["seed"], fa_budget=esn["fa_budget"]
    )


def git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def run_e1(
    episodes: pd.DataFrame, cfg: dict, make_monitor: MonitorFactory | None = None
) -> tuple[dict, pd.DataFrame]:
    make_monitor = make_monitor or default_monitor_factory(cfg)
    plan_args = {
        "source_corpus": cfg["source_corpus"],
        "target_corpus": cfg["target_corpus"],
        "target_family": cfg["target_family"],
        "k": cfg["k"],
        "val_fraction": cfg["val_fraction"],
    }
    plans = plan_grouped_folds(episodes, fold_seed=cfg["fold_seed"], **plan_args)
    scores, fold_info = score_plans(episodes, plans, make_monitor)
    result = {
        "what": "SP3 E1: transfer collapse of the paper's esn_cusum_max, task-disjoint protocol",
        "episode_score": EPISODE_SCORE_DEFINITION,
        "standardisation": STANDARDISATION,
        "config": cfg,
        "git": git_state(),
        "reference": cfg["reference"],
        "counts": {
            "eval_runs": len(scores),
            "healthy": int((scores["label"] == 0).sum()),
            "failed": int((scores["label"] == 1).sum()),
            "per_fold": fold_info,
        },
        **summarise(scores, cfg),
    }

    sens = cfg["sensitivity"]
    sensitivity = {}
    if sens.get("cluster_bootstrap"):
        head = _headline(scores, "task_group", cfg)
        sensitivity["task_group_cluster_bootstrap"] = {
            name: {metric: _ci(ci) for metric, ci in head[name].items()} for name in MONITORS
        } | {"paired_auroc_B_minus_A": _ci(head["diff"])}
    seed_runs = {}
    for fold_seed in sens.get("fold_seeds", []):
        seed_plans = plan_grouped_folds(episodes, fold_seed=fold_seed, **plan_args)
        seed_runs[str(fold_seed)] = _pooled_aurocs(score_plans(episodes, seed_plans, make_monitor)[0])
    sensitivity["fold_seeds"] = seed_runs
    if sens.get("leave_one_task_group_out"):
        lotgo_plans = plan_leave_one_group_out(
            episodes,
            source_corpus=cfg["source_corpus"],
            target_corpus=cfg["target_corpus"],
            val_fraction=cfg["val_fraction"],
            seed=cfg["fold_seed"],
        )
        lotgo = score_plans(episodes, lotgo_plans, make_monitor)[0]
        sensitivity["leave_one_task_group_out"] = _pooled_aurocs(lotgo) | {"n_folds": len(lotgo_plans)}
    result["sensitivity"] = sensitivity
    return result, scores


def _to_json(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def write_outputs(result: dict, scores: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "e1_collapse.json"
    csv_path = out_dir / "e1_episode_scores.csv"
    json_path.write_text(json.dumps(result, indent=2, default=_to_json) + "\n")
    scores[CSV_COLUMNS].to_csv(csv_path, index=False)
    return json_path, csv_path
