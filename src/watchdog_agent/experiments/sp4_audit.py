"""SP4 Step 4: feature diagnostics. No monitor is trained here; no llama/gemini failure label is
used except to score the ESN baseline (4d), exactly as E1 did.

4a family predictability (healthy runs only, label = family), 4b univariate TL AUROC on qwen,
4c the tool-level ceiling, 4d the ESN at matched steps.
"""

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from watchdog_agent.baselines.esn.monitor import Run
from watchdog_agent.evaluation import CheckpointSet, prefix_eval_set
from watchdog_agent.experiments.e1 import MonitorFactory, plan_grouped_folds
from watchdog_agent.metrics import auroc

# ---------------------------------------------------------------- 4a


def family_predictability_auroc(
    X: pd.DataFrame, family: np.ndarray, groups: np.ndarray, n_splits: int = 5
) -> float:
    """Out-of-fold AUROC of a logistic regression telling two families apart, folds grouped by
    task group (a task is never in its own training fold, so task identity cannot stand in for
    family). Near 0.5 = the features do not identify the family.
    """
    oof = np.full(len(X), np.nan)
    # GroupKFold: each group lands in exactly one test fold; deterministic (no shuffle).
    for train, test in GroupKFold(n_splits=n_splits).split(X, family, groups):
        model = make_pipeline(
            # add_indicator: a 0/1 "was missing" column per feature with any NaN, so missingness
            # (e.g. no logprobs) is visible to the model as it would be to XGBoost in SP5.
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=5000),
        )
        model.fit(X.iloc[train], family[train])
        oof[test] = model.predict_proba(X.iloc[test])[:, 1]
    return auroc(family, oof)


def _with_nan_lowest(values: np.ndarray) -> np.ndarray:
    """NaN ranked below every observed value, so a univariate AUROC keeps every run."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    floor = (finite.min() - 1.0) if finite.size else 0.0
    return np.where(np.isnan(values), floor, values)


def top_family_features(X: pd.DataFrame, family: np.ndarray, k: int = 5) -> list[dict]:
    """The k features whose univariate AUROC for family is furthest from 0.5."""
    rows = [{"feature": c, "auroc": auroc(family, _with_nan_lowest(X[c].to_numpy()))} for c in X.columns]
    rows.sort(key=lambda r: -abs(r["auroc"] - 0.5))
    return rows[:k]


# ---------------------------------------------------------------- 4b


def univariate_auroc(table: pd.DataFrame, cset: CheckpointSet, feature: str) -> float:
    """AUROC of one feature's value at step cset.t over the checkpoint set (NaN ranked lowest)."""
    # .loc with a list of (uid, t) tuples: one row per run of the set, in cset order.
    values = table.loc[[(uid, cset.t) for uid in cset.uids], feature].to_numpy(dtype=float)
    return auroc(cset.y, _with_nan_lowest(values))


# ---------------------------------------------------------------- 4c


def tool_prefix_signature(steps: Sequence[dict], t: int) -> tuple:
    """Everything a tool-level monitor can see in steps 0..t: per step the call (name, args), the
    result and the error flag, or None for a step without a call. Timing and text are excluded.
    """
    sig = []
    for step in steps[: t + 1]:
        events = step.get("tool_events") or []
        sig.append(
            tuple(
                (e.get("name"), json.dumps(e.get("args"), sort_keys=True), e.get("result"), bool(e.get("is_error")))
                for e in events
            )
            or None
        )
    return tuple(sig)


def indistinguishable_positives(runs: pd.DataFrame, t: int) -> set[str]:
    """Positives at t (D8) whose tool-level prefix equals a healthy run's prefix in the same task
    group. A tool-level monitor gives both the same score, so it cannot rank the failed one higher.
    """
    cset = prefix_eval_set(runs, t)
    frame = runs.set_index("uid")
    healthy_sigs: dict[str, set] = {}
    for uid, y in zip(cset.uids, cset.y):
        if y == 0:
            healthy_sigs.setdefault(frame.at[uid, "task_group"], set()).add(
                tool_prefix_signature(frame.at[uid, "steps_parsed"], t)
            )
    return {
        uid
        for uid, y in zip(cset.uids, cset.y)
        if y == 1
        and tool_prefix_signature(frame.at[uid, "steps_parsed"], t)
        in healthy_sigs.get(frame.at[uid, "task_group"], set())
    }


def ceiling_counts(runs: pd.DataFrame, t: int) -> dict:
    """Counts of indistinguishable_positives at t, overall and per failure class."""
    cset = prefix_eval_set(runs, t)
    frame = runs.set_index("uid")
    hits = indistinguishable_positives(runs, t)
    by_class: dict[str, dict] = {}
    for uid, y in zip(cset.uids, cset.y):
        if y == 1:
            entry = by_class.setdefault(frame.at[uid, "failure_class"], {"n_pos": 0, "n_indistinguishable": 0})
            entry["n_pos"] += 1
            entry["n_indistinguishable"] += int(uid in hits)
    n_ind = sum(e["n_indistinguishable"] for e in by_class.values())
    return {
        "n_pos": cset.n_pos,
        "n_indistinguishable": n_ind,
        "share": n_ind / cset.n_pos if cset.n_pos else None,
        "by_class": dict(sorted(by_class.items())),
    }


# ---------------------------------------------------------------- 4d


def standardised_running_max(
    stream: np.ndarray, val_streams: Sequence[np.ndarray], checkpoints: Sequence[int]
) -> tuple[np.ndarray, dict[int, str]]:
    """Per-step score at each checkpoint t: the running max of the ESN-CUSUM stream over steps
    0..t, standardised by the mean/std of the same quantity on the fold model's healthy validation
    runs that have a step t. The matched-step analogue of E1's episode-score standardisation.

    Returns an array as long as the stream (NaN off the checkpoints) and a flag per checkpoint
    where the validation scores were constant (washout steps 0..2 always are): those are centred
    but not scaled.
    """
    # np.maximum.accumulate: running maximum along the array (value at i = max of 0..i).
    run_max = np.maximum.accumulate(np.asarray(stream, dtype=float))
    out = np.full(len(run_max), np.nan)
    flags: dict[int, str] = {}
    for t in checkpoints:
        if t >= len(run_max):
            continue
        val = np.array([np.maximum.accumulate(v)[t] for v in val_streams if len(v) > t])
        if val.size < 2:
            raise ValueError(f"fewer than 2 validation runs with a step {t}")
        mean, std = float(val.mean()), float(val.std())
        if std == 0.0:
            flags[t] = "constant_val_scores_centred_only"
            out[t] = run_max[t] - mean
        else:
            out[t] = (run_max[t] - mean) / std
    return out, flags


def matched_step_esn_scores(
    episodes: pd.DataFrame, e1_cfg: dict, checkpoints: Sequence[int], make_monitor: MonitorFactory
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict]]:
    """E1's folds and fit/val pools (plan_grouped_folds), each fold model's eval runs scored at
    matched steps. Returns {monitor: {uid: per-step score}} and per-fold info.
    """
    plans = plan_grouped_folds(
        episodes,
        source_corpus=e1_cfg["source_corpus"],
        target_corpus=e1_cfg["target_corpus"],
        target_family=e1_cfg["target_family"],
        k=e1_cfg["k"],
        fold_seed=e1_cfg["fold_seed"],
        val_fraction=e1_cfg["val_fraction"],
    )
    frame = episodes.set_index("uid")

    def to_run(uid: str) -> Run:
        cls = frame.at[uid, "failure_class"]
        return Run(uid, frame.at[uid, "steps_parsed"], None if pd.isna(cls) else cls)

    scores: dict[str, dict[str, np.ndarray]] = {"A": {}, "B": {}}
    info = []
    for plan in plans:
        fold_info: dict = {"fold": plan.fold, "eval_runs": len(plan.eval_uids)}
        for name in ("A", "B"):
            monitor = make_monitor()
            monitor.fit([to_run(u) for u in plan.fit[name]], [to_run(u) for u in plan.val[name]])
            val_streams = [monitor.score(to_run(u)).per_step for u in plan.val[name]]
            flags_seen: dict[int, str] = {}
            for uid in plan.eval_uids:
                s, flags = standardised_running_max(monitor.score(to_run(uid)).per_step, val_streams, checkpoints)
                scores[name][uid] = s
                flags_seen |= flags
            fold_info |= {
                f"n_fit_{name}": len(plan.fit[name]),
                f"n_val_{name}": len(plan.val[name]),
                f"constant_val_checkpoints_{name}": sorted(flags_seen),
            }
        info.append(fold_info)
    return scores, info
