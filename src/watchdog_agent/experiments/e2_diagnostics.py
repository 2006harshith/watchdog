"""SP5 Step 3b, E2 diagnostics (interpretation only; the pre-registered gate is not revisited).

Why the llama ESN scores 0.846 in E2 (ESN_i, headline) but 0.646 in SP4 4d (monitor A, all classes).
The two differ in class mix, in pooling (5 fold models z-scored and pooled vs 1 model, raw) and in
fit pool (ollama7b healthy runs outside the test fold's task groups vs outside all llama groups).
Each row below swaps one of those; per-fold columns remove pooling from the metric itself. Two more
rows ask where the context_corruption signal lives: the ESN without its surprisal channel u, and
MS-raw with and without its surprisal features.
"""

import copy
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn.monitor import Run
from watchdog_agent.evaluation import matched_step_auroc, prefix_eval_set
from watchdog_agent.experiments.e1 import (
    MonitorFactory,
    default_monitor_factory,
    git_state,
    plan_grouped_folds,
)
from watchdog_agent.experiments.e2 import (
    Scores,
    build_test_set,
    build_training_set,
    default_esn_factory,
    esn_scores,
    evaluate,
    feature_columns,
    fit_and_score,
    to_score_dict,
)
from watchdog_agent.experiments.sp4_audit import standardised_running_max
from watchdog_agent.features import prefix_feature_table
from watchdog_agent.splits import leave_one_family_out

REQUIRED_KEYS = {"what", "git", "family", "rows", "counts"}
FAMILY = "llama"


def restrict_classes(runs: pd.DataFrame, classes: Sequence[str]) -> pd.DataFrame:
    return runs[runs["failure_class"].isna() | runs["failure_class"].isin(classes)]


def without_channel(cfg: dict, channel: str) -> dict:
    out = copy.deepcopy(cfg)
    out["esn"]["channels"] = [c for c in out["esn"]["channels"] if c != channel]
    return out


def per_fold_mean_auroc(scores: Scores, runs: pd.DataFrame, folds: list[list[str]], checkpoints: Sequence[int]) -> dict:
    """Matched-step AUROC inside each fold, then the unweighted mean over folds. No score is ever
    compared with a score from another fold, so scale differences between fold models cannot matter.
    A fold with a one-class checkpoint is skipped and reported (as SP4 did)."""
    per_fold = []
    for f, uids in enumerate(folds):
        sub = runs[runs["uid"].isin(set(uids))]
        sets = [prefix_eval_set(sub, t) for t in checkpoints]
        if not all(s.n_pos and s.n_neg for s in sets):
            per_fold.append({"fold": f, "n_runs": len(sub), "skipped": "one-class checkpoint"})
            continue
        per_fold.append({"fold": f, "n_runs": len(sub), "auroc": matched_step_auroc(scores, sub, checkpoints).mean})
    vals = [p["auroc"] for p in per_fold if "auroc" in p]
    return {"mean": float(np.mean(vals)) if vals else float("nan"), "n_folds_used": len(vals), "per_fold": per_fold}


def sp4_fold_scores(
    episodes: pd.DataFrame, e1_cfg: dict, checkpoints: Sequence[int], make_monitor: MonitorFactory
) -> tuple[Scores, Scores, list[list[str]]]:
    """SP4 4d's monitor A (E1 folds and pools), each fold model scoring its own eval runs: the raw
    running max and SP4's per-fold, per-checkpoint standardised version, from the same fitted models.
    Mirrors sp4_audit.matched_step_esn_scores for A only, keeping the raw scores it discards."""
    plans = plan_grouped_folds(
        episodes, source_corpus=e1_cfg["source_corpus"], target_corpus=e1_cfg["target_corpus"],
        target_family=e1_cfg["target_family"], k=e1_cfg["k"], fold_seed=e1_cfg["fold_seed"],
        val_fraction=e1_cfg["val_fraction"],
    )
    frame = episodes.set_index("uid")

    def to_run(uid: str) -> Run:
        cls = frame.at[uid, "failure_class"]
        return Run(uid, frame.at[uid, "steps_parsed"], None if pd.isna(cls) else cls)

    raw, std = {}, {}
    for plan in plans:
        monitor = make_monitor()
        monitor.fit([to_run(u) for u in plan.fit["A"]], [to_run(u) for u in plan.val["A"]])
        val_streams = [monitor.score(to_run(u)).per_step for u in plan.val["A"]]
        for uid in plan.eval_uids:
            stream = monitor.score(to_run(uid)).per_step
            raw[uid] = np.maximum.accumulate(np.asarray(stream, dtype=float))
            std[uid], _ = standardised_running_max(stream, val_streams, checkpoints)
    return raw, std, [p.eval_uids for p in plans]


def e2_healthy_pool(episodes: pd.DataFrame, family: str, cfg: dict, variant: str) -> pd.DataFrame:
    """The healthy pool E2's ESN variant was fitted on (as in e2.run_family)."""
    train_uids, _, _ = leave_one_family_out(episodes, family, eval_only_families=tuple(cfg["families"]["eval_only"]))
    pool = episodes[episodes["uid"].isin(set(train_uids)) & episodes["failure_class"].isna()]
    corpora = cfg["esn"]["variants"][variant]["corpora"]
    return pool if corpora == "all" else pool[pool["corpus"].isin(corpora)]


def _summary(scores: Scores, runs: pd.DataFrame, folds: list[list[str]], cfg: dict) -> dict:
    cl, n, primary = cfg["classes"], cfg["bootstrap"]["n_secondary"], cfg["checkpoints"]["primary"]
    all_classes = cl["headline"] + cl["error_visible"] + cl["api"]

    def mean_iv(classes):
        m = evaluate(scores, runs, classes, cfg, n).get("mean_primary")
        return m if m is not None else {"point": float("nan"), "skipped": "no mean over the primary checkpoints"}

    per_class = {}
    for c in all_classes:
        m = evaluate(scores, runs, [c], cfg, n).get("mean_primary")
        if m is not None:
            per_class[c] = m["point"]
    return {
        "headline": mean_iv(cl["headline"]),
        "all_classes": mean_iv(all_classes),
        "per_fold_headline": per_fold_mean_auroc(scores, restrict_classes(runs, cl["headline"]), folds, primary),
        "per_fold_all_classes": per_fold_mean_auroc(scores, restrict_classes(runs, all_classes), folds, primary),
        "per_class": per_class,
    }


def ms_raw_scores(table: pd.DataFrame, episodes: pd.DataFrame, cfg: dict) -> dict[str, Scores]:
    """E2's MS-raw arm, refitted with its surprisal features removed, and with only them."""
    train = build_training_set(table, episodes, FAMILY, cfg)
    test = build_test_set(table, episodes, FAMILY, cfg, cfg["recalibration"]["target_cross_fit"]["seed"])
    cols = feature_columns("MS", cfg)
    surprisal = [c for c in cols if "surprisal" in c]
    variants = {"e2_MS_raw": cols, "e2_MS_raw_no_surprisal": [c for c in cols if c not in surprisal],
                "e2_MS_raw_surprisal_only": surprisal}
    out = {}
    for name, use in variants.items():
        _, s = fit_and_score(train, test.X["raw"], train.X["raw"], use, cfg["model"]["primary"])
        out[name] = to_score_dict(s, test.runs)
    return out


def run_e2_diagnostics(
    episodes: pd.DataFrame,
    cfg: dict,
    e1_cfg: dict,
    make_esn: Callable | None = None,
    make_monitor: MonitorFactory | None = None,
    table: pd.DataFrame | None = None,
    git: dict | None = None,
) -> dict:
    make_monitor = make_monitor or default_monitor_factory(e1_cfg)
    runs = episodes[episodes["family"] == FAMILY]
    score_t = cfg["checkpoints"]["score"]

    sp4_raw, sp4_std, folds = sp4_fold_scores(episodes, e1_cfg, score_t, make_monitor)
    scores: dict[str, Scores] = {"sp4_A_standardised": sp4_std, "sp4_A_raw": sp4_raw}

    pool = e2_healthy_pool(episodes, FAMILY, cfg, "i")
    no_u = without_channel(cfg, "u")
    scores["e2_ESN_i"], info_i = esn_scores(pool, runs, cfg, make_esn or default_esn_factory(cfg))
    scores["e2_ESN_i_no_u"], info_no_u = esn_scores(pool, runs, no_u, make_esn or default_esn_factory(no_u))

    if table is None:
        needed = sorted(set(cfg["checkpoints"]["train"]) | set(score_t))
        table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    scores |= ms_raw_scores(table, episodes, cfg)

    return {
        "what": "SP5 Step 3b, interpretation only: one-factor swaps between SP4 4d's ESN (0.646) and "
        "E2's ESN_i (0.846) on llama, plus surprisal ablations of ESN_i and MS-raw. Mean matched-step "
        "AUROC over the primary checkpoints; pooled rows carry task-group cluster CIs (n_secondary), "
        "per-fold rows are the unweighted mean of within-fold AUROCs over SP4's E1 folds (no CI).",
        "git": git if git is not None else git_state(),
        "family": FAMILY,
        "counts": {
            "llama_runs": len(runs),
            "sp4_folds": [len(f) for f in folds],
            "e2_ESN_i_pool": info_i,
            "e2_ESN_i_no_u_pool": info_no_u,
        },
        "rows": {name: _summary(s, runs, folds, cfg) for name, s in scores.items()},
    }


def check_reproduction(result: dict, e2: dict, sp4: dict, tol: float = 1e-9) -> dict[str, tuple[float, float]]:
    """The three rows that re-run a published number must match it exactly, or the diagnostic is
    not measuring the same model and every swap is meaningless."""
    arms = e2["families"][FAMILY]["arms"]
    pairs = {
        "e2_ESN_i": (result["rows"]["e2_ESN_i"]["headline"]["point"], arms["ESN_i"]["headline"]["mean_primary"]["point"]),
        "e2_MS_raw": (result["rows"]["e2_MS_raw"]["headline"]["point"], arms["MS-raw"]["headline"]["mean_primary"]["point"]),
        "sp4_A_standardised": (result["rows"]["sp4_A_standardised"]["all_classes"]["point"],
                               sp4["d_esn_matched_step"]["monitors"]["A"]["mean_over_primary"]["point"]),
    }
    bad = {k: v for k, v in pairs.items() if not abs(v[0] - v[1]) <= tol}
    if bad:
        raise AssertionError(f"diagnostic does not reproduce the published number: {bad}")
    return pairs
