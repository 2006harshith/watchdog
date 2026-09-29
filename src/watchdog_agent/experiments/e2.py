"""SP5 E2 ablation (GATE): feature-set x transform arms against the author's ESN on a held-out family.

Pre-registered in configs/e2.yaml (docs/decisions.md, "SP5 Decisions and GATE"). Per test family:
one HistGradientBoosting model per arm, trained on the leave_one_family_out pool (gemini never, D6)
and scored at matched steps on the test family. Nothing is tuned on a test family.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from watchdog_agent.baselines.esn.monitor import ESNBaseline, Run
from watchdog_agent.evaluation import (
    matched_step_auroc,
    matched_step_cluster_ci,
    paired_matched_step_diff,
    prefix_eval_set,
)
from watchdog_agent.experiments.e1 import git_state, split_by_task_group
from watchdog_agent.experiments.sp4_audit import indistinguishable_positives
from watchdog_agent.features import FEATURE_GROUPS, prefix_feature_table
from watchdog_agent.metrics import tpr_at_fpr
from watchdog_agent.recalibration import (
    cross_fitted_target_transform,
    fit_healthy_reference,
    percentile_transform,
    zscore_transform,
)
from watchdog_agent.splits import leave_one_family_out

REQUIRED_KEYS = {
    "what", "config", "git", "families", "gate", "secondary_expectations", "interpretation_only",
}
ESN_VARIANTS = ("ESN_i", "ESN_ii")
Scores = dict[str, np.ndarray]  # uid -> per-step score (index = step)


# ---------------------------------------------------------------- data


class TrainingSet:
    def __init__(self, X: dict[str, pd.DataFrame], y: np.ndarray, w: np.ndarray, runs: pd.DataFrame, counts: dict):
        self.X, self.y, self.w, self.runs, self.counts = X, y, w, runs, counts


class TestSet:
    def __init__(self, X: dict[str, pd.DataFrame], runs: pd.DataFrame):
        self.X, self.runs = X, runs


def _rows(table: pd.DataFrame, uids, checkpoints) -> pd.DataFrame:
    idx = table.index
    return table[idx.get_level_values("uid").isin(set(uids)) & idx.get_level_values("t").isin(checkpoints)]


def build_training_set(table: pd.DataFrame, episodes: pd.DataFrame, test_family: str, cfg: dict) -> TrainingSet:
    """Prefix rows t in checkpoints.train of the leave_one_family_out pool, D8 labels, weights
    1 / rows-per-run, in three versions: raw, healthy-percentile, z-score (sensitivity)."""
    train_uids, _, n_overlap = leave_one_family_out(
        episodes, test_family, eval_only_families=tuple(cfg["families"]["eval_only"])
    )
    pool = episodes[episodes["uid"].isin(set(train_uids))]
    info = pool.set_index("uid")
    rows = _rows(table, train_uids, cfg["checkpoints"]["train"])
    uid = rows.index.get_level_values("uid")
    t = rows.index.get_level_values("t").to_numpy()
    failed = info.loc[uid, "failure_class"].notna().to_numpy()
    tau = info.loc[uid, "tau"].to_numpy(dtype=float)
    # D8: 1 = failed with tau <= t, 0 = healthy, -1 = failed with tau > t (not yet failed: excluded).
    label = np.where(~failed, 0, np.where(tau <= t, 1, -1))
    keep = label >= 0
    n_d8 = int((~keep).sum())
    rows, label = rows[keep], label[keep]

    by = cfg["recalibration"]["source_reference_by"]
    reference = fit_healthy_reference(rows, pool, by=by)
    group = info.loc[rows.index.get_level_values("uid"), by].to_numpy()
    tt = rows.index.get_level_values("t").to_numpy()
    has_ref = np.array([(g, int(x)) in reference.n_healthy_rows for g, x in zip(group, tt)], dtype=bool)
    dropped = pd.Series(list(zip(group[~has_ref], tt[~has_ref]))).value_counts()
    if not cfg["training"]["drop_rows_without_healthy_reference"] and (~has_ref).any():
        raise ValueError(f"training rows without a healthy reference: {dropped.to_dict()}")
    rows, label = rows[has_ref], label[has_ref]

    run_of_row = rows.index.get_level_values("uid")
    # map(value_counts): each row gets its run's row count, so every run's weights sum to 1.
    weight = 1.0 / pd.Series(run_of_row).map(pd.Series(run_of_row).value_counts()).to_numpy(dtype=float)
    X = {
        "raw": rows,
        "pct": percentile_transform(rows, reference, pool),
        "z": zscore_transform(rows, reference, pool, cfg["recalibration"]["zscore_sensitivity"]["std_floor"]),
    }
    counts = {
        "pool_runs": len(pool),
        "pool_runs_dropped_for_task_overlap": int(n_overlap),
        "pool_families": sorted(pool["family"].unique()),
        "rows": len(rows),
        "positive_rows": int(label.sum()),
        "negative_rows": int((label == 0).sum()),
        "rows_excluded_d8_tau_after_t": n_d8,
        "rows_dropped_no_healthy_reference": {f"{g}@t={x}": int(n) for (g, x), n in dropped.items()},
    }
    return TrainingSet(X, label.astype(int), weight, pool, counts)


def build_test_set(
    table: pd.DataFrame, episodes: pd.DataFrame, test_family: str, cfg: dict, cross_fit_seed: int
) -> TestSet:
    """The test family's rows at checkpoints.score: raw, and re-expressed by task-group cross-fitting."""
    runs = episodes[episodes["family"] == test_family]
    rows = _rows(table, runs["uid"], cfg["checkpoints"]["score"])
    k = cfg["recalibration"]["target_cross_fit"]["k"]
    floor = cfg["recalibration"]["zscore_sensitivity"]["std_floor"]
    X = {
        "raw": rows,
        "pct": cross_fitted_target_transform(rows, runs, k=k, seed=cross_fit_seed),
        "z": cross_fitted_target_transform(rows, runs, k=k, seed=cross_fit_seed, method="zscore", std_floor=floor),
    }
    return TestSet(X, runs)


# ---------------------------------------------------------------- models


def feature_columns(feature_set: str, cfg: dict) -> list[str]:
    return [c for group in cfg["arms"]["feature_sets"][feature_set] for c in FEATURE_GROUPS[group]]


class Fitted:
    def __init__(self, model: HistGradientBoostingClassifier, cols: list[str], dropped: list[str]):
        self.model, self.cols, self.dropped = model, cols, dropped

    def score(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(X[self.cols])[:, 1], index=X.index)


def fit_and_score(
    train: TrainingSet, test_X: pd.DataFrame, train_X: pd.DataFrame, cols: list[str], params: dict
) -> tuple[Fitted, pd.Series]:
    # A column that is NaN on every training row carries no information, and sklearn 1.9.1's HGB
    # binning crashes on it ("window shape cannot be larger than input array shape"); it is left out
    # of that model and recorded.
    dropped = [c for c in cols if train_X[c].isna().all()]
    used = [c for c in cols if c not in set(dropped)]
    model = HistGradientBoostingClassifier(**params)
    model.fit(train_X[used], train.y, sample_weight=train.w)
    fitted = Fitted(model, used, dropped)
    return fitted, fitted.score(test_X)


def to_score_dict(series: pd.Series, runs: pd.DataFrame) -> Scores:
    """(uid, t) -> score series as uid -> array over steps, NaN where no score was produced."""
    out = {uid: np.full(int(T), np.nan) for uid, T in zip(runs["uid"], runs["T"])}
    for (uid, t), v in series.items():
        out[uid][t] = v
    return out


def logprob_coverage(runs: pd.DataFrame) -> float:
    steps = [s for steps in runs["steps_parsed"] for s in steps]
    return float(np.mean([bool(s.get("token_logprobs")) for s in steps])) if steps else 0.0


def esn_channels(fit_pool: pd.DataFrame, test_runs: pd.DataFrame, cfg: dict) -> list[str]:
    esn = cfg["esn"]
    coverage = min(logprob_coverage(fit_pool), logprob_coverage(test_runs))
    return [c for c in esn["channels"] if not (c == "u" and coverage < esn["drop_u_below_logprob_coverage"])]


def default_esn_factory(cfg: dict) -> Callable[[list[str]], ESNBaseline]:
    esn = cfg["esn"]
    return lambda channels: ESNBaseline(channels=tuple(channels), K=esn["K"], seed=esn["seed"], fa_budget=esn["fa_budget"])


def esn_scores(
    healthy_pool: pd.DataFrame, test_runs: pd.DataFrame, cfg: dict, make_esn: Callable
) -> tuple[Scores, dict]:
    """One ESN fitted on healthy_pool (whole-task-group fit/val split), scored on every test run:
    per-step score = running max of the CUSUM stream over steps 0..t (matched-step, unstandardised)."""
    channels = esn_channels(healthy_pool, test_runs, cfg)
    fit_uids, val_uids = split_by_task_group(healthy_pool, cfg["esn"]["val_fraction"], cfg["esn"]["val_seed"])
    frame = pd.concat([healthy_pool, test_runs]).drop_duplicates("uid").set_index("uid")

    def to_run(uid: str) -> Run:
        cls = frame.at[uid, "failure_class"]
        return Run(uid, frame.at[uid, "steps_parsed"], None if pd.isna(cls) else cls)

    monitor = make_esn(channels)
    monitor.fit([to_run(u) for u in fit_uids], [to_run(u) for u in val_uids])
    # np.maximum.accumulate: running maximum, so the score at t only uses steps 0..t.
    scores = {u: np.maximum.accumulate(monitor.score(to_run(u)).per_step) for u in test_runs["uid"]}
    return scores, {"channels": channels, "n_fit": len(fit_uids), "n_val": len(val_uids),
                    "corpora": sorted(healthy_pool["corpus"].unique())}


# ---------------------------------------------------------------- evaluation


def _iv(interval) -> dict:
    return interval._asdict()


def evaluate(scores: Scores, runs: pd.DataFrame, classes: list[str], cfg: dict, n: int) -> dict:
    """Matched-step AUROC of one arm on one class subset (vs all healthy runs): per checkpoint with a
    task-group cluster CI, the mean over the primary checkpoints with its CI, and TPR at 5% FPR."""
    sub = runs[runs["failure_class"].isna() | runs["failure_class"].isin(classes)]
    seed, primary = cfg["bootstrap"]["seed"], cfg["checkpoints"]["primary"]
    sets = {t: prefix_eval_set(sub, t) for t in cfg["checkpoints"]["score"]}
    out: dict = {
        "classes": list(classes),
        "n": {str(t): {"n_pos": s.n_pos, "n_healthy": s.n_neg,
                       "healthy_task_groups": len(set(s.task_groups[s.y == 0])),
                       "positive_task_groups": len(set(s.task_groups[s.y == 1]))} for t, s in sets.items()},
    }
    valid = [t for t, s in sets.items() if s.n_pos and s.n_neg]
    if not valid:
        return out | {"skipped": "no positives or no healthy runs at any checkpoint"}
    out["auroc"] = {str(t): v for t, v in matched_step_auroc(scores, sub, valid).per_checkpoint.items()}
    alpha = cfg["evaluation"]["tpr_at_fpr_alpha"]
    out["tpr_at_fpr"] = {
        str(t): tpr_from_set(scores, sets[t], alpha) for t in valid
    }
    try:
        ci = matched_step_cluster_ci(scores, sub, valid, n, seed=seed)
        out["auroc_ci"] = {str(t): _iv(iv) for t, iv in ci.per_checkpoint.items()}
    except ValueError as e:  # too few task groups of one class for a cluster CI
        out["auroc_ci_unavailable"] = str(e)
    if all(t in valid for t in primary):
        try:
            out["mean_primary"] = _iv(matched_step_cluster_ci(scores, sub, primary, n, seed=seed).mean)
        except ValueError as e:
            out["mean_primary"] = {"point": matched_step_auroc(scores, sub, primary).mean, "ci_unavailable": str(e)}
    return out


def tpr_from_set(scores: Scores, cset, alpha: float) -> float:
    s = np.array([scores[u][cset.t] for u in cset.uids])
    return float(tpr_at_fpr(cset.y, s, alpha))


def paired(scores_a: Scores, scores_b: Scores, runs: pd.DataFrame, classes: list[str], cfg: dict, n: int) -> dict:
    """B - A, both on the same resampled task groups, over the primary checkpoints."""
    sub = runs[runs["failure_class"].isna() | runs["failure_class"].isin(classes)]
    d = paired_matched_step_diff(scores_a, scores_b, sub, cfg["checkpoints"]["primary"], n, seed=cfg["bootstrap"]["seed"])
    return {"per_checkpoint": {str(t): _iv(iv) for t, iv in d.per_checkpoint.items()}, "mean": _iv(d.mean)}


def decide_gate(paired_result: dict, cfg: dict) -> dict:
    gate = cfg["gate"]
    lo = paired_result["mean"]["lo"]
    passed = bool(lo > gate["pass_if_lower_bound_above"])
    return {
        "test_family": gate["test_family"],
        "arm": gate["arm"],
        "baseline": gate["baseline"],
        "metric": gate["metric"],
        "threshold": gate["pass_if_lower_bound_above"],
        "difference": paired_result["mean"],
        "pass": passed,
        "outcome": "PASS" if passed else "FAIL",
        "on_fail": None if passed else gate["on_fail"],
    }


def _arm_block(scores: Scores, runs: pd.DataFrame, cfg: dict, excluded: set[str]) -> dict:
    cl, b = cfg["classes"], cfg["bootstrap"]
    all_classes = cl["headline"] + cl["error_visible"] + cl["api"]
    return {
        "headline": evaluate(scores, runs, cl["headline"], cfg, b["n_primary"]),
        "error_visible": evaluate(scores, runs, cl["error_visible"], cfg, b["n_secondary"]),
        "api": evaluate(scores, runs, cl["api"], cfg, b["n_secondary"]),
        "per_class": {c: evaluate(scores, runs, [c], cfg, b["n_secondary"]) for c in all_classes},
        "headline_excluding_tool_level_indistinguishable": evaluate(
            scores, runs[~runs["uid"].isin(excluded)], cl["headline"], cfg, b["n_secondary"]
        ),
    }


def _mean_point(block: dict) -> float:
    m = block["headline"].get("mean_primary")
    return float("nan") if m is None else float(m["point"])


# ---------------------------------------------------------------- orchestration


def run_family(
    table: pd.DataFrame, episodes: pd.DataFrame, family: str, cfg: dict, make_esn: Callable
) -> tuple[dict, pd.DataFrame]:
    train = build_training_set(table, episodes, family, cfg)
    test = build_test_set(table, episodes, family, cfg, cfg["recalibration"]["target_cross_fit"]["seed"])
    runs = test.runs
    primary_params = cfg["model"]["primary"]

    series: dict[str, pd.Series] = {}
    models = {}
    for fs in cfg["arms"]["feature_sets"]:
        cols = feature_columns(fs, cfg)
        for tr in cfg["arms"]["transforms"]:
            models[f"{fs}-{tr}"], series[f"{fs}-{tr}"] = fit_and_score(train, test.X[tr], train.X[tr], cols, primary_params)
    scores: dict[str, Scores] = {arm: to_score_dict(s, runs) for arm, s in series.items()}

    healthy_pool = train.runs[train.runs["failure_class"].isna()]
    variants = cfg["esn"]["variants"]
    esn_info = {}
    for name, key in zip(ESN_VARIANTS, ("i", "ii")):
        corpora = variants[key]["corpora"]
        pool = healthy_pool if corpora == "all" else healthy_pool[healthy_pool["corpus"].isin(corpora)]
        scores[name], esn_info[name] = esn_scores(pool, runs, cfg, make_esn)

    excluded = set().union(*(indistinguishable_positives(runs, t) for t in cfg["checkpoints"]["score"]))
    arms = {arm: _arm_block(s, runs, cfg, excluded) for arm, s in scores.items()}
    # ESN_best: the higher headline mean on the test family (decided on the test result, which is
    # conservative toward TL-pct); a tie goes to (i), E1's setting; NaN (no mean) ranks last.
    best = max(ESN_VARIANTS, key=lambda v: (np.nan_to_num(_mean_point(arms[v]), nan=-np.inf), v == "ESN_i"))
    scores["ESN_best"] = scores[best]

    head, n = cfg["classes"]["headline"], cfg["bootstrap"]["n_primary"]
    paired_out = {
        "TL-pct_vs_ESN_best": paired(scores["ESN_best"], scores["TL-pct"], runs, head, cfg, n),
        "TL-pct_vs_TL-raw": paired(scores["TL-raw"], scores["TL-pct"], runs, head, cfg, n),
        "TL-pct_vs_MS-pct": paired(scores["MS-pct"], scores["TL-pct"], runs, head, cfg, n),
    }

    # Sensitivity, not gated: z-score arm, two other HGB configs, the target cross-fit seed.
    n2, tl = cfg["bootstrap"]["n_secondary"], feature_columns("TL", cfg)
    sens = {}
    _, s = fit_and_score(train, test.X["z"], train.X["z"], tl, primary_params)
    sens["TL-z"] = evaluate(to_score_dict(s, runs), runs, head, cfg, n2)
    for name, params in cfg["model"]["sensitivity"].items():
        _, s = fit_and_score(train, test.X["pct"], train.X["pct"], tl, params)
        sens[f"hgb_{name}"] = evaluate(to_score_dict(s, runs), runs, head, cfg, n2)
    seeds = {}
    for seed in cfg["model"]["seed_sensitivity"]["target_cross_fit_seeds"]:
        X = build_test_set(table, episodes, family, cfg, seed).X["pct"]
        s = models["TL-pct"].score(X)
        seeds[str(seed)] = evaluate(to_score_dict(s, runs), runs, head, cfg, n2)
    sens["target_cross_fit_seeds"] = seeds

    frame = _score_frame(scores, runs, test.X["raw"].index, family)
    result = {
        "counts": {"train": train.counts, "test_runs": len(runs),
                   "tool_level_indistinguishable_positives_excluded": len(excluded)},
        "arms": arms,
        "esn": {"best": best, **esn_info},
        "paired": paired_out,
        "sensitivity": sens,
    }
    pi_cfg = cfg["evaluation"]["permutation_importance"]
    if family == pi_cfg["family"]:
        result["_permutation_importance"] = _permutation_importance(models[pi_cfg["arm"]], test, runs, cfg)
    result["all_nan_training_columns_left_out"] = {arm: f.dropped for arm, f in models.items() if f.dropped}
    return result, frame


def _score_frame(scores: dict[str, Scores], runs: pd.DataFrame, index: pd.MultiIndex, family: str) -> pd.DataFrame:
    info = runs.set_index("uid")
    rows = []
    for uid, t in index:
        cls, tau = info.at[uid, "failure_class"], info.at[uid, "tau"]
        label = 0 if pd.isna(cls) else (1 if tau <= t else np.nan)
        row = {"uid": uid, "family": family, "t": int(t), "label": label, "class": "healthy" if pd.isna(cls) else cls}
        row |= {arm: s[uid][t] for arm, s in scores.items() if arm != "ESN_best"}
        rows.append(row)
    return pd.DataFrame(rows)


def _permutation_importance(fitted: Fitted, test: TestSet, runs: pd.DataFrame, cfg: dict) -> dict:
    """INTERPRETATION ONLY, on the test family: which TL-pct features the llama scores lean on.
    Computed after every result above and never read by anything that chooses."""
    pi = cfg["evaluation"]["permutation_importance"]
    head = cfg["classes"]["headline"]
    sub = runs[runs["failure_class"].isna() | runs["failure_class"].isin(head)]
    keys, y = [], []
    for t in cfg["checkpoints"]["primary"]:
        s = prefix_eval_set(sub, t)
        keys += [(u, t) for u in s.uids]
        y += s.y.tolist()
    transform = pi["arm"].rsplit("-", 1)[1]
    cols = fitted.cols
    X = test.X[transform].loc[keys, cols]
    r = permutation_importance(fitted.model, X, np.array(y), scoring="roc_auc", n_repeats=pi["n_repeats"],
                               random_state=pi["random_state"])
    order = np.argsort(-r.importances_mean)[: pi["top_k"]]
    return {
        "note": "computed on the TEST family (llama) for interpretation only; feeds back into no choice",
        "arm": pi["arm"], "rows": len(keys), "scoring": "roc_auc, pooled over the primary checkpoints",
        "top": [{"feature": cols[i], "mean": float(r.importances_mean[i]), "std": float(r.importances_std[i])}
                for i in order],
    }


def run_e2(
    episodes: pd.DataFrame,
    cfg: dict,
    make_esn: Callable | None = None,
    table: pd.DataFrame | None = None,
    git: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    make_esn = make_esn or default_esn_factory(cfg)
    if table is None:
        needed = sorted(set(cfg["checkpoints"]["train"]) | set(cfg["checkpoints"]["score"]))
        table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)

    families, frames, importance = {}, [], {}
    for family in cfg["families"]["test"]:
        res, frame = run_family(table, episodes, family, cfg, make_esn)
        if "_permutation_importance" in res:
            importance[family] = res.pop("_permutation_importance")
        families[family] = res
        frames.append(frame)

    gate_family = families[cfg["gate"]["test_family"]]
    expectations = []
    for exp in cfg["secondary_expectations"]:
        arms = families[exp["family"]]["arms"]
        a, b = _mean_point(arms[exp["a"]]), _mean_point(arms[exp["b"]])
        holds = a > b if exp["expect"] == "a_gt_b" else a >= b
        expectations.append(exp | {"a_mean_primary": a, "b_mean_primary": b, "holds": bool(holds)})

    result = {
        "what": "SP5 E2 ablation: HistGradientBoosting per feature set x transform vs the author's ESN, "
        "held-out family, matched-step AUROC with task-group cluster CIs",
        "config": cfg,
        "git": git if git is not None else git_state(),
        "gate": decide_gate(gate_family["paired"]["TL-pct_vs_ESN_best"], cfg),
        "secondary_expectations": expectations,
        "families": families,
        "interpretation_only": {"permutation_importance": importance},
    }
    return result, pd.concat(frames, ignore_index=True)
