"""SP6 E3 recalibration curve: alarms on a new model from n healthy target runs.

Pre-registered in configs/e3.yaml (docs/decisions.md, "SP6 Decisions and pre-registered
expectations"). No gate. For every target fold f and draw r, n healthy runs of the OTHER folds pay
for everything the monitor needs on the new model (percentile reference, ESN refit, threshold);
fold f's runs are only ever test runs. FA and TPR pool the 5 folds' test runs per draw.

The spread over draws is calibration-set variability only; test-set sampling is not in it.
"""

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2 import (
    Scores,
    build_training_set,
    default_esn_factory,
    esn_channels,
    feature_columns,
    fit_model,
    to_score_dict,
)

# _esn_running_max and _headline_mean are D-6's own scoring and metric: ESN-src must score its fit
# draws exactly as D-6 did, so the per-draw AUROCs can be checked against results/e2_closeout.json.
from watchdog_agent.experiments.e2_diagnostics import (
    _esn_running_max,
    _headline_mean,
    e2_healthy_pool,
    run_getter,
    subset_draws,
)
from watchdog_agent.features import prefix_feature_table
from watchdog_agent.recalibration import (
    cross_fitted_target_transform,
    fit_healthy_reference,
    percentile_transform,
)
from watchdog_agent.splits import grouped_kfold
from watchdog_agent.thresholds import (
    calibration_score,
    conformal_threshold,
    fa_tpr,
    first_alarm_steps,
    plugin_threshold,
)

REQUIRED_KEYS = {"what", "config", "git", "caveat", "families", "expectations"}
FAMILY_KEYS = {"counts", "n0", "esn_src", "reproduction", "cells", "comparisons"}
CAVEAT = "spread over draws = calibration-set variability only; test-set sampling is NOT in it"
RULES = {"conformal": conformal_threshold, "plugin": plugin_threshold}
COUNT_FIELDS = ("n_healthy", "n_healthy_alarm", "n_pos", "n_pos_alarm", "n_early")
GROUP_FIELDS = ("n_healthy", "n_healthy_alarm", "n_pos", "n_pos_alarm", "n_pos_alarm_pre_tau", "n_pos_alarm_post")
GROUP_META = ("family", "arm", "n", "rule", "alpha", "t", "draw", "fold")


# ---------------------------------------------------------------- Step 0


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


# ---------------------------------------------------------------- draws


@dataclass(frozen=True)
class Draw:
    uids: tuple[str, ...]  # the n target healthy runs; m = n arms calibrate on all of them
    half_a: tuple[str, ...]  # percentile reference (TL-pct, ALL-pct) or fit set (ESN-refit)
    half_b: tuple[str, ...]  # calibration runs of the m = n/2 arms; never in half_a


def eligible_uids(runs: pd.DataFrame, other_uids: Sequence[str], t: int) -> list[str]:
    """Healthy runs of the other folds with a step t: the only runs a fold may calibrate on."""
    ok = runs["uid"].isin(set(other_uids)) & runs["failure_class"].isna() & (runs["T"] >= t + 1)
    return sorted(runs.loc[ok, "uid"])


def calibration_draws(
    eligible: Sequence[str], n: int, R: int, seed: int, half_seed: int, key: Sequence[int]
) -> list[Draw] | None:
    """R draws of n runs; None when n > len(eligible) (the fold is skipped at this n).

    Seeded per (key, n, r), not by arm, so every arm at the same (family, t, fold, n, r) gets the
    same n runs (pairing). default_rng([...]) seeds from the whole integer list (a SeedSequence), so
    draws do not depend on the order the loops run in.
    """
    if n > len(eligible):
        return None
    pool = sorted(eligible)
    out = []
    for r in range(R):
        picked = sorted(np.random.default_rng([seed, *key, n, r]).choice(pool, size=n, replace=False).tolist())
        perm = np.random.default_rng([half_seed, *key, n, r]).permutation(n)
        out.append(Draw(
            tuple(picked),
            tuple(sorted(picked[i] for i in perm[: n // 2])),
            tuple(sorted(picked[i] for i in perm[n // 2:])),
        ))
    return out


def task_group_folds(pool: pd.DataFrame, k: int, seed: int) -> list[tuple[list[str], list[str]]]:
    """k folds over a multi-family source pool's task groups (grouped_kfold works on one family)."""
    groups = sorted(pool["task_group"].unique())
    random.Random(seed).shuffle(groups)
    fold_of = {g: i % k for i, g in enumerate(groups)}
    f = pool["task_group"].map(fold_of)
    return [(pool.loc[f != i, "uid"].tolist(), pool.loc[f == i, "uid"].tolist()) for i in range(k)]


# ---------------------------------------------------------------- thresholds and pooled rates


def _thetas(scores: Scores, cal_uids: Sequence[str], t: int, alphas: Sequence[float]) -> tuple[dict, int]:
    cal = [calibration_score(scores[u], t) for u in cal_uids]
    return {(rule, a): fn(cal, a) for rule, fn in RULES.items() for a in alphas}, len(cal)


def _rates(scores: Scores, eval_runs: pd.DataFrame, thetas: dict, t: int, classes: Sequence[str]) -> dict:
    sub = {u: scores[u] for u in eval_runs["uid"]}
    return {key: fa_tpr(first_alarm_steps(sub, theta), eval_runs, t, classes) for key, theta in thetas.items()}


def group_counts(scores: Scores, theta: float, eval_runs: pd.DataFrame, t: int, classes: Sequence[str]) -> dict:
    """Per task group of eval_runs: fa_tpr's counts at t, plus post-onset alarms (a positive with
    some score > theta at a step in tau..t). Post-onset is not n_pos_alarm - pre_tau: a run whose
    first alarm precedes tau can alarm again after it. Groups with no counted run are left out.

    Task-group counts are all a task-group cluster bootstrap needs, so per-run rows are not kept.
    """
    first = first_alarm_steps({u: scores[u] for u in eval_runs["uid"]}, theta)
    tau = eval_runs["tau"].to_numpy(float)
    has_step = (eval_runs["T"] >= t + 1).to_numpy()
    neg = has_step & eval_runs["failure_class"].isna().to_numpy()
    pos = has_step & eval_runs["failure_class"].isin(list(classes)).to_numpy() & (tau <= t)
    step = eval_runs["uid"].map(first).to_numpy(float)  # never alarms -> NaN, and NaN <= t is False
    alarm = step <= t
    # NaN > theta is False, so unscored steps (HGB arms) cannot alarm post-onset either.
    post = np.array([bool(p) and bool((np.asarray(scores[u], dtype=float)[int(ta): t + 1] > theta).any())
                     for u, p, ta in zip(eval_runs["uid"], pos, tau)], dtype=bool)
    masks = {"n_healthy": neg, "n_healthy_alarm": neg & alarm, "n_pos": pos, "n_pos_alarm": pos & alarm,
             "n_pos_alarm_pre_tau": pos & alarm & (step < tau), "n_pos_alarm_post": post}
    # pd.factorize: integer code per row plus the distinct groups, so np.bincount can sum per group.
    codes, groups = pd.factorize(eval_runs["task_group"])
    counts = {k: np.bincount(codes, weights=v, minlength=len(groups)).astype(int) for k, v in masks.items()}
    keep = (counts["n_healthy"] + counts["n_pos"]) > 0
    return {"task_group": np.asarray(groups)[keep], **{k: v[keep] for k, v in counts.items()}}


def group_frame(group_record: list) -> pd.DataFrame:
    """run_e3's group_record as one row per (family, arm, n, rule, alpha, t, draw, fold, task group);
    fold = -1 at n = 0, where every target run is tested at once."""
    cols: dict[str, list] = {k: [] for k in (*GROUP_META, "task_group", *GROUP_FIELDS)}
    for meta, counts in group_record:
        k = len(counts["task_group"])
        for name in GROUP_META:
            cols[name].extend([meta[name]] * k)
        for name in ("task_group", *GROUP_FIELDS):
            cols[name].extend(counts[name].tolist())
    return pd.DataFrame(cols)


class _Pool:
    """Per (arm, n, rule, alpha, t, draw): fold counts summed, thetas listed, skips counted."""

    def __init__(self):
        self.cells: dict[tuple, dict] = {}

    def _cell(self, key: tuple) -> dict:
        return self.cells.setdefault(key, {**{f: 0 for f in COUNT_FIELDS}, "thetas": [], "m": None,
                                           "folds_used": 0, "folds_skipped": 0, "undefined": False,
                                           "fit_draw": None})

    def add(self, arm, n, t, r, rates: dict, thetas: dict, m: int, fit_draw=None):
        for (rule, a), rt in rates.items():
            c = self._cell((arm, n, rule, a, t, r))
            for f in COUNT_FIELDS:
                c[f] += getattr(rt, f)
            c["thetas"].append(thetas[(rule, a)])
            c["m"], c["fit_draw"] = m, fit_draw
            c["folds_used"] += 1

    def skip(self, arm, n, t, r, alphas):
        for rule in RULES:
            for a in alphas:
                self._cell((arm, n, rule, a, t, r))["folds_skipped"] += 1

    def undefined(self, arm, n, t, r, alphas):
        for rule in RULES:
            for a in alphas:
                self._cell((arm, n, rule, a, t, r))["undefined"] = True

    def frame(self, family: str) -> pd.DataFrame:
        rows = []
        for (arm, n, rule, a, t, r), c in self.cells.items():
            fa = c["n_healthy_alarm"] / c["n_healthy"] if c["n_healthy"] else math.nan
            tpr = c["n_pos_alarm"] / c["n_pos"] if c["n_pos"] else math.nan
            early = c["n_early"] / c["n_pos_alarm"] if c["n_pos_alarm"] else math.nan
            undefined = c["undefined"] or c["folds_used"] == 0
            rows.append({
                "family": family, "arm": arm, "n": n, "rule": rule, "alpha": a, "t": t, "draw": r,
                "fa": math.nan if undefined else fa, "tpr": math.nan if undefined else tpr,
                "early_alarm_rate": math.nan if undefined else early,
                **{f: c[f] for f in COUNT_FIELDS},
                "m": c["m"], "folds_used": c["folds_used"], "folds_skipped": c["folds_skipped"],
                "undefined": undefined, "fit_draw": c["fit_draw"],
                "theta_inf_folds": sum(math.isinf(x) for x in c["thetas"]),
                "thetas": ";".join(f"{x:.6g}" for x in c["thetas"]),
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------- arms


def _esn_src(episodes, runs, family, e2_cfg, cfg, make_esn, to_run) -> tuple[list[dict], dict]:
    """The fit draws of ESN-src: per draw, the model's running-max scores on the target runs and on
    the draw's left-out pool runs (n = 0 calibration). Scored once per fit draw, then reused."""
    spec = cfg["arms"]["ESN-src"]["fit_draws"][family]
    pool = e2_healthy_pool(episodes, family, e2_cfg, "i")
    if len(pool) != spec["pool_runs"]:
        raise ValueError(f"{family} ESN-src pool has {len(pool)} runs, config says {spec['pool_runs']}")
    channels = esn_channels(pool, runs, e2_cfg)
    draws = []
    for fit in subset_draws(pool["uid"], spec["fit_size"], spec["n"], spec["seed"]):
        val = sorted(set(pool["uid"]) - set(fit))
        monitor = make_esn(channels)
        monitor.fit([to_run(u) for u in fit], [to_run(u) for u in val])
        draws.append({"fit": fit, "val": val,
                      "target": _esn_running_max(monitor, runs, to_run),
                      "val_scores": _esn_running_max(monitor, pool[pool["uid"].isin(val)], to_run)})
    aurocs = [_headline_mean(d["target"], runs, e2_cfg) for d in draws]
    return draws, {"channels": channels, "pool_runs": len(pool), "fit_size": spec["fit_size"],
                   "n_fit_draws": len(draws), "headline_auroc_per_fit_draw": aurocs}


def _oof_source_scores(train, cols, params, cfg, make_model) -> Scores:
    """n = 0 for the HGB arms: out-of-fold scores of the source pool's healthy runs (5 task-group
    folds), at the scored checkpoints. Only used to set the source threshold."""
    spec = cfg["n0_source_threshold"]["hgb"]
    X, idx = train.X["raw"], train.X["raw"].index
    uid = idx.get_level_values("uid")
    healthy = train.runs.loc[train.runs["failure_class"].isna()]
    parts = []
    for fit_uids, held_uids in task_group_folds(train.runs, spec["k"], spec["seed"]):
        fit_rows = uid.isin(set(fit_uids))
        model = fit_model(X[fit_rows], train.y[fit_rows], train.w[fit_rows], cols, params, make_model)
        held = uid.isin(set(held_uids) & set(healthy["uid"])) & idx.get_level_values("t").isin(cfg["checkpoints"]["scored"])
        parts.append(model.score(X[held]))
    return to_score_dict(pd.concat(parts), healthy)


def _pct_scores(models: dict, rows: pd.DataFrame, runs: pd.DataFrame, ref_uids, score_uids, t: int) -> dict[str, Scores]:
    """TL-pct / ALL-pct scores of score_uids at checkpoints <= t, re-expressed against a reference
    built from ref_uids' healthy rows only (the draw's n/2 reference runs)."""
    rows_t = rows[rows.index.get_level_values("t") <= t]
    in_ref = rows_t.index.get_level_values("uid").isin(set(ref_uids))
    ref = fit_healthy_reference(rows_t[in_ref], runs[runs["uid"].isin(set(ref_uids))], by="family")
    target = rows_t[rows_t.index.get_level_values("uid").isin(set(score_uids))]
    X = percentile_transform(target, ref, runs)
    sub = runs[runs["uid"].isin(set(score_uids))]
    return {arm: to_score_dict(m.score(X), sub) for arm, m in models.items()}


# ---------------------------------------------------------------- summaries


def _q(x: np.ndarray, p: float) -> float:
    # np.nanpercentile with numpy's default linear interpolation; NaN draws (undefined) are left out.
    return float(np.nanpercentile(x, p)) if np.isfinite(x).any() else math.nan


def _stats(fa: np.ndarray, tpr: np.ndarray, early: np.ndarray, alpha: float) -> dict:
    ok = np.isfinite(fa)
    return {
        "n_draws": int(fa.size), "n_draws_undefined": int((~ok).sum()),
        "fa": {"median": _q(fa, 50), "p10": _q(fa, 10), "p90": _q(fa, 90),
               "mean": float(np.nanmean(fa)) if ok.any() else math.nan,
               "p_fa_gt_2alpha": float((fa[ok] > 2 * alpha).mean()) if ok.any() else math.nan},
        "tpr": {"median": _q(tpr, 50), "p10": _q(tpr, 10), "p90": _q(tpr, 90),
                "mean": float(np.nanmean(tpr)) if np.isfinite(tpr).any() else math.nan},
        "early_alarm_rate": {"median": _q(early, 50),
                             "mean": float(np.nanmean(early)) if np.isfinite(early).any() else math.nan},
    }


def _headline(frame: pd.DataFrame, report: Sequence[int]) -> pd.DataFrame:
    """Per (arm, n, rule, alpha, draw): mean over t in report of FA, TPR, early-alarm rate. NaN if any t is NaN."""
    sub = frame[frame["t"].isin(report)]
    keys = ["arm", "n", "rule", "alpha", "draw"]
    g = sub.groupby(keys)
    out = g[["fa", "tpr", "early_alarm_rate"]].agg(lambda s: s.mean() if s.notna().all() and len(s) == len(report) else np.nan)
    return out.reset_index()


def summarise(frame: pd.DataFrame, cfg: dict) -> dict:
    """cells[arm][n][rule][alpha][t | "headline"] -> stats over draws, plus m, skips, inf thetas."""
    report, min_m = cfg["checkpoints"]["report"], {float(a): m for a, m in cfg["thresholds"]["conformal_min_m"].items()}
    head = _headline(frame, report)
    cells: dict = {}
    for (arm, n, rule, a), block in frame.groupby(["arm", "n", "rule", "alpha"]):
        node = cells.setdefault(arm, {}).setdefault(str(n), {}).setdefault(rule, {}).setdefault(str(a), {})
        defined_all = True
        for t, b in block.groupby("t"):
            b = b.sort_values("draw")
            # m is fixed per (arm, n) except at n = 0 for ESN-src, where each fit draw leaves out its own
            # runs: conformal counts as defined only if every draw has enough calibration runs.
            ms = b["m"].dropna()
            m_min, m_max = (int(ms.min()), int(ms.max())) if len(ms) else (None, None)
            defined = rule != "conformal" or (m_min is not None and m_min >= min_m[float(a)])
            defined_all &= defined
            node[str(t)] = _stats(b["fa"].to_numpy(float), b["tpr"].to_numpy(float), b["early_alarm_rate"].to_numpy(float), a) | {
                "m_min": m_min, "m_max": m_max, "conformal_defined": defined if rule == "conformal" else None,
                "folds_skipped_per_draw": int(b["folds_skipped"].max()),
                "theta_inf_fold_share": float(b["theta_inf_folds"].sum() / max(1, b["folds_used"].sum())),
            }
        h = head[(head["arm"] == arm) & (head["n"] == n) & (head["rule"] == rule) & (head["alpha"] == a)].sort_values("draw")
        node["headline"] = _stats(h["fa"].to_numpy(float), h["tpr"].to_numpy(float), h["early_alarm_rate"].to_numpy(float), a) | {
            "conformal_defined": defined_all if rule == "conformal" else None}
    return cells


def paired_difference(frame: pd.DataFrame, a: str, b: str, n: int, rule: str, alpha: float, report) -> dict:
    """Per draw r present in both arms: headline TPR(a) - TPR(b); summary over draws."""
    head = _headline(frame, report)
    sel = (head["n"] == n) & (head["rule"] == rule) & (head["alpha"] == alpha)
    ta = head[sel & (head["arm"] == a)].set_index("draw")["tpr"]
    tb = head[sel & (head["arm"] == b)].set_index("draw")["tpr"]
    both = ta.index.intersection(tb.index)
    d = (ta.loc[both] - tb.loc[both]).dropna().to_numpy(float)
    out = {"a": a, "b": b, "n": n, "rule": rule, "alpha": alpha, "n_paired_draws": int(d.size),
           "a_median_tpr_paired": _q(ta.loc[both].to_numpy(float), 50),
           "b_median_tpr_paired": _q(tb.loc[both].to_numpy(float), 50),
           "b_median_tpr_all_draws": _q(tb.to_numpy(float), 50)}
    if d.size:
        out |= {"diff_median": _q(d, 50), "diff_p10": _q(d, 10), "diff_p90": _q(d, 90),
                "share_draws_gt_0": float((d > 0).mean())}
    return out


# ---------------------------------------------------------------- expectations


def _check(value: float, holds_if: dict) -> bool:
    op, x = next(iter(holds_if.items()))
    return {"le": value <= x, "ge": value >= x, "gt": value > x, "lt": value < x}[op]


def evaluate_expectations(cfg: dict, families: dict, frames: dict) -> list[dict]:
    """Each pre-registered expectation as HOLDS / FAILS / UNDEFINED, read from cfg["expectations"]."""
    out = []
    for exp in cfg["expectations"]:
        fam = families.get(exp["family"])
        res = dict(exp)
        if fam is None:
            out.append(res | {"outcome": "UNDEFINED", "why": "family not run"})
            continue
        cells, alpha = fam["cells"], exp["alpha"]
        if exp["kind"] == "every_cell_mean_fa":
            arms = list(cells) if exp["arms"] == "all" else exp["arms"]
            checked = []
            for arm in arms:
                for n in exp["n"] if isinstance(exp["n"], list) else [exp["n"]]:
                    node = cells.get(arm, {}).get(str(n), {}).get(exp["rule"], {}).get(str(alpha), {}).get("headline")
                    if node is None or (exp.get("only_where_defined") and node["conformal_defined"] is False):
                        continue
                    v = node["fa"]["mean"]
                    checked.append({"arm": arm, "n": n, "mean_fa": v,
                                    "ok": None if math.isnan(v) else _check(v, exp["holds_if"])})
            oks = [c["ok"] for c in checked if c["ok"] is not None]
            outcome = "UNDEFINED" if not oks else ("HOLDS" if all(oks) else "FAILS")
            out.append(res | {"outcome": outcome, "cells": checked})
        elif exp["kind"] in ("paired_median_tpr_diff", "median_tpr_comparison"):
            comp = paired_difference(frames[exp["family"]], exp["a"], exp["b"], exp["n"], exp["rule"], alpha,
                                     cfg["checkpoints"]["report"])
            defined = all(
                cells.get(arm, {}).get(str(exp["n"]), {}).get(exp["rule"], {}).get(str(alpha), {})
                .get("headline", {}).get("conformal_defined") is not False for arm in (exp["a"], exp["b"]))
            if not defined or not comp["n_paired_draws"]:
                out.append(res | {"outcome": "UNDEFINED", "comparison": comp})
                continue
            if exp["kind"] == "paired_median_tpr_diff":
                ok = _check(comp["diff_median"], exp["holds_if"])
            else:
                ok = comp["a_median_tpr_paired"] <= comp["b_median_tpr_paired"]
            out.append(res | {"outcome": "HOLDS" if ok else "FAILS", "comparison": comp})
        else:
            raise ValueError(f"unknown expectation kind {exp['kind']!r}")
    return out


# ---------------------------------------------------------------- one family


def run_family(
    table, episodes, family, cfg, e2_cfg, make_esn, make_model, record=None, e2_scores=None, group_record=None,
) -> tuple[dict, pd.DataFrame]:
    runs = episodes[episodes["family"] == family]

    def emit(arm, n, t, r, fold, scores, eval_runs, thetas):
        # Called next to every pool_cells.add with the same thetas and test runs, so the group
        # counts sum exactly to the pooled counts (checked by scripts/run_e3.py).
        if group_record is not None:
            for (rule, a), theta in thetas.items():
                meta = {"family": family, "arm": arm, "n": n, "rule": rule, "alpha": a, "t": t, "draw": r, "fold": fold}
                group_record.append((meta, group_counts(scores, theta, eval_runs, t, classes)))

    fam_idx = [cfg["families"]["primary"], *cfg["families"]["descriptive"]].index(family)
    scored, report, alphas = cfg["checkpoints"]["scored"], cfg["checkpoints"]["report"], cfg["thresholds"]["alphas"]
    classes = cfg["metrics"]["classes"]["headline"]
    R, dseed, hseed = cfg["draws"]["R"], cfg["draws"]["seed"], cfg["budgets"]["half_split"]["seed"]
    to_run = run_getter(episodes)
    params = e2_cfg["model"]["primary"]
    pool_cells = _Pool()

    # Source models: the E2 arms, trained once (deterministic: early stopping off, checked in SP5).
    train = build_training_set(table, episodes, family, e2_cfg)
    rows = table[table.index.get_level_values("uid").isin(set(runs["uid"]))
                 & table.index.get_level_values("t").isin(scored)]
    models = {}
    for arm in cfg["arms"]["hgb"]["names"]:
        fs, tr = arm.rsplit("-", 1)
        models[arm] = fit_model(train.X[tr], train.y, train.w, feature_columns(fs, e2_cfg), params, make_model)
    raw_arms = [a for a in models if a.endswith("-raw")]
    pct_arms = {a: m for a, m in models.items() if a.endswith("-pct")}
    raw_scores = {a: to_score_dict(models[a].score(rows), runs) for a in raw_arms}

    reproduction = {}
    if e2_scores is not None:
        e2f = e2_scores[e2_scores["family"] == family].set_index(["uid", "t"])
        k, s = e2_cfg["recalibration"]["target_cross_fit"]["k"], e2_cfg["recalibration"]["target_cross_fit"]["seed"]
        pct_X = cross_fitted_target_transform(rows, runs, k=k, seed=s)
        ours = {a: models[a].score(rows) for a in raw_arms} | {a: m.score(pct_X) for a, m in pct_arms.items()}
        reproduction = {a: float(np.nanmax(np.abs(v.to_numpy() - e2f.loc[v.index, a].to_numpy()))) for a, v in ours.items()}

    esn_draws, esn_info = _esn_src(episodes, runs, family, e2_cfg, cfg, make_esn, to_run)
    refit = family in cfg["arms"]["ESN-refit"]["families"]
    folds = grouped_kfold(runs, family, cfg["cross_fit"]["folds"]["k"], cfg["cross_fit"]["folds"]["seed"],
                          eval_only_families=())

    # n = 0: source thresholds, tested on every target run (no target data used).
    n0 = {}
    for arm in [a for a in raw_arms if a in cfg["budgets"]["spend"]["all_n_for_calibration"]]:
        fs = arm.rsplit("-", 1)[0]
        src = _oof_source_scores(train, feature_columns(fs, e2_cfg), params, cfg, make_model)
        for t in report:
            cal = [u for u, s in src.items() if len(s) > t and not np.isnan(s[t])]
            thetas, m = _thetas(src, cal, t, alphas)
            pool_cells.add(arm, 0, t, 0, _rates(raw_scores[arm], runs, thetas, t, classes), thetas, m)
            emit(arm, 0, t, 0, -1, raw_scores[arm], runs, thetas)
            n0.setdefault(arm, {})[str(t)] = {"m_source_healthy": m}
    for j, d in enumerate(esn_draws):
        for t in report:
            cal = [u for u in d["val"] if len(d["val_scores"][u]) > t]
            thetas, m = _thetas(d["val_scores"], cal, t, alphas)
            pool_cells.add("ESN-src", 0, t, j, _rates(d["target"], runs, thetas, t, classes), thetas, m, fit_draw=j)
            emit("ESN-src", 0, t, j, -1, d["target"], runs, thetas)
            n0.setdefault("ESN-src", {}).setdefault(str(t), {"m_per_fit_draw": []})["m_per_fit_draw"].append(m)

    for n in [x for x in cfg["budgets"]["n"] if x > 0]:
        for t in report:
            for f, (other, held) in enumerate(folds):
                eval_runs = runs[runs["uid"].isin(set(held))]
                draws = calibration_draws(eligible_uids(runs, other, t), n, R["hgb_arms"], dseed, hseed, (fam_idx, t, f))
                arms_here = [*raw_arms, *pct_arms, "ESN-src"] + (["ESN-refit"] if refit else [])
                if draws is None:
                    for arm in arms_here:
                        for r in range(R["ESN-refit"] if arm == "ESN-refit" else R["hgb_arms"]):
                            pool_cells.skip(arm, n, t, r, alphas)
                    continue
                for r, d in enumerate(draws):
                    plan = {a: (d.uids, ()) for a in raw_arms} | {a: (d.half_b, d.half_a) for a in pct_arms}
                    for arm in raw_arms:
                        thetas, m = _thetas(raw_scores[arm], d.uids, t, alphas)
                        pool_cells.add(arm, n, t, r, _rates(raw_scores[arm], eval_runs, thetas, t, classes), thetas, m)
                        emit(arm, n, t, r, f, raw_scores[arm], eval_runs, thetas)
                    pct = _pct_scores(pct_arms, rows, runs, d.half_a, [*held, *d.half_b], t)
                    for arm, sc in pct.items():
                        thetas, m = _thetas(sc, d.half_b, t, alphas)
                        pool_cells.add(arm, n, t, r, _rates(sc, eval_runs, thetas, t, classes), thetas, m)
                        emit(arm, n, t, r, f, sc, eval_runs, thetas)
                    # ESN-src: one source fit draw per (n, r), shared by every fold and t of that draw.
                    j = int(np.random.default_rng([dseed, fam_idx, 999, n, r]).integers(len(esn_draws)))
                    target = esn_draws[j]["target"]
                    thetas, m = _thetas(target, d.uids, t, alphas)
                    pool_cells.add("ESN-src", n, t, r, _rates(target, eval_runs, thetas, t, classes), thetas, m, fit_draw=j)
                    emit("ESN-src", n, t, r, f, target, eval_runs, thetas)
                    plan["ESN-src"] = (d.uids, ())
                    if refit and r < R["ESN-refit"]:
                        plan["ESN-refit"] = (d.half_b, d.half_a)
                        sc = _esn_refit(d, held, runs, episodes, e2_cfg, make_esn, to_run)
                        if sc is None:
                            pool_cells.undefined("ESN-refit", n, t, r, alphas)
                        else:
                            thetas, m = _thetas(sc, d.half_b, t, alphas)
                            pool_cells.add("ESN-refit", n, t, r, _rates(sc, eval_runs, thetas, t, classes), thetas, m)
                            emit("ESN-refit", n, t, r, f, sc, eval_runs, thetas)
                    if record is not None:
                        for arm, (cal, other_half) in plan.items():
                            record.append({"family": family, "arm": arm, "n": n, "t": t, "fold": f, "draw": r,
                                           "cal": tuple(cal), "other_half": tuple(other_half), "eval": tuple(held)})

    frame = pool_cells.frame(family)
    comparisons = {}
    for exp in cfg["expectations"]:
        if "a" in exp:
            comparisons[f"{exp['a']}_vs_{exp['b']}"] = paired_difference(
                frame, exp["a"], exp["b"], exp["n"], exp["rule"], exp["alpha"], report)
    result = {
        "counts": {"target_runs": len(runs), "folds": [len(h) for _, h in folds],
                   "healthy_with_step": {str(t): int((runs["failure_class"].isna() & (runs["T"] >= t + 1)).sum()) for t in report}},
        "n0": n0,
        "esn_src": esn_info,
        "reproduction": reproduction,
        "cells": summarise(frame, cfg),
        "comparisons": comparisons,
    }
    return result, frame


def _esn_refit(d: Draw, held, runs, episodes, e2_cfg, make_esn, to_run) -> Scores | None:
    """ESN fitted on the draw's n/2 target healthy runs; the calibration half is passed as its
    validation set (it only sets the ESN's own threshold, which E3 ignores). A fit or score that
    fails, or gives non-finite scores, makes the draw UNDEFINED for this fold (counted, not patched)."""
    fit_pool = episodes[episodes["uid"].isin(set(d.half_a))]
    score_runs = runs[runs["uid"].isin(set(held) | set(d.half_b))]
    try:
        monitor = make_esn(esn_channels(fit_pool, runs, e2_cfg))
        monitor.fit([to_run(u) for u in d.half_a], [to_run(u) for u in d.half_b])
        scores = _esn_running_max(monitor, score_runs, to_run)
    except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
        return None
    if not all(np.isfinite(s).all() for s in scores.values()):
        return None
    return scores


# ---------------------------------------------------------------- orchestration


def run_e3(
    episodes: pd.DataFrame,
    cfg: dict,
    e2_cfg: dict,
    make_esn: Callable | None = None,
    make_model: Callable = HistGradientBoostingClassifier,
    table: pd.DataFrame | None = None,
    git: dict | None = None,
    record: list | None = None,
    e2_scores: pd.DataFrame | None = None,
    group_record: list | None = None,
) -> tuple[dict, pd.DataFrame]:
    """group_record, if given, collects per-(cell, draw, fold, task group) counts; see group_frame."""
    make_esn = make_esn or default_esn_factory(e2_cfg)
    if table is None:
        needed = sorted(set(e2_cfg["checkpoints"]["train"]) | set(cfg["checkpoints"]["scored"]))
        table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    families, frames = {}, {}
    for family in [cfg["families"]["primary"], *cfg["families"]["descriptive"]]:
        families[family], frames[family] = run_family(
            table, episodes, family, cfg, e2_cfg, make_esn, make_model, record, e2_scores, group_record)
    result = {
        "what": "SP6 E3 recalibration curve: FA and TPR of alarms whose thresholds come from n healthy "
        "target runs (conformal and plug-in), per arm and budget; llama primary, gemini descriptive",
        "config": cfg,
        "git": git if git is not None else git_state(),
        "caveat": CAVEAT,
        "expectations": evaluate_expectations(cfg, families, frames),
        "families": families,
    }
    return result, pd.concat(frames.values(), ignore_index=True)
