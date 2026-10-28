"""SP6 D-14 (POST HOC): E3's split arms with their two halves drawn from disjoint task groups.

Design and expected outcome: docs/decisions.md, "D-14 Half split by task group", committed before the
run. Only the half split changes. Source models, test runs, folds, draw counts, seeds, threshold
rules and skips are E3's, and the scoring, threshold and counting functions are E3's own. The
loop also runs with E3's by-run draws (split="run") to check it reproduces E3's counts exactly.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from watchdog_agent.experiments.e2 import (
    build_training_set,
    default_esn_factory,
    feature_columns,
    fit_model,
)
from watchdog_agent.experiments.e2_diagnostics import run_getter
from watchdog_agent.experiments.e3 import (
    GROUP_FIELDS,
    _esn_refit,
    _pct_scores,
    _Pool,
    _rates,
    _thetas,
    calibration_draws,
    eligible_uids,
    group_counts,
    group_frame,
    group_split_draws,
)
from watchdog_agent.experiments.e3_diagnostics import (
    FA,
    TPR,
    TPR_FIRST,
    FamilyCounts,
    _j,
    bootstrap_weights,
    conformal_alpha_eff,
    fold_rows,
    mean_over_draws,
    median_over_draws,
    percentile_ci,
)
from watchdog_agent.splits import grouped_kfold

GROUP_SUFFIX = "@group"
CAVEAT = ("POST HOC (docs/decisions.md D-14). Draw spread = calibration-set variability only; r's interval = "
          "test-set (task-group) sampling with both splits' draws held fixed. The two splits draw different "
          "runs, so they are compared as distributions on the same test runs, not draw by draw")


def run_d14(table, episodes, cfg: dict, e3_cfg: dict, e2_cfg: dict, split: str = "task_group",
            only: tuple[int, int, int] | None = None, make_esn: Callable | None = None,
            make_model: Callable = HistGradientBoostingClassifier) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The split arms' loop of e3.run_family with the half split swapped. only = (n, t, fold)
    restricts it to one cell (the code path check). Returns the draw frame (as E3's), the
    task-group counts (as E3's parquet) and split/placement records."""
    make_esn = make_esn or default_esn_factory(e2_cfg)
    family = cfg["family"]
    runs = episodes[episodes["family"] == family]
    fam_idx = [e3_cfg["families"]["primary"], *e3_cfg["families"]["descriptive"]].index(family)
    scored, report, alphas = e3_cfg["checkpoints"]["scored"], e3_cfg["checkpoints"]["report"], e3_cfg["thresholds"]["alphas"]
    classes = e3_cfg["metrics"]["classes"]["headline"]
    R, dseed, hseed = e3_cfg["draws"]["R"], e3_cfg["draws"]["seed"], e3_cfg["budgets"]["half_split"]["seed"]
    to_run, params = run_getter(episodes), e2_cfg["model"]["primary"]

    # The same source models as E3 (deterministic: early stopping off, checked in SP5).
    train = build_training_set(table, episodes, family, e2_cfg)
    rows = table[table.index.get_level_values("uid").isin(set(runs["uid"]))
                 & table.index.get_level_values("t").isin(scored)]
    pct_arms = {a: fit_model(train.X["pct"], train.y, train.w, feature_columns(a.rsplit("-", 1)[0], e2_cfg), params,
                             make_model) for a in cfg["arms"] if a.endswith("-pct")}
    refit = "ESN-refit" in cfg["arms"]
    folds = grouped_kfold(runs, family, e3_cfg["cross_fit"]["folds"]["k"], e3_cfg["cross_fit"]["folds"]["seed"],
                          eval_only_families=())
    group_of = dict(zip(runs["uid"], runs["task_group"]))
    healthy_sizes = runs.loc[runs["failure_class"].isna(), "task_group"].value_counts()
    big = set(healthy_sizes[healthy_sizes >= cfg["big_group_min_runs"]].index)

    pool, group_record, splits, placement = _Pool(), [], [], []

    def add(arm, n, t, r, f, scores, eval_runs, cal):
        thetas, m = _thetas(scores, cal, t, alphas)
        pool.add(arm, n, t, r, _rates(scores, eval_runs, thetas, t, classes), thetas, m)
        for (rule, a), theta in thetas.items():
            meta = {"family": family, "arm": arm, "n": n, "rule": rule, "alpha": a, "t": t, "draw": r, "fold": f}
            group_record.append((meta, group_counts(scores, theta, eval_runs, t, classes)))

    for n in cfg["n"]:
        for t in report:
            for f, (other, held) in enumerate(folds):
                if only is not None and (n, t, f) != tuple(only):
                    continue
                eval_runs = runs[runs["uid"].isin(set(held))]
                eligible, key = eligible_uids(runs, other, t), (fam_idx, t, f)
                if split == "run":
                    got = calibration_draws(eligible, n, R["hgb_arms"], dseed, hseed, key)
                    got = None if got is None else (got, [0] * len(got))
                else:
                    got = group_split_draws(eligible, group_of, n, R["hgb_arms"], dseed, hseed, key,
                                            cfg["split"]["max_tries"])
                if got is None:
                    for arm in cfg["arms"]:
                        for r in range(R["ESN-refit"] if arm == "ESN-refit" else R["hgb_arms"]):
                            pool.skip(arm, n, t, r, alphas)
                    splits.append({"n": n, "t": t, "fold": f, "feasible": False})
                    continue
                draws, tries = got
                halves = [(frozenset(group_of[u] for u in d.half_a), frozenset(group_of[u] for u in d.half_b))
                          for d in draws]
                splits.append({"n": n, "t": t, "fold": f, "feasible": True, "max_tries": max(tries),
                               "distinct_group_pairs": len(set(halves)),
                               "shared_groups_max": max(len(a & b) for a, b in halves)})
                for r, d in enumerate(draws):
                    ga, gb = halves[r]
                    placement.append({"n": n, "t": t, "fold": f, "draw": r,
                                      "big_in": "reference" if big & ga else ("calibration" if big & gb else "neither")})
                    if pct_arms:
                        for arm, sc in _pct_scores(pct_arms, rows, runs, d.half_a, [*held, *d.half_b], t).items():
                            add(arm, n, t, r, f, sc, eval_runs, d.half_b)
                    if refit and r < R["ESN-refit"]:
                        sc = _esn_refit(d, held, runs, episodes, e2_cfg, make_esn, to_run)
                        if sc is None:
                            pool.undefined("ESN-refit", n, t, r, alphas)
                        else:
                            add("ESN-refit", n, t, r, f, sc, eval_runs, d.half_b)
    return pool.frame(family), group_frame(group_record), {"splits": splits, "placement": placement,
                                                           "big_groups": sorted(big)}


def code_path_check(ours: pd.DataFrame, e3_groups: pd.DataFrame, cfg: dict) -> dict:
    """D-14's loop on E3's by-run draws vs E3's own group counts for the same (n, t, fold)."""
    c = cfg["code_path_check"]
    keys = ["arm", "n", "rule", "alpha", "t", "draw", "fold", "task_group"]
    ref = e3_groups[(e3_groups["family"] == cfg["family"]) & e3_groups["arm"].isin(cfg["arms"])
                    & (e3_groups["n"] == c["n"]) & (e3_groups["t"] == c["t"]) & (e3_groups["fold"] == c["fold"])]
    both = ours.merge(ref, on=keys, how="outer", suffixes=("", "_e3"), indicator=True)
    unmatched = int((both["_merge"] != "both").sum())
    m = both[both["_merge"] == "both"]
    diff = max((int((m[f] != m[f"{f}_e3"]).sum()) for f in GROUP_FIELDS), default=0)
    return {"rows": len(ours), "e3_rows": len(ref), "unmatched_rows": unmatched, "max_field_mismatches": diff,
            "ok": unmatched == 0 and diff == 0 and len(ours) > 0}


def d14_reading(r: float | None, excess: float | None, ci: dict, rd: dict) -> str:
    """decisions.md D-14: FALLS / PARTIAL / DOES NOT FALL for r = FA_run - FA_group."""
    # A NaN interval compares False both ways, which would read as "excludes 0".
    if r is None or excess is None or not (np.isfinite(ci["lo"]) and np.isfinite(ci["hi"])):
        return "UNDEFINED"
    if r >= rd["falls_share_of_excess"] * excess and not ci["lo"] <= 0 <= ci["hi"]:
        return "FALLS"
    return "PARTIAL" if r > rd["no_fall_le"] else "DOES NOT FALL"


def _p_gt(x: np.ndarray, level: float) -> float | None:
    ok = x[np.isfinite(x)]
    return float((ok > level).mean()) if ok.size else None


def compare(e3_groups, e3_draws, d14_groups, d14_draws, extra: dict, cfg: dict, e3_cfg: dict) -> dict:
    """By-run (E3) vs by-group (D-14) on one FamilyCounts, so both use the same test task groups and
    one bootstrap resample per replicate."""
    fam, arms, ns, alpha = cfg["family"], cfg["arms"], cfg["n"], float(cfg["alpha"])

    def e3_part(df):
        return df[(df["family"] == fam) & df["arm"].isin(arms) & df["n"].isin(ns)]

    def d14_part(df):
        return df.assign(arm=df["arm"] + GROUP_SUFFIX)

    groups = pd.concat([e3_part(e3_groups), d14_part(d14_groups)], ignore_index=True)
    draws = pd.concat([e3_part(e3_draws), d14_part(d14_draws)], ignore_index=True)
    fc = FamilyCounts(groups, draws, e3_cfg["checkpoints"]["report"])
    ones = fc.ones()
    M = bootstrap_weights(len(fc.task_groups), cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"])
    boot = {t: M for t in fc.report}
    rd = cfg["reading"]
    place = pd.DataFrame(extra["placement"])
    out: dict = {}
    for arm in arms:
        for n in ns:
            run_c, grp_c = fc.cell(arm, n, "conformal", alpha), fc.cell(arm + GROUP_SUFFIX, n, "conformal", alpha)
            fa_run, fa_grp = fc.headline(run_c, FA, ones), fc.headline(grp_c, FA, ones)
            run_mean, grp_mean = _j(mean_over_draws(fa_run)), _j(mean_over_draws(fa_grp))
            a_eff = conformal_alpha_eff(n // 2, alpha)
            r = None if run_mean is None or grp_mean is None else run_mean - grp_mean
            r_boot = mean_over_draws(fc.headline(run_c, FA, boot)) - mean_over_draws(fc.headline(grp_c, FA, boot))
            ci = percentile_ci(np.asarray(r_boot, dtype=float))
            excess = None if run_mean is None else run_mean - a_eff
            reading = d14_reading(r, excess, ci, rd)
            per_fold, by_place = {}, []
            for t in fc.report:
                per_fold[str(t)] = {"run": fold_rows(fc, run_c, t, alpha, ones), "group": fold_rows(fc, grp_c, t, alpha, ones)}
                for f in sorted(set(fc.fold_of[fc.fold_of >= 0].tolist())):
                    fold_fa = fc.ratio(grp_c, t, FA, fc.only_folds(ones, [f]))
                    p = place[(place["n"] == n) & (place["t"] == t) & (place["fold"] == f)].set_index("draw")["big_in"]
                    if p.empty:
                        continue
                    where = p.reindex(grp_c.draws).to_numpy()
                    for k in ("reference", "calibration", "neither"):
                        sel_k = (where == k) & np.isfinite(fold_fa)
                        if sel_k.any():
                            by_place.append({"t": t, "fold": f, "big_in": k, "draws": int(sel_k.sum()),
                                             "fold_fa_mean": float(fold_fa[sel_k].mean())})
            out.setdefault(arm, {})[str(n)] = {
                "alpha_eff": a_eff, "fa_run_mean": run_mean, "fa_group_mean": grp_mean, "excess_run": excess,
                "r": r, "r_ci": ci, "reading": reading,
                "p_fa_gt_2alpha": {"run": _p_gt(fa_run, 2 * alpha), "group": _p_gt(fa_grp, 2 * alpha)},
                "tpr_median": {"run": _j(median_over_draws(fc.headline(run_c, TPR, ones))),
                               "group": _j(median_over_draws(fc.headline(grp_c, TPR, ones)))},
                "tpr_first_median": {"run": _j(median_over_draws(fc.headline(run_c, TPR_FIRST, ones))),
                                     "group": _j(median_over_draws(fc.headline(grp_c, TPR_FIRST, ones)))},
                "draws": {"run": int(np.isfinite(fa_run).sum()), "group": int(np.isfinite(fa_grp).sum())},
                "per_fold": per_fold, "fa_by_big_group_placement": by_place,
            }
    checks = [{"arm": arm, "n": 20, "reading": out[arm]["20"]["reading"], "ok": out[arm]["20"]["reading"] == "FALLS"}
              for arm in arms if "20" in out.get(arm, {})]
    verdict = "HOLDS" if checks and all(c["ok"] for c in checks) else "FAILS"
    return {"rule": "conformal", "alpha": alpha, "cells": out, "expectation": {"verdict": verdict, "checks": checks}}
