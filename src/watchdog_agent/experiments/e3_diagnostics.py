"""SP6 close-out diagnostics D-8..D-13. POST HOC.

Design and expected outcomes: docs/decisions.md, "2026-10-27 — SP6 close-out diagnostics" (D-8..D-12)
and "2026-10-28 — SP6 close-out part 2, D-13", each committed before its first run. Input: E3's per-(cell, draw, fold, task group) alarm counts
(results/e3_group_counts.parquet), so nothing is refitted or rescored here.

Every statistic is a function of task-group weights w: w = 1 gives E3's own numbers, a fold mask gives
a sub-test-set (D-8, D-9), and bootstrap multiplicities give D-12's test-side interval. The same code
path computes the point estimate and the bootstrap, so the two cannot drift apart.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

CELL_KEYS = ["arm", "n", "rule", "alpha"]
FA = ("n_healthy_alarm", "n_healthy")
TPR = ("n_pos_alarm", "n_pos")
TPR_POST = ("n_pos_alarm_post", "n_pos")
TPR_FIRST = ("n_pos_alarm_first_post", "n_pos")  # first alarm at a step in tau..t
COUNT_FIELDS = ("n_healthy", "n_healthy_alarm", "n_pos", "n_pos_alarm", "n_pos_alarm_pre_tau", "n_pos_alarm_post")
ESN_ARMS = ("ESN-src", "ESN-refit")


@dataclass
class Cell:
    draws: np.ndarray  # E3 draw ids, sorted
    counts: dict  # t -> field -> (draws, task groups) array
    undefined: dict  # t -> (draws,) bool: E3 reports the draw as NaN (e.g. an ESN-refit fit failure)
    m: dict  # t -> (draws,) calibration size


def _quiet(fn, x, axis=0):
    # nanmean / nanmedian of an all-NaN slice warn and return NaN; NaN is what we want.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(x, axis=axis)


class FamilyCounts:
    """One family's group counts as dense (draws x task groups) arrays per cell and t."""

    def __init__(self, groups: pd.DataFrame, draws: pd.DataFrame, report: Sequence[int]):
        groups = groups.astype({c: str for c in ("arm", "rule", "task_group")})
        self.report = list(report)
        self.task_groups = sorted(groups["task_group"].unique())
        gpos = groups["task_group"].map({g: i for i, g in enumerate(self.task_groups)}).to_numpy()
        # Folds are by task group, so a group is tested in exactly one fold at n > 0.
        fold_sets = groups[groups["n"] > 0].groupby("task_group")["fold"].unique()
        if (fold_sets.map(len) > 1).any():
            raise ValueError("a task group is tested in more than one fold")
        self.fold_of = np.array([int(fold_sets[g][0]) if g in fold_sets else -1 for g in self.task_groups])
        G = len(self.task_groups)
        self.cells: dict[tuple, Cell] = {}
        by_cell = {k: idx for k, idx in groups.groupby(CELL_KEYS).indices.items()}
        for key, d in draws.groupby(CELL_KEYS):
            ids = np.sort(d["draw"].unique())
            counts, undefined, m = {}, {}, {}
            rows = by_cell.get(key, np.array([], dtype=int))
            sub = groups.iloc[rows]
            for t in self.report:
                st = sub[sub["t"] == t]
                dpos, gp = np.searchsorted(ids, st["draw"].to_numpy()), gpos[rows][sub["t"].to_numpy() == t]
                counts[t] = {}
                for f in COUNT_FIELDS:
                    a = np.zeros((len(ids), G))
                    # np.add.at: unbuffered a[i, j] += v, so repeated (draw, group) pairs (one per fold) add up.
                    np.add.at(a, (dpos, gp), st[f].to_numpy(float))
                    counts[t][f] = a
                counts[t]["n_pos_alarm_first_post"] = counts[t]["n_pos_alarm"] - counts[t]["n_pos_alarm_pre_tau"]
                dt = d[d["t"] == t].set_index("draw").reindex(ids)
                undefined[t] = dt["undefined"].fillna(True).to_numpy(bool)
                m[t] = dt["m"].to_numpy(float)
            self.cells[(key[0], int(key[1]), key[2], float(key[3]))] = Cell(ids, counts, undefined, m)

    def cell(self, arm: str, n: int, rule: str, alpha: float) -> Cell:
        return self.cells[(arm, int(n), rule, float(alpha))]

    # ---------------------------------------------------------------- weights

    def ones(self) -> dict:
        return {t: np.ones(len(self.task_groups)) for t in self.report}

    def drop_folds(self, w: dict, drop: dict) -> dict:
        """Zero the weight of every task group in the dropped folds at each t."""
        return {t: w[t] * (~np.isin(self.fold_of, drop.get(t, [])))[(...,) + (None,) * (w[t].ndim - 1)] for t in w}

    def only_folds(self, w: dict, folds: Sequence[int]) -> dict:
        keep = np.isin(self.fold_of, list(folds))
        return {t: w[t] * keep[(...,) + (None,) * (w[t].ndim - 1)] for t in w}

    # ---------------------------------------------------------------- rates

    def ratio(self, cell: Cell, t: int, fields: tuple[str, str], w: dict) -> np.ndarray:
        """Per draw (and per bootstrap column if w is 2-D): sum of num / sum of den over weighted groups."""
        num, den = cell.counts[t][fields[0]] @ w[t], cell.counts[t][fields[1]] @ w[t]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        r[cell.undefined[t]] = np.nan
        return r

    def headline(self, cell: Cell, fields: tuple[str, str], w: dict) -> np.ndarray:
        """Mean over the report checkpoints per draw; NaN if any t is NaN (as E3's _headline)."""
        per_t = np.stack([self.ratio(cell, t, fields, w) for t in self.report])
        return np.where(np.isnan(per_t).any(axis=0), np.nan, per_t.mean(axis=0))

    def pooled_over_draws(self, cell: Cell, t: int, fields: tuple[str, str], w: dict) -> np.ndarray:
        """Sum over draws of num / sum over draws of den (D-9's per-fold FA); undefined draws left out."""
        ok = ~cell.undefined[t]
        num, den = (cell.counts[t][f][ok] @ w[t] for f in fields)
        num, den = num.sum(axis=0), den.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def mean_over_draws(x: np.ndarray) -> np.ndarray:
    return _quiet(np.nanmean, x)


def median_over_draws(x: np.ndarray) -> np.ndarray:
    return _quiet(np.nanmedian, x)


def paired(fc: FamilyCounts, a: Cell, b: Cell, fields, w) -> np.ndarray:
    """Per shared draw id: headline(a) - headline(b)."""
    # np.intersect1d(..., return_indices=True): the shared ids plus where each sits in a and in b.
    _, ia, ib = np.intersect1d(a.draws, b.draws, return_indices=True)
    return fc.headline(a, fields, w)[ia] - fc.headline(b, fields, w)[ib]


# ---------------------------------------------------------------- D-10: exchangeable benchmark


def conformal_alpha_eff(m: int, alpha: float) -> float:
    """P(new > theta) for iid continuous scores: 1 - k/(m+1); 0 when k > m (theta = +inf)."""
    k = math.ceil((m + 1) * (1 - alpha))
    return 0.0 if k > m else 1 - k / (m + 1)


def plugin_expected_fa(m: int, alpha: float) -> float:
    """E[FA] of np.quantile(method="linear") at 1 - alpha, exact for iid Uniform scores:
    theta = U(j+1) + g (U(j+2) - U(j+1)) at position h = (m-1)(1-alpha), and E[U(i)] = i/(m+1)."""
    return 1 - ((m - 1) * (1 - alpha) + 1) / (m + 1)


DISTRIBUTIONS: dict[str, Callable] = {
    "uniform": lambda rng, size: rng.random(size),
    "normal": lambda rng, size: rng.standard_normal(size),
    "exponential": lambda rng, size: rng.exponential(size=size),
}


def vectorised_thresholds(cal: np.ndarray, rule: str, alpha: float) -> np.ndarray:
    """Row-wise thresholds.conformal_threshold / plugin_threshold for a (reps, m) array."""
    m = cal.shape[1]
    if rule == "plugin":
        return np.quantile(cal, 1 - alpha, axis=1, method="linear")
    k = math.ceil((m + 1) * (1 - alpha))
    return np.full(cal.shape[0], np.inf) if k > m else np.sort(cal, axis=1)[:, k - 1]


def simulate_fa(rule: str, alpha: float, m: int, reps: int, dist: str, rng: np.random.Generator) -> float:
    cal, new = DISTRIBUTIONS[dist](rng, (reps, m)), DISTRIBUTIONS[dist](rng, reps)
    return float((new > vectorised_thresholds(cal, rule, alpha)).mean())


def expected_fa(rule: str, m: float, alpha: float) -> float:
    if not np.isfinite(m):
        return math.nan
    return conformal_alpha_eff(int(m), alpha) if rule == "conformal" else plugin_expected_fa(int(m), alpha)


# ---------------------------------------------------------------- D-11 helper


def subset_median_percentile(values: np.ndarray, observed: float, size: int, n: int, seed: int) -> dict:
    """Where `observed` (the median of one size-`size` subset) falls among the medians of n random
    size-`size` subsets of `values`, drawn without replacement within a subset."""
    rng = np.random.default_rng(seed)
    v = values[np.isfinite(values)]
    meds = np.array([np.median(rng.choice(v, size=size, replace=False)) for _ in range(n)])
    return {"observed": float(observed), "share_below": float((meds < observed).mean()),
            "share_at_or_below": float((meds <= observed).mean()),
            "subset_median_p10": float(np.percentile(meds, 10)), "subset_median_p50": float(np.percentile(meds, 50)),
            "subset_median_p90": float(np.percentile(meds, 90)), "n_subsets": n, "subset_size": size,
            "n_values": int(v.size)}


# ---------------------------------------------------------------- D-12 helper


def bootstrap_weights(n_groups: int, B: int, seed: int) -> np.ndarray:
    """(groups, B) multiplicities: each column resamples the task groups with replacement once."""
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(B, n_groups))
    # bincount per replicate: how often each group was drawn.
    return np.stack([np.bincount(p, minlength=n_groups) for p in picks], axis=1).astype(float)


def percentile_ci(boot: np.ndarray) -> dict:
    ok = np.isfinite(boot)
    if not ok.any():
        return {"lo": math.nan, "hi": math.nan, "n_skipped": int(boot.size)}
    lo, hi = np.percentile(boot[ok], [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_skipped": int((~ok).sum())}


# ---------------------------------------------------------------- orchestration

CAVEAT = ("POST HOC (docs/decisions.md 2026-10-27). Draw spread = calibration-set variability only; "
          "D-12 intervals = test-set (task-group) sampling with the calibration draws held fixed")


def _j(x) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _diff(a, b) -> float | None:
    return None if a is None or b is None else a - b


def _summary(x: np.ndarray) -> dict:
    ok = x[np.isfinite(x)]
    if not ok.size:
        return {"median": None, "p10": None, "p90": None, "mean": None, "share_gt_0": None, "n": 0}
    return {"median": _j(np.median(ok)), "p10": _j(np.percentile(ok, 10)), "p90": _j(np.percentile(ok, 90)),
            "mean": _j(ok.mean()), "share_gt_0": _j((ok > 0).mean()), "n": int(ok.size)}


def common_folds(groups: pd.DataFrame, ns: Sequence[int], report: Sequence[int]) -> dict:
    """Per t, the folds to drop for a test set shared by every n in ns: folds tested at some n but
    not at all of them (llama t=4: fold 1, skipped at n=40)."""
    sub = groups[groups["n"].isin(list(ns))]
    drop = {}
    for t in report:
        per_n = [set(sub[(sub["n"] == n) & (sub["t"] == t)]["fold"].unique()) for n in ns]
        drop[t] = sorted(int(f) for f in set.union(*per_n) - set.intersection(*per_n))
    return drop


def _n_test(fc: FamilyCounts, cell: Cell, w: dict, field: str = "n_healthy") -> dict:
    # Fixed per draw except where a fold was undefined (no counts for it); report the largest.
    return {str(t): int((cell.counts[t][field] @ w[t]).max()) if cell.draws.size else 0 for t in fc.report}


def run_diagnostics(groups: pd.DataFrame, draws: pd.DataFrame, e3: dict, cfg: dict, e3_cfg: dict,
                    evaluate_expectations: bool = True) -> dict:
    """evaluate_expectations=False skips the verdicts, whose n values (10, 20, 40) are E3's own."""
    report, alpha = e3_cfg["checkpoints"]["report"], float(cfg["alpha"])
    families = [e3_cfg["families"]["primary"], *e3_cfg["families"]["descriptive"]]
    out: dict = {"what": "SP6 close-out diagnostics D-8..D-13 on E3's task-group counts", "post_hoc": True,
                 "caveat": CAVEAT, "config": cfg, "families": {}}
    out["d10_simulation"] = d10_simulation(cfg["d10"])
    for family in families:
        g, d = groups[groups["family"] == family], draws[draws["family"] == family]
        fc = FamilyCounts(g, d, report)
        out["families"][family] = family_diagnostics(fc, g, e3["families"][family], cfg, alpha)
    if evaluate_expectations:
        out["expectations"] = evaluate(out, cfg, alpha)
    return out


def d10_simulation(spec: dict) -> list[dict]:
    rows = []
    for rule in ("conformal", "plugin"):
        for a in spec["alphas"]:
            for m in spec["m"]:
                formula = expected_fa(rule, m, a)
                for dist in spec["distributions"]:
                    # One generator per row, seeded by the row, so rows do not depend on loop order.
                    rng = np.random.default_rng([spec["seed"], int(rule == "plugin"), round(a * 1000), m,
                                                 list(DISTRIBUTIONS).index(dist)])
                    sim = simulate_fa(rule, a, m, spec["reps"], dist, rng)
                    exact = rule == "conformal" or dist == "uniform"
                    tol = spec["tolerance"] if exact else spec["other_distribution_tolerance"]
                    rows.append({"rule": rule, "alpha": a, "m": m, "distribution": dist, "formula": formula,
                                 "simulated": sim, "diff": sim - formula, "tolerance": tol,
                                 "ok": abs(sim - formula) <= tol})
    return rows


def family_diagnostics(fc: FamilyCounts, g: pd.DataFrame, e3_fam: dict, cfg: dict, alpha: float) -> dict:
    ones = fc.ones()
    out: dict = {"task_groups": len(fc.task_groups),
                 "fold_of_task_group": {tg: int(f) for tg, f in zip(fc.task_groups, fc.fold_of)}}

    # Reproduction: every E3 headline cell from the group counts with w = 1.
    worst = 0.0
    for (arm, n, rule, a), cell in fc.cells.items():
        ref = e3_fam["cells"][arm][str(n)][rule][str(a)]["headline"]
        fa, tpr = fc.headline(cell, FA, ones), fc.headline(cell, TPR, ones)
        for ours, theirs in [(mean_over_draws(fa), ref["fa"]["mean"]), (median_over_draws(fa), ref["fa"]["median"]),
                             (median_over_draws(tpr), ref["tpr"]["median"])]:
            ours = _j(ours)
            if (ours is None) != (theirs is None):
                worst = math.inf
            elif ours is not None:
                worst = max(worst, abs(ours - theirs))
    out["reproduction"] = {"max_abs_diff_vs_e3_headline": worst, "cells": len(fc.cells), "ok": worst <= 1e-12}

    # Post-onset alarms are a subset of alarms by t, so TPR_post <= TPR in every draw.
    out["tpr_post_le_tpr_violations"] = sum(
        int(np.sum(fc.ratio(c, t, TPR_POST, ones) > fc.ratio(c, t, TPR, ones))) for c in fc.cells.values() for t in fc.report)

    arms = sorted({k[0] for k in fc.cells})
    ns = sorted({k[1] for k in fc.cells if k[1] > 0})
    drop = common_folds(g[g["n"] > 0], cfg["d8"]["n"], fc.report)
    common = fc.drop_folds(ones, drop)
    out["d8"] = d8(fc, arms, cfg["d8"]["n"], alpha, drop, ones, common, e3_fam)
    out["d9"] = d9(fc, [a for a in cfg["d9"]["arms"] if a in arms], cfg["d9"], alpha, ones)
    out["d10_cells"] = d10_cells(fc, ones, common)
    out["d11"] = d11(fc, cfg["d11"], alpha, ones, has_refit="ESN-refit" in arms)
    out["d12"] = d12(fc, arms, ns, alpha, cfg, drop, has_refit="ESN-refit" in arms)
    out["d13a"] = d13a(fc, cfg["d13"]["a"], alpha, ones)
    out["d13b"] = d13b(fc, g, cfg["d13"]["b"])
    out["d13c"] = d13c(fc, cfg["d13"]["c"], cfg["d12"], alpha, ones)
    return out


def d8(fc, arms, ns, alpha, drop, ones, common, e3_fam) -> dict:
    rows = []
    for arm in arms:
        for n in ns:
            if (arm, n, "conformal", alpha) not in fc.cells:
                continue
            c = fc.cell(arm, n, "conformal", alpha)
            ref = e3_fam["cells"][arm][str(n)]["conformal"][str(alpha)]["headline"]
            common_fa = _j(mean_over_draws(fc.headline(c, FA, common)))
            rows.append({"arm": arm, "n": n, "conformal_defined": ref["conformal_defined"],
                         "e3_fa_mean": ref["fa"]["mean"], "common_fa_mean": common_fa,
                         "delta_fa_mean": _diff(common_fa, ref["fa"]["mean"]),
                         "e3_tpr_median": ref["tpr"]["median"],
                         "common_tpr_median": _j(median_over_draws(fc.headline(c, TPR, common))),
                         "common_tpr_post_median": _j(median_over_draws(fc.headline(c, TPR_POST, common))),
                         "e3_n_test_healthy": _n_test(fc, c, ones), "common_n_test_healthy": _n_test(fc, c, common),
                         "e3_n_pos": _n_test(fc, c, ones, "n_pos"), "common_n_pos": _n_test(fc, c, common, "n_pos")})
    return {"dropped_folds": {str(t): v for t, v in drop.items()}, "rule": "conformal", "alpha": alpha, "rows": rows}


def _folds(fc: FamilyCounts) -> list[int]:
    return sorted(set(fc.fold_of[fc.fold_of >= 0].tolist()))


def fold_rows(fc: FamilyCounts, c: Cell, t: int, alpha: float, ones: dict) -> list[dict]:
    """Per fold at t: FA pooled over draws, healthy test runs, excess alarms per draw over alpha."""
    ok = ~c.undefined[t]
    rows = []
    for f in _folds(fc):
        wf = fc.only_folds(ones, [f])
        alarms = float((c.counts[t]["n_healthy_alarm"][ok] @ wf[t]).sum())
        healthy = float((c.counts[t]["n_healthy"][ok] @ wf[t]).sum())
        rows.append({"fold": f, "fa": _j(fc.pooled_over_draws(c, t, FA, wf)),
                     "n_test_healthy": int((c.counts[t]["n_healthy"] @ wf[t]).max()),
                     "excess_alarms_per_draw": (alarms - alpha * healthy) / max(1, int(ok.sum()))})
    return rows


def d9(fc, arms, spec, alpha, ones) -> dict:
    rows = []
    folds = _folds(fc)
    rest = fc.only_folds(ones, [f for f in folds if f != spec["fold"]])
    for arm in arms:
        for n in spec["n"]:
            c = fc.cell(arm, n, "conformal", alpha)
            per_t = {}
            for t in fc.report:
                folds_t = fold_rows(fc, c, t, alpha, ones)
                total = sum(r["excess_alarms_per_draw"] for r in folds_t)
                f1 = next(r for r in folds_t if r["fold"] == spec["fold"])
                per_t[str(t)] = {"folds": folds_t, "fold_tested": f1["n_test_healthy"] > 0, "fold_fa": f1["fa"],
                                 "rest_fa": _j(fc.pooled_over_draws(c, t, FA, rest)),
                                 "fold_share_of_excess": _j(f1["excess_alarms_per_draw"] / total) if total else None}
            tested = [v for v in per_t.values() if v["fold_tested"]]
            conc = bool(tested) and all(v["fold_fa"] > spec["concentrated_if"]["fold_fa_gt"]
                                        and v["rest_fa"] <= spec["concentrated_if"]["rest_fa_le"] for v in tested)
            spread = any(v["rest_fa"] is not None and v["rest_fa"] > spec["spread_if"]["rest_fa_gt"] for v in per_t.values())
            rows.append({"arm": arm, "n": n, "per_t": per_t,
                         "reading": "CONCENTRATED" if conc else ("SPREAD" if spread else "NEITHER")})
    return {"fold": spec["fold"], "rule": "conformal", "alpha": alpha, "rows": rows}


def d10_cells(fc, ones, common) -> list[dict]:
    rows = []
    for (arm, n, rule, a), c in sorted(fc.cells.items()):
        fa = fc.headline(c, FA, ones)
        # Expected FA per draw follows the draw's own m at each t (m varies per fit draw for ESN-src at n=0).
        expected = np.stack([[expected_fa(rule, m, a) for m in c.m[t]] for t in fc.report]).mean(axis=0)
        ok = np.isfinite(fa)
        e = _j(expected[ok].mean()) if ok.any() else None
        obs = _j(mean_over_draws(fa))
        row = {"arm": arm, "n": n, "rule": rule, "alpha": a,
               "m": {str(t): [int(np.nanmin(c.m[t])), int(np.nanmax(c.m[t]))] if np.isfinite(c.m[t]).any() else None
                     for t in fc.report},
               "expected_fa": e, "observed_fa_mean": obs, "observed_minus_expected": _diff(obs, e),
               "n_test_healthy": _n_test(fc, c, ones)}
        if n > 0:
            cobs = _j(mean_over_draws(fc.headline(c, FA, common)))
            row |= {"common_observed_fa_mean": cobs, "common_observed_minus_expected": _diff(cobs, e)}
        rows.append(row)
    return rows


def d11(fc, spec, alpha, ones, has_refit) -> dict:
    n = spec["n"]
    out: dict = {"n": n, "rule": "conformal", "alpha": alpha, "arms": {}}
    for arm in sorted({k[0] for k in fc.cells if k[1] == n}):
        c = fc.cell(arm, n, "conformal", alpha)
        fa, tpr, post = (fc.headline(c, f, ones) for f in (FA, TPR, TPR_POST))
        med, med_post = _j(median_over_draws(tpr)), _j(median_over_draws(post))
        out["arms"][arm] = {"fa_mean": _j(mean_over_draws(fa)), "tpr_median": med, "tpr_post_median": med_post,
                            "tpr_drop": _diff(med, med_post),
                            "points": [{"draw": int(r), "fa": _j(x), "tpr": _j(y), "tpr_post": _j(z)}
                                       for r, x, y, z in zip(c.draws, fa, tpr, post)]}
    src, tl = fc.cell("ESN-src", n, "conformal", alpha), fc.cell("TL-pct", n, "conformal", alpha)
    out["e2_restated"] = {name: _summary(paired(fc, tl, src, f, ones)) for name, f in (("tpr", TPR), ("tpr_post", TPR_POST))}
    if has_refit:
        ref = fc.cell("ESN-refit", n, "conformal", alpha)
        in_pair = np.isin(src.draws, ref.draws)
        e3 = {}
        for name, f in (("tpr", TPR), ("tpr_post", TPR_POST)):
            src_all = fc.headline(src, f, ones)
            refit_med, src_med = _j(median_over_draws(fc.headline(ref, f, ones))), _j(median_over_draws(src_all))
            e3[name] = {
                "paired": _summary(paired(fc, ref, src, f, ones)) | {
                    "refit_median": refit_med, "src_median_paired_draws": _j(median_over_draws(src_all[in_pair]))},
                "refit_30_vs_src_all": {"refit_median": refit_med, "src_median_all_draws": src_med,
                                        "diff": _diff(refit_med, src_med)},
                "src_paired_median_among_subsets": subset_median_percentile(
                    src_all, float(np.nanmedian(src_all[in_pair])), int(in_pair.sum()), spec["subsets"], spec["seed"]),
            }
        out["e3_restated"] = e3
    return out


def d12(fc, arms, ns, alpha, cfg, drop, has_refit) -> dict:
    spec, d9s, n11 = cfg["d12"], cfg["d9"], cfg["d11"]["n"]
    M = bootstrap_weights(len(fc.task_groups), spec["B"], spec["seed"])
    stats: dict[str, Callable] = {}
    for arm in arms:
        for n in [0, *ns]:
            for rule in ("conformal", "plugin"):
                if (arm, n, rule, alpha) not in fc.cells:
                    continue
                c = fc.cell(arm, n, rule, alpha)
                stats[f"{arm}.n{n}.{rule}.fa_mean"] = lambda w, c=c: mean_over_draws(fc.headline(c, FA, w))
                if rule != "conformal":
                    continue
                stats[f"{arm}.n{n}.conformal.tpr_median"] = lambda w, c=c: median_over_draws(fc.headline(c, TPR, w))
                stats[f"{arm}.n{n}.conformal.tpr_post_median"] = lambda w, c=c: median_over_draws(fc.headline(c, TPR_POST, w))
                if n > 0:
                    stats[f"{arm}.n{n}.conformal.common.fa_mean"] = \
                        lambda w, c=c: mean_over_draws(fc.headline(c, FA, fc.drop_folds(w, drop)))
                    stats[f"{arm}.n{n}.conformal.common.tpr_median"] = \
                        lambda w, c=c: median_over_draws(fc.headline(c, TPR, fc.drop_folds(w, drop)))
    src, tl = fc.cell("ESN-src", n11, "conformal", alpha), fc.cell("TL-pct", n11, "conformal", alpha)
    for name, f in (("tpr", TPR), ("tpr_post", TPR_POST)):
        stats[f"E-2.TL-pct_minus_ESN-src.n{n11}.{name}.diff_median"] = \
            lambda w, f=f: median_over_draws(paired(fc, tl, src, f, w))
        if has_refit:
            ref = fc.cell("ESN-refit", n11, "conformal", alpha)
            stats[f"E-3.ESN-refit_minus_ESN-src.n{n11}.{name}.paired_diff_median"] = \
                lambda w, f=f, ref=ref: median_over_draws(paired(fc, ref, src, f, w))
            stats[f"E-3.ESN-refit_minus_ESN-src.n{n11}.{name}.refit30_minus_src_all"] = \
                lambda w, f=f, ref=ref: median_over_draws(fc.headline(ref, f, w)) - median_over_draws(fc.headline(src, f, w))
    folds = sorted(set(fc.fold_of[fc.fold_of >= 0].tolist()))
    for arm in [a for a in d9s["arms"] if a in arms]:
        for n in d9s["n"]:
            c = fc.cell(arm, n, "conformal", alpha)
            for t in fc.report:
                stats[f"{arm}.n{n}.t{t}.fold{d9s['fold']}_fa"] = \
                    lambda w, c=c, t=t: fc.pooled_over_draws(c, t, FA, fc.only_folds(w, [d9s["fold"]]))
                stats[f"{arm}.n{n}.t{t}.rest_fa"] = \
                    lambda w, c=c, t=t: fc.pooled_over_draws(c, t, FA, fc.only_folds(w, [f for f in folds if f != d9s["fold"]]))
    ones, boot = fc.ones(), {t: M for t in fc.report}
    return {"B": spec["B"], "seed": spec["seed"], "task_groups": len(fc.task_groups),
            "stats": {name: {"point": _j(fn(ones))} | percentile_ci(np.asarray(fn(boot), dtype=float))
                      for name, fn in stats.items()}}


def d13a_reading(pooled: float | None, rated_fas: Sequence[float], spec: dict) -> str:
    """decisions.md 2026-10-28: HIGH / NOMINAL / OFFSETTING over the rated folds, UNRATED if none."""
    lo, hi = spec["nominal_band"]
    if not rated_fas:
        return "UNRATED"
    if pooled > spec["high_if_pooled_gt"]:
        return "HIGH"
    return "NOMINAL" if all(lo <= x <= hi for x in rated_fas) else "OFFSETTING"


def d13a(fc, spec, alpha, ones) -> dict:
    rows = []
    for arm in spec["arms"]:
        for n in spec["n"]:
            if (arm, n, "conformal", alpha) not in fc.cells:
                continue
            c = fc.cell(arm, n, "conformal", alpha)
            per_t = {}
            for t in fc.report:
                folds = fold_rows(fc, c, t, alpha, ones)
                rated = [r for r in folds if r["n_test_healthy"] >= spec["min_healthy"]]
                pooled = _j(fc.pooled_over_draws(c, t, FA, ones))
                per_t[str(t)] = {"folds": folds, "pooled_fa": pooled, "rated_folds": [r["fold"] for r in rated],
                                 "reading": d13a_reading(pooled, [r["fa"] for r in rated], spec)}
            m = {str(t): [int(np.nanmin(c.m[t])), int(np.nanmax(c.m[t]))] if np.isfinite(c.m[t]).any() else None
                 for t in fc.report}
            rows.append({"arm": arm, "n": n, "m": m, "per_t": per_t})
    return {"rule": "conformal", "alpha": alpha, "spec": spec,
            "caveat": "per-fold FA carries test-sampling noise that pooling over draws does not remove",
            "rows": rows}


def d13b(fc, g, spec) -> dict:
    sub = g[g["n"] > 0]
    out = {"fold": spec["fold"], "per_t": {}}
    for t in fc.report:
        # A group's healthy test runs at t are a property of the data, not of the cell or draw.
        per_group = sub[sub["t"] == t].groupby("task_group")["n_healthy"].agg(["min", "max"])
        if (per_group["min"] != per_group["max"]).any():
            raise ValueError(f"healthy test runs per task group differ across cells at t={t}")
        healthy = per_group["max"].astype(int)
        in_fold = [tg for tg in healthy.index if fc.fold_of[fc.task_groups.index(tg)] == spec["fold"]]
        fold = healthy[in_fold].sort_values(ascending=False, kind="stable")
        out["per_t"][str(t)] = {"healthy_per_task_group": {tg: int(v) for tg, v in fold.items() if v > 0},
                                "fold_total": int(fold.sum()), "all_folds_total": int(healthy.sum())}
    return out


def d13c(fc, spec, d12_spec, alpha, ones) -> dict:
    first_gt_post = post_gt_alarm = esn_post_ne_alarm = hgb_post_gt_first = 0
    for (arm, _, _, _), c in fc.cells.items():
        for t in fc.report:
            k = c.counts[t]
            first_gt_post += int((k["n_pos_alarm_first_post"] > k["n_pos_alarm_post"]).sum())
            post_gt_alarm += int((k["n_pos_alarm_post"] > k["n_pos_alarm"]).sum())
            if arm in ESN_ARMS:
                esn_post_ne_alarm += int((k["n_pos_alarm_post"] != k["n_pos_alarm"]).sum())
            else:
                hgb_post_gt_first += int((k["n_pos_alarm_post"] > k["n_pos_alarm_first_post"]).sum())
    cells: dict = {}
    for (arm, n, rule, a), c in sorted(fc.cells.items()):
        if rule == "conformal" and a == alpha:
            cells.setdefault(arm, {})[str(n)] = {
                name: _j(median_over_draws(fc.headline(c, f, ones)))
                for name, f in (("tpr_median", TPR), ("tpr_post_median", TPR_POST), ("tpr_first_median", TPR_FIRST))}
    e2 = spec["e2"]
    tl, src = fc.cell(e2["a"], e2["n"], "conformal", alpha), fc.cell(e2["b"], e2["n"], "conformal", alpha)
    # Same B, seed and generator as D-12, so this interval uses D-12's resamples.
    M = bootstrap_weights(len(fc.task_groups), d12_spec["B"], d12_spec["seed"])
    boot = median_over_draws(paired(fc, tl, src, TPR_FIRST, {t: M for t in fc.report}))
    return {
        "definitions": {
            "early_alarm_rate": "per (cell, t, draw), pooled over folds: n_early / n_pos_alarm. n_pos_alarm = positives "
                                "(headline classes, tau <= t, a step t) whose first alarm is <= t; n_early = those whose "
                                "first alarm is < tau. Denominator: alarmed positives, not all positives. NaN when "
                                "n_pos_alarm = 0 at some t; such draws leave E3's headline median",
            "tpr_post": "positives with some score > theta at a step in tau..t / all positives (alarm state at >= tau, not "
                        "first crossing >= tau). ESN arms score a running max, so for them tpr_post = TPR by construction",
            "tpr_first": "(n_pos_alarm - n_pos_alarm_pre_tau) / n_pos: first alarm at a step in tau..t; same meaning for "
                         "every arm",
        },
        "checks": {"first_gt_post_entries": first_gt_post, "post_gt_alarm_entries": post_gt_alarm,
                   "esn_post_ne_alarm_entries": esn_post_ne_alarm, "hgb_post_gt_first_entries": hgb_post_gt_first},
        "cells": cells,
        # The summary nests: its own "n" counts draws and must not overwrite the budget n.
        "e2_tpr_first": {"a": e2["a"], "b": e2["b"], "n": e2["n"], "B": d12_spec["B"], "seed": d12_spec["seed"],
                         "paired_diff": _summary(paired(fc, tl, src, TPR_FIRST, ones)),
                         "ci": percentile_ci(np.asarray(boot, dtype=float))},
    }


# ---------------------------------------------------------------- expectations (docs/decisions.md 2026-10-27, 2026-10-28)


def _verdict(checks: list) -> str:
    vals = [c["ok"] for c in checks if c["ok"] is not None]
    return "UNDEFINED" if not vals else ("HOLDS" if all(vals) else "FAILS")


def _block(**parts) -> dict:
    return {k: {"verdict": _verdict(v), "checks": v} for k, v in parts.items()}


def evaluate(out: dict, cfg: dict, alpha: float) -> dict:
    fam = out["families"][cfg["primary_family"]]
    high, low = cfg["d9"]["arms"], ["MS-raw", "ESN-src"]
    d8r = {(r["arm"], r["n"]): r for r in fam["d8"]["rows"]}
    res: dict = {}

    res["D-8"] = _block(
        # As E-1: only where conformal is defined. At n=10 the m=5 arms have theta=+inf, FA=0 on any test set.
        a=[{"arm": arm, "n": n, "delta": d8r[(arm, n)]["delta_fa_mean"], "ok": abs(d8r[(arm, n)]["delta_fa_mean"]) > 0.02}
           for arm in high for n in (10, 20) if d8r[(arm, n)]["conformal_defined"]],
        b=[{"arm": arm, "common_fa_mean": d8r[(arm, 20)]["common_fa_mean"], "ok": d8r[(arm, 20)]["common_fa_mean"] > 0.12}
           for arm in high],
        c=[{"arm": arm, "n": n, "delta": d8r[(arm, n)]["delta_fa_mean"], "ok": abs(d8r[(arm, n)]["delta_fa_mean"]) <= 0.02}
           for arm in low for n in (10, 20)],
        d=[{"arm": arm, "n20": d8r[(arm, 20)]["common_fa_mean"], "n40": d8r[(arm, 40)]["common_fa_mean"],
            "ok": d8r[(arm, 40)]["common_fa_mean"] >= d8r[(arm, 20)]["common_fa_mean"] - 0.03} for arm in high])
    res["D-8"]["a"]["note"] = "evaluated where conformal is defined (as E-1); at n=10 the m=5 arms have theta=+inf"

    res["D-9"] = {"verdict": "HOLDS" if all(r["reading"] == "CONCENTRATED" for r in fam["d9"]["rows"]) else "FAILS",
                  "readings": [{"arm": r["arm"], "n": r["n"], "reading": r["reading"]} for r in fam["d9"]["rows"]]}

    cells = {(r["arm"], r["n"], r["rule"], r["alpha"]): r for r in fam["d10_cells"]}
    res["D-10"] = _block(
        a=[{k: s[k] for k in ("rule", "alpha", "m", "distribution", "diff", "tolerance")} | {"ok": s["ok"]}
           for s in out["d10_simulation"]],
        b=[{"arm": k[0], "n": k[1], "rule": k[2], "alpha": k[3], "diff": r["observed_minus_expected"],
            "ok": abs(r["observed_minus_expected"]) <= 0.03} for k, r in cells.items() if k[0] in low and k[1] >= 10],
        c=[{"arm": arm, "n": n, "diff": cells[(arm, n, "conformal", alpha)]["observed_minus_expected"],
            "ok": cells[(arm, n, "conformal", alpha)]["observed_minus_expected"] > 0.10} for arm in high for n in (20, 40)],
        d=[{"m": m, "expected": plugin_expected_fa(m, alpha), "ok": plugin_expected_fa(m, alpha) > alpha} for m in (5, 10)])

    e11 = fam["d11"]
    drops = {arm: e11["arms"][arm]["tpr_drop"] for arm in e11["arms"]}
    e2p, e3p = e11["e2_restated"]["tpr_post"], e11["e3_restated"]["tpr_post"]
    sub = e11["e3_restated"]["tpr"]["src_paired_median_among_subsets"]
    res["D-11"] = _block(
        a=[{"drops": drops, "ok": min(drops[x] for x in high) > max(drops[x] for x in ["TL-raw", *low])}],
        b=[{"diff_median": e2p["median"], "share_gt_0": e2p["share_gt_0"],
            "ok": e2p["median"] < 0.115 and e2p["share_gt_0"] < 0.80}],
        c=[{"reading": "paired", "diff_median": e3p["paired"]["median"], "ok": e3p["paired"]["median"] > 0},
           {"reading": "refit_30_vs_src_all", "diff": e3p["refit_30_vs_src_all"]["diff"],
            "ok": e3p["refit_30_vs_src_all"]["diff"] > 0}],
        d=[{"share_below": sub["share_below"], "ok": sub["share_below"] < 0.10}])

    st = fam["d12"]["stats"]

    def width(name):
        return st[name]["hi"] - st[name]["lo"]

    e2 = st[f"E-2.TL-pct_minus_ESN-src.n{cfg['d11']['n']}.tpr.diff_median"]
    res["D-12"] = _block(
        a=[{"arm": arm, "width": width(f"{arm}.n20.conformal.fa_mean"), "ok": width(f"{arm}.n20.conformal.fa_mean") > 0.15}
           for arm in high],
        b=[{"n": n, "ci": [st[f"MS-raw.n{n}.conformal.fa_mean"][k] for k in ("lo", "hi")],
            "ok": st[f"MS-raw.n{n}.conformal.fa_mean"]["lo"] <= 0.10 <= st[f"MS-raw.n{n}.conformal.fa_mean"]["hi"]}
           for n in (10, 20)],
        c=[{"ci": [e2["lo"], e2["hi"]], "ok": e2["lo"] <= 0 <= e2["hi"]}])

    at_n = cfg["d13"]["a"]["expect_offsetting_at_n"]
    readings = [{"arm": r["arm"], "n": r["n"], "t": int(t), "reading": v["reading"]}
                for r in fam["d13a"]["rows"] for t, v in r["per_t"].items()]
    res["D-13a"] = {"verdict": "HOLDS" if any(x["reading"] == "OFFSETTING" for x in readings if x["n"] == at_n)
                    else "FAILS", "readings": readings}
    ch = fam["d13c"]["checks"]
    res["D-13c"] = _block(
        i=[{"first_gt_post": ch["first_gt_post_entries"], "post_gt_alarm": ch["post_gt_alarm_entries"],
            "ok": ch["first_gt_post_entries"] == 0 and ch["post_gt_alarm_entries"] == 0}],
        ii=[{"esn_post_ne_alarm": ch["esn_post_ne_alarm_entries"], "ok": ch["esn_post_ne_alarm_entries"] == 0}],
        iii=[{"hgb_post_gt_first": ch["hgb_post_gt_first_entries"], "ok": ch["hgb_post_gt_first_entries"] > 0}])
    return res
