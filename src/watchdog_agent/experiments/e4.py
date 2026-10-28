"""SP7 E4: the cheap monitors vs J-local at matched steps on held-out llama. NO GATE (configs/e4.yaml).

Every monitor number comes from the code that produced it in E2/E3 and must reproduce it exactly:
the HGB arms from results/e2_prefix_scores.csv (E2's headline AUROCs), ESN-src's 30 fit draws from
e3._esn_src (D-6's AUROCs), and the J-2 operating point from E3's own folds, draws and threshold
functions (E3's MS-raw and ESN-src cells). The judge is the only new scorer.

D = median over ESN-src's 30 fit draws of (AUROC_draw - AUROC_judge). Its CI is a paired task-group
cluster bootstrap that makes the same random draws as evaluation._cluster_bootstrap, so with one fit
draw it is exactly paired_matched_step_diff (tested). The fit draws are held fixed: the CI is
test-side sampling only.
"""

import math
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from watchdog_agent.evaluation import (
    _rows_by_group,
    matched_step_auroc,
    prefix_eval_set,
)
from watchdog_agent.experiments.e2_diagnostics import restrict_classes
from watchdog_agent.experiments.e3 import (
    _Pool,
    _rates,
    _thetas,
    calibration_draws,
    eligible_uids,
    group_counts,
    summarise,
)
from watchdog_agent.experiments.e4_dev import collect, preflight
from watchdog_agent.judges import JudgeCache, forbidden_hits, mean_score, prompt_hash
from watchdog_agent.metrics import MAX_SKIPPED_FRACTION, auroc
from watchdog_agent.splits import grouped_kfold
from watchdog_agent.thresholds import calibration_score

Scores = Mapping[str, Sequence[float]]


class RunStopped(RuntimeError):
    """A pre-registered stop condition (parse failures, digest, leakage); the message says which."""


# ---------------------------------------------------------------- judge


def check_digest(pulled: str, cfg: dict) -> None:
    if pulled != cfg["judge"]["digest"]:
        raise RunStopped(f"pulled {cfg['judge']['model']} digest {pulled} != frozen {cfg['judge']['digest']}")


def judge_scores(template: str, runs: pd.DataFrame, judge, cache: JudgeCache, cfg: dict) -> tuple[dict, dict]:
    """J-local's scores for every llama run (NaN where not judged: tool_cascade, steps not scored)."""
    if prompt_hash(template) != cfg["prompt"]["sha256"]:
        raise RunStopped("template does not match the frozen sha256 in configs/e4.yaml")
    judged = restrict_classes(runs, cfg["classes"]["headline"])
    todo = preflight(template, judged, cfg["checkpoints"]["scored"])  # forbidden template words, label invariance
    leaks = {(u, t): forbidden_hits(p) for u, t, p in todo if forbidden_hits(p)}
    if leaks:
        raise RunStopped(f"forbidden words in {len(leaks)} llama prompts, e.g. {next(iter(leaks.items()))}")
    jc = cfg["judge"]
    texts, new = collect(todo, judge, cache, jc, prompt_hash(template))
    scores = {uid: np.full(int(T), np.nan) for uid, T in zip(runs["uid"], runs["T"])}
    k, failures, identical, sds = jc["samples"], 0, [], []
    for (uid, t), rs in texts.items():
        m, bad = mean_score([r["text"] for r in rs[:k]])
        failures += bad
        if m is not None:
            scores[uid][t] = m
        if len(rs) > k:
            identical.append(len({r["text"] for r in rs}) == 1)
            vals = [v for v in (mean_score([r["text"]])[0] for r in rs) if v is not None]
            if len(vals) >= 2:
                sds.append(float(np.std(vals, ddof=1)))
    rate = failures / (k * len(texts))
    info = {"prompts": len(texts), "scoring_calls": k * len(texts), "calls": sum(len(v) for v in texts.values()),
            "new_calls": len(new), "parse_failures": failures, "parse_failure_rate": rate,
            "refusals": sum(r.get("stop_reason") == "refusal" for rs in texts.values() for r in rs),
            "determinism": {"prompts": len(identical), "share_identical_text": float(np.mean(identical)) if identical else None,
                            "sd_median": float(np.median(sds)) if sds else None},
            "responses": [r for rs in texts.values() for r in rs]}
    if rate > cfg["judge"]["max_parse_failure_rate"]:
        raise RunStopped(f"parse failure rate {rate:.1%} > {cfg['judge']['max_parse_failure_rate']:.0%} "
                         f"({failures} of {k * len(texts)}): stopped before any statistic")
    return scores, info


# ---------------------------------------------------------------- AUROC and D


def auroc_block(scores: Scores, runs: pd.DataFrame, cfg: dict) -> dict:
    report, head = cfg["checkpoints"]["report"], cfg["classes"]["headline"]
    h = matched_step_auroc(scores, restrict_classes(runs, head), report)
    return {"headline": h.mean, "per_t": {str(t): v for t, v in h.per_checkpoint.items()},
            "per_class": {c: matched_step_auroc(scores, restrict_classes(runs, [c]), report).mean for c in head}}


def paired_median_diff(draws: Sequence[Scores], ref: Scores, runs: pd.DataFrame, checkpoints: Sequence[int],
                       n: int, seed: int) -> dict:
    """median_j (AUROC_j - AUROC_ref), each AUROC the mean over checkpoints, with a paired task-group
    cluster-bootstrap 95% CI. The resampling is evaluation._cluster_bootstrap's, call for call: one
    group draw per replicate shared by every checkpoint, every fit draw and the reference."""
    sets = {t: prefix_eval_set(runs, t) for t in checkpoints}
    s_ref = {t: np.array([ref[u][t] for u in s.uids], dtype=float) for t, s in sets.items()}
    s_d = [{t: np.array([d[u][t] for u in s.uids], dtype=float) for t, s in sets.items()} for d in draws]
    groups = sorted({g for s in sets.values() for g in s.task_groups})
    group_index = {g: k for k, g in enumerate(groups)}
    rows = {t: _rows_by_group(s, group_index) for t, s in sets.items()}

    def stat(idx: dict) -> tuple[float, np.ndarray]:
        y = {t: sets[t].y[idx[t]] for t in sets}
        ref_m = np.mean([auroc(y[t], s_ref[t][idx[t]]) for t in sets])
        diffs = np.array([np.mean([auroc(y[t], d[t][idx[t]]) for t in sets]) - ref_m for d in s_d])
        return float(np.median(diffs)), diffs

    point, per_draw = stat({t: np.arange(len(s.uids)) for t, s in sets.items()})
    rng = np.random.default_rng(seed)  # the same Generator calls as _cluster_bootstrap
    reps, skipped = [], 0
    for _ in range(n):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = {t: np.concatenate([rows[t][k] for k in picked]) for t in sets}
        if any(np.unique(sets[t].y[idx[t]]).size < 2 for t in sets):
            skipped += 1
            continue
        reps.append(stat(idx)[0])
    if skipped > MAX_SKIPPED_FRACTION * n:
        raise ValueError(f"{skipped}/{n} replicates had a one-class checkpoint")
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi), "n_skipped": skipped, "B": n, "seed": seed,
            "per_draw_diff": per_draw.tolist(), "share_draws_gt_0": float((per_draw > 0).mean())}


# ---------------------------------------------------------------- J-2 operating point (E3's machinery)


def operating_point(arms: dict[str, Callable[[int], Scores]], runs: pd.DataFrame, e3_cfg: dict, n: int,
                    family: str = "llama") -> dict:
    """E3's n > 0 loop for arms that spend no target runs (m = n): the same folds, draws, theta rules and
    pooled counts, so an arm that E3 ran reproduces E3's cells. arms[name](r) gives the scores for draw r."""
    fam_idx = [e3_cfg["families"]["primary"], *e3_cfg["families"]["descriptive"]].index(family)
    report, alphas = e3_cfg["checkpoints"]["report"], e3_cfg["thresholds"]["alphas"]
    classes = e3_cfg["metrics"]["classes"]["headline"]
    R, dseed, hseed = e3_cfg["draws"]["R"]["hgb_arms"], e3_cfg["draws"]["seed"], e3_cfg["budgets"]["half_split"]["seed"]
    folds = grouped_kfold(runs, family, e3_cfg["cross_fit"]["folds"]["k"], e3_cfg["cross_fit"]["folds"]["seed"],
                          eval_only_families=())
    pool = _Pool()
    extra: dict[tuple, dict] = {}  # (arm, rule, alpha, t, r) -> first-crossing counts and ties
    for t in report:
        for f, (other, held) in enumerate(folds):
            eval_runs = runs[runs["uid"].isin(set(held))]
            draws = calibration_draws(eligible_uids(runs, other, t), n, R, dseed, hseed, (fam_idx, t, f))
            if draws is None:
                for arm in arms:
                    for r in range(R):
                        pool.skip(arm, n, t, r, alphas)
                continue
            for r, d in enumerate(draws):
                for arm, provider in arms.items():
                    sc = provider(r)
                    thetas, m = _thetas(sc, d.uids, t, alphas)
                    pool.add(arm, n, t, r, _rates(sc, eval_runs, thetas, t, classes), thetas, m)
                    cal = np.array([calibration_score(sc[u], t) for u in d.uids])
                    for (rule, a), theta in thetas.items():
                        g = group_counts(sc, theta, eval_runs, t, classes)
                        e = extra.setdefault((arm, rule, a, t, r), {"pos": 0, "first": 0, "tied": 0, "m": 0})
                        e["pos"] += int(g["n_pos"].sum())
                        e["first"] += int((g["n_pos_alarm"] - g["n_pos_alarm_pre_tau"]).sum())
                        e["tied"] += int((cal == theta).sum())
                        e["m"] += m
    frame = pool.frame(family)
    cells = summarise(frame, e3_cfg)
    for arm in arms:
        for rule in ("conformal", "plugin"):
            for a in alphas:
                firsts, ties = [], []
                for r in range(R):
                    per_t = [extra.get((arm, rule, a, t, r)) for t in report]
                    if all(x and x["pos"] for x in per_t):
                        firsts.append(np.mean([x["first"] / x["pos"] for x in per_t]))
                        ties.append(np.mean([x["tied"] / x["m"] for x in per_t]))
                node = cells[arm][str(n)][rule][str(a)]["headline"]
                node["tpr_first_crossing_median"] = float(np.median(firsts)) if firsts else math.nan
                node["calibration_share_tied_at_theta_mean"] = float(np.mean(ties)) if ties else math.nan
    return {"cells": cells, "frame": frame}


def max_abs_diff(a, b, path: str = "") -> tuple[float, str]:
    """Largest numeric difference between two nested JSON-like values over b's keys (a may hold more)."""
    if isinstance(b, dict):
        worst = (0.0, "")
        for k, v in b.items():
            if k not in a:
                return math.inf, f"{path}/{k} missing"
            worst = max(worst, max_abs_diff(a[k], v, f"{path}/{k}"), key=lambda x: x[0])
        return worst
    if b is None:  # the JSON writer turned NaN into null
        ok = a is None or (isinstance(a, float) and math.isnan(a))
        return (0.0, path) if ok else (math.inf, f"{path} {a!r} != null")
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        x, y = float(a), float(b)
        if math.isnan(x) and math.isnan(y):
            return 0.0, path
        return abs(x - y), path
    if a != b:
        return math.inf, f"{path} {a!r} != {b!r}"
    return 0.0, path


# ---------------------------------------------------------------- cost: CPU time per step


def cpu_per_step(fn: Callable[[], object], steps: int) -> dict:
    """process_time (CPU, not wall) of one call of fn, per step it scores."""
    start = time.process_time()
    fn()
    cpu = time.process_time() - start
    return {"cpu_s_total": cpu, "steps": steps, "cpu_ms_per_step": 1000 * cpu / steps if steps else math.nan}


def require_complete(scores: Scores, runs: pd.DataFrame, cfg: dict) -> None:
    """Every healthy and headline run needs a judge score at every scored step it has. A missing score
    would drop a run from one scorer's AUROC set (unpairing D) and break a calibration draw; how to
    handle it was not pre-registered, so the run stops and reports instead of improvising."""
    missing = [(r["uid"], t) for _, r in restrict_classes(runs, cfg["classes"]["headline"]).iterrows()
               for t in cfg["checkpoints"]["scored"] if r["T"] >= t + 1 and not np.isfinite(scores[r["uid"]][t])]
    if missing:
        raise RunStopped(f"{len(missing)} judge scores missing (parse failures), e.g. {missing[:3]}; "
                         "the handling of missing scores needs an owner decision")
