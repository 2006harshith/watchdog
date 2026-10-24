"""SP5 Step 3b, E2 diagnostics (interpretation only; the pre-registered gate is not revisited).

Why the llama ESN scores 0.846 in E2 (ESN_i, headline) but 0.646 in SP4 4d (monitor A, all classes).
The two differ in class mix, in pooling (5 fold models z-scored and pooled vs 1 model, raw) and in
fit pool (ollama7b healthy runs outside the test fold's task groups vs outside all llama groups).
Each row below swaps one of those; per-fold columns remove pooling from the metric itself. Two more
rows ask where the context_corruption signal lives: the ESN without its surprisal channel u, and
MS-raw with and without its surprisal features.
"""

import copy
import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn.monitor import Run
from watchdog_agent.evaluation import matched_step_auroc, paired_matched_step_diff, prefix_eval_set
from watchdog_agent.experiments.e1 import (
    FoldPlan,
    MonitorFactory,
    default_monitor_factory,
    git_state,
    plan_grouped_folds,
    split_by_task_group,
)
from watchdog_agent.experiments.e2 import (
    Scores,
    build_test_set,
    build_training_set,
    default_esn_factory,
    esn_channels,
    esn_scores,
    evaluate,
    feature_columns,
    fit_and_score,
    to_score_dict,
)
from watchdog_agent.experiments.sp4_audit import standardised_running_max

# _aggregate is features.py's private prefix aggregation: D-7 must aggregate masked step rows exactly
# as prefix_feature_table does, and re-implementing it here could drift from the real features.
from watchdog_agent.features import (
    FEATURE_GROUPS,
    STEP_FEATURES,
    _aggregate,
    prefix_feature_table,
    step_feature_rows,
)
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


def sp4_plans(episodes: pd.DataFrame, e1_cfg: dict) -> list[FoldPlan]:
    return plan_grouped_folds(
        episodes, source_corpus=e1_cfg["source_corpus"], target_corpus=e1_cfg["target_corpus"],
        target_family=e1_cfg["target_family"], k=e1_cfg["k"], fold_seed=e1_cfg["fold_seed"],
        val_fraction=e1_cfg["val_fraction"],
    )


def run_getter(episodes: pd.DataFrame) -> Callable[[str], Run]:
    frame = episodes.set_index("uid")

    def to_run(uid: str) -> Run:
        cls = frame.at[uid, "failure_class"]
        return Run(uid, frame.at[uid, "steps_parsed"], None if pd.isna(cls) else cls)

    return to_run


def sp4_fold_scores(
    episodes: pd.DataFrame,
    e1_cfg: dict,
    checkpoints: Sequence[int],
    make_monitor: MonitorFactory,
    choose_pool: Callable[[FoldPlan], tuple[list[str], list[str]]] | None = None,
) -> tuple[Scores, Scores, list[list[str]]]:
    """SP4 4d's monitor A (E1 folds and pools), each fold model scoring its own eval runs: the raw
    running max and SP4's per-fold, per-checkpoint standardised version, from the same fitted models.
    Mirrors sp4_audit.matched_step_esn_scores for A only, keeping the raw scores it discards.
    choose_pool(plan) -> (fit uids, val uids) replaces a fold's pool (D-6 swaps); None keeps SP4's."""
    plans = sp4_plans(episodes, e1_cfg)
    to_run = run_getter(episodes)

    raw, std = {}, {}
    for plan in plans:
        fit, val = choose_pool(plan) if choose_pool else (plan.fit["A"], plan.val["A"])
        monitor = make_monitor()
        monitor.fit([to_run(u) for u in fit], [to_run(u) for u in val])
        val_streams = [monitor.score(to_run(u)).per_step for u in val]
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


# ---------------------------------------------------------------- SP5 close-out: D-6

CLOSEOUT_KEYS = {"what", "git", "config", "d6", "d7"}


def subset_draws(uids: Sequence[str], size: int, n: int, seed: int) -> list[list[str]]:
    """n distinct random size-subsets of uids from one seeded Generator (uids sorted first, so the
    draws do not depend on the pool's row order)."""
    pool = sorted(uids)
    if size > len(pool) or math.comb(len(pool), size) < n:
        raise ValueError(f"cannot draw {n} distinct {size}-subsets of {len(pool)} runs")
    rng = np.random.default_rng(seed)
    draws, seen = [], set()
    while len(draws) < n:
        draw = sorted(rng.choice(pool, size=size, replace=False).tolist())
        if frozenset(draw) not in seen:
            seen.add(frozenset(draw))
            draws.append(draw)
    return draws


def _headline_mean(scores: Scores, runs: pd.DataFrame, cfg: dict) -> float:
    return matched_step_auroc(
        scores, restrict_classes(runs, cfg["classes"]["headline"]), cfg["checkpoints"]["primary"]
    ).mean


def _esn_running_max(
    monitor, runs: pd.DataFrame, to_run: Callable[[str], Run], carry: dict[str, int] | None = None
) -> Scores:
    out = {}
    for uid in runs["uid"]:
        step = (carry or {}).get(uid)
        res = monitor.score(to_run(uid)) if step is None else monitor.score(to_run(uid), carry_forward_step=step)
        # np.maximum.accumulate: running maximum, so the score at t only uses steps 0..t.
        out[uid] = np.maximum.accumulate(np.asarray(res.per_step, dtype=float))
    return out


def _fit_esn(make_esn: Callable, channels: list[str], fit: Sequence[str], val: Sequence[str], to_run: Callable):
    monitor = make_esn(channels)
    monitor.fit([to_run(u) for u in fit], [to_run(u) for u in val])
    return monitor


def d6_subset_sweep(episodes: pd.DataFrame, cfg: dict, make_esn: Callable) -> dict:
    """ESN_i refitted on random fit_size-of-pool subsets of its own healthy pool (the rest is the
    validation set, which only sets the alarm threshold and so cannot move AUROC)."""
    runs, co, esn = episodes[episodes["family"] == FAMILY], cfg["closeout"], cfg["esn"]
    pool = e2_healthy_pool(episodes, FAMILY, cfg, "i")
    channels, to_run = esn_channels(pool, runs, cfg), run_getter(episodes)
    e2_fit, e2_val = split_by_task_group(pool, esn["val_fraction"], esn["val_seed"])
    e2_model = _fit_esn(make_esn, channels, e2_fit, e2_val, to_run)
    e2_value = _headline_mean(_esn_running_max(e2_model, runs, to_run), runs, cfg)
    draws = subset_draws(pool["uid"], co["fit_size"], co["n_subsets"], co["seed"])
    aurocs = []
    for fit in draws:
        val = sorted(set(pool["uid"]) - set(fit))
        model = _fit_esn(make_esn, channels, fit, val, to_run)
        aurocs.append(_headline_mean(_esn_running_max(model, runs, to_run), runs, cfg))
    a = np.array(aurocs)
    return {
        "pool_runs": len(pool), "fit_size": co["fit_size"], "channels": channels, "seed": co["seed"],
        "e2_split_auroc": e2_value,
        "aurocs": aurocs, "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
        "share_of_draws_at_or_above_e2_split": float((a >= e2_value).mean()),
        "e2_split_is_one_of_the_draws": frozenset(e2_fit) in {frozenset(d) for d in draws},
    }


def d6_fit_pool_only(episodes: pd.DataFrame, cfg: dict, e1_cfg: dict, make_monitor: MonitorFactory) -> dict:
    """(a) SP4's structure (5 fold models, per-fold z-scoring, pooled) with every fold model fitted on
    E2's pool and split. Identical data and seed give identical models and one shared validation set,
    so this must equal E2's ESN_i: the structure alone adds nothing."""
    runs, esn = episodes[episodes["family"] == FAMILY], cfg["esn"]
    pool = e2_healthy_pool(episodes, FAMILY, cfg, "i")
    fit, val = split_by_task_group(pool, esn["val_fraction"], esn["val_seed"])
    raw, std, folds = sp4_fold_scores(episodes, e1_cfg, cfg["checkpoints"]["score"], make_monitor, lambda plan: (fit, val))
    head = restrict_classes(runs, cfg["classes"]["headline"])
    return {
        "sp4_structure_e2_pool_standardised": _headline_mean(std, runs, cfg),
        "sp4_structure_e2_pool_raw": _headline_mean(raw, runs, cfg),
        "per_fold_headline": per_fold_mean_auroc(raw, head, folds, cfg["checkpoints"]["primary"]),
    }


def d6_size_matched(episodes: pd.DataFrame, cfg: dict, e1_cfg: dict, make_monitor: MonitorFactory) -> dict:
    """(b') SP4's own fold pools subsampled to E2's fit size, validation sets unchanged. Leak-free
    (each fit set stays inside its fold's pool). Per-fold AUROC only: no score crosses a fold."""
    runs, co = episodes[episodes["family"] == FAMILY], cfg["closeout"]
    head = restrict_classes(runs, cfg["classes"]["headline"])
    llama_groups = set(runs["task_group"])
    group_of = episodes.set_index("uid")["task_group"]
    draws = []
    for d in range(co["size_matched_draws"]):
        rng = np.random.default_rng(co["seed"] + d)
        folds_info: list[dict] = []

        def choose(plan: FoldPlan, rng=rng, folds_info=folds_info):
            pool = sorted(plan.fit["A"])
            fit = sorted(rng.choice(pool, size=min(co["fit_size"], len(pool)), replace=False).tolist())
            eval_groups = set(group_of[plan.eval_uids])
            folds_info.append({
                "fold": plan.fold, "n_fit": len(fit), "n_fold_pool": len(pool),
                "fit_inside_fold_pool": set(fit) <= set(pool) and not (set(group_of[fit]) & eval_groups),
                "n_fit_on_llama_task_groups": int(group_of[fit].isin(llama_groups).sum()),
            })
            return fit, plan.val["A"]

        raw, _, folds = sp4_fold_scores(episodes, e1_cfg, cfg["checkpoints"]["score"], make_monitor, choose)
        draws.append({
            "draw": d,
            "per_fold_headline": per_fold_mean_auroc(raw, head, folds, cfg["checkpoints"]["primary"]),
            "folds": folds_info,
        })
    means = np.array([x["per_fold_headline"]["mean"] for x in draws])
    return {
        "fit_size": co["fit_size"], "draws": draws,
        "per_fold_mean": {"median": float(np.median(means)), "min": float(means.min()), "max": float(means.max())},
    }


# ---------------------------------------------------------------- SP5 close-out: D-7


def _group_step_features(group: str) -> list[str]:
    return [n for n, (g, _) in STEP_FEATURES.items() if g == group]


def mask_rows(rows: Sequence[dict[str, float]], tau: int, group: str, mode: str) -> list[dict[str, float]]:
    """Step tau's per-step inputs of one feature group, masked: "nan" (continuous features only; a
    NaN in a count makes the count NaN) or "carry" (step tau-1's values, position kept coherent)."""
    if not 1 <= tau < len(rows):
        raise ValueError(f"tau={tau} needs a step tau-1 and tau inside a {len(rows)}-step run")
    names = _group_step_features(group)
    if mode == "nan":
        kinds = {STEP_FEATURES[n][1] for n in names}
        if kinds != {"continuous"}:
            raise ValueError(f"nan masking needs continuous features; group {group} has {sorted(kinds)}")
    elif mode != "carry":
        raise ValueError(f"unknown mode {mode!r}")
    out, new, prev = list(rows), dict(rows[tau]), rows[tau - 1]
    for n in names:
        if mode == "nan":
            new[n] = math.nan
        elif n not in ("step_index", "steps_since_error"):  # step_index is position, not an input
            new[n] = prev[n]
    if mode == "carry" and "steps_since_error" in names:
        # Consistent with the carried is_error: 0 on an error step, else one more than at tau-1.
        new["steps_since_error"] = 0 if new["is_error"] else prev["steps_since_error"] + 1
    out[tau] = new
    return out


def masked_prefix_table(runs: pd.DataFrame, checkpoints: Sequence[int], group: str, mode: str) -> pd.DataFrame:
    """prefix_feature_table for these runs with each failed run's step tau masked (mask_rows).
    Rows after tau keep their real-history values: only step tau's own inputs are hidden."""
    records, index = [], []
    for uid, steps, cls, tau in zip(runs["uid"], runs["steps_parsed"], runs["failure_class"], runs["tau"]):
        rows = step_feature_rows(steps)
        if not pd.isna(cls) and int(tau) < len(rows):
            rows = mask_rows(rows, int(tau), group, mode)
        for t in checkpoints:
            if t < len(steps):
                records.append(_aggregate(rows[: t + 1], t))
                index.append((uid, t))
    columns = [c for names in FEATURE_GROUPS.values() for c in names]
    return pd.DataFrame.from_records(
        records, index=pd.MultiIndex.from_tuples(index, names=["uid", "t"]), columns=columns
    )


def d7_evidence(episodes: pd.DataFrame) -> dict:
    """What a tool step's MS inputs and the ESN's text embedding can see of the tool result."""
    out = {}
    for fam, runs in episodes.groupby("family"):
        n_tool, lat_both, lat_ge, text_has_result = 0, 0, 0, 0
        for steps in runs["steps_parsed"]:
            for s in steps:
                events = s.get("tool_events") or []
                if not events:
                    continue
                n_tool += 1
                ev = events[0]
                if s.get("latency_s") is not None and ev.get("latency_s") is not None:
                    lat_both += 1
                    lat_ge += float(s["latency_s"]) >= float(ev["latency_s"])
                result = str(ev.get("result") or "")[:40]
                text_has_result += bool(result) and result in str(s.get("text") or "")
        out[fam] = {
            "tool_steps": n_tool,
            "tool_steps_with_both_latencies": lat_both,
            "share_step_latency_ge_tool_latency": lat_ge / lat_both if lat_both else None,
            "share_step_text_contains_tool_result": text_has_result / n_tool if n_tool else None,
        }
    return out


def _before_after(before: Scores, after: Scores, runs: pd.DataFrame, cfg: dict) -> dict:
    cl, b_cfg, primary = cfg["classes"], cfg["bootstrap"], cfg["checkpoints"]["primary"]
    n, seed = b_cfg["n_secondary"], b_cfg["seed"]
    subsets = {"headline": cl["headline"], **{c: [c] for c in cl["headline"] + cl["error_visible"] + cl["api"]}}
    out: dict = {"per_class": {}}
    for name, classes in subsets.items():
        b = evaluate(before, runs, classes, cfg, n).get("mean_primary")
        if b is None:
            continue
        a = evaluate(after, runs, classes, cfg, n)["mean_primary"]
        sub = restrict_classes(runs, classes)
        try:
            diff = paired_matched_step_diff(before, after, sub, primary, n, seed=seed).mean._asdict()
        except ValueError as e:  # too few task groups of one class for a cluster CI
            diff = {"point": a["point"] - b["point"], "ci_unavailable": str(e)}
        block = {"before": b, "after": a, "after_minus_before": diff}
        if name == "headline":
            out["headline"] = block
        else:
            out["per_class"][name] = block
    return out


def d7_masking(table: pd.DataFrame, episodes: pd.DataFrame, cfg: dict, make_esn: Callable) -> dict:
    seed = cfg["recalibration"]["target_cross_fit"]["seed"]
    score_t, params = cfg["checkpoints"]["score"], cfg["model"]["primary"]
    out = {}
    llama = episodes[episodes["family"] == FAMILY]

    train = build_training_set(table, episodes, FAMILY, cfg)
    X_before = build_test_set(table, episodes, FAMILY, cfg, seed).X["raw"]
    fitted, s_before = fit_and_score(train, X_before, train.X["raw"], feature_columns("MS", cfg), params)
    for mode in ("nan", "carry"):
        X_after = build_test_set(masked_prefix_table(llama, score_t, "MS", mode), episodes, FAMILY, cfg, seed).X["raw"]
        out[f"MS-raw@llama:{mode}"] = _before_after(
            to_score_dict(s_before, llama), to_score_dict(fitted.score(X_after), llama), llama, cfg
        )

    esn, to_run = cfg["esn"], run_getter(episodes)
    pool = e2_healthy_pool(episodes, FAMILY, cfg, "i")
    fit, val = split_by_task_group(pool, esn["val_fraction"], esn["val_seed"])
    monitor = _fit_esn(make_esn, esn_channels(pool, llama, cfg), fit, val, to_run)
    failed = llama[llama["failure_class"].notna()]
    carry = {u: int(t) for u, t in zip(failed["uid"], failed["tau"])}
    out["ESN_i@llama:carry"] = _before_after(
        _esn_running_max(monitor, llama, to_run), _esn_running_max(monitor, llama, to_run, carry), llama, cfg
    )

    gemini = episodes[episodes["family"] == "gemini"]
    train_g = build_training_set(table, episodes, "gemini", cfg)
    Xg_before = build_test_set(table, episodes, "gemini", cfg, seed).X["pct"]
    fitted_g, sg_before = fit_and_score(train_g, Xg_before, train_g.X["pct"], feature_columns("TL", cfg), params)
    Xg_after = build_test_set(masked_prefix_table(gemini, score_t, "TL", "carry"), episodes, "gemini", cfg, seed).X["pct"]
    out["TL-pct@gemini:carry"] = _before_after(
        to_score_dict(sg_before, gemini), to_score_dict(fitted_g.score(Xg_after), gemini), gemini, cfg
    )
    return out


# D-7c: masking step tau alone cannot test context_corruption, because the injector keeps
# corrupting later results. Instead every run (fit, val, test; healthy and failed) loses its tool
# result content, label-free, while every size is kept.

RESULT_FILLER = "x"
ARROW = " -> "


def blind_results(steps: Sequence[dict]) -> list[dict]:
    """Each tool step's result replaced by a same-length filler, in tool_events[].result and in the
    "[name(args) -> result]" text. The text shows the result cut to 100 chars, so the span is found
    by position (after the one arrow, before the closing bracket), not by searching for the result.
    Reasoning before the call is model output and stays. Raises rather than leave content in."""
    out = []
    for i, step in enumerate(steps):
        events = step.get("tool_events") or []
        if not events:
            out.append(step)
            continue
        text = str(step.get("text") or "")
        if len(events) != 1 or text.count(ARROW) != 1 or not text.endswith("]"):
            raise ValueError(f"step {i}: expected one tool event rendered as '[name(args) -> result]'")
        start = text.index(ARROW) + len(ARROW)
        new = dict(step)
        new["text"] = text[:start] + RESULT_FILLER * (len(text) - 1 - start) + "]"
        ev = dict(events[0])
        if ev.get("result") is not None:
            ev["result"] = RESULT_FILLER * len(str(ev["result"]))
        new["tool_events"] = [ev]
        out.append(new)
    return out


def blind_episodes(episodes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """blind_results on every run; every other column (labels, tau, groups, T) is untouched."""
    out = episodes.copy()
    out["steps_parsed"] = [blind_results(steps) for steps in episodes["steps_parsed"]]
    counts = {}
    for fam, before, after in zip(episodes["family"], episodes["steps_parsed"], out["steps_parsed"]):
        c = counts.setdefault(fam, {"tool_steps_blinded": 0, "text_chars_changed": 0})
        for b, a in zip(before, after):
            if b is not a:
                c["tool_steps_blinded"] += 1
                c["text_chars_changed"] += sum(x != y for x, y in zip(b["text"], a["text"]))
    return out, counts


def d7_result_blind(table: pd.DataFrame, episodes: pd.DataFrame, cfg: dict, make_esn: Callable) -> dict:
    """ESN_i (llama, gemini) and TEXT-raw (llama), refitted and rescored with result content blinded
    in every run. Same pools, splits, channels and training rows as E2: only the steps differ."""
    blind, counts = blind_episodes(episodes)
    out: dict = {"counts": counts}
    esn = cfg["esn"]
    for family in (FAMILY, "gemini"):
        scored = []
        for eps in (episodes, blind):
            runs, to_run = eps[eps["family"] == family], run_getter(eps)
            pool = e2_healthy_pool(eps, family, cfg, "i")
            fit, val = split_by_task_group(pool, esn["val_fraction"], esn["val_seed"])
            monitor = _fit_esn(make_esn, esn_channels(pool, runs, cfg), fit, val, to_run)
            scored.append(_esn_running_max(monitor, runs, to_run))
        out[f"ESN_i@{family}"] = _before_after(*scored, episodes[episodes["family"] == family], cfg)

    seed, params, cols = cfg["recalibration"]["target_cross_fit"]["seed"], cfg["model"]["primary"], feature_columns("TEXT", cfg)
    needed = sorted(set(cfg["checkpoints"]["train"]) | set(cfg["checkpoints"]["score"]))
    blind_table = prefix_feature_table(dict(zip(blind["uid"], blind["steps_parsed"])), needed)
    llama = episodes[episodes["family"] == FAMILY]
    scored = []
    for tab, eps in ((table, episodes), (blind_table, blind)):
        train = build_training_set(tab, eps, FAMILY, cfg)
        _, s = fit_and_score(train, build_test_set(tab, eps, FAMILY, cfg, seed).X["raw"], train.X["raw"], cols, params)
        scored.append(to_score_dict(s, llama))
    out["TEXT-raw@llama"] = _before_after(*scored, llama, cfg)
    return out


def run_e2_closeout(
    episodes: pd.DataFrame,
    cfg: dict,
    e1_cfg: dict,
    make_esn: Callable | None = None,
    make_monitor: MonitorFactory | None = None,
    table: pd.DataFrame | None = None,
    git: dict | None = None,
) -> dict:
    make_esn = make_esn or default_esn_factory(cfg)
    make_monitor = make_monitor or default_monitor_factory(e1_cfg)
    if table is None:
        needed = sorted(set(cfg["checkpoints"]["train"]) | set(cfg["checkpoints"]["score"]))
        table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    return {
        "what": "SP5 close-out diagnostics, interpretation only (the E2 gate is final). D-6: ESN_i "
        "instability over random fit subsets; (a) SP4 structure with E2's pool; (b') SP4 fold pools "
        "size-matched to E2's fit size. D-7: what MS and the ESN read on a tool step; MS-raw, ESN_i "
        "(llama) and TL-pct (gemini) rescored with step tau's inputs masked; (c) ESN_i (llama, gemini) "
        "and TEXT-raw (llama) refitted and rescored with every run's tool results replaced by "
        "same-length filler (label-free, all runs). Mean matched-step AUROC "
        "over the primary checkpoints, headline classes unless per class.",
        "git": git if git is not None else git_state(),
        "config": cfg["closeout"],
        "d6": {
            "subsets": d6_subset_sweep(episodes, cfg, make_esn),
            "a_fit_pool_only": d6_fit_pool_only(episodes, cfg, e1_cfg, make_monitor),
            "b_size_matched": d6_size_matched(episodes, cfg, e1_cfg, make_monitor),
        },
        "d7": {
            "a_evidence": d7_evidence(episodes),
            "b_masking": d7_masking(table, episodes, cfg, make_esn),
            "c_result_blind": d7_result_blind(table, episodes, cfg, make_esn),
        },
    }


def check_closeout_reproduction(result: dict, e2: dict, tol: float = 1e-9) -> dict[str, tuple[float, float]]:
    """Every re-run of an E2 number must match it exactly; otherwise before/after compares a
    different model."""
    fam = e2["families"]
    ref_esn = fam[FAMILY]["arms"]["ESN_i"]["headline"]["mean_primary"]["point"]
    pairs = {
        "d6.subsets.e2_split_auroc": (result["d6"]["subsets"]["e2_split_auroc"], ref_esn),
        "d6.a_fit_pool_only.sp4_structure_e2_pool_raw": (
            result["d6"]["a_fit_pool_only"]["sp4_structure_e2_pool_raw"], ref_esn),
    }
    refs = {("b_masking", "MS-raw@llama:nan"): (FAMILY, "MS-raw"), ("b_masking", "ESN_i@llama:carry"): (FAMILY, "ESN_i"),
            ("b_masking", "TL-pct@gemini:carry"): ("gemini", "TL-pct"),
            ("c_result_blind", "ESN_i@llama"): (FAMILY, "ESN_i"), ("c_result_blind", "ESN_i@gemini"): ("gemini", "ESN_i"),
            ("c_result_blind", "TEXT-raw@llama"): (FAMILY, "TEXT-raw")}
    for (part, key), (family, arm) in refs.items():
        block, arms = result["d7"][part][key], fam[family]["arms"][arm]
        pairs[f"d7.{part}.{key}.headline.before"] = (block["headline"]["before"]["point"],
                                                     arms["headline"]["mean_primary"]["point"])
        for cls, v in block["per_class"].items():
            pairs[f"d7.{part}.{key}.{cls}.before"] = (v["before"]["point"], arms["per_class"][cls]["mean_primary"]["point"])
    bad = {k: v for k, v in pairs.items() if not abs(v[0] - v[1]) <= tol}
    if bad:
        raise AssertionError(f"close-out does not reproduce E2: {bad}")
    return pairs
