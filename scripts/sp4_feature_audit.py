"""SP4 Step 4 feature audit -> results/sp4_feature_audit.json.

Run from a clean commit: uv run python scripts/sp4_feature_audit.py [--config configs/sp4_feature_audit.yaml]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.evaluation import (
    matched_step_auroc,
    matched_step_cluster_ci,
    paired_matched_step_diff,
    prefix_eval_set,
)
from watchdog_agent.experiments.e1 import default_monitor_factory, git_state
from watchdog_agent.experiments.sp4_audit import (
    ceiling_counts,
    family_predictability_auroc,
    matched_step_esn_scores,
    top_family_features,
    univariate_auroc,
)
from watchdog_agent.features import FEATURE_GROUPS, prefix_feature_table

OUT = Path("results/sp4_feature_audit.json")


def _iv(interval) -> dict:
    return interval._asdict()


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def checkpoint_counts(episodes: pd.DataFrame, cfg: dict) -> dict:
    out = {}
    for family, runs in episodes.groupby("family"):
        out[family] = {}
        for t in cfg["checkpoints"]:
            s = prefix_eval_set(runs, t)
            pos_cls = runs.set_index("uid").loc[[u for u, y in zip(s.uids, s.y) if y == 1], "failure_class"]
            out[family][t] = {
                "n_pos": s.n_pos,
                "n_pos_behaviour": int(pos_cls.isin(cfg["behaviour_classes"]).sum()),
                "n_pos_api": int(pos_cls.isin(cfg["api_classes"]).sum()),
                "n_healthy": s.n_neg,
                "n_excluded_tau_after_t": s.n_excluded,
                "n_without_step_t": s.n_without_step,
                "healthy_task_groups": len(set(s.task_groups[s.y == 0])),
            }
    return out


def _family_block(healthy: pd.DataFrame, table: pd.DataFrame, fam_a: str, fam_b: str, t: int, cfg: dict) -> dict:
    sub = healthy[healthy["T"] >= t + 1]
    fam = (sub["family"] == fam_b).to_numpy(dtype=int)
    groups = sub["task_group"].to_numpy()
    rows = table.loc[[(u, t) for u in sub["uid"]]]
    fp = cfg["family_predictability"]
    block = {
        "n": {fam_a: int((fam == 0).sum()), fam_b: int((fam == 1).sum())},
        "task_groups": {fam_a: len(set(groups[fam == 0])), fam_b: len(set(groups[fam == 1]))},
    }
    # Each family needs a task group in every fold: with fewer groups than folds, some training
    # folds hold almost none of that family and the pooled out-of-fold AUROC turns meaningless
    # (the dry run gave 0.000 at booking t = 5, where llama's healthy runs span 2 task groups).
    if min(block["task_groups"].values()) < fp["n_splits"]:
        return block | {"skipped": f"a family has fewer than {fp['n_splits']} task groups"}
    block["auroc"] = {
        g: family_predictability_auroc(rows[FEATURE_GROUPS[g]].reset_index(drop=True), fam, groups, fp["n_splits"])
        for g in fp["groups"]
    }
    block["top_tl_features"] = top_family_features(rows[FEATURE_GROUPS["TL"]], fam, fp["top_k"])
    return block


def family_predictability(episodes: pd.DataFrame, table: pd.DataFrame, cfg: dict) -> dict:
    fp = cfg["family_predictability"]
    healthy = episodes[episodes["failure_class"].isna()]
    matched = {}
    for pair, sides in fp["matched_pairs"].items():
        (fam_a, corpora_a), (fam_b, corpora_b) = sides.items()
        pool = healthy[
            ((healthy["family"] == fam_a) & healthy["corpus"].isin(corpora_a))
            | ((healthy["family"] == fam_b) & healthy["corpus"].isin(corpora_b))
        ]
        matched[pair] = {"corpora": sides} | {
            str(t): _family_block(pool, table, fam_a, fam_b, t, cfg) for t in fp["checkpoints"]
        }
    all_runs = {}
    for fam_a, fam_b in fp["all_runs_pairs"]:
        pool = healthy[healthy["family"].isin([fam_a, fam_b])]
        all_runs[f"{fam_a}_vs_{fam_b}"] = {
            str(t): _family_block(pool, table, fam_a, fam_b, t, cfg) for t in fp["checkpoints"]
        }
    return {
        "what": "out-of-fold AUROC telling two families apart from HEALTHY runs; folds grouped by task "
        "group; ~0.5 = the feature set does not identify the family",
        "matched_pairs": matched,
        "all_runs": all_runs,
    }


def univariate_tl(episodes: pd.DataFrame, table: pd.DataFrame, cfg: dict) -> dict:
    uv = cfg["univariate"]
    runs = episodes[episodes["family"] == uv["family"]]
    healthy = runs[runs["failure_class"].isna()]
    out, flags = {}, []
    for t in cfg["checkpoints"]:
        out[str(t)] = {}
        for cls in cfg["behaviour_classes"] + cfg["api_classes"]:
            s = prefix_eval_set(pd.concat([healthy, runs[runs["failure_class"] == cls]]), t)
            if s.n_pos == 0:
                continue
            aurocs = {f: univariate_auroc(table, s, f) for f in FEATURE_GROUPS["TL"]}
            out[str(t)][cls] = {
                "kind": "behaviour" if cls in cfg["behaviour_classes"] else "api",
                "n_pos": s.n_pos,
                "n_healthy": s.n_neg,
                "auroc": aurocs,
            }
            for f, a in aurocs.items():
                if max(a, 1 - a) > uv["flag_threshold"]:
                    flags.append({"t": t, "class": cls, "feature": f, "auroc": a})
    return {
        "what": f"each TL feature alone at matched step t, {uv['family']} runs, healthy vs one class; "
        f"flag if max(AUROC, 1 - AUROC) > {uv['flag_threshold']}",
        "per_checkpoint": out,
        "flags": sorted(flags, key=lambda r: (r["class"], r["t"], -abs(r["auroc"] - 0.5))),
    }


def ceiling(episodes: pd.DataFrame, cfg: dict) -> dict:
    return {
        "what": "failed runs (tau <= t) whose tool-level prefix (calls, args, results, error flags, "
        "steps 0..t) equals a healthy run's prefix in the same task group",
        "per_family": {
            fam: {str(t): ceiling_counts(runs, t) for t in cfg["checkpoints"]}
            for fam, runs in episodes.groupby("family")
        },
    }


def esn_matched_step(episodes: pd.DataFrame, cfg: dict) -> dict:
    e1_cfg = yaml.safe_load(Path(cfg["esn"]["e1_config"]).read_text())
    checkpoints, primary = cfg["checkpoints"], cfg["primary_checkpoints"]
    n, seed = cfg["bootstrap"]["n"], cfg["bootstrap"]["seed"]
    scores, fold_info = matched_step_esn_scores(episodes, e1_cfg, checkpoints, default_monitor_factory(e1_cfg))
    runs = episodes[episodes["corpus"] == e1_cfg["target_corpus"]]

    monitors = {}
    for name in ("A", "B"):
        point = matched_step_auroc(scores[name], runs, checkpoints)
        per_t = matched_step_cluster_ci(scores[name], runs, checkpoints, n, seed=seed)
        mean_primary = matched_step_cluster_ci(scores[name], runs, primary, n, seed=seed)
        monitors[name] = {
            "per_checkpoint": {str(t): _iv(iv) for t, iv in per_t.per_checkpoint.items()},
            "mean_over_primary": _iv(mean_primary.mean),
            "mean_over_all_checkpoints": point.mean,
        }
    diff = paired_matched_step_diff(scores["A"], scores["B"], runs, primary, n, seed=seed)

    length = {uid: np.arange(1, T + 1, dtype=float) for uid, T in zip(runs["uid"], runs["T"])}
    length_res = matched_step_auroc(length, runs, checkpoints)
    if any(v != 0.5 for v in length_res.per_checkpoint.values()):
        raise AssertionError(f"prefix length is not 0.5 at a matched step: {length_res.per_checkpoint}")

    sets = {t: prefix_eval_set(runs, t) for t in checkpoints}
    e1 = json.loads(Path("results/e1_collapse.json").read_text())
    return {
        "what": "author ESN (esn_cusum_max), E1 folds/pools/seed; per-step score at t = running max of "
        "the CUSUM stream over steps 0..t, standardised per fold and checkpoint on that fold model's "
        "healthy validation runs with a step t; task-group cluster bootstrap",
        "e1_config": e1_cfg["esn"] | {"k": e1_cfg["k"], "fold_seed": e1_cfg["fold_seed"]},
        "washout_note": "the ESN emits 0 on steps 0..2 (vendored esn.py _WASHOUT = 3), so at t = 2 every "
        "run scores 0 and AUROC is 0.5 by construction",
        "counts": {
            str(t): {"n_pos": s.n_pos, "n_healthy": s.n_neg, "n_excluded_tau_after_t": s.n_excluded,
                     "healthy_task_groups": len(set(s.task_groups[s.y == 0]))}
            for t, s in sets.items()
        },
        "monitors": monitors,
        "paired_B_minus_A_primary": {
            "per_checkpoint": {str(t): _iv(iv) for t, iv in diff.per_checkpoint.items()},
            "mean": _iv(diff.mean),
        },
        "prefix_length_as_score": {str(t): v for t, v in length_res.per_checkpoint.items()},
        "reference_e1_episode_level_auroc": {m: e1["monitors"][m]["auroc"]["point"] for m in ("A", "B")},
        "per_fold": fold_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sp4_feature_audit.yaml")
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    episodes = load_episodes()
    table_checkpoints = sorted(set(cfg["checkpoints"]) | set(cfg["family_predictability"]["checkpoints"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), table_checkpoints)
    print(f"feature table {table.shape} in {time.time() - start:.0f}s")

    result = {
        "what": "SP4 Step 4 feature audit. No monitor is trained on features. a and b use no llama or "
        "gemini failure label; c counts failed runs per family (a data description); d scores the "
        "ESN baseline on llama labels exactly as E1 did",
        "git": git,
        "config": cfg,
        "n_features": {g: len(cols) for g, cols in FEATURE_GROUPS.items()},
        "counts": checkpoint_counts(episodes, cfg),
        "a_family_predictability": family_predictability(episodes, table, cfg),
    }
    print(f"4a done in {time.time() - start:.0f}s")
    result["b_univariate_tl_qwen"] = univariate_tl(episodes, table, cfg)
    result["c_tool_level_ceiling"] = ceiling(episodes, cfg)
    print(f"4b, 4c done in {time.time() - start:.0f}s")
    result["d_esn_matched_step"] = esn_matched_step(episodes, cfg)
    print(f"4d done in {time.time() - start:.0f}s")

    OUT.write_text(json.dumps(result, indent=2, default=_json_default))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
