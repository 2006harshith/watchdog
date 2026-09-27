"""SP3 diagnostics: why does the paper's in-domain monitor score 0.885 but only ~0.63 task-disjoint?

D-1  twin audit: runs whose tool-call sequences (name, args, result per step) are identical.
D-2  monitor B (llama-refit) under three splits of the same size: author random, random with
     twin clusters kept on one side, task-disjoint; 20 seeds each.
D-3  monitor A (qwen-fit) scored on the author's llama test set, fitted on qwen runs whose task
     group does / does not appear in that test set.
The ESN seed stays at the author default, so the spread across seeds is the split's alone.
"""

import hashlib
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn import AUTHOR_CHANNELS, ESNBaseline, Run
from watchdog_agent.experiments.author_protocol import author_split
from watchdog_agent.metrics import auroc

SplitResult = tuple[list[str], list[str], list[str], dict]  # fit, val, test uids, extra counts


def trajectory_hash(steps: list[dict]) -> str:
    """sha256 of the per-step tool-call sequence: (name, args, result) only, step boundaries kept.
    Text, timing, logprobs and event ids are ignored on purpose: a twin is a run that did the
    same thing and saw the same results, however it was worded or timed.
    """
    sequence = [
        [[ev.get("name"), ev.get("args"), ev.get("result")] for ev in (step.get("tool_events") or [])]
        for step in steps
    ]
    # sort_keys makes {"a":1,"b":2} and {"b":2,"a":1} hash the same.
    return hashlib.sha256(json.dumps(sequence, sort_keys=True).encode("utf-8")).hexdigest()


def _has_tool_calls(steps: list[dict]) -> bool:
    return any(step.get("tool_events") for step in steps)


# ---------------------------------------------------------------- D-1


def twin_audit(episodes: pd.DataFrame) -> dict:
    """Per corpus: exact-duplicate trajectories. Runs without any tool call are counted and left
    out, since all of them would hash alike without having done the same thing."""
    out = {}
    for corpus, sub in episodes.groupby("corpus"):
        with_tools = sub[sub["steps_parsed"].map(_has_tool_calls)]
        hashes = with_tools["steps_parsed"].map(trajectory_hash)
        clusters = [g for _, g in with_tools.groupby(hashes) if len(g) > 1]
        mixing = [g for g in clusters if g["failure_class"].isna().any() and g["failure_class"].notna().any()]
        mixing_runs = pd.concat(mixing) if mixing else with_tools.iloc[:0]
        out[corpus] = {
            "runs": len(sub),
            "runs_without_tool_calls": len(sub) - len(with_tools),
            "twin_clusters": len(clusters),
            "runs_in_twin_clusters": int(sum(len(g) for g in clusters)),
            "largest_cluster": int(max((len(g) for g in clusters), default=0)),
            "clusters_mixing_healthy_and_failed": len(mixing),
            "clusters_spanning_task_groups": int(sum(g["task_group"].nunique() > 1 for g in clusters)),
            # A failed run whose tool calls match a healthy run's exactly cannot be told apart by
            # any tool-level signal; these counts size that label-quality problem.
            "runs_in_mixing_clusters": {
                "total": len(mixing_runs),
                "by_label": {
                    "healthy": int(mixing_runs["failure_class"].isna().sum()),
                    "failed": int(mixing_runs["failure_class"].notna().sum()),
                },
                "by_failure_class": {
                    str(cls): int(n) for cls, n in mixing_runs["failure_class"].value_counts().items()
                },
            },
        }
    return out


def cross_corpus_twins(episodes: pd.DataFrame, corpus_a: str, corpus_b: str) -> dict:
    hashes = {
        c: set(episodes.loc[episodes["corpus"] == c, "steps_parsed"].map(trajectory_hash))
        for c in (corpus_a, corpus_b)
    }
    shared = hashes[corpus_a] & hashes[corpus_b]
    runs = episodes[episodes["corpus"].isin([corpus_a, corpus_b])]
    return {
        "shared_trajectories": len(shared),
        "runs_with_a_twin_in_the_other_corpus": int(runs["steps_parsed"].map(trajectory_hash).isin(shared).sum()),
    }


def overlap_counts(test_uids: list[str], fit_uids: list[str], hash_of: dict, group_of: dict) -> dict:
    """How many test runs have an exact twin, or share a task group, with the fit runs."""
    fit_hashes = {hash_of[u] for u in fit_uids}
    fit_groups = {group_of[u] for u in fit_uids}
    return {
        "test_runs": len(test_uids),
        "test_runs_with_twin_in_fit": sum(hash_of[u] in fit_hashes for u in test_uids),
        "test_runs_sharing_task_group_with_fit": sum(group_of[u] in fit_groups for u in test_uids),
    }


# ---------------------------------------------------------------- D-2 splits


def random_split(runs: list[Run], seed: int) -> SplitResult:
    """The author's procedure with a different seed (seed 0 is the author's exact split)."""
    fit, val, test = author_split(runs, seed)
    return [r.uid for r in fit], [r.uid for r in val], [r.uid for r in test], {}


def twin_aware_split(runs: list[Run], hash_of: dict, seed: int, n_fit: int, n_val: int) -> SplitResult:
    """Random split of healthy runs in whole twin clusters. Clusters fill fit, then val; the one
    cluster that overflows each is trimmed to hit n_fit / n_val exactly and its extra twins are
    dropped (used nowhere, counted), never sent to test. A failed run whose twin was fitted or
    used for the threshold is dropped from test too, since fit and val are healthy-only."""
    clusters = defaultdict(list)
    for r in runs:
        if r.failure_class is None:
            clusters[hash_of[r.uid]].append(r.uid)
    units = [clusters[h] for h in sorted(clusters)]
    random.Random(seed).shuffle(units)
    fit, val, test, dropped_healthy = [], [], [], 0
    for unit in units:
        if len(fit) < n_fit:
            take = unit[: n_fit - len(fit)]
            fit += take
            dropped_healthy += len(unit) - len(take)
        elif len(val) < n_val:
            take = unit[: n_val - len(val)]
            val += take
            dropped_healthy += len(unit) - len(take)
        else:
            test += unit
    if len(fit) != n_fit or len(val) != n_val:
        raise ValueError(f"not enough healthy runs for fit={n_fit}/val={n_val}")
    seen = {hash_of[u] for u in fit + val}
    failed = [r.uid for r in runs if r.failure_class is not None]
    kept = [u for u in failed if hash_of[u] not in seen]
    return fit, val, test + kept, {
        "healthy_dropped_trimmed_twins": dropped_healthy,
        "failed_dropped_twin_of_fit_or_val": len(failed) - len(kept),
    }


def task_disjoint_split(
    runs: list[Run], group_of: dict, seed: int, n_fit: int, n_val: int
) -> SplitResult | str:
    """Whole task groups to fit until it holds >= n_fit healthy runs, then to val until >= n_val;
    each is subsampled to exactly n_fit / n_val. Test = every run of the remaining groups.
    Returns a skip reason instead when this seed's group order cannot fill fit/val (a large
    group overshooting fit) or leaves test with one class."""
    rng = random.Random(seed)
    healthy_by_group = defaultdict(list)
    for r in runs:
        if r.failure_class is None:
            healthy_by_group[group_of[r.uid]].append(r.uid)
    groups = sorted({group_of[r.uid] for r in runs})
    rng.shuffle(groups)
    fit_groups, val_groups = [], []
    n_fit_pool = n_val_pool = 0
    for g in groups:
        if n_fit_pool < n_fit:
            fit_groups.append(g)
            n_fit_pool += len(healthy_by_group[g])
        elif n_val_pool < n_val:
            val_groups.append(g)
            n_val_pool += len(healthy_by_group[g])
    if n_fit_pool < n_fit or n_val_pool < n_val:
        return "whole task groups cannot fill fit/val"
    fit = rng.sample([u for g in fit_groups for u in healthy_by_group[g]], n_fit)
    val = rng.sample([u for g in val_groups for u in healthy_by_group[g]], n_val)
    held = set(fit_groups) | set(val_groups)
    test_runs = [r for r in runs if group_of[r.uid] not in held]
    if len({r.failure_class is None for r in test_runs}) < 2:
        return "test has one class"
    return fit, val, [r.uid for r in test_runs], {"test_task_groups": len(groups) - len(held)}


# ---------------------------------------------------------------- scoring


def fit_and_auroc(
    by_uid: dict[str, Run], fit: list[str], val: list[str], test: list[str], channels: tuple[str, ...]
) -> float:
    baseline = ESNBaseline(channels=channels)
    baseline.fit([by_uid[u] for u in fit], [by_uid[u] for u in val])
    y = np.array([by_uid[u].failure_class is not None for u in test], dtype=int)
    s = np.array([baseline.score(by_uid[u]).episode_score for u in test])
    return auroc(y, s)


def summarise_aurocs(values: list[float]) -> dict:
    arr = np.asarray(values)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "min": float(arr.min()),
            "max": float(arr.max()), "n_seeds": len(arr)}


def split_ablation_b(
    runs: list[Run], hash_of: dict, group_of: dict, n_seeds: int, max_seed: int = 500
) -> dict:
    by_uid = {r.uid: r for r in runs}
    fit0, val0, _, _ = random_split(runs, 0)
    n_fit, n_val = len(fit0), len(val0)
    makers = {
        "random": lambda s: random_split(runs, s),
        "twin_aware": lambda s: twin_aware_split(runs, hash_of, s, n_fit, n_val),
        "task_disjoint": lambda s: task_disjoint_split(runs, group_of, s, n_fit, n_val),
    }
    out = {"n_fit": n_fit, "n_val": n_val, "esn_seed": "author default (1300)"}
    for name, make in makers.items():
        per_seed, skipped, seed = [], [], 0
        while len(per_seed) < n_seeds and seed < max_seed:
            split = make(seed)
            if isinstance(split, str):
                skipped.append({"seed": seed, "reason": split})
            else:
                fit, val, test, extra = split
                y_test = [by_uid[u].failure_class is not None for u in test]
                per_seed.append({
                    "seed": seed,
                    "auroc": fit_and_auroc(by_uid, fit, val, test, AUTHOR_CHANNELS),
                    "n_test_healthy": y_test.count(False),
                    "n_test_failed": y_test.count(True),
                    **overlap_counts(test, fit, hash_of, group_of),
                    **extra,
                })
            seed += 1
        out[name] = {
            **summarise_aurocs([p["auroc"] for p in per_seed]),
            "skipped_seeds": skipped,
            "per_seed": per_seed,
        }
    return out


# ---------------------------------------------------------------- D-3


def source_pools(
    source_runs: list[Run], target_runs: list[Run], group_of: dict
) -> tuple[list[str], dict[str, list[str]]]:
    """The author's llama test uids, and the healthy source runs split by whether their task
    group appears among those test runs."""
    _, _, test = author_split(target_runs, 0)
    test_uids = [r.uid for r in test]
    test_groups = {group_of[u] for u in test_uids}
    healthy = [r for r in source_runs if r.failure_class is None]
    return test_uids, {
        "overlap": [r.uid for r in healthy if group_of[r.uid] in test_groups],
        "excluded": [r.uid for r in healthy if group_of[r.uid] not in test_groups],
    }


def source_overlap_a(
    source_runs: list[Run],
    target_runs: list[Run],
    group_of: dict,
    hash_of: dict,
    source_channels: tuple[str, ...],
    n_fit: int,
    n_val: int,
    n_seeds: int,
) -> dict:
    """Monitor A on the author's llama test set (seed 0), fitted on qwen healthy runs whose task
    group is inside vs outside that test set's task groups; same n_fit, n_seeds draws each."""
    test_uids, pools = source_pools(source_runs, target_runs, group_of)
    for name, pool in pools.items():
        if len(pool) < n_fit + n_val:
            raise ValueError(f"{name} pool has {len(pool)} runs, needs {n_fit + n_val}")
    by_uid = {r.uid: r for r in source_runs + target_runs}
    out = {"n_fit": n_fit, "n_val": n_val, "test": "author llama test set (seed 0)",
           "test_runs": len(test_uids), "pool_sizes": {k: len(v) for k, v in pools.items()}}
    for name, pool in pools.items():
        per_seed = []
        for seed in range(n_seeds):
            drawn = random.Random(seed).sample(pool, n_fit + n_val)
            fit, val = drawn[:n_fit], drawn[n_fit:]
            per_seed.append({
                "seed": seed,
                "auroc": fit_and_auroc(by_uid, fit, val, test_uids, source_channels),
                **overlap_counts(test_uids, fit, hash_of, group_of),
            })
        out[name] = {**summarise_aurocs([p["auroc"] for p in per_seed]), "per_seed": per_seed}
    return out
