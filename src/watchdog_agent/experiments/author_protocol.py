"""The author's evaluation protocol for the paper's section 5 transfer table.

Mirrors derail/experiments/run_hybrid_study.load_real and run_model_transfer.py @ 1b3e07f:
runs with T >= 4 in each corpus's manifest order; healthy runs split 60/20/20 into fit /
threshold / test with rng_for(seed, "real-split") (the author uses seed 0); test = held-out
healthy runs + all failed runs. Not task-disjoint: that is what SP3's diagnostics measure.
"""

import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from watchdog_agent.baselines.esn import Run
from watchdog_agent.baselines.esn._vendor.common import rng_for
from watchdog_agent.data import REPO_ID

MIN_T = 4


def load_corpus(episodes: pd.DataFrame, corpus: str) -> tuple[list[Run], dict[str, int], tuple[str, ...]]:
    """Runs in manifest order (the order the author's split permutes), onset step per uid, and
    the channel set the author's rule picks for this corpus."""
    manifest_path = hf_hub_download(REPO_ID, repo_type="dataset", filename=f"traces/{corpus}/manifest.json")
    manifest = json.loads(Path(manifest_path).read_text("utf-8"))
    rows = episodes[episodes["corpus"] == corpus].set_index("episode_id")
    runs, taus = [], {}
    for entry in manifest:
        if entry["T"] < MIN_T:
            continue
        row = rows.loc[entry["episode_id"]]
        failure_class = None if entry["tau"] is None else entry["failure_class"]
        runs.append(Run(uid=row["uid"], steps=row["steps_parsed"], failure_class=failure_class))
        taus[row["uid"]] = entry["tau"]
    # Author's channel rule (load_real): drop surprisal if < 90% of the corpus has logprobs.
    n_logprobs = sum(bool(e.get("has_logprobs")) for e in manifest)
    base = ("e", "u", "m") if n_logprobs >= 0.9 * len(manifest) else ("e", "m")
    return runs, taus, base + ("x",)


def author_split(runs: list[Run], seed: int = 0) -> tuple[list[Run], list[Run], list[Run]]:
    healthy = [r for r in runs if r.failure_class is None]
    failed = [r for r in runs if r.failure_class is not None]
    perm = rng_for(seed, "real-split").permutation(len(healthy))
    # round() as in the author's code: Python rounds half to even.
    n_fit = round(0.6 * len(healthy))
    n_val = round(0.2 * len(healthy))
    fit = [healthy[i] for i in perm[:n_fit]]
    val = [healthy[i] for i in perm[n_fit : n_fit + n_val]]
    test = [healthy[i] for i in perm[n_fit + n_val :]] + failed
    return fit, val, test
