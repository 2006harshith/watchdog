"""SP3 Step 2 acceptance check: reproduce the paper's section 5 numbers with the vendored monitor.

Rebuilds the AUTHOR's protocol (derail/experiments/run_hybrid_study.load_real and
run_model_transfer.py @ 1b3e07f), not watchdog's task-disjoint E1 protocol:
  - runs with T >= 4, in each corpus's manifest order;
  - healthy runs split 60/20/20 (fit / threshold / test) with rng_for(0, "real-split");
  - test = the target's held-out healthy runs + all its failed runs;
  - in-domain: fit on llama's own splits; transfer: fit on qwen's splits, no refit.
Pass: esn_cusum_max AUROC within TOLERANCE of 0.8847 (in-domain) and 0.5275 (transfer).
If it passes, our parquet -> features -> monitor path is the author's.

Run: uv run python scripts/reproduce_author_transfer.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from watchdog_agent.baselines.esn import ESNBaseline, Run
from watchdog_agent.baselines.esn._vendor.common import rng_for
from watchdog_agent.data import REPO_ID, load_episodes
from watchdog_agent.metrics import auprc, auroc

SOURCE, TARGET = "ollama7b", "ollama_llama8b"
MIN_T = 4
TOLERANCE = 0.01
# results/tables/model_transfer_family.csv @ 1b3e07f, monitor esn_cusum_max
AUTHOR = {
    "in_domain": {"auroc": 0.8847, "auprc": 0.9552, "detection_rate": 0.7672, "healthy_fa_rate": 0.125},
    "transfer": {"auroc": 0.5275, "auprc": 0.8702, "detection_rate": 0.9052, "healthy_fa_rate": 0.75},
}
OUT = Path("results/e1_author_repro.json")


def load_corpus(episodes: pd.DataFrame, corpus: str) -> tuple[list[Run], dict[str, int], tuple[str, ...]]:
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


def author_split(runs: list[Run]) -> tuple[list[Run], list[Run], list[Run]]:
    healthy = [r for r in runs if r.failure_class is None]
    failed = [r for r in runs if r.failure_class is not None]
    perm = rng_for(0, "real-split").permutation(len(healthy))
    # round() as in the author's code: Python rounds half to even.
    n_fit = round(0.6 * len(healthy))
    n_val = round(0.2 * len(healthy))
    fit = [healthy[i] for i in perm[:n_fit]]
    val = [healthy[i] for i in perm[n_fit : n_fit + n_val]]
    test = [healthy[i] for i in perm[n_fit + n_val :]] + failed
    return fit, val, test


def evaluate(baseline: ESNBaseline, test: list[Run], taus: dict[str, int]) -> dict:
    scores = [baseline.score(r) for r in test]
    y = np.array([0 if r.failure_class is None else 1 for r in test])
    s = np.array([sc.episode_score for sc in scores])
    healthy_alarms = [sc.alarm_step is not None for r, sc in zip(test, scores) if r.failure_class is None]
    # A detection is an alarm at or after onset; an alarm before tau is an early (false) alarm.
    detections = [
        sc.alarm_step is not None and sc.alarm_step >= taus[r.uid]
        for r, sc in zip(test, scores)
        if r.failure_class is not None
    ]
    return {
        "auroc": auroc(y, s),
        "auprc": auprc(y, s),
        "detection_rate": float(np.mean(detections)),
        "healthy_fa_rate": float(np.mean(healthy_alarms)),
        "theta": baseline.theta,
        "n_test_healthy": int((y == 0).sum()),
        "n_test_failed": int((y == 1).sum()),
    }


def main() -> int:
    load_dotenv()
    episodes = load_episodes()
    tgt_runs, tgt_taus, tgt_channels = load_corpus(episodes, TARGET)
    src_runs, _, src_channels = load_corpus(episodes, SOURCE)
    tgt_fit, tgt_val, test = author_split(tgt_runs)
    src_fit, src_val, _ = author_split(src_runs)

    arms = {
        "in_domain": (tgt_fit, tgt_val, tgt_channels),
        "transfer": (src_fit, src_val, src_channels),
    }
    results = {}
    for arm, (fit, val, channels) in arms.items():
        baseline = ESNBaseline(channels=channels)
        baseline.fit(fit, val)
        results[arm] = {
            **evaluate(baseline, test, tgt_taus),
            "channels": list(channels),
            "n_fit": len(fit),
            "n_val": len(val),
        }

    deltas = {arm: round(results[arm]["auroc"] - AUTHOR[arm]["auroc"], 4) for arm in AUTHOR}
    passed = all(abs(d) <= TOLERANCE for d in deltas.values())
    OUT.write_text(
        json.dumps(
            {
                "what": "author-protocol reproduction of arXiv 2608.02464 section 5 (esn_cusum_max)",
                "author_commit": "1b3e07fee53ae13407173c3ea932adb4a43e8230",
                "source": SOURCE,
                "target": TARGET,
                "tolerance_auroc": TOLERANCE,
                "author": AUTHOR,
                "ours": results,
                "auroc_delta_ours_minus_author": deltas,
                "passed": passed,
            },
            indent=2,
        )
        + "\n"
    )
    for arm, ref in AUTHOR.items():
        ours = results[arm]
        print(
            f"{arm:>9}: AUROC {ours['auroc']:.4f} (author {ref['auroc']:.4f}) | "
            f"AUPRC {ours['auprc']:.4f} ({ref['auprc']:.4f}) | "
            f"det {ours['detection_rate']:.4f} ({ref['detection_rate']:.4f}) | "
            f"FA {ours['healthy_fa_rate']:.4f} ({ref['healthy_fa_rate']:.4f}) | "
            f"test {ours['n_test_healthy']}h/{ours['n_test_failed']}f"
        )
    print(f"{'PASS' if passed else 'FAIL'} (tolerance {TOLERANCE}) -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
