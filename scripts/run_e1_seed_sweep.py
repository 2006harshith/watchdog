"""SP3 D-5: paired ESN-seed sweep of E1 + seed-1300 dominant-task / length checks.

Writes results/e1_seed_sweep.json twice: once before the loop (config, git, pre-registered rule,
status "running"), and again with the per-seed rows, summary and outcome.

Run: uv run python scripts/run_e1_seed_sweep.py [--config configs/e1_seed_sweep.yaml]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import (
    decide_seed_sweep,
    dominant_task_and_length_checks,
    git_state,
    seed_sweep_rows,
)

OUT = Path("results/e1_seed_sweep.json")
E1_RESULTS = Path("results/e1_collapse.json")
E1_SCORES = Path("results/e1_episode_scores.csv")


def write(result: dict) -> None:
    OUT.write_text(json.dumps(result, indent=2) + "\n")


def matches_e1(row: dict) -> dict:
    """The seed-1300 row must reproduce the committed E1 result exactly."""
    e1 = json.loads(E1_RESULTS.read_text())
    cluster = e1["sensitivity"]["task_group_cluster_bootstrap"]["paired_auroc_B_minus_A"]
    return {
        "auroc_A": row["auroc_A"] == e1["monitors"]["A"]["auroc"]["point"],
        "auroc_B": row["auroc_B"] == e1["monitors"]["B"]["auroc"]["point"],
        "cluster_ci_B_minus_A": row["cluster_ci_B_minus_A"] == cluster,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1_seed_sweep.yaml")
    args = parser.parse_args()
    load_dotenv()
    sweep_cfg = yaml.safe_load(Path(args.config).read_text())
    base_cfg = yaml.safe_load(Path(sweep_cfg["base_config"]).read_text())
    result = {
        "what": "SP3 D-5: E1 repeated over ESN seeds (paired: same folds, splits and runs)",
        "git": git_state(),
        "config": {"sweep": sweep_cfg, "base": base_cfg},
        "rule": sweep_cfg["rule"],
        "status": "running",
    }
    write(result)  # rule on disk before any seed is scored

    episodes = load_episodes()

    def progress(row: dict) -> None:
        ci = row["cluster_ci_B_minus_A"]
        print(f"seed {row['esn_seed']}: A {row['auroc_A']:.4f} B {row['auroc_B']:.4f} "
              f"B-A {row['gap_B_minus_A']:+.4f} cluster CI [{ci['lo']:.4f}, {ci['hi']:.4f}]", flush=True)

    with warnings.catch_warnings():
        # pick_threshold's <19-validation-runs warning; the sweep reports AUROC only.
        warnings.simplefilter("ignore", RuntimeWarning)
        rows = seed_sweep_rows(
            episodes, base_cfg, sweep_cfg["esn_seeds"],
            sweep_cfg["bootstrap"]["n"], sweep_cfg["bootstrap"]["seed"], on_row=progress,
        )
    summary = decide_seed_sweep(rows, sweep_cfg["rule"])
    row_1300 = next(r for r in rows if r["esn_seed"] == base_cfg["esn"]["seed"])
    length_of = dict(zip(episodes["uid"], episodes["T"].astype(int)))
    result |= {
        "status": "done",
        "rows": rows,
        "summary": summary,
        "outcome": summary["outcome"],
        "seed_1300_matches_e1_collapse": matches_e1(row_1300),
        "seed_1300_checks": dominant_task_and_length_checks(pd.read_csv(E1_SCORES), length_of),
    }
    write(result)
    print(json.dumps({k: result[k] for k in ("summary", "seed_1300_matches_e1_collapse",
                                              "seed_1300_checks")}, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
