"""SP6 Step 0: calibration-pool counts per family, checkpoint and target fold -> results/e3_step0_counts.json.

Run: uv run python scripts/run_e3_step0.py
Says where n = 40 is infeasible, and checks the ESN-src pool sizes configs/e3.yaml assumes.
"""

import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2_diagnostics import e2_healthy_pool
from watchdog_agent.experiments.e3 import calibration_pool_counts


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(cfg["arms"]["hgb"]["config"]).read_text())
    episodes = load_episodes()
    folds, checkpoints, budgets = cfg["cross_fit"]["folds"], cfg["checkpoints"]["scored"], cfg["budgets"]["n"]
    result = {"what": "SP6 Step 0: healthy test runs and eligible calibration runs per target fold",
              "git": git_state(), "families": {}}
    for family in [cfg["families"]["primary"], *cfg["families"]["descriptive"]]:
        counts = calibration_pool_counts(episodes, family, checkpoints, folds["k"], folds["seed"], budgets)
        counts["esn_src_pool_runs"] = len(e2_healthy_pool(episodes, family, e2_cfg, "i"))
        counts["esn_src_pool_runs_in_config"] = cfg["arms"]["ESN-src"]["fit_draws"][family]["pool_runs"]
        result["families"][family] = counts
        print(f"\n{family}: ESN-src pool {counts['esn_src_pool_runs']} (config {counts['esn_src_pool_runs_in_config']})")
        for t, block in counts["per_t"].items():
            print(f"  t={t}: healthy with step {block['healthy_with_step']}; folds skipped {block['folds_skipped']}")
            for f in block["per_fold"]:
                print(f"    fold {f['fold']}: runs {f['runs']:3d}  test healthy {f['test_healthy']:3d}  eligible {f['eligible']:3d}")
    path = Path("results/e3_step0_counts.json")
    path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
