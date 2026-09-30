"""SP5 Step 3b: E2 diagnostics -> results/e2_diagnostics.json (interpretation only; no MLflow run).

Run from a clean commit: uv run python scripts/run_e2_diagnostics.py
Needs results/e2_ablation.json and results/sp4_feature_audit.json: three rows must reproduce them.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2_diagnostics import check_reproduction, run_e2_diagnostics

RESULTS_DIR = Path("results")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _fmt(iv: dict) -> str:
    if "lo" in iv:
        return f"{iv['point']:.3f} [{iv['lo']:.3f}, {iv['hi']:.3f}]"
    return f"{iv['point']:.3f}"


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e2.yaml").read_text())
    e1_cfg = yaml.safe_load(Path("configs/e1.yaml").read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    result = run_e2_diagnostics(load_episodes(), cfg, e1_cfg, git=git)
    print(f"diagnostics done in {time.time() - start:.0f}s")

    e2 = json.loads((RESULTS_DIR / "e2_ablation.json").read_text())
    sp4 = json.loads((RESULTS_DIR / "sp4_feature_audit.json").read_text())
    result["reproduction"] = {k: {"diagnostic": a, "published": b} for k, (a, b) in check_reproduction(result, e2, sp4).items()}

    path = RESULTS_DIR / "e2_diagnostics.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default))

    print(f"{'row':28s} {'headline (pooled)':24s} {'all classes (pooled)':24s} per-fold head / all")
    for name, row in result["rows"].items():
        pf_h, pf_a = row["per_fold_headline"], row["per_fold_all_classes"]
        print(f"{name:28s} {_fmt(row['headline']):24s} {_fmt(row['all_classes']):24s} "
              f"{pf_h['mean']:.3f} ({pf_h['n_folds_used']}) / {pf_a['mean']:.3f} ({pf_a['n_folds_used']})")
    print("\nper class (pooled):")
    for name, row in result["rows"].items():
        print(f"  {name:28s} " + "  ".join(f"{c} {v:.3f}" for c, v in row["per_class"].items()))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
