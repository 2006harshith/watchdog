"""SP5 close-out: D-6 and D-7 diagnostics -> results/e2_closeout.json (interpretation only; no MLflow run).

Run from a clean commit: uv run python scripts/run_e2_closeout.py
Needs results/e2_ablation.json: every re-run of an E2 number must reproduce it.
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
from watchdog_agent.experiments.e2_diagnostics import check_closeout_reproduction, run_e2_closeout

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
    return f"{iv['point']:.3f} [{iv['lo']:.3f}, {iv['hi']:.3f}]" if "lo" in iv else f"{iv['point']:.3f}"


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e2.yaml").read_text())
    cfg["closeout"] = yaml.safe_load(Path("configs/e2_closeout.yaml").read_text())
    e1_cfg = yaml.safe_load(Path("configs/e1.yaml").read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    result = run_e2_closeout(load_episodes(), cfg, e1_cfg, git=git)
    print(f"close-out done in {time.time() - start:.0f}s")

    e2 = json.loads((RESULTS_DIR / "e2_ablation.json").read_text())
    pairs = check_closeout_reproduction(result, e2)
    result["reproduction"] = {k: {"closeout": a, "e2": b} for k, (a, b) in pairs.items()}
    path = RESULTS_DIR / "e2_closeout.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default))

    d6 = result["d6"]
    s, a, b = d6["subsets"], d6["a_fit_pool_only"], d6["b_size_matched"]["per_fold_mean"]
    print(f"D-6 subsets ({s['fit_size']} of {s['pool_runs']}, n={len(s['aurocs'])}): median {s['median']:.3f} "
          f"[min {s['min']:.3f}, max {s['max']:.3f}]; E2 split {s['e2_split_auroc']:.3f}, "
          f"share of draws >= it {s['share_of_draws_at_or_above_e2_split']:.2f}")
    print(f"D-6 (a) SP4 structure + E2 pool: standardised {a['sp4_structure_e2_pool_standardised']:.3f}, "
          f"raw {a['sp4_structure_e2_pool_raw']:.3f}, per-fold {a['per_fold_headline']['mean']:.3f}")
    print(f"D-6 (b') SP4 fold pools at fit size {d6['b_size_matched']['fit_size']}: per-fold mean median "
          f"{b['median']:.3f} [min {b['min']:.3f}, max {b['max']:.3f}]")
    print("\nD-7a evidence:", json.dumps(result["d7"]["a_evidence"], indent=1))
    for key, block in result["d7"]["b_masking"].items():
        h = block["headline"]
        print(f"\nD-7b {key}: headline {_fmt(h['before'])} -> {_fmt(h['after'])}, diff {_fmt(h['after_minus_before'])}")
        for cls, v in block["per_class"].items():
            print(f"  {cls:20s} {_fmt(v['before'])} -> {_fmt(v['after'])}   diff {_fmt(v['after_minus_before'])}")
    blind = result["d7"]["c_result_blind"]
    print("\nD-7c blinded:", json.dumps(blind["counts"]))
    for key, block in blind.items():
        if key == "counts":
            continue
        h = block["headline"]
        print(f"\nD-7c {key}: headline {_fmt(h['before'])} -> {_fmt(h['after'])}, diff {_fmt(h['after_minus_before'])}")
        for cls, v in block["per_class"].items():
            print(f"  {cls:20s} {_fmt(v['before'])} -> {_fmt(v['after'])}   diff {_fmt(v['after_minus_before'])}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
