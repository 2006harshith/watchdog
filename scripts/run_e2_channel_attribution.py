"""SP5 close-out: ESN channel attribution -> results/e2_channel_attribution.json (interpretation only).

Run from a clean commit: uv run python scripts/run_e2_channel_attribution.py
Needs results/e2_ablation.json and results/e2_diagnostics.json: the 'before' ESNs must reproduce them.
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
from watchdog_agent.experiments.e2_diagnostics import (
    check_attribution_reproduction,
    run_channel_attribution,
)

RESULTS_DIR = Path("results")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _fmt(iv: dict | None) -> str:
    if iv is None:
        return "  n/a"
    return f"{iv['point']:.3f} [{iv['lo']:.3f}, {iv['hi']:.3f}]" if "lo" in iv else f"{iv['point']:.3f}"


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e2.yaml").read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    result = run_channel_attribution(load_episodes(), cfg, git=git)
    print(f"channel attribution done in {time.time() - start:.0f}s")

    e2 = json.loads((RESULTS_DIR / "e2_ablation.json").read_text())
    diag = json.loads((RESULTS_DIR / "e2_diagnostics.json").read_text())
    pairs = check_attribution_reproduction(result, e2, diag)
    result["reproduction"] = {k: {"attribution": a, "published": b} for k, (a, b) in pairs.items()}
    path = RESULTS_DIR / "e2_channel_attribution.json"
    path.write_text(json.dumps(result, indent=2, default=_json_default))

    for key, block in result["esn"].items():
        h = block["auroc"]["headline"]
        print(f"\n{key} {block['channels']}: headline {_fmt(h['before'])} -> blind {_fmt(h['after'])}")
        for when, by_t in block["attribution"].items():
            for t, rows in by_t.items():
                print(f"  {when:11s} t={t}: " + "; ".join(f"{cls} {rows[cls]}" for cls in sorted(rows)))
    print("\nm inputs on llama, single-input AUROC (mean over primary checkpoints):")
    for name, views in result["m_inputs"]["contrast"].items():
        for view, by_cls in views.items():
            print(f"  {name:20s} {view:9s} " + "  ".join(f"{c} {_fmt(v)}" for c, v in by_cls.items()))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
