"""SP6 close-out diagnostics D-8..D-12 (POST HOC) -> results/e3_diagnostics.json.

Run from a clean commit: uv run python scripts/run_e3_diagnostics.py
Needs results/e3_group_counts.parquet from `uv run python scripts/run_e3.py` (gitignored): this script
only re-weights E3's counts, it fits and scores nothing. Design and expected outcomes:
docs/decisions.md, "2026-10-27 — SP6 close-out diagnostics", committed before the first run.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from run_e3 import _clean, _json_default

from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e3_diagnostics import run_diagnostics


def main() -> int:
    cfg = yaml.safe_load(Path("configs/e3_diagnostics.yaml").read_text())
    e3_cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")
    groups = pd.read_parquet(cfg["inputs"]["groups"])
    draws = pd.read_csv(cfg["inputs"]["draws"])
    e3 = json.loads(Path(cfg["inputs"]["e3"]).read_text())
    out = run_diagnostics(groups, draws, e3, cfg, e3_cfg) | {"git": git, "e3_git": e3["git"]}

    path = Path("results/e3_diagnostics.json")
    path.write_text(json.dumps(_clean(out), indent=2, default=_json_default))
    ok = True
    for family, fam in out["families"].items():
        r = fam["reproduction"]
        ok &= r["ok"] and fam["tpr_post_le_tpr_violations"] == 0
        print(f"{family}: E3 headline from group counts max diff {r['max_abs_diff_vs_e3_headline']:.3g} over "
              f"{r['cells']} cells; TPR_post > TPR in {fam['tpr_post_le_tpr_violations']} draws")
    for name, block in out["expectations"].items():
        parts = {k: v["verdict"] for k, v in block.items() if isinstance(v, dict) and "verdict" in v}
        print(f"{name}: {block.get('verdict', parts)}")
    print(f"wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
