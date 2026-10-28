"""SP8 Step 2: fit MS-raw-noLat on source -> results/e5_nolat_fingerprint.json. No SP8 run is read.

Run from a clean commit: uv run python scripts/fit_e5_nolat.py
The fingerprint is the sha256 of the model's scores on the llama E2 prefix rows (not SP8 data); e5.py
refits the same deterministic model and must reproduce it. MS-raw is fitted alongside as the check that
this path is E2's: its llama scores must equal results/e2_prefix_scores.csv.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e5_models import fingerprint, fit_ms_models, ms_nolat_columns
from watchdog_agent.features import prefix_feature_table


def main() -> int:
    e3_cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(e3_cfg["arms"]["hgb"]["config"]).read_text())
    episodes = load_episodes()  # excludes organic* (D1): no SP8 run is read here
    needed = sorted(set(e2_cfg["checkpoints"]["train"]) | set(e3_cfg["checkpoints"]["scored"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    models = fit_ms_models(table, episodes, e2_cfg, HistGradientBoostingClassifier)
    llama = episodes[episodes["family"] == "llama"]
    rows = table[table.index.get_level_values("uid").isin(set(llama["uid"]))
                 & table.index.get_level_values("t").isin(e3_cfg["checkpoints"]["scored"])]
    ms = models["MS-raw"].score(rows)
    ref = pd.read_csv(e3_cfg["arms"]["hgb"]["reproduce_from"])
    ref = ref[ref["family"] == "llama"].set_index(["uid", "t"])["MS-raw"]
    diff = float(np.nanmax(np.abs(ms.to_numpy() - ref.loc[ms.index].to_numpy())))
    nolat = models["MS-raw-noLat"].score(rows)
    out = {"what": "SP8 MS-raw-noLat fitted on source before any SP8 score (O-3 control)",
           "git": git_state(), "features": ms_nolat_columns(e2_cfg), "n_features": len(ms_nolat_columns(e2_cfg)),
           "hgb_params": e2_cfg["model"]["primary"], "fingerprint_rows": "llama E2 prefix rows at t in {2, 3, 4}",
           "n_rows": len(rows), "MS-raw-noLat_sha256": fingerprint(nolat), "MS-raw_sha256": fingerprint(ms),
           "MS-raw_vs_e2_prefix_scores_max_abs_diff": diff}
    Path("results/e5_nolat_fingerprint.json").write_text(json.dumps(out, indent=2))
    print(f"MS-raw vs E2 prefix scores: max diff {diff:.2e}; noLat sha256 {out['MS-raw-noLat_sha256'][:16]}")
    return 0 if diff <= 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
