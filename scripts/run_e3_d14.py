"""SP6 D-14 (POST HOC) -> results/e3_d14.json, results/e3_d14_draws.csv,
results/e3_d14_group_counts.parquet (gitignored; regenerable).

Run from a clean commit: uv run python scripts/run_e3_d14.py
Needs results/e3_group_counts.parquet and results/e3_draws.csv (E3's by-run side). Design and
expected outcome: docs/decisions.md, "D-14 Half split by task group", committed before the run.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from run_e3 import _clean, _fmt, _json_default

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e3 import summarise
from watchdog_agent.experiments.e3_d14 import CAVEAT, code_path_check, compare, run_d14
from watchdog_agent.features import prefix_feature_table


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e3_d14.yaml").read_text())
    e3_cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(e3_cfg["arms"]["hgb"]["config"]).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")
    e3_groups = pd.read_parquet(cfg["inputs"]["e3_groups"])
    e3_draws = pd.read_csv(cfg["inputs"]["e3_draws"])
    e3_git = json.loads(Path("results/e3_recalibration.json").read_text())["git"]

    start = time.time()
    episodes = load_episodes()
    needed = sorted(set(e2_cfg["checkpoints"]["train"]) | set(e3_cfg["checkpoints"]["scored"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    c = cfg["code_path_check"]
    _, check_groups, _ = run_d14(table, episodes, cfg, e3_cfg, e2_cfg, split="run", only=(c["n"], c["t"], c["fold"]))
    check = code_path_check(check_groups, e3_groups, cfg)
    print(f"code path check (E3 draws, n={c['n']} t={c['t']} fold {c['fold']}): {check}")
    if not check["ok"]:
        print("code path check FAILED: D-14's loop does not reproduce E3; not running the group split")
        return 1
    draws, groups, extra = run_d14(table, episodes, cfg, e3_cfg, e2_cfg, split=cfg["split"]["by"])
    print(f"D-14 done in {time.time() - start:.0f}s")

    groups.to_parquet("results/e3_d14_group_counts.parquet", index=False)
    draws.to_csv("results/e3_d14_draws.csv", index=False)
    comparison = compare(e3_groups, e3_draws, groups, draws, extra, cfg, e3_cfg)
    placement = (pd.DataFrame(extra["placement"]).groupby(["n", "t", "fold", "big_in"]).size()
                 .rename("draws").reset_index().to_dict("records"))
    out = {"what": "SP6 D-14: TL-pct, ALL-pct, ESN-refit with halves from disjoint task groups vs E3's by-run split",
           "post_hoc": True, "caveat": CAVEAT, "config": cfg, "git": git, "e3_git": e3_git,
           "code_path_check": check, "splits": extra["splits"], "big_groups": extra["big_groups"],
           "big_group_placement": placement, "comparison": comparison,
           "cells": summarise(draws, e3_cfg)}
    path = Path("results/e3_d14.json")
    path.write_text(json.dumps(_clean(out), indent=2, default=_json_default))

    skipped = [(s["n"], s["t"], s["fold"]) for s in extra["splits"] if not s["feasible"]]
    print(f"skipped (n, t, fold): {skipped}")
    for arm, by_n in comparison["cells"].items():
        for n, v in by_n.items():
            print(f"{arm:9} n={n}: FA run {_fmt(v['fa_run_mean'])} group {_fmt(v['fa_group_mean'])} "
                  f"(alpha_eff {_fmt(v['alpha_eff'])}) r {_fmt(v['r'])} [{_fmt(v['r_ci']['lo'])}, {_fmt(v['r_ci']['hi'])}] "
                  f"-> {v['reading']}")
    print(f"D-14 expectation: {comparison['expectation']['verdict']}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
