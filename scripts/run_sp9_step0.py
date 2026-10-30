"""SP9 Step 0 runner -> results/sp9_step0.json. POST HOC; design pre-registered in configs/sp9_step0.yaml (11589d3).

Run from a clean commit: uv run python scripts/run_sp9_step0.py
Order: L-3 (read-only), then L-1 after MS-raw reproduces E2's prefix scores and MS-raw-noLat its
fingerprint, then L-2, where each fit draw's as-is AUROC must equal D-6's before its constant-latency
model is fitted. A failed check stops the run and writes only the checks.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from run_e3 import _clean, _json_default
from sklearn.ensemble import HistGradientBoostingClassifier

from watchdog_agent.data import load_episodes
from watchdog_agent.evaluation import matched_step_cluster_ci
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2 import default_esn_factory, paired, to_score_dict
from watchdog_agent.experiments.e2_diagnostics import _headline_mean, restrict_classes, run_getter
from watchdog_agent.experiments.e5_models import fingerprint, fit_ms_models
from watchdog_agent.experiments.sp9_step0 import (
    ReproductionFailed,
    drop_summary,
    esn_constant_latency_draws,
    latency_anatomy,
    verdict,
)
from watchdog_agent.features import prefix_feature_table

OUT = Path("results/sp9_step0.json")
TOL = 1e-9
MAX_DROP = 0.10


def main() -> int:
    cfg = yaml.safe_load(Path("configs/sp9_step0.yaml").read_text(encoding="utf-8"))
    e3_cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(e3_cfg["arms"]["hgb"]["config"]).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")
    fp = json.loads(Path("results/e5_nolat_fingerprint.json").read_text())
    co = json.loads(Path("results/e2_closeout.json").read_text())
    out = {"what": "SP9 Step 0: latency diagnostics (L-1, L-2, L-3); POST HOC", "config": cfg, "git": git}

    def stop(checks: dict) -> int:
        print(f"REPRODUCTION FAILED, stopped: {checks}")
        OUT.write_text(json.dumps(_clean(out | {"stopped": "reproduction", "checks": checks}), indent=2,
                                  default=_json_default))
        return 1

    episodes = load_episodes()
    llama = episodes[episodes["family"] == "llama"]

    # L-3, read-only.
    out["L-3"] = latency_anatomy(llama)
    print(f"L-3: {out['L-3']['share_fast']:.3f} of llama steps < 50 ms")

    # L-1: reproduction, then MS-raw-noLat on E2's llama setting.
    needed = sorted(set(e2_cfg["checkpoints"]["train"]) | set(e3_cfg["checkpoints"]["scored"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    models = fit_ms_models(table, episodes, e2_cfg, HistGradientBoostingClassifier)
    rows = table[table.index.get_level_values("uid").isin(set(llama["uid"]))
                 & table.index.get_level_values("t").isin(e3_cfg["checkpoints"]["scored"])]
    ref = pd.read_csv(e3_cfg["arms"]["hgb"]["reproduce_from"])
    ref = ref[ref["family"] == "llama"].set_index(["uid", "t"])
    ms, nolat = models["MS-raw"].score(rows), models["MS-raw-noLat"].score(rows)
    diff = float(np.nanmax(np.abs(ms.to_numpy() - ref.loc[ms.index, "MS-raw"].to_numpy())))
    nolat_fp = fingerprint(nolat)
    checks = {"MS-raw.llama_vs_e2_prefix_scores": {"value": diff, "ok": diff <= TOL},
              "MS-raw-noLat.fingerprint": {"value": nolat_fp, "ok": nolat_fp == fp["MS-raw-noLat_sha256"]}}
    if not all(c["ok"] for c in checks.values()):
        return stop(checks)
    head = e2_cfg["classes"]["headline"]
    s_ms, s_nolat = to_score_dict(ms, llama), to_score_dict(nolat, llama)
    a_ms, a_nolat = _headline_mean(s_ms, llama, e2_cfg), _headline_mean(s_nolat, llama, e2_cfg)
    n, seed = e2_cfg["bootstrap"]["n_primary"], e2_cfg["bootstrap"]["seed"]
    ci = matched_step_cluster_ci(s_nolat, restrict_classes(llama, head), e2_cfg["checkpoints"]["primary"], n,
                                 seed=seed).mean
    out["L-1"] = {"MS-raw": a_ms, "MS-raw-noLat": a_nolat, "MS-raw-noLat_ci": {"lo": ci.lo, "hi": ci.hi},
                  "paired_noLat_minus_MS-raw": paired(s_ms, s_nolat, llama, head, e2_cfg, n)["mean"],
                  "expectation": verdict(a_ms, a_nolat, MAX_DROP)}
    print(f"L-1: MS-raw {a_ms:.3f}, noLat {a_nolat:.3f} [{ci.lo:.3f}, {ci.hi:.3f}] -> {out['L-1']['expectation']['outcome']}")

    # L-2: ESN-src's 30 fit draws, as is (checked against D-6 per draw) and with constant step latency.
    try:
        r = esn_constant_latency_draws(episodes, llama, "llama", e2_cfg, e3_cfg, default_esn_factory(e2_cfg),
                                       run_getter(episodes), expected_as_is=co["d6"]["subsets"]["aurocs"], tol=TOL)
    except ReproductionFailed as e:
        return stop(checks | {"ESN-src.fit_draw_aurocs_vs_d6": {"value": str(e), "ok": False}})
    checks["ESN-src.fit_draw_aurocs_vs_d6"] = {"value": float(np.max(np.abs(
        np.array(r["as_is"]) - np.array(co["d6"]["subsets"]["aurocs"])))), "ok": True}
    ref_med = cfg["L-2"]["reference"]["median"]
    summ = drop_summary(r["as_is"], r["constant"], MAX_DROP)
    out["L-2"] = {"channels": r["channels"], "constants_s": r["constants"], "as_is": r["as_is"],
                  "constant": r["constant"], "summary": summ,
                  "expectation": verdict(ref_med, summ["constant_median"], MAX_DROP),
                  "caveat": "spread = fit-set variability only; test-set sampling is not in it"}
    out["checks"] = checks
    OUT.write_text(json.dumps(_clean(out), indent=2, default=_json_default))
    print(f"L-2: ESN-src median {summ['as_is_median']:.3f} -> {summ['constant_median']:.3f} "
          f"(paired drop median {summ['drop_median']:+.3f}) -> {out['L-2']['expectation']['outcome']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
