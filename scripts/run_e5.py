"""SP8 E5 runner -> results/e5_organic.json. GATE. Pre-registered config: configs/e5.yaml (51c7723).

Run from a clean commit: uv run python scripts/run_e5.py
Order: every reproduction check first (HGB arms vs E2's prefix scores, MS-raw-noLat vs its fingerprint,
ESN-src vs D-6); a mismatch stops the run before any SP8 statistic. Then the organic scores, the gate,
O-1, O-3, and the latency shift table.
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
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2 import build_training_set, default_esn_factory
from watchdog_agent.experiments.e2_diagnostics import run_getter
from watchdog_agent.experiments.e3 import _esn_src
from watchdog_agent.experiments.e5 import (
    esn_block,
    esn_reproduction,
    evaluate,
    fit_tl_models,
    gate_verdict,
    latency_table,
    organic_scores,
)
from watchdog_agent.experiments.e5_models import fingerprint, fit_ms_models
from watchdog_agent.features import prefix_feature_table
from watchdog_agent.organic import load_organic
from watchdog_agent.recalibration import cross_fitted_target_transform

OUT = Path("results/e5_organic.json")
TOL = 1e-9


def main() -> int:
    cfg = yaml.safe_load(Path("configs/e5.yaml").read_text(encoding="utf-8"))
    e3_cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(e3_cfg["arms"]["hgb"]["config"]).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")
    fp = json.loads(Path("results/e5_nolat_fingerprint.json").read_text())
    co = json.loads(Path("results/e2_closeout.json").read_text())
    scored = cfg["checkpoints"]["scored"]
    B, seed = cfg["gate"]["B"], cfg["gate"]["seed"]

    episodes = load_episodes()
    organic = load_organic()
    needed = sorted(set(e2_cfg["checkpoints"]["train"]) | set(e3_cfg["checkpoints"]["scored"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    org_table = prefix_feature_table(dict(zip(organic["uid"], organic["steps_parsed"])), scored)

    # 1. Frozen models and every reproduction check.
    train = build_training_set(table, episodes, "llama", e2_cfg)
    models = fit_ms_models(table, episodes, e2_cfg, HistGradientBoostingClassifier) | fit_tl_models(
        train, e2_cfg, HistGradientBoostingClassifier)
    llama = episodes[episodes["family"] == "llama"]
    rows_l = table[table.index.get_level_values("uid").isin(set(llama["uid"]))
                   & table.index.get_level_values("t").isin(scored)]
    ref = pd.read_csv(e3_cfg["arms"]["hgb"]["reproduce_from"])
    ref = ref[ref["family"] == "llama"].set_index(["uid", "t"])
    k, s = e2_cfg["recalibration"]["target_cross_fit"]["k"], e2_cfg["recalibration"]["target_cross_fit"]["seed"]
    checks = {}
    for arm in ("MS-raw", "TL-raw", "TL-pct"):
        X = cross_fitted_target_transform(rows_l, llama, k=k, seed=s) if arm.endswith("-pct") else rows_l
        v = models[arm].score(X)
        diff = float(np.nanmax(np.abs(v.to_numpy() - ref.loc[v.index, arm].to_numpy())))
        checks[f"{arm}.llama_vs_e2_prefix_scores"] = {"value": diff, "ok": diff <= TOL}
    nolat_fp = fingerprint(models["MS-raw-noLat"].score(rows_l))
    checks["MS-raw-noLat.fingerprint"] = {"value": nolat_fp, "ok": nolat_fp == fp["MS-raw-noLat_sha256"]}
    both = pd.concat([llama, organic], ignore_index=True)
    frame = pd.concat([episodes, organic], ignore_index=True)
    draws, info = _esn_src(episodes, both, "llama", e2_cfg, e3_cfg, default_esn_factory(e2_cfg), run_getter(frame))
    esn_scores = [d["target"] for d in draws]
    diff = esn_reproduction(esn_scores, llama, e2_cfg, co["d6"]["subsets"]["aurocs"])
    checks["ESN-src.fit_draw_aurocs_vs_d6"] = {"value": diff, "ok": diff <= TOL}
    if not all(c["ok"] for c in checks.values()):
        print(f"REPRODUCTION FAILED, stopped before any SP8 statistic: {checks}")
        OUT.write_text(json.dumps(_clean({"stopped": "reproduction", "checks": checks, "git": git}), indent=2))
        return 1
    print("reproduction: exact")

    # 2. Each corpus: frozen arms, ESN-src's draw distribution.
    per_corpus = {}
    for corpus in cfg["data"]["corpora"]:
        runs = organic[organic["corpus"] == corpus]
        rows = org_table[org_table.index.get_level_values("uid").isin(set(runs["uid"]))]
        arms = {}
        for arm in ("MS-raw", "MS-raw-noLat", "TL-raw", "TL-pct"):
            try:
                arms[arm] = organic_scores({arm: models[arm]}, rows, runs, e2_cfg)[arm]
            except ValueError as e:  # TL-pct needs healthy runs in every fold's reference
                arms[arm] = str(e)
        defined = {a: sc for a, sc in arms.items() if not isinstance(sc, str)}
        per_corpus[corpus] = {"runs": len(runs), "healthy": int(runs["healthy"].sum()),
                              "arms": evaluate(defined, runs, B, seed)
                              | {a: {"undefined": msg} for a, msg in arms.items() if isinstance(msg, str)},
                              "ESN-src": esn_block(esn_scores, runs, B, seed)}

    # 3. Gate and expectations, on the gate corpus.
    g = cfg["gate"]
    gc = per_corpus[g["corpus"]]["arms"]
    ms, nolat = gc["MS-raw"], gc["MS-raw-noLat"]
    gate = {"text": g["text"], "auroc": ms["headline"], "ci": ms["ci"], "c": g["c"],
            "outcome": gate_verdict(ms["ci"]["lo"], g["c"])}
    o1 = {"auroc": ms["headline"], "synthetic_e2": 0.858, "outcome": "HOLDS" if ms["headline"] < 0.858 else "FAILS",
          "estimand_note": cfg["expectations"][0]["note"]}
    o3 = {"noLat": nolat["headline"], "MS-raw": ms["headline"], "difference": nolat["headline"] - ms["headline"],
          "outcome": "HOLDS" if nolat["headline"] >= ms["headline"] - 0.05 else "FAILS"}

    # 4. The latency shift the frozen MS arm sees.
    lat = latency_table({
        "organic_cold.healthy": organic[(organic["corpus"] == g["corpus"]) & organic["healthy"]],
        "organic_cold.failed": organic[(organic["corpus"] == g["corpus"]) & ~organic["healthy"]],
        "organic_t09.all": organic[organic["corpus"] == cfg["data"]["descriptive_corpus"]],
        "llama_source.healthy": llama[llama["failure_class"].isna()],
        "llama_source.failed": llama[llama["failure_class"].notna()],
        "qwen_source.healthy": episodes[(episodes["family"] == "qwen") & episodes["failure_class"].isna()],
        "qwen_source.failed": episodes[(episodes["family"] == "qwen") & episodes["failure_class"].notna()]})

    out = {"what": "SP8 E5: frozen source monitors on organic llama8b runs (run-level labels); GATE",
           "config": cfg, "git": git, "checks": checks, "esn_channels": info["channels"],
           "caveat": cfg["caveat"], "gate": gate, "O-1": o1, "O-3": o3,
           "O-2": cfg["expectations"][1], "O-4": cfg["expectations"][3],
           "per_corpus": per_corpus, "latency_shift": lat,
           "unparsed_tool_steps": int(organic["unparsed_tool_steps"].sum())}
    OUT.write_text(json.dumps(_clean(out), indent=2, default=_json_default))
    print(f"GATE {gate['outcome']}: MS-raw cold AUROC {ms['headline']:.3f} [{ms['ci']['lo']:.3f}, {ms['ci']['hi']:.3f}]")
    print(f"O-1 {o1['outcome']}; O-3 {o3['outcome']} (noLat {nolat['headline']:.3f}, diff {o3['difference']:+.3f})")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
