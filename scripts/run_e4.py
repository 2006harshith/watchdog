"""SP7 E4 runner -> results/e4_judges.json. NO GATE. Frozen config: configs/e4.yaml (15b31e8).

Run from a clean commit, with Ollama running and phi4-mini:3.8b pulled: uv run python scripts/run_e4.py
Order: monitor reproduction checks (cheap) first, so a mismatch stops before any judge call; then the
judge on llama (cached in results/e4_judge_cache.jsonl; a rerun makes no calls); then D, J-1, J-2, cost.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from run_e3 import _clean, _json_default
from sklearn.ensemble import HistGradientBoostingClassifier

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2 import (
    build_training_set,
    default_esn_factory,
    feature_columns,
    fit_model,
    to_score_dict,
)
from watchdog_agent.experiments.e2_diagnostics import (
    _esn_running_max,
    _fit_esn,
    restrict_classes,
    run_getter,
)
from watchdog_agent.experiments.e3 import _esn_src
from watchdog_agent.experiments.e4 import (
    RunStopped,
    auroc_block,
    check_digest,
    cpu_per_step,
    judge_scores,
    max_abs_diff,
    mean_auroc_ci,
    missing_policy,
    missing_scores,
    operating_point,
    paired_median_diff,
    worst_case_fill,
)
from watchdog_agent.experiments.e4_dev import load_template
from watchdog_agent.features import prefix_feature_table
from watchdog_agent.judges import JudgeCache, OllamaJudge, cost_latency, ollama_digest

HGB_ARMS = ["TL-raw", "TL-pct", "MS-raw"]
OUT = Path("results/e4_judges.json")
TOL = 1e-9


def main() -> int:
    cfg = yaml.safe_load(Path("configs/e4.yaml").read_text(encoding="utf-8"))
    e3_cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(e3_cfg["arms"]["hgb"]["config"]).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")
    ab = json.loads(Path("results/e2_ablation.json").read_text())
    co = json.loads(Path("results/e2_closeout.json").read_text())
    rc = json.loads(Path("results/e3_recalibration.json").read_text())
    family, head, report = cfg["family"], cfg["classes"]["headline"], cfg["checkpoints"]["report"]
    episodes = load_episodes()
    runs = episodes[episodes["family"] == family]
    eval_runs = restrict_classes(runs, head)
    checks: dict = {}

    # 1. Monitors, reproduced before anything else.
    prefix = pd.read_csv(e3_cfg["arms"]["hgb"]["reproduce_from"])
    prefix = prefix[prefix["family"] == family].set_index(["uid", "t"])
    hgb = {arm: to_score_dict(prefix[arm], runs) for arm in HGB_ARMS}
    for arm in HGB_ARMS:
        ours = auroc_block(hgb[arm], runs, cfg)["headline"]
        ref = ab["families"][family]["arms"][arm]["headline"]["mean_primary"]["point"]
        checks[f"{arm}.headline_auroc_vs_e2_ablation"] = {"value": abs(ours - ref), "ok": abs(ours - ref) <= TOL}
    to_run, make_esn = run_getter(episodes), default_esn_factory(e2_cfg)
    t0 = time.time()
    esn_draws, esn_info = _esn_src(episodes, runs, family, e2_cfg, e3_cfg, make_esn, to_run)
    print(f"ESN-src: 30 fit draws in {time.time() - t0:.0f}s")
    aur = np.array(esn_info["headline_auroc_per_fit_draw"])
    for name, ref in [("d6_aurocs", co["d6"]["subsets"]["aurocs"]),
                      ("e3_esn_src", rc["families"][family]["esn_src"]["headline_auroc_per_fit_draw"])]:
        diff = float(np.max(np.abs(aur - np.array(ref))))
        checks[f"ESN-src.fit_draw_aurocs_vs_{name}"] = {"value": diff, "ok": diff <= TOL}
    bad = [k for k, v in checks.items() if not v["ok"]]
    if bad:
        print(f"REPRODUCTION FAILED before any judge call: {bad}")
        OUT.write_text(json.dumps(_clean({"stopped": "reproduction", "checks": checks, "git": git}), indent=2))
        return 1
    print("monitor reproduction: exact")

    # 2. The judge on llama (frozen prompt, frozen weights).
    jc = cfg["judge"]
    digest = ollama_digest(jc["model"])
    check_digest(digest, cfg)
    judge = OllamaJudge(jc["model"], digest, num_predict=jc["num_predict"], num_ctx=jc["num_ctx"], seed=jc["seed"],
                        temperature=jc["temperature"], stop=jc["stop"])
    t0 = time.time()
    try:
        js, jinfo = judge_scores(load_template(cfg["prompt"]["file"]), runs, judge, JudgeCache(cfg["cache"]), cfg)
        missing = missing_scores(js, runs, cfg)
        excl = missing_policy(missing)  # decisions.md 2026-10-28 (post hoc, option 1)
    except RunStopped as e:
        print(f"STOPPED: {e}")
        OUT.write_text(json.dumps(_clean({"stopped": str(e), "checks": checks, "git": git}), indent=2))
        return 2
    print(f"judge: {jinfo['calls']} calls ({jinfo['new_calls']} new) in {time.time() - t0:.0f}s, "
          f"parse failures {jinfo['parse_failures']}")

    # 3. AUROCs, D, J-1, all on the same sets: a failed run's missing (uid, t) dropped for every scorer.
    B, seed = cfg["statistic_D"]["B"], cfg["statistic_D"]["seed"]
    judge_block = auroc_block(js, runs, cfg, excl)
    ci = mean_auroc_ci(js, eval_runs, report, B, seed, excl)
    esn_blocks = [auroc_block(d["target"], runs, cfg, excl) for d in esn_draws]
    D = paired_median_diff([d["target"] for d in esn_draws], js, eval_runs, report, B, seed, excl)
    esn_median = float(np.median([b["headline"] for b in esn_blocks]))
    j1 = {"judge_headline_auroc": judge_block["headline"], "esn_src_median_headline_auroc": esn_median,
          "outcome": "HOLDS" if judge_block["headline"] > esn_median else "FAILS"}
    # Sensitivity: the missing score set to 0 (worst case for the judge), nothing dropped.
    worst = worst_case_fill(js, missing)
    sensitivity = {"missing": [list(m[:2]) for m in missing], "rule": "missing failed-run score set to 0, no pair dropped",
                   "judge_headline_auroc": auroc_block(worst, runs, cfg)["headline"],
                   "D": {k: v for k, v in paired_median_diff([d["target"] for d in esn_draws], worst, eval_runs,
                                                             report, B, seed).items() if k != "per_draw_diff"}}

    # 4. J-2 at E3's operating point (E3's folds and draws, n = 20), with MS-raw and ESN-src rerun as the check.
    j2cfg = cfg["expectations"][1]
    n = j2cfg["n"]
    fam_idx = [e3_cfg["families"]["primary"], *e3_cfg["families"]["descriptive"]].index(family)

    def esn(r):  # E3's fit-draw choice for draw r at budget n
        j = int(np.random.default_rng([e3_cfg["draws"]["seed"], fam_idx, 999, n, r]).integers(len(esn_draws)))
        return esn_draws[j]["target"]

    op = operating_point({"J-local": lambda r: js, "MS-raw": lambda r: hgb["MS-raw"], "ESN-src": esn},
                         runs, e3_cfg, n)["cells"]
    for arm in ("MS-raw", "ESN-src"):
        worst, where = max_abs_diff(op[arm][str(n)], rc["families"][family]["cells"][arm][str(n)])
        checks[f"{arm}.e3_operating_point_cells_n{n}"] = {"value": worst, "where": where, "ok": worst <= 1e-12}
    jl = op["J-local"][str(n)][j2cfg["rule"]][str(j2cfg["alpha"])]["headline"]
    j2 = {"judge_mean_fa": jl["fa"]["mean"], "threshold": j2cfg["max_mean_fa"],
          "outcome": "HOLDS" if jl["fa"]["mean"] <= j2cfg["max_mean_fa"] else "FAILS",
          "judge_headline": jl,
          "context_at_same_point": {a: op[a][str(n)][j2cfg["rule"]][str(j2cfg["alpha"])]["headline"]
                                    for a in ("MS-raw", "ESN-src")}
          | {"ESN-refit (E3, reused)": rc["families"][family]["cells"]["ESN-refit"][str(n)][j2cfg["rule"]][str(j2cfg["alpha"])]["headline"]}}

    # 5. Cost and latency on this machine.
    llama_resp = jinfo["responses"]
    cost = {"judge": cost_latency(llama_resp, 0.0, 0.0, calls_per_run=jinfo["scoring_calls"] / len(eval_runs))
            | {"model_load_s_max": max(r.get("server_load_s", 0.0) for r in llama_resp), "usd": 0.0,
               "note": "wall-clock per call on this machine; local GPU (GTX 1650), no per-token price"}}
    steps = int(runs["T"].sum())
    fit0 = esn_draws[0]
    monitor = _fit_esn(make_esn, esn_info["channels"], fit0["fit"], fit0["val"], to_run)
    cost["ESN-src"] = cpu_per_step(lambda: _esn_running_max(monitor, runs, to_run), steps)
    needed = sorted(set(e2_cfg["checkpoints"]["train"]) | set(e3_cfg["checkpoints"]["scored"]))
    steps_by_uid = dict(zip(runs["uid"], runs["steps_parsed"]))
    n_rows = len(prefix_feature_table(steps_by_uid, needed))  # one row per (run, scored prefix)
    cost["HGB.features"] = cpu_per_step(lambda: prefix_feature_table(steps_by_uid, needed), n_rows)
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    train = build_training_set(table, episodes, family, e2_cfg)
    rows = table[table.index.get_level_values("uid").isin(set(runs["uid"]))]
    model = fit_model(train.X["raw"], train.y, train.w, feature_columns("MS", e2_cfg), e2_cfg["model"]["primary"],
                      HistGradientBoostingClassifier)
    cost["HGB.MS-raw.predict"] = cpu_per_step(lambda: model.score(rows), len(rows))
    cost["machine"] = "Windows 10, GTX 1650 4 GB, Ollama 0.40.12; measured " + time.strftime("%Y-%m-%d")

    texts = pd.Series([r["text"] for r in llama_resp]).value_counts().to_dict()
    out = {"what": "SP7 E4: ESN-src and HGB monitors vs J-local (phi4-mini, local) at matched steps on llama; no gate",
           "config": cfg, "git": git, "checks": checks,
           "caveat": "monitor spread over fit draws and E3 draws = fit-set / calibration variability; D's CI is "
                     "test-side (task-group) only; one 3.8B local judge",
           "judge": {k: v for k, v in jinfo.items() if k != "responses"} | {"digest": digest, "model_key": judge.model,
                                                                         "reply_counts": texts},
           "excluded_pairs": sorted(map(list, excl)),
           "auroc": {"J-local": judge_block | {"headline_ci": ci},
                     **{arm: auroc_block(hgb[arm], runs, cfg, excl) for arm in HGB_ARMS},
                     "ESN-src": {"headline_per_fit_draw": aur.tolist(), "headline_median": esn_median,
                                 "headline_min": float(aur.min()), "headline_max": float(aur.max()),
                                 "per_class_median": {c: float(np.median([b["per_class"][c] for b in esn_blocks]))
                                                      for c in head}}},
           "labels": {"MS-raw": "post hoc"},
           "D": D, "J-1": j1, "J-2": j2, "sensitivity_missing_as_0": sensitivity, "cost_latency": cost}
    OUT.write_text(json.dumps(_clean(out), indent=2, default=_json_default))
    ok = all(v["ok"] for v in checks.values())
    print(f"J-local headline AUROC {judge_block['headline']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]; "
          f"ESN-src median {esn_median:.3f}; D {D['point']:+.3f} [{D['lo']:+.3f}, {D['hi']:+.3f}]")
    print(f"J-1 {j1['outcome']}; J-2 {j2['outcome']} (judge mean FA {j2['judge_mean_fa']:.3f}); checks ok: {ok}")
    print(f"wrote {OUT}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
