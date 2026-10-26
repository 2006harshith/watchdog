"""SP6 E3 runner -> results/e3_recalibration.json, results/e3_draws.csv, MLflow run.

Run from a clean commit: uv run python scripts/run_e3.py
Needs results/e2_prefix_scores.csv and results/e2_closeout.json: the retrained E2 models and the
ESN-src fit draws must reproduce them exactly (checked and recorded under "reproduction_checks").
"""

import json
import math
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e3 import run_e3

RESULTS_DIR = Path("results")
TOL = 1e-9


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _clean(obj):
    """NaN / inf -> None, so the JSON is strict (json.dumps would write the non-standard NaN)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def reproduction_checks(result: dict, closeout: dict) -> dict:
    checks = {}
    for family, fam in result["families"].items():
        for arm, diff in fam["reproduction"].items():
            checks[f"{family}.{arm}.max_abs_diff_vs_e2_prefix_scores"] = {"value": diff, "ok": diff <= TOL}
    ours = result["families"]["llama"]["esn_src"]["headline_auroc_per_fit_draw"]
    d6 = closeout["d6"]["subsets"]["aurocs"]
    diff = max(abs(a - b) for a, b in zip(ours, d6)) if len(ours) == len(d6) else math.inf
    checks["llama.ESN-src.fit_draw_aurocs_vs_d6"] = {"value": diff, "ok": diff <= TOL}
    return checks


def log_to_mlflow(cfg: dict, result: dict, paths: list[Path]) -> str:
    # Same sqlite store as E1/E2 (MLflow 3.16 refuses the ./mlruns file store).
    tracking_dir = Path(cfg["mlflow"]["tracking_dir"]).absolute()
    tracking_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (tracking_dir / "mlflow.db").as_posix())
    name = cfg["mlflow"]["experiment"]
    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else mlflow.create_experiment(name, artifact_location=(tracking_dir / "artifacts").as_uri())
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name="e3-recalibration") as run:
        mlflow.set_tags({"git_commit": result["git"]["commit"], "git_dirty": result["git"]["dirty"],
                         **{f"expectation_{e['id']}": e["outcome"] for e in result["expectations"]}})
        a = str(cfg["thresholds"]["primary_alpha"])
        metrics = {}
        for family, fam in result["families"].items():
            for arm, by_n in fam["cells"].items():
                for n, by_rule in by_n.items():
                    h = by_rule.get("conformal", {}).get(a, {}).get("headline")
                    if h and h["fa"]["mean"] is not None and math.isfinite(h["fa"]["mean"]):
                        metrics[f"{family}.{arm}.n{n}.conformal.mean_fa"] = h["fa"]["mean"]
                        metrics[f"{family}.{arm}.n{n}.conformal.median_tpr"] = h["tpr"]["median"]
        mlflow.log_metrics({k.replace("+", "_plus_"): v for k, v in metrics.items() if v is not None})
        for path in paths:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def _fmt(x) -> str:
    return "  nan" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def main() -> int:
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(cfg["arms"]["hgb"]["config"]).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    e2_scores = pd.read_csv(cfg["arms"]["hgb"]["reproduce_from"])
    result, frame = run_e3(load_episodes(), cfg, e2_cfg, git=git, e2_scores=e2_scores)
    print(f"E3 done in {time.time() - start:.0f}s")
    result["reproduction_checks"] = reproduction_checks(result, json.loads((RESULTS_DIR / "e2_closeout.json").read_text()))

    json_path, csv_path = RESULTS_DIR / "e3_recalibration.json", RESULTS_DIR / "e3_draws.csv"
    json_path.write_text(json.dumps(_clean(result), indent=2, default=_json_default))
    frame.to_csv(csv_path, index=False, float_format="%.6g")
    run_id = log_to_mlflow(cfg, result, [json_path, csv_path])

    for key, c in result["reproduction_checks"].items():
        print(f"{'ok  ' if c['ok'] else 'FAIL'} {key}: {c['value']:.3g}")
    for e in result["expectations"]:
        print(f"{e['id']}: {e['outcome']}")
    a = str(cfg["thresholds"]["primary_alpha"])
    for family, fam in result["families"].items():
        print(f"\n{family}, conformal alpha={a}, headline ({result['caveat']})")
        for arm, by_n in fam["cells"].items():
            for n, by_rule in by_n.items():
                h = by_rule["conformal"][a]["headline"]
                print(f"  {arm:10s} n={n:>2s}  FA median {_fmt(h['fa']['median'])} [{_fmt(h['fa']['p10'])}, "
                      f"{_fmt(h['fa']['p90'])}] mean {_fmt(h['fa']['mean'])}  TPR median {_fmt(h['tpr']['median'])} "
                      f"[{_fmt(h['tpr']['p10'])}, {_fmt(h['tpr']['p90'])}]  defined {h['conformal_defined']}")
    print(f"\nwrote {json_path}, {csv_path}; MLflow run {run_id}")
    return 0 if all(c["ok"] for c in result["reproduction_checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
