"""SP5 E2 runner -> results/e2_ablation.json, results/e2_prefix_scores.csv, MLflow run.

Run from a clean commit: uv run python scripts/run_e2.py [--config configs/e2.yaml]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e2 import run_e2

RESULTS_DIR = Path("results")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= _flatten(value, f"{name}.")
        else:
            flat[name] = value
    return flat


def log_to_mlflow(cfg: dict, result: dict, paths: list[Path]) -> str:
    # Same sqlite store as E1 (MLflow 3.16 refuses the ./mlruns file store).
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
    gate = result["gate"]
    with mlflow.start_run(experiment_id=experiment_id, run_name="e2-ablation") as run:
        params = {k: str(v)[:500] for k, v in _flatten(cfg).items() if k != "mlflow.tracking_dir"}
        mlflow.log_params(params)
        mlflow.set_tags({"git_commit": result["git"]["commit"], "git_dirty": result["git"]["dirty"],
                         "gate_outcome": gate["outcome"]})
        metrics = {"gate_diff": gate["difference"]["point"], "gate_diff_lo": gate["difference"]["lo"],
                   "gate_diff_hi": gate["difference"]["hi"]}
        for family, fam in result["families"].items():
            for arm, block in fam["arms"].items():
                m = block["headline"].get("mean_primary")
                if m is not None:
                    metrics[f"{family}.{arm}.headline_mean"] = m["point"]
        mlflow.log_metrics({k.replace("+", "_plus_"): v for k, v in metrics.items()})
        for path in paths:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e2.yaml")
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    result, scores = run_e2(load_episodes(), cfg, git=git)
    print(f"E2 done in {time.time() - start:.0f}s")

    json_path, csv_path = RESULTS_DIR / "e2_ablation.json", RESULTS_DIR / "e2_prefix_scores.csv"
    json_path.write_text(json.dumps(result, indent=2, default=_json_default))
    scores.to_csv(csv_path, index=False)
    run_id = log_to_mlflow(cfg, result, [json_path, csv_path])

    gate = result["gate"]
    d = gate["difference"]
    print(f"GATE {gate['outcome']}: {gate['arm']} - {gate['baseline']} on {gate['test_family']} = "
          f"{d['point']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}], pass if lo > {gate['threshold']}")
    for family, fam in result["families"].items():
        print(f"\n{family} (ESN_best = {fam['esn']['best']})")
        for arm, block in fam["arms"].items():
            m = block["headline"].get("mean_primary")
            if m is not None and "lo" in m:
                print(f"  {arm:10s} {m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]")
    print(f"\nwrote {json_path}, {csv_path}; MLflow run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
