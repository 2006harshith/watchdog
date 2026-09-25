"""SP3 E1 runner: python -> results/e1_collapse.json, results/e1_episode_scores.csv, MLflow run.

Run: uv run python scripts/run_e1.py [--config configs/e1.yaml]
"""

import argparse
import sys
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import run_e1, write_outputs

RESULTS_DIR = Path("results")


def _flatten(d: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat |= _flatten(value, f"{name}.")
        else:
            flat[name] = value
    return flat


def log_to_mlflow(cfg: dict, result: dict, paths: tuple[Path, Path]) -> str:
    # MLflow 3.16 refuses the ./mlruns file store ("maintenance mode"); a sqlite store inside the
    # gitignored tracking dir keeps runs and artifacts out of git and out of the repo root.
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
    monitors, gate = result["monitors"], result["gate"]
    with mlflow.start_run(experiment_id=experiment_id, run_name="e1-collapse") as run:
        mlflow.log_params({k: v for k, v in _flatten(cfg).items() if k != "mlflow.tracking_dir"})
        mlflow.set_tags({"git_commit": result["git"]["commit"], "git_dirty": result["git"]["dirty"],
                         "gate_outcome": gate["outcome"]})
        metrics = {"gap_B_minus_A": gate["gap_B_minus_A"],
                   "paired_diff_lo": gate["rule_b"]["paired_ci_B_minus_A"]["lo"],
                   "paired_diff_hi": gate["rule_b"]["paired_ci_B_minus_A"]["hi"],
                   "gate_rule_b_pass": float(gate["rule_b"]["pass"]),
                   "gate_rule_a_pass": float(gate["rule_a"]["pass"])}
        for name, m in monitors.items():
            for metric in ("auroc", "auprc", "tpr_at_fpr"):
                metrics[f"{metric}_{name}"] = m[metric]["point"]
                metrics[f"{metric}_{name}_lo"] = m[metric]["lo"]
                metrics[f"{metric}_{name}_hi"] = m[metric]["hi"]
            metrics[f"healthy_fa_rate_{name}"] = m["default_alarm"]["healthy_fa_rate"]
            metrics[f"detection_rate_{name}"] = m["default_alarm"]["detection_rate"]
        mlflow.log_metrics(metrics)
        for path in paths:
            mlflow.log_artifact(str(path))
        return run.info.run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e1.yaml")
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(Path(args.config).read_text())
    result, scores = run_e1(load_episodes(), cfg)
    paths = write_outputs(result, scores, RESULTS_DIR)
    run_id = log_to_mlflow(cfg, result, paths)

    m, gate = result["monitors"], result["gate"]
    for name in ("A", "B"):
        a = m[name]["auroc"]
        print(f"AUROC {name}: {a['point']:.4f} [{a['lo']:.4f}, {a['hi']:.4f}] | "
              f"mean per-fold {m[name]['mean_per_fold_auroc']:.4f} | "
              f"FA at default {m[name]['default_alarm']['healthy_fa_rate']:.3f}")
    d = gate["rule_b"]["paired_ci_B_minus_A"]
    print(f"gap B-A {gate['gap_B_minus_A']:.4f}, paired CI [{d['lo']:.4f}, {d['hi']:.4f}] -> "
          f"rule b {gate['outcome']}, rule a {'PASS' if gate['rule_a']['pass'] else 'FAIL'}")
    if result["flags"]["baseline_B_below_sanity_auroc"]:
        print("SANITY FLAG: refitted baseline B AUROC < 0.70")
    print(f"wrote {paths[0]}, {paths[1]}; MLflow run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
