"""SP6 E3 runner -> results/e3_recalibration.json, results/e3_draws.csv,
results/e3_group_counts.parquet (gitignored; regenerable), MLflow run.

Run from a clean commit: uv run python scripts/run_e3.py [--reference <commit>]
Needs results/e2_prefix_scores.csv and results/e2_closeout.json: the retrained E2 models and the
ESN-src fit draws must reproduce them exactly (checked and recorded under "reproduction_checks").
--reference: also check every cell and draw row against the results committed at <commit>.
"""

import argparse
import io
import json
import math
import subprocess
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
from watchdog_agent.experiments.e3 import GROUP_FIELDS, group_frame, run_e3

RESULTS_DIR = Path("results")
TOL = 1e-9
REFERENCE_TOL = 1e-12
DRAW_KEYS = ["family", "arm", "n", "rule", "alpha", "t", "draw"]


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


def group_sum_check(groups: pd.DataFrame, frame: pd.DataFrame) -> dict:
    """Task-group counts summed over folds and groups must equal the pooled counts of every draw
    row exactly. Cells with every fold skipped have no group rows, hence fillna(0)."""
    summed = groups.groupby(DRAW_KEYS)[list(GROUP_FIELDS)].sum().add_suffix("_g").reset_index()
    both = frame.merge(summed, on=DRAW_KEYS, how="left")
    both[summed.columns.difference(DRAW_KEYS)] = both[summed.columns.difference(DRAW_KEYS)].fillna(0)
    pairs = [(f, f"{f}_g") for f in ("n_healthy", "n_healthy_alarm", "n_pos", "n_pos_alarm")]
    pairs.append(("n_early", "n_pos_alarm_pre_tau_g"))  # fa_tpr's "early" is the pre-tau first alarm
    bad = sum(int((both[a] != both[b]).sum()) for a, b in pairs)
    return {"value": bad, "ok": bad == 0}


def _git_show(commit: str, path: str) -> str:
    return subprocess.run(["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=True).stdout


def _max_diff(a, b, path: str, mismatches: list) -> float:
    """Max abs difference over every number in two JSON trees; any non-number or shape difference is a mismatch."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            mismatches.append(path)
        return max([_max_diff(a[k], b[k], f"{path}.{k}", mismatches) for k in set(a) & set(b)], default=0.0)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            mismatches.append(path)
        return max([_max_diff(x, y, f"{path}[{i}]", mismatches) for i, (x, y) in enumerate(zip(a, b))], default=0.0)
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(a - b)
    if a != b:
        mismatches.append(path)
    return 0.0


def reference_checks(result: dict, frame_path: Path, commit: str) -> dict:
    """Every existing cell and draw row against the files committed at `commit` (git and the
    reproduction checks themselves excluded). The CSV is compared as written (%.6g)."""
    ref = json.loads(_git_show(commit, "results/e3_recalibration.json"))
    ours = json.loads(json.dumps(_clean(result), default=_json_default))
    mismatches: list = []
    skip = {"git", "reproduction_checks"}
    diff = _max_diff({k: v for k, v in ours.items() if k not in skip},
                     {k: v for k, v in ref.items() if k not in skip}, "", mismatches)
    checks = {f"json_vs_{commit}": {"value": diff, "ok": diff <= REFERENCE_TOL and not mismatches,
                                    "mismatches": mismatches[:20]}}
    new, old = pd.read_csv(frame_path), pd.read_csv(io.StringIO(_git_show(commit, "results/e3_draws.csv")))
    same_shape = new.shape == old.shape and list(new.columns) == list(old.columns)
    csv_diff, bad_cols = math.inf, ["shape"]
    if same_shape:
        num = new.select_dtypes("number").columns
        csv_diff = float(np.nanmax(np.abs(new[num].to_numpy(float) - old[num].to_numpy(float)))) if len(num) else 0.0
        # NaN in the same place is equal; NaN against a number is a mismatch.
        nan_moved = [c for c in num if not (new[c].isna() == old[c].isna()).all()]
        bad_cols = nan_moved + [c for c in new.columns.difference(num) if not new[c].fillna("").astype(str).equals(old[c].fillna("").astype(str))]
    checks[f"draws_csv_vs_{commit}"] = {"value": csv_diff, "ok": csv_diff <= REFERENCE_TOL and not bad_cols,
                                        "mismatched_columns": bad_cols}
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="commit whose results/e3_* every cell must reproduce")
    args = parser.parse_args()
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    e2_cfg = yaml.safe_load(Path(cfg["arms"]["hgb"]["config"]).read_text())
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    start = time.time()
    e2_scores = pd.read_csv(cfg["arms"]["hgb"]["reproduce_from"])
    group_record: list = []
    result, frame = run_e3(load_episodes(), cfg, e2_cfg, git=git, e2_scores=e2_scores, group_record=group_record)
    print(f"E3 done in {time.time() - start:.0f}s")
    groups = group_frame(group_record)
    result["reproduction_checks"] = reproduction_checks(result, json.loads((RESULTS_DIR / "e2_closeout.json").read_text()))
    result["reproduction_checks"]["group_counts_sum_to_pooled_counts"] = group_sum_check(groups, frame)

    json_path, csv_path = RESULTS_DIR / "e3_recalibration.json", RESULTS_DIR / "e3_draws.csv"
    groups_path = RESULTS_DIR / "e3_group_counts.parquet"
    frame.to_csv(csv_path, index=False, float_format="%.6g")
    if args.reference:
        result["reproduction_checks"] |= reference_checks(result, csv_path, args.reference)
    json_path.write_text(json.dumps(_clean(result), indent=2, default=_json_default))
    # Categorical string columns: parquet stores each distinct value once (dictionary encoding).
    groups.astype({c: "category" for c in ("family", "arm", "rule", "task_group")}).to_parquet(groups_path, index=False)
    run_id = log_to_mlflow(cfg, result, [json_path, csv_path, groups_path])

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
    print(f"\nwrote {json_path}, {csv_path}, {groups_path} ({len(groups)} rows); MLflow run {run_id}")
    return 0 if all(c["ok"] for c in result["reproduction_checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
