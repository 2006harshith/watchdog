"""SP8 E5: frozen source monitors on the organic llama8b runs (configs/e5.yaml, pre-registered 51c7723).

Nothing is fitted on SP8 data. The HGB arms are E2's models refitted deterministically from source (and
reproduce E2's llama scores before they score anything organic), MS-raw-noLat must reproduce its
committed fingerprint, and ESN-src's 30 fit draws must reproduce D-6. AUROC and its CI are SP7's tested
code (e4.auroc_block / mean_auroc_ci) with the organic failure class: every run is its own task group,
so the cluster bootstrap is a run-level bootstrap.
"""

import numpy as np
import pandas as pd

from watchdog_agent.experiments.e2 import feature_columns, fit_model, to_score_dict
from watchdog_agent.experiments.e2_diagnostics import _headline_mean
from watchdog_agent.experiments.e4 import auroc_block, mean_auroc_ci
from watchdog_agent.organic import FAILED
from watchdog_agent.recalibration import cross_fitted_target_transform

ORGANIC_CFG = {"checkpoints": {"report": [3, 4]}, "classes": {"headline": [FAILED]}}


class ReproductionFailed(RuntimeError):
    pass


def fit_tl_models(train, e2_cfg: dict, make_model) -> dict:
    """TL-raw and TL-pct exactly as E2 fits them (one model each, on the llama target's source rows)."""
    params = e2_cfg["model"]["primary"]
    return {f"TL-{tr}": fit_model(train.X[tr], train.y, train.w, feature_columns("TL", e2_cfg), params, make_model)
            for tr in ("raw", "pct")}


def organic_scores(models: dict, rows: pd.DataFrame, runs: pd.DataFrame, e2_cfg: dict) -> dict:
    """Each frozen HGB arm's scores on one corpus. TL-pct's percentile reference is cross-fitted on the
    corpus's own healthy runs (E2's transform), so a healthy run is never inside its own reference."""
    k, seed = e2_cfg["recalibration"]["target_cross_fit"]["k"], e2_cfg["recalibration"]["target_cross_fit"]["seed"]
    out = {}
    for arm, model in models.items():
        X = cross_fitted_target_transform(rows, runs, k=k, seed=seed) if arm.endswith("-pct") else rows
        out[arm] = to_score_dict(model.score(X), runs)
    return out


def _ci_or_undefined(scores: dict, runs: pd.DataFrame, B: int, seed: int) -> dict:
    # With one healthy run, ~44% of resamples hold no healthy run and the bootstrap refuses (more than
    # 5% one-class replicates): the CI is undefined there and reported so, not patched.
    try:
        return mean_auroc_ci(scores, runs, ORGANIC_CFG["checkpoints"]["report"], B, seed)
    except ValueError as e:
        return {"lo": None, "hi": None, "undefined": str(e)}


def evaluate(scores: dict, runs: pd.DataFrame, B: int, seed: int) -> dict:
    """Per arm: headline (mean of t = 3, 4), per t, and a run-level cluster-bootstrap 95% CI."""
    out = {}
    for arm, sc in scores.items():
        block = auroc_block(sc, runs, ORGANIC_CFG)
        out[arm] = {"headline": block["headline"], "per_t": block["per_t"], "ci": _ci_or_undefined(sc, runs, B, seed)}
    return out


def esn_block(draw_scores: list, runs: pd.DataFrame, B: int, seed: int) -> dict:
    """ESN-src as a distribution over its fit draws; the CI is the median draw's (test side only)."""
    heads = np.array([auroc_block(s, runs, ORGANIC_CFG)["headline"] for s in draw_scores])
    j = int(np.argsort(heads)[len(heads) // 2])  # the upper-median draw, an actual draw
    return {"headline_per_fit_draw": heads.tolist(), "headline_median": float(np.median(heads)),
            "headline_min": float(heads.min()), "headline_max": float(heads.max()),
            "median_draw": j, "median_draw_ci": _ci_or_undefined(draw_scores[j], runs, B, seed)}


def esn_reproduction(draw_scores: list, llama_runs: pd.DataFrame, e2_cfg: dict, d6_aurocs) -> float:
    ours = np.array([_headline_mean(s, llama_runs, e2_cfg) for s in draw_scores])
    return float(np.max(np.abs(ours - np.array(d6_aurocs))))


def gate_verdict(ci_lo: float, c: float) -> str:
    return "PASS" if ci_lo > c else "FAIL"


def latency_table(groups: dict[str, pd.DataFrame]) -> dict:
    """Step latency and output-token quantiles per group of runs (the shift the frozen MS arm sees)."""
    out = {}
    for name, runs in groups.items():
        lat = np.array([s["latency_s"] for st in runs["steps_parsed"] for s in st], dtype=float)
        tok = np.array([s["output_tokens"] for st in runs["steps_parsed"] for s in st], dtype=float)
        out[name] = {"runs": len(runs), "steps": int(lat.size),
                     "latency_s_q10_q50_q90": np.percentile(lat, [10, 50, 90]).tolist(),
                     "share_latency_below_0_05s": float((lat < 0.05).mean()),
                     "output_tokens_q10_q50_q90": np.percentile(tok, [10, 50, 90]).tolist()}
    return out
