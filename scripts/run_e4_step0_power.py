"""SP7 Step 0c power check -> results/e4_step0_power.json. Read-only: no judge score exists.

Real llama labels, checkpoints (t = 3, 4) and the 25 task groups (evaluation.prefix_eval_set on
healthy + headline runs); synthetic scores. One monitor M and two judges with EQUAL true AUROC
(0.77, about ESN-src's median fit-draw AUROC): score = d*y + task-group effect + run effect +
checkpoint noise, the three scorers' noises correlated (rho monitor-judge, rho_jj judge-judge).
D = AUROC_M - max(AUROC_J1, AUROC_J2), the best judge re-selected in every replicate; the CI is a
task-group cluster bootstrap of the mean over t = 3, 4. Power at delta = P(lower bound > -delta).
It is why SP7 has no gate (docs/decisions.md 2026-10-28): 80% power needs delta > 0.15 unless
rho >= 0.6. Assumed: a single monitor stands in for the median over 30 fit draws; scores are
continuous (judge scores will tie); task-group effect sd 0.5, checkpoint noise sd 0.3.

Run: uv run python scripts/run_e4_step0_power.py [S B]   (default S=200 datasets, B=1000)
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

from watchdog_agent.data import load_episodes
from watchdog_agent.evaluation import prefix_eval_set
from watchdog_agent.experiments.e1 import git_state

CFG = {"target_auroc": 0.77, "sigma_group": 0.5, "sigma_checkpoint": 0.3, "rho_judges": 0.6,
       "rho_monitor_judge": [0.0, 0.3, 0.6], "deltas": [0.10, 0.15, 0.20], "seed": 0,
       "headline": ["looping", "goal_drift", "context_corruption"], "checkpoints": [3, 4]}


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    r = rankdata(s)  # average ranks for ties: the Mann-Whitney form of AUROC
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def main() -> int:
    S, B = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (200, 1000)
    ep = load_episodes()
    runs = ep[(ep.family == "llama") & (ep.failure_class.isna() | ep.failure_class.isin(CFG["headline"]))]
    runs = runs.reset_index(drop=True)
    sets = {t: prefix_eval_set(runs, t) for t in CFG["checkpoints"]}
    uid_pos = {u: i for i, u in enumerate(runs.uid)}
    groups = sorted(runs.task_group.unique())
    G = len(groups)
    run_group = runs.task_group.map({g: k for k, g in enumerate(groups)}).to_numpy()
    y_run = runs.failure_class.notna().to_numpy(int)
    rows = {t: [np.flatnonzero(sets[t].task_groups == g) for g in groups] for t in sets}
    cols = {t: np.array([uid_pos[u] for u in sets[t].uids]) for t in sets}
    sg, st = CFG["sigma_group"], CFG["sigma_checkpoint"]
    # Var(noise) = sg^2 + 1 + st^2, and AUROC = Phi(d / sqrt(2 Var)) for a normal shift d.
    d = np.sqrt(2 * (sg**2 + 1 + st**2)) * norm.ppf(CFG["target_auroc"])

    def scores(rng, rho):
        rj = CFG["rho_judges"]
        L = np.linalg.cholesky(np.array([[1, rho, rho], [rho, 1, rj], [rho, rj, 1]]))
        g = rng.standard_normal((G, 3)) @ L.T * sg
        e = rng.standard_normal((len(runs), 3)) @ L.T
        return {t: (d * y_run[:, None] + g[run_group] + e + rng.standard_normal((len(runs), 3)) @ L.T * st)[cols[t]]
                for t in sets}

    def D(sc, idx):
        m = np.mean([[auroc(sets[t].y[idx[t]], sc[t][idx[t], k]) for k in range(3)] for t in sets], axis=0)
        return m[0] - max(m[1], m[2])

    check = scores(np.random.default_rng(CFG["seed"]), 0.3)
    sk = abs(auroc(sets[3].y, check[3][:, 0]) - roc_auc_score(sets[3].y, check[3][:, 0]))
    assert sk < 1e-12, "AUROC disagrees with scikit-learn"

    out = {"what": "SP7 Step 0c power check for D (monitor - best judge), synthetic equal-AUROC scores",
           "config": CFG | {"S": S, "B": B}, "git": git_state(), "sklearn_auroc_check_max_abs_diff": sk,
           "data": {"runs": len(runs), "task_groups": G,
                    "per_t": {str(t): {"pos": s.n_pos, "neg": s.n_neg} for t, s in sets.items()}},
           "per_rho": []}
    for rho in CFG["rho_monitor_judge"]:
        rng = np.random.default_rng([CFG["seed"], round(rho * 10)])
        lbs, halves, points = [], [], []
        for _ in range(S):
            sc = scores(rng, rho)
            points.append(D(sc, {t: np.arange(len(sets[t].uids)) for t in sets}))
            reps = []
            for _ in range(B):
                picked = rng.integers(0, G, size=G)
                idx = {t: np.concatenate([rows[t][k] for k in picked]) for t in sets}
                if all(np.unique(sets[t].y[idx[t]]).size == 2 for t in sets):
                    reps.append(D(sc, idx))
            lo, hi = np.percentile(reps, [2.5, 97.5])
            lbs.append(lo)
            halves.append((hi - lo) / 2)
        lbs = np.array(lbs)
        row = {"rho": rho, "mean_D_under_equality": float(np.mean(points)),
               "ci_half_width_median": float(np.median(halves)),
               "power": {str(dl): float(np.mean(lbs > -dl)) for dl in CFG["deltas"]},
               "delta_for_80pct_power": float(-np.percentile(lbs, 20))}
        out["per_rho"].append(row)
        print(row)
    path = Path("results/e4_step0_power.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
