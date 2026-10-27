import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from test_e2 import FakeESN, _episodes
from test_e2 import _cfg as _e2_cfg

from watchdog_agent.experiments.e2_diagnostics import e2_healthy_pool
from watchdog_agent.experiments.e3 import (
    FAMILY_KEYS,
    REQUIRED_KEYS,
    calibration_draws,
    calibration_pool_counts,
    evaluate_expectations,
    group_counts,
    group_frame,
    run_e3,
    task_group_folds,
)
from watchdog_agent.features import prefix_feature_table
from watchdog_agent.thresholds import fa_tpr, first_alarm_steps


def _runs():
    # 6 task groups x 4 runs; T = 3..6 per group, so "has a step t" (T >= t + 1) thins out with t.
    # Within each group the T=6 run is failed: it has every step but must never count as calibration.
    rows = []
    for g in range(6):
        for i, T in enumerate([3, 4, 5, 6]):
            rows.append({
                "uid": f"g{g}-{i}", "family": "fam", "task_group": f"g{g}", "T": T,
                "failure_class": "looping" if T == 6 else None,
            })
    other = {"uid": "other-0", "family": "x", "task_group": "g0", "T": 9, "failure_class": None}
    return pd.DataFrame([*rows, other])


def test_counts_partition_and_exclude_failed_and_short_runs():
    out = calibration_pool_counts(_runs(), "fam", [2, 3, 4], k=3, seed=0, budgets=[0, 5, 10])
    # Healthy runs with a step t per group: t=2 -> T in {3,4,5} = 3; t=3 -> 2; t=4 -> 1 (T=5 only).
    for t, per_group in [(2, 3), (3, 2), (4, 1)]:
        block = out["per_t"][str(t)]
        assert block["healthy_with_step"] == 6 * per_group  # the other family's run never counts
        folds = block["per_fold"]
        assert sum(f["test_healthy"] for f in folds) == block["healthy_with_step"]
        for f in folds:
            # Eligible = everything healthy with a step t that is not in the eval fold itself.
            assert f["eligible"] == block["healthy_with_step"] - f["test_healthy"]
            assert f["feasible"] == {"0": True, "5": 5 <= f["eligible"], "10": 10 <= f["eligible"]}
        assert block["folds_skipped"]["10"] == sum(not f["feasible"]["10"] for f in folds)
    # k=3 over 6 groups: 2 groups per fold, so at t=4 each fold holds 2 test runs and 4 eligible.
    assert [f["eligible"] for f in out["per_t"]["4"]["per_fold"]] == [4, 4, 4]
    assert out["per_t"]["4"]["folds_skipped"] == {"0": 0, "5": 3, "10": 3}


def test_e3_config_parses_and_holds_the_four_expectations():
    cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    assert [e["id"] for e in cfg["expectations"]] == ["E-1", "E-2", "E-3", "E-4"]
    assert cfg["thresholds"]["primary_alpha"] == 0.10
    assert cfg["budgets"]["n"] == [0, 10, 20, 40]
    assert cfg["arms"]["ESN-refit"]["families"] == ["llama"]  # D6: no model fit on gemini


# ---------------------------------------------------------------- Step 2: experiment

FITTED_UIDS: list[set] = []
GROUP_RECORD: list = []


class RecordingHGB(HistGradientBoostingClassifier):
    def fit(self, X, y, sample_weight=None):
        FITTED_UIDS.append(set(X.index.get_level_values("uid")))
        return super().fit(X, y, sample_weight=sample_weight)


class RecordingESN(FakeESN):
    def fit(self, fit_runs, val_runs):
        FITTED_UIDS.append({r.uid for r in fit_runs})
        super().fit(fit_runs, val_runs)


def _e3_cfg(episodes, e2_cfg):
    cfg = yaml.safe_load(Path("configs/e3.yaml").read_text())
    cfg["budgets"]["n"] = [0, 4, 8, 14]
    cfg["draws"]["R"] = {"hgb_arms": 3, "ESN-src": 3, "ESN-refit": 2}
    cfg["thresholds"]["alphas"] = [0.25, 0.5]
    cfg["thresholds"]["primary_alpha"] = 0.5
    cfg["thresholds"]["conformal_min_m"] = {0.25: 3, 0.5: 1}
    for fam in ("llama", "gemini"):
        pool = len(e2_healthy_pool(episodes, fam, e2_cfg, "i"))
        cfg["arms"]["ESN-src"]["fit_draws"][fam] = {"n": 3, "fit_size": 8, "pool_runs": pool, "seed": 0}
    for exp in cfg["expectations"]:
        exp["alpha"] = 0.5
        exp["n"] = [4, 8] if isinstance(exp["n"], list) else 8
    return cfg


@pytest.fixture(scope="module")
def e3_run():
    episodes, e2_cfg = _episodes(), _e2_cfg()
    cfg = _e3_cfg(episodes, e2_cfg)
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), [2, 3, 4, 5])
    FITTED_UIDS.clear()
    GROUP_RECORD.clear()
    record: list = []
    result, frame = run_e3(episodes, cfg, e2_cfg, make_esn=lambda ch: RecordingESN(), make_model=RecordingHGB,
                           table=table, git={"commit": "test", "dirty": False}, record=record,
                           group_record=GROUP_RECORD)
    return episodes, cfg, e2_cfg, result, frame, record, [set(s) for s in FITTED_UIDS]


def test_synthetic_run_yields_every_key(e3_run):
    _, _, _, result, frame, _, _ = e3_run
    assert REQUIRED_KEYS <= set(result)
    assert set(result["families"]) == {"llama", "gemini"}
    for fam, block in result["families"].items():
        assert FAMILY_KEYS <= set(block)
        arms = {"TL-raw", "TL-pct", "MS-raw", "ALL-pct", "ESN-src"} | ({"ESN-refit"} if fam == "llama" else set())
        assert set(block["cells"]) == arms
        node = block["cells"]["TL-pct"]["8"]["conformal"]["0.5"]
        assert {"3", "4", "headline"} <= set(node)
        assert {"median", "p10", "p90", "mean", "p_fa_gt_2alpha"} <= set(node["headline"]["fa"])
        assert set(block["cells"]["TL-raw"]) == {"0", "4", "8", "14"}  # n = 0 only for arms needing no target data
        assert "0" not in block["cells"]["TL-pct"]
    assert [e["id"] for e in result["expectations"]] == ["E-1", "E-2", "E-3", "E-4"]
    assert all(e["outcome"] in {"HOLDS", "FAILS", "UNDEFINED"} for e in result["expectations"])
    assert {"family", "arm", "n", "rule", "alpha", "t", "draw", "fa", "tpr", "thetas"} <= set(frame.columns)


def test_calibration_and_reference_runs_never_in_eval_fold_and_never_positive(e3_run):
    episodes, _, _, _, _, record, _ = e3_run
    info = episodes.set_index("uid")
    assert record
    for rec in record:
        used = set(rec["cal"]) | set(rec["other_half"])
        assert used.isdisjoint(rec["eval"])
        assert info.loc[list(used), "failure_class"].isna().all()
        assert (info.loc[list(used), "T"] >= rec["t"] + 1).all()  # every calibration run has a step t
        assert (info.loc[list(used), "family"] == rec["family"]).all()


def test_reference_and_calibration_halves_are_disjoint_and_cover_the_draw(e3_run):
    *_, record, _ = e3_run
    for rec in record:
        if rec["arm"] in ("TL-pct", "ALL-pct", "ESN-refit"):
            assert set(rec["cal"]).isdisjoint(rec["other_half"])
            assert len(rec["cal"]) + len(rec["other_half"]) == rec["n"]


def test_paired_arms_share_calibration_uids(e3_run):
    *_, record, _ = e3_run
    by_cell: dict = {}
    for rec in record:
        by_cell.setdefault((rec["family"], rec["n"], rec["t"], rec["fold"], rec["draw"]), {})[rec["arm"]] = rec
    for cell in by_cell.values():
        full = {cell[a]["cal"] for a in ("TL-raw", "MS-raw", "ESN-src")}
        assert len(full) == 1  # m = n arms: the same n runs
        halves = {(cell[a]["cal"], cell[a]["other_half"]) for a in ("TL-pct", "ALL-pct", "ESN-refit") if a in cell}
        assert len(halves) == 1  # m = n/2 arms: the same split of the same n runs
        (cal, other), = halves
        assert set(cal) | set(other) == set(next(iter(full)))


def test_gemini_never_in_any_model_fit_only_in_calibration(e3_run):
    episodes, *_, record, fitted = e3_run
    gemini = set(episodes.loc[episodes["family"] == "gemini", "uid"])
    assert fitted and all(s.isdisjoint(gemini) for s in fitted)
    assert not any(r["arm"] == "ESN-refit" and r["family"] == "gemini" for r in record)
    assert any(set(r["cal"]) & gemini for r in record)  # gemini healthy runs do calibrate gemini thresholds


def test_infeasible_budget_skips_are_counted(e3_run):
    episodes, cfg, _, _, frame, _, _ = e3_run
    counts = calibration_pool_counts(episodes, "llama", [3, 4], 5, 0, cfg["budgets"]["n"])
    assert any(counts["per_t"][str(t)]["folds_skipped"]["14"] for t in (3, 4))  # the fixture does exercise a skip
    for t in (3, 4):
        expected = counts["per_t"][str(t)]["folds_skipped"]["14"]
        got = frame[(frame["family"] == "llama") & (frame["arm"] == "TL-raw") & (frame["n"] == 14) & (frame["t"] == t)]
        assert (got["folds_skipped"] == expected).all()
        assert (got["folds_used"] == 5 - expected).all()


def test_expectations_are_read_from_the_config(e3_run):
    _, cfg, _, result, frame, _, _ = e3_run
    frames = {f: frame[frame["family"] == f] for f in result["families"]}
    loose, strict = copy.deepcopy(cfg), copy.deepcopy(cfg)
    loose["expectations"][0]["holds_if"] = {"le": 1.0}  # FA <= 1 always holds
    strict["expectations"][0]["holds_if"] = {"le": -1.0}  # FA <= -1 never does
    assert evaluate_expectations(loose, result["families"], frames)[0]["outcome"] == "HOLDS"
    assert evaluate_expectations(strict, result["families"], frames)[0]["outcome"] == "FAILS"


def test_calibration_draws_deterministic_and_none_when_infeasible():
    pool = [f"u{i}" for i in range(12)]
    a = calibration_draws(pool, 6, 4, 0, 0, (0, 3, 1))
    assert a == calibration_draws(list(reversed(pool)), 6, 4, 0, 0, (0, 3, 1))  # pool order does not matter
    assert a != calibration_draws(pool, 6, 4, 0, 0, (0, 3, 2))  # another fold, other draws
    assert calibration_draws(pool, 13, 4, 0, 0, (0, 3, 1)) is None
    for d in a:
        assert len(set(d.uids)) == 6 and set(d.half_a).isdisjoint(d.half_b)
        assert set(d.half_a) | set(d.half_b) == set(d.uids)


def test_task_group_folds_partition_without_splitting_a_group():
    episodes = _episodes()
    pool = episodes[episodes["family"] == "qwen"]
    held = [set(h) for _, h in task_group_folds(pool, 5, 0)]
    assert set().union(*held) == set(pool["uid"]) and sum(map(len, held)) == len(pool)
    group = pool.set_index("uid")["task_group"]
    for h in held:
        assert set(group.loc[list(h)]).isdisjoint(set(group.drop(list(h))))


def test_pooled_test_runs_cover_every_target_run_once_per_draw(e3_run):
    episodes, *_, frame, _, _ = e3_run
    llama = episodes[episodes["family"] == "llama"]
    for t in (3, 4):
        healthy = int((llama["failure_class"].isna() & (llama["T"] >= t + 1)).sum())
        rows = frame[(frame["family"] == "llama") & (frame["n"] == 4) & (frame["t"] == t) & ~frame["undefined"]]
        assert (rows["n_healthy"] == healthy).all()  # every fold used once; no run counted twice
        assert (rows["fa"] == rows["n_healthy_alarm"] / rows["n_healthy"]).all()


def test_n0_esn_src_has_one_row_per_fit_draw(e3_run):
    *_, frame, _, _ = e3_run
    rows = frame[(frame["arm"] == "ESN-src") & (frame["n"] == 0) & (frame["rule"] == "plugin")]
    assert sorted(rows["fit_draw"].unique()) == [0, 1, 2]


# ---------------------------------------------------------------- Step 3: task-group counts

NAN = float("nan")


def _alarm_runs():
    # t = 4, theta = 0.5, headline class "looping". Comments give each run's expected role.
    rows = [
        ("h1", "g0", 6, None, NAN, [0, 0, 0.9, 0, 0, 0]),  # healthy, alarms at 2
        ("h2", "g0", 6, None, NAN, [0] * 6),  # healthy, quiet
        ("p_pre", "g0", 6, "looping", 3, [0, 0.9, 0, 0, 0, 0]),  # alarms only before tau: pre, not post
        ("p_both", "g1", 6, "looping", 3, [0, 0.9, 0, 0.9, 0, 0]),  # first alarm pre-tau, alarms again after
        ("p_post", "g1", 6, "looping", 2, [0, 0, 0, 0, 0.9, 0]),  # first alarm at 4 >= tau
        ("p_nan", "g1", 7, "looping", 2, [NAN, NAN, 0.2, NAN, 0.3, NAN, 0.9]),  # alarms at 6 > t: not by t
        ("p_late", "g1", 7, "looping", 5, [0.9] * 7),  # tau > t: not a positive yet
        ("h_short", "g2", 3, None, NAN, [0.9] * 3),  # no step 4: not counted
        ("other", "g2", 6, "rate_limit", 1, [0.9] * 6),  # not a headline class
    ]
    runs = pd.DataFrame([{"uid": u, "task_group": g, "T": T, "failure_class": c, "tau": tau} for u, g, T, c, tau, _ in rows])
    return runs, {u: s for u, *_, s in rows}


def test_group_counts_split_pre_and_post_onset_alarms_per_task_group():
    runs, scores = _alarm_runs()
    out = group_counts(scores, 0.5, runs, 4, ["looping"])
    got = {g: {k: int(out[k][i]) for k in out if k != "task_group"} for i, g in enumerate(out["task_group"])}
    assert got == {  # g2 has no counted run, so it is left out
        "g0": {"n_healthy": 2, "n_healthy_alarm": 1, "n_pos": 1, "n_pos_alarm": 1,
               "n_pos_alarm_pre_tau": 1, "n_pos_alarm_post": 0},
        "g1": {"n_healthy": 0, "n_healthy_alarm": 0, "n_pos": 3, "n_pos_alarm": 2,
               "n_pos_alarm_pre_tau": 1, "n_pos_alarm_post": 2},
    }
    # Summed over groups, the counts are fa_tpr's.
    rates = fa_tpr(first_alarm_steps(scores, 0.5), runs, 4, ["looping"])
    for k, f in [("n_healthy", "n_healthy"), ("n_healthy_alarm", "n_healthy_alarm"), ("n_pos", "n_pos"),
                 ("n_pos_alarm", "n_pos_alarm"), ("n_pos_alarm_pre_tau", "n_early")]:
        assert out[k].sum() == getattr(rates, f)


def test_group_counts_sum_to_the_pooled_counts_of_every_draw_row(e3_run):
    _, _, _, _, frame, _, _ = e3_run
    groups = group_frame(GROUP_RECORD)
    assert (groups["n_pos_alarm_post"] <= groups["n_pos"]).all()
    assert set(groups.loc[groups["n"] == 0, "fold"]) == {-1} and (groups.loc[groups["n"] > 0, "fold"] >= 0).all()
    keys = ["family", "arm", "n", "rule", "alpha", "t", "draw"]
    summed = groups.groupby(keys)[["n_healthy", "n_healthy_alarm", "n_pos", "n_pos_alarm", "n_pos_alarm_pre_tau"]].sum()
    pooled = frame.set_index(keys)[["n_healthy", "n_healthy_alarm", "n_pos", "n_pos_alarm", "n_early"]]
    pooled = pooled[pooled["n_healthy"] + pooled["n_pos"] > 0]  # cells with every fold skipped have no group rows
    assert len(summed) == len(pooled)
    assert (summed.loc[pooled.index].to_numpy() == pooled.to_numpy()).all()
    # A task group sits in one fold only (folds are by task group).
    assert (groups[groups["n"] > 0].groupby(["family", "task_group"])["fold"].nunique() == 1).all()
