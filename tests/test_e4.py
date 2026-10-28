import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from test_e3 import RecordingESN, RecordingHGB, e3_run  # noqa: F401  (e3_run is a fixture)
from test_e4_dev import FakeJudge
from test_judges import _run

from watchdog_agent.evaluation import (
    matched_step_auroc,
    matched_step_cluster_ci,
    paired_matched_step_diff,
)
from watchdog_agent.experiments.e2 import (
    build_training_set,
    feature_columns,
    fit_model,
    to_score_dict,
)
from watchdog_agent.experiments.e2_diagnostics import run_getter
from watchdog_agent.experiments.e3 import _esn_src
from watchdog_agent.experiments.e4 import (
    RunStopped,
    auroc_block,
    check_digest,
    eval_sets,
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
from watchdog_agent.judges import JudgeCache

E4 = yaml.safe_load(Path("configs/e4.yaml").read_text(encoding="utf-8"))
TEMPLATE = load_template(E4["prompt"]["file"])


def _runs(n_groups=12, per_group=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for i in range(per_group):
            failed = i % 2 == 1
            rows.append({"uid": f"g{g}-{i}", "task_group": f"g{g}", "T": 6,
                         "failure_class": "looping" if failed else None, "tau": 2.0 if failed else math.nan})
    runs = pd.DataFrame(rows)
    y = runs["failure_class"].notna().to_numpy()

    def scores(shift):
        return {u: rng.standard_normal(6) + shift * yy for u, yy in zip(runs["uid"], y)}

    return runs, scores


# ---------------------------------------------------------------- D


def test_one_draw_D_is_exactly_the_paired_matched_step_difference():
    runs, scores = _runs()
    ref, draw = scores(0.5), scores(1.0)
    ours = paired_median_diff([draw], ref, runs, [3, 4], n=300, seed=0)
    theirs = paired_matched_step_diff(ref, draw, runs, [3, 4], n=300, seed=0).mean  # AUROC(draw) - AUROC(ref)
    assert (ours["point"], ours["lo"], ours["hi"]) == pytest.approx((theirs.point, theirs.lo, theirs.hi), abs=1e-12)
    assert ours["n_skipped"] == theirs.n_skipped


def test_D_is_the_median_over_fit_draws_and_a_better_reference_lowers_it():
    runs, scores = _runs()
    draws = [scores(s) for s in (0.2, 0.8, 1.6)]
    weak, strong = scores(0.0), scores(3.0)
    d = paired_median_diff(draws, weak, runs, [3, 4], n=200, seed=1)
    assert d["point"] == pytest.approx(float(np.median(d["per_draw_diff"])))
    assert len(d["per_draw_diff"]) == 3 and d["lo"] <= d["point"] <= d["hi"]
    assert paired_median_diff(draws, strong, runs, [3, 4], n=200, seed=1)["point"] < d["point"]


# ---------------------------------------------------------------- J-2 reproduces E3


def test_operating_point_reproduces_e3_cells_for_ms_raw_and_esn_src(e3_run):  # noqa: F811
    episodes, e3_cfg, e2_cfg, result, _, _, _ = e3_run
    family, n = "llama", 8
    runs = episodes[episodes["family"] == family]
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), [2, 3, 4, 5])
    # MS-raw exactly as e3.run_family scores it.
    train = build_training_set(table, episodes, family, e2_cfg)
    rows = table[table.index.get_level_values("uid").isin(set(runs["uid"]))
                 & table.index.get_level_values("t").isin(e3_cfg["checkpoints"]["scored"])]
    model = fit_model(train.X["raw"], train.y, train.w, feature_columns("MS", e2_cfg), e2_cfg["model"]["primary"],
                      RecordingHGB)
    ms = to_score_dict(model.score(rows), runs)
    esn_draws, _ = _esn_src(episodes, runs, family, e2_cfg, e3_cfg, lambda ch: RecordingESN(), run_getter(episodes))

    def esn(r):  # E3's fit-draw choice for draw r at budget n
        j = int(np.random.default_rng([e3_cfg["draws"]["seed"], 0, 999, n, r]).integers(len(esn_draws)))
        return esn_draws[j]["target"]

    ours = operating_point({"MS-raw": lambda r: ms, "ESN-src": esn}, runs, e3_cfg, n)["cells"]
    for arm in ("MS-raw", "ESN-src"):
        worst, where = max_abs_diff(ours[arm][str(n)], result["families"][family]["cells"][arm][str(n)])
        assert worst <= 1e-12, (arm, where)
        head = ours[arm][str(n)]["conformal"]["0.5"]["headline"]
        assert "tpr_first_crossing_median" in head and "calibration_share_tied_at_theta_mean" in head


def test_max_abs_diff_treats_json_null_as_nan_and_flags_missing_keys():
    assert max_abs_diff({"a": float("nan"), "b": 1.0}, {"a": None, "b": 1.0})[0] == 0.0
    assert max_abs_diff({"a": 0.5}, {"a": None})[0] == math.inf
    assert max_abs_diff({}, {"a": 1})[0] == math.inf
    assert max_abs_diff({"a": {"b": 2.0}}, {"a": {"b": 2.5}}) == (0.5, "/a/b")


# ---------------------------------------------------------------- judge on llama


def _llama_runs():
    out = []
    for i in range(16):
        cls = [None, "looping", "goal_drift", "tool_cascade"][i % 4]
        r = _run(uid=f"ollama_llama8b/x-{i:02d}", family="llama", failure_class=cls,
                 tau=math.nan if cls is None else 2.0, task_group=f"text:task {i % 5}")
        r["steps_parsed"] = copy.deepcopy(r["steps_parsed"])
        if cls is not None:
            r["steps_parsed"][1]["tool_events"][0]["result"] = "ERROR-MARK"
        r["T"] = len(r["steps_parsed"])
        out.append(r)
    return pd.DataFrame(out)


def _cfg(subset=4):
    cfg = copy.deepcopy(E4)
    cfg["judge"]["determinism_subset"] = {"prompts": subset, "samples": 3, "seed": 0}
    return cfg


def test_judge_scores_cover_healthy_and_headline_runs_only(tmp_path):
    runs = _llama_runs()
    judge = FakeJudge()
    scores, info = judge_scores(TEMPLATE, runs, judge, JudgeCache(tmp_path / "c.jsonl"), _cfg())
    judged = runs[runs["failure_class"] != "tool_cascade"]
    assert info["prompts"] == 3 * len(judged) and info["calls"] == judge.calls == info["prompts"] + 2 * 4
    assert info["parse_failures"] == 0 and info["determinism"]["share_identical_text"] == 1.0
    tc = runs.loc[runs["failure_class"] == "tool_cascade", "uid"].iloc[0]
    assert np.isnan(scores[tc]).all()  # tool_cascade is not judged
    u = judged["uid"].iloc[0]
    assert np.isfinite(scores[u][[2, 3, 4]]).all() and np.isnan(scores[u][[0, 1]]).all()
    assert missing_scores(scores, runs, _cfg()) == []


def test_judge_run_stops_on_a_changed_template_or_too_many_parse_failures(tmp_path):
    runs = _llama_runs()
    with pytest.raises(RunStopped, match="sha256"):
        judge_scores(TEMPLATE + " ", runs, FakeJudge(), JudgeCache(tmp_path / "a.jsonl"), _cfg())
    with pytest.raises(RunStopped, match="parse failure rate"):
        judge_scores(TEMPLATE, runs, FakeJudge(bad_text="maybe"), JudgeCache(tmp_path / "b.jsonl"), _cfg(subset=0))


def test_a_failed_runs_missing_score_is_excluded_but_a_healthy_runs_stops_the_run():
    runs = _llama_runs()
    scores = {u: np.full(int(T), 5.0) for u, T in zip(runs["uid"], runs["T"])}
    failed = runs.loc[runs["failure_class"] == "looping", "uid"].iloc[0]
    healthy = runs.loc[runs["failure_class"].isna(), "uid"].iloc[0]
    scores[failed][4] = np.nan
    missing = missing_scores(scores, runs, _cfg())
    assert missing == [(failed, 4, False)] and missing_policy(missing) == frozenset({(failed, 4)})
    assert worst_case_fill(scores, missing)[failed][4] == 0.0 and np.isnan(scores[failed][4])
    scores[healthy][3] = np.nan
    with pytest.raises(RunStopped, match="healthy"):
        missing_policy(missing_scores(scores, runs, _cfg()))


def test_exclusion_drops_a_pair_from_its_checkpoint_only_and_is_a_no_op_when_empty():
    runs, scores = _runs()
    s = scores(1.0)
    cfg = {"checkpoints": {"report": [3, 4]}, "classes": {"headline": ["looping"]}}
    assert auroc_block(s, runs, cfg)["headline"] == pytest.approx(matched_step_auroc(s, runs, [3, 4]).mean, abs=1e-15)
    ci = matched_step_cluster_ci(s, runs, [3, 4], 300, seed=0).mean
    ours = mean_auroc_ci(s, runs, [3, 4], 300, seed=0)
    assert (ours["lo"], ours["hi"]) == pytest.approx((ci.lo, ci.hi), abs=1e-12)
    u = runs.loc[runs["failure_class"].notna(), "uid"].iloc[0]
    sets = eval_sets(runs, [3, 4], frozenset({(u, 4)}))
    assert u in sets[3].uids and u not in sets[4].uids and sets[4].n_pos == eval_sets(runs, [4])[4].n_pos - 1
    d = paired_median_diff([scores(0.5)], s, runs, [3, 4], 50, 0, exclude=frozenset({(u, 4)}))
    assert d["excluded_pairs"] == [[u, 4]]


def test_a_different_model_digest_stops_the_run():
    check_digest(E4["judge"]["digest"], E4)
    check_digest(E4["judge"]["digest"].removeprefix("sha256:"), E4)  # /api/tags reports the bare hex
    with pytest.raises(RunStopped, match="digest"):
        check_digest("sha256:other", E4)
