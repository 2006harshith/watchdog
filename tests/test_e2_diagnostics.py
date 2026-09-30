import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# tests/ has no __init__.py, so pytest puts it on sys.path: reuse E2's synthetic corpus and fake ESN.
from test_e2 import _cfg, _episodes, _fake_esn

from watchdog_agent.experiments.e2_diagnostics import (
    REQUIRED_KEYS,
    check_reproduction,
    per_fold_mean_auroc,
    restrict_classes,
    run_e2_diagnostics,
    sp4_fold_scores,
    without_channel,
)
from watchdog_agent.features import prefix_feature_table


def _runs(classes, taus, groups):
    return pd.DataFrame({
        "uid": [f"r{i}" for i in range(len(classes))],
        "failure_class": classes,
        "tau": taus,
        "T": 6,
        "task_group": groups,
    })


def test_restrict_classes_keeps_healthy_and_only_the_named_classes():
    runs = _runs([None, "looping", "tool_cascade", None, "goal_drift"], [np.nan, 2, 2, np.nan, 3], list("abcde"))
    kept = restrict_classes(runs, ["looping", "goal_drift"])
    assert kept["uid"].tolist() == ["r0", "r1", "r3", "r4"]


def test_per_fold_mean_equals_pooled_with_one_fold():
    from watchdog_agent.evaluation import matched_step_auroc

    rng = np.random.default_rng(0)
    runs = _runs([None] * 6 + ["looping"] * 6, [np.nan] * 6 + [2] * 6, [f"g{i % 4}" for i in range(12)])
    scores = {u: rng.random(6) for u in runs["uid"]}
    pooled = matched_step_auroc(scores, runs, [3, 4]).mean
    out = per_fold_mean_auroc(scores, runs, [runs["uid"].tolist()], [3, 4])
    assert out["n_folds_used"] == 1 and out["mean"] == pytest.approx(pooled, abs=1e-12)


def test_per_fold_skips_a_fold_with_a_one_class_checkpoint():
    runs = _runs([None, None, "looping", "looping", None, None], [np.nan, np.nan, 2, 2, np.nan, np.nan], list("aabbcc"))
    scores = {u: np.arange(6, dtype=float) + i for i, u in enumerate(runs["uid"])}
    out = per_fold_mean_auroc(scores, runs, [["r0", "r1", "r2", "r3"], ["r4", "r5"]], [3, 4])
    assert out["n_folds_used"] == 1
    assert "skipped" in out["per_fold"][1] and "auroc" in out["per_fold"][0]


def test_without_channel_keeps_order_and_does_not_mutate():
    cfg = {"esn": {"channels": ["e", "u", "m", "x"], "K": 8}}
    out = without_channel(cfg, "u")
    assert out["esn"]["channels"] == ["e", "m", "x"]
    assert cfg["esn"]["channels"] == ["e", "u", "m", "x"]


@pytest.fixture(scope="module")
def synthetic():
    episodes, cfg = _episodes(), _cfg()
    e1_cfg = yaml.safe_load(Path("configs/e1.yaml").read_text())
    needed = sorted(set(cfg["checkpoints"]["train"]) | set(cfg["checkpoints"]["score"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    result = run_e2_diagnostics(
        episodes, cfg, e1_cfg, make_esn=_fake_esn, make_monitor=lambda: _fake_esn(None), table=table,
        git={"commit": "test", "dirty": False},
    )
    return episodes, cfg, e1_cfg, result


def test_standardisation_does_not_change_within_fold_auroc(synthetic):
    # Per fold and checkpoint the SP4 standardisation is (x - mean) / std with std > 0 (or centring
    # only): a monotone map, so it can only move AUROC through pooling across folds, never within one.
    episodes, cfg, e1_cfg, _ = synthetic
    raw, std, folds = sp4_fold_scores(episodes, e1_cfg, cfg["checkpoints"]["score"], lambda: _fake_esn(None))
    runs = episodes[episodes["family"] == "llama"]
    a = per_fold_mean_auroc(raw, runs, folds, cfg["checkpoints"]["primary"])
    b = per_fold_mean_auroc(std, runs, folds, cfg["checkpoints"]["primary"])
    assert a["n_folds_used"] == b["n_folds_used"] > 0
    assert a["mean"] == pytest.approx(b["mean"], abs=1e-12)


def test_sp4_folds_cover_every_llama_run_once(synthetic):
    episodes, cfg, e1_cfg, _ = synthetic
    _, _, folds = sp4_fold_scores(episodes, e1_cfg, cfg["checkpoints"]["score"], lambda: _fake_esn(None))
    flat = [u for f in folds for u in f]
    assert len(flat) == len(set(flat))
    assert set(flat) == set(episodes.loc[episodes["family"] == "llama", "uid"])


def test_synthetic_end_to_end_produces_every_key(synthetic):
    _, _, _, result = synthetic
    assert REQUIRED_KEYS <= set(result)
    expected = {"sp4_A_standardised", "sp4_A_raw", "e2_ESN_i", "e2_ESN_i_no_u",
                "e2_MS_raw", "e2_MS_raw_no_surprisal", "e2_MS_raw_surprisal_only"}
    assert expected == set(result["rows"])
    for row in result["rows"].values():
        assert {"headline", "all_classes", "per_fold_headline", "per_fold_all_classes", "per_class"} <= set(row)
    assert result["git"] == {"commit": "test", "dirty": False}


def test_check_reproduction_raises_on_a_mismatch():
    result = {"rows": {"e2_ESN_i": {"headline": {"point": 0.846}}, "sp4_A_standardised": {"all_classes": {"point": 0.646}},
                       "e2_MS_raw": {"headline": {"point": 0.858}}}}
    e2 = {"families": {"llama": {"arms": {"ESN_i": {"headline": {"mean_primary": {"point": 0.846}}},
                                          "MS-raw": {"headline": {"mean_primary": {"point": 0.858}}}}}}}
    sp4 = {"d_esn_matched_step": {"monitors": {"A": {"mean_over_primary": {"point": 0.646}}}}}
    assert set(check_reproduction(result, e2, sp4)) == {"e2_ESN_i", "sp4_A_standardised", "e2_MS_raw"}
    bad = copy.deepcopy(result)
    bad["rows"]["e2_ESN_i"]["headline"]["point"] = 0.8
    with pytest.raises(AssertionError, match="e2_ESN_i"):
        check_reproduction(bad, e2, sp4)
