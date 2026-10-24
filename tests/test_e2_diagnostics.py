import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

# tests/ has no __init__.py, so pytest puts it on sys.path: reuse E2's synthetic corpus and fake ESN.
from test_e2 import _cfg, _episodes, _fake_esn

from watchdog_agent.baselines.esn.monitor import RunScore
from watchdog_agent.experiments.e2_diagnostics import (
    CLOSEOUT_KEYS,
    REQUIRED_KEYS,
    blind_episodes,
    blind_results,
    check_reproduction,
    mask_rows,
    masked_prefix_table,
    per_fold_mean_auroc,
    restrict_classes,
    run_e2_closeout,
    run_e2_diagnostics,
    sp4_fold_scores,
    subset_draws,
    without_channel,
)
from watchdog_agent.features import STEP_FEATURES, prefix_feature_table, step_feature_rows


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


# ---------------------------------------------------------------- SP5 close-out: D-6, D-7


def _group_features(group):
    return [n for n, (g, _) in STEP_FEATURES.items() if g == group]


@pytest.fixture(scope="module")
def one_failed_run():
    episodes = _episodes()
    run = episodes[episodes["failure_class"].notna() & (episodes["tau"] >= 2)].iloc[0]
    return run, step_feature_rows(run["steps_parsed"])


def test_carry_mask_changes_only_row_tau_and_keeps_position_coherent(one_failed_run):
    run, rows = one_failed_run
    tau = int(run["tau"])
    out = mask_rows(rows, tau, "TL", "carry")
    for i, (a, b) in enumerate(zip(rows, out)):
        if i != tau:
            assert a == b
    for name in _group_features("TL"):
        if name == "step_index":
            assert out[tau][name] == rows[tau][name]
        elif name == "steps_since_error":
            # Consistent with the carried is_error: 0 on an error step, else one more than at tau-1.
            assert out[tau][name] == (0 if out[tau]["is_error"] else rows[tau - 1][name] + 1)
        else:
            assert out[tau][name] == rows[tau - 1][name]
    # Other groups on row tau are untouched.
    assert all(out[tau][n] == rows[tau][n] for n in _group_features("MS") if not np.isnan(rows[tau][n]))
    assert rows[tau] is not out[tau]  # the input rows are not mutated


def test_nan_mask_is_ms_only(one_failed_run):
    run, rows = one_failed_run
    tau = int(run["tau"])
    out = mask_rows(rows, tau, "MS", "nan")
    assert all(np.isnan(out[tau][n]) for n in _group_features("MS"))
    assert all(out[tau][n] == rows[tau][n] for n in _group_features("TL"))
    with pytest.raises(ValueError, match="nan"):
        mask_rows(rows, tau, "TL", "nan")


def test_masked_table_equals_unmasked_for_healthy_runs_and_before_tau():
    episodes = _episodes()
    runs = episodes[episodes["family"] == "llama"]
    plain = prefix_feature_table(dict(zip(runs["uid"], runs["steps_parsed"])), [2, 3, 4])
    masked = masked_prefix_table(runs, [2, 3, 4], "TL", "carry")
    assert masked.index.equals(plain.index) and list(masked.columns) == list(plain.columns)
    info = runs.set_index("uid")
    for (uid, t), row in plain.iterrows():
        cls, tau = info.at[uid, "failure_class"], info.at[uid, "tau"]
        if pd.isna(cls) or t < tau:
            pd.testing.assert_series_equal(masked.loc[(uid, t)], row)
    changed = [(u, t) for (u, t) in plain.index if not masked.loc[(u, t)].equals(plain.loc[(u, t)])]
    assert changed and all(t >= info.at[u, "tau"] for u, t in changed)


def test_subset_draws_are_distinct_seeded_and_inside_the_pool():
    pool = [f"u{i}" for i in range(28)]
    a, b = subset_draws(pool, 21, 30, seed=0), subset_draws(pool, 21, 30, seed=0)
    assert a == b and len(a) == 30
    assert len({frozenset(d) for d in a}) == 30
    assert all(len(d) == 21 == len(set(d)) and set(d) <= set(pool) for d in a)
    with pytest.raises(ValueError):
        subset_draws(pool[:3], 21, 1, seed=0)


class CarryFakeESN:
    theta = 0.0

    def fit(self, fit_runs, val_runs):
        assert all(r.failure_class is None for r in (*fit_runs, *val_runs))

    def score(self, run, carry_forward_step=None):
        steps = list(run.steps)
        if carry_forward_step is not None:
            steps[carry_forward_step] = steps[carry_forward_step - 1]
        per_step = np.cumsum([float(s["error"]) + 0.01 * i for i, s in enumerate(steps)])
        return RunScore(per_step, None, float(per_step.max()))


@pytest.fixture(scope="module")
def closeout():
    episodes, cfg = _episodes(), _cfg()
    cfg["closeout"] = {"n_subsets": 4, "fit_size": 6, "seed": 0, "size_matched_draws": 2}
    e1_cfg = yaml.safe_load(Path("configs/e1.yaml").read_text())
    needed = sorted(set(cfg["checkpoints"]["train"]) | set(cfg["checkpoints"]["score"]))
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), needed)
    result = run_e2_closeout(
        episodes, cfg, e1_cfg, make_esn=lambda ch: CarryFakeESN(), make_monitor=CarryFakeESN, table=table,
        git={"commit": "test", "dirty": False},
    )
    return episodes, cfg, result


def test_closeout_end_to_end_produces_every_key(closeout):
    _, cfg, result = closeout
    assert CLOSEOUT_KEYS <= set(result)
    d6 = result["d6"]
    assert len(d6["subsets"]["aurocs"]) == cfg["closeout"]["n_subsets"]
    assert {"median", "min", "max"} <= set(d6["subsets"])
    assert {"sp4_structure_e2_pool_standardised", "sp4_structure_e2_pool_raw"} <= set(d6["a_fit_pool_only"])
    assert len(d6["b_size_matched"]["draws"]) == cfg["closeout"]["size_matched_draws"]
    for key in ("MS-raw@llama:nan", "MS-raw@llama:carry", "ESN_i@llama:carry", "TL-pct@gemini:carry"):
        block = result["d7"]["b_masking"][key]
        assert {"before", "after", "after_minus_before"} <= set(next(iter(block["per_class"].values())))
    for key in ("ESN_i@llama", "ESN_i@gemini", "TEXT-raw@llama"):
        block = result["d7"]["c_result_blind"][key]
        assert {"before", "after", "after_minus_before"} <= set(block["headline"])


def test_size_matched_fit_sets_stay_inside_their_fold_pool(closeout):
    _, _, result = closeout
    for draw in result["d6"]["b_size_matched"]["draws"]:
        for fold in draw["folds"]:
            assert fold["fit_inside_fold_pool"] and fold["n_fit"] <= fold["n_fold_pool"]


# ---------------------------------------------------------------- D-7c result-blind transform


def _tool_step(result, text_result=None, reasoning=""):
    shown = result if text_result is None else text_result
    return {
        "action": "tool_call", "latency_s": 0.5, "output_tokens": 20, "error": False,
        "text": f'{reasoning}[lookup({{"q": 1}}) -> {shown}]',
        "tool_events": [{"name": "lookup", "args": {"q": 1}, "result": result, "result_chars": len(result),
                         "result_truncated": False, "is_error": False, "latency_s": 0.01}],
    }


def test_blind_keeps_every_length_and_the_step_count():
    long = "Anomaly Detection of Tabular Data | " * 5
    steps = [_tool_step("$172"), _tool_step(long, text_result=long[:100]), {"text": "done", "tool_events": []}]
    out = blind_results(steps)
    assert len(out) == len(steps)
    for a, b in zip(steps, out):
        assert len(a["text"]) == len(b["text"])
        for ea, eb in zip(a["tool_events"], b["tool_events"]):
            assert len(ea["result"]) == len(eb["result"]) and ea["result_chars"] == eb["result_chars"]


def test_blind_removes_content_and_keeps_the_call_and_reasoning():
    a = blind_results([_tool_step("$172", reasoning="rainy mild ")])[0]
    b = blind_results([_tool_step("Oslo", reasoning="rainy mild ")])[0]
    assert a == b  # same length, different content -> identical after the transform
    assert a["text"] == 'rainy mild [lookup({"q": 1}) -> xxxx]'
    assert a["tool_events"][0]["result"] == "xxxx"


def test_blind_leaves_a_run_without_tool_events_unchanged_and_does_not_mutate():
    plain = [{"text": "thinking", "tool_events": []}, {"text": "done"}]
    assert blind_results(plain) == plain
    steps = [_tool_step("$172")]
    original = copy.deepcopy(steps)
    blind_results(steps)
    assert steps == original


def test_blind_raises_on_a_tool_step_it_cannot_parse():
    bad = _tool_step("$172")
    bad["text"] = "[lookup({})]"
    with pytest.raises(ValueError):
        blind_results([bad])


def test_blind_episodes_changes_no_label_and_only_steps():
    episodes = _episodes()
    blind, counts = blind_episodes(episodes)
    other = [c for c in episodes.columns if c != "steps_parsed"]
    pd.testing.assert_frame_equal(blind[other], episodes[other])
    assert (blind["T"] == blind["steps_parsed"].map(len)).all()
    assert counts["llama"]["tool_steps_blinded"] > 0
