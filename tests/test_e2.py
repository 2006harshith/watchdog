import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from watchdog_agent.baselines.esn.monitor import RunScore
from watchdog_agent.experiments.e2 import (
    REQUIRED_KEYS,
    build_test_set,
    build_training_set,
    decide_gate,
    run_e2,
)
from watchdog_agent.features import prefix_feature_table

TOOLS = ["calculator", "lookup_flight", "search_catalog", "get_weather"]
HEADLINE = ["looping", "goal_drift", "context_corruption"]


def _step(name, args, result, is_error, logprobs):
    return {
        "action": "tool_call", "latency_s": 0.5, "output_tokens": 20, "error": is_error,
        "token_logprobs": [-0.1, -0.7] if logprobs else [],
        "text": f"[{name}({args}) -> {result}]",
        "tool_events": [{"name": name, "args": args, "result": result, "result_chars": len(result),
                         "result_truncated": False, "is_error": is_error, "latency_s": 0.01}],
    }


def _run(uid, family, corpus, group, cls, rng, logprobs=True):
    tau = int(rng.integers(2, 4)) if cls else None
    steps = []
    for t in range(6):
        failing = cls is not None and t >= tau
        if failing and cls == "looping" and t > tau:
            steps.append(copy.deepcopy(steps[-1]))
            continue
        err = bool(failing and cls in ("tool_cascade", "timeout")) or bool(rng.random() < 0.05)
        result = "Error: boom" if err else ("x" * int(rng.integers(5, 50)))
        steps.append(_step(str(rng.choice(TOOLS)), {"q": int(rng.integers(0, 3))}, result, err, logprobs))
    steps.append({"action": "synthesis", "latency_s": 1.0, "output_tokens": 30, "error": False,
                  "token_logprobs": [-0.2] if logprobs else [], "text": "done", "tool_events": []})
    return {"uid": uid, "family": family, "corpus": corpus, "task_group": group, "failure_class": cls,
            "tau": np.nan if cls is None else float(tau), "T": len(steps), "steps_parsed": steps}


def _episodes(seed=0):
    rng = np.random.default_rng(seed)
    classes = HEADLINE + ["tool_cascade", "timeout"]
    rows = []
    specs = [("qwen", "ollama7b", "q", 8, True), ("qwen", "autogen7b", "a", 6, True),
             ("llama", "ollama_llama8b", "l", 8, True), ("gemini", "real_gemini_long", "m", 6, False)]
    for family, corpus, prefix, n_groups, lp in specs:
        for g in range(n_groups):
            for i in range(4):
                cls = None if i < 2 else classes[(g + i) % len(classes)]
                rows.append(_run(f"{corpus}-{g}-{i}", family, corpus, f"{prefix}{g}", cls, rng, lp))
    # qwen runs sharing a task group with llama (l0) and with gemini (m0): must leave the pools.
    rows.append(_run("ollama7b-shared-l0", "qwen", "ollama7b", "l0", None, rng))
    rows.append(_run("autogen7b-shared-m0", "qwen", "autogen7b", "m0", "looping", rng))
    return pd.DataFrame(rows)


def _cfg():
    cfg = yaml.safe_load(Path("configs/e2.yaml").read_text())
    cfg["checkpoints"]["train"] = [2, 3, 4, 5]
    for params in [cfg["model"]["primary"], *cfg["model"]["sensitivity"].values()]:
        params["max_iter"] = 15
    cfg["model"]["seed_sensitivity"]["target_cross_fit_seeds"] = [0, 1]
    cfg["bootstrap"] = {"seed": 0, "n_primary": 60, "n_secondary": 30}
    cfg["evaluation"]["permutation_importance"]["n_repeats"] = 2
    return cfg


class FakeESN:
    theta = 0.0

    def fit(self, fit_runs, val_runs):
        assert all(r.failure_class is None for r in (*fit_runs, *val_runs))

    def score(self, run):
        per_step = np.cumsum([float(s["error"]) + 0.01 * i for i, s in enumerate(run.steps)])
        return RunScore(per_step, None, float(per_step.max()))


def _fake_esn(channels):
    return FakeESN()


@pytest.fixture(scope="module")
def synthetic():
    episodes, cfg = _episodes(), _cfg()
    table = prefix_feature_table(dict(zip(episodes["uid"], episodes["steps_parsed"])), cfg["checkpoints"]["train"])
    result, scores = run_e2(episodes, cfg, make_esn=_fake_esn, table=table, git={"commit": "test", "dirty": False})
    return episodes, cfg, table, result, scores


@pytest.mark.parametrize("test_family", ["llama", "gemini"])
def test_no_test_family_run_or_task_group_in_training_rows(synthetic, test_family):
    episodes, cfg, table, _, _ = synthetic
    train = build_training_set(table, episodes, test_family, cfg)
    train_uids = set(train.X["raw"].index.get_level_values("uid"))
    test = episodes[episodes["family"] == test_family]
    assert train_uids.isdisjoint(set(test["uid"]))
    train_groups = set(episodes.loc[episodes["uid"].isin(train_uids), "task_group"])
    assert train_groups.isdisjoint(set(test["task_group"]))
    assert "ollama7b-shared-l0" not in train_uids or test_family != "llama"


@pytest.mark.parametrize("test_family", ["llama", "gemini"])
def test_gemini_never_in_training(synthetic, test_family):
    episodes, cfg, table, _, _ = synthetic
    train = build_training_set(table, episodes, test_family, cfg)
    families = set(episodes.set_index("uid").loc[list(set(train.X["raw"].index.get_level_values("uid"))), "family"])
    assert "gemini" not in families


def test_training_rows_follow_d8_and_weights_sum_to_one_per_run(synthetic):
    episodes, cfg, table, _, _ = synthetic
    train = build_training_set(table, episodes, "llama", cfg)
    idx = train.X["raw"].index
    info = episodes.set_index("uid")
    for (uid, t), y in zip(idx, train.y):
        cls, tau = info.at[uid, "failure_class"], info.at[uid, "tau"]
        assert y == (0 if pd.isna(cls) else 1)
        if not pd.isna(cls):
            assert tau <= t
    per_run = pd.Series(train.w, index=idx.get_level_values("uid")).groupby(level=0).sum()
    assert np.allclose(per_run.to_numpy(), 1.0)
    assert all(train.X[k].index.equals(idx) for k in train.X)


def test_esn_and_every_arm_are_scored_on_the_identical_uid_t_set(synthetic):
    _, cfg, _, _, scores = synthetic
    arm_cols = [c for c in scores.columns if c not in ("uid", "family", "t", "label", "class")]
    assert {"ESN_i", "ESN_ii"} <= set(arm_cols)
    assert len(arm_cols) == 2 * len(cfg["arms"]["feature_sets"]) + 2
    for family, block in scores.groupby("family"):
        assert block[arm_cols].notna().all().all(), family
        assert set(block["t"]) == set(cfg["checkpoints"]["score"])


def test_synthetic_end_to_end_produces_every_key(synthetic):
    _, cfg, _, result, _ = synthetic
    assert REQUIRED_KEYS <= set(result)
    arms = {f"{fs}-{tr}" for fs in cfg["arms"]["feature_sets"] for tr in cfg["arms"]["transforms"]}
    for family in cfg["families"]["test"]:
        fam = result["families"][family]
        assert arms | {"ESN_i", "ESN_ii"} <= set(fam["arms"])
        assert fam["esn"]["best"] in ("ESN_i", "ESN_ii")
        for arm in fam["arms"].values():
            assert {"headline", "error_visible", "api", "per_class",
                    "headline_excluding_tool_level_indistinguishable"} <= set(arm)
        assert {"TL-pct_vs_ESN_best", "TL-pct_vs_TL-raw", "TL-pct_vs_MS-pct"} <= set(fam["paired"])
        assert {"TL-z", "hgb_shallow", "hgb_deep", "target_cross_fit_seeds"} <= set(fam["sensitivity"])
    assert "llama" in result["interpretation_only"]["permutation_importance"]
    assert result["git"] == {"commit": "test", "dirty": False}


def test_gate_is_computed_from_the_config(synthetic):
    _, cfg, _, result, _ = synthetic
    paired = {"mean": {"point": 0.05, "lo": 0.01, "hi": 0.09, "n_skipped": 0}}
    assert decide_gate(paired, cfg)["pass"] is True
    stricter = copy.deepcopy(cfg)
    stricter["gate"]["pass_if_lower_bound_above"] = 0.02
    assert decide_gate(paired, stricter)["pass"] is False
    gate = result["gate"]
    assert gate["threshold"] == cfg["gate"]["pass_if_lower_bound_above"]
    assert gate["test_family"] == cfg["gate"]["test_family"]
    lo = result["families"]["llama"]["paired"]["TL-pct_vs_ESN_best"]["mean"]["lo"]
    assert gate["pass"] == (lo > cfg["gate"]["pass_if_lower_bound_above"])


def test_build_test_set_transforms_only_the_test_family_at_score_checkpoints(synthetic):
    episodes, cfg, table, _, _ = synthetic
    test = build_test_set(table, episodes, "llama", cfg, cross_fit_seed=0)
    uids = set(test.X["raw"].index.get_level_values("uid"))
    assert uids <= set(episodes.loc[episodes["family"] == "llama", "uid"])
    assert set(test.X["pct"].index.get_level_values("t")) == set(cfg["checkpoints"]["score"])
    v = test.X["pct"].to_numpy()
    assert np.nanmin(v) >= 0 and np.nanmax(v) <= 1


def test_fit_and_score_leaves_out_all_nan_training_columns(synthetic):
    from watchdog_agent.experiments.e2 import fit_and_score

    episodes, cfg, table, _, _ = synthetic
    train = build_training_set(table, episodes, "llama", cfg)
    test = build_test_set(table, episodes, "llama", cfg, cross_fit_seed=0)
    X_train = train.X["raw"].assign(all_nan=np.nan)
    X_test = test.X["raw"].assign(all_nan=1.0)
    fitted, scores = fit_and_score(train, X_test, X_train, ["tl__is_error__count", "all_nan"], cfg["model"]["primary"])
    assert fitted.dropped == ["all_nan"] and fitted.cols == ["tl__is_error__count"]
    assert scores.index.equals(X_test.index) and scores.notna().all()
