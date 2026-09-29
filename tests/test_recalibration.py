import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from watchdog_agent import recalibration as R
from watchdog_agent.recalibration import (
    cross_fitted_target_transform,
    fit_healthy_reference,
    percentile_transform,
    zscore_transform,
)


def _runs(rows):
    """rows: (uid, family, corpus, task_group, failure_class)."""
    df = pd.DataFrame.from_records(rows, columns=["uid", "family", "corpus", "task_group", "failure_class"])
    df["tau"] = np.where(df["failure_class"].isna(), np.nan, 2.0)
    return df


def _table(values):
    """values: {(uid, t): {feature: value}} -> prefix table indexed by (uid, t)."""
    index = pd.MultiIndex.from_tuples(list(values), names=["uid", "t"])
    return pd.DataFrame.from_records(list(values.values()), index=index)


def _toy():
    """5 healthy runs with f = 1, 2, 2, 3, 5 at t = 3, and failed runs probing chosen x values."""
    healthy = [1.0, 2.0, 2.0, 3.0, 5.0]
    probes = [0.0, 1.0, 2.0, 2.5, 4.0, 5.0, 6.0]
    rows = [(f"h{i}", "qwen", "c", f"g{i}", None) for i in range(len(healthy))]
    rows += [(f"p{i}", "qwen", "c", f"gp{i}", "looping") for i in range(len(probes))]
    vals = {(f"h{i}", 3): {"f": v} for i, v in enumerate(healthy)}
    vals |= {(f"p{i}", 3): {"f": v} for i, v in enumerate(probes)}
    return _table(vals), _runs(rows)


def test_hand_computed_mid_rank_on_five_value_toy():
    table, runs = _toy()
    ref = fit_healthy_reference(table, runs, by="corpus")
    out = percentile_transform(table, ref, runs)
    got = [out.loc[(f"p{i}", 3), "f"] for i in range(7)]
    # mid-rank fraction = (#healthy < x + 0.5 * #healthy == x) / 5
    # x=0: 0 | x=1: 0.5/5 | x=2: (1 + 1)/5 | x=2.5: 3/5 | x=4: 4/5 | x=5: 4.5/5 | x=6: 5/5
    assert got == pytest.approx([0.0, 0.1, 0.4, 0.6, 0.8, 0.9, 1.0])


def test_healthy_value_at_the_reference_median_maps_to_half():
    rows = [(f"h{i}", "qwen", "c", f"g{i}", None) for i in range(5)]
    table = _table({(f"h{i}", 2): {"f": v} for i, v in enumerate([10.0, 20.0, 30.0, 40.0, 50.0])})
    runs = _runs(rows)
    out = percentile_transform(table, fit_healthy_reference(table, runs, by="family"), runs)
    assert out.loc[("h2", 2), "f"] == pytest.approx(0.5)


def test_values_stay_in_unit_interval():
    rng = np.random.default_rng(0)
    rows, vals = [], {}
    for i in range(60):
        cls = None if i % 3 else "looping"
        rows.append((f"u{i}", "qwen", f"c{i % 2}", f"g{i % 12}", cls))
        for t in (2, 3, 4):
            vals[(f"u{i}", t)] = {"a": rng.normal(), "b": float(rng.integers(0, 3)), "c": rng.exponential() * 50}
    table, runs = _table(vals), _runs(rows)
    out = percentile_transform(table, fit_healthy_reference(table, runs, by="corpus"), runs)
    v = out.to_numpy()
    assert np.isfinite(v).all()
    assert (v >= 0).all() and (v <= 1).all()


def test_feature_constant_on_healthy_runs_gives_no_inf_or_nan():
    rows = [(f"h{i}", "qwen", "c", f"g{i}", None) for i in range(4)]
    rows += [("f0", "qwen", "c", "gx", "tool_cascade"), ("f1", "qwen", "c", "gy", "tool_cascade")]
    vals = {(f"h{i}", 3): {"is_error_count": 0.0} for i in range(4)}
    vals |= {("f0", 3): {"is_error_count": 0.0}, ("f1", 3): {"is_error_count": 2.0}}
    table, runs = _table(vals), _runs(rows)
    ref = fit_healthy_reference(table, runs, by="family")
    pct = percentile_transform(table, ref, runs)
    z = zscore_transform(table, ref, runs, std_floor=0.1)
    assert np.isfinite(pct.to_numpy()).all() and np.isfinite(z.to_numpy()).all()
    assert pct.loc[("f0", 3), "is_error_count"] == 0.5
    assert pct.loc[("f1", 3), "is_error_count"] == 1.0
    assert z.loc[("f1", 3), "is_error_count"] == pytest.approx(2.0 / 0.1)


def test_nan_passes_through_and_nan_is_not_counted_in_the_reference():
    rows = [("h0", "qwen", "c", "g0", None), ("h1", "qwen", "c", "g1", None), ("h2", "qwen", "c", "g2", None),
            ("f0", "qwen", "c", "g3", "looping"), ("f1", "qwen", "c", "g4", "looping")]
    vals = {("h0", 3): {"f": 1.0}, ("h1", 3): {"f": np.nan}, ("h2", 3): {"f": 3.0},
            ("f0", 3): {"f": np.nan}, ("f1", 3): {"f": 2.0}}
    table, runs = _table(vals), _runs(rows)
    out = percentile_transform(table, fit_healthy_reference(table, runs, by="family"), runs)
    assert np.isnan(out.loc[("f0", 3), "f"])
    assert np.isnan(out.loc[("h1", 3), "f"])
    assert out.loc[("f1", 3), "f"] == pytest.approx(0.5)  # reference is {1, 3}, n = 2, not 3


def test_all_nan_reference_gives_nan_not_an_error():
    rows = [("h0", "gemini", "c", "g0", None), ("f0", "gemini", "c", "g1", "looping")]
    table = _table({("h0", 3): {"surprisal": np.nan}, ("f0", 3): {"surprisal": 1.0}})
    runs = _runs(rows)
    out = percentile_transform(table, fit_healthy_reference(table, runs, by="family"), runs)
    assert np.isnan(out.loc[("f0", 3), "surprisal"])


def test_reference_is_per_group_and_per_checkpoint():
    rows = [("a0", "qwen", "A", "g0", None), ("a1", "qwen", "A", "g1", None),
            ("b0", "qwen", "B", "g2", None), ("b1", "qwen", "B", "g3", None),
            ("pa", "qwen", "A", "g4", "looping"), ("pb", "qwen", "B", "g5", "looping")]
    vals = {("a0", 3): {"f": 0.0}, ("a1", 3): {"f": 1.0}, ("b0", 3): {"f": 10.0}, ("b1", 3): {"f": 11.0},
            ("a0", 4): {"f": 5.0}, ("a1", 4): {"f": 6.0}, ("b0", 4): {"f": 5.0}, ("b1", 4): {"f": 6.0},
            ("pa", 3): {"f": 5.0}, ("pb", 3): {"f": 5.0}, ("pa", 4): {"f": 5.0}, ("pb", 4): {"f": 5.0}}
    table, runs = _table(vals), _runs(rows)
    out = percentile_transform(table, fit_healthy_reference(table, runs, by="corpus"), runs)
    assert out.loc[("pa", 3), "f"] == 1.0  # above corpus A's healthy values at t = 3
    assert out.loc[("pb", 3), "f"] == 0.0  # below corpus B's healthy values at t = 3
    assert out.loc[("pa", 4), "f"] == pytest.approx(0.25)  # t = 4 has its own reference {5, 6}


def test_missing_group_or_checkpoint_in_reference_raises():
    table, runs = _toy()
    ref = fit_healthy_reference(table, runs, by="corpus")
    other = runs.assign(corpus="unseen")
    with pytest.raises(ValueError, match="unseen"):
        percentile_transform(table, ref, other)


def test_zscore_uses_healthy_mean_and_std_with_a_floor():
    table, runs = _toy()
    ref = fit_healthy_reference(table, runs, by="corpus")
    z = zscore_transform(table, ref, runs, std_floor=0.1)
    healthy = np.array([1.0, 2.0, 2.0, 3.0, 5.0])
    assert z.loc[("p4", 3), "f"] == pytest.approx((4.0 - healthy.mean()) / healthy.std())


# ---- cross-fitting -----------------------------------------------------------------------------


def _target(seed=0, n_groups=10):
    rng = np.random.default_rng(seed)
    rows, vals = [], {}
    for g in range(n_groups):
        for i, cls in enumerate((None, None, "looping")):
            uid = f"u{g}_{i}"
            rows.append((uid, "llama", "ollama_llama8b", f"g{g}", cls))
            for t in (3, 4):
                vals[(uid, t)] = {"f": rng.normal(), "k": float(rng.integers(0, 4))}
    return _table(vals), _runs(rows)


def test_cross_fit_a_run_never_enters_its_own_reference():
    # Both healthy runs of task group g0 get a value far above everything else. If either run, or
    # its task-group mate, were in the reference that transforms it, its mid-rank would be < 1.
    table, runs = _target()
    for uid in ("u0_0", "u0_1"):
        table.loc[(uid, 3), "f"] = 1e9
    out = cross_fitted_target_transform(table, runs, k=5, seed=0)
    assert out.loc[("u0_0", 3), "f"] == 1.0
    assert out.loc[("u0_1", 3), "f"] == 1.0


def test_cross_fit_keeps_shape_order_and_bounds():
    table, runs = _target()
    out = cross_fitted_target_transform(table, runs, k=5, seed=0)
    assert out.index.equals(table.index) and list(out.columns) == list(table.columns)
    v = out.to_numpy()
    assert ((v >= 0) & (v <= 1)).all()


def test_cross_fit_value_of_a_run_does_not_depend_on_its_own_label():
    table, runs = _target()
    flipped = runs.copy()
    flipped.loc[flipped["uid"] == "u3_0", "failure_class"] = "looping"  # healthy -> failed
    a = cross_fitted_target_transform(table, runs, k=5, seed=0)
    b = cross_fitted_target_transform(table, flipped, k=5, seed=0)
    pd.testing.assert_series_equal(a.loc["u3_0"].stack(), b.loc["u3_0"].stack())


def test_cross_fit_zscore_method_and_seed_changes_folds():
    table, runs = _target()
    z = cross_fitted_target_transform(table, runs, k=5, seed=0, method="zscore", std_floor=0.1)
    assert np.isfinite(z.to_numpy()).all()
    a = cross_fitted_target_transform(table, runs, k=5, seed=0)
    b = cross_fitted_target_transform(table, runs, k=5, seed=1)
    assert not a.equals(b)


def test_cross_fit_requires_one_family():
    table, runs = _target()
    runs.loc[0, "family"] = "qwen"
    with pytest.raises(ValueError, match="one family"):
        cross_fitted_target_transform(table, runs, k=5, seed=0)


@pytest.mark.smoke
def test_cross_fit_on_real_target_families_at_scored_checkpoints():
    from watchdog_agent.data import load_episodes
    from watchdog_agent.features import prefix_feature_table

    df = load_episodes()
    for family in ("llama", "gemini"):
        runs = df[df["family"] == family]
        table = prefix_feature_table(dict(zip(runs["uid"], runs["steps_parsed"])), [2, 3, 4])
        v = cross_fitted_target_transform(table, runs, k=5, seed=0).to_numpy()
        assert np.nanmin(v) >= 0 and np.nanmax(v) <= 1


# ---- D9 ----------------------------------------------------------------------------------------

# The reference must know which runs are healthy (failure_class) and how to group/fold them
# (family, corpus, task_group): that is the premise, "a few healthy runs". No other D9 field may be
# read, and a run's own label never changes its own transformed value (test above).
FORBIDDEN = {"tau", "T", "n_steps", "metadata", "metadata_parsed", "steps_parsed", "model",
             "episode_id", "has_logprobs"}


def test_no_other_d9_field_is_read_by_the_recalibration_module():
    keys = set()
    for node in ast.walk(ast.parse(Path(R.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            keys.add(node.slice.value)
        if isinstance(node, ast.Attribute):
            keys.add(node.attr)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            keys.add(node.value)
    assert keys.isdisjoint(FORBIDDEN), keys & FORBIDDEN
