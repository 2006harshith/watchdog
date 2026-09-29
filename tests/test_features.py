import ast
import copy
import math
from pathlib import Path

import numpy as np
import pytest

from watchdog_agent import features as F
from watchdog_agent.baselines.esn._vendor.adapter import embed_text
from watchdog_agent.features import (
    FEATURE_GROUPS,
    prefix_feature_table,
    prefix_features,
    step_feature_rows,
    step_features,
)


def tool(name="calc", args=None, result="ok", is_error=False, result_chars=None, tool_latency=0.01,
         latency=1.0, tokens=10, logprobs=(-0.1, -2.0), text=None):
    args = {"x": 1} if args is None else args
    return {
        "action": "tool_call",
        "error": is_error,
        "latency_s": latency,
        "output_tokens": tokens,
        "token_logprobs": list(logprobs),
        "text": text if text is not None else f"[{name}({args}) -> {result}]",
        "schema": 5,
        "tool_events": [
            {
                "id": "fc-0",
                "name": name,
                "args": args,
                "result": result,
                "result_chars": len(result) if result_chars is None else result_chars,
                "result_truncated": False,
                "is_error": is_error,
                "latency_s": tool_latency,
            }
        ],
    }


def synth(text="The answer is 42.", latency=2.0, tokens=30, logprobs=(-0.5,)):
    return {
        "action": "synthesis",
        "error": False,
        "latency_s": latency,
        "output_tokens": tokens,
        "token_logprobs": list(logprobs),
        "text": text,
        "schema": 5,
        "tool_events": [],
    }


def per_step(steps, key):
    return [row[key] for row in step_feature_rows(steps)]


def nan_equal(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


# ---- per-step features: one test each -------------------------------------------------------


def test_is_tool_call():
    assert per_step([tool(), synth()], "is_tool_call") == [1, 0]


def test_exact_repeat_matches_name_and_args_regardless_of_key_order():
    steps = [
        tool("a", {"x": 1, "y": 2}),
        tool("a", {"y": 2, "x": 1}),  # same call, keys reordered
        tool("a", {"x": 9, "y": 2}),  # same tool, different args
        tool("b", {"x": 1, "y": 2}),  # different tool, same args
        synth(),
    ]
    assert per_step(steps, "exact_repeat") == [0, 1, 0, 0, 0]


def test_consecutive_identical_counts_the_current_run_of_identical_calls():
    a, b = tool("a"), tool("b")
    steps = [a, a, a, b, a, synth(), a]
    # a synthesis step breaks the run; the call after it starts again at 1
    assert per_step(steps, "consecutive_identical") == [1, 2, 3, 1, 1, 0, 1]


def test_new_tool_marks_first_use_of_a_tool_name():
    steps = [tool("a"), tool("b", {"z": 1}), tool("a", {"q": 3}), synth()]
    assert per_step(steps, "new_tool") == [1, 1, 0, 0]


def test_n_arg_keys_and_nan_on_synthesis():
    vals = per_step([tool(args={"a": 1, "b": 2}), synth()], "n_arg_keys")
    assert vals[0] == 2
    assert math.isnan(vals[1])


def test_is_error():
    assert per_step([tool(is_error=True), tool(), synth()], "is_error") == [1, 0, 0]


def test_result_error_prefix_is_independent_of_the_is_error_flag():
    steps = [
        tool(result="Error: invalid syntax", is_error=False),
        tool(result="  error: timed out", is_error=True),
        tool(result="No error here"),
        synth(text="Error in my reasoning"),  # synthesis text is not a tool result
    ]
    assert per_step(steps, "result_error_prefix") == [1, 1, 0, 0]


def test_result_chars_bucket_is_floor_log1p_and_nan_on_synthesis():
    steps = [tool(result_chars=0), tool(result_chars=7), tool(result_chars=1000), synth()]
    vals = per_step(steps, "result_chars_bucket")
    assert vals[:3] == [0, math.floor(math.log1p(7)), math.floor(math.log1p(1000))]
    assert math.isnan(vals[3])


def test_steps_since_error():
    ok, err = tool(), tool(is_error=True)
    assert per_step([ok, err, ok, ok], "steps_since_error") == [1, 0, 1, 2]
    # no error yet: as if one had happened at step -1
    assert per_step([ok, ok, synth()], "steps_since_error") == [1, 2, 3]


def test_step_index():
    assert per_step([tool(), tool(), synth()], "step_index") == [0, 1, 2]


def test_log_latency():
    assert per_step([tool(latency=3.0)], "log_latency") == [pytest.approx(math.log1p(3.0))]


def test_log_output_tokens():
    assert per_step([tool(tokens=24)], "log_output_tokens") == [pytest.approx(math.log1p(24))]


def test_synth_text_log_chars_only_on_synthesis_steps():
    vals = per_step([tool(), synth(text="abcd")], "synth_text_log_chars")
    assert math.isnan(vals[0])
    assert vals[1] == pytest.approx(math.log1p(4))


def test_surprisal_mean_and_max_and_nan_without_logprobs():
    rows = step_feature_rows([tool(logprobs=(-0.1, -2.0)), tool(logprobs=())])
    assert rows[0]["surprisal_mean"] == pytest.approx(1.05)
    assert rows[0]["surprisal_max"] == pytest.approx(2.0)
    assert math.isnan(rows[1]["surprisal_mean"]) and math.isnan(rows[1]["surprisal_max"])


def test_tool_log_latency_nan_when_missing_or_no_call():
    vals = per_step([tool(tool_latency=0.5), tool(tool_latency=None), synth()], "tool_log_latency")
    assert vals[0] == pytest.approx(math.log1p(0.5))
    assert math.isnan(vals[1]) and math.isnan(vals[2])


def test_text_hash_is_the_esn_embedding_of_the_step_text():
    step = tool(text="[calc({'x': 1}) -> 3]")
    row = step_features(step, [])
    got = np.array([row[f"text_hash_{i:02d}"] for i in range(32)])
    np.testing.assert_allclose(got, embed_text(step["text"]))
    assert np.linalg.norm(got) == pytest.approx(1.0)


# ---- prefix aggregation ---------------------------------------------------------------------


def _five_steps():
    a = tool("a")
    return [
        a,
        tool("b", {"k": 1}, is_error=True, result_chars=7, latency=1.0),
        a,
        tool("a", is_error=True, result_chars=1000, latency=4.0),
        tool("c", {"m": 2}, result_chars=0, latency=2.0),
    ]


def test_binary_aggregation_count_rate_last3():
    row = prefix_features(_five_steps(), t=4)
    assert row["mi__is_error__count"] == 2
    assert row["mi__is_error__rate"] == pytest.approx(2 / 5)
    assert row["mi__is_error__last3"] == 1  # steps 2, 3, 4


def test_continuous_aggregation_mean_max_last_last3_mean():
    row = prefix_features(_five_steps(), t=4)
    lat = [math.log1p(v) for v in (1.0, 1.0, 1.0, 4.0, 2.0)]
    assert row["ms__log_latency__mean"] == pytest.approx(np.mean(lat))
    assert row["ms__log_latency__max"] == pytest.approx(max(lat))
    assert row["ms__log_latency__last"] == pytest.approx(lat[4])
    assert row["ms__log_latency__last3_mean"] == pytest.approx(np.mean(lat[2:]))


def test_continuous_aggregation_ignores_nan_and_all_nan_window_is_nan():
    steps = [tool(), synth(text="ab"), synth(text="abcd"), tool(), tool(), tool()]
    row = prefix_features(steps, t=5)
    assert row["ms__synth_text_log_chars__mean"] == pytest.approx(np.mean([math.log1p(2), math.log1p(4)]))
    assert math.isnan(row["ms__synth_text_log_chars__last3_mean"])  # steps 3-5 are all tool steps
    assert math.isnan(row["ms__synth_text_log_chars__last"])


def test_state_features_keep_only_the_value_at_t():
    row = prefix_features(_five_steps(), t=4)
    assert row["mi__step_index__last"] == 4
    assert row["mi__steps_since_error__last"] == 1
    assert "mi__step_index__mean" not in row


def test_max_consecutive_identical_calls():
    a = tool("a")
    row = prefix_features([a, a, a, tool("b"), a], t=4)
    assert row["mi__consecutive_identical__max"] == 3


def test_distinct_tools_per_call():
    row = prefix_features([tool("a"), tool("b", {"z": 1}), tool("a", {"q": 1}), synth()], t=3)
    assert row["mi__distinct_tools_per_call"] == pytest.approx(2 / 3)
    row = prefix_features([synth(), synth(), synth()], t=2)
    assert math.isnan(row["mi__distinct_tools_per_call"])


def test_text_hash_aggregates_to_mean_and_last():
    steps = _five_steps()
    row = prefix_features(steps, t=2)
    emb = np.array([embed_text(s["text"]) for s in steps[:3]])
    assert row["text__text_hash_00__mean"] == pytest.approx(emb[:, 0].mean())
    assert row["text__text_hash_31__last"] == pytest.approx(emb[2, 31])


def test_prefix_features_requires_a_step_t():
    with pytest.raises(ValueError):
        prefix_features(_five_steps(), t=5)


# ---- no future leakage: one case per feature -------------------------------------------------


def _altered_future(steps, t):
    """Change everything about steps after t: errors, calls, repeats, text, timing, logprobs."""
    future = [
        tool("zzz", {"new": i}, result="Error: boom", is_error=True, result_chars=5000,
             tool_latency=None, latency=99.0, tokens=500, logprobs=(-9.0,), text=f"totally different {i}")
        for i in range(len(steps) - t - 1)
    ]
    future[:1] = [copy.deepcopy(steps[0])] if future else []  # an exact repeat of step 0
    return copy.deepcopy(steps[: t + 1]) + future + [synth(text="late"), steps[1]]


ALL_FEATURES = sorted(name for names in FEATURE_GROUPS.values() for name in names)


@pytest.mark.parametrize("name", ALL_FEATURES)
def test_feature_at_t_ignores_steps_after_t(name):
    steps = _five_steps() + [tool("d"), synth()]
    t = 3
    before = prefix_features(steps, t)[name]
    after = prefix_features(_altered_future(steps, t), t)[name]
    assert nan_equal(before, after)


# ---- registry and D9 ------------------------------------------------------------------------


def test_feature_groups_are_disjoint_and_cover_every_column():
    seen = [name for names in FEATURE_GROUPS.values() for name in names]
    assert len(seen) == len(set(seen))
    assert set(seen) == set(prefix_features(_five_steps(), t=3))
    assert set(FEATURE_GROUPS) == {"MI", "MS", "ENV", "TEXT"}
    for group, names in FEATURE_GROUPS.items():
        assert all(n.startswith(f"{group.lower()}__") for n in names)


# D9 (docs/decisions.md). Run-level fields cannot be read at all: features take a list of step
# dicts, never the run row. This guards the step- and event-level ones.
D9_STEP_KEYS = {"schema", "id", "source"}
D9_RUN_KEYS = {"failure_class", "tau", "T", "n_steps", "metadata", "corpus", "model", "family",
               "uid", "episode_id", "task_group", "has_logprobs"}


def _dict_keys_read_in(module_path: Path) -> set[str]:
    """String keys used as d["k"] or d.get("k") anywhere in the module."""
    keys = set()
    for node in ast.walk(ast.parse(module_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            keys.add(node.slice.value)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args and isinstance(node.args[0], ast.Constant)):
            keys.add(node.args[0].value)
    return keys


def test_no_d9_field_is_read_by_the_feature_module():
    read = _dict_keys_read_in(Path(F.__file__))
    assert read.isdisjoint(D9_STEP_KEYS | D9_RUN_KEYS), read & (D9_STEP_KEYS | D9_RUN_KEYS)


def test_changing_d9_step_fields_changes_no_feature():
    steps = _five_steps()
    altered = copy.deepcopy(steps)
    for i, s in enumerate(altered):
        s["schema"] = 99
        s["task"] = f"another task {i}"
        for e in s["tool_events"]:
            e["id"] = f"other-{i}"
            e["source"] = "live_external"
    a, b = prefix_features(steps, 4), prefix_features(altered, 4)
    assert all(nan_equal(a[k], b[k]) for k in a)


# ---- table ------------------------------------------------------------------------------------


def test_prefix_feature_table_matches_prefix_features_and_skips_short_runs():
    runs = {"r1": _five_steps(), "r2": _five_steps()[:3]}
    table = prefix_feature_table(runs, checkpoints=[2, 4])
    assert list(table.index) == [("r1", 2), ("r1", 4), ("r2", 2)]
    for (uid, t), row in table.iterrows():
        expected = prefix_features(runs[uid], t)
        assert all(nan_equal(row[k], expected[k]) for k in expected)


@pytest.mark.smoke
def test_prefix_feature_table_on_real_runs():
    from watchdog_agent.data import load_episodes

    df = load_episodes()
    sample = df.groupby("family").head(40)
    table = prefix_feature_table(dict(zip(sample["uid"], sample["steps_parsed"])), checkpoints=[2, 3, 4])
    assert len(table) > 0
    values = table.to_numpy(dtype=float)
    assert not np.isinf(values).any()
    mi = table[FEATURE_GROUPS["MI"]]
    # MI must be defined on every run with a tool call (D7 guarantees one); only the
    # "not applicable" continuous features may be NaN.
    nan_ok = {c for c in mi.columns if "n_arg_keys" in c or "result_chars_bucket" in c}
    assert not mi.drop(columns=list(nan_ok)).isna().any().any()
