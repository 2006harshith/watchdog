import json

import pandas as pd
import pytest

from watchdog_agent.data import assign_task_groups, load_episodes, model_to_family, to_step_table


def test_model_to_family_maps_known_prefixes():
    assert model_to_family("qwen2.5:7b") == "qwen"
    assert model_to_family("qwen2.5:3b") == "qwen"
    assert model_to_family("llama3.1:8b") == "llama"
    assert model_to_family("gemini-2.5-flash") == "gemini"


def test_model_to_family_raises_on_unknown_model():
    with pytest.raises(ValueError):
        model_to_family("mistral-7b")


def _episode(uid, episode_id, corpus, model, failure_class, tau, steps, task_sha256=None):
    metadata = {"provenance": {"task_sha256": task_sha256}} if task_sha256 else {}
    return {
        "uid": uid,
        "episode_id": episode_id,
        "corpus": corpus,
        "model": model,
        "failure_class": failure_class,
        "tau": tau,
        "T": len(steps),
        "has_logprobs": True,
        "n_steps": len(steps),
        "steps": json.dumps(steps),
        "metadata": json.dumps(metadata),
    }


def _make_episodes_frame(rows):
    df = pd.DataFrame.from_records(rows)
    df["metadata_parsed"] = df["metadata"].map(json.loads)
    df["steps_parsed"] = df["steps"].map(json.loads)
    return df


def test_assign_task_groups_union_find_transitivity():
    # A~B via shared task_sha256, B~C via shared task text; all three land in
    # one task_group even though A and C share neither field directly.
    steps_ab_text_1 = [{"task": "solve x"}]
    steps_bc_text = [{"task": "  solve   y  "}]
    rows = [
        _episode("uA", "eA", "c", "qwen2.5:7b", None, None, steps_ab_text_1, task_sha256="sha1"),
        _episode("uB", "eB", "c", "qwen2.5:7b", None, None, steps_bc_text, task_sha256="sha1"),
        _episode("uC", "eC", "c", "qwen2.5:7b", None, None, steps_bc_text),
    ]
    df = _make_episodes_frame(rows)
    df = assign_task_groups(df)

    groups = dict(zip(df["uid"], df["task_group"]))
    assert groups["uA"] == groups["uB"] == groups["uC"]
    assert df["task_known"].all()


def test_assign_task_groups_orphans_are_singletons_not_merged():
    steps_no_task = [{"action": "tool_call"}]
    rows = [
        _episode("uA", "eA", "same_corpus", "qwen2.5:7b", None, None, steps_no_task),
        _episode("uB", "eB", "same_corpus", "qwen2.5:7b", None, None, steps_no_task),
    ]
    df = _make_episodes_frame(rows)
    df = assign_task_groups(df)

    assert not df["task_known"].any()
    assert df["task_group"].nunique() == 2


def test_to_step_table_step_label_and_step_count():
    steps = [{"task": "t"} for _ in range(4)]
    rows = [
        _episode("u1", "e1", "c", "qwen2.5:7b", None, None, steps),  # healthy
        _episode("u2", "e2", "c", "qwen2.5:7b", "looping", 2, steps),  # failed, tau=2
    ]
    df = _make_episodes_frame(rows)
    df["family"] = df["model"].map(model_to_family)
    df["task_group"] = ["g1", "g2"]
    df["task_known"] = [False, False]

    step_df = to_step_table(df)

    healthy_steps = step_df[step_df["uid"] == "u1"]
    assert len(healthy_steps) == 4
    assert not healthy_steps["step_label"].any()

    failed_steps = step_df[step_df["uid"] == "u2"].sort_values("step_idx")
    assert len(failed_steps) == 4
    assert failed_steps["step_label"].tolist() == [False, False, True, True]


@pytest.mark.smoke
def test_load_episodes_real_data_smoke():
    df = load_episodes()
    assert len(df) > 0
    assert not df["corpus"].str.startswith("organic").any()
    assert df["family"].isin(["qwen", "llama", "gemini"]).all()
    assert df["task_group"].notna().all()
    # every uid gets a group; orphan rows are singletons, never merged together
    orphan = df[~df["task_known"]]
    assert orphan["task_group"].nunique() == len(orphan)
