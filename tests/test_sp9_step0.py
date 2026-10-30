import numpy as np
import pandas as pd
import pytest

from watchdog_agent.baselines.esn._vendor.adapter import ExtFeatureState, step_signal_ext
from watchdog_agent.baselines.esn._vendor.common import D_SEM, IDX_TOOL_LATENCY
from watchdog_agent.baselines.esn.monitor import Run
from watchdog_agent.experiments.sp9_step0 import (
    constant_latency_getter,
    drop_summary,
    latency_anatomy,
    step_latencies,
    verdict,
    with_constant_latency,
)

M_LOG_LATENCY = D_SEM + 4 + 4  # adapter.step_signal: x[D_SEM + 4 + 4] = log(latency)


def _step(latency, event_latency=None):
    ev = {"name": "search", "args": {"q": "x"}, "result": "a result"}
    if event_latency is not None:
        ev["latency_s"] = event_latency
    return {"text": "calling search", "action": "tool_call", "latency_s": latency, "output_tokens": 20,
            "error": False, "tool_events": [ev]}


def _signal(step):
    return step_signal_ext(step, ExtFeatureState(), use_sentence_transformers=False)


def test_constant_latency_changes_only_latency_and_leaves_the_original_alone():
    steps = [_step(0.01), _step(4.2, event_latency=0.3)]
    out = with_constant_latency(steps, 1.5)
    assert [s["latency_s"] for s in out] == [1.5, 1.5]
    assert [s["latency_s"] for s in steps] == [0.01, 4.2]
    for a, b in zip(steps, out):
        assert {k: v for k, v in a.items() if k != "latency_s"} == {k: v for k, v in b.items() if k != "latency_s"}


def test_without_measured_tool_latency_only_m_latency_and_derived_x_latency_move():
    a, b = _signal(_step(0.01)), _signal(with_constant_latency([_step(0.01)], 2.0)[0])
    changed = set(np.flatnonzero(~np.isclose(a, b)))
    assert changed == {M_LOG_LATENCY, IDX_TOOL_LATENCY}
    assert b[M_LOG_LATENCY] == pytest.approx(np.log(2.0))


def test_measured_tool_latency_is_untouched():
    a, b = _signal(_step(0.01, event_latency=0.3)), _signal(with_constant_latency([_step(0.01, 0.3)], 2.0)[0])
    assert set(np.flatnonzero(~np.isclose(a, b))) == {M_LOG_LATENCY}
    assert b[IDX_TOOL_LATENCY] == pytest.approx(np.log(0.3))


def test_getter_wraps_runs_and_keeps_label():
    run = Run("u1", [_step(0.01)], "looping")
    got = constant_latency_getter(lambda uid: run, 3.0)("u1")
    assert got.uid == "u1" and got.failure_class == "looping" and got.steps[0]["latency_s"] == 3.0


def test_step_latencies_refuses_missing_values():
    with pytest.raises(ValueError, match="no latency_s"):
        step_latencies([Run("u", [{"text": "x"}])])
    assert step_latencies([Run("u", [_step(0.5), _step(2.0)])]).tolist() == [0.5, 2.0]


def test_drop_summary_is_paired_per_draw():
    s = drop_summary([0.8, 0.7, 0.9], [0.6, 0.7, 0.85], threshold=0.10)
    assert s["per_draw_drop"] == pytest.approx([0.2, 0.0, 0.05])
    assert s["drop_median"] == pytest.approx(0.05)
    assert s["share_drop_above_threshold"] == pytest.approx(1 / 3)
    assert s["constant_median"] == pytest.approx(0.7)


def test_verdict_boundary_holds_at_exactly_the_max_drop():
    assert verdict(0.75, 0.65, 0.10)["outcome"] == "HOLDS"
    assert verdict(0.770, 0.669, 0.10)["outcome"] == "FAILS"


def test_latency_anatomy_counts_fast_steps_by_position_and_label():
    runs = pd.DataFrame({"failure_class": [None, "looping"],
                         "steps_parsed": [[_step(0.01), _step(3.0)], [_step(0.02), _step(0.03, event_latency=0.2)]]})
    a = latency_anatomy(runs)
    assert a["steps"] == 4 and a["missing_latency"] == 0
    assert a["share_fast"] == pytest.approx(0.75)
    assert a["by_position"]["0"] == {"steps": 2, "share_fast": 1.0}
    assert a["by_healthy"]["True"]["share_fast"] == pytest.approx(0.5)
    assert a["fast"]["tool_event_latency_present_share"] == pytest.approx(1 / 3)
