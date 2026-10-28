import json
import math

import pandas as pd
import pytest

from watchdog_agent.features import prefix_feature_table
from watchdog_agent.organic import (
    FAILED,
    adapt_steps,
    load_organic,
    numbers_in,
    parse_tool_text,
    strict_label,
)


def _step(action, text, error=False):
    return {"text": text, "token_logprobs": [-0.1], "action": action, "latency_s": 2.7, "output_tokens": 23,
            "error": error}


def test_tool_text_parses_into_one_event_and_odd_shapes_do_not():
    e = parse_tool_text('[lookup_flight({"destination": "Bergen", "origin": "Cusco"}) -> $126]')
    assert e["name"] == "lookup_flight" and e["args"] == {"destination": "Bergen", "origin": "Cusco"}
    assert e["result"] == "$126" and e["result_chars"] == 4 and e["latency_s"] is None
    err = parse_tool_text('[calculator({"args": "[126, 773]", "op": "+"}) -> Error: invalid syntax (<string>, line 0)]')
    assert err["result"].startswith("Error:")
    assert parse_tool_text('}\n{"name": "lookup_weather", "parameters": {"city": "Bergen"}}') is None
    assert parse_tool_text("[calc(not json) -> 3]") is None


def test_adapted_steps_carry_the_error_flag_count_unparsed_and_feed_the_features():
    steps = [_step("tool_call", '[lookup_flight({"origin": "A"}) -> $1]'),
             _step("tool_call", '[calculator({"op": "+"}) -> Error: bad]', error=True),
             _step("tool_call", '{"name": "x"} {"name": "y"}'),
             _step("synthesis", "The grand total is $1.")]
    out, unparsed = adapt_steps(steps)
    assert unparsed == 1 and [len(s["tool_events"]) for s in out] == [1, 1, 0, 0]
    assert out[1]["tool_events"][0]["is_error"] is True and out[0]["tool_events"][0]["is_error"] is False
    assert all(s["text"] == o["text"] for s, o in zip(steps, out))  # the old fields are untouched
    table = prefix_feature_table({"r": out}, [2, 3])
    assert len(table) == 2  # the feature code accepts adapted steps (at most one event per step)


@pytest.mark.parametrize("text, expected, healthy", [
    ("The grand total for the trip is $1630.", 1630, True),
    ("Total: $1,630.00", 1630, True),
    ("The grand total is 1630.4", 1630, True),
    ("The grand total is $1631.", 1630, False),       # off by one is wrong
    ("I could not compute the total.", 1630, False),  # no number
    ("Flights $302 + $611 = $913; hotels unknown", 1630, False),
    ('{"name": "hotel_price", "parameters": {"nights": "2"}}', 2, True),  # the rule is literal: 2 is stated
])
def test_strict_label(text, expected, healthy):
    assert strict_label([_step("tool_call", "[f({}) -> 1]"), _step("synthesis", text)], expected) is healthy


def test_label_reads_the_last_synthesis_step_only():
    steps = [_step("synthesis", "Total so far $1630"), _step("tool_call", "[f({}) -> 1]"),
             _step("synthesis", "The total is $2000")]
    assert strict_label(steps, 1630) is False and strict_label(steps, 2000) is True
    assert strict_label([_step("tool_call", "[f({}) -> 1]")], 1) is False  # no synthesis step at all


def test_numbers_ignore_digits_inside_words():
    assert numbers_in("item-3 costs $1,200.50 in 2 nights; x9") == [3.0, 1200.5, 2.0]


def _raw():
    rows = []
    for corpus, k in (("organic_llama8b", 3), ("organic_llama8b_cold", 3), ("ollama_llama8b", 1)):
        for i in range(k):
            final = f"The grand total is ${100 + i}." if i == 0 else "No idea."
            steps = [_step("tool_call", '[f({"a": 1}) -> $5]')] * 4 + [_step("synthesis", final)]
            rows.append({"uid": f"{corpus}/r{i}", "episode_id": f"r{i}", "corpus": corpus, "model": "llama3.1:8b",
                         "failure_class": None, "tau": math.nan, "T": 5, "has_logprobs": True, "n_steps": 5,
                         "steps": json.dumps(steps), "metadata": json.dumps({"expected_total": 100 + i})})
    return pd.DataFrame(rows)


def test_load_organic_keeps_the_two_corpora_labels_runs_and_encodes_the_estimand():
    df = load_organic(_raw())
    assert set(df["corpus"]) == {"organic_llama8b", "organic_llama8b_cold"} and len(df) == 6
    healthy = df[df["healthy"]]
    assert len(healthy) == 2 and healthy["failure_class"].isna().all() and healthy["tau"].isna().all()
    failed = df[~df["healthy"]]
    assert (failed["failure_class"] == FAILED).all() and (failed["tau"] == 0.0).all()
    assert (df["family"] == "llama").all() and df["task_group"].is_unique
