import json

import pytest

from watchdog_agent.judges import (
    FORBIDDEN,
    AnthropicJudge,
    CacheKey,
    JudgeAPIError,
    JudgeCache,
    build_prompt,
    cost_latency,
    forbidden_hits,
    judge_samples,
    label_invariant,
    mean_score,
    parse_score,
    prompt_for_run,
    prompt_hash,
)

TEMPLATE = "Task: $task\n\nTrace so far:\n$trace\n\nAnswer with one integer from 0 to 100."
TASK = "You are a booking assistant. Task: price a trip from Cusco to Osaka. Report the total."


def _step(i, events=None, text=None):
    return {"text": text or f"[lookup_flight(...) -> ${100 + i}]", "token_logprobs": [-0.1, -0.2],
            "logprobs_available": True, "action": "tool_call" if events else "synthesis", "latency_s": 6.0 + i,
            "output_tokens": 18, "error": False, "task": TASK, "schema": 5, "tool_events": events or []}


def _run(**over):
    events = lambda i: [{"id": f"fc-{i}", "name": "lookup_flight", "args": {"origin": "Cusco", "destination": "Osaka"},
                         "result": f"${100 + i}", "result_chars": 4, "result_truncated": False, "is_error": i == 2,
                         "latency_s": 1e-5}]
    run = {"uid": "ollama_llama8b/real-healthy-000", "episode_id": "real-healthy-000", "corpus": "ollama_llama8b",
           "model": "llama3.1:8b", "failure_class": None, "tau": float("nan"), "metadata": '{"injection": {}}',
           "metadata_parsed": {"injection": {"kind": "context_corruption"}}, "family": "llama",
           "task_group": "text:" + TASK, "task_known": True, "has_logprobs": True,
           "steps_parsed": [_step(i, events(i)) for i in range(4)] + [_step(4, text="The total is $412.")]}
    return run | over


# ---------------------------------------------------------------- prompt and leakage


def test_prompt_holds_task_and_steps_up_to_t_only():
    p = prompt_for_run(TEMPLATE, _run(), 2)
    assert TASK in p and "Step 3:" in p and "Step 4:" not in p
    assert 'tool call: lookup_flight({"destination": "Osaka", "origin": "Cusco"})' in p  # args in sorted order
    assert "tool result (error): $102" in p and "tool result: $101" in p
    p4 = prompt_for_run(TEMPLATE, _run(), 4)
    assert "model: The total is $412." in p4  # a step without a tool call shows the model's text


def test_dollar_signs_in_trace_are_values_not_placeholders():
    p = build_prompt(TEMPLATE, "cost $task", [_step(0, text="$trace $1")], 0)
    assert "cost $task" in p and "model: $trace $1" in p


@pytest.mark.parametrize("t", [0, 2, 4])
def test_prompt_is_invariant_to_every_label_provenance_and_monitor_field(t):
    run = _run(failure_class="looping", tau=2.0, uid="ollama_llama8b/looping-007")
    assert label_invariant(TEMPLATE, run, t)
    other = _run(failure_class=None, tau=float("nan"), metadata="{}", family="qwen", model="qwen2.5:7b")
    assert prompt_for_run(TEMPLATE, run, t) == prompt_for_run(TEMPLATE, other, t)


def test_invariance_check_catches_a_leaky_builder():
    def leaky(template, run, t):
        return prompt_for_run(template, run, t) + f"\nrun id: {run['uid']}"

    def latency_leak(template, run, t):
        return prompt_for_run(template, run, t) + str(run["steps_parsed"][0]["latency_s"])

    assert not label_invariant(TEMPLATE, _run(), 2, builder=leaky)
    assert not label_invariant(TEMPLATE, _run(), 2, builder=latency_leak)


def test_forbidden_words_are_word_bounded_and_absent_from_the_template():
    assert forbidden_hits(TEMPLATE) == []
    assert forbidden_hits("the agent looks healthy") == ["healthy"]
    assert forbidden_hits("Goal drift happened") == ["goal drift"]
    assert forbidden_hits("unhealthy, taurus, Llamas") == []  # word boundaries: no partial-word hits
    assert {"looping", "healthy", "llama", "tau"} <= set(FORBIDDEN)


def test_prompt_hash_depends_on_the_template_only():
    assert prompt_hash(TEMPLATE) == prompt_hash(str(TEMPLATE))
    assert prompt_hash(TEMPLATE) != prompt_hash(TEMPLATE + " ")


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize("text, score", [
    ("42", 42), (" 7\n", 7), ("100", 100), ("0", 0), ("63.", 63),
    ("", None), (None, None), ("101", None), ("-3", None), ("4.5", None), ("42 or 43", None),
    ("Score: 42", None), ("forty", None), ("1000", None), ("42%", None),
])
def test_parse_score(text, score):
    assert parse_score(text) == score


def test_mean_score_ignores_unparsed_samples_and_counts_them():
    assert mean_score(["40", "60", "x"]) == (50.0, 1)
    assert mean_score(["x", "", "999"]) == (None, 3)


# ---------------------------------------------------------------- cache and client


def _ok(text="55", status=200):
    body = {"content": [{"type": "text", "text": text}], "model": "claude-haiku-5-5", "stop_reason": "end_turn",
            "usage": {"input_tokens": 300, "output_tokens": 2}}
    return status, json.dumps(body).encode()


class FakeTransport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, headers, body):
        self.calls.append((url, headers, json.loads(body)))
        return self.replies.pop(0)


def test_request_sends_thinking_off_and_no_sampling_parameters(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    fake = FakeTransport([_ok()])
    r = AnthropicJudge("claude-haiku-5-5", transport=fake).call("prompt")
    _, headers, body = fake.calls[0]
    assert body["thinking"] == {"type": "disabled"} and body["model"] == "claude-haiku-5-5"
    assert not {"temperature", "top_p", "top_k"} & set(body)  # rejected by API version 2026-10-21
    assert headers["anthropic-version"] == "2026-10-21" and headers["x-api-key"] == "test-key"
    assert r["text"] == "55" and r["input_tokens"] == 300 and r["output_tokens"] == 2 and r["model"] == "claude-haiku-5-5"


def test_retries_rate_limits_then_raises_on_client_errors(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    sleeps = []
    fake = FakeTransport([(429, b"{}"), (529, b"{}"), _ok("10")])
    r = AnthropicJudge("m", transport=fake, sleep=sleeps.append).call("p")
    assert r["text"] == "10" and r["attempts"] == 3 and sleeps == [2.0, 4.0]
    with pytest.raises(JudgeAPIError, match="HTTP 400"):
        AnthropicJudge("m", transport=FakeTransport([(400, b'{"error": "bad"}')]), sleep=sleeps.append).call("p")


def test_missing_api_key_fails_before_any_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake = FakeTransport([_ok()])
    with pytest.raises(JudgeAPIError, match="ANTHROPIC_API_KEY"):
        AnthropicJudge("m", transport=fake).call("p")
    assert fake.calls == []


def test_cache_spends_once_and_rereads_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    path = tmp_path / "cache.jsonl"
    fake = FakeTransport([_ok("20"), _ok("30"), _ok("40")])
    judge = AnthropicJudge("claude-haiku-5-5", transport=fake)
    prompt = "UNIQUE-PROMPT-TEXT about Cusco"
    first = judge_samples(judge, JudgeCache(path), "J-api", "h", "u1", 3, prompt, 3)
    assert [r["text"] for r in first] == ["20", "30", "40"] and len(fake.calls) == 3
    # A new cache object reads the file: no further calls, identical records.
    again = judge_samples(judge, JudgeCache(path), "J-api", "h", "u1", 3, prompt, 3)
    assert again == first and len(fake.calls) == 3
    assert "UNIQUE-PROMPT-TEXT" not in path.read_text()  # no prompt text stored


def test_cache_refuses_a_different_response_for_the_same_key(tmp_path):
    cache = JudgeCache(tmp_path / "c.jsonl")
    key = CacheKey("J-api", "m", "h", "u", 3, 0)
    cache.put(key, {"text": "1"})
    assert cache.put(key, {"text": "1"})["response"]["text"] == "1"
    with pytest.raises(ValueError, match="different response"):
        cache.put(key, {"text": "2"})


def test_cost_latency_from_recorded_tokens():
    rs = [{"latency_s": 1.0, "input_tokens": 1_000_000, "output_tokens": 0},
          {"latency_s": 3.0, "input_tokens": 0, "output_tokens": 1_000_000}]
    c = cost_latency(rs, 0.10, 0.50, calls_per_run=9)
    assert c["usd_per_call_mean"] == pytest.approx(0.30) and c["usd_per_1000_runs"] == pytest.approx(2700.0)
    assert c["latency_s_p50"] == pytest.approx(2.0) and c["calls"] == 2
    assert cost_latency([], 0.1, 0.5, 9) == {"calls": 0}
