"""SP8 Step 0d pilot: AgentDojo no-attack runs of llama3.1:8b via Ollama, recorded raw. NO monitor scores.

Run (Ollama running, llama3.1:8b pulled): uv run python pilot.py   (from envs/agentdojo)
Writes ../../data/agentdojo_pilot/pilot.jsonl (gitignored /data/) and pilot_tasks.json here (committed):
the pilot's user tasks, which are excluded from SP8 permanently.

The run itself is AgentDojo's own: its default pipeline (system message, query, LLM, tools loop), its
3-attempt retry when the model gives no output, and its utility check. The only change is the LLM
element: temperature 0.7, a seed per (run seed, call index), logprobs requested, and per-call timing.
"""

import json
import random
import time
import urllib.request
from pathlib import Path

import openai
from agentdojo.agent_pipeline import AgentPipeline, InitQuery, SystemMessage, ToolsExecutionLoop, ToolsExecutor
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms.openai_llm import (
    OpenAILLM,
    _function_to_openai,
    _message_to_openai,
    _openai_to_assistant_message,
)
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages

BENCHMARK = "v1.2.2"
MODEL = "llama3.1:8b"
OLLAMA = "http://localhost:11434"
TEMPERATURE = 0.7
SEEDS = [0, 1]
OUT = Path(__file__).resolve().parents[2] / "data" / "agentdojo_pilot" / "pilot.jsonl"
TASKS_FILE = Path(__file__).with_name("pilot_tasks.json")


class RecordingLLM(OpenAILLM):
    """OpenAILLM with a fixed temperature, a deterministic seed per call, logprobs, and a call log."""

    def __init__(self, client, model, run_seed: int):
        super().__init__(client, model, temperature=TEMPERATURE)
        self.run_seed, self.calls = run_seed, []

    def query(self, query, runtime, env=None, messages=(), extra_args={}):  # noqa: B006  (AgentDojo's signature)
        oa_messages = [_message_to_openai(m, self.model) for m in messages]
        tools = [_function_to_openai(t) for t in runtime.functions.values()]
        seed = self.run_seed * 1000 + len(self.calls)
        start = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model, messages=oa_messages, tools=tools or openai.NOT_GIVEN,
            tool_choice="auto" if tools else openai.NOT_GIVEN, temperature=self.temperature, seed=seed, logprobs=True)
        latency = time.perf_counter() - start
        choice = completion.choices[0]
        lp = getattr(choice, "logprobs", None)
        self.calls.append({
            "seed": seed, "latency_s": latency, "finish_reason": choice.finish_reason,
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
            "completion_tokens": getattr(completion.usage, "completion_tokens", None),
            "logprobs": [t.logprob for t in lp.content] if lp is not None and lp.content else None,
            "content": choice.message.content,
            "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                           for tc in (choice.message.tool_calls or [])],
        })
        output = _openai_to_assistant_message(choice.message)
        return query, runtime, env, [*messages, output], extra_args


class TimedToolsExecutor(ToolsExecutor):
    """ToolsExecutor that records each execution's wall time (tool latency, the ENV feature)."""

    def __init__(self, log: list):
        super().__init__()
        self.log = log

    def query(self, query, runtime, env=None, messages=(), extra_args={}):  # noqa: B006
        start = time.perf_counter()
        out = super().query(query, runtime, env, messages, extra_args)
        self.log.append({"after_message": len(messages), "latency_s": time.perf_counter() - start})
        return out


def pick_pilot_tasks(suites) -> list[tuple[str, str]]:
    """One user task per suite plus one more from workspace (the largest), seeded; set aside for good."""
    rng = random.Random(0)
    picked = []
    for name in sorted(suites):
        ids = sorted(suites[name].user_tasks, key=lambda s: int(s.rsplit("_", 1)[1]))
        picked.append((name, rng.choice(ids)))
    ws = [t for t in sorted(suites["workspace"].user_tasks, key=lambda s: int(s.rsplit("_", 1)[1]))
          if ("workspace", t) not in picked]
    picked.append(("workspace", rng.choice(ws)))
    return picked


def serialise(messages) -> list:
    out = []
    for m in messages:
        d = dict(m)
        d["content"] = [dict(c) for c in d["content"]] if isinstance(d.get("content"), list) else d.get("content")
        if d.get("tool_calls"):
            d["tool_calls"] = [{"id": tc.id, "function": tc.function, "args": dict(tc.args)} for tc in d["tool_calls"]]
        if d.get("tool_call"):
            tc = d["tool_call"]
            d["tool_call"] = {"id": tc.id, "function": tc.function, "args": dict(tc.args)}
        out.append(json.loads(json.dumps(d, default=str)))
    return out


def run_one(suite, task_id: str, seed: int, client) -> dict:
    """AgentDojo's run_task_with_pipeline for a no-attack run, keeping the trace it discards."""
    user_task = suite.get_user_task_by_id(task_id)
    llm, tool_log = RecordingLLM(client, MODEL, seed), []
    pipeline = AgentPipeline([SystemMessage(load_system_message(None)), InitQuery(), llm,
                              ToolsExecutionLoop([TimedToolsExecutor(tool_log), llm])])
    environment = suite.load_and_inject_default_environment({})
    task_env = user_task.init_environment(environment)
    pre_env = task_env.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    model_output, messages, attempts, start = None, [], 0, time.perf_counter()
    for _ in range(3):  # AgentDojo retries when the model produces no output
        attempts += 1
        llm.calls.clear()
        tool_log.clear()
        try:
            _, _, task_env, messages, _ = pipeline.query(user_task.PROMPT, runtime, task_env)
        except AbortAgentError as e:
            task_env, messages = e.task_environment, e.messages
        model_output = model_output_from_messages(messages)
        if model_output is not None:
            break
    utility = suite._check_task_result(user_task, model_output or [], pre_env, task_env,
                                       functions_stack_trace_from_messages(messages))
    return {"suite": suite.name, "user_task": task_id, "seed": seed, "attempts": attempts, "utility": bool(utility),
            "wall_s": time.perf_counter() - start, "prompt": user_task.PROMPT, "llm_calls": llm.calls,
            "tool_executions": tool_log, "messages": serialise(messages)}


def ollama_meta() -> dict:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=10) as r:
        tags = {m["name"]: m for m in json.loads(r.read())["models"]}
    with urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=10) as r:
        version = json.loads(r.read())["version"]
    m = tags[MODEL]
    return {"ollama_version": version, "model": MODEL, "digest": m["digest"],
            "quantization": m["details"].get("quantization_level"), "size_bytes": m["size"]}


def main() -> None:
    suites = get_suites(BENCHMARK)
    tasks = pick_pilot_tasks(suites)
    TASKS_FILE.write_text(json.dumps({"benchmark": BENCHMARK, "seed": 0, "excluded_from_sp8": tasks}, indent=2))
    meta = ollama_meta() | {"benchmark": BENCHMARK, "temperature": TEMPERATURE}
    client = openai.OpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama")  # Ollama ignores the key; the client needs one
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():  # resumable: one line per finished run
        done = {(r["suite"], r["user_task"], r["seed"]) for r in map(json.loads, OUT.read_text().splitlines())}
    for suite_name, task_id in tasks:
        for seed in SEEDS:
            if (suite_name, task_id, seed) in done:
                continue
            try:
                rec = run_one(suites[suite_name], task_id, seed, client)
            except Exception as e:  # a harness crash is excluded and counted, never relabelled
                rec = {"suite": suite_name, "user_task": task_id, "seed": seed, "crash": f"{type(e).__name__}: {e}"}
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec | {"meta": meta}, default=str) + "\n")
            print(suite_name, task_id, seed, "crash" if "crash" in rec else
                  f"utility={rec['utility']} calls={len(rec['llm_calls'])} wall={rec['wall_s']:.0f}s attempts={rec['attempts']}")


if __name__ == "__main__":
    main()
