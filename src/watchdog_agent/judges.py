"""SP7 E4: an LLM judge read on the same trace prefixes as the monitors.

The judge sees a run's task text and its steps 0..t (model text, tool calls, tool results) and
answers one integer 0-100: how likely the run has already gone wrong. Design: docs/decisions.md
2026-10-28 (SP7): one judge, claude-haiku-5-5, thinking off, default sampling, 3 samples per (run, t).

- Leakage: the prompt is built from (task, steps) only. A run's labels and provenance (failure_class,
  tau, injection metadata, uid, corpus, model, family) never reach the builder, and
  label_invariant() checks it: scrambling every one of those fields must leave the prompt
  byte-identical. A word ban alone would not do, because qwen's own tool text contains "healthy"
  and "failure" in a few runs.
- Monitor-only step fields (latency, output tokens, logprobs) are left out too: they are the
  monitors' signals, and the judge is meant to read the trace as a person would.
- Money is spent once: every response is cached by (judge, model, prompt hash, uid, t, sample), and
  a rerun reads the cache instead of calling.
"""

import hashlib
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2026-10-21"  # this version rejects temperature/top_p/top_k, so none is sent
RETRY_STATUS = {429, 500, 502, 503, 504, 529}

# Labels, injection metadata and provenance: none may change a prompt (label_invariant).
LABEL_FIELDS = ("uid", "episode_id", "corpus", "model", "failure_class", "tau", "metadata",
                "metadata_parsed", "family", "task_group", "task_known", "has_logprobs")
# Monitor signals inside a step; the judge sees the trace content only.
MONITOR_STEP_FIELDS = ("latency_s", "output_tokens", "token_logprobs", "logprobs_available", "schema")
# Words that would leak a label or a family. The template must contain none; on llama no prompt
# contains any (checked on the data in Step 3). Word-bounded and case-insensitive.
FORBIDDEN = ("looping", "goal_drift", "goal drift", "context_corruption", "context corruption", "tool_cascade",
             "healthy", "inject", "injected", "injection", "tau", "llama", "qwen", "gemini", "ollama", "corpus")


def forbidden_hits(text: str) -> list[str]:
    low = text.lower()
    return [w for w in FORBIDDEN if re.search(r"\b" + re.escape(w) + r"\b", low)]


# ---------------------------------------------------------------- prompt


def render_trace(steps: Sequence[Mapping], t: int) -> str:
    """Steps 0..t as text: tool calls with their results, or the model's text for a step without one."""
    lines = []
    for i, step in enumerate(steps[: t + 1]):
        lines.append(f"Step {i + 1}:")
        events = step.get("tool_events") or []
        if not events:
            lines.append(f"  model: {step['text']}")
        for e in events:
            # sort_keys: the same call always renders the same way, so the prompt hash is stable.
            lines.append(f"  tool call: {e['name']}({json.dumps(e['args'], sort_keys=True)})")
            lines.append(f"  tool result{' (error)' if e.get('is_error') else ''}: {e['result']}")
    return "\n".join(lines)


def build_prompt(template: str, task: str, steps: Sequence[Mapping], t: int) -> str:
    # string.Template substitutes $task / $trace in the template only; a "$172" inside a tool result
    # is a value, not a placeholder, so trace text cannot break or alter the template.
    return string.Template(template).substitute(task=task, trace=render_trace(steps, t))


def prompt_for_run(template: str, run: Mapping, t: int) -> str:
    """The only bridge from a run row to a prompt: it reads the steps and nothing else."""
    steps = run["steps_parsed"]
    return build_prompt(template, steps[0]["task"], steps, t)


def label_invariant(template: str, run: Mapping, t: int,
                    builder: Callable[[str, Mapping, int], str] = prompt_for_run) -> bool:
    """True if scrambling every label, provenance and monitor field leaves the prompt unchanged."""
    junk = "goal_drift healthy injected llama tau=1 corpus"
    scrambled = dict(run) | {f: junk for f in LABEL_FIELDS if f in run}
    scrambled["steps_parsed"] = [dict(s) | {f: junk for f in MONITOR_STEP_FIELDS if f in s} for s in run["steps_parsed"]]
    return builder(template, run, t) == builder(template, scrambled, t)


def prompt_hash(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- parsing

_SCORE = re.compile(r"\s*(\d{1,3})\.?\s*")


def parse_score(text: str | None) -> int | None:
    """Exactly one integer 0-100 (surrounding whitespace and one trailing period allowed), else None."""
    m = _SCORE.fullmatch(text or "")
    if m is None:
        return None
    v = int(m.group(1))
    return v if v <= 100 else None


def mean_score(texts: Sequence[str]) -> tuple[float | None, int]:
    """Mean of the samples that parse, and how many did not. None only if none parses."""
    vals = [v for v in map(parse_score, texts) if v is not None]
    return (float(np.mean(vals)) if vals else None), len(texts) - len(vals)


# ---------------------------------------------------------------- cache


@dataclass(frozen=True)
class CacheKey:
    judge: str
    model: str  # the model id requested; the version the API reports is stored in the record
    prompt_hash: str
    uid: str
    t: int
    sample: int

    def as_tuple(self) -> tuple:
        return (self.judge, self.model, self.prompt_hash, self.uid, self.t, self.sample)


class JudgeCache:
    """Append-only JSON-lines file of judge responses. No prompt text is stored, only its hash."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.records: dict[tuple, dict] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[CacheKey(**rec["key"]).as_tuple()] = rec

    def get(self, key: CacheKey) -> dict | None:
        return self.records.get(key.as_tuple())

    def put(self, key: CacheKey, response: dict) -> dict:
        old = self.get(key)
        if old is not None:
            if old["response"]["text"] != response["text"]:
                raise ValueError(f"cache already holds a different response for {key}")
            return old
        rec = {"key": asdict(key), "response": response}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        self.records[key.as_tuple()] = rec
        return rec


# ---------------------------------------------------------------- API client

Transport = Callable[[str, dict, bytes], tuple[int, bytes]]


class JudgeAPIError(RuntimeError):
    pass


def urllib_transport(url: str, headers: dict, body: bytes, timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:  # 4xx/5xx still carry a JSON error body worth keeping
        return e.code, e.read()


class AnthropicJudge:
    """One Messages API call per sample, over plain HTTP (no SDK: one POST is all E4 needs)."""

    def __init__(self, model: str, max_tokens: int = 16, transport: Transport = urllib_transport,
                 max_retries: int = 4, backoff_s: float = 2.0, sleep: Callable[[float], None] = time.sleep):
        self.model, self.max_tokens, self.transport = model, max_tokens, transport
        self.max_retries, self.backoff_s, self.sleep = max_retries, backoff_s, sleep

    def body(self, prompt: str) -> dict:
        return {"model": self.model, "max_tokens": self.max_tokens, "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": prompt}]}

    def call(self, prompt: str) -> dict:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeAPIError("ANTHROPIC_API_KEY is not set (put it in .env, which is gitignored)")
        headers = {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}
        payload = json.dumps(self.body(prompt)).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            status, raw = self.transport(API_URL, headers, payload)
            latency = time.perf_counter() - start
            if status == 200:
                msg = json.loads(raw)
                text = "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")
                usage = msg.get("usage", {})
                return {"text": text, "model": msg.get("model"), "stop_reason": msg.get("stop_reason"),
                        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"),
                        "latency_s": latency, "attempts": attempt + 1,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if status not in RETRY_STATUS or attempt == self.max_retries:
                raise JudgeAPIError(f"HTTP {status}: {raw[:500].decode('utf-8', 'replace')}")
            self.sleep(self.backoff_s * 2 ** attempt)
        raise AssertionError("unreachable")


def judge_samples(judge, cache: JudgeCache, judge_name: str, p_hash: str, uid: str, t: int, prompt: str,
                  samples: int) -> list[dict]:
    """The responses for samples 0..samples-1, calling the API only for keys not yet cached."""
    out = []
    for s in range(samples):
        key = CacheKey(judge_name, judge.model, p_hash, uid, t, s)
        rec = cache.get(key)
        if rec is None:
            rec = cache.put(key, judge.call(prompt))
        out.append(rec["response"])
    return out


# ---------------------------------------------------------------- cost and latency


def cost_latency(responses: Sequence[dict], price_in_per_mtok: float, price_out_per_mtok: float,
                 calls_per_run: float) -> dict:
    """Per-call tokens and wall-clock, and $ per 1,000 runs at calls_per_run calls each."""
    if not responses:
        return {"calls": 0}
    lat = np.array([r["latency_s"] for r in responses], dtype=float)
    tin = np.array([r["input_tokens"] for r in responses], dtype=float)
    tout = np.array([r["output_tokens"] for r in responses], dtype=float)
    usd = (tin * price_in_per_mtok + tout * price_out_per_mtok) / 1e6
    return {"calls": len(responses), "latency_s_p50": float(np.percentile(lat, 50)),
            "latency_s_p95": float(np.percentile(lat, 95)), "input_tokens_mean": float(tin.mean()),
            "output_tokens_mean": float(tout.mean()), "usd_per_call_mean": float(usd.mean()),
            "calls_per_run": calls_per_run, "usd_per_1000_runs": float(usd.mean() * calls_per_run * 1000),
            "usd_total": float(usd.sum())}
