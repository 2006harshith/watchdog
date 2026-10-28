"""SP8 core piece: the organic llama8b corpora as labelled episodes (run-level label, strict rule).

Design: docs/decisions.md 2026-10-28 (SP8, option 3). load_episodes() keeps excluding organic* corpora
(D1); this module is the one place they are read, labelled, and adapted to the current step schema.

- Label (strict): a run is HEALTHY iff its last synthesis step states metadata.expected_total (some
  number in the text within 0.5 of it); otherwise FAILED. The label is about the run as a whole: there
  is no onset, so a failed run is a positive at every checkpoint where it has a step. That is encoded
  as tau = 0, because evaluation counts a positive at t only if tau <= t. The estimand is therefore
  "will this run end wrong", read at matched steps, not "has it already gone wrong by t".
- Steps: the corpora use the author's old step schema (no tool_events, no task text). Each tool step's
  text "[name(args) -> result]" becomes one tool event; the step's own error flag is the event's
  is_error (as on the current schema); tool latency does not exist, so it is None (ENV features NaN).
- Task group = the run: the itineraries are random per run (236 distinct expected totals in 240 runs).
"""

import json
import math
import re

import pandas as pd
from huggingface_hub import hf_hub_download

from watchdog_agent.data import REPO_ID, model_to_family

CORPORA = ("organic_llama8b", "organic_llama8b_cold")
FAILED = "organic_failed"
_TOOL_TEXT = re.compile(r"^\[(\w+)\((.*)\) -> (.*)\]$", re.DOTALL)
# A number as the model writes it: optional $, thousands commas, optional decimals.
_NUMBER = re.compile(r"(?<![\w.])\$?\s*(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")


def parse_tool_text(text: str) -> dict | None:
    """One tool event from "[name(args) -> result]", or None if the text is not that shape."""
    m = _TOOL_TEXT.match(text)
    if m is None:
        return None
    try:
        args = json.loads(m.group(2))
    except json.JSONDecodeError:
        return None
    result = m.group(3)
    return {"name": m.group(1), "args": args, "result": result, "result_chars": len(result),
            "result_truncated": False, "latency_s": None}


def adapt_steps(steps: list[dict]) -> tuple[list[dict], int]:
    """Old-schema steps with tool_events added; and how many tool steps did not parse (no event)."""
    out, unparsed = [], 0
    for s in steps:
        events = []
        if s["action"] == "tool_call":
            e = parse_tool_text(s["text"])
            if e is None:
                unparsed += 1
            else:
                events = [e | {"is_error": bool(s["error"]), "id": None}]
        out.append(dict(s) | {"tool_events": events, "task": None, "schema": "organic-adapted"})
    return out, unparsed


def numbers_in(text: str) -> list[float]:
    vals = []
    for whole, frac in _NUMBER.findall(text or ""):
        vals.append(float(whole.replace(",", "") + ("." + frac if frac else "")))
    return vals


def final_synthesis_text(steps: list[dict]) -> str | None:
    syn = [s for s in steps if s["action"] == "synthesis"]
    return syn[-1]["text"] if syn else None


def strict_label(steps: list[dict], expected_total: float) -> bool:
    """True = healthy: the last synthesis step states the expected total (within 0.5)."""
    return any(abs(v - float(expected_total)) <= 0.5 for v in numbers_in(final_synthesis_text(steps) or ""))


def load_organic(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """The organic llama8b runs, labelled and adapted, with load_episodes()'s columns."""
    if raw is None:
        raw = pd.read_parquet(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="data/episodes.parquet"))
    df = raw[raw["corpus"].isin(CORPORA)].copy().reset_index(drop=True)
    df["metadata_parsed"] = df["metadata"].map(json.loads)
    adapted = df["steps"].map(lambda s: adapt_steps(json.loads(s)))
    df["steps_parsed"] = adapted.map(lambda x: x[0])
    df["unparsed_tool_steps"] = adapted.map(lambda x: x[1])
    df["healthy"] = [strict_label(st, md["expected_total"]) for st, md in zip(df["steps_parsed"], df["metadata_parsed"])]
    df["failure_class"] = [None if h else FAILED for h in df["healthy"]]
    df["tau"] = [math.nan if h else 0.0 for h in df["healthy"]]  # run-level label: positive from step 0
    df["family"] = df["model"].map(model_to_family)
    df["task_group"] = "run:" + df["uid"]
    df["task_known"] = False
    return df
