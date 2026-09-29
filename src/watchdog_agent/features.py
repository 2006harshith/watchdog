"""SP4 core piece: per-step features and their causal prefix aggregation.

Groups (docs/decisions.md, SP4 Step 0):
- MI, model-independent: the structure of the tool calls and of what the tools returned
  (repeats, new tools, errors, result size, time since the last error, step index).
- MS, model-specific: the model's own output and timing (latency, output tokens, length of the
  final-answer text, token surprisal).
- ENV: tool latency, which the environment sets, not the model.
- TEXT: the ESN's 32-dim char-3-gram hash of the step text (vendored `embed_text`, so the SP5
  ablation is like-for-like). On a tool step the text is "[name(args) -> result]", so this group
  mixes tool results with model output; it is kept apart from MS for that reason.

Dropped after Step 0 because they never vary in this data: args empty, args parse as JSON,
schema validation (`schema` is the trace-format version, always 5), empty result,
result_truncated. The step `error` flag equals the tool event's `is_error` on every step, so
only `is_error` is used.

Inputs are lists of step dicts, never the run row, so run-level fields (labels, corpus, model,
T, metadata) cannot reach a feature. The step- and event-level D9 fields are not read either
(tests/test_features.py checks both).

NaN policy: a continuous feature that does not apply to a step is NaN (e.g. surprisal without
logprobs, result size on a synthesis step), never a made-up 0. XGBoost (SP5) handles NaN
natively: each split learns a default branch for missing values ("sparsity-aware split
finding"), so missingness itself becomes usable signal, which is why surprisal (missing for every
gemini step) belongs to MS and not MI.
"""

import json
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn._vendor.adapter import embed_text

TEXT_DIM = 32

# name -> (group, kind). The kind decides the prefix aggregation (AGGREGATIONS).
STEP_FEATURES: dict[str, tuple[str, str]] = {
    "is_tool_call": ("MI", "binary"),
    "exact_repeat": ("MI", "binary"),
    "new_tool": ("MI", "binary"),
    "is_error": ("MI", "binary"),
    "result_error_prefix": ("MI", "binary"),
    "consecutive_identical": ("MI", "continuous"),
    "n_arg_keys": ("MI", "continuous"),
    "result_chars_bucket": ("MI", "continuous"),
    "steps_since_error": ("MI", "state"),
    "step_index": ("MI", "state"),
    "log_latency": ("MS", "continuous"),
    "log_output_tokens": ("MS", "continuous"),
    "synth_text_log_chars": ("MS", "continuous"),
    "surprisal_mean": ("MS", "continuous"),
    "surprisal_max": ("MS", "continuous"),
    "tool_log_latency": ("ENV", "continuous"),
    **{f"text_hash_{i:02d}": ("TEXT", "vector") for i in range(TEXT_DIM)},
}

AGGREGATIONS = {
    "binary": ("count", "rate", "last3"),
    "continuous": ("mean", "max", "last", "last3_mean"),
    # Already a summary of the whole history (or just the position): only its value at t means
    # anything; its mean over the prefix would mostly re-encode t.
    "state": ("last",),
    "vector": ("mean", "last"),
}

# Features defined on the prefix as a whole, not per step.
PREFIX_ONLY: dict[str, str] = {"distinct_tools_per_call": "MI"}


def _column(group: str, name: str, agg: str | None = None) -> str:
    return f"{group.lower()}__{name}" + (f"__{agg}" if agg else "")


def _build_groups() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"MI": [], "MS": [], "ENV": [], "TEXT": []}
    for name, (group, kind) in STEP_FEATURES.items():
        groups[group].extend(_column(group, name, agg) for agg in AGGREGATIONS[kind])
    for name, group in PREFIX_ONLY.items():
        groups[group].append(_column(group, name))
    return groups


FEATURE_GROUPS: dict[str, list[str]] = _build_groups()


def _event(step: dict) -> dict | None:
    events = step.get("tool_events") or []
    if len(events) > 1:
        # Step 0 found at most one tool event per step in all 14,648 steps; features like
        # "the step's call" assume it, so a trace that breaks it must not be silently truncated.
        raise ValueError(f"step has {len(events)} tool events; features assume at most one")
    return events[0] if events else None


def _call_key(event: dict | None) -> tuple[str, str] | None:
    if event is None:
        return None
    # sort_keys: {"x": 1, "y": 2} and {"y": 2, "x": 1} are the same call.
    return event.get("name"), json.dumps(event.get("args"), sort_keys=True)


def _log1p_or_nan(value) -> float:
    return math.nan if value is None else math.log1p(float(value))


def step_features(step: dict, history: Sequence[dict]) -> dict[str, float]:
    """Features of one step given the steps before it (history = steps 0..i-1, oldest first)."""
    event = _event(step)
    past_events = [_event(h) for h in history]
    key = _call_key(event)
    past_keys = [_call_key(e) for e in past_events]

    consecutive = 0
    if key is not None:
        consecutive = 1
        for k in reversed(past_keys):
            if k != key:
                break
            consecutive += 1

    is_error = int(event is not None and bool(event.get("is_error")))
    errors = [e is not None and bool(e.get("is_error")) for e in past_events] + [bool(is_error)]
    last_error = max((i for i, e in enumerate(errors) if e), default=-1)

    logprobs = step.get("token_logprobs") or []
    surprisal = -np.asarray(logprobs, dtype=float)
    result_chars = event.get("result_chars") if event is not None else None

    row: dict[str, float] = {
        "is_tool_call": int(event is not None),
        "exact_repeat": int(key is not None and key in past_keys),
        "new_tool": int(event is not None and event.get("name") not in {e.get("name") for e in past_events if e}),
        "is_error": is_error,
        # Read from the result text, not the flag: llama's own calculator failures return
        # "Error: invalid syntax" and injected errors return "Error: ...", and the two flags are
        # set by different code paths.
        "result_error_prefix": int(
            event is not None and str(event.get("result") or "").lstrip().lower().startswith("error")
        ),
        "consecutive_identical": consecutive,
        "n_arg_keys": len(event.get("args") or {}) if event is not None else math.nan,
        # Coarse log bucket: result size varies by orders of magnitude between tools (7 chars for
        # a price, thousands for a paper list); the bucket keeps the order, drops the detail.
        "result_chars_bucket": math.floor(math.log1p(result_chars)) if result_chars is not None else math.nan,
        # No error yet counts as one at step -1, so the value is always defined (i + 1).
        "steps_since_error": len(history) - last_error,
        "step_index": len(history),
        "log_latency": _log1p_or_nan(step.get("latency_s")),
        "log_output_tokens": _log1p_or_nan(step.get("output_tokens")),
        # Only a synthesis step's text is the model's own words; a tool step's text is mostly
        # the rendered tool result.
        "synth_text_log_chars": math.log1p(len(step.get("text") or "")) if event is None else math.nan,
        "surprisal_mean": float(surprisal.mean()) if surprisal.size else math.nan,
        # "min token surprisal" read as the least likely token (max -logprob, = min logprob): the
        # literal min is ~0 on almost every step. The author's ESN uses mean/max the same way.
        "surprisal_max": float(surprisal.max()) if surprisal.size else math.nan,
        "tool_log_latency": _log1p_or_nan(event.get("latency_s")) if event is not None else math.nan,
    }
    emb = embed_text(str(step.get("text") or ""))
    row.update({f"text_hash_{i:02d}": float(emb[i]) for i in range(TEXT_DIM)})
    return row


def step_feature_rows(steps: Sequence[dict]) -> list[dict[str, float]]:
    """step_features for every step. Row i depends only on steps 0..i."""
    return [step_features(step, steps[:i]) for i, step in enumerate(steps)]


def _nanmean(values: np.ndarray) -> float:
    # np.nanmean on an all-NaN slice returns NaN but warns; the all-NaN window is expected here
    # (e.g. no synthesis step yet), so it is handled explicitly.
    return math.nan if np.isnan(values).all() else float(np.nanmean(values))


def _nanmax(values: np.ndarray) -> float:
    return math.nan if np.isnan(values).all() else float(np.nanmax(values))


def _aggregate(rows: Sequence[dict[str, float]], t: int) -> dict[str, float]:
    if len(rows) != t + 1:
        raise ValueError(f"need the rows of steps 0..{t}, got {len(rows)}")
    out: dict[str, float] = {}
    for name, (group, kind) in STEP_FEATURES.items():
        v = np.array([r[name] for r in rows], dtype=float)
        if kind == "binary":
            vals = {"count": v.sum(), "rate": v.sum() / (t + 1), "last3": v[-3:].sum()}
        elif kind == "continuous":
            vals = {"mean": _nanmean(v), "max": _nanmax(v), "last": v[-1], "last3_mean": _nanmean(v[-3:])}
        elif kind == "state":
            vals = {"last": v[-1]}
        else:  # vector
            vals = {"mean": v.mean(), "last": v[-1]}
        for agg in AGGREGATIONS[kind]:
            out[_column(group, name, agg)] = float(vals[agg])
    n_calls = sum(r["is_tool_call"] for r in rows)
    n_distinct = sum(r["new_tool"] for r in rows)
    out[_column("MI", "distinct_tools_per_call")] = n_distinct / n_calls if n_calls else math.nan
    return out


def prefix_features(steps: Sequence[dict], t: int) -> dict[str, float]:
    """One feature row for a run observed up to and including step t (steps 0..t only)."""
    if not 0 <= t < len(steps):
        raise ValueError(f"run has {len(steps)} steps; no step t={t}")
    # Slicing before computing anything makes it impossible for a later step to reach a feature.
    return _aggregate(step_feature_rows(steps[: t + 1]), t)


def prefix_feature_table(steps_by_uid: Mapping[str, Sequence[dict]], checkpoints: Sequence[int]) -> pd.DataFrame:
    """Rows indexed by (uid, t) for every run that has a step t; runs without one are skipped."""
    records, index = [], []
    for uid, steps in steps_by_uid.items():
        # Computed once per run and sliced per checkpoint: row i depends only on steps 0..i, so
        # rows[: t + 1] equals recomputing on steps[: t + 1] (tested against prefix_features).
        rows = step_feature_rows(steps)
        for t in checkpoints:
            if t < len(steps):
                records.append(_aggregate(rows[: t + 1], t))
                index.append((uid, t))
    columns = [c for names in FEATURE_GROUPS.values() for c in names]
    return pd.DataFrame.from_records(
        records, index=pd.MultiIndex.from_tuples(index, names=["uid", "t"]), columns=columns
    )
