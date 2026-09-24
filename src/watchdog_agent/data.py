"""SP2 data layer: load episodes, derive family/task_group, and build the per-step table.

Nothing here trains a model or defines learned features (SP4).
"""

import json
import re

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "sunnydubey1111/agent-trajectory-sentinel"

# D1 (docs/data_card.md): organic* corpora have no verified per-episode label
# (SP1 audit) and are excluded from SP2-SP6.
ORGANIC_PREFIX = "organic"

# Model -> family (D2). Unmapped models raise rather than silently falling into
# the wrong family, since family is the held-out unit for SP3/leave_one_family_out.
_FAMILY_PREFIXES = {
    "qwen2.5": "qwen",
    "llama3.1": "llama",
    "gemini-2.5": "gemini",
}


def model_to_family(model: str) -> str:
    for prefix, family in _FAMILY_PREFIXES.items():
        if model.startswith(prefix):
            return family
    raise ValueError(f"unmapped model family for model={model!r}")


def normalise_task_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _first_task_text(steps: list) -> str | None:
    if not steps:
        return None
    return steps[0].get("task")


class _UnionFind:
    """Plain-Python union-find: groups rows linked (transitively) by a shared
    task_sha256 or a shared normalised-task-text hash into one task_group.
    """

    def __init__(self):
        self._parent: dict[str, str] = {}

    def _find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self._parent[ra] = rb

    def group_of(self, x: str) -> str:
        return self._find(x)


def assign_task_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add task_group/task_known to a frame that already has metadata_parsed,
    steps_parsed, episode_id, corpus. Split out from load_episodes() so the
    union-find grouping (D3) can be unit-tested on small synthetic frames
    without hitting the network.
    """
    task_sha = df["metadata_parsed"].map(lambda m: m.get("provenance", {}).get("task_sha256"))
    task_text = df["steps_parsed"].map(_first_task_text)
    task_text_hash = task_text.map(lambda t: None if pd.isna(t) else normalise_task_text(t))

    uf = _UnionFind()
    task_known = pd.Series(False, index=df.index)
    for idx, sha, text_key in zip(df.index, task_sha, task_text_hash):
        # A row with a sha and/or a text key gets an edge from each key it has to
        # itself; union-find merges rows that share either key (SP2 spec D3:
        # "connected components over rows linked by the same task_sha256 OR the
        # same normalised task-text hash"). pd.isna, not `is None`: this
        # pandas build stores missing object values (dict.get() returning None)
        # as NaN, not Python None, once round-tripped through Series.map.
        keys = [k for k in (("sha", sha), ("text", text_key)) if not pd.isna(k[1])]
        if not keys:
            continue
        task_known.at[idx] = True
        first_key = f"{keys[0][0]}:{keys[0][1]}"
        for kind, value in keys[1:]:
            uf.union(first_key, f"{kind}:{value}")

    def resolve_group(idx, sha, text_key, episode_id, corpus) -> str:
        if not task_known.at[idx]:
            # Singleton per orphan row: never link unrelated rows just because
            # they lack task identity (confirmed with owner over the literal
            # "unknown:<corpus>" spec wording, which would have merged them).
            return f"unknown:{corpus}:{episode_id}"
        key = f"sha:{sha}" if not pd.isna(sha) else f"text:{text_key}"
        return uf.group_of(key)

    df = df.copy()
    df["task_group"] = [
        resolve_group(idx, sha, text_key, episode_id, corpus)
        for idx, sha, text_key, episode_id, corpus in zip(
            df.index, task_sha, task_text_hash, df["episode_id"], df["corpus"]
        )
    ]
    df["task_known"] = task_known.values
    return df


def load_episodes() -> pd.DataFrame:
    """Load episodes.parquet, drop organic* corpora, add family and task_group.

    Returns one row per episode (uid is the primary key) with the parsed
    metadata/steps still attached as *_parsed columns for to_step_table().
    """
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="data/episodes.parquet")
    df = pd.read_parquet(path)
    df["metadata_parsed"] = df["metadata"].map(json.loads)
    df["steps_parsed"] = df["steps"].map(json.loads)

    df = df[~df["corpus"].str.startswith(ORGANIC_PREFIX)].reset_index(drop=True)
    df["family"] = df["model"].map(model_to_family)
    df = assign_task_groups(df)

    return df


def _step_tool_names(step: dict) -> list[str]:
    return [event.get("name") for event in step.get("tool_events", [])]


def to_step_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """One row per (episode, step). No learned features here (SP4)."""
    records = []
    for row in episodes.itertuples(index=False):
        # failure_class/tau come back as NaN (float), not None, even though the
        # column dtype is pandas' native "str" — pd.notna is required, `is not
        # None` silently always passes.
        is_failed_run = pd.notna(row.failure_class)
        tau = row.tau
        for step_idx, step in enumerate(row.steps_parsed):
            step_label = bool(is_failed_run and pd.notna(tau) and step_idx >= tau)
            records.append(
                {
                    "uid": row.uid,
                    "corpus": row.corpus,
                    "model": row.model,
                    "family": row.family,
                    "task_group": row.task_group,
                    "task_known": row.task_known,
                    "step_idx": step_idx,
                    "T": row.T,
                    "is_failed_run": is_failed_run,
                    "failure_class": row.failure_class,
                    "tau": tau,
                    "step_label": step_label,
                    "action": step.get("action"),
                    "error": step.get("error"),
                    "latency_s": step.get("latency_s"),
                    "output_tokens": step.get("output_tokens"),
                    "logprobs_available": step.get("logprobs_available"),
                    "n_tool_events": len(step.get("tool_events", [])),
                    "tool_names": _step_tool_names(step),
                }
            )
    return pd.DataFrame.from_records(records)
