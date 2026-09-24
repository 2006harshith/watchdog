"""E0 — data audit for agent-trajectory-sentinel.

Downloads only the files the checks below need (the episodes table, the one
organic label CSV that exists, and the small per-corpus rejected/landing-failure
JSON files) into the Hugging Face cache, runs seven schema/leakage checks, and
writes results/e0_audit_v2.json. Nothing here trains a model.

Run: uv run python scripts/e0_data_audit.py
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

REPO_ID = "sunnydubey1111/agent-trajectory-sentinel"

# Found via the HF API's file listing (not guessed): every rejected.json /
# landing_failures.json in the dataset. Not every corpus has one of these.
LABEL_FILES = [
    "traces/autogen/rejected.json",
    "traces/autogen7b/rejected.json",
    "traces/autogen7b_real/landing_failures.json",
    "traces/autogen7b_real2/landing_failures.json",
    "traces/demo_real_varied/rejected.json",
    "traces/demo_real_varied_ext/rejected.json",
    "traces/langgraph/rejected.json",
    "traces/langgraph7b/rejected.json",
    "traces/langgraph7b_real2/landing_failures.json",
    "traces/ollama/rejected.json",
    "traces/ollama7b/rejected.json",
    "traces/ollama_llama8b/rejected.json",
    "traces/real/rejected.json",
    "traces/real_gemini_long/rejected.json",
    "traces/real_ollama7b/rejected.json",
    "traces/real_research3b/rejected.json",
    "traces/real_research7b/rejected.json",
]
# The only organic* corpus with a verified-label CSV in the dataset.
ORGANIC_LABELS_FILE = "traces/organic7b/organic_labels.csv"

# Found via HfApi.list_repo_files (not guessed): every non-jsonl file under an
# organic* corpus. All 8 have manifest.json; two also have collection_meta.json
# (aggregate collection params, not per-episode); organic7b alone also has
# organic_labels.csv (handled separately above) and review_sheet.txt (free text).
ORGANIC_MANIFEST_FILES = {
    "organic7b": ["manifest.json", "review_sheet.txt"],
    "organic_demo7b": ["manifest.json"],
    "organic_demo7b_cold": ["manifest.json"],
    "organic_demo7b_ext": ["manifest.json"],
    "organic_demo7b_holdout": ["manifest.json", "collection_meta.json"],
    "organic_demo7b_provoked": ["manifest.json"],
    "organic_llama8b": ["manifest.json"],
    "organic_llama8b_cold": ["manifest.json", "collection_meta.json"],
}


def counts_dict(s: pd.Series) -> dict:
    """value_counts() as a plain {str: int} dict, JSON-serialisable, NaN kept as 'none'."""
    vc = s.value_counts(dropna=False)
    return {("none" if pd.isna(k) else str(k)): int(v) for k, v in vc.items()}


def describe_dict(s: pd.Series) -> dict:
    return {k: float(v) for k, v in s.describe().to_dict().items()}


def load_episodes() -> pd.DataFrame:
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="data/episodes.parquet")
    df = pd.read_parquet(path)
    # steps/metadata are stored as JSON strings (per the dataset card); parse once, reuse everywhere.
    df["metadata_parsed"] = df["metadata"].map(json.loads)
    df["steps_parsed"] = df["steps"].map(json.loads)
    return df


def build_steps_df(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (episode, step) for the per-step stats in check 6."""
    records = []
    for episode_id, corpus, model, fc, steps in zip(
        df["episode_id"], df["corpus"], df["model"], df["failure_class"], df["steps_parsed"]
    ):
        for step in steps:
            records.append(
                (episode_id, corpus, model, fc, step.get("latency_s"), step.get("output_tokens"), step.get("error"))
            )
    return pd.DataFrame.from_records(
        records, columns=["episode_id", "corpus", "model", "failure_class", "latency_s", "output_tokens", "error"]
    )


def download_organic_labels() -> Path:
    return Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=ORGANIC_LABELS_FILE))


def download_label_files() -> dict[str, Path]:
    paths = {}
    for rel in LABEL_FILES:
        local = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=rel))
        corpus = rel.split("/")[1]
        paths.setdefault(corpus, {})[local.name] = local
    return paths


def download_organic_manifest_files() -> dict[str, dict[str, Path]]:
    paths: dict[str, dict[str, Path]] = {}
    for corpus, names in ORGANIC_MANIFEST_FILES.items():
        for name in names:
            local = Path(
                hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=f"traces/{corpus}/{name}")
            )
            paths.setdefault(corpus, {})[name] = local
    return paths


def check_corpus_model_failure(df: pd.DataFrame) -> dict:
    nested: dict = {}
    for (corpus, model), sub in df.groupby(["corpus", "model"], dropna=False):
        nested.setdefault(corpus, {})[model] = counts_dict(sub["failure_class"])
    return {
        "corpus_counts": counts_dict(df["corpus"]),
        "corpus_model_failure_class": nested,
        # data-quality note found during this audit, not one of the 7 asked-for checks:
        # some corpora ship with metadata == {} entirely.
        "rows_with_empty_metadata_by_corpus": counts_dict(
            df.loc[df["metadata_parsed"].map(lambda m: len(m) == 0), "corpus"]
        ),
    }


def check_organic_labels(df: pd.DataFrame, organic_labels_path: Path) -> dict:
    organic = df[df["corpus"].str.startswith("organic")]
    meta_keys: set[str] = set()
    for m in organic["metadata_parsed"]:
        meta_keys.update(m.keys())
    label_like_keys = sorted(k for k in meta_keys if "label" in k.lower() or "success" in k.lower())

    labels_df = pd.read_csv(organic_labels_path)
    verified_healthy_ids = set(labels_df.loc[labels_df["label"] == "healthy", "episode_id"])
    none_fc = df[df["failure_class"].isna()]
    not_verified_healthy = none_fc[~none_fc["episode_id"].isin(verified_healthy_ids)]

    return {
        "organic_row_count": len(organic),
        "metadata_keys_seen": sorted(meta_keys),
        "label_like_keys": label_like_keys,
        "success_value_counts": counts_dict(organic["metadata_parsed"].map(lambda m: m.get("success"))),
        "accepted_because_value_counts": counts_dict(
            organic["metadata_parsed"].map(lambda m: m.get("accepted_because"))
        ),
        "organic_labels_csv": {
            "columns": list(labels_df.columns),
            "row_count": len(labels_df),
            "label_value_counts": counts_dict(labels_df["label"]),
            "failure_mode_value_counts": counts_dict(labels_df["failure_mode"]),
        },
        "failure_class_none_total": len(none_fc),
        "failure_class_none_not_verified_healthy": len(not_verified_healthy),
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    # ~44% of rows have completely empty metadata ({}), not just a missing field
    # (confirmed by inspection, e.g. all of organic_demo7b_cold) — every metadata_parsed
    # lookup below has to tolerate that instead of indexing directly.
    task_sha = df["metadata_parsed"].map(lambda m: m.get("provenance", {}).get("task_sha256"))
    trace_sha = df["metadata_parsed"].map(lambda m: m.get("trace_sha256"))
    has_task_sha = task_sha.notna()
    has_trace_sha = trace_sha.notna()

    task_model = pd.DataFrame({"task_sha256": task_sha[has_task_sha], "model": df.loc[has_task_sha, "model"]})
    multi_model_count = int((task_model.groupby("task_sha256")["model"].nunique() > 1).sum())

    episode_corpus_count = df.groupby("episode_id")["corpus"].nunique()
    episode_multi_corpus_count = int((episode_corpus_count > 1).sum())

    duplicate_trace_count = int((trace_sha[has_trace_sha].value_counts() > 1).sum())

    return {
        "rows_missing_task_sha256": int((~has_task_sha).sum()),
        "rows_missing_trace_sha256": int((~has_trace_sha).sum()),
        "rows_per_task_sha256": describe_dict(task_sha[has_task_sha].value_counts()),
        "task_sha256_multi_model_count": multi_model_count,
        "episode_id_in_multiple_corpora_count": episode_multi_corpus_count,
        "duplicate_trace_sha256_value_count": duplicate_trace_count,
    }


def check_qwen_gap(df: pd.DataFrame) -> dict:
    qwen = df[df["model"].str.startswith("qwen")]
    return {
        "qwen_total_rows": len(qwen),
        "dataset_card_count": 2247,
        "gap": int(len(qwen) - 2247),
        "by_corpus": counts_dict(qwen["corpus"]),
    }


def check_tau(df: pd.DataFrame) -> dict:
    with_tau = df[df["tau"].notna()].copy()
    with_tau["tau_over_T"] = with_tau["tau"] / with_tau["T"]

    by_failure = {
        ("none" if pd.isna(fc) else fc): describe_dict(sub["tau_over_T"])
        for fc, sub in with_tau.groupby("failure_class", dropna=False)
    }

    healthy = df[df["failure_class"].isna()]
    failed = df[df["failure_class"].notna()]

    return {
        "tau_over_T_by_failure_class": by_failure,
        "share_tau_eq_2": float((with_tau["tau"] == 2).mean()),
        "T_distribution_healthy_by_corpus": {c: describe_dict(sub["T"]) for c, sub in healthy.groupby("corpus")},
        "T_distribution_failed_by_corpus": {c: describe_dict(sub["T"]) for c, sub in failed.groupby("corpus")},
    }


def check_logprobs_latency(df: pd.DataFrame, steps_df: pd.DataFrame) -> dict:
    has_logprobs_x_model = {model: counts_dict(sub["has_logprobs"]) for model, sub in df.groupby("model")}

    per_failure_class = {}
    for fc, sub in steps_df.groupby("failure_class", dropna=False):
        key = "none" if pd.isna(fc) else fc
        per_failure_class[key] = {
            "mean_latency_s": float(sub["latency_s"].mean()),
            "mean_output_tokens": float(sub["output_tokens"].mean()),
            "share_error": float(sub["error"].mean()),
        }
    return {
        "has_logprobs_x_model": has_logprobs_x_model,
        "per_failure_class_step_stats": per_failure_class,
    }


def check_injection_provenance(df: pd.DataFrame, label_file_paths: dict) -> dict:
    failed = df[df["failure_class"].notna()].copy()
    injection_nonempty = failed["metadata_parsed"].map(lambda m: bool(m.get("injection")))

    requested = failed["metadata_parsed"].map(lambda m: m.get("provenance", {}).get("requested_class"))
    has_requested = requested.notna()
    agreement = (
        float((requested[has_requested] == failed.loc[has_requested, "failure_class"]).mean())
        if has_requested.any()
        else None
    )

    label_file_sizes = {
        corpus: {name: path.stat().st_size for name, path in files.items()}
        for corpus, files in label_file_paths.items()
    }

    return {
        "share_failures_with_injection": float(injection_nonempty.mean()),
        "requested_class_coverage": int(has_requested.sum()),
        "requested_class_vs_failure_class_agreement": agreement,
        "label_file_sizes_bytes": label_file_sizes,
    }


def check_organic_manifest_labels(manifest_paths: dict[str, dict[str, Path]]) -> dict:
    per_corpus = {}
    for corpus, files in manifest_paths.items():
        entry: dict = {}
        if "manifest.json" in files:
            manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
            entry["manifest_type"] = type(manifest).__name__
            episode_keys = sorted(manifest[0].keys()) if manifest else []
            entry["manifest_episode_keys"] = episode_keys
            label_like_keys = sorted(k for k in episode_keys if "label" in k.lower() or "success" in k.lower())
            entry["label_like_keys"] = label_like_keys
            entry["manifest_row_count"] = len(manifest)
            entry["manifest_rows_with_failure_class"] = sum(1 for row in manifest if row.get("failure_class"))
            for key in label_like_keys:
                entry[f"manifest_rows_with_{key}"] = sum(1 for row in manifest if row.get(key))
        if "collection_meta.json" in files:
            meta = json.loads(files["collection_meta.json"].read_text(encoding="utf-8"))
            entry["collection_meta_keys"] = sorted(meta.keys())
        per_corpus[corpus] = entry
    return per_corpus


def normalise_task_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def check_derived_task_id(df: pd.DataFrame) -> dict:
    def first_task(steps: list) -> str | None:
        if not steps:
            return None
        return steps[0].get("task")

    task_text = df["steps_parsed"].map(first_task)
    has_task = task_text.notna()
    derived_id = task_text[has_task].map(lambda t: hashlib.sha256(normalise_task_text(t).encode()).hexdigest())

    task_sha = df["metadata_parsed"].map(lambda m: m.get("provenance", {}).get("task_sha256"))
    both = has_task & task_sha.notna()
    agree = None
    if both.any():
        derived_groups = df.loc[both, "episode_id"].groupby(derived_id[both]).apply(frozenset)
        sha_groups = df.loc[both, "episode_id"].groupby(task_sha[both]).apply(frozenset)
        agree = bool(set(derived_groups) == set(sha_groups))

    task_name = df["metadata_parsed"].map(lambda m: m.get("provenance", {}).get("task_name"))
    has_family = has_task & task_name.notna()
    ids_per_family = (
        pd.Series(derived_id[has_family].values, index=task_name[has_family].values)
        .groupby(level=0)
        .nunique()
        .describe()
    )

    ollama7b_ids = set(derived_id[has_task & (df.loc[has_task, "corpus"] == "ollama7b")])
    llama8b_ids = set(derived_id[has_task & (df.loc[has_task, "corpus"] == "ollama_llama8b")])

    return {
        "rows_with_no_task_text": int((~has_task).sum()),
        "distinct_derived_task_ids": int(derived_id.nunique()),
        "derived_id_vs_task_sha256_same_grouping": agree,
        "rows_with_both_ids": int(both.sum()),
        "distinct_derived_ids_per_task_name_family": describe_dict(ids_per_family),
        "derived_ids_shared_ollama7b_and_ollama_llama8b": len(ollama7b_ids & llama8b_ids),
        "derived_ids_ollama7b_total": len(ollama7b_ids),
        "derived_ids_ollama_llama8b_total": len(llama8b_ids),
    }


def check_ollama_comparison(df: pd.DataFrame) -> dict:
    result = {}
    for corpus in ["ollama7b", "ollama_llama8b"]:
        sub = df[df["corpus"] == corpus]
        tool_sets = sub["metadata_parsed"].map(lambda m: tuple(sorted(m.get("provenance", {}).get("tools", []))))
        result[corpus] = {
            "tool_roster_union": sorted({t for ts in tool_sets for t in ts}),
            "distinct_tool_rosters": tool_sets.nunique(),
            "failure_class_counts": counts_dict(sub["failure_class"]),
        }
    return result


def main() -> None:
    # HF_TOKEN isn't exported into the shell on Windows; load .env explicitly so
    # huggingface_hub (which reads HF_TOKEN from the environment) picks it up.
    load_dotenv()

    df = load_episodes()
    organic_labels_path = download_organic_labels()
    label_file_paths = download_label_files()
    organic_manifest_paths = download_organic_manifest_files()
    steps_df = build_steps_df(df)

    results = {
        "corpus_model_failure": check_corpus_model_failure(df),
        "organic_labels": check_organic_labels(df, organic_labels_path),
        "duplicates": check_duplicates(df),
        "qwen_gap": check_qwen_gap(df),
        "tau": check_tau(df),
        "logprobs_latency": check_logprobs_latency(df, steps_df),
        "injection_provenance": check_injection_provenance(df, label_file_paths),
        "organic_manifest_labels": check_organic_manifest_labels(organic_manifest_paths),
        "derived_task_id": check_derived_task_id(df),
        "ollama_comparison": check_ollama_comparison(df),
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "e0_audit_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
