"""E0 — data audit for agent-trajectory-sentinel.

Downloads only the files the checks below need (the episodes table, the one
organic label CSV that exists, and the small per-corpus rejected/landing-failure
JSON files) into the Hugging Face cache, runs seven schema/leakage checks, and
writes results/e0_audit_v2.json. Nothing here trains a model.

Run: uv run python scripts/e0_data_audit.py
"""

import json
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


def main() -> None:
    # HF_TOKEN isn't exported into the shell on Windows; load .env explicitly so
    # huggingface_hub (which reads HF_TOKEN from the environment) picks it up.
    load_dotenv()

    df = load_episodes()
    organic_labels_path = download_organic_labels()
    label_file_paths = download_label_files()
    steps_df = build_steps_df(df)

    results = {
        "corpus_model_failure": check_corpus_model_failure(df),
        "organic_labels": check_organic_labels(df, organic_labels_path),
        "duplicates": check_duplicates(df),
        "qwen_gap": check_qwen_gap(df),
        "tau": check_tau(df),
        "logprobs_latency": check_logprobs_latency(df, steps_df),
        "injection_provenance": check_injection_provenance(df, label_file_paths),
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "e0_audit_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
