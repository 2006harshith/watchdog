"""SP2 summary: rows/task-groups per family, unknown-task rows per corpus, and
train-run drop counts for each leave-one-family-out split.

Run: uv run python scripts/sp2_summary.py
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.splits import leave_one_family_out


def main() -> None:
    load_dotenv()
    df = load_episodes()

    rows_per_family = df["family"].value_counts().to_dict()
    task_groups_per_family = df.groupby("family")["task_group"].nunique().to_dict()
    unknown_rows_per_corpus = (
        df.loc[~df["task_known"]].groupby("corpus").size().to_dict()
    )

    overlap_drops = {}
    for family in sorted(df["family"].unique()):
        train_uids, test_uids, n_dropped = leave_one_family_out(df, test_family=family)
        overlap_drops[family] = {
            "train_runs": len(train_uids),
            "test_runs": len(test_uids),
            "train_runs_dropped_for_task_overlap": n_dropped,
        }

    results = {
        "rows_per_family": rows_per_family,
        "task_groups_per_family": task_groups_per_family,
        "unknown_task_rows_per_corpus": unknown_rows_per_corpus,
        "leave_one_family_out_overlap_drops": overlap_drops,
    }

    out_path = Path(__file__).resolve().parent.parent / "results" / "sp2_split_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
