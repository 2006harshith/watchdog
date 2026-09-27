"""SP3 diagnostics D-1..D-3 (+ D-4 comparison) -> results/e1_diagnostics.json.

Run: uv run python scripts/run_e1_diagnostics.py [--clean-e1 <e1_collapse.json from a clean commit>]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.author_protocol import author_split, load_corpus
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e1_diagnostics import (
    cross_corpus_twins,
    overlap_counts,
    source_overlap_a,
    split_ablation_b,
    trajectory_hash,
    twin_audit,
)

SOURCE, TARGET = "ollama7b", "ollama_llama8b"
N_SEEDS = 20
D3_N_FIT, D3_N_VAL = 20, 5
E1_RESULTS = Path("results/e1_collapse.json")
OUT = Path("results/e1_diagnostics.json")


def compare_clean_e1(clean_path: Path) -> dict:
    clean = json.loads(clean_path.read_text())
    current = json.loads(E1_RESULTS.read_text())
    clean_git = clean.pop("git")
    current.pop("git")
    return {
        "clean_run_git": clean_git,
        "identical_to_results_e1_collapse_excluding_git": clean == current,
        "differing_top_level_keys": sorted(k for k in clean.keys() | current.keys()
                                           if clean.get(k) != current.get(k)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-e1", type=Path, help="e1_collapse.json produced from a clean commit")
    args = parser.parse_args()
    load_dotenv()
    git = git_state()
    episodes = load_episodes()
    hash_of = dict(zip(episodes["uid"], episodes["steps_parsed"].map(trajectory_hash)))
    group_of = dict(zip(episodes["uid"], episodes["task_group"]))

    tgt_runs, _, _ = load_corpus(episodes, TARGET)
    src_runs, _, src_channels = load_corpus(episodes, SOURCE)
    tgt_fit, tgt_val, tgt_test = author_split(tgt_runs, 0)
    src_fit, _, _ = author_split(src_runs, 0)
    test_uids = [r.uid for r in tgt_test]

    result = {
        "what": "SP3 diagnostics: twins, split ablation (B), source task overlap (A), clean rerun",
        "git": git,
        "d1_twins": {
            "definition": "sha256 of per-step tool calls (name, args, result); text/timing ignored",
            "per_corpus": twin_audit(episodes),
            "cross_corpus": cross_corpus_twins(episodes, SOURCE, TARGET),
            "author_in_domain_split": {
                "vs_fit": overlap_counts(test_uids, [r.uid for r in tgt_fit], hash_of, group_of),
                "vs_fit_and_val": overlap_counts(
                    test_uids, [r.uid for r in tgt_fit + tgt_val], hash_of, group_of
                ),
            },
            "author_transfer_split_vs_qwen_fit": overlap_counts(
                test_uids, [r.uid for r in src_fit], hash_of, group_of
            ),
        },
    }
    with warnings.catch_warnings():
        # pick_threshold warns that <19 validation runs cannot reach a 5% budget; only AUROC,
        # which does not use the threshold, is reported here.
        warnings.simplefilter("ignore", RuntimeWarning)
        result["d2_split_ablation_B"] = split_ablation_b(tgt_runs, hash_of, group_of, N_SEEDS)
        result["d3_source_overlap_A"] = source_overlap_a(
            src_runs, tgt_runs, group_of, hash_of, src_channels, D3_N_FIT, D3_N_VAL, N_SEEDS
        )
    if args.clean_e1:
        result["d4_clean_rerun"] = compare_clean_e1(args.clean_e1)
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    d1 = result["d1_twins"]
    print("D-1 per corpus (runs / in twin clusters / clusters):",
          {c: (v["runs"], v["runs_in_twin_clusters"], v["twin_clusters"]) for c, v in d1["per_corpus"].items()
           if c in (SOURCE, TARGET)})
    print("D-1 author in-domain test vs fit:", d1["author_in_domain_split"]["vs_fit"])
    for name in ("random", "twin_aware", "task_disjoint"):
        s = result["d2_split_ablation_B"][name]
        print(f"D-2 B {name:>13}: AUROC mean {s['mean']:.3f} sd {s['sd']:.3f} [{s['min']:.3f}, {s['max']:.3f}]")
    for name in ("overlap", "excluded"):
        s = result["d3_source_overlap_A"][name]
        print(f"D-3 A {name:>8}: AUROC mean {s['mean']:.3f} sd {s['sd']:.3f} [{s['min']:.3f}, {s['max']:.3f}]")
    if "d4_clean_rerun" in result:
        print("D-4:", result["d4_clean_rerun"])
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
