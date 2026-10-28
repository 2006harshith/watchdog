"""SP7 Step 2: judge one prompt version on the qwen dev set -> results/e4_prompt_dev.json.

Run: uv run python scripts/run_e4_prompt_dev.py v1   (template: configs/e4_prompts/v1.txt)
Spends money on uncached calls only (hard stop: configs/e4_prompt_dev.yaml budget_usd). Responses go to
results/e4_dev_cache.jsonl (committed: no trace text). Reads qwen runs only; never a llama run.
"""

import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from watchdog_agent.data import load_episodes
from watchdog_agent.experiments.e1 import git_state
from watchdog_agent.experiments.e4_dev import load_template, run_version, select_dev_runs, spent_usd
from watchdog_agent.judges import AnthropicJudge, JudgeCache

OUT = Path("results/e4_prompt_dev.json")


def main() -> int:
    version = sys.argv[1]
    load_dotenv()
    cfg = yaml.safe_load(Path("configs/e4_prompt_dev.yaml").read_text())
    if cfg["family"] != "qwen":
        raise ValueError("prompt development reads the source family (qwen) only")
    out = json.loads(OUT.read_text()) if OUT.exists() else {"versions": {}}
    if version not in out["versions"] and len(out["versions"]) >= cfg["max_versions"]:
        raise ValueError(f"already {len(out['versions'])} versions; the rule allows {cfg['max_versions']}")
    git = git_state()
    if git["dirty"]:
        print("WARNING: working tree is dirty; the result will carry git.dirty = true")

    template = load_template(Path(cfg["prompts_dir"]) / f"{version}.txt")
    runs = select_dev_runs(load_episodes(), cfg["dev_set"], cfg["family"])
    cache = JudgeCache(cfg["cache"])
    price = cfg["price"]
    print(f"{version}: {len(runs)} qwen dev runs; spent so far ${spent_usd(cache, price['input_usd_per_mtok'], price['output_usd_per_mtok']):.4f}")
    judge = AnthropicJudge(cfg["judge"]["model"], max_tokens=cfg["judge"]["max_tokens"])
    summary = run_version(template, runs, judge, cache, cfg) | {"git": git}
    out["versions"][version] = summary
    out["config"] = cfg
    OUT.write_text(json.dumps(out, indent=2))

    h = summary["headline_auroc"]
    print(f"{version}: calls {summary['calls']} (new {summary['new_calls']}), parse failures "
          f"{summary['parse_failures']} ({summary['parse_failure_rate']:.1%}), refusals {summary['refusals']}; "
          f"qwen headline AUROC {h['mean']:.3f} (t=3 {h['per_t']['3']['auroc']:.3f}, t=4 {h['per_t']['4']['auroc']:.3f}); "
          f"sample SD median {summary['sample_sd']['median']}")
    print(f"total dev spend ${spent_usd(cache, price['input_usd_per_mtok'], price['output_usd_per_mtok']):.4f}; wrote {OUT}")
    return 0 if summary["parse_failure_rate"] <= cfg["max_parse_failure_rate"] else 2


if __name__ == "__main__":
    sys.exit(main())
