"""SP7 Step 2: judge prompt development on qwen source runs (configs/e4_prompt_dev.yaml).

Only the source family is read, so the frozen prompt has never seen a llama trace. Every check that
guards Step 3 runs here first: the template has no forbidden word, every prompt is label-invariant,
and spending stops at a hard budget counted from the usage the API reports.
"""

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from watchdog_agent.evaluation import prefix_eval_set
from watchdog_agent.judges import (
    CacheKey,
    JudgeCache,
    cost_latency,
    forbidden_hits,
    label_invariant,
    mean_score,
    parse_score,
    prompt_for_run,
    prompt_hash,
)
from watchdog_agent.metrics import auroc


def load_template(path: Path | str) -> str:
    # Git on Windows checks text out with CRLF; the hash must not depend on the checkout.
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")


def select_dev_runs(episodes: pd.DataFrame, spec: dict, family: str) -> pd.DataFrame:
    """A seeded, stratified sample: healthy runs plus per_class runs of each headline class."""
    runs = episodes[(episodes["family"] == family) & (episodes["T"] >= spec["min_T"])]
    healthy = runs[runs["failure_class"].isna()]
    parts = [_sample(healthy, spec["healthy"], spec["seed"], 0)]
    for k, cls in enumerate(spec["classes"], start=1):
        pos = runs[(runs["failure_class"] == cls) & (runs["tau"] <= spec["max_tau"])]
        parts.append(_sample(pos, spec["per_class"], spec["seed"], k))
    return pd.concat(parts).sort_values("uid").reset_index(drop=True)


def _sample(df: pd.DataFrame, n: int, seed: int, stratum: int) -> pd.DataFrame:
    if len(df) < n:
        raise ValueError(f"stratum {stratum} has {len(df)} runs, needs {n}")
    uids = sorted(df["uid"])
    picked = np.random.default_rng([seed, stratum]).choice(uids, size=n, replace=False)
    return df[df["uid"].isin(set(picked))]


def spent_usd(cache: JudgeCache, price_in: float, price_out: float) -> float:
    return sum((r["response"]["input_tokens"] * price_in + r["response"]["output_tokens"] * price_out) / 1e6
               for r in cache.records.values())


class BudgetExceeded(RuntimeError):
    pass


def preflight(template: str, runs: pd.DataFrame, scored: Sequence[int]) -> list[tuple[str, int, str]]:
    """Every (uid, t, prompt) to send, after the leakage checks; raises before any call if one fails."""
    hits = forbidden_hits(template)
    if hits:
        raise ValueError(f"template contains forbidden words {hits}")
    out = []
    for _, run in runs.iterrows():
        row = run.to_dict()
        for t in scored:
            if run["T"] < t + 1:
                continue
            if not label_invariant(template, row, t):
                raise ValueError(f"prompt for {run['uid']} t={t} depends on a label field")
            out.append((run["uid"], t, prompt_for_run(template, row, t)))
    return out


def determinism_subset(n_prompts: int, spec: dict) -> set[int]:
    """Indices of the prompts that get extra samples, to measure whether a repeat call repeats."""
    k = min(spec["prompts"], n_prompts)
    return set(np.random.default_rng(spec["seed"]).choice(n_prompts, size=k, replace=False).tolist())


def run_version(template: str, runs: pd.DataFrame, judge, cache: JudgeCache, cfg: dict) -> dict:
    """Judge every dev prompt (cache first), stop at the budget, and summarise the version. Each
    prompt gets judge.samples scoring samples; the determinism subset gets more, used only for SD."""
    jc, price = cfg["judge"], cfg["price"]
    p_hash = prompt_hash(template)
    todo = preflight(template, runs, cfg["checkpoints"]["scored"])
    subset = determinism_subset(len(todo), jc["determinism_subset"])
    texts: dict[tuple[str, int], list[str]] = {}
    new = []
    for i, (uid, t, prompt) in enumerate(todo):
        got = []
        n = max(jc["samples"], jc["determinism_subset"]["samples"]) if i in subset else jc["samples"]
        for s in range(n):
            key = CacheKey(jc["name"], judge.model, p_hash, uid, t, s)
            rec = cache.get(key)
            if rec is None:
                if spent_usd(cache, price["input_usd_per_mtok"], price["output_usd_per_mtok"]) >= cfg["budget_usd"]:
                    raise BudgetExceeded(f"dev spend reached ${cfg['budget_usd']}; stopping before {uid} t={t}")
                rec = cache.put(key, judge.call(prompt))
                new.append(rec["response"])
            got.append(rec["response"])
        texts[(uid, t)] = got
    return summarise(runs, texts, cfg, p_hash, new)


def summarise(runs: pd.DataFrame, texts: dict, cfg: dict, p_hash: str, new: list) -> dict:
    report, k = cfg["checkpoints"]["report"], cfg["judge"]["samples"]
    scores = {uid: np.full(int(T), np.nan) for uid, T in zip(runs["uid"], runs["T"])}
    failures = calls = refusals = 0
    sds, identical = [], []
    for (uid, t), rs in texts.items():
        # Scores use the first k samples only, so a determinism-subset prompt is scored like the rest.
        m, bad = mean_score([r["text"] for r in rs[:k]])
        failures += bad
        calls += len(rs)
        refusals += sum(r.get("stop_reason") == "refusal" for r in rs)
        if m is not None:
            scores[uid][t] = m
        if len(rs) > k:
            vals = [parse_score(r["text"]) for r in rs]
            identical.append(len({r["text"] for r in rs}) == 1)
            ok_vals = [v for v in vals if v is not None]
            if len(ok_vals) >= 2:
                sds.append(float(np.std(ok_vals, ddof=1)))
    scoring_calls = k * len(texts)
    per_t, per_class = {}, {}
    for t in report:
        cs = prefix_eval_set(runs, t)
        s = np.array([scores[u][t] for u in cs.uids])
        ok = np.isfinite(s)
        per_t[str(t)] = {"auroc": float(auroc(cs.y[ok], s[ok])), "n_pos": int(cs.y[ok].sum()),
                         "n_neg": int((1 - cs.y[ok]).sum()), "missing": int((~ok).sum())}
        cls = runs.set_index("uid").loc[cs.uids, "failure_class"].to_numpy()
        for c in cfg["dev_set"]["classes"]:
            keep = ok & ((cs.y == 0) | (cls == c))
            per_class.setdefault(c, {})[str(t)] = float(auroc(cs.y[keep], s[keep]))
    price = cfg["price"]
    calls_per_run = calls / max(1, len(runs))
    return {"prompt_sha256": p_hash, "runs": len(runs), "prompts": len(texts), "calls": calls,
            "new_calls": len(new), "parse_failures": failures,
            "parse_failure_rate": failures / scoring_calls if scoring_calls else math.nan, "refusals": refusals,
            "headline_auroc": {"per_t": per_t, "mean": float(np.mean([v["auroc"] for v in per_t.values()]))},
            "per_class_auroc_mean": {c: float(np.mean(list(v.values()))) for c, v in per_class.items()},
            "determinism": {"prompts": len(identical),
                            "share_identical_text": float(np.mean(identical)) if identical else None,
                            "sd_median": float(np.median(sds)) if sds else None, "n_sd": len(sds)},
            "cost_latency_new_calls": cost_latency(new, price["input_usd_per_mtok"], price["output_usd_per_mtok"],
                                                   calls_per_run)}
