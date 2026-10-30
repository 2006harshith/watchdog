"""SP9 Step 0: is the frozen monitors' llama signal latency? (configs/sp9_step0.yaml, pre-registered 11589d3).

L-1 is MS-raw-noLat on E2's llama setting (scored in the runner with E2's own code). L-2 refits
ESN-src's 30 D-6 fit draws with every step's latency_s replaced by a per-draw constant: done on the
step dict, not the signal vector, because the vendored adapter reads step latency twice (m's log
latency, and x's tool latency when a tool event has no measured latency of its own). L-3 describes
the < 50 ms llama steps, read-only.
"""

from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd

from watchdog_agent.baselines.esn.monitor import Run
from watchdog_agent.experiments.e2 import esn_channels
from watchdog_agent.experiments.e2_diagnostics import (
    _esn_running_max,
    _headline_mean,
    e2_healthy_pool,
    subset_draws,
)

FAST_S = 0.05


def with_constant_latency(steps: list[dict], c: float) -> list[dict]:
    """Copies of the steps with latency_s = c. tool_events (and their measured latency_s) are kept
    as they are: tool time is set by the environment, not by the model's generation."""
    return [{**s, "latency_s": c} for s in steps]


def step_latencies(runs: list[Run]) -> np.ndarray:
    vals = [s.get("latency_s") for r in runs for s in r.steps]
    if any(v is None for v in vals):
        # The adapter would silently read a missing latency as 1.0 s; a constant built on that
        # would not be the source's healthy median.
        raise ValueError(f"{sum(v is None for v in vals)} steps have no latency_s")
    return np.asarray(vals, dtype=float)


def constant_latency_getter(to_run: Callable[[str], Run], c: float) -> Callable[[str], Run]:
    def get(uid: str) -> Run:
        r = to_run(uid)
        return Run(r.uid, with_constant_latency(r.steps, c), r.failure_class)

    return get


class ReproductionFailed(RuntimeError):
    pass


def esn_constant_latency_draws(episodes, target_runs, family, e2_cfg, e3_cfg, make_esn, to_run,
                               expected_as_is=None, tol: float = 1e-9) -> dict:
    """E3's ESN-src fit draws (same pool, channels, subsets and val runs), each fitted and scored
    twice: as is, and with every step's latency_s set to the median latency of that draw's fit runs.
    With expected_as_is (D-6's AUROCs), a draw's as-is AUROC must match before its constant-latency
    model is fitted."""
    spec = e3_cfg["arms"]["ESN-src"]["fit_draws"][family]
    pool = e2_healthy_pool(episodes, family, e2_cfg, "i")
    if len(pool) != spec["pool_runs"]:
        raise ValueError(f"{family} ESN-src pool has {len(pool)} runs, config says {spec['pool_runs']}")
    channels = esn_channels(pool, target_runs, e2_cfg)
    fits = subset_draws(pool["uid"], spec["fit_size"], spec["n"], spec["seed"])
    # Every constant before any fit, so a missing latency fails in seconds, not after a refit.
    consts = [float(np.median(step_latencies([to_run(u) for u in fit]))) for fit in fits]
    as_is, const = [], []
    for i, (fit, c) in enumerate(zip(fits, consts)):
        val = sorted(set(pool["uid"]) - set(fit))
        for getter, out in ((to_run, as_is), (constant_latency_getter(to_run, c), const)):
            monitor = make_esn(channels)
            monitor.fit([getter(u) for u in fit], [getter(u) for u in val])
            out.append(_headline_mean(_esn_running_max(monitor, target_runs, getter), target_runs, e2_cfg))
            if out is as_is and expected_as_is is not None and abs(as_is[i] - expected_as_is[i]) > tol:
                raise ReproductionFailed(f"draw {i}: as-is AUROC {as_is[i]!r} != D-6 {expected_as_is[i]!r}")
    return {"channels": channels, "fits": fits, "constants": consts, "as_is": as_is, "constant": const}


def drop_summary(as_is, constant, threshold: float) -> dict:
    """Paired per-draw drop (as is - constant latency); its spread is fit-set variability only."""
    a, c = np.asarray(as_is, dtype=float), np.asarray(constant, dtype=float)
    d = a - c
    return {"as_is_median": float(np.median(a)), "constant_median": float(np.median(c)),
            "constant_min": float(c.min()), "constant_max": float(c.max()),
            "drop_median": float(np.median(d)), "drop_p10": float(np.percentile(d, 10)),
            "drop_p90": float(np.percentile(d, 90)), "drop_min": float(d.min()), "drop_max": float(d.max()),
            "share_drop_above_threshold": float((d > threshold).mean()), "per_draw_drop": d.tolist()}


def verdict(reference: float, value: float, max_drop: float) -> dict:
    return {"reference": reference, "value": value, "drop": reference - value,
            "outcome": "HOLDS" if reference - value <= max_drop else "FAILS"}


def latency_anatomy(runs: pd.DataFrame) -> dict:
    """L-3: where the < 50 ms steps sit (position, action, run label) and what they carry."""
    rows = []
    for healthy, steps in zip(runs["failure_class"].isna(), runs["steps_parsed"]):
        for i, s in enumerate(steps):
            events = s.get("tool_events") or []
            rows.append({"pos": i, "healthy": bool(healthy), "action": str(s.get("action")),
                         "has_latency": "latency_s" in s and s["latency_s"] is not None,
                         "latency": s.get("latency_s"), "tokens": s.get("output_tokens"),
                         "logprobs": bool(s.get("token_logprobs")), "n_events": len(events),
                         "event_latency": [e.get("latency_s") for e in events if isinstance(e, dict)],
                         "keys": tuple(sorted(s))})
    df = pd.DataFrame(rows)
    lat = df["latency"].astype(float)
    fast = lat < FAST_S

    def share_by(col: str) -> dict:
        g = df.assign(fast=fast).groupby(col)["fast"]
        return {str(k): {"steps": int(n), "share_fast": float(m)} for k, n, m in zip(g.size().index, g.size(), g.mean())}

    def profile(mask) -> dict:
        sub = df[mask]
        ev = [x for xs in sub["event_latency"] for x in xs]
        return {"steps": len(sub),
                "latency_q": np.percentile(sub["latency"].astype(float), [0, 10, 50, 90, 100]).tolist(),
                "output_tokens_q": np.percentile(sub["tokens"].astype(float), [10, 50, 90]).tolist(),
                "logprobs_share": float(sub["logprobs"].mean()),
                "tool_events_per_step_mean": float(sub["n_events"].mean()),
                "tool_event_latency_present_share": float(np.mean([x is not None for x in ev])) if ev else None,
                "tool_event_latency_q": (np.percentile([x for x in ev if x is not None], [10, 50, 90]).tolist()
                                         if any(x is not None for x in ev) else None),
                "key_sets": {", ".join(k): int(n) for k, n in Counter(sub["keys"]).most_common(5)}}

    return {"steps": len(df), "missing_latency": int((~df["has_latency"]).sum()),
            "share_fast": float(fast.mean()),
            "exact_values_among_fast": {f"{v:.6g}": int(n) for v, n in Counter(lat[fast].round(6)).most_common(5)},
            "by_position": share_by("pos"), "by_action": share_by("action"), "by_healthy": share_by("healthy"),
            "fast": profile(fast), "not_fast": profile(~fast)}
