"""SP8 E5 frozen source models: MS-raw (SP5, as is) and MS-raw-noLat, fitted before any SP8 score.

MS-raw-noLat is MS-raw's HGB config retrained on the same source rows without the latency features
(the four ms__log_latency__* aggregates). It is the pre-registered O-3 control: SP5 found MS-raw's
llama signal in the m channel (latency among it), and llama's source latencies (53-57% of steps under
50 ms) differ from the organic runs' (2.7-10 s), so a latency-driven score could be measuring the
collection, not the run.
"""

import hashlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from watchdog_agent.experiments.e2 import build_training_set, feature_columns, fit_model

LATENCY_PREFIX = "ms__log_latency__"


def ms_nolat_columns(e2_cfg: dict) -> list[str]:
    cols = [c for c in feature_columns("MS", e2_cfg) if not c.startswith(LATENCY_PREFIX)]
    if len(cols) == len(feature_columns("MS", e2_cfg)):
        raise ValueError("no latency column was dropped: the MS feature names changed")
    return cols


def fit_ms_models(table: pd.DataFrame, episodes: pd.DataFrame, e2_cfg: dict, make_model: Callable,
                  family: str = "llama") -> dict:
    """MS-raw and MS-raw-noLat on E2's training rows for the llama target (source pool, D6)."""
    train = build_training_set(table, episodes, family, e2_cfg)
    params = e2_cfg["model"]["primary"]
    return {"MS-raw": fit_model(train.X["raw"], train.y, train.w, feature_columns("MS", e2_cfg), params, make_model),
            "MS-raw-noLat": fit_model(train.X["raw"], train.y, train.w, ms_nolat_columns(e2_cfg), params, make_model)}


def fingerprint(scores: pd.Series) -> str:
    """sha256 of scores rounded to 12 decimals, in index order: the same model gives the same hash."""
    vals = np.round(scores.sort_index().to_numpy(dtype=float), 12)
    return hashlib.sha256(vals.tobytes()).hexdigest()
