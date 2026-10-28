from pathlib import Path

import pandas as pd
import pytest
import yaml

import watchdog_agent.experiments.e5_models as e5m
from watchdog_agent.experiments.e2 import feature_columns

E2 = yaml.safe_load(Path("configs/e2.yaml").read_text(encoding="utf-8"))


def test_nolat_drops_exactly_the_latency_aggregates():
    full, nolat = feature_columns("MS", E2), e5m.ms_nolat_columns(E2)
    dropped = sorted(set(full) - set(nolat))
    assert dropped == sorted(c for c in full if c.startswith(e5m.LATENCY_PREFIX)) and len(dropped) == 4
    assert [c for c in full if c in nolat] == nolat  # order kept


def test_nolat_refuses_if_no_latency_column_matches(monkeypatch):
    monkeypatch.setattr(e5m, "LATENCY_PREFIX", "no_such_prefix__")
    with pytest.raises(ValueError, match="no latency column"):
        e5m.ms_nolat_columns(E2)


def test_fingerprint_is_order_free_and_sensitive_to_any_score():
    s = pd.Series([0.1, 0.2, 0.3], index=["b", "a", "c"])
    assert e5m.fingerprint(s) == e5m.fingerprint(s.sort_index()) and e5m.fingerprint(s) != e5m.fingerprint(s + 1e-9)
