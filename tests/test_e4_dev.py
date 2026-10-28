import copy
from pathlib import Path

import pandas as pd
import pytest
import yaml
from test_judges import _run

from watchdog_agent.experiments.e4_dev import (
    BudgetExceeded,
    load_template,
    preflight,
    run_version,
    select_dev_runs,
)
from watchdog_agent.judges import JudgeCache, forbidden_hits, prompt_hash

CFG = yaml.safe_load(Path("configs/e4_prompt_dev.yaml").read_text(encoding="utf-8"))
TEMPLATE = load_template("configs/e4_prompts/v1.txt")


def _episodes():
    rows = []
    for i in range(40):
        cls = [None, "looping", "goal_drift", "context_corruption"][i % 4]
        run = _run(uid=f"qwen/run-{i:02d}", family="qwen", failure_class=cls, tau=float("nan") if cls is None else 2.0)
        run["steps_parsed"] = copy.deepcopy(run["steps_parsed"])
        if cls is not None:  # a visible trace difference the fake judge keys on
            run["steps_parsed"][1]["tool_events"][0]["result"] = "ERROR-MARK"
        run["T"] = len(run["steps_parsed"])
        rows.append(run)
    rows.append(_run(uid="llama/run-x", family="llama") | {"T": 5})
    rows.append(_run(uid="qwen/short", family="qwen") | {"T": 3, "steps_parsed": _run()["steps_parsed"][:3]})
    return pd.DataFrame(rows)


def _cfg(**over):
    cfg = copy.deepcopy(CFG)
    cfg["dev_set"] |= {"healthy": 6, "per_class": 3}
    return cfg | over


class FakeJudge:
    model = "claude-haiku-5-5"

    def __init__(self, bad_text=None, tokens=(300, 2)):
        self.calls, self.bad_text, self.tokens = 0, bad_text, tokens

    def call(self, prompt):
        self.calls += 1
        text = "80" if "ERROR-MARK" in prompt else "20"
        if self.bad_text is not None and self.calls % 3 == 0:
            text = self.bad_text
        return {"text": text, "input_tokens": self.tokens[0], "output_tokens": self.tokens[1], "latency_s": 0.5,
                "stop_reason": "end_turn", "model": self.model}


def test_v1_template_is_clean_and_has_both_placeholders_once():
    assert forbidden_hits(TEMPLATE) == []
    assert TEMPLATE.count("$task") == 1 and TEMPLATE.count("$trace") == 1


def test_template_hash_ignores_crlf_checkouts(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes(TEMPLATE.replace("\n", "\r\n").encode("utf-8"))
    assert prompt_hash(load_template(p)) == prompt_hash(TEMPLATE)


def test_dev_set_is_qwen_only_stratified_and_seeded():
    ep = _episodes()
    runs = select_dev_runs(ep, _cfg()["dev_set"], "qwen")
    assert set(runs["family"]) == {"qwen"} and (runs["T"] >= 5).all()
    counts = runs["failure_class"].fillna("healthy").value_counts().to_dict()
    assert counts == {"healthy": 6, "looping": 3, "goal_drift": 3, "context_corruption": 3}
    assert runs["uid"].tolist() == select_dev_runs(ep, _cfg()["dev_set"], "qwen")["uid"].tolist()
    with pytest.raises(ValueError, match="needs"):
        select_dev_runs(ep, _cfg()["dev_set"] | {"per_class": 50}, "qwen")


def test_preflight_refuses_a_template_with_a_forbidden_word():
    runs = select_dev_runs(_episodes(), _cfg()["dev_set"], "qwen")
    assert len(preflight(TEMPLATE, runs, [2, 3, 4])) == 3 * len(runs)
    with pytest.raises(ValueError, match="forbidden"):
        preflight(TEMPLATE + "\nIs it healthy?", runs, [2, 3, 4])


def test_run_version_scores_caches_and_reruns_for_free(tmp_path):
    runs = select_dev_runs(_episodes(), _cfg()["dev_set"], "qwen")
    judge = FakeJudge()
    out = run_version(TEMPLATE, runs, judge, JudgeCache(tmp_path / "c.jsonl"), _cfg())
    assert out["calls"] == judge.calls == 3 * 3 * len(runs) and out["new_calls"] == out["calls"]
    assert out["headline_auroc"]["mean"] == 1.0 and out["parse_failures"] == 0
    assert out["sample_sd"]["share_zero"] == 1.0 and out["prompt_sha256"] == prompt_hash(TEMPLATE)
    again = run_version(TEMPLATE, runs, judge, JudgeCache(tmp_path / "c.jsonl"), _cfg())
    assert again["new_calls"] == 0 and judge.calls == out["calls"]
    assert again["headline_auroc"] == out["headline_auroc"]


def test_parse_failures_are_counted_not_scored(tmp_path):
    runs = select_dev_runs(_episodes(), _cfg()["dev_set"], "qwen")
    out = run_version(TEMPLATE, runs, FakeJudge(bad_text="maybe 40"), JudgeCache(tmp_path / "c.jsonl"), _cfg())
    assert out["parse_failures"] == out["calls"] // 3 and out["parse_failure_rate"] == pytest.approx(1 / 3)
    assert out["headline_auroc"]["mean"] == 1.0  # the mean of the two samples that parse


def test_budget_stops_spending_before_the_next_call(tmp_path):
    runs = select_dev_runs(_episodes(), _cfg()["dev_set"], "qwen")
    judge = FakeJudge(tokens=(1_000_000, 0))  # $0.10 per call
    with pytest.raises(BudgetExceeded):
        run_version(TEMPLATE, runs, judge, JudgeCache(tmp_path / "c.jsonl"), _cfg(budget_usd=0.25))
    assert judge.calls == 3
