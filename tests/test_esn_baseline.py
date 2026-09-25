import numpy as np
import pytest

from watchdog_agent.baselines.esn import ESNBaseline, Run

T = 8
ANOMALY_FROM = 4
TOOLS = ("lookup_flight", "lookup_hotel", "calculator")


def _healthy_step(rng: np.random.Generator, t: int) -> dict:
    tool = TOOLS[t % len(TOOLS)]
    price = int(rng.integers(50, 500))
    return {
        "text": f'[{tool}({{"q": "item-{t}"}}) -> {{"price": {price}}}] Adding this to the total.',
        "token_logprobs": (-rng.exponential(0.3, size=30)).tolist(),
        "action": "tool_call",
        "latency_s": float(rng.uniform(0.8, 1.5)),
        "output_tokens": int(rng.integers(25, 40)),
        "error": False,
    }


def _broken_step(rng: np.random.Generator) -> dict:
    return {
        "text": "ERROR ERROR timeout ### retrying retrying ??? " + "x" * int(rng.integers(50, 80)),
        "token_logprobs": (-rng.exponential(5.0, size=300)).tolist(),
        "action": "tool_call",
        "latency_s": 40.0,
        "output_tokens": 400,
        "error": True,
    }


def _run(seed: int, broken: bool = False, failure_class: str | None = None) -> Run:
    rng = np.random.default_rng(seed)
    steps = [
        _broken_step(rng) if broken and t >= ANOMALY_FROM else _healthy_step(rng, t)
        for t in range(T)
    ]
    return Run(uid=f"run-{seed}", steps=steps, failure_class=failure_class)


def _fitted() -> ESNBaseline:
    baseline = ESNBaseline()
    baseline.fit([_run(s) for s in range(30)], [_run(s) for s in range(100, 120)])
    return baseline


@pytest.fixture(scope="module")
def baseline() -> ESNBaseline:
    return _fitted()


def test_fit_rejects_failed_runs():
    healthy = [_run(s) for s in range(5)]
    failed = _run(99, broken=True, failure_class="looping")
    with pytest.raises(ValueError, match="healthy runs only"):
        ESNBaseline().fit(healthy + [failed], healthy)
    with pytest.raises(ValueError, match="healthy runs only"):
        ESNBaseline().fit(healthy, healthy + [failed])


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        ESNBaseline().score(_run(0))


def test_score_is_deterministic(baseline):
    other = _fitted()
    run = _run(500, broken=True)
    assert other.theta == baseline.theta
    assert np.array_equal(other.score(run).per_step, baseline.score(run).per_step)


def test_obvious_anomaly_scores_above_normal_runs(baseline):
    normal = [baseline.score(_run(s)).episode_score for s in range(200, 210)]
    assert baseline.score(_run(300, broken=True)).episode_score > max(normal)


def test_washout_steps_score_zero_and_cusum_is_nonnegative(baseline):
    per_step = baseline.score(_run(301, broken=True)).per_step
    assert per_step.shape == (T,)
    assert np.all(per_step[:3] == 0.0)
    assert np.all(per_step >= 0.0)


def test_labels_do_not_change_scores(baseline):
    unlabelled = _run(302, broken=True)
    labelled = Run(unlabelled.uid, unlabelled.steps, failure_class="looping")
    assert np.array_equal(baseline.score(unlabelled).per_step, baseline.score(labelled).per_step)


def test_alarm_step_is_first_strict_crossing(baseline):
    result = baseline.score(_run(303, broken=True))
    assert result.alarm_step is not None
    assert result.per_step[result.alarm_step] > baseline.theta
    assert np.all(result.per_step[: result.alarm_step] <= baseline.theta)
    assert result.episode_score == result.per_step.max()
