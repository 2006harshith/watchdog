"""The paper's `esn_cusum_max` monitor (arXiv 2608.02464) behind a watchdog-shaped interface.

All modelling is the vendored author code; this file only wires it up exactly as the
author's make_hybrids() + run_model_transfer.py do (derail/monitor/hybrid.py:416 and
derail/experiments/run_model_transfer.py @ 1b3e07f):
  - features: adapter.episode_from_trace(extended=True), the published 51-dim view;
  - Standardizer fit on the fit runs; ChannelMaxESNMonitor(K=8, cusum=True, seed=1300)
    fit on the same runs;
  - alarm threshold: pick_threshold on healthy validation runs, 5% false-alarm budget;
  - episode score: max of the per-step stream (the author's episode_auc definition).
"""

from dataclasses import dataclass

import numpy as np

from ._vendor.adapter import episode_from_trace
from ._vendor.common import Episode, Standardizer
from ._vendor.esn import ChannelMaxESNMonitor
from ._vendor.thresholds import first_alarm, pick_threshold

# Author defaults for esn_cusum_max (make_hybrids signature and run_model_transfer.FA_BUDGET).
AUTHOR_K = 8
AUTHOR_SEED = 1300
AUTHOR_FA_BUDGET = 0.05
# The author drops "u" (surprisal) when < 90% of a corpus has logprobs; SP3's two corpora
# have 100%, so all four channel groups apply.
AUTHOR_CHANNELS = ("e", "u", "m", "x")


@dataclass(frozen=True)
class Run:
    uid: str
    steps: list[dict]
    failure_class: str | None = None  # None = healthy


@dataclass(frozen=True)
class RunScore:
    per_step: np.ndarray
    alarm_step: int | None
    episode_score: float


def _to_episode(run: Run) -> Episode:
    # Labels are never passed on: the vendored code only ever sees unlabelled runs, so no
    # label can leak into features or scores.
    return episode_from_trace(run.steps, run.uid, extended=True)


class ESNBaseline:
    def __init__(
        self,
        channels: tuple[str, ...] = AUTHOR_CHANNELS,
        K: int = AUTHOR_K,
        seed: int = AUTHOR_SEED,
        fa_budget: float = AUTHOR_FA_BUDGET,
    ):
        self.channels = tuple(channels)
        self.K = K
        self.seed = seed
        self.fa_budget = fa_budget
        self._monitor: ChannelMaxESNMonitor | None = None
        self.theta: float | None = None

    def fit(self, fit_runs: list[Run], val_runs: list[Run]) -> None:
        """One-class fit: both sets must be healthy. fit_runs train the standardiser and the
        ESN; val_runs only set the alarm threshold (as the author's train/val healthy splits).
        """
        labelled = [r.uid for r in (*fit_runs, *val_runs) if r.failure_class is not None]
        if labelled:
            raise ValueError(f"ESNBaseline.fit takes healthy runs only; got failed runs {labelled[:5]}")
        fit_eps = [_to_episode(r) for r in fit_runs]
        standardizer = Standardizer().fit(fit_eps)
        monitor = ChannelMaxESNMonitor(
            standardizer, channels=self.channels, K=self.K, cusum=True, seed=self.seed
        )
        monitor.fit(fit_eps)
        self._monitor = monitor
        val_streams = [monitor.score_episode(_to_episode(r)) for r in val_runs]
        self.theta = pick_threshold(val_streams, fa_budget=self.fa_budget)

    def score(self, run: Run) -> RunScore:
        if self._monitor is None or self.theta is None:
            raise RuntimeError("ESNBaseline.score called before fit")
        per_step = self._monitor.score_episode(_to_episode(run))
        return RunScore(per_step, first_alarm(per_step, self.theta), float(per_step.max()))
