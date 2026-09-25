# Vendored from github.com/sunnydubey1111/agent-trajectory-sentinel
# commit 1b3e07fee53ae13407173c3ea932adb4a43e8230, file derail/telemetry/goal_consistency.py
# Copyright 2026 Sunny Dubey. Apache License 2.0: see LICENSE and NOTICE in this directory.
# Modified by watchdog (2026-09-25): `derail.*` imports rewritten as relative imports and this
# header added. No other change.
"""Goal-consistency telemetry: is the agent still working on the task it was given?

The behavioural channels answer "is this step unusual". They cannot answer
"is this step still about the goal", because nothing in `e`, `u`, `m` or `x`
compares the trajectory to the task. A run that switches to a *different*
topic and pursues it competently produces perfectly ordinary latencies, action
types and step-to-step embedding drift -- which is exactly why the framework
corpora contain goal-drift episodes whose entire interior sits inside the
healthy null (`results/framework_miss_forensics.md`).

The signal here is a provenance question about each action the agent takes:

    where did the content of this action come from -- the goal, or the
    environment?

An agent doing its job draws the substance of its tool calls and its prose
from the task it was given. An agent that has been steered draws it from
something it read along the way. That framing is what makes the signal
general: it names no framework, no failure class and no topic, and it works
against any task that arrives as text. A prompt-injection hijack is caught not
because the payload is recognised -- it never is -- but because the agent
starts sourcing its actions from tool output that the goal does not mention.

Two surfaces, because the corpora show two distinct derailment modes:

  * the arguments the agent SENDS (its actions). Action-level drift: the
    agent's tool calls themselves go off-goal.
  * the prose the agent WRITES (its reasoning and its answer). Answer-level
    contamination: the calls stay on-goal but the output absorbs injected
    material.

Two quantities per surface, because leaving the goal and being captured by the
environment are different events and either can happen alone:

  * abandonment -- the share of this action's content that the goal does NOT
    account for;
  * environment echo -- the share that the goal does not account for but
    earlier tool OUTPUT does.

Every dim is in [0, 1] and oriented so HIGHER = more anomalous, matching the
grounding channel's convention. Everything is computed from steps already
seen, so the value at step t never depends on step t+1. Cost is set algebra
over a few hundred tokens per step: no model, no embedding, no second judge.

Deliberately NOT used: the tool RESULT at the current step. The payload lands
in the observation, but landing is the environment's doing; only the agent's
next action shows whether it was adopted. Reading the current result would
fire on every episode where a hijack arrived and was correctly ignored -- five
of the twelve framework goal-drift episodes -- which is a false alarm, not a
detection.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .events import parse_step_events

#: Content tokens: lowercase alphanumeric runs of 3+ characters. Short tokens
#: are dropped with the closed class below rather than kept, because at length
#: 1-2 the overlap is dominated by units and list markers shared by every task.
_WORD = re.compile(r"[a-z0-9]+")

#: Generic English function words. Closed-class only -- no topic, domain or
#: benchmark term appears here, and removing it entirely changes the direction
#: of no measurement, only its contrast: a long task statement contains most
#: function words, so leaving them in inflates every overlap toward 1.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its into about over under not no
you your we our they their he she his her i me my do does did done have has
had will would can could should may might must shall use used using please
step steps each all both any some more most other another such only own same
so too very just now new next first second third also
""".split())


def content_tokens(text: object) -> frozenset[str]:
    """The content vocabulary of a piece of text. Deterministic, order-free."""
    return frozenset(w for w in _WORD.findall(str(text).lower())
                     if len(w) >= 3 and w not in _STOPWORDS)


def action_tokens(events) -> frozenset[str]:
    """What the agent SENT this step: tool names and argument values.

    The name is split on underscores so `wikipedia_search` contributes
    `wikipedia` and `search` rather than one opaque token; argument values are
    serialised whole, so a query string contributes its own words.
    """
    out: set[str] = set()
    for ev in events:
        out |= content_tokens(ev.name.replace("_", " "))
        out |= content_tokens(json.dumps(ev.args, sort_keys=True))
    return frozenset(out)


def _shares(tokens: frozenset[str], goal: frozenset[str],
            seen: frozenset[str]) -> tuple[float, float]:
    """(abandonment, environment echo) for one surface at one step.

    Abandonment counts everything the goal does not account for; echo counts
    the part of that which earlier tool output does. Echo is therefore a
    subset of abandonment by construction, and the pair is scale-free: it
    measures composition, not volume, so a verbose step and a terse one on the
    same subject score the same.
    """
    n = len(tokens)
    if n == 0:
        return 0.0, 0.0
    off_goal = tokens - goal
    return len(off_goal) / n, len(off_goal & seen) / n


class GoalConsistencyState:
    """Causal per-episode state. One per episode, fed in step order.

    Holds the goal vocabulary, the tool output seen so far, and the last value
    of each surface. The hold matters: a synthesis step issues no tool call,
    and letting the argument dims collapse to a default there would make "the
    run finished" look like "the run went off-goal" -- measured on the
    framework corpora, that artifact alone puts healthy episodes BELOW drifted
    ones on argument anchoring, which is the opposite of the truth.
    """

    def __init__(self, goal_text: object) -> None:
        self.goal = content_tokens(goal_text)
        self.seen: set[str] = set()
        #: Nothing has been issued yet, so nothing has been abandoned. Starting
        #: at 0 rather than at a midpoint keeps the opening steps of a healthy
        #: run and a drifted run identical, which is what they are.
        self.last = np.zeros(4)

    @property
    def inert(self) -> bool:
        """True when no goal text was recorded, so nothing can be measured.

        The channel then emits zeros for the whole episode, the same way the
        grounding channel goes inert on traces with no recorded tool results.
        Silence is the honest reading: an absent goal is not a consistent one.
        """
        return not self.goal


def goal_features(step: dict, state: GoalConsistencyState) -> np.ndarray:
    """One step -> the 4-dim goal-consistency vector. Mutates `state`.

    Order is the whole causality argument: the features are read against tool
    output from steps STRICTLY BEFORE this one, and this step's own results
    join that pool only afterwards.
    """
    events, prose = parse_step_events(step)
    if state.inert:
        return np.zeros(4)

    seen = frozenset(state.seen)
    values = state.last.copy()
    sent = action_tokens(events)
    if sent:
        values[0], values[1] = _shares(sent, state.goal, seen)
    written = content_tokens(prose)
    if written:
        values[2], values[3] = _shares(written, state.goal, seen)
    state.last = values

    for ev in events:
        state.seen |= content_tokens(ev.result)
    return values
