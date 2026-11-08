"""chameleon - balanced by default, a counter-strategy once the opponent shows a clear leak.

Chops come from real but small edges. The opponents we only chop with share
one of two leaks, measured with OppStats against agent.py:
  * passive callers (eq_shade, eq_tight, trapper, adaptive): almost never
    re-raise preflop, rarely bet, and fold to bets LESS than a solver;
  * over-folders (nit, cbet, blinder, shover): fold to postflop bets 66-85%.
For each leak there is a counter-strategy table: a best response (train.c
BR_VS, 150M iterations) to strategy.npy distorted to fold x0.7 / raise x0.3
(practice/tables/counter_passive.npy) or to fold x1.6 (counter_folder.npy).

Until MIN_SEEN observations of each kind, and whenever the opponent looks
solver-like, it plays the shipped strategy, so strong opponents get nothing
extra to exploit. Leaving a mode takes a little more evidence than entering
it, so it doesn't flicker.

RESULT (2026-11-07, 6 seeds vs 12 opponents): worse than agent.py, 61.5 vs
71.5 points. A best response to an approximate model ("our strategy,
distorted") is brittle against the real bots, and the pure counter tables
lose to the exploiter 11 of 12. Kept as a documented negative result; ROLE is
"experimental" so default runs skip it. (train.c's restricted-response mode
was tried first and changed almost nothing: its opponent model is a best
response, mostly pure, and multiplicative distortion of a pure action is a
no-op.)
"""

import os
import sys

ROLE = "experimental"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _tablebot  # noqa: E402
from _tablebot import solver  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation  # noqa: E402
from opponent import OppStats  # noqa: E402

TABLES = {
    "base": solver.STRATEGY,
    "passive": _tablebot.load("counter_passive.npy"),
    "folder": _tablebot.load("counter_folder.npy"),
}
MIN_SEEN = 40
# Solver priors (agent.py vs itself), used to shrink small samples.
PRIOR_FOLD_POST, PRIOR_RERAISE_PRE, PRIOR_BET_POST = 0.496, 0.464, 0.352


def classify(stats: OppStats, current: str) -> str:
    if min(stats.faced[0], stats.faced[1], stats.free[1]) < MIN_SEEN:
        return "base"
    fold = stats.fold_rate(1, PRIOR_FOLD_POST)
    reraise_pre = stats.reraise_rate(0, PRIOR_RERAISE_PRE)
    bet = stats.bet_rate(1, PRIOR_BET_POST)
    stay = 0.03   # hysteresis
    if fold >= 0.60 - (stay if current == "folder" else 0):
        return "folder"
    s = stay if current == "passive" else 0
    if fold <= 0.43 + s and reraise_pre <= 0.15 + s and bet <= 0.33 + s:
        return "passive"
    return "base"


def new_half(config: AgentConfig, memory: dict) -> None:
    solver.new_half(config, memory)
    memory["opp"] = OppStats()
    memory["last"] = None
    memory["mode"] = "base"
    memory["mode_hands"] = {m: 0 for m in TABLES}


def _decide(obs: Observation, memory: dict) -> Action:
    return _tablebot.decide(obs, memory, TABLES[memory["mode"]])


def choose_action(obs: Observation, memory: dict) -> Action:
    action = _tablebot.choose(obs, memory, TABLES[memory.get("mode", "base")])
    memory["last"] = (obs, int(action))
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        last = memory.get("last")
        if last is not None and last[0].hand_index == result.hand_index:
            memory["opp"].learn(last[0], last[1], result)
        else:
            memory["opp"].learn(None, 0, result)
        memory["mode_hands"][memory["mode"]] += 1
        memory["mode"] = classify(memory["opp"], memory["mode"])
    except Exception:
        memory["mode"] = "base"
    memory["last"] = None
