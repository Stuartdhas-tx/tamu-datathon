"""combo - exploit's opponent modelling plus smallball's lower-variance sizing.

Solver strategy -> re-weighted by OppStats (see exploit.py) -> pot-sized
raises moved to half-pot unless the hand is strong (see smallball.py).
"""

import os
import sys
import time

ROLE = "candidate"

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import agent as solver  # noqa: E402
import exploit  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation  # noqa: E402
from smallball import BIG_HAND  # noqa: E402


def new_half(config: AgentConfig, memory: dict) -> None:
    exploit.new_half(config, memory)


def _decide(obs: Observation, memory: dict) -> Action:
    bucket, strength, _ = solver._street_info(obs, memory)
    probs = solver._strategy(obs, bucket)
    if probs is None:
        return solver._fallback(obs, strength)
    probs = exploit._adjust(obs, probs, strength, memory["opp"])
    if strength < BIG_HAND and Action.RAISE_HALF in obs.legal_actions:
        weights = dict(probs)
        weights[Action.RAISE_HALF] = weights.get(Action.RAISE_HALF, 0) + weights.pop(Action.RAISE_POT, 0)
        probs = list(weights.items())
    x = memory["rng"].random() * sum(p for _, p in probs)
    for action, p in probs:
        x -= p
        if x <= 0:
            return action
    return probs[-1][0]


def choose_action(obs: Observation, memory: dict) -> Action:
    start = time.perf_counter()
    try:
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        action = Action.CHECK_CALL
    memory["used"] = memory.get("used", 0.0) + time.perf_counter() - start
    memory["last"] = (obs, int(action))
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    exploit.on_hand_end(result, memory)
