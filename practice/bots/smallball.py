"""smallball - the solver bot with lower variance.

Moves the solver's pot-sized raises to half-pot unless the hand is strong, so
bluffs and medium hands build smaller pots. Gives up some EV for a tighter
confidence interval, which can turn chops into wins (the verdict needs the
whole 95% interval above 0).
"""

import os
import sys

ROLE = "candidate"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import agent as solver  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation  # noqa: E402

BIG_HAND = 0.8   # strength at or above which pot-sized raises are kept


def new_half(config: AgentConfig, memory: dict) -> None:
    solver.new_half(config, memory)


def _decide(obs: Observation, memory: dict) -> Action:
    bucket, strength, _ = solver._street_info(obs, memory)
    probs = solver._strategy(obs, bucket)
    if probs is None:
        return solver._fallback(obs, strength)
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
    try:
        action = _decide(obs, memory)
        return action if action in obs.legal_actions else Action.CHECK_CALL
    except Exception:
        return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
