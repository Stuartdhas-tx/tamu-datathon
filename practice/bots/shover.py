"""shover - push / fold. Maximum pressure, minimum thinking.

Preflop: pot-raises the top 30% of starting hands (with 40 bb stacks, two
pot raises are nearly all-in) and folds the rest. Postflop: pot-raises strong
hands, check-calls medium ones, folds weak ones to a bet. A plausible
"simple but aggressive" competitor; tests how we handle big early pots.
"""

import os
import random
import sys

import numpy as np

ROLE = "opponent"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import brain  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation, Street  # noqa: E402

TOP_PREFLOP = float(np.quantile(brain.PF_MU, 0.70))   # strength of a top-30% starting hand


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = random.Random()
    memory["key"] = None


def _strength(obs: Observation, memory: dict) -> float:
    key = (obs.hand_index, len(obs.board))
    if memory["key"] != key:
        memory["key"] = key
        memory["strength"] = brain.street_info(obs.hole_cards, obs.board, 0.5)[1]
    return memory["strength"]


def choose_action(obs: Observation, memory: dict) -> Action:
    try:
        s = _strength(obs, memory)
        legal = obs.legal_actions
        if obs.street == Street.PREFLOP:
            if s >= TOP_PREFLOP:
                return Action.RAISE_POT if Action.RAISE_POT in legal else Action.CHECK_CALL
            return Action.FOLD if Action.FOLD in legal else Action.CHECK_CALL
        if s > 0.72:
            return Action.RAISE_POT if Action.RAISE_POT in legal else Action.CHECK_CALL
        if obs.to_call == 0 or s > 0.55:
            return Action.CHECK_CALL
        return Action.FOLD
    except Exception:
        return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
