"""greedy - the solver bot, but always plays the solver's most likely action.

Tests whether the solver's randomising (mixing) earns its keep. A pure
strategy is easier to read and exploit, but against bots that don't adapt it
can win more, because it never plays the solver's lower-value options.
"""

import os
import sys

ROLE = "candidate"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import agent as solver  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation  # noqa: E402


def new_half(config: AgentConfig, memory: dict) -> None:
    solver.new_half(config, memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    try:
        bucket, strength, _ = solver._street_info(obs, memory)
        probs = solver._strategy(obs, bucket)
        action = solver._fallback(obs, strength) if probs is None else max(probs, key=lambda x: x[1])[0]
        return action if action in obs.legal_actions else Action.CHECK_CALL
    except Exception:
        return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
