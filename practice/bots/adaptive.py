"""adaptive - an opponent that learns and tries to exploit US.

Tracks our tendencies with the same OppStats our exploit bot uses (from its
side, we are the opponent), then:
  * bluffs more when we fold to bets a lot, and stops bluffing when we don't;
  * calls our bets more when we bet often (we must be bluffing), less when
    we rarely bet.
Its base game is equity plus pot odds. If one of our bots loses chips to it,
that bot is exploitable by a learning competitor.
"""

import os
import random
import sys

ROLE = "opponent"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import brain  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation, Street  # noqa: E402
from opponent import OppStats  # noqa: E402


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = random.Random()
    memory["key"] = None
    memory["opp"] = OppStats()
    memory["last"] = None


def _strength(obs: Observation, memory: dict) -> float:
    key = (obs.hand_index, len(obs.board))
    if memory["key"] != key:
        memory["key"] = key
        memory["strength"] = brain.street_info(obs.hole_cards, obs.board, 0.5)[1]
    return memory["strength"]


def _decide(obs: Observation, memory: dict) -> Action:
    e, legal, stats = _strength(obs, memory), obs.legal_actions, memory["opp"]
    g = 0 if obs.street == Street.PREFLOP else 1
    their_fold = stats.fold_rate(g, 0.4)
    their_bet = stats.bet_rate(g, 0.4)
    if e > 0.8 and Action.RAISE_POT in legal:
        return Action.RAISE_POT
    if e > 0.66 and Action.RAISE_HALF in legal and obs.raises_this_street < 2:
        return Action.RAISE_HALF
    if obs.to_call == 0:
        bluff = min(0.7, max(0.0, 2 * (their_fold - 0.4)))
        if Action.RAISE_HALF in legal and e < 0.45 and memory["rng"].random() < bluff:
            return Action.RAISE_HALF
        return Action.CHECK_CALL
    need = obs.to_call / (obs.pot + obs.to_call)
    margin = 0.08 - 0.25 * (their_bet - 0.4)   # they bet a lot -> call lighter
    return Action.CHECK_CALL if e >= need + margin else Action.FOLD


def choose_action(obs: Observation, memory: dict) -> Action:
    try:
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        action = Action.CHECK_CALL
    memory["last"] = (obs, int(action))
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    try:
        last = memory.get("last")
        if last is not None and last[0].hand_index == result.hand_index:
            memory["opp"].learn(last[0], last[1], result)
        else:
            memory["opp"].learn(None, 0, result)
    except Exception:
        pass
    memory["last"] = None
