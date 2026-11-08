"""mc_equity - the guide's "textbook" bot, built only on the official evaluator.

Monte Carlo equity: deal the missing board cards and a random opponent hand
SAMPLES times, score both with omaha_score, count wins. Then pot odds decide
calls, equity decides raises, and it bluffs a little in position. Independent
of brain.py, so it also cross-checks our own strength estimates.
"""

import os
import random
import sys

ROLE = "opponent"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluator import NUM_CARDS, omaha_score, warmup  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation, Street  # noqa: E402

warmup()
SAMPLES = 60           # about 1.5 ms per street
POT_RAISE, HALF_RAISE = 0.78, 0.64
CALL_MARGIN = 0.03
BLUFF = 0.12           # bet chance with a weak hand when checked to on the button


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = random.Random()
    memory["key"] = None


def _equity(obs: Observation, rng: random.Random) -> float:
    dead = set(obs.hole_cards) | set(obs.board)
    deck = [c for c in range(NUM_CARDS) if c not in dead]
    need = 5 - len(obs.board)
    score = 0.0
    for _ in range(SAMPLES):
        draw = rng.sample(deck, need + 5)
        board = list(obs.board) + draw[:need]
        mine, theirs = omaha_score(obs.hole_cards, board), omaha_score(draw[need:], board)
        score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
    return score / SAMPLES


def choose_action(obs: Observation, memory: dict) -> Action:
    try:
        key = (obs.hand_index, len(obs.board))
        if memory["key"] != key:
            memory["key"] = key
            memory["eq"] = _equity(obs, memory["rng"])
        e, legal = memory["eq"], obs.legal_actions
        if e > POT_RAISE and Action.RAISE_POT in legal:
            return Action.RAISE_POT
        if e > HALF_RAISE and Action.RAISE_HALF in legal and obs.raises_this_street < 2:
            return Action.RAISE_HALF
        if obs.to_call == 0:
            if (obs.is_button and obs.street != Street.PREFLOP and Action.RAISE_HALF in legal
                    and memory["rng"].random() < BLUFF):
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        need = obs.to_call / (obs.pot + obs.to_call)
        return Action.CHECK_CALL if e >= need + CALL_MARGIN else Action.FOLD
    except Exception:
        return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
