"""fortress - built to lose as few matches as possible.

A match is lost only when the opponent's 95% interval is entirely above 0, so
the goal is to give no opponent a real edge: play the least exploitable
strategy table available (practice/tables/fortress.npy, measured with
solver.py exploitability), mixed exactly as solved. No exploitation and no
fixed patterns (pure or smallball-style sizing rules): both measurably raise
what a best response can win. Same never-crash wrapper as agent.py.

The table slot currently holds the shipped strategy.npy: it measured the
least exploitable of everything tried (0.27 bb/deal), so today fortress plays
exactly like agent.py. Drop in a table that measures lower to test it, e.g.
`compare.py --bots agent,fortress`. ROLE is "experimental" so default runs
don't play two identical bots.
"""

import os
import sys

ROLE = "experimental"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _tablebot  # noqa: E402
from _tablebot import solver  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation  # noqa: E402

TABLE = _tablebot.load("fortress.npy")


def new_half(config: AgentConfig, memory: dict) -> None:
    solver.new_half(config, memory)


def _decide(obs: Observation, memory: dict) -> Action:
    return _tablebot.decide(obs, memory, TABLE)


def choose_action(obs: Observation, memory: dict) -> Action:
    return _tablebot.choose(obs, memory, TABLE)


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass
