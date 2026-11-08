"""_tablebot.py - agent.py's decision procedure with any strategy table.

Shared by the table candidates (fortress, exploiter, sharp, ...). Same bucket
and context lookup, same purification, same pot-odds fallback and the same
never-crash wrapper as agent.py; only the table differs. Optional
`smallball` moves pot-sized raises to half-pot unless the hand is strong;
optional `purify` drops actions below that share of the row (agent.py: 0.04).
If every action falls below it, the most likely one is kept.
"""

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(os.path.dirname(HERE), "tables")
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import agent as solver  # noqa: E402
from game import Action, Observation  # noqa: E402

BIG_HAND = 0.8


def load(name):
    return np.load(os.path.join(TABLES, name))


def _probs(obs: Observation, bucket: int, table, purify):
    ctx = solver.brain.CTX.get(solver.brain.context_key(obs))
    if ctx is None:
        return None
    row = table[ctx, bucket]
    probs = [(a, int(row[int(a)])) for a in obs.legal_actions]
    total = sum(p for _, p in probs)
    if total == 0:
        return None
    return [(a, p) for a, p in probs if p >= purify * total] or [max(probs, key=lambda x: x[1])]


def decide(obs: Observation, memory: dict, table, smallball=False, purify=None) -> Action:
    bucket, strength, _ = solver._street_info(obs, memory)
    probs = _probs(obs, bucket, table, solver.PURIFY if purify is None else purify)
    if probs is None:
        return solver._fallback(obs, strength)
    if smallball and strength < BIG_HAND and Action.RAISE_HALF in obs.legal_actions:
        weights = dict(probs)
        weights[Action.RAISE_HALF] = weights.get(Action.RAISE_HALF, 0) + weights.pop(Action.RAISE_POT, 0)
        probs = list(weights.items())
    x = memory["rng"].random() * sum(p for _, p in probs)
    for action, p in probs:
        x -= p
        if x <= 0:
            return action
    return probs[-1][0]


def choose(obs: Observation, memory: dict, table, smallball=False, purify=None) -> Action:
    start = time.perf_counter()
    try:
        action = decide(obs, memory, table, smallball, purify)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        action = Action.CHECK_CALL
    memory["used"] = memory.get("used", 0.0) + time.perf_counter() - start
    return action
