"""selector - profile the opponent first, then switch to the model that beats that kind of bot.

Each half (1000 hands) is split in two:
  1. hands 0..PROFILE_HANDS-1: play agent.py (the safest model) while
     building the opponent's behaviour profile (bots/_features.py: fold /
     raise / bet rates, showdown looseness, how often they quit preflop, and
     risk = how much our results swing);
  2. the rest: play the portfolio model that did best against the K most
     similar of the training opponents in practice/tables/selector.json
     (train_selector.py: generated opponents, never the test bots).
The choice is risk-aware: a model is only eligible if, against EVERY one of
the K neighbours, it scored at least as many points as agent.py (worst case,
not average), and agent.py stays unless an eligible model is expected to win
more points (MARGIN) or, at equal points, at least BB_GAIN more bb/deal (the
tournament's tiebreak). Neighbours must be similar (MAX_DIST). Results within
a half are mostly card luck, so the choice is made from behaviour, not from
chips won.
"""

import importlib.util
import json
import math
import os
import sys

ROLE = "experimental"      # trained against the old solver-only agent.py; retrain before trusting it

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import _features  # noqa: E402
import agent as solver  # noqa: E402
import greedy  # noqa: E402
import smallball  # noqa: E402
from game import Action, AgentConfig, HandResult, Observation  # noqa: E402

K, MARGIN, MAX_DIST = 7, 0.02, 3.0
BB_GAIN = 1.0              # bb/deal gain needed to switch at equal expected points


def _load_shark():
    """The poker-bot/ Shark, or None where that folder is not checked out."""
    folder = os.path.join(ROOT, "poker-bot")
    if not os.path.exists(os.path.join(folder, "agent.py")):
        return None
    if folder not in sys.path:
        sys.path.append(folder)
    spec = importlib.util.spec_from_file_location("selector_shark", os.path.join(folder, "agent.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODELS = {"agent": solver, "greedy": greedy, "smallball": smallball}
_SHARK = _load_shark()
if _SHARK is not None:
    MODELS["poker-bot/agent.py"] = _SHARK
with open(os.path.join(os.path.dirname(HERE), "tables", "selector.json")) as _fh:
    DATA = json.load(_fh)
PROFILE_HANDS = DATA["profile_hands"]
_MEAN, _STD = DATA["scale"]["mean"], DATA["scale"]["std"]


def _z(v):
    return [(x - m) / s for x, m, s in zip(v, _MEAN, _STD)]


def pick(vector):
    """(model, details) for an opponent profile."""
    z = _z(vector)
    near = sorted((math.dist(z, _z(o["profile"])), o) for o in DATA["opponents"])[:K]
    if not near or near[0][0] > MAX_DIST:
        return DATA["default"], {"reason": "no similar training opponent"}
    weights = [1.0 / (d + 0.25) for d, _ in near]
    mean = lambda m, k: sum(w * o[k][m] for w, (_, o) in zip(weights, near)) / sum(weights)
    portfolio = [m for m in DATA["portfolio"] if m in MODELS]
    pts = {m: mean(m, "points") for m in portfolio}
    bb = {m: mean(m, "bb") for m in portfolio}
    base = DATA["default"]
    eligible = [m for m in portfolio if m != base and pts[m] >= pts[base] - 0.005
                and all(o["points"][m] >= o["points"][base] for _, o in near)]
    best = base
    if eligible:
        top = max(eligible, key=lambda m: (pts[m], bb[m]))
        if pts[top] - pts[base] >= MARGIN or bb[top] - bb[base] >= BB_GAIN:
            best = top
    return best, {"points": pts, "bb": bb, "eligible": eligible, "neighbours": [o["name"] for _, o in near]}


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["sub"] = {}
    for name, model in MODELS.items():
        memory["sub"][name] = {}
        model.new_half(config, memory["sub"][name])
    memory["active"] = DATA["default"]
    memory["profile"] = _features.Profile()
    memory["hands"] = 0
    memory["last"] = None
    memory["choice"] = None


def _decide(obs: Observation, memory: dict) -> Action:
    return MODELS[memory["active"]].choose_action(obs, memory["sub"][memory["active"]])


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
        memory["profile"].learn(memory.get("last"), result)
        active = memory["active"]
        MODELS[active].on_hand_end(result, memory["sub"][active])
        memory["hands"] += 1
        if memory["hands"] == PROFILE_HANDS:
            memory["active"], memory["choice"] = pick(memory["profile"].vector())
    except Exception:
        memory["active"] = DATA["default"]
    memory["last"] = None
