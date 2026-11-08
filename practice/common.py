"""common.py - load any bot by name and play judge-accurate matches in-process.

Bot names understood everywhere in practice/:
    agent             the submission, agent.py
    <name>            a bot in practice/bots/<name>.py (e.g. exploit, shover)
    zoo:<name>        an opponent from zoo.py (e.g. zoo:nit); bare zoo names work too
    strat=<table.npy> a solver strategy table played straight (zoo.Strat), e.g. a best response
    thr:... / noisy:...  generated training opponents (see opponents_gen.py)
    path/to/bot.py    any file with new_half / choose_action / on_hand_end

Matches follow the judge's rules: duplicate deals, the 8 s time bank with the
first 1 ms of every decision free, 1 s per decision, forfeit after 5 errors,
and the judge's own verdict function. Every half loads a fresh copy of a bot
file, like the tournament's fresh server per half.
"""

import importlib.util
import math
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BOTS_DIR = os.path.join(HERE, "bots")
sys.path.insert(0, ROOT)

from game import (ACTION_TIMEOUT_SECONDS, BIG_BLIND, CHOP_Z, ERROR_LIMIT,  # noqa: E402
                  NETWORK_GRACE_SECONDS, TIME_BANK_SECONDS, Action, AgentConfig,
                  Game, button_for_deal, default_action, make_deal)
from judge_engine import coerce_action, decide_verdict  # noqa: E402  (the judge's own rules)

_loaded = 0


def load_module(path):
    """Load a bot file as a brand-new module (fresh globals, like a fresh server)."""
    global _loaded
    _loaded += 1
    # The bot's own folder goes LAST on the path: its helper modules (e.g.
    # strategy.py) are found, but `import agent` still means the repo's agent.py.
    folder = os.path.dirname(os.path.abspath(path))
    if folder not in sys.path:
        sys.path.append(folder)
    spec = importlib.util.spec_from_file_location(f"practice_bot_{_loaded}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModuleBot:
    """A bot file in agent.py style."""

    def __init__(self, path, name):
        self.path, self.name = path, name

    def new_half(self, config):
        self.m = load_module(self.path)
        self.mem = {}
        self.m.new_half(config, self.mem)

    def act(self, obs):
        return self.m.choose_action(obs, self.mem)

    def end(self, result):
        self.m.on_hand_end(result, self.mem)


class ZooBot:
    """An opponent class from zoo.py."""

    def __init__(self, zoo_name, name):
        self.zoo_name, self.name = zoo_name, name

    def new_half(self, config):
        import zoo
        self.b = zoo.make(self.zoo_name)
        self.b.new_half(config)

    def act(self, obs):
        return self.b.act(obs)

    def end(self, result):
        self.b.end(result)


class GenBot(ZooBot):
    """A generated opponent from opponents_gen.py (thr:... or noisy:...)."""

    def __init__(self, name):
        self.zoo_name = self.name = name

    def new_half(self, config):
        import opponents_gen
        self.b = opponents_gen.make(self.name)
        self.b.new_half(config)


def bot_files(role=None):
    """Names of the bots in practice/bots/, optionally only one ROLE."""
    out = []
    for f in sorted(os.listdir(BOTS_DIR)):
        if f.endswith(".py") and not f.startswith("_"):
            if role is None or _role(os.path.join(BOTS_DIR, f)) == role:
                out.append(f[:-3])
    return out


def _role(path):
    with open(path) as fh:
        for line in fh:
            if line.startswith("ROLE"):
                return line.split("=")[1].strip().strip("\"'")
    return "opponent"


def zoo_names():
    import zoo
    return ["zoo:" + n for n in zoo.names()]


def resolve(name):
    """Turn a bot name into a bot object with new_half / act / end."""
    if name == "agent":
        return ModuleBot(os.path.join(ROOT, "agent.py"), name)
    if name.endswith(".py"):
        return ModuleBot(os.path.abspath(name), name)
    path = os.path.join(BOTS_DIR, name + ".py")
    if os.path.exists(path):
        return ModuleBot(path, name)
    import zoo
    z = name[4:] if name.startswith("zoo:") else name
    if z in zoo.names() or z.startswith("strat="):   # strat=table.npy[@purify]: play a solver table
        return ZooBot(z, name)
    if name.startswith(("thr:", "noisy:")):           # generated opponents (opponents_gen.py)
        return GenBot(name)
    raise SystemExit(f"unknown bot {name!r}. Known: agent, {', '.join(bot_files())}, "
                     f"{', '.join(zoo_names())}")


class Seat:
    """One bot in one half, with the judge's time bank, errors and forfeit."""

    def __init__(self, bot):
        self.bot = bot
        self.time = 0.0          # charged thinking time (first 1 ms of each decision free)
        self.max_decision = 0.0
        self.decisions = 0
        self.errors = 0
        self.forfeited = False
        self.timed_out = False
        self.first_error = None

    def _error(self, message):
        self.errors += 1
        if self.first_error is None:
            self.first_error = message
        if self.errors >= ERROR_LIMIT:
            self.forfeited = True

    def start(self, config):
        try:
            self.bot.new_half(config)
        except Exception:
            self._error("new_half raised\n" + traceback.format_exc())
            self.forfeited = True

    def act(self, obs):
        fallback = default_action(obs.to_call)
        if self.forfeited or self.timed_out:
            return fallback
        t0 = time.perf_counter()
        try:
            raw = self.bot.act(obs)
        except Exception:
            self._charge(t0)
            self._error("choose_action raised\n" + traceback.format_exc())
            return fallback
        elapsed = self._charge(t0)
        self.decisions += 1
        if elapsed > ACTION_TIMEOUT_SECONDS:
            self._error(f"a decision took {elapsed:.2f}s (limit {ACTION_TIMEOUT_SECONDS:g}s)")
            return fallback
        action = coerce_action(raw.name if isinstance(raw, Action) else raw)
        if action is None or action not in obs.legal_actions:
            self._error(f"illegal answer {raw!r}; legal: {[a.name for a in obs.legal_actions]}")
            return fallback
        return action

    def end(self, result):
        if self.forfeited:
            return
        t0 = time.perf_counter()
        try:
            self.bot.end(result)
        except Exception:
            self._error("on_hand_end raised\n" + traceback.format_exc())
        self.time += time.perf_counter() - t0

    def _charge(self, t0):
        elapsed = time.perf_counter() - t0
        self.max_decision = max(self.max_decision, elapsed)
        self.time += max(0.0, elapsed - NETWORK_GRACE_SECONDS)
        if self.time >= TIME_BANK_SECONDS:
            self.timed_out = True
        return elapsed


def play_half(bot_a, bot_b, seed, deals, half):
    """Bot a sits in seat 0 in half 1 and seat 1 in half 2, like agent1 in the judge.
    Returns (a's chip result per deal, a's Seat, b's Seat)."""
    config = AgentConfig(num_hands=deals)
    a, b = Seat(bot_a), Seat(bot_b)
    seats = (a, b) if half == 1 else (b, a)
    a.start(config)
    b.start(config)
    nets = []
    for d in range(deals):
        game = Game(make_deal(str(seed), d), button_for_deal(d), hand_index=d)
        while not game.is_terminal:
            s = game.to_act
            game.apply(seats[s].act(game.observation(s)))
        nets.append(game.result().nets[half - 1])
        for s in (0, 1):
            seats[s].end(game.hand_result(s))
    return nets, a, b


def _seat_summary(s):
    return {"errors": s.errors, "forfeited": s.forfeited, "timed_out": s.timed_out,
            "think": round(s.time, 3), "max_decision": round(s.max_decision, 4),
            "first_error": s.first_error}


def play_match(name_a, name_b, seed, deals=1000):
    """One duplicate match, scored exactly like judge_engine.combine_halves."""
    bot_a, bot_b = resolve(name_a), resolve(name_b)
    n1, a1, b1 = play_half(bot_a, bot_b, seed, deals, 1)
    n2, a2, b2 = play_half(bot_a, bot_b, seed, deals, 2)
    paired = [(x + y) / BIG_BLIND for x, y in zip(n1, n2)]
    mean = sum(paired) / len(paired)
    sd = statistics.stdev(paired) if len(paired) > 1 else 0.0
    se = sd / math.sqrt(len(paired))
    lo, hi = mean - CHOP_Z * se, mean + CHOP_Z * se
    verdict, points, _ = decide_verdict(lo, hi)
    a_forfeit, b_forfeit = a1.forfeited or a2.forfeited, b1.forfeited or b2.forfeited
    if a_forfeit or b_forfeit:
        points = {"agent1": float(not a_forfeit), "agent2": float(not b_forfeit)}
        verdict = "BOTH_FORFEIT" if a_forfeit and b_forfeit else (
            "AGENT2_WIN" if a_forfeit else "AGENT1_WIN")
    return {
        "a": name_a, "b": name_b, "seed": str(seed), "deals": deals,
        "mean": mean, "sd": sd, "se": se, "ci": (lo, hi), "verdict": verdict,
        "points_a": points["agent1"], "points_b": points["agent2"],
        "a_halves": [_seat_summary(a1), _seat_summary(a2)],
        "b_halves": [_seat_summary(b1), _seat_summary(b2)],
    }
