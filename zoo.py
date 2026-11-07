"""Practice opponents modelled on what other teams are likely to submit."""
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from game import Action, Street
import brain

F, C, H, P = Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT


class Base:
    def new_half(self, cfg):
        self.rng = random.Random(random.random())
        self.key = None
        self.eq = 0.5
        self.t = 0
    def end(self, r):
        pass
    def strength(self, o):
        k = (o.hand_index, len(o.board))
        if k != self.key:
            self.key = k
            self.eq = brain.street_info(o.hole_cards, o.board, 0.5)[1]
        return self.eq
    def raise_or(self, o, a, other=C):
        return a if a in o.legal_actions else other


class Fixed(Base):
    def __init__(self, f): self.f = f
    def act(self, o): return self.f(self, o)


def checkfold(s, o): return C if o.to_call == 0 else F
def station(s, o): return C
def alwayspot(s, o): return s.raise_or(o, P)
def alwayshalf(s, o): return s.raise_or(o, H)
def rand(s, o): return s.rng.choice(o.legal_actions)


class Threshold(Base):
    """Monte Carlo equity against a random hand plus pot odds: the textbook
    bot from the guide's strategy section, the most likely competitor."""
    def __init__(self, value=.8, half=.65, margin=.0, bluff=0., shade=0.):
        self.value, self.half, self.margin, self.bluff, self.shade = value, half, margin, bluff, shade
    def act(self, o):
        e = self.strength(o)
        need = o.to_call / (o.pot + o.to_call) if o.to_call else 0
        # shade: discount equity when facing big bets, as better bots do
        if o.to_call:
            e -= self.shade * min(1.0, o.to_call / max(1, o.pot))
        if e > self.value and P in o.legal_actions:
            return P
        if e > self.half and H in o.legal_actions and o.raises_this_street < 2:
            return H
        if o.to_call == 0:
            if self.bluff and self.rng.random() < self.bluff and H in o.legal_actions:
                return H
            return C
        return C if e >= need + self.margin else F


class Passive(Base):
    """Calls with anything half decent, raises only monsters."""
    def act(self, o):
        e = self.strength(o)
        if e > .93 and P in o.legal_actions:
            return P
        if o.to_call == 0:
            return C
        return C if e > .3 else F


class Maniac(Base):
    """Raises a lot regardless, gives up only with air against big bets."""
    def act(self, o):
        e = self.strength(o)
        if self.rng.random() < .55 and P in o.legal_actions:
            return P
        if self.rng.random() < .3 and H in o.legal_actions:
            return H
        if o.to_call and e < .2:
            return F
        return C


class Nit(Base):
    """Folds to almost any bet without a strong hand."""
    def act(self, o):
        e = self.strength(o)
        if e > .85 and P in o.legal_actions:
            return P
        if o.to_call == 0:
            return C
        if o.to_call <= 2 and e > .4:
            return C
        return C if e > .72 else F


class Trapper(Base):
    """Slow plays strong hands, then check raises or jams on the river."""
    def act(self, o):
        e = self.strength(o)
        if o.street == Street.RIVER:
            if e > .9 and P in o.legal_actions:
                return P
            if o.to_call == 0:
                return C
            return C if e > .75 else F
        if o.to_call == 0:
            return C
        if e > .85:
            return C
        need = o.to_call / (o.pot + o.to_call)
        return C if e >= need else F


class Blinder(Base):
    """Hand-rule bot: ignores position and sizing, bets made strength."""
    def act(self, o):
        e = self.strength(o)
        if o.street == Street.PREFLOP:
            if e > .56:
                return self.raise_or(o, H)
            if o.to_call <= 2:
                return C
            return C if e > .52 else F
        if e > .7:
            return self.raise_or(o, H)
        if o.to_call == 0:
            return C
        return C if e > .55 else F


class CBet(Base):
    """Always continuation bets after raising, gives up when raised."""
    def act(self, o):
        e = self.strength(o)
        mine = [a for a in o.action_history if a.is_me and int(a.action) >= 2]
        if o.street == Street.PREFLOP:
            if e > .5:
                return self.raise_or(o, P)
            return C if o.to_call <= 2 else F
        if o.to_call == 0:
            if mine or e > .6:
                return self.raise_or(o, H)
            return C
        return C if e > .68 else F


_MADE = {
    "checkfold": lambda: Fixed(checkfold), "station": lambda: Fixed(station),
    "alwayspot": lambda: Fixed(alwayspot), "alwayshalf": lambda: Fixed(alwayshalf),
    "random": lambda: Fixed(rand),
    "eq_basic": lambda: Threshold(), "eq_loose": lambda: Threshold(.85, .72, -.03),
    "eq_tight": lambda: Threshold(.8, .7, .12), "eq_aggro": lambda: Threshold(.68, .55, .02, .2),
    "eq_shade": lambda: Threshold(.82, .68, .05, .1, .25), "eq_shade2": lambda: Threshold(.86, .72, .02, .15, .35),
    "passive": Passive, "maniac": Maniac, "nit": Nit, "trapper": Trapper, "blinder": Blinder, "cbet": CBet,
}


class Strat(Base):
    """A solved strategy table played straight, for strategy vs strategy tests."""
    def __init__(self, path, purify=.04):
        import numpy as np
        self.table = np.load(path)
        self.purify = purify
    def act(self, o):
        k = (o.hand_index, len(o.board))
        if k != self.key:
            self.key = k
            self.bucket = brain.street_info(o.hole_cards, o.board)[0]
        ctx = brain.CTX.get(brain.context_key(o))
        row = self.table[ctx, self.bucket]
        p = [(a, int(row[int(a)])) for a in o.legal_actions]
        t = sum(x for _, x in p)
        if t == 0:
            return C
        p = [(a, x) for a, x in p if x >= self.purify * t] or p
        return self.rng.choices([a for a, _ in p], [x for _, x in p])[0]


def names():
    return list(_MADE)


def make(name):
    if name.startswith("strat="):
        bits = name[6:].split("@")
        return Strat(bits[0], float(bits[1]) if len(bits) > 1 else .04)
    if name.startswith("agent:"):
        import arena
        return arena.AgentBot(path=name[6:])
    return _MADE[name]()
