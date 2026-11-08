"""opponents_gen.py - generated opponents, for training the selector on bots it will not be tested on.

Names (usable anywhere a bot name is, e.g. compare.py --field):
    thr:V,H,M,B,S      zoo.Threshold(value=V, half=H, margin=M, bluff=B, shade=S):
                       equity + pot odds, the most likely competitor style
    noisy:NAME:Q       zoo opponent NAME, but a random legal action with probability Q

training_set(n, seed) returns a reproducible mix of both, spread over styles
(tight / loose, passive / aggressive, shading or not, noisy or not).
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zoo  # noqa: E402


class Noisy(zoo.Base):
    """Wraps a zoo opponent: plays a random legal action with probability q."""

    def __init__(self, inner, q):
        self.inner, self.q = inner, q

    def new_half(self, cfg):
        super().new_half(cfg)
        self.inner.new_half(cfg)

    def act(self, o):
        if self.rng.random() < self.q:
            return self.rng.choice(o.legal_actions)
        return self.inner.act(o)

    def end(self, r):
        self.inner.end(r)


def make(name):
    kind, _, rest = name.partition(":")
    if kind == "thr":      # thr:V/H/M/B/S also works, for comma-separated bot lists
        v, h, m, b, s = (float(x) for x in rest.replace("/", ",").split(","))
        return zoo.Threshold(v, h, m, b, s)
    if kind == "noisy":
        inner, q = rest.rsplit(":", 1)
        return Noisy(zoo.make(inner), float(q))
    raise ValueError(name)


def training_set(n=40, seed=2026):
    rng = random.Random(seed)
    names = []
    for _ in range(n * 3 // 5):
        v = rng.uniform(0.62, 0.95)
        h = rng.uniform(0.5, v)
        m = rng.uniform(-0.06, 0.16)
        b = rng.choice([0.0, 0.0, rng.uniform(0.02, 0.3)])
        s = rng.choice([0.0, rng.uniform(0.05, 0.45)])
        names.append(f"thr:{v:.2f},{h:.2f},{m:.2f},{b:.2f},{s:.2f}")
    styles = ["passive", "maniac", "nit", "trapper", "blinder", "cbet", "station", "alwayshalf", "eq_loose"]
    noisy = [f"noisy:{s}:{q:.2f}" for s in styles for q in (0.05, 0.15, 0.3)]
    rng.shuffle(noisy)
    return names + noisy[:max(0, n - len(names))]
