"""Express rule-based opponents as strategy tables the solver can best-respond to.

    python models.py <workdir>
Writes model_<name>.bin (float32, same shape as a solver dump) and bucket_strength.npy.
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brain

F, C, H, P = 0, 1, 2, 3


def decode(key):
    """(street, player, start pot, aggressor, action digits) from a context key."""
    k = key - 1
    aggr, pl, street = (k >> 1) & 3, (k >> 3) & 1, (k >> 4) & 15
    ps, seq = (k >> 8) & 4095, k >> 20
    digits = []
    while seq > 1:
        digits.append(seq & 3)
        seq >>= 2
    return street, pl, ps, aggr, digits[::-1]


def replay(street, ps, digits):
    """Game state at a context: mirrors game.py's betting arithmetic."""
    if street == 0:
        b, c, to = [1, 2], [1, 2], 0
    else:
        b, c, to = [0, 0], [ps // 2, ps // 2], 1
    inc, raises = 2, 0
    for a in digits:
        tc = max(0, b[1 - to] - b[to])
        stack = 80 - c[to]
        if a == C:
            amt = min(tc, stack)
        else:
            pot = c[0] + c[1]
            mn = b[1 - to] + max(inc, 2)
            potto = b[1 - to] + pot + tc
            half = b[1 - to] + (pot + tc) // 2
            target = half if a == H else potto
            x = min(max(target, mn), potto, b[to] + stack)
            amt = x - b[to]
            inc = x - b[1 - to]
            raises += 1
        c[to] += amt
        b[to] += amt
        to = 1 - to
    tc = max(0, b[1 - to] - b[to])
    stack, ostack = 80 - c[to], 80 - c[1 - to]
    return dict(pot=c[0] + c[1], to_call=tc, raises=raises,
                can_raise=raises < 4 and stack > tc and ostack > 0, to=to)


def bucket_strength(work):
    """Average strength of each bucket on each street (street, bucket)."""
    out = np.zeros((4, 128))
    x = np.fromfile(f"{work}/feat_sample.bin", dtype=np.float32).reshape(-1, 5)
    pm, pb = brain.PF_MU, brain.PF_BUCKET
    for b in range(pb.max() + 1):
        out[0, b] = pm[pb == b].mean()
    for s, g, i in ((1, brain.GRID_F, 0), (2, brain.GRID_T, 2)):
        lv = np.searchsorted(g.me, x[:, i], side="right")
        sub = np.array([np.searchsorted(g.ne[l], n, side="right") for l, n in zip(lv, x[:, i + 1])])
        bk = lv * g.nn + sub
        for b in range(g.nm * g.nn):
            m = bk == b
            if m.any():
                out[s, b] = x[m, i].mean()
            else:          # empty nu cell: use its mu level
                out[s, b] = x[lv == b // g.nn, i].mean()
    bk = np.searchsorted(brain.RIVER_EDGES, x[:, 4], side="right")
    for b in range(bk.max() + 1):
        out[3, b] = x[bk == b, 4].mean() if (bk == b).any() else 1.0
    return out


def threshold_table(strength, value, half, margin=0., bluff=0., shade=0.):
    """The zoo's Threshold bot as probabilities over (context, bucket)."""
    t = np.zeros((len(brain.CTX), 128, 4), dtype=np.float32)
    for key, ctx in brain.CTX.items():
        street, pl, ps, aggr, d = decode(key)
        st = replay(street, ps, d)
        pot, tc = st["pot"], st["to_call"]
        need = tc / (pot + tc) if tc else 0
        disc = shade * min(1.0, tc / max(1, pot)) if tc else 0
        for b in range(128):
            e = strength[street, b] - disc
            row = t[ctx, b]
            if e > value and st["can_raise"]:
                row[P] = 1
            elif e > half and st["can_raise"] and st["raises"] < 2:
                row[H] = 1
            elif tc == 0:
                if bluff and st["can_raise"]:
                    row[H], row[C] = bluff, 1 - bluff
                else:
                    row[C] = 1
            elif e >= need + margin:
                row[C] = 1
            else:
                row[F] = 1
    return t


def rule_table(strength, policy):
    """Any rule bot as a table. policy(e, st, pl, aggr, street) -> 4 probabilities."""
    t = np.zeros((len(brain.CTX), 128, 4), dtype=np.float32)
    for key, ctx in brain.CTX.items():
        street, pl, ps, aggr, d = decode(key)
        st = replay(street, ps, d)
        for b in range(128):
            t[ctx, b] = policy(strength[street, b], st, pl, aggr, street)
    return t


def cbet_policy(e, st, pl, aggr, street):
    r, tc = st["can_raise"], st["to_call"]
    if street == 0:
        if e > .5 and r:
            return (0, 0, 0, 1)
        return (0, 1, 0, 0) if tc <= 2 else (1, 0, 0, 0)
    if tc == 0:
        return (0, 0, 1, 0) if r and (aggr == pl + 1 or e > .6) else (0, 1, 0, 0)
    return (0, 1, 0, 0) if e > .68 else (1, 0, 0, 0)


def maniac_policy(e, st, pl, aggr, street):
    r, tc = st["can_raise"], st["to_call"]
    pot = .55 if r else 0
    half = (1 - pot) * .3 if r else 0
    rest = 1 - pot - half
    return (rest, 0, half, pot) if tc and e < .2 else (0, rest, half, pot)


def always_policy(a):
    def f(e, st, pl, aggr, street):
        p = [0, 0, 0, 0]
        p[a if st["can_raise"] else 1] = 1
        return p
    return f


def random_policy(e, st, pl, aggr, street):
    legal = [1] + ([0] if st["to_call"] else []) + ([2, 3] if st["can_raise"] else [])
    return [1 / len(legal) if a in legal else 0 for a in range(4)]


AGGRESSIVE = {
    "cbet": cbet_policy, "maniac": maniac_policy, "pot": always_policy(3),
    "half": always_policy(2), "rand": random_policy,
}

TYPES = {
    "basic": dict(value=.8, half=.65), "loose": dict(value=.85, half=.72, margin=-.03),
    "tight": dict(value=.8, half=.7, margin=.12), "aggro": dict(value=.68, half=.55, margin=.02, bluff=.2),
    "shade": dict(value=.82, half=.68, margin=.05, bluff=.1, shade=.25),
    "shade2": dict(value=.86, half=.72, margin=.02, bluff=.15, shade=.35),
}

if __name__ == "__main__":
    work = sys.argv[1]
    s = bucket_strength(work)
    np.save(f"{work}/bucket_strength.npy", s)
    for name, p in TYPES.items():
        threshold_table(s, **p).tofile(f"{work}/model_{name}.bin")
    threshold_table(s, value=.65, half=.5, margin=-.05, bluff=.5).tofile(f"{work}/model_lag.bin")
    for name, f in AGGRESSIVE.items():
        rule_table(s, f).tofile(f"{work}/model_{name}.bin")
    print("models", list(TYPES), list(AGGRESSIVE), "lag")
