import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import brain
from game import *
d = sys.argv[1]
# 1. every reachable decision has a solver context, under many action mixes
rng = random.Random(1)
seen = set(); dec = 0
for h in range(60000):
    g = Game(make_deal("x", h), button_seat=h % 2, hand_index=h)
    w = rng.choice([0.2, 0.5, 0.8, 1.0])
    while not g.is_terminal:
        o = g.observation(g.to_act)
        k = brain.context_key(o); dec += 1
        assert k in brain.CTX, (k, o)
        seen.add(k)
        r = [a for a in o.legal_actions if a >= 2]
        a = rng.choice(r) if r and rng.random() < w else (Action.CHECK_CALL if rng.random() > .05 or Action.FOLD not in o.legal_actions else Action.FOLD)
        g.apply(a)
print("decisions", dec, "contexts seen", len(seen), "of", len(brain.CTX))
# 2. numpy features vs C features: same distribution?
F = np.fromfile(f"{d}/feat_sample.bin", dtype=np.float32).reshape(-1, 5)
N = 3000
out = np.zeros((N, 5))
t = [0, 0, 0]
for i in range(N):
    c = rng.sample(range(36), 10); h, b = sorted(c[:5]), c[5:]
    t0 = time.perf_counter(); out[i, :2] = brain.flop_features(h, b[:3])
    t1 = time.perf_counter(); out[i, 2:4] = brain.turn_features(h, b[:4])
    t2 = time.perf_counter(); out[i, 4] = brain.river_rank(h, b)
    t3 = time.perf_counter(); t[0] += t1 - t0; t[1] += t2 - t1; t[2] += t3 - t2
print("ms flop/turn/river", [round(x / N * 1000, 3) for x in t])
for j, n in enumerate(["fmu", "fnu", "tmu", "tnu", "rr"]):
    q = [.1, .25, .5, .75, .9, .97]
    print(n, "C ", np.round(np.quantile(F[:, j], q), 3))
    print(n, "np", np.round(np.quantile(out[:, j], q), 3))
