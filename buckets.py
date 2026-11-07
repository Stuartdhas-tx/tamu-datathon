"""Turn feature samples into bucket thresholds (thresholds.bin, pf_bucket.bin)."""
import sys, numpy as np
d = sys.argv[1]
F = np.fromfile(f"{d}/feat_sample.bin", dtype=np.float32).reshape(-1, 5)
Q_M = np.array([.06,.12,.18,.25,.32,.40,.48,.56,.64,.72,.79,.85,.90,.94,.97])   # 16 levels, top heavy
N_NU = 8

def grid(mu, nu):
    me = np.quantile(mu, Q_M)
    lv = np.searchsorted(me, mu, side="right")
    out = [float(len(me) + 1), float(N_NU)] + list(me)
    for k in range(len(me) + 1):
        x = nu[lv == k]
        out += list(np.quantile(x, np.arange(1, N_NU) / N_NU))
    return out

rr = F[:, 4]
tail = 1 - np.array([.008,.02,.035,.05,.065,.08,.10,.12,.14,.16,.18,.21,.24,.27,.30,.34,.38,.42,.46,.50,.54,.58,.62,.66,.70])
re = sorted(set(np.round(np.concatenate([
    np.linspace(.05, .55, 12),
    np.linspace(.60, .90, 13),
    1 - np.array([.09,.08,.07,.06,.05,.042,.035,.03,.025,.02,.016,.013,.01,.008,.006,.0045,.003,.002,.001]),
]), 5)))
out = grid(F[:, 0], F[:, 1]) + grid(F[:, 2], F[:, 3]) + [float(len(re) + 1)] + list(re)
np.array(out, dtype=np.float32).tofile(f"{d}/thresholds.bin")
print("river levels", len(re) + 1)

P = np.fromfile(f"{d}/pf_feat.bin", dtype=np.float32).reshape(2, -1)
mu, nu = P
print("preflop mu", np.quantile(mu, [0, .5, .9, 1]), "nu", np.quantile(nu, [0, .5, .9, 1]))
qs = np.array([.08,.16,.25,.34,.43,.52,.61,.70,.78,.85,.91,.96])
me = np.quantile(mu, qs)
lv = np.searchsorted(me, mu, side="right")
b = np.zeros(mu.shape, dtype=np.uint8)
NN = 4
for k in range(len(qs) + 1):
    m = lv == k
    ne = np.quantile(nu[m], np.arange(1, NN) / NN)
    b[m] = k * NN + np.searchsorted(ne, nu[m], side="right")
b.tofile(f"{d}/pf_bucket.bin")
print("preflop buckets", b.max() + 1)
for k, n in [(0, "flop"), (2, "turn")]:
    print(n, "mu quantiles", np.round(np.quantile(F[:, k], [.1, .5, .9, .99]), 3), "nu", np.round(np.quantile(F[:, k + 1], [.1, .5, .9, .99]), 3))
print("river", np.round(np.quantile(rr, [.1, .5, .9, .99]), 3))
