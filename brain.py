"""
brain.py - fast hand features and strategy lookup for the bot.

Everything here is table driven so one hand costs about a millisecond in
numpy. The features match training/train.c:

  mu  average rank of my final hand against a random holding, over runouts
  nu  how often the final rank is in the top 10%
  r   (river) exact rank against a random holding

(mu, nu) or r picks a card bucket. Betting history picks a context, and
strategy[context, bucket] holds the solved probabilities for the 4 actions.
"""

import os
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = HERE

NP, NT = 630, 7140
TAB = np.fromfile(os.path.join(DATA, "tab.bin"), dtype=np.uint16).reshape(NT, NP)

# Index helpers -------------------------------------------------------------
PAIR = np.zeros((36, 36), dtype=np.intp)
for i, (a, b) in enumerate(combinations(range(36), 2)):
    PAIR[a, b] = PAIR[b, a] = i
TRI = np.zeros((36, 36, 36), dtype=np.intp)
for i, (a, b, c) in enumerate(combinations(range(36), 3)):
    for x, y, z in ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)):
        TRI[x, y, z] = i
B3 = np.array(list(combinations(range(5), 3)), dtype=np.intp)   # 10 ways to take 3 of 5
H2 = np.array(list(combinations(range(5), 2)), dtype=np.intp)   # 10 ways to take 2 of 5

_CHOOSE = [[0] * 7 for _ in range(40)]
for n in range(40):
    _CHOOSE[n][0] = 1
    for r in range(1, 7):
        _CHOOSE[n][r] = (_CHOOSE[n - 1][r - 1] + _CHOOSE[n - 1][r]) if n else 0


def hand_index(sorted_cards) -> int:
    """Combinadic index of a sorted 5-card holding (matches train.c)."""
    return sum(_CHOOSE[c][i + 1] for i, c in enumerate(sorted_cards))


# A fixed shuffled list of random holdings, used as the "random opponent".
_rng0 = np.random.default_rng(20261107)
NH = 60000
_h = np.argsort(_rng0.random((NH, 36)), axis=1)[:, :5].astype(np.intp)
HOLD_PAIRS = PAIR[_h[:, H2[:, 0]], _h[:, H2[:, 1]]]                 # (NH, 10)
HOLD_MASK = (np.uint64(1) << _h.astype(np.uint64)).sum(axis=1).astype(np.uint64)
del _h

M_FLOP, N_FLOP, N_TURN, N_RIVER = 60, 190, 240, 5900
NU_T = 0.9
RNG = np.random.default_rng()

_pairs28 = np.array(list(combinations(range(28), 2)), dtype=np.intp)


def _holdings(dead: int, base: int):
    """Random holdings that avoid the dead cards, from a rotating window."""
    start = int(RNG.integers(0, NH - base))
    m = HOLD_MASK[start:start + base]
    keep = (m & np.uint64(dead)) == 0
    return HOLD_PAIRS[start:start + base][keep], m[keep]


def _bits(cards) -> int:
    x = 0
    for c in cards:
        x |= 1 << c
    return x


def _strengths(boards, my_pairs):
    """boards: (M, 5). Returns (S per pair (M, 630), my best (M,))."""
    tri = TRI[boards[:, B3[:, 0]], boards[:, B3[:, 1]], boards[:, B3[:, 2]]]
    m = boards.shape[0]
    s = TAB.take(tri.ravel(), axis=0).reshape(m, 10, NP).max(axis=1)
    return s, s[:, my_pairs].max(axis=1)


def river_rank(hole, board, scale: float = 1.0) -> float:
    hole = list(hole)
    b = np.array([board], dtype=np.intp)
    my_pairs = PAIR[np.array(hole)[H2[:, 0]], np.array(hole)[H2[:, 1]]]
    s, mine = _strengths(b, my_pairs)
    hp, _ = _holdings(_bits(hole) | _bits(board), int(N_RIVER * scale))
    v = s[0].take(hp).max(axis=1)
    my = mine[0]
    return float(((v < my).sum() + 0.5 * (v == my).sum()) / max(1, v.size))


def _runout_ranks(boards, run_masks, hole, dead, base):
    h = np.array(hole)
    my_pairs = PAIR[h[H2[:, 0]], h[H2[:, 1]]]
    s, mine = _strengths(boards, my_pairs)
    hp, hm = _holdings(dead, base)
    m = boards.shape[0]
    v = s[np.arange(m)[:, None, None], hp[None, :, :]].max(axis=2)  # (M, n)
    ok = (hm[None, :] & run_masks[:, None]) == 0
    mine = mine[:, None]
    win = ((v < mine) & ok).sum(axis=1) + 0.5 * ((v == mine) & ok).sum(axis=1)
    return win / np.maximum(1, ok.sum(axis=1))


def turn_features(hole, board, scale: float = 1.0):
    dead = _bits(hole) | _bits(board)
    rivers = np.array([c for c in range(36) if not dead >> c & 1], dtype=np.intp)
    n = rivers.size
    boards = np.empty((n, 5), dtype=np.intp)
    boards[:, :4] = board
    boards[:, 4] = rivers
    run_masks = np.uint64(1) << rivers.astype(np.uint64)
    r = _runout_ranks(boards, run_masks, hole, dead, int(N_TURN * scale))
    return float(r.mean()), float((r >= NU_T).mean())


def flop_features(hole, board, scale: float = 1.0):
    dead = _bits(hole) | _bits(board)
    live = np.array([c for c in range(36) if not dead >> c & 1], dtype=np.intp)
    m = max(8, int(M_FLOP * scale))
    pick = _pairs28[RNG.integers(0, _pairs28.shape[0], m)]
    run = live[pick]
    boards = np.empty((m, 5), dtype=np.intp)
    boards[:, :3] = board
    boards[:, 3:] = run
    run_masks = (np.uint64(1) << run[:, 0].astype(np.uint64)) | (np.uint64(1) << run[:, 1].astype(np.uint64))
    r = _runout_ranks(boards, run_masks, hole, dead, N_FLOP)
    return float(r.mean()), float((r >= NU_T).mean())


# Buckets ---------------------------------------------------------------------

class Grid:
    def __init__(self, nm, nn, me, ne):
        self.nm, self.nn = nm, nn
        self.me = me
        self.ne = ne

    def bucket(self, mu: float, nu: float) -> int:
        a = int(np.searchsorted(self.me, mu, side="right"))
        return a * self.nn + int(np.searchsorted(self.ne[a], nu, side="right"))


def _load_thresholds():
    x = np.fromfile(os.path.join(DATA, "thresholds.bin"), dtype=np.float32)
    p = 0

    def grid():
        nonlocal p
        nm, nn = int(x[p]), int(x[p + 1])
        p += 2
        me = x[p:p + nm - 1].copy()
        p += nm - 1
        ne = x[p:p + nm * (nn - 1)].reshape(nm, nn - 1).copy()
        p += nm * (nn - 1)
        return Grid(nm, nn, me, ne)

    gf, gt = grid(), grid()
    nr = int(x[p])
    return gf, gt, x[p + 1:p + nr].copy()


GRID_F, GRID_T, RIVER_EDGES = _load_thresholds()
PF_BUCKET = np.fromfile(os.path.join(DATA, "pf_bucket.bin"), dtype=np.uint8)
_pf = np.fromfile(os.path.join(DATA, "pf_feat.bin"), dtype=np.float32).reshape(2, -1)
PF_MU, PF_NU = _pf[0], _pf[1]


def street_info(hole, board, scale: float = 1.0):
    """Return (bucket, strength in 0..1, nut chance) for the current street."""
    n = len(board)
    if n == 0:
        i = hand_index(sorted(hole))
        return int(PF_BUCKET[i]), float(PF_MU[i]), float(PF_NU[i])
    if n == 3:
        mu, nu = flop_features(hole, board, scale)
        return GRID_F.bucket(mu, nu), mu, nu
    if n == 4:
        mu, nu = turn_features(hole, board, scale)
        return GRID_T.bucket(mu, nu), mu, nu
    r = river_rank(hole, board, scale)
    return int(np.searchsorted(RIVER_EDGES, r, side="right")), r, float(r >= NU_T)


# Betting context ---------------------------------------------------------------

def _load_contexts():
    raw = np.load(os.path.join(DATA, "ctx.npy"))
    return {int(k): int(v) for k, v in raw}


CTX = _load_contexts()


def context_key(obs) -> int:
    """The solver's context key for this decision (see ctx_id in train.c)."""
    me_pl = 0 if obs.is_button else 1
    street = int(obs.street)
    seq, aggr = 1, 0
    for a in obs.action_history:
        pl = me_pl if a.is_me else 1 - me_pl
        act = int(a.action)
        if act >= 2:
            aggr = pl + 1
        if int(a.street) == street:
            seq = seq * 4 + act
    ps = 3 if street == 0 else obs.pot - obs.my_bet - obs.opp_bet
    return 1 + ((seq << 20) | (ps << 8) | (street << 4) | (me_pl << 3) | (aggr << 1))
