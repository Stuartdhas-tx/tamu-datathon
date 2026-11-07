"""In-process arena: same rules, same scoring as the judge, no HTTP.

    python arena.py --deals 1000 --seeds 2 --opps tag,lag,nit
"""
import argparse, importlib.util, math, multiprocessing as mp, os, sys, time, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from game import Game, make_deal, AgentConfig, Action, default_action


def load_agent(path=None):
    spec = importlib.util.spec_from_file_location("agent_%d" % id(path), path or os.path.join(ROOT, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class AgentBot:
    def __init__(self, mod=None, path=None):
        self.m = mod or load_agent(path)
        self.t = 0.0
    def new_half(self, cfg):
        self.mem = {}
        self.m.new_half(cfg, self.mem)
    def act(self, o):
        t = time.perf_counter()
        a = self.m.choose_action(o, self.mem)
        self.t += time.perf_counter() - t
        return a
    def end(self, r):
        self.m.on_hand_end(r, self.mem)


def play_half(a, b, seed, deals, half):
    """a in seat (half-1), returns list of a's chip results per deal."""
    cfg = AgentConfig(num_hands=deals)
    bots = [a, b] if half == 1 else [b, a]
    a.new_half(cfg); b.new_half(cfg)
    out = []
    err = 0
    for d in range(deals):
        g = Game(make_deal(seed, d), button_seat=d % 2, hand_index=d)
        while not g.is_terminal:
            s = g.to_act
            o = g.observation(s)
            try:
                act = bots[s].act(o)
                if act not in o.legal_actions:
                    raise ValueError("illegal %r" % (act,))
            except Exception:
                err += bots[s] is a
                if bots[s] is a:
                    traceback.print_exc()
                act = default_action(o.to_call)
            g.apply(act)
        for s in (0, 1):
            bots[s].end(g.hand_result(s))
        out.append(g.result().nets[half - 1])
    return out, err


def match(a_factory, b_factory, seed, deals):
    a1, b1 = a_factory(), b_factory()
    r1, e1 = play_half(a1, b1, seed, deals, 1)
    a2, b2 = a_factory(), b_factory()
    r2, e2 = play_half(a2, b2, seed, deals, 2)
    x = [(p + q) / 2.0 for p, q in zip(r1, r2)]  # chips -> big blinds
    n = len(x); m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / max(1, n - 1))
    se = sd / math.sqrt(n)
    return dict(mean=m, se=se, sd=sd, errors=e1 + e2, t=max(a1.t, a2.t))


def _job(args):
    name, seed, deals, agent = args
    import zoo
    res = match(lambda: AgentBot(path=agent), lambda: zoo.make(name), seed, deals)
    res.update(name=name, seed=seed)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--opps", default="")
    p.add_argument("--agent", default=None)
    p.add_argument("--procs", type=int, default=8)
    a = p.parse_args()
    import zoo
    names = [x for x in a.opps.split(",") if x] or zoo.names()
    jobs = [(n, s + 100, a.deals, a.agent) for n in names for s in range(a.seeds)]
    with mp.Pool(a.procs) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    by = {}
    for r in res:
        by.setdefault(r["name"], []).append(r)
    wins = chop = lost = 0
    print(f"{'opponent':16s} {'bb/deal':>8s} {'+-95%':>6s} {'W/C/L':>7s} {'sd':>6s} {'err':>3s} {'time':>5s}")
    tot = 0
    for n in names:
        rs = by[n]
        m = sum(r["mean"] for r in rs) / len(rs)
        w = sum(r["mean"] - 1.96 * r["se"] > 0 for r in rs)
        l = sum(r["mean"] + 1.96 * r["se"] < 0 for r in rs)
        c = len(rs) - w - l
        wins += w; chop += c; lost += l; tot += m
        print(f"{n:16s} {m:8.2f} {1.96*rs[0]['se']:6.2f} {w:3d}/{c}/{l} {rs[0]['sd']:6.1f} {sum(r['errors'] for r in rs):3d} {max(r['t'] for r in rs):5.1f}")
    print(f"TOTAL wins {wins} chops {chop} losses {lost}  mean {tot/len(names):.2f} bb/deal")


if __name__ == "__main__":
    main()
