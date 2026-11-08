"""solver.py - measure and improve how exploitable a strategy table is.

    python practice/solver.py exploitability                       # the shipped strategy.npy
    python practice/solver.py exploitability practice/tables/fortress.npy --export br.npy
    python practice/solver.py refine strategy.npy --out practice/tables/refined.npy --iters 2000

exploitability: trains a best response (train.c's BR_VS mode) against the
table, purified the way agent.py plays it (its PURIFY, or --purify), on an
evaluation pool of deals, and reports how many bb/deal a perfect opponent
could win against it inside the solver's card abstraction (a lower bound for
the real game). --export saves that best response as a table to play
against: compare.py --field strat=br.npy

refine: continues CFR self-play from the table instead of from zero, on a
separate training pool, and exports the result as a new table. train.c is
copied into the work folder and patched for the warm start (WARM / NODISC);
the original is untouched.

Heavy files (compiled solver, deal pools, raw dumps) live in --work
(default practice/work/, ignored by git and Docker). Pools are reused.
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
THREADS = max(1, (os.cpu_count() or 2) - 1)
PER_ROUND = 10_000_000
EVAL_SEED, TRAIN_SEED = 4242, 9001

_ALLOC = "  REG=calloc(nf,4); AVG=calloc(nf,4);\n"
_WARM = _ALLOC + """  if(getenv("WARM")){ /* start regret matching at a saved strategy: REG = AVG = K * probabilities */
    FILE *wf=fopen(getenv("WARM"),"rb"); float *W=malloc(nf*4); if(!wf||fread(W,4,nf,wf)!=nf){perror("WARM");exit(1);} fclose(wf);
    float K=getenv("WARM_K")?atof(getenv("WARM_K")):1000.0f;
    for(size_t i=0;i<nf;i+=4){ float t=W[i]+W[i+1]+W[i+2]+W[i+3]; if(t>0) for(int a=0;a<4;a++){ REG[i+a]=K*W[i+a]/t; AVG[i+a]=K*W[i+a]/t; } }
    free(W); printf("warm start from %s (K=%g)\\n",getenv("WARM"),K); }
"""
_DISC = "  double t0=now(); int disc_rounds=rounds*2/5;"
_NODISC = "  double t0=now(); int disc_rounds=getenv(\"NODISC\")?0:rounds*2/5;"


def solver_binary(work):
    exe = os.path.join(work, "train_warm")
    if os.path.exists(exe):
        return exe
    src = open(os.path.join(ROOT, "train.c")).read()
    if src.count(_ALLOC) != 1 or src.count(_DISC) != 1:
        sys.exit("train.c changed: the warm-start patch no longer applies")
    patched = os.path.join(work, "train_warm.c")
    with open(patched, "w") as fh:
        fh.write(src.replace(_ALLOC, _WARM).replace(_DISC, _NODISC))
    for flags in (["-O3", "-mcpu=native"], ["-O3", "-march=native"], ["-O3"]):
        if subprocess.run(["clang", *flags, "-pthread", patched, "-o", exe, "-lm"],
                          capture_output=True).returncode == 0:
            print(f"built {exe} ({' '.join(flags)})")
            return exe
    sys.exit("could not compile train.c with clang")


def data_dir(work):
    d = os.path.join(work, "data")
    os.makedirs(d, exist_ok=True)
    for f in ("tab.bin", "thresholds.bin", "pf_bucket.bin", "pf_feat.bin"):
        link = os.path.join(d, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(ROOT, f), link)
    return d


def pool(work, exe, name, deals, seed):
    path = os.path.join(work, f"pool_{name}_{deals}.bin")
    if not os.path.exists(path):
        deals -= deals % THREADS
        print(f"building the {name} pool: {deals:,} deals (about {deals / 12000 / 60:.0f} min)...")
        subprocess.run([exe, "pool", data_dir(work), str(deals), str(THREADS), path, str(seed)],
                       check=True, stdout=subprocess.DEVNULL)
    return path


def agent_purify():
    """agent.py's PURIFY, so the measured strategy is the one actually played."""
    m = re.search(r"^PURIFY\s*=\s*([0-9.]+)", open(os.path.join(ROOT, "agent.py")).read(), re.M)
    return float(m.group(1)) if m else 0.0


def to_float(table_path, out, purify=0.0):
    """Table -> raw floats, purified like the bots play it: drop actions under
    `purify` of the row, keeping the most likely one if none survive."""
    t = np.load(table_path).astype(np.float32)
    if purify > 0:
        total = t.sum(axis=2, keepdims=True)
        keep = t >= purify * total
        keep = np.where(keep.any(axis=2, keepdims=True), keep, t == t.max(axis=2, keepdims=True))
        t = np.where(keep & (total > 0), t, 0).astype(np.float32)
    t.tofile(out)


def export(raw, out):
    """Raw float dump -> uint8 table, exactly like export.py."""
    a = np.fromfile(raw, dtype=np.float32).reshape(-1, 128, 4)
    t = a.sum(axis=2, keepdims=True)
    p = np.where(t > 0, a / np.maximum(t, 1e-12), 0)
    np.save(out, np.round(p * 255).astype(np.uint8))


def train(exe, work, pool_path, out, iters_m, env):
    rounds = max(1, int(iters_m * 1_000_000 // PER_ROUND))
    cmd = [exe, "train", data_dir(work), pool_path, out, str(PER_ROUND), str(rounds), str(THREADS)]
    values = []
    with subprocess.Popen(cmd, env={**os.environ, **env}, stdout=subprocess.PIPE, text=True) as p:
        for line in p.stdout:
            m = re.search(r"round (\d+)/(\d+).*avgv (-?[\d.]+)", line)
            if m:
                values.append(float(m.group(3)))
                if int(m.group(1)) % max(1, rounds // 10) == 0:
                    print(f"  round {m.group(1)}/{m.group(2)}  value {values[-1]:+.4f}", flush=True)
    if p.returncode:
        sys.exit(f"solver failed ({p.returncode})")
    return values


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exploitability")
    e.add_argument("table", nargs="?", default=os.path.join(ROOT, "strategy.npy"))
    e.add_argument("--iters", type=float, default=200, help="millions of iterations (default 200)")
    e.add_argument("--export", default="", help="save the best response as a .npy table")
    e.add_argument("--purify", type=float, default=None,
                   help="purify the table like the bot plays it (default: agent.py's PURIFY)")
    r = sub.add_parser("refine")
    r.add_argument("table")
    r.add_argument("--out", required=True)
    r.add_argument("--iters", type=float, default=2000, help="millions of iterations (default 2000)")
    r.add_argument("--k", type=float, default=1000, help="warm-start weight (default 1000)")
    for s in (e, r):
        s.add_argument("--work", default=os.path.join(HERE, "work"))
        s.add_argument("--pool-deals", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    exe = solver_binary(a.work)
    frozen = os.path.join(a.work, "frozen.bin")
    purify = (agent_purify() if a.purify is None else a.purify) if a.cmd == "exploitability" else 0.0
    to_float(a.table, frozen, purify)

    if a.cmd == "exploitability":
        p = pool(a.work, exe, "eval", a.pool_deals or 1_440_000, EVAL_SEED)
        raw = os.path.join(a.work, "br_raw.bin")
        print(f"best response vs {a.table} purified at {purify:g}, {a.iters:g}M iterations...")
        v = train(exe, a.work, p, raw, a.iters, {"BR_VS": frozen})
        tail = v[-5:]
        # avgv = chips per hand per seat; a duplicate deal is one hand in each seat
        # (2 hands) and 1 bb = 2 chips, so chips per hand == bb per deal.
        print(f"\nexploitability: {sum(tail) / len(tail):.3f} bb/deal "
              f"(a perfect opponent's edge inside the abstraction; lower is safer)")
        if a.export:
            export(raw, a.export)
            print(f"best response saved to {a.export}  (play it: compare.py --field strat={a.export})")
    else:
        p = pool(a.work, exe, "train", a.pool_deals or 2_880_000, TRAIN_SEED)
        raw = os.path.join(a.work, "refined_raw.bin")
        print(f"refining {a.table}, {a.iters:g}M iterations from a warm start...")
        train(exe, a.work, p, raw, a.iters, {"WARM": frozen, "WARM_K": str(a.k), "NODISC": "1"})
        export(raw, a.out)
        print(f"refined table saved to {a.out}. Measure it: solver.py exploitability {a.out}")


if __name__ == "__main__":
    main()
