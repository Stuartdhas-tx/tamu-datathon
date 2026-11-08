"""compare.py - pit bots against each other and rank them like the tournament.

    python practice/compare.py                            # candidates vs the full field
    python practice/compare.py --bots agent,exploit --field zoo:nit,shover --seeds 5
    python practice/compare.py --round-robin --bots agent,exploit,greedy,smallball

Candidates (--bots; default agent + every practice bot with ROLE = "candidate")
each play every field bot (--field; default every zoo.py opponent + every
practice bot with ROLE = "opponent") and each other. Every pairing plays
--seeds duplicate matches of --deals deals; all pairings use the same seeds,
so every candidate faces the same cards. Ranked by points, then by average
bb/deal (the tournament's tiebreak).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402


def _job(args):
    a, b, seed, deals = args
    return common.play_match(a, b, seed, deals)


def _side(r, bot):
    """(points, bb/deal, halves) from bot's point of view."""
    if r["a"] == bot:
        return r["points_a"], r["mean"], r["a_halves"]
    return r["points_b"], -r["mean"], r["b_halves"]


def _letter(points):
    return "W" if points == 1 else "L" if points == 0 else "C"


def label(name):
    """Short display name: strat=/long/path/br.npy -> strat:br."""
    if name.startswith("strat="):
        return "strat:" + os.path.splitext(os.path.basename(name[6:].split("@")[0]))[0]
    return name if len(name) <= 24 else "..." + name[-21:]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bots", default="", help="comma list of candidates")
    p.add_argument("--field", default="", help="comma list of opponents")
    p.add_argument("--round-robin", action="store_true", help="every bot plays every other bot")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0, help="first seed number (use fresh seeds to confirm a result)")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--json", default="", help="save every match result here")
    a = p.parse_args()

    cands = [x for x in a.bots.split(",") if x] or ["agent"] + common.bot_files("candidate")
    field = [x for x in a.field.split(",") if x] or common.zoo_names() + common.bot_files("opponent")
    for name in cands + field:
        common.resolve(name)   # fail fast on a typo
    if a.round_robin:
        everyone = list(dict.fromkeys(cands + field))
        pairs = [(x, y) for i, x in enumerate(everyone) for y in everyone[i + 1:]]
        cands = everyone
    else:
        field = [f for f in field if f not in cands]
        pairs = [(c, f) for c in cands for f in field]
        pairs += [(x, y) for i, x in enumerate(cands) for y in cands[i + 1:]]
    seeds = [f"practice-{i}" for i in range(a.seed_start, a.seed_start + a.seeds)]
    jobs = [(x, y, s, a.deals) for x, y in pairs for s in seeds]

    print(f"{len(cands)} candidates, {len(jobs)} matches of {a.deals} deals on {a.procs} processes...",
          file=sys.stderr)
    t0 = time.time()
    results = []
    with mp.Pool(a.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            results.append(r)
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} done ({time.time() - t0:.0f}s)", file=sys.stderr)

    # --- leaderboard -------------------------------------------------------------
    table = []
    for bot in cands:
        mine = [r for r in results if bot in (r["a"], r["b"])]
        pts = [_side(r, bot)[0] for r in mine]
        bbs = [_side(r, bot)[1] for r in mine]
        halves = [h for r in mine for h in _side(r, bot)[2]]
        per_opp = defaultdict(list)
        for r in mine:
            per_opp[r["b"] if r["a"] == bot else r["a"]].append(_side(r, bot)[1])
        worst = min(per_opp.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
        table.append({
            "bot": bot, "points": sum(pts), "of": len(pts),
            "w": pts.count(1.0), "c": pts.count(0.5), "l": pts.count(0.0),
            "bb": sum(bbs) / len(bbs), "worst": (worst[0], sum(worst[1]) / len(worst[1])),
            "errors": sum(h["errors"] for h in halves),
            "forfeits": sum(h["forfeited"] for h in halves),
            "think": max(h["think"] for h in halves),
        })
    table.sort(key=lambda t: (t["points"], t["bb"]), reverse=True)

    w = max([16] + [len(label(n)) for n in cands + field])
    print(f"\nLEADERBOARD  ({a.seeds} seeds x {a.deals} deals per pairing; ranked by points, then bb/deal)")
    print(f"{'#':>2} {'bot':{w}s} {'points':>11s} {'W/C/L':>10s} {'bb/deal':>8s}  {'worst matchup':{w + 8}s} "
          f"{'err':>3s} {'forf':>4s} {'think':>6s}")
    for i, t in enumerate(table, 1):
        print(f"{i:2d} {label(t['bot']):{w}s} {t['points']:5.1f}/{t['of']:<5d} {t['w']:3d}/{t['c']}/{t['l']:<3d} "
              f"{t['bb']:+8.2f}  {label(t['worst'][0]):{w}s} {t['worst'][1]:+7.2f} {t['errors']:3d} "
              f"{t['forfeits']:4d} {t['think']:5.2f}s")

    # --- matrix ----------------------------------------------------------------
    shown = [t["bot"] for t in table][:6]   # top six keep the table terminal-width
    rows = list(dict.fromkeys([r["b"] for r in results] + [r["a"] for r in results]))
    print("\nMATCHUPS  (bb/deal from the column bot's side, then one letter per seed: W win, C chop, L loss)")
    print(f"{'opponent':{w}s}" + "".join(f"{label(c)[-14:]:>16s}" for c in shown))
    for opp in rows:
        cells = []
        for c in shown:
            rs = [r for r in results if {r["a"], r["b"]} == {c, opp}] if c != opp else []
            if rs:
                sides = [_side(r, c) for r in rs]
                bb = sum(s[1] for s in sides) / len(sides)
                cells.append(f"{bb:+7.2f} {''.join(_letter(s[0]) for s in sides):>7s} ")
            else:
                cells.append(f"{'':>16s}")
        print(f"{label(opp):{w}s}" + "".join(cells))

    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"leaderboard": table, "matches": results}, fh, indent=1)
        print(f"\nsaved {len(results)} match results to {a.json}")


if __name__ == "__main__":
    main()
