"""check_rules.py - do the solver's tables still match the game's rules?

    python practice/check_rules.py                # local checks
    python practice/check_rules.py --upstream     # also compare with the official starter repo

The strategy tables were solved for one exact rule set. If the organizers
tune game.py (stacks, blinds, raise cap), contexts stop matching and the bot
quietly falls back to simple pot odds. This checks:
  1. the game constants the solver assumes;
  2. that every decision in many random hands has a solver context, and how
     often the table has an answer (vs. the pot-odds fallback), per street;
  3. with --upstream: that the organizer files are identical to
     github.com/tamu-datathon-org/heads-up-bundle (main branch).
Exit code 1 if anything doesn't match.
"""

import argparse
import os
import random
import re
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import numpy as np  # noqa: E402

import brain  # noqa: E402
import game  # noqa: E402
from evaluator import NUM_CARDS  # noqa: E402
from game import Action, Game, Street, make_deal  # noqa: E402

SOLVER_ASSUMES = {
    "STARTING_STACK": 80, "BIG_BLIND": 2, "SMALL_BLIND": 1, "RAISE_CAP_PER_STREET": 4,
    "NUM_HOLE_CARDS": 5, "NUM_BOARD_CARDS": 5,
}
UPSTREAM = "https://raw.githubusercontent.com/tamu-datathon-org/heads-up-bundle/main/"
ORGANIZER_FILES = [
    "game.py", "evaluator.py", "judge_engine.py", "test_agent.py", "README.md",
    "docker/Dockerfile.base", "docker/Dockerfile.agent", "tools/check_build.sh", "tools/smoke_test.py",
    "baseline-bots/baseline_random.py", "baseline-bots/baseline_calling_station.py",
    "baseline-bots/baseline_always_pot.py", "baseline-bots/baseline_always_half.py",
]


def check_constants():
    ok = True
    print("1. game constants the solver was trained for")
    found = dict(SOLVER_ASSUMES, NUM_CARDS=36)
    for name, want in found.items():
        have = NUM_CARDS if name == "NUM_CARDS" else getattr(game, name, None)
        good = have == want
        ok &= good
        print(f"   {'ok      ' if good else 'MISMATCH'} {name} = {have}" + ("" if good else f"  (solver assumes {want})"))
    return ok


def check_coverage(hands, seed):
    print(f"\n2. solver coverage over {hands} random hands")
    strategy = np.load(os.path.join(common.ROOT, "strategy.npy"))
    rng = random.Random(seed)
    seen, missing, table, empty = set(), Counter(), Counter(), Counter()
    for h in range(hands):
        g = Game(make_deal(f"rules{seed}", h), button_seat=h % 2, hand_index=h)
        raise_rate = rng.choice([0.2, 0.5, 0.8, 1.0])
        buckets = {}
        while not g.is_terminal:
            o = g.observation(g.to_act)
            street = Street(o.street).name
            key = brain.context_key(o)
            ctx = brain.CTX.get(key)
            if ctx is None:
                missing[street] += 1
            else:
                seen.add(key)
                bk = (g.to_act, len(o.board))
                if bk not in buckets:
                    buckets[bk] = brain.street_info(o.hole_cards, o.board, 0.2)[0]
                row = strategy[ctx, buckets[bk]]
                (table if sum(int(row[int(a)]) for a in o.legal_actions) else empty)[street] += 1
            raises = [a for a in o.legal_actions if a >= 2]
            if raises and rng.random() < raise_rate:
                a = rng.choice(raises)
            elif Action.FOLD in o.legal_actions and rng.random() < 0.05:
                a = Action.FOLD
            else:
                a = Action.CHECK_CALL
            g.apply(a)
    print(f"   {'street':8s} {'decisions':>9s} {'table':>7s} {'empty row':>9s} {'NO CONTEXT':>10s}")
    for s in ("PREFLOP", "FLOP", "TURN", "RIVER"):
        n = table[s] + empty[s] + missing[s]
        if n:
            print(f"   {s:8s} {n:9d} {100 * table[s] / n:6.1f}% {100 * empty[s] / n:8.1f}% {missing[s]:10d}")
    print(f"   contexts reached: {len(seen)} of {len(brain.CTX)}")
    print("   ('empty row' = the solver never trained that spot; the bot uses pot odds there)")
    return sum(missing.values()) == 0


def check_upstream():
    print(f"\n3. organizer files vs {UPSTREAM}")
    ok = True
    for f in ORGANIZER_FILES:
        try:
            with urllib.request.urlopen(UPSTREAM + f, timeout=20) as r:
                theirs = r.read()
        except Exception as exc:
            print(f"   ??       {f}  (could not download: {exc})")
            ok = False
            continue
        with open(os.path.join(common.ROOT, f), "rb") as fh:
            mine = fh.read()
        same = mine == theirs
        ok &= same
        print(f"   {'same    ' if same else 'CHANGED '} {f}")
        if not same and f == "game.py":
            pattern = re.compile(r"^[A-Z_]+\s*=.*$", re.M)
            old, new = set(pattern.findall(mine.decode())), set(pattern.findall(theirs.decode()))
            for line in sorted(new - old):
                print(f"            upstream now has: {line}")
    if not ok:
        print("   -> pull the organizer changes, then rerun this check and the solver tests.")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hands", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--upstream", action="store_true", help="also diff organizer files against GitHub")
    a = p.parse_args()
    ok = check_constants()
    ok &= check_coverage(a.hands, a.seed)
    if a.upstream:
        ok &= check_upstream()
    print("\nPASSED" if ok else "\nFAILED: see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
