"""check_timing.py - how close does a bot get to the time limits?

    python practice/check_timing.py                    # agent.py
    python practice/check_timing.py exploit --slowdown 3

Measures on one thread, like the tournament (OMP_NUM_THREADS=1):
  * startup: a cold process loading the bot file (limit 15 s);
  * every decision by street (mean / p50 / p99 / max), and on_hand_end;
  * the time bank charged per half (first 1 ms of every decision free),
    on this machine and on an emulated machine --slowdown times slower.
The slow machine is emulated by making time.perf_counter run --slowdown
times faster while the bot plays, so bots that budget their own time (stop a
simulation at a deadline) behave as they really would on slower hardware.
The opponents create many decisions: a calling station (hands reach the
river) and a maniac (lots of raises).

Fails (exit 1) if the slow machine runs out of time bank or exceeds 1 s on a
decision; warns above half the bank.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from game import (ACTION_TIMEOUT_SECONDS, NETWORK_GRACE_SECONDS, STARTUP_TIMEOUT_SECONDS,  # noqa: E402
                  TIME_BANK_SECONDS, AgentConfig, Game, Street, button_for_deal, make_deal)

OPPONENTS = ("zoo:station", "zoo:maniac")
_real_clock = time.perf_counter


def cold_startup(path):
    code = ("import importlib.util, os, sys, time; t = time.perf_counter(); sys.path.insert(0, sys.argv[1]); "
            "sys.path.append(os.path.dirname(sys.argv[2])); "
            "s = importlib.util.spec_from_file_location('bot', sys.argv[2]); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print(time.perf_counter() - t)")
    out = subprocess.run([sys.executable, "-c", code, common.ROOT, path], cwd=common.ROOT,
                         capture_output=True, text=True, timeout=120)
    return float(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def measure(name, deals, slowdown):
    """Play halves against OPPONENTS on a machine `slowdown` times slower.
    Returns (latencies by street, [(opponent, half, decisions, bank used)], slowest)."""
    time.perf_counter = (lambda: slowdown * _real_clock()) if slowdown != 1 else _real_clock
    clock = time.perf_counter
    try:
        bot = common.resolve(name)
        by_street, halves, worst = defaultdict(list), [], 0.0
        for opp in OPPONENTS:
            for half in (1, 2):
                other = common.resolve(opp)
                me_seat = half - 1
                bot.new_half(AgentConfig(num_hands=deals))
                other.new_half(AgentConfig(num_hands=deals))
                bank, n = 0.0, 0
                for d in range(deals):
                    game = Game(make_deal(f"timing-{opp}", d), button_for_deal(d), hand_index=d)
                    while not game.is_terminal:
                        s = game.to_act
                        obs = game.observation(s)
                        if s != me_seat:
                            game.apply(other.act(obs))
                            continue
                        t0 = clock()
                        action = bot.act(obs)
                        el = clock() - t0
                        by_street[Street(obs.street).name].append(el)
                        bank += max(0.0, el - NETWORK_GRACE_SECONDS)
                        worst = max(worst, el)
                        n += 1
                        game.apply(action)
                    t0 = clock()
                    bot.end(game.hand_result(me_seat))
                    el = clock() - t0
                    by_street["on_hand_end"].append(el)
                    bank += el
                    other.end(game.hand_result(1 - me_seat))
                halves.append((opp, half, n, bank))
        return by_street, halves, worst
    finally:
        time.perf_counter = _real_clock


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bot", nargs="?", default="agent")
    p.add_argument("--deals", type=int, default=1000)
    p.add_argument("--slowdown", type=float, default=3.0,
                   help="how much slower the tournament machine might be (default 3x)")
    a = p.parse_args()
    k = a.slowdown

    if isinstance(common.resolve(a.bot), common.ModuleBot):
        t = cold_startup(common.resolve(a.bot).path)
        if t is None:
            print("startup: FAILED to load the bot file in a fresh process")
            sys.exit(1)
        print(f"startup (cold process, module load): {t:.2f} s   (limit {STARTUP_TIMEOUT_SECONDS:g} s,"
              f" x{k:g} slower machine: {t * k:.2f} s)")

    by_street, fast, _ = measure(a.bot, a.deals, 1.0)
    _, slow, worst = measure(a.bot, a.deals, k)

    print(f"\n{'call':12s} {'count':>7s} {'mean':>8s} {'p50':>8s} {'p99':>8s} {'max':>8s}   (ms, this machine)")
    for name in ("PREFLOP", "FLOP", "TURN", "RIVER", "on_hand_end"):
        xs = by_street.get(name, [])
        if xs:
            print(f"{name:12s} {len(xs):7d} {1000 * sum(xs) / len(xs):8.3f} {1000 * pct(xs, .5):8.3f} "
                  f"{1000 * pct(xs, .99):8.3f} {1000 * max(xs):8.3f}")

    print(f"\n{'opponent':14s} {'half':>4s} {'decisions':>9s} {'bank used':>10s} {f'x{k:g} slower':>11s}"
          f"   (of {TIME_BANK_SECONDS:g} s)")
    for (opp, half, n, used), (_, _, _, used_slow) in zip(fast, slow):
        print(f"{opp:14s} {half:4d} {n:9d} {used:9.2f}s {used_slow:10.2f}s")

    worst_bank = max(h[3] for h in slow)
    print()
    if worst_bank > TIME_BANK_SECONDS or worst > ACTION_TIMEOUT_SECONDS:
        print(f"FAILED: on a {k:g}x slower machine the bot would run out of time "
              f"(bank {worst_bank:.2f} s, slowest decision {worst:.2f} s).")
        sys.exit(1)
    if worst_bank > TIME_BANK_SECONDS / 2:
        print(f"WARNING: on a {k:g}x slower machine the bot would use {worst_bank:.2f} s of the "
              f"{TIME_BANK_SECONDS:g} s bank. Little safety margin.")
    else:
        print(f"PASSED: even {k:g}x slower, at most {worst_bank:.2f} s of the {TIME_BANK_SECONDS:g} s bank "
              f"per half; slowest decision there {worst * 1000:.1f} ms.")


if __name__ == "__main__":
    main()
