"""check_robustness.py - try hard to make a bot crash, stall or play an illegal move.

    python practice/check_robustness.py                    # agent.py
    python practice/check_robustness.py exploit --hands 6000

Plays the bot against a "chaos" opponent that switches style every 25 hands
(random, raise-everything, call-everything, half-pot raise wars, folder), so
the bot meets the rare spots: raise cap reached, facing an all-in, a call
that puts it all-in, the big blind's option. Every observation and hand
result goes through the same JSON round trip as the real server.

Fails (exit 1) on any exception, illegal answer, or decision over 1 s, and
reports which rare spots were covered. Bots that catch their own errors and
quietly answer CHECK_CALL would hide bugs, so if the bot file has a
_decide(obs, memory) function it is also called directly: an exception there
means the strategy silently fell back, and also fails the check.
"""

import argparse
import os
import random
import sys
import time
import traceback
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from game import (ACTION_TIMEOUT_SECONDS, Action, AgentConfig, Game, HandResult,  # noqa: E402
                  Observation, Street, button_for_deal, default_action, make_deal)

C, H, P = Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
STYLES = {
    "random": lambda o, r: r.choice(o.legal_actions),
    "raise-everything": lambda o, r: P if P in o.legal_actions else C,
    "call-everything": lambda o, r: C,
    "half-pot wars": lambda o, r: H if H in o.legal_actions else C,
    "folder": lambda o, r: Action.FOLD if Action.FOLD in o.legal_actions and r.random() < .6 else C,
}
SPOTS = [
    "big blind option (free check preflop)",
    "first to act after the flop",
    "raise cap reached (4 raises)",
    "only FOLD / CHECK_CALL legal",
    "facing an all-in",
    "calling puts us all-in",
    "river, facing a bet",
]


def spots(o: Observation):
    found = []
    if o.street == Street.PREFLOP and o.to_call == 0:
        found.append(SPOTS[0])
    if o.street != Street.PREFLOP and not any(a.street == o.street for a in o.action_history):
        found.append(SPOTS[1])
    if o.raises_this_street >= 4:
        found.append(SPOTS[2])
    if set(o.legal_actions) == {Action.FOLD, C}:
        found.append(SPOTS[3])
    if o.opp_stack == 0 and o.to_call > 0:
        found.append(SPOTS[4])
    if 0 < o.my_stack <= o.to_call:
        found.append(SPOTS[5])
    if o.street == Street.RIVER and o.to_call > 0:
        found.append(SPOTS[6])
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bot", nargs="?", default="agent")
    p.add_argument("--hands", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    bot = common.resolve(a.bot)
    rng = random.Random(a.seed)
    styles = list(STYLES)
    coverage, failures, slowest, decisions = Counter(), [], 0.0, 0
    half_size = a.hands // 2

    for half in (1, 2):
        my_seat = half - 1
        try:
            bot.new_half(AgentConfig(num_hands=half_size))
        except Exception:
            failures.append(("new_half raised", traceback.format_exc()))
            break
        for d in range(half_size):
            style = STYLES[styles[(d // 25) % len(styles)]]
            game = Game(make_deal(f"robust{a.seed}-{half}", d), button_for_deal(d), hand_index=d)
            while not game.is_terminal:
                seat = game.to_act
                obs = Observation.from_dict(game.observation(seat).to_dict())
                if seat != my_seat:
                    game.apply(style(obs, rng))
                    continue
                coverage.update(spots(obs))
                decisions += 1
                t0 = time.perf_counter()
                try:
                    raw = bot.act(obs)
                except Exception:
                    failures.append((f"choose_action raised (hand {d}, half {half})", traceback.format_exc()))
                    raw = None
                elapsed = time.perf_counter() - t0
                slowest = max(slowest, elapsed)
                if elapsed > ACTION_TIMEOUT_SECONDS:
                    failures.append((f"decision took {elapsed:.2f}s (hand {d})", ""))
                probe = getattr(getattr(bot, "m", None), "_decide", None)
                if probe is not None:
                    try:
                        probe(obs, bot.mem)
                    except Exception:
                        failures.append((f"_decide raised, hidden by the bot's own try/except "
                                         f"(hand {d}, half {half})", traceback.format_exc()))
                action = common.coerce_action(raw.name if isinstance(raw, Action) else raw)
                if raw is not None and (action is None or action not in obs.legal_actions):
                    failures.append((f"illegal answer {raw!r} (hand {d}); legal: "
                                     f"{[x.name for x in obs.legal_actions]}", ""))
                    action = None
                game.apply(action if action is not None else default_action(obs.to_call))
            try:
                bot.end(HandResult.from_dict(game.hand_result(my_seat).to_dict()))
            except Exception:
                failures.append((f"on_hand_end raised (hand {d}, half {half})", traceback.format_exc()))
        hidden = getattr(bot, "mem", {}).get("errors", 0)   # bots that count their own caught errors
        if isinstance(hidden, int) and hidden > 0:
            failures.append((f"the bot caught {hidden} of its own errors in half {half} "
                             f"(memory['errors']) and played a fallback action", ""))

    print(f"bot: {a.bot}   hands: {2 * half_size}   decisions: {decisions}   "
          f"slowest decision: {slowest * 1000:.1f} ms")
    print("\nrare spots covered:")
    for s in SPOTS:
        print(f"  {coverage[s]:6d}  {s}{'' if coverage[s] else '   <- never reached'}")
    if failures:
        print(f"\nFAILED: {len(failures)} problem(s). First ones:")
        for title, tb in failures[:3]:
            print(f"\n--- {title}\n{tb}")
        sys.exit(1)
    print("\nPASSED: no exceptions, no illegal answers, no slow decisions.")


if __name__ == "__main__":
    main()
