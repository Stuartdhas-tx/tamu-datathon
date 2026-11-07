"""OppStats must agree with ground truth taken straight from the game engine."""
import random

from game import Action, Game, make_deal
from opponent import OppStats


def play(hands, raise_rate, fold_rate, seed):
    rng = random.Random(seed)
    stats, truth = OppStats(), OppStats()
    for h in range(hands):
        g = Game(make_deal(seed, h), button_seat=h % 2, hand_index=h)
        last, mine = None, None
        while not g.is_terminal:
            s = g.to_act
            o = g.observation(s)
            raises = [a for a in o.legal_actions if a >= 2]
            if raises and rng.random() < raise_rate:
                a = rng.choice(raises)
            elif Action.FOLD in o.legal_actions and rng.random() < fold_rate:
                a = Action.FOLD
            else:
                a = Action.CHECK_CALL
            if s == 0:
                last, mine = o, int(a)
            else:   # seat 1 is "the opponent": record the truth
                grp = 0 if int(o.street) == 0 else 1
                if o.to_call > 0:
                    truth.faced[grp] += 1
                    truth.folded[grp] += a == Action.FOLD
                    truth.reraised[grp] += a >= 2
                else:
                    truth.free[grp] += 1
                    truth.bet[grp] += a >= 2
            g.apply(a)
        stats.learn(last, mine, g.hand_result(0))
    return stats, truth


def test_stats_match_engine_truth():
    for seed, (r, f) in enumerate([(.3, .2), (.6, .1), (.1, .5), (.9, .05), (0, .3)]):
        s, t = play(3000, r, f, seed)
        assert s.folded == t.folded and s.reraised == t.reraised, (s.__dict__, t.__dict__)
        assert s.bet == t.bet, (s.__dict__, t.__dict__)
        for g in (0, 1):
            # Only the rare hand where they act after our last decision without
            # it being visible can differ, so require near-exact agreement.
            assert abs(s.faced[g] - t.faced[g]) <= 0.02 * max(1, t.faced[g]), (s.__dict__, t.__dict__)
            assert abs(s.free[g] - t.free[g]) <= 0.02 * max(1, t.free[g]), (s.__dict__, t.__dict__)
