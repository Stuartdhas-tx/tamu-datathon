"""The Python betting replay must agree with the real game on every context."""
import random

import brain
from game import Action, Game, make_deal
from models import decode, replay


def test_replay_matches_game():
    rng = random.Random(3)
    for h in range(4000):
        g = Game(make_deal("m", h), button_seat=h % 2, hand_index=h)
        w = rng.choice([.2, .5, .9])
        while not g.is_terminal:
            o = g.observation(g.to_act)
            street, pl, ps, aggr, d = decode(brain.context_key(o))
            st = replay(street, ps, d)
            assert st["pot"] == o.pot and st["to_call"] == o.to_call
            assert st["can_raise"] == (Action.RAISE_POT in o.legal_actions)
            assert pl == (0 if o.is_button else 1)
            raises = [a for a in o.legal_actions if a >= 2]
            g.apply(rng.choice(raises) if raises and rng.random() < w else Action.CHECK_CALL)
