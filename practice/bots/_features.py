"""_features.py - an opponent's behaviour profile, built from what our bot sees.

Shared by the selector bot (online, during the first hands of a half) and by
train_selector.py (offline), so both describe opponents the same way.

    fold_pre / fold_post        folds when facing our bet (OppStats)
    reraise_pre / reraise_post  raises when facing our bet
    bet_pre / bet_post          bets when nothing to call
    showdown                    share of hands that reach showdown
    looseness                   preflop strength of the hands they show down
    fold_preflop                share of hands they give up preflop
    risk                        std. dev. of our result per hand (bb): pot sizes
Rates are shrunk toward solver-like priors so a few hands can't produce
extreme values.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import brain  # noqa: E402
from opponent import OppStats  # noqa: E402

NAMES = ["fold_pre", "fold_post", "reraise_pre", "reraise_post", "bet_pre", "bet_post",
         "showdown", "looseness", "fold_preflop", "risk"]
WEIGHT = 10  # prior pseudo-observations


class Profile:
    def __init__(self):
        self.stats = OppStats()
        self.hands = self.showdowns = self.pre_folds = 0
        self.shown, self.net, self.net2 = [], 0.0, 0.0

    def learn(self, last, result):
        """last = (our last Observation of the hand, our action as int) or None."""
        if last is not None and last[0].hand_index == result.hand_index:
            self.stats.learn(last[0], last[1], result)
        else:
            self.stats.learn(None, 0, result)
        self.hands += 1
        self.showdowns += bool(result.went_to_showdown)
        if not result.went_to_showdown and result.winner == "me" and len(result.board) == 0:
            self.pre_folds += 1
        if result.opp_hole:
            self.shown.append(float(brain.PF_MU[brain.hand_index(sorted(result.opp_hole))]))
        bb = result.my_net / 2.0
        self.net += bb
        self.net2 += bb * bb

    def vector(self):
        s, w = self.stats, WEIGHT
        n = max(1, self.hands)
        mean = self.net / n
        return [
            s.fold_rate(0, 0.175, w), s.fold_rate(1, 0.496, w),
            s.reraise_rate(0, 0.464, w), s.reraise_rate(1, 0.140, w),
            s.bet_rate(0, 0.491, w), s.bet_rate(1, 0.352, w),
            (self.showdowns + 0.3 * w) / (n + w),
            (sum(self.shown) + 0.5 * w) / (len(self.shown) + w),
            (self.pre_folds + 0.15 * w) / (n + w),
            math.sqrt(max(0.0, self.net2 / n - mean * mean)),
        ]
