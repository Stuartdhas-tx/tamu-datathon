"""
opponent.py - what we have learned about the opponent this half.

Stats come from replaying each finished hand. The last observation of a hand
stops at our final decision, so whatever came after is inferred from the
result: a pot won without a showdown means they folded to our last raise, a
showdown after our raise means they called it.

Group 0 is preflop, group 1 is postflop.
"""

RAISE = 2  # int(Action.RAISE_HALF); RAISE_POT is 3


class OppStats:
    def __init__(self):
        self.faced = [0, 0]      # times they faced a bet from us
        self.folded = [0, 0]     # ... and folded
        self.reraised = [0, 0]   # ... and raised
        self.free = [0, 0]       # times they could bet with nothing to call
        self.bet = [0, 0]        # ... and did
        self.hands = 0

    def learn(self, obs, my_action: int, result) -> None:
        """Replay one hand. obs is our last observation of that hand, or None
        if we never got to act in it."""
        self.hands += 1
        if obs is None:      # no decision for us: they folded their button
            self.faced[0] += 1
            self.folded[0] += 1
            return
        my_bet, opp_bet = (1, 2) if obs.is_button else (2, 1)
        street = 0
        for a in obs.action_history:
            s = int(a.street)
            if s != street:
                street, my_bet, opp_bet = s, 0, 0
            if a.is_me:
                my_bet = a.street_total
                continue
            g = 0 if s == 0 else 1
            if my_bet > opp_bet:
                self.faced[g] += 1
                self.reraised[g] += int(a.action) >= RAISE
            else:
                self.free[g] += 1
                self.bet[g] += int(a.action) >= RAISE
            opp_bet = a.street_total

        # What happened after our final decision.
        g = 0 if int(obs.street) == 0 else 1
        i_raised = my_action >= RAISE
        if not result.went_to_showdown:
            if result.winner == "me":
                self.faced[g] += 1
                self.folded[g] += 1
        elif i_raised:
            self.faced[g] += 1
        elif my_action == 1 and obs.to_call == 0:
            acted_this_street = any(int(a.street) == int(obs.street) and not a.is_me
                                    for a in obs.action_history)
            if not acted_this_street:      # we checked first, they checked back
                self.free[g] += 1

    def fold_rate(self, g: int, prior_rate: float, weight: float = 25.0) -> float:
        return (self.folded[g] + prior_rate * weight) / (self.faced[g] + weight)

    def bet_rate(self, g: int, prior_rate: float, weight: float = 25.0) -> float:
        return (self.bet[g] + prior_rate * weight) / (self.free[g] + weight)

    def reraise_rate(self, g: int, prior_rate: float, weight: float = 25.0) -> float:
        return (self.reraised[g] + prior_rate * weight) / (self.faced[g] + weight)


def opp_decisions(obs, my_action: int):
    """Every decision the opponent made in a hand that reached showdown, as
    (street, context key, action). obs is our last observation of the hand.

    A context key is built exactly like brain.context_key, from their seat.
    Their last action can come after our final decision, so it is inferred:
    a showdown after our raise means they called, and a showdown after we
    checked first means they checked behind.
    """
    me = 0 if obs.is_button else 1
    them = 1 - me
    out = []
    street, seq, aggr, pot, ps = 0, 1, 0, 3, 3

    def key(s):
        return 1 + ((seq << 20) | (ps << 8) | (s << 4) | (them << 3) | (aggr << 1))

    for a in obs.action_history:
        s = int(a.street)
        if s != street:
            street, seq, ps = s, 1, pot
        act = int(a.action)
        if not a.is_me:
            out.append((s, key(s), act))
        seq = seq * 4 + act
        if act >= RAISE:
            aggr = (me if a.is_me else them) + 1
        pot += a.amount

    s = int(obs.street)
    if s != street:
        seq, ps = 1, pot
    if my_action >= RAISE:
        seq = seq * 4 + my_action
        aggr = me + 1
        out.append((s, key(s), 1))
    elif my_action == 1 and obs.to_call == 0 and seq == 1:
        seq = seq * 4 + 1
        out.append((s, key(s), 1))
    return out
