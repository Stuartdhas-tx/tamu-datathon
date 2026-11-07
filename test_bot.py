"""Failure paths and the rules that keep the counters safe."""
import importlib.util
import os
import random

import numpy as np
import pytest

from game import Action, AgentConfig, Game, make_deal

HERE = os.path.dirname(os.path.abspath(__file__))


def fresh_agent():
    spec = importlib.util.spec_from_file_location("agent_bot_test", os.path.join(HERE, "agent.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture
def agent():
    return fresh_agent()


def play(m, hands=150, seed="bot", raise_rate=.35):
    """Seat 0 is the bot, seat 1 a random raiser. Every action must be legal."""
    rng = random.Random(seed)
    mem = {}
    m.new_half(AgentConfig(num_hands=hands), mem)
    for h in range(hands):
        g = Game(make_deal(seed, h), button_seat=h % 2, hand_index=h)
        while not g.is_terminal:
            o = g.observation(g.to_act)
            if g.to_act == 0:
                a = m.choose_action(o, mem)
                assert a in o.legal_actions
            else:
                r = [x for x in o.legal_actions if x >= 2]
                a = rng.choice(r) if r and rng.random() < raise_rate else Action.CHECK_CALL
            g.apply(a)
        m.on_hand_end(g.hand_result(0), mem)
    return mem


def test_all_strategy_files_load(agent):
    assert len(agent.STRATEGIES) == 3 and agent.EVIDENCE is not None
    for t in agent.STRATEGIES:
        assert t.shape == agent.STRATEGIES[0].shape


def test_plays_with_no_tables_at_all(agent):
    agent.STRATEGIES = []
    agent.EVIDENCE = None
    play(agent)


def test_plays_when_features_crash(agent, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("feature failure")
    monkeypatch.setattr(agent.brain, "street_info", boom)
    play(agent)


def test_a_strange_reply_from_the_solver_cannot_be_illegal(agent, monkeypatch):
    monkeypatch.setattr(agent, "_decide", lambda o, m: Action.FOLD)
    mem = play(agent, hands=30)
    assert mem["used"] >= 0


def test_counters_stay_locked_without_evidence(agent):
    mem = {}
    agent.new_half(AgentConfig(), mem)
    for _ in range(500):
        assert agent._pick_arm(mem) == 0


def unlocked_memory(agent, honest, aggressive, balanced, showdowns):
    mem = {}
    agent.new_half(AgentConfig(), mem)
    mem["ll_models"][:] = -1e9
    mem["ll_models"][0] = honest
    mem["ll_models"][agent.NUM_HONEST] = aggressive
    mem["ll_balanced"] = balanced
    mem["showdowns"] = showdowns
    return mem


def test_gate_rules(agent):
    m = unlocked_memory(agent, 100, 0, 0, 10)
    assert agent._open_arms(m) == [0, 1]                   # clearly honest
    m = unlocked_memory(agent, 0, 100, 0, 10)
    assert agent._open_arms(m) == [0, 2]                   # clearly aggressive
    m = unlocked_memory(agent, 100, 90, 0, 10)
    assert agent._open_arms(m) == [0, 1]                   # aggressive must beat both clearly
    m = unlocked_memory(agent, 39, 0, 0, 2)
    assert agent._open_arms(m) == [0]                      # not enough evidence yet
    m = unlocked_memory(agent, 100, 0, 0, 40)
    assert agent._open_arms(m) == [0]                      # evidence too thin per showdown
    m = unlocked_memory(agent, 100, 100, 200, 10)
    assert agent._open_arms(m) == [0]                      # balanced fits best


def test_slow_machine_cuts_work_and_stops_optional_evidence(agent):
    mem = {}
    agent.new_half(AgentConfig(num_hands=1000), mem)
    mem["used"] = 0.0
    assert agent._work_scale(500, mem) == 1.0
    mem["used"] = 2.4          # 4.8 ms a hand with 0.6 s left for 500 hands
    assert agent._work_scale(500, mem) == 0.2
    assert agent._time_headroom(500, mem) < 1.0
    mem["used"] = 9.9          # past the target: must not crash or go negative in a bad way
    assert agent._work_scale(900, mem) == 0.2


def test_one_strategy_per_hand(agent):
    mem = {}
    agent.new_half(AgentConfig(), mem)
    g = Game(make_deal("s", 0), button_seat=0, hand_index=0)
    mem["arm"] = 2
    mem["arm_hand"] = 0
    o = g.observation(g.to_act)
    agent._decide(o, mem)
    assert mem["arm"] == 2
