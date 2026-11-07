"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Team Tamu Datathon"
AGENT_NAME = "ShortStackSolver"


# ===== YOUR CODE HERE =====

import math
import random
import time

import numpy as np

import brain
from opponent import OppStats, opp_decisions

PURIFY = 0.04                # never play an action the solver gives under 4%
# The judge also charges for HTTP and JSON work we cannot see from in here, and on
# a slow machine that nearly doubles our own compute. Running out of bank means
# check-or-fold for the rest of the half, so we aim well below the 8 seconds.
TIME_TARGET = 3.0            # seconds of our own compute to aim for per half


def _load_strategies() -> list:
    """Solved strategies. A missing file must never stop the server starting:
    with no tables the bot still plays sound pot odds through _fallback()."""
    tables = []
    for name in ("strategy.npy", "counter.npy", "counter_aggressive.npy"):
        try:
            tables.append(np.load(os.path.join(brain.DATA, name)))
        except Exception:
            pass
    return tables


def _log_probs(p: np.ndarray) -> np.ndarray:
    """Smoothed log probabilities, so one surprising action is never fatal."""
    total = p.sum(axis=-1, keepdims=True)
    q = np.where(total > 0, p / np.maximum(total, 1e-9), 0.25)
    return np.log(0.9 * q + 0.1 * 0.25).astype(np.float32)


def _load_evidence():
    """(opponent models, balanced model) as log probabilities, or None.
    models.npz holds NUM_HONEST honest models, then the aggressive ones."""
    try:
        models = np.load(os.path.join(brain.DATA, "models.npz"))["models"]
        return _log_probs(models.astype(np.float32) / 255), _log_probs(STRATEGIES[0].astype(np.float32) / 255)
    except Exception:
        return None


STRATEGIES = _load_strategies()
EVIDENCE = _load_evidence()

# Arm 0 is the near-equilibrium strategy that nobody beats by much. Arm 1
# counters honest opponents and arm 2 counters over-aggressive ones. Each counter
# earns several times what arm 0 does against its target, and loses badly to
# everyone else.
#
# Chips alone are too noisy to tell these apart against a strong opponent, so
# the counters stay locked until the cards shown at showdown identify the
# opponent: each of their actions is scored under the honest models, the
# aggressive models and the balanced strategy. A counter unlocks only when its
# family fits better than both others. Once unlocked, the chips each arm wins
# decide, which protects us if the evidence was wrong.
#                  mean chips per hand, weight (pseudo hands)
ARM_PRIORS = ((0.5, 60.0), (2.0, 20.0), (2.0, 20.0))
NUM_HONEST = 6               # models.npz: honest models first, aggressive after
HAND_SD = 16.0               # chips; floor on an arm's per-hand spread
UNLOCK_TOTAL = 40.0          # evidence needed to unlock the counter ...
UNLOCK_PER_SHOWDOWN = 3.0    # ... and how strong it must be on average
EVIDENCE_DECAY = 0.98        # per showdown, so a change of style closes the gate


def new_half(config: AgentConfig, memory: dict) -> None:
    """Fresh state for a half."""
    memory["rng"] = random.Random()
    memory["key"] = None
    memory["info"] = None
    memory["used"] = 0.0
    memory["num_hands"] = config.num_hands
    memory["stats"] = OppStats()
    memory["last_obs"] = None
    memory["last_action"] = 1
    memory["arm_hand"] = -1
    memory["arm"] = 0
    n = max(1, len(STRATEGIES))
    memory["arms"] = [[0, 0.0, 0.0] for _ in range(n)]   # hands, sum, sum of squares
    memory["arm_log"] = [0] * n
    memory["ll_models"] = np.zeros(EVIDENCE[0].shape[0]) if EVIDENCE else None
    memory["ll_balanced"] = 0.0
    memory["showdowns"] = 0.0


def _open_arms(memory: dict) -> list:
    """The strategies the showdowns so far allow against this opponent."""
    arms = [0]
    ll = memory["ll_models"]
    if ll is None or memory["showdowns"] <= 0:
        return arms
    honest = float(ll[:NUM_HONEST].max())
    aggressive = float(ll[NUM_HONEST:].max())
    enough = max(UNLOCK_TOTAL, UNLOCK_PER_SHOWDOWN * memory["showdowns"])
    balanced = memory["ll_balanced"]
    # The honest counter only has to fit better than the aggressive one.
    if honest - balanced >= enough and honest >= aggressive:
        arms.append(1)
    # The aggressive counter loses most when wrong, so it must beat both clearly.
    if len(STRATEGIES) > 2 and min(aggressive - balanced, aggressive - honest) >= enough:
        arms.append(2)
    return arms


def _weigh_showdown(obs: Observation, my_action: int, result: HandResult, memory: dict) -> None:
    """Score the opponent's actions in a shown-down hand under each model."""
    models, balanced = EVIDENCE
    memory["ll_models"] *= EVIDENCE_DECAY
    memory["ll_balanced"] *= EVIDENCE_DECAY
    memory["showdowns"] = memory["showdowns"] * EVIDENCE_DECAY + 1
    buckets = {}
    for street, key, action in opp_decisions(obs, my_action):
        ctx = brain.CTX.get(key)
        if ctx is None:
            continue
        if street not in buckets:
            shown = result.board[:(0, 3, 4, 5)[street]]
            buckets[street] = brain.street_info(result.opp_hole, shown, 0.5)[0]
        b = buckets[street]
        memory["ll_models"] += models[:, ctx, b, action]
        memory["ll_balanced"] += float(balanced[ctx, b, action])


def _arm_posterior(arm: int, memory: dict):
    """(mean, standard error) of an arm's chips per hand."""
    n, total, squares = memory["arms"][arm]
    mean0, weight = ARM_PRIORS[arm] if arm < len(ARM_PRIORS) else ARM_PRIORS[-1]
    mean = (mean0 * weight + total) / (weight + n)
    var = HAND_SD ** 2
    if n >= 30:
        var = max(var, squares / n - (total / n) ** 2)
    return mean, math.sqrt(var / (weight + n))


def _pick_arm(memory: dict) -> int:
    """Thompson sampling: play each arm as often as it might be the best."""
    arms = _open_arms(memory)
    if len(arms) == 1:
        return 0
    rng = memory["rng"]
    best, best_draw = 0, -1e18
    for arm in arms:
        mean, se = _arm_posterior(arm, memory)
        draw = rng.gauss(mean, se)
        if draw > best_draw:
            best, best_draw = arm, draw
    return best


def _time_headroom(hand_index: int, memory: dict) -> float:
    """Pace we can afford over the pace we are running at. Below 1 means that
    at this rate we would miss TIME_TARGET by the end of the half."""
    if hand_index < 20:
        return 1.0
    pace = memory["used"] / hand_index
    left = max(1, memory["num_hands"] - hand_index)
    return (TIME_TARGET - memory["used"]) / left / max(pace, 1e-6)


def _work_scale(hand_index: int, memory: dict) -> float:
    """Spend less on simulation as soon as the half looks too slow."""
    room = _time_headroom(hand_index, memory)
    return 1.0 if room >= 1.0 else 0.5 if room >= 0.6 else 0.2


def _street_info(obs: Observation, memory: dict):
    """Card bucket for this street, computed once per street."""
    key = (obs.hand_index, len(obs.board))
    if memory["key"] != key:
        memory["info"] = brain.street_info(obs.hole_cards, obs.board, _work_scale(obs.hand_index, memory))
        memory["key"] = key
    return memory["info"]


def _strategy(obs: Observation, bucket: int, arm: int):
    """Solver probabilities for the legal actions, or None if unknown."""
    ctx = brain.CTX.get(brain.context_key(obs))
    if ctx is None or arm >= len(STRATEGIES):
        return None
    row = STRATEGIES[arm][ctx, bucket]
    probs = [(a, int(row[int(a)])) for a in obs.legal_actions]
    total = sum(p for _, p in probs)
    if total == 0:
        return None
    return [(a, p) for a, p in probs if p >= PURIFY * total] or probs


def _fallback(obs: Observation, strength: float) -> Action:
    """Simple pot odds, used only if the tables have no answer."""
    if obs.to_call == 0:
        return Action.CHECK_CALL
    need = obs.to_call / (obs.pot + obs.to_call)
    if strength >= need + 0.1:
        return Action.CHECK_CALL
    return Action.FOLD


def _decide(obs: Observation, memory: dict) -> Action:
    if memory["arm_hand"] != obs.hand_index:      # one strategy per hand
        memory["arm_hand"] = obs.hand_index
        memory["arm"] = _pick_arm(memory)
    bucket, strength, _ = _street_info(obs, memory)
    probs = _strategy(obs, bucket, memory["arm"])
    if probs is None:
        probs = _strategy(obs, bucket, 0)
    if probs is None:
        return _fallback(obs, strength)
    x = memory["rng"].random() * sum(p for _, p in probs)
    for action, p in probs:
        x -= p
        if x <= 0:
            return action
    return probs[-1][0]


def choose_action(obs: Observation, memory: dict) -> Action:
    """Every path returns a legal action: an error here can never cost a forfeit."""
    start = time.perf_counter()
    try:
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        action = Action.CHECK_CALL
    memory["last_obs"] = obs
    memory["last_action"] = int(action)
    memory["used"] = memory.get("used", 0.0) + time.perf_counter() - start
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Credit the hand to the strategy that played it and weigh any showdown."""
    start = time.perf_counter()
    try:
        obs = memory["last_obs"]
        played = obs is not None and obs.hand_index == result.hand_index
        memory["stats"].learn(obs if played else None, memory["last_action"], result)
        if played:
            arm = memory["arms"][memory["arm"]]
            arm[0] += 1
            arm[1] += result.my_net
            arm[2] += result.my_net * result.my_net
            memory["arm_log"][memory["arm"]] += 1
            # Only when there is time to spare: this is a bonus, not a need.
            if (EVIDENCE and result.opp_hole is not None
                    and _time_headroom(result.hand_index, memory) >= 1.0):
                _weigh_showdown(obs, memory["last_action"], result, memory)
    except Exception:
        pass
    memory["used"] = memory.get("used", 0.0) + time.perf_counter() - start

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
