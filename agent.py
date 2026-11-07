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

import random
import time

import numpy as np

import brain

# A missing or broken strategy file must never stop the server from starting:
# without it the bot still plays sound pot odds through _fallback().
try:
    STRATEGY = np.load(os.path.join(brain.DATA, "strategy.npy"))
except Exception:
    STRATEGY = None
PURIFY = 0.04                # never play an action the solver gives under 4%
TIME_LIMIT = 6.0             # seconds of our own compute we allow per half


def new_half(config: AgentConfig, memory: dict) -> None:
    """Fresh state for a half."""
    memory["rng"] = random.Random()
    memory["key"] = None
    memory["info"] = None
    memory["used"] = 0.0
    memory["num_hands"] = config.num_hands


def _street_info(obs: Observation, memory: dict):
    """Card bucket for this street, computed once per street."""
    key = (obs.hand_index, len(obs.board))
    if memory["key"] != key:
        # Slow down gracefully if the machine is much slower than expected.
        left = max(1, memory["num_hands"] - obs.hand_index)
        spare = (TIME_LIMIT - memory["used"]) / left
        scale = 1.0 if spare > 0.002 else 0.5 if spare > 0.0008 else 0.2
        memory["info"] = brain.street_info(obs.hole_cards, obs.board, scale)
        memory["key"] = key
    return memory["info"]


def _strategy(obs: Observation, bucket: int):
    """Solver probabilities for the legal actions, or None if unknown."""
    ctx = brain.CTX.get(brain.context_key(obs))
    if ctx is None or STRATEGY is None:
        return None
    row = STRATEGY[ctx, bucket]
    probs = [(a, int(row[int(a)])) for a in obs.legal_actions]
    total = sum(p for _, p in probs)
    if total == 0:
        return None
    kept = [(a, p) for a, p in probs if p >= PURIFY * total] or probs
    return kept


def _fallback(obs: Observation, strength: float) -> Action:
    """Simple pot odds, used only if the table has no answer."""
    if obs.to_call == 0:
        return Action.CHECK_CALL
    need = obs.to_call / (obs.pot + obs.to_call)
    if strength >= need + 0.1:
        return Action.CHECK_CALL
    return Action.FOLD


def _decide(obs: Observation, memory: dict) -> Action:
    bucket, strength, _ = _street_info(obs, memory)
    probs = _strategy(obs, bucket)
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
    memory["used"] = memory.get("used", 0.0) + time.perf_counter() - start
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    pass

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
