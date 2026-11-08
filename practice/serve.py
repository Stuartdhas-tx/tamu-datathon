"""serve.py - run any practice bot as an HTTP server, for matches through the real judge.

    python practice/serve.py exploit                     # port 5010 (or $PORT)
    python practice/serve.py zoo:maniac --port 5011
    python judge_engine.py --agent1 5008 --agent2 5010   # agent.py vs exploit

Uses agent.py's own server plumbing, in a separate copy, so bots that build
on agent.py keep calling the original functions. Listens on 127.0.0.1 only.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import uvicorn  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bot")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5010")))
    a = p.parse_args()

    server = common.load_module(os.path.join(common.ROOT, "agent.py"))
    bot = common.resolve(a.bot)
    if isinstance(bot, common.ModuleBot):
        m = common.load_module(bot.path)
        server.new_half, server.choose_action, server.on_hand_end = m.new_half, m.choose_action, m.on_hand_end
    else:
        server.new_half = lambda config, memory: bot.new_half(config)
        server.choose_action = lambda obs, memory: bot.act(obs)
        server.on_hand_end = lambda result, memory: bot.end(result)
    server.PARTICIPANT, server.AGENT_NAME = "practice", a.bot
    print(f"serving {a.bot} on http://127.0.0.1:{a.port}")
    uvicorn.run(server.app, host="127.0.0.1", port=a.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
