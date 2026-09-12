"""What an agent loop actually gets from the prefix cache, on the real serve path.

An agent resends its whole conversation every step: system prompt + tool schemas, then turn after
turn of (assistant tool call, tool result). The prompt only grows, and every step is a blocking
dependency for the agent, so the question that decides whether this engine is pleasant to build an
agent on is: how much of step k's prompt does step k actually have to compute?

Runs the fake engine over the production shape (BLOCK 768, CHUNK 6,912) with no GPU. It measures
reuse, not latency -- latency needs a window, and production is this engine.
"""
from __future__ import annotations

import concurrent.futures
import json
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from engine.base.kv import BlockPool, SlotPool                              # noqa: E402
from engine.base.prefix import PrefixCache                                  # noqa: E402
from engine.base.record import Ring                                        # noqa: E402
from engine.base.runner import Runner, STEP_RECORD                         # noqa: E402
from engine.base.scheduler import Contract                                 # noqa: E402
from engine.base.serve import Server                                       # noqa: E402
from test_engine_serve import Comm, Engine, Tokenizer, drive               # noqa: E402

BLOCK, CHUNK, SNAPSHOTS = 768, 6912, 96            # profiles/glm53/facts.py and boot.PREFIX_SNAPSHOTS


def agent_server(rows=4, blocks=512):
    engine = Engine(rows + 1)
    cache = PrefixCache(BLOCK, CHUNK, SNAPSHOTS)
    runner = Runner(engine, Contract(BLOCK, CHUNK, 0, 0.0, rows), BlockPool(blocks, BLOCK, rows, blocks),
                    SlotPool(rows + 1), Ring(64, STEP_RECORD.size), prefix=cache)
    s = Server(engine, runner, Comm(), host="127.0.0.1", port=0, max_pending=64)
    s.tok = Tokenizer()
    s.model_name = "fake"
    # A chat template that is append-only in its messages, which is what every real one is for a growing
    # conversation: the rendering of turns 1..k is a prefix of the rendering of turns 1..k+1.
    s.chat = lambda messages, kwargs, *, generation_prompt=True, continue_final=False: "".join(
        (m.get("content") or "") for m in messages)
    return s


def main(turns=8, system=6000, call=900, result=1500):
    s = agent_server()
    httpd = s._serve_http()
    url = f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"

    def post(body):
        req = urllib.request.Request(url, data=json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.load(r)

    # the system block: an agent's instructions plus its tool schemas, the part that never changes
    messages = [{"role": "system", "content": "s" * system},
                {"role": "user", "content": "u" * 400}]
    rows = []
    try:
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            for turn in range(1, turns + 1):
                out = drive(s, pool.submit(post, {"messages": list(messages), "max_tokens": 2}))
                usage = out["usage"]
                prompt = usage["prompt_tokens"]
                # what the client is told, not what the engine knows: the whole point is that these are the same number
                reused = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                rows.append((turn, prompt, reused, len(s.runner.prefix.entries),
                             getattr(s.runner, "snapshot_self_evicts", 0), getattr(s.runner.prefix, "snapshot_denials", 0)))
                # the step's answer and the tool's reply, appended for the next step
                messages.append({"role": "assistant", "content": chr(97 + turn % 26) * call})
                messages.append({"role": "tool", "content": chr(65 + turn % 26) * result})
    finally:
        httpd.shutdown(); httpd.server_close()

    print(f"  agent loop: system {system}, +{call} answer +{result} tool result per turn, BLOCK {BLOCK}\n")
    print(f"  {'turn':>4} {'prompt':>8} {'computed':>9} {'reused':>8} {'reuse':>7} {'entries':>8} {'self-evict':>11} {'denied':>7}")
    for turn, prompt, reused, entries, evicts, denied in rows:
        print(f"  {turn:>4} {prompt:>8} {prompt - reused:>9} {reused:>8} {reused / prompt:>6.1%} "
              f"{entries:>8} {evicts:>11} {denied:>7}")
    total = sum(p for _, p, _, _, _, _ in rows)
    saved = sum(r for _, _, r, _, _, _ in rows)
    print(f"\n  whole loop: {total} prompt tokens, {total - saved} computed, {saved / total:.1%} reused")


if __name__ == "__main__":
    main(*[int(a) for a in sys.argv[1:]])
