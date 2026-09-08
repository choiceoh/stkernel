#!/usr/bin/env python3
"""Does the long-prefill engine death follow prefix caching? (40차)

Two deaths on 2026-09-08/09 carry the same fingerprint, both on arms with NO
knobs set, during onepass's 128K stage:

    total_num_scheduled_tokens = 6912 (= 3 x the 2304 block, the APC align-mode
    chunk), at a block-aligned deep prefix (89,856 = 39 blocks; 76,032 = 33),
    new_block_ids_to_zero six blocks in two runs of three, KV at 11-13% (so not
    memory), and Worker proc VllmWorker-0 dying with no Python traceback.

With PREFIX_CACHE=0 the mamba cache leaves "align" mode and the chunk is no
longer clipped to a block multiple. So: send the same shape repeatedly on each
side and count how far it gets. The death is intermittent -- a defaults arm
passed 128K 9/9 earlier the same day -- so one request decides nothing and this
sends N.

Every prompt carries a fresh nonce FIRST, so prefix caching cannot hit whatever
the setting is: what is under test is the CHUNKING the setting selects, not
cache hits.

  python3 probes/prefill_long_repeat.py [--ctx 128000] [--n 6] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
try:
    from bench_common import resolve_model
except Exception:                                   # running from a copy outside the repo
    def resolve_model(default: str = "glm-5.3-flash") -> str:
        try:
            with urllib.request.urlopen(BASE + "/v1/models", timeout=5) as r:
                return json.loads(r.read().decode())["data"][0]["id"]
        except Exception:
            return default

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8000")
WORDS = ["reactor", "harbor", "lattice", "quarry", "ember", "meridian", "syntax",
         "granite", "voltage", "cirrus", "tundra", "beacon", "ledger", "prism"]


def one(model: str, ctx: int, rng: random.Random) -> tuple[int, float]:
    nonce = f"{rng.getrandbits(48):012x}"
    text = " ".join(rng.choice(WORDS) for _ in range(int(ctx / 1.3)))
    body = json.dumps({"model": model, "max_tokens": 1, "temperature": 0,
                       "messages": [{"role": "user", "content": f"{nonce} {text} End."}],
                       "chat_template_kwargs": {"thinking": False}}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return int(out["usage"]["prompt_tokens"]), time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=128000)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    model = resolve_model()
    rng = random.Random(20260909)
    print(f"model={model} ctx={args.ctx} n={args.n}")
    runs, died = [], None
    for i in range(args.n):
        try:
            tok, wall = one(model, args.ctx, rng)
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:400]
            except Exception:
                detail = "(no body)"
            died = {"at": i + 1, "code": exc.code, "detail": detail}
            print(f"  request {i + 1}/{args.n}: DIED HTTP {exc.code}: {detail}", file=sys.stderr)
            break
        except Exception as exc:                    # connection refused = engine gone
            died = {"at": i + 1, "code": 0, "detail": f"{type(exc).__name__}: {exc}"}
            print(f"  request {i + 1}/{args.n}: DIED {type(exc).__name__}: {exc}", file=sys.stderr)
            break
        runs.append({"tok": tok, "wall": wall})
        print(f"  request {i + 1}/{args.n}: tok {tok}  wall {wall:6.1f}s  {tok / wall:,.0f} tok/s")
    ok = len(runs)
    print(f"\nSURVIVED {ok}/{args.n} long prefills" + ("" if died is None else f"; died on #{died['at']}"))
    if runs:
        rates = sorted(r["tok"] / r["wall"] for r in runs)
        print(f"  median {rates[len(rates) // 2]:,.0f} tok/s over {ok} runs")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"ctx": args.ctx, "n": args.n, "survived": ok, "died": died,
                       "runs": runs, "when": time.strftime("%F %T")}, fh, indent=1)
        print(f"wrote {args.json}")
    return 0 if died is None else 1


if __name__ == "__main__":
    sys.exit(main())
