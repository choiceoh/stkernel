#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""deneb fork: engine-level prefill JIT warmup for the GLM-5.3 lane.

PR #192's ledger leaves exactly one open prefill target: the COLD tax
(11-41% on the first request per length; warm throughput is already at
97% of the goal line). The tax is first-use JIT -- FLA chunk autotuners,
deepgemm M-buckets, TileLang big_fuse, chunk boundaries -- and the same
ledger's ladder shows repeated same-length requests collapsing onto the
warm number (35% spread at 2048 -> 0.9% at 8192).

So warm the way the measurements were warmed: real requests. Each length
in PREFILL_WARMUP_LENS fires REPS completions with EXACTLY that many
prompt tokens (server /tokenize + id trim, so the scheduled-token shape
matches production) and DISTINCT content per request (a shared prefix
would prefix-cache-hit and skip the compute entirely -- the ladder probe
enforces the same rule). rep1-vs-rep2 in the log is the receipt: if rep2
lands on the warm number, the tax was real and is now paid at boot
instead of on the first user.

Numerics-neutral: outputs are discarded, nothing is armed or compared.
Run by hand after a boot, or let the launcher do it (PREFILL_WARMUP=1).

    python3 launchers/prefill-warmup.py [--port 8000]
        [--lens 2048,4096,8192] [--reps 2] [--timeout-min 25]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

WORDS = (
    "kernel schedule lattice vector fabric quantum ledger beacon cascade "
    "meridian tundra cobalt harbor signal vantage prism anchor delta "
    "cascade lantern quiver summit ridge fathom echo zenith marrow "
    "thicket galley obsidian pinnacle wander catalyst fissure beacon"
).split()


def build_distinct_prompt(seed: int, approx_tokens: int) -> str:
    """Deterministic per-seed text, ~4 chars/token, no shared long prefix.

    The first 8 words are the seed's own (a shared header would let the
    prefix cache serve every later request from the first one's blocks).
    """
    rng = random.Random(seed)
    head = [WORDS[seed % len(WORDS)], str(seed)] + [
        rng.choice(WORDS) for _ in range(6)]
    body = [rng.choice(WORDS) for _ in range(max(1, approx_tokens))]
    return " ".join(head + body)


def trim_to_exact_tokens(text: str, want: int, tokenize) -> list[int]:
    """Server-side token ids trimmed to exactly `want` tokens."""
    ids = tokenize(text)
    if len(ids) < want:
        # rare (tokenizer overhead ~0): top up with repeats of the body
        while len(ids) < want:
            ids = ids + ids[8 : 8 + min(want - len(ids), len(ids) - 8)]
    return ids[:want]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lens", default="2048,4096,8192")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout-min", type=int, default=25)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",") if x.strip()]
    if not lens or any(n <= 0 for n in lens):
        print("ABORT: no positive lengths in --lens")
        return 1
    base = f"http://127.0.0.1:{args.port}"

    def post(path: str, payload: dict, timeout: int = 600):
        req = urllib.request.Request(
            base + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    deadline = time.time() + args.timeout_min * 60
    while True:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=10):
                break
        except Exception:
            if time.time() > deadline:
                print(f"ABORT: server not up within {args.timeout_min} min")
                return 1
            time.sleep(10)
    print("[prefill-warmup] server up; warming shapes:", lens)

    def tokenize(text: str) -> list[int]:
        out = post("/tokenize", {"prompt": text})
        ids = out.get("tokens") or out.get("prompt_token_ids")
        if not ids:
            raise RuntimeError(f"tokenize gave no ids: keys={list(out)}")
        return ids

    worst = 0.0
    for n in lens:
        for rep in range(1, args.reps + 1):
            seed = n * 1000 + rep
            text = build_distinct_prompt(seed, n)
            ids = trim_to_exact_tokens(text, n, tokenize)
            if len(ids) != n:
                print(f"ABORT: len={n} rep={rep} got {len(ids)} ids")
                return 1
            t0 = time.time()
            # This fork's /v1/completions ignores a top-level prompt_token_ids
            # (400 "Either prompt or prompt_embeds must be provided") and its
            # prompt field does not take the {"prompt_token_ids": ...} object
            # form either -- a bare list of ints IS the token-id form here.
            post("/v1/completions",
                 {"prompt": ids, "max_tokens": 1,
                  "temperature": 0.0, "ignore_eos": True})
            dt = time.time() - t0
            tps = n / dt if dt > 0 else 0.0
            worst = max(worst, dt)
            print(f"[prefill-warmup] len={n} rep={rep}: {dt:.2f}s "
                  f"({tps:,.0f} tok/s prefill)")
    print("[prefill-warmup] done — slowest request "
          f"{worst:.2f}s; first-user prefill is now warm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
