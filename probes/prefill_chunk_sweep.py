#!/usr/bin/env python3
"""What a prefill step actually costs, as a function of chunk and context
(40차; 39차 DF4's leftover: "프리필 스텝 고정 비용(~0.25 s@32K) 자체 절감").

The ledger has two points -- a 1,152-token chunk step is 0.64 s at 32K and
1.3 s at 100K -- and a residual named "fixed cost 0.25 s" that was DERIVED by
subtracting the solo 8,192-chunk rate (2,950 tok/s), not measured. Substituting
both points into ``T = a + b*C + c*ctx`` makes ``a`` come out NEGATIVE, so the
name is probably wrong and the term that actually hurts is the one proportional
to the prefix each chunk re-reads (the indexer's top-k over the whole context
plus MLA over the KV). This separates them:

    T_step(C, ctx) = a + b*C + c*ctx
    wall(C, ctx)   = n*(a + b*C) + c*ctx^2/(2C),   n = steps for the request

a  host + launch glue + collectives that every step pays regardless of shape
b  per-token work (what a big chunk amortises well)
c  per-step work proportional to the PREFIX -- the structural cost of chunking

Sweeping C needs no boot per size: ``VLLM_GLM53_SCHED_CHUNK_FILE`` (the
decode-first scheduler's dev instrument) pins the chunk from a file, and that
file lives on the container's /prof mount, so this script writes it directly.
Requests are solo (no decoder): the small-chunk penalty is a pure-prefill
property, and a solo request keeps the cadence, the floor and the decoders out
of the number.

Run ON the head (srv2), against an idle boot that has DECODE_FIRST=1 and
VLLM_GLM53_SCHED_CHUNK_FILE set. PREFIX_CACHE=0 is expected; every request
carries a fresh nonce anyway.

  python3 probes/prefill_chunk_sweep.py [--ctx 32000,128000]
      [--chunks 1152,2304,4608,8192] [--reps 2] [--chunk-file PATH] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
from bench_common import resolve_model  # noqa: E402

BASE = os.environ.get("BENCH_BASE", "http://127.0.0.1:8000")
WORDS = [
    "reactor", "harbor", "lattice", "quarry", "ember", "meridian", "syntax",
    "granite", "voltage", "cirrus", "tundra", "beacon", "ledger", "prism",
    "cobalt", "willow", "cascade", "anvil", "nocturne", "vellum",
]


def metrics() -> str:
    try:
        return urllib.request.urlopen(BASE + "/metrics", timeout=5).read().decode()
    except Exception:
        return ""


def histogram(text: str, name: str) -> tuple[float, float]:
    """(count, sum) of a vLLM histogram. vllm:iteration_tokens_total counts one
    observation per ENGINE STEP, so its count is the step count -- the honest
    denominator, instead of assuming ceil(ctx / chunk) (chunk ends also align
    to the 2304-token block)."""
    out = []
    for suffix in ("_count", "_sum"):
        m = re.search(r"^vllm:%s%s\{[^}]*\}\s+([0-9.e+]+)" % (re.escape(name), suffix), text, re.M)
        out.append(float(m.group(1)) if m else 0.0)
    return out[0], out[1]


def set_chunk(path: str, chunk: int) -> None:
    """0 = off = the boot's own chunking (MAX_BATCHED)."""
    with open(path, "w") as fh:
        fh.write(f"{chunk}\n")
    # The scheduler re-reads at most 5x/s, and only from inside a step; an idle
    # engine takes no steps, so the next request reads the new value on its
    # first one. The pause is for the case where a request just ended.
    time.sleep(0.5)


def prefill(model: str, ctx_tokens: int, rng: random.Random) -> tuple[int, float]:
    """One fresh prefill, max_tokens=1. Returns (prompt_tokens, wall)."""
    nonce = f"{rng.getrandbits(48):012x}"
    text = " ".join(rng.choice(WORDS) for _ in range(int(ctx_tokens / 1.3)))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": f"{nonce} {text} End."}],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return int(out["usage"]["prompt_tokens"]), time.time() - t0


def fit(rows: list[dict]) -> dict:
    """Least squares for a, b, c over wall = n*a + n*C*b + (ctx^2/2C)*c.

    Solved with the normal equations (3x3, Gaussian elimination) so the probe
    needs no numpy on the head."""
    cols = []
    rhs = []
    for r in rows:
        n = r["steps"]
        cols.append([n, n * r["chunk"], r["prompt_tokens"] ** 2 / (2.0 * r["chunk"])])
        rhs.append(r["wall"])
    if len(cols) < 3:
        return {}
    ata = [[sum(cols[k][i] * cols[k][j] for k in range(len(cols))) for j in range(3)] for i in range(3)]
    atb = [sum(cols[k][i] * rhs[k] for k in range(len(cols))) for i in range(3)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(ata[r][i]))
        if abs(ata[p][i]) < 1e-12:
            return {}
        ata[i], ata[p] = ata[p], ata[i]
        atb[i], atb[p] = atb[p], atb[i]
        for r in range(i + 1, 3):
            f = ata[r][i] / ata[i][i]
            for c in range(i, 3):
                ata[r][c] -= f * ata[i][c]
            atb[r] -= f * atb[i]
    x = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        x[i] = (atb[i] - sum(ata[i][j] * x[j] for j in range(i + 1, 3))) / ata[i][i]
    return {"a_s": x[0], "b_s_per_tok": x[1], "c_s_per_ctx_tok": x[2]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32000,128000")
    ap.add_argument("--chunks", default="1152,2304,4608,8192")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--chunk-file", default="/home/choiceoh/vllm-prof/sched_chunk")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    ctxs = [int(v) for v in args.ctx.split(",") if v.strip()]
    chunks = [int(v) for v in args.chunks.split(",") if v.strip()]
    d = os.path.dirname(args.chunk_file)
    if not os.path.isdir(d):
        print(f"chunk file directory missing: {d} (is this the head node?)", file=sys.stderr)
        return 2
    model = resolve_model()
    rng = random.Random(20260908)
    print(f"model={model} chunk-file={args.chunk_file} reps={args.reps}")
    if not histogram(metrics(), "iteration_tokens_total")[0]:
        print("note: vllm:iteration_tokens_total not exported; falling back to ceil(tokens/chunk) for the step count")

    rows: list[dict] = []
    try:
        for chunk in chunks:
            set_chunk(args.chunk_file, chunk)
            for ctx in ctxs:
                for rep in range(args.reps):
                    before = histogram(metrics(), "iteration_tokens_total")
                    ptok, wall = prefill(model, ctx, rng)
                    after = histogram(metrics(), "iteration_tokens_total")
                    steps = int(round(after[0] - before[0])) if after[0] else 0
                    measured = steps > 0
                    if not measured:
                        steps = math.ceil(ptok / chunk)
                    rows.append({"chunk": chunk, "ctx": ctx, "rep": rep, "prompt_tokens": ptok,
                                 "wall": wall, "steps": steps, "steps_measured": measured,
                                 "sched_tokens": after[1] - before[1]})
                    print(f"  chunk {chunk:>5}  ctx {ctx:>7}  rep{rep}  "
                          f"tok {ptok:>7}  wall {wall:6.2f}s  steps {steps:>4}"
                          f"{'' if measured else '(est)'}  "
                          f"{ptok / wall:7.0f} tok/s  step {wall / steps * 1000:7.1f} ms")
    finally:
        set_chunk(args.chunk_file, 0)
        print("chunk override cleared (0 = the boot's own chunking)")

    print("\n== per (chunk, ctx): median of reps ==")
    print(f"{'chunk':>6} {'ctx':>8} {'tok/s':>8} {'step ms':>9} {'ms/1k tok':>10}")
    best: dict[int, float] = {}
    for chunk in chunks:
        for ctx in ctxs:
            sel = sorted(r["wall"] / r["steps"] for r in rows if r["chunk"] == chunk and r["ctx"] == ctx)
            if not sel:
                continue
            step_s = sel[len(sel) // 2]
            rate = [r["prompt_tokens"] / r["wall"] for r in rows if r["chunk"] == chunk and r["ctx"] == ctx]
            rate = sorted(rate)[len(rate) // 2]
            print(f"{chunk:>6} {ctx:>8} {rate:>8.0f} {step_s * 1000:>9.1f} {step_s / chunk * 1e6:>10.1f}")
            best[ctx] = max(best.get(ctx, 0.0), rate)

    print("\n== small-chunk penalty (vs the best chunk at the same ctx) ==")
    for ctx in ctxs:
        for chunk in chunks:
            rate = [r["prompt_tokens"] / r["wall"] for r in rows if r["chunk"] == chunk and r["ctx"] == ctx]
            if not rate or not best.get(ctx):
                continue
            rate = sorted(rate)[len(rate) // 2]
            print(f"  ctx {ctx:>7} chunk {chunk:>5}: {rate:7.0f} tok/s = {100 * rate / best[ctx]:5.1f}% of best")

    f = fit(rows)
    if f:
        print("\n== fit T_step = a + b*C + c*ctx ==")
        print(f"  a = {f['a_s'] * 1000:8.1f} ms   (shape-independent: host, launch glue, collectives)")
        print(f"  b = {f['b_s_per_tok'] * 1e6:8.2f} us/token   (=> {1 / f['b_s_per_tok']:,.0f} tok/s ceiling)"
              if f["b_s_per_tok"] > 0 else f"  b = {f['b_s_per_tok'] * 1e6:8.2f} us/token (NEGATIVE: model does not hold)")
        print(f"  c = {f['c_s_per_ctx_tok'] * 1e6:8.3f} us per 1 token of PREFIX per step"
              f"  (= {f['c_s_per_ctx_tok'] * 32000 * 1000:.0f} ms/step at 32K, "
              f"{f['c_s_per_ctx_tok'] * 128000 * 1000:.0f} ms at 128K)")
        for ctx in ctxs:
            for chunk in chunks:
                pred = f["a_s"] + f["b_s_per_tok"] * chunk + f["c_s_per_ctx_tok"] * ctx / 2
                print(f"    predicted step at ctx {ctx:>7} chunk {chunk:>5}: {pred * 1000:7.1f} ms "
                      f"= a {f['a_s'] * 1000:.0f} + b*C {f['b_s_per_tok'] * chunk * 1000:.0f} "
                      f"+ c*ctx/2 {f['c_s_per_ctx_tok'] * ctx / 2 * 1000:.0f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "fit": f, "model": model,
                       "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
