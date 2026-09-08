#!/usr/bin/env python3
"""What a prefill step actually costs, as a function of chunk size.

39차 DF4 left "프리필 스텝 고정 비용(~0.25 s@32K) 자체 절감" open, and the ledger's
0.25 s was a residual (small-chunk step time minus the solo 8,192-chunk rate),
not a measurement. This measures it directly:

    wall = a*steps + b*tokens          steps = ceil(prompt_tokens / chunk)

a  what every step pays regardless of shape -- host, launch glue, collectives
b  the token work, which does not care how the prompt is split

Sweeping the chunk needs no boot per size: VLLM_GLM53_SCHED_CHUNK_FILE (the
decode-first scheduler's dev instrument) pins it from a file on the container's
/prof mount, so this script writes it directly. Requests are solo -- the
small-chunk penalty is a pure-prefill property, and a solo request keeps the
cadence, the floor and the decoders out of the number.

2026-09-08 (pstep0908v1, 15 points, chunks 1152/2304/4608/8192 x 32K/128K):
a = 211 ms/step, b = 281 us/token (ceiling 3,554 tok/s). The 32K and 128K
curves lie on top of each other, so the third term this probe originally
carried -- c*ctx, the prefix each chunk re-scores -- is not resolvable and was
removed. At chunk 1,152 the fixed cost is 39% of the step; at 8,192 it is 8%.

Run ON the head (srv2), against an idle boot with DECODE_FIRST=1 and
VLLM_GLM53_SCHED_CHUNK_FILE set. Every request carries a fresh nonce FIRST, so
prefix caching cannot hit whatever PREFIX_CACHE is.

  python3 probes/prefill_chunk_sweep.py [--ctx 32000,128000] [--reps 2]
      [--long-reps 1] [--chunks 1152,2304,4608,8192] [--chunk-file PATH]
      [--trace-chunks 8192,1152] [--json OUT]
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
import urllib.error
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
    """(count, sum) of a vLLM histogram, summed over every label set.

    Only a cross-check. The 2026-09-08 sweep trusted this as the step count and
    got 1 step for a 33-chunk request on one arm and 101 on another -- whatever
    vllm:iteration_tokens_total counts on this build, it is not one observation
    per engine step, and a single re.search also read only the first label set.
    The denominator is now ceil(prompt_tokens / chunk), which is exactly what
    the pinned chunk produces, and this number only prints when it disagrees."""
    out = []
    for suffix in ("_count", "_sum"):
        vals = re.findall(r"^vllm:%s%s(?:\{[^}]*\})?\s+([0-9.eE+-]+)" % (re.escape(name), suffix),
                          text, re.M)
        out.append(sum(float(v) for v in vals))
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
    try:
        out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    except urllib.error.HTTPError as exc:
        # The 2026-09-08 sweep died on a bare "HTTP Error 500" and the arm's head
        # log had already been snapshotted, so the cause was unrecoverable. The
        # body carries vLLM's own message; print it before re-raising.
        try:
            detail = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:
            detail = "(no body)"
        print(f"  !! HTTP {exc.code} on a {ctx_tokens}-token prefill after "
              f"{time.time() - t0:.1f}s: {detail}", file=sys.stderr)
        raise
    return int(out["usage"]["prompt_tokens"]), time.time() - t0


def profile_capture(model: str, ctx_tokens: int, rng: random.Random, trace_dir: str) -> tuple[str, int, float]:
    """One traced prefill: /start_profile, a fresh request, /stop_profile, then
    the newest rank-0 trace once its size stops growing (the profiler flushes
    asynchronously). tools/trace_prefill_attribution.py reads what this returns
    and reports each chunk separately, which is the attribution half of the
    question this probe answers in wall time."""
    import glob

    def post(path: str, timeout: int = 600) -> None:
        req = urllib.request.Request(BASE + path, data=b"", method="POST")
        urllib.request.urlopen(req, timeout=timeout).read()

    before = set(glob.glob(os.path.join(trace_dir, "*.json*")))
    post("/start_profile", 60)
    ptok, wall = prefill(model, ctx_tokens, rng)
    post("/stop_profile")
    newest, size = "", -1
    for _ in range(120):
        fresh = [f for f in glob.glob(os.path.join(trace_dir, "*.json*")) if f not in before]
        if fresh:
            newest = max(fresh, key=os.path.getmtime)
            now = os.path.getsize(newest)
            if now == size and now > 0:
                break
            size = now
        time.sleep(2)
    return newest, ptok, wall


def fit(rows: list[dict]) -> dict:
    """Least squares for ``wall = a*steps + b*tokens`` on one context.

    Every step pays ``a`` once; the token work ``b*tokens`` is the same however
    the prompt is split. That is the whole model, and it is exact for the last,
    PARTIAL chunk -- the earlier form (``wall/tokens = a/C + b``) silently
    assumed ``steps = tokens/C`` and so charged a full chunk for the remainder,
    which biased `a` low and `b` high by ~15% on a real 33-step request.

    The three-parameter version (a + b*C + c*ctx) is gone: the 2026-09-08 sweep
    put the 32K and 128K curves on top of each other, the pooled `c` came out
    NEGATIVE, and `c` is nearly collinear with `a` once C is swept. One (a, b)
    per context, and the spread between contexts, is what the data supports.
    Normal equations on a 2x2, so the head needs no numpy."""
    if len(rows) < 2:
        return {}
    snn = sum(r["steps"] ** 2 for r in rows)
    stt = sum(r["prompt_tokens"] ** 2 for r in rows)
    snt = sum(r["steps"] * r["prompt_tokens"] for r in rows)
    swn = sum(r["wall"] * r["steps"] for r in rows)
    swt = sum(r["wall"] * r["prompt_tokens"] for r in rows)
    det = snn * stt - snt * snt
    if det == 0:
        return {}
    a = (swn * stt - swt * snt) / det
    b = (snn * swt - snt * swn) / det
    worst = max(abs(r["wall"] - (a * r["steps"] + b * r["prompt_tokens"])) / r["wall"] for r in rows)
    return {"a_s": a, "b_s_per_tok": b, "n": len(rows),
            "ceiling_tok_s": (1.0 / b) if b > 0 else float("inf"),
            "worst_residual_pct": 100 * worst}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="32000,128000")
    ap.add_argument("--chunks", default="1152,2304,4608,8192")
    ap.add_argument("--reps", type=int, default=2)
    # A 128K row costs ~4x a 32K row and buys only the ctx sensitivity, which
    # the 2026-09-08 sweep found to be within noise: one rep confirms it.
    ap.add_argument("--long-reps", type=int, default=1,
                    help="reps for contexts above --long-ctx (default 1)")
    ap.add_argument("--long-ctx", type=int, default=64000)
    ap.add_argument("--chunk-file", default="/home/choiceoh/vllm-prof/sched_chunk")
    ap.add_argument("--json", default="")
    ap.add_argument("--trace-chunks", default="",
                    help="after the sweep, capture one torch trace per chunk at --trace-ctx")
    ap.add_argument("--trace-ctx", type=int, default=128000)
    ap.add_argument("--trace-dir", default="/home/choiceoh/vllm-prof")
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
        # Reps outermost, so every rep is a full pass over the chunk sizes. With
        # the chunk loop outermost a drift over the run (clocks, cache state, a
        # neighbour on the node) aliases straight into the chunk coefficient,
        # which is the one number this probe exists to produce.
        for rep in range(max(args.reps, args.long_reps)):
            for chunk in chunks:
                set_chunk(args.chunk_file, chunk)
                for ctx in ctxs:
                    if rep >= (args.long_reps if ctx > args.long_ctx else args.reps):
                        continue
                    before = histogram(metrics(), "iteration_tokens_total")
                    ptok, wall = prefill(model, ctx, rng)
                    after = histogram(metrics(), "iteration_tokens_total")
                    # The pinned chunk IS the denominator; the metric only gets
                    # to disagree out loud.
                    steps = math.ceil(ptok / chunk)
                    seen = int(round(after[0] - before[0]))
                    note = "" if not seen or abs(seen - steps) <= max(2, steps // 10) else f" (metric says {seen})"
                    rows.append({"chunk": chunk, "ctx": ctx, "rep": rep, "prompt_tokens": ptok,
                                 "wall": wall, "steps": steps, "metric_steps": seen,
                                 "sched_tokens": after[1] - before[1]})
                    print(f"  chunk {chunk:>5}  ctx {ctx:>7}  rep{rep}  "
                          f"tok {ptok:>7}  wall {wall:6.2f}s  steps {steps:>4}  "
                          f"{ptok / wall:7.0f} tok/s  step {wall / steps * 1000:7.1f} ms{note}")
    finally:
        set_chunk(args.chunk_file, 0)
        print("chunk override cleared (0 = the boot's own chunking)")

    def med(values):
        values = sorted(values)
        return values[len(values) // 2] if values else 0.0

    print("\n== per (chunk, ctx): median of reps ==")
    print(f"{'chunk':>6} {'ctx':>8} {'tok/s':>8} {'step ms':>9} {'us/token':>9}")
    best: dict[int, float] = {}
    for ctx in ctxs:
        for chunk in chunks:
            sel = [r for r in rows if r["chunk"] == chunk and r["ctx"] == ctx]
            if not sel:
                continue
            rate = med(r["prompt_tokens"] / r["wall"] for r in sel)
            step_s = med(r["wall"] / r["steps"] for r in sel)
            print(f"{chunk:>6} {ctx:>8} {rate:>8.0f} {step_s * 1000:>9.1f} {1e6 / rate:>9.1f}")
            best[ctx] = max(best.get(ctx, 0.0), rate)

    print("\n== small-chunk penalty (vs the best chunk at the same ctx) ==")
    for ctx in ctxs:
        for chunk in chunks:
            sel = [r for r in rows if r["chunk"] == chunk and r["ctx"] == ctx]
            if not sel or not best.get(ctx):
                continue
            rate = med(r["prompt_tokens"] / r["wall"] for r in sel)
            print(f"  ctx {ctx:>7} chunk {chunk:>5}: {rate:7.0f} tok/s = {100 * rate / best[ctx]:5.1f}% of best")

    # One (a, b) per context. Their spread IS the context sensitivity: a real
    # prefix-proportional term would push `a` up with ctx.
    fits = {ctx: fit([r for r in rows if r["ctx"] == ctx]) for ctx in ctxs}
    fits = {ctx: f for ctx, f in fits.items() if f}
    if fits:
        print("\n== fit T_step = a + b*C, one per context ==")
        for ctx, f in fits.items():
            print(f"  ctx {ctx:>7}: a = {f['a_s'] * 1000:7.1f} ms/step   "
                  f"b = {f['b_s_per_tok'] * 1e6:6.2f} us/token (ceiling {f['ceiling_tok_s']:,.0f} tok/s)   "
                  f"n={f['n']} worst residual {f['worst_residual_pct']:.1f}%")
        avals = [f["a_s"] for f in fits.values()]
        if len(avals) > 1:
            spread = 100 * (max(avals) - min(avals)) / (sum(avals) / len(avals))
            print(f"  a across contexts: {min(avals) * 1000:.0f}-{max(avals) * 1000:.0f} ms "
                  f"(spread {spread:.0f}%) -- a prefix-proportional term would show up here")
        a_med = med(avals)
        print("\n  what the fixed cost costs, per chunk size:")
        for chunk in chunks:
            b = med(f["b_s_per_tok"] for f in fits.values())
            print(f"    chunk {chunk:>5}: {100 * a_med / (a_med + b * chunk):4.0f}% of the step "
                  f"(step {(a_med + b * chunk) * 1000:6.0f} ms, {chunk / (a_med + b * chunk):,.0f} tok/s)")
        fits = {str(k): v for k, v in fits.items()}

    traces = {}
    for chunk in [int(v) for v in args.trace_chunks.split(",") if v.strip()]:
        set_chunk(args.chunk_file, chunk)
        path, ptok, wall = profile_capture(model, args.trace_ctx, rng, args.trace_dir)
        traces[chunk] = {"trace": path, "prompt_tokens": ptok, "wall": wall}
        print(f"trace chunk {chunk}: {path or '(none found)'}  tok {ptok}  wall {wall:.1f}s")
        print(f"  python3 tools/trace_prefill_attribution.py {path} --out attr-{chunk}.json")
    if traces:
        set_chunk(args.chunk_file, 0)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "fit": fits, "model": model, "traces": traces,
                       "when": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
