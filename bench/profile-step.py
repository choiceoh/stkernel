#!/usr/bin/env python3
"""One-shot GPU-kernel attribution for the unattributed step residual.

The deneb CUDA-event probes closed the step budget down to MoE 27.6% /
attention_impl 17.3% / in_gemms 3.7% / o_proj 2.6%, leaving ~49%
unattributed (mHC + TP collectives + launch glue). Python-level spans
CANNOT split that residual: fuse_gemm_comms / fuse_allreduce_rms melt the
collectives into compiled custom ops, so any wrapper around
tensor_model_parallel_all_reduce structurally under-counts comms. The torch
profiler sees every GPU kernel — NCCL device kernels included, fused or not.

Flow: POST /start_profile -> one fresh (uncached) long prefill with
max_tokens=1 -> POST /stop_profile -> parse the newest rank trace in the
mounted trace dir -> bucket kernel time by name + print a top-kernel table
(use it to identify mHC kernel names and refine the buckets).

Prereq: server launched with VLLM_TORCH_PROFILER_DIR=/prof (start-hy4-tp4.sh
sets it; traces land in /home/choiceoh/vllm-prof on the head). Run ON srv2.

Usage:
  profile-step.py [ctx_tokens=32000] [trace_dir=/home/choiceoh/vllm-prof]
                  [batched_tokens=4096]
  profile-step.py decode:600   # decode-shaped step budget instead of prefill
The per-step column divides by ceil(prompt_tokens / batched_tokens) so the
numbers line up with the deneb probe tree (429 / 268 / 57 / ~40 ms per
4,096-token step). idle = wall minus the union of kernel intervals = launch
glue + host stalls (comm waits sit INSIDE nccl kernel durations, not here).
"""
import glob
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
# Served name of the model under test. It was the dsv4 literal, so pointing
# this tool at any other lane returned 404 from /v1/chat/completions and the
# capture died after /start_profile had already armed. Same default as
# bench-dec.py so existing invocations and recorded numbers still line up.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from bench_common import resolve_model as _resolve_model
MODEL = _resolve_model("deepseek-v4-flash")
WORDS = [
    "reactor", "harbor", "lattice", "quarry", "ember", "meridian", "syntax",
    "granite", "voltage", "cirrus", "tundra", "beacon", "ledger", "prism",
    "cobalt", "willow", "cascade", "anvil", "nocturne", "vellum",
]

# First match wins; names are lower-cased before matching. Refine using the
# top-kernel table this script prints (that is how mHC gets its own bucket).
BUCKETS = [
    ("comms", r"nccl|all_reduce|allreduce|reduce_scatter|all_gather|allgather"),
    ("moe", r"moe|expert|grouped_gemm|group_gemm|router|routing"),
    ("attn/indexer", r"mla|fmha|flashinfer|attn|sparse|indexer|mqa|paged|topk"),
    ("gemm", r"gemm|matmul|cutlass|cublas|nvjet|wgmma|einsum|f8f8"),
    ("norm/rope/quant", r"norm|rope|rotary|quant"),
]
_COMPILED = [(label, re.compile(pat)) for label, pat in BUCKETS]


def post(path: str, timeout: int = 30) -> None:
    req = urllib.request.Request(BASE + path, data=b"", method="POST")
    try:
        urllib.request.urlopen(req, timeout=timeout).read()
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            sys.exit(
                f"{path} -> HTTP {e.code}: server was not launched with "
                "VLLM_TORCH_PROFILER_DIR (relaunch via start-hy4-tp4.sh once) "
                "or this fork lacks the profiler endpoints (fallback: nsys)."
            )
        raise


def run_prefill(ctx_tokens: int) -> tuple[int, float]:
    rng = random.Random(os.getpid() * 6067 + int(time.time()) % 100000)
    text = " ".join(rng.choice(WORDS) for _ in range(int(ctx_tokens / 1.3)))
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text + " End."}],
        "max_tokens": 1,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", body,
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return out["usage"]["prompt_tokens"], time.time() - t0


def draft_count() -> float:
    """spec_decode draft batches so far = decode steps taken (one draft set per
    step). Used instead of dividing tokens by an assumed acceptance rate."""
    try:
        text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:
        return 0.0
    total = 0.0
    for line in text.splitlines():
        if line.startswith("#") or "spec_decode" not in line or "accept" in line:
            continue
        m = re.match(r"[A-Za-z0-9_:]*num_drafts[A-Za-z0-9_:]*(?:\{[^}]*\})?\s+"
                     r"([0-9.eE+-]+)\s*$", line)
        if m:
            total += float(m.group(1))
    return total


def run_decode(out_tokens: int) -> tuple[int, float]:
    """Short prompt, long generation — the step budget here is decode-shaped.

    Returns (decode steps, elapsed). Steps come from the /metrics draft counter
    so the per-step column stays correct whatever the acceptance rate rolls.
    """
    rng = random.Random(os.getpid() * 6067 + int(time.time()) % 100000)
    topic = rng.choice(WORDS)
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"Write a long technical essay about {topic}."}],
        "max_tokens": out_tokens,
        "chat_template_kwargs": {"thinking": False},
    }).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", body,
        {"Content-Type": "application/json"},
    )
    d0 = draft_count()
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    el = time.time() - t0
    steps = int(draft_count() - d0) or out["usage"]["completion_tokens"]
    print(f"decode {out['usage']['completion_tokens']} tok in {el:.1f}s "
          f"over {steps} steps")
    return steps, el


def wait_for_trace(trace_dir: str, after: float, timeout: int = 180) -> str:
    """Newest trace file written after `after`, waited until its size is
    stable (the profiler flushes asynchronously after /stop_profile)."""
    deadline = time.time() + timeout
    path, last_size = None, -1
    while time.time() < deadline:
        cands = [
            p for p in glob.glob(os.path.join(trace_dir, "**", "*.json*"),
                                 recursive=True)
            if os.path.getmtime(p) >= after
        ]
        if cands:
            newest = max(cands, key=os.path.getmtime)
            size = os.path.getsize(newest)
            if newest == path and size == last_size and size > 0:
                return newest
            path, last_size = newest, size
        time.sleep(2)
    sys.exit(f"no stable trace appeared in {trace_dir} within {timeout}s")


def load_events(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        doc = json.load(f)
    return doc.get("traceEvents", doc if isinstance(doc, list) else [])


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, pat in _COMPILED:
        if pat.search(low):
            return label
    return "other"


def merged_busy_us(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of (start, end) intervals, in µs.

    Distinguishes true GPU idle (launch glue / host stalls) from stream
    overlap: per-bucket sums can exceed wall, the union cannot.
    """
    busy = 0.0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy


def main() -> None:
    arg1 = sys.argv[1] if len(sys.argv) > 1 else ""
    ctx = int(arg1) if arg1.isdigit() else 32_000
    trace_dir = sys.argv[2] if len(sys.argv) > 2 else "/home/choiceoh/vllm-prof"
    batched = int(sys.argv[3]) if len(sys.argv) > 3 else 4_096

    # `decode:600` swaps the prefill for a 600-token generation, so the same
    # bucketing answers "what does a DECODE step spend on" (the two step shapes
    # have completely different comms/weight ratios).
    decode_tokens = 0
    if len(sys.argv) > 1 and sys.argv[1].startswith("decode:"):
        decode_tokens = int(sys.argv[1].split(":", 1)[1] or 600)

    post("/start_profile")
    t0 = time.time()
    try:
        if decode_tokens:
            steps_override, el = run_decode(decode_tokens)
            pt = 0
        else:
            pt, el = run_prefill(ctx)
    finally:
        post("/stop_profile", timeout=600)
    if not decode_tokens:
        print(f"prefill {pt:,} tok in {el:.1f}s "
              f"({pt / el:,.0f} tok/s WITH profiler overhead — use ratios, "
              "not absolutes)")

    path = wait_for_trace(trace_dir, after=t0)
    print(f"trace: {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    events = load_events(path)

    totals: dict[str, float] = {}
    per_kernel: dict[str, list[float]] = {}
    intervals: list[tuple[float, float]] = []
    t_min, t_max = float("inf"), 0.0
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        dur = ev.get("dur")
        if dur is None:
            continue
        if cat in ("gpu_memcpy", "gpu_memset"):
            label = "memops"
        elif "kernel" in cat:
            label = bucket_of(ev.get("name", ""))
        else:
            continue
        ts = ev.get("ts", 0.0)
        t_min, t_max = min(t_min, ts), max(t_max, ts + dur)
        intervals.append((ts, ts + dur))
        totals[label] = totals.get(label, 0.0) + dur
        k = per_kernel.setdefault(ev.get("name", "?"), [0.0, 0])
        k[0] += dur
        k[1] += 1

    if not totals:
        sys.exit("no GPU kernel events in trace — was the request served "
                 "during the capture window?")

    wall_ms = (t_max - t_min) / 1e3
    total_ms = sum(totals.values()) / 1e3
    busy_ms = merged_busy_us(intervals) / 1e3
    idle_ms = max(0.0, wall_ms - busy_ms)
    steps = steps_override if decode_tokens else max(1, -(-pt // batched))
    print(f"\n=== GPU kernel-time attribution "
          f"(wall {wall_ms:,.0f} ms · {steps} "
          f"{'decode' if decode_tokens else f'prefill steps of <={batched} tok'}) ===")
    for label, us in sorted(totals.items(), key=lambda kv: -kv[1]):
        ms = us / 1e3
        print(f"  {label:<16} {ms:>9,.1f} ms  ({100.0 * ms / total_ms:>5.1f}%)"
              f"  {ms / steps:>7,.1f} ms/step")
    print(f"  {'total kernels':<16} {total_ms:>9,.1f} ms   "
          f"(sum > busy = stream overlap)")
    print(f"  {'gpu busy':<16} {busy_ms:>9,.1f} ms  "
          f"({100.0 * busy_ms / wall_ms:>5.1f}% of wall)"
          f"  {busy_ms / steps:>7,.1f} ms/step")
    print(f"  {'idle/launch glue':<16} {idle_ms:>9,.1f} ms  "
          f"({100.0 * idle_ms / wall_ms:>5.1f}% of wall)"
          f"  {idle_ms / steps:>7,.1f} ms/step")

    print("\n=== top kernels (refine BUCKETS with these; find mHC here) ===")
    top = sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:40]
    for name, (us, n) in top:
        print(f"  {us / 1e3:>9,.1f} ms  x{n:<6} [{bucket_of(name):<15}] "
              f"{name[:110]}")


if __name__ == "__main__":
    main()
