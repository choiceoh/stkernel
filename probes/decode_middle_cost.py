"""What the decode chain's eager middle costs, at the shape production actually runs.

pipeline.launch replays three captured graphs -- target, observe_rows, propose_rows -- with an eager stretch
between the first two: all_gather, distribution_batch, note_ceilings, block_verify_batch, advance. Closing that
stretch is the remaining half of "one continuous decode chain". Whether it is worth closing is a number, and this
is the number.

Shape from production /metrics on 2026-09-12: 97.9% of decode steps are ONE sequence (10,820 against 236 at two),
spec_k 5 so six positions, vocabulary 154,880, sel_top_k 16.

    docker run --rm --gpus all -v $PWD:/w:ro -v probes:/s:ro --entrypoint python3 st-engine:<tag> /s/decode_middle_cost.py

Answered (ledger 45차): block_verify_batch is 2.5 ms of wall for 399 us of GPU work, issued as ~180 device-side
operations. It is not a fusion problem, it is a dispatch problem: at n=1 and K=5 most of those operations are on
five-element tensors. The sampler beside it, one kernel, does a [6, 154,880] threshold search in 555 us.
"""
import sys

import torch

sys.path.insert(0, "/w")
from engine.base.sampler import block_verify_batch, rows as sampler_rows   # noqa: E402

N, T, V, C = 1, 6, 154880, 16
DEV = torch.device("cuda")


def timed(fn, iters=200, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b) * 1000)
    out.sort()
    return out[len(out) // 2], out[0]


def main():
    g = torch.Generator(device=DEV).manual_seed(3)
    logits = torch.randn(N * T, V, generator=g, device=DEV, dtype=torch.bfloat16)
    probs = torch.empty(N * T, V, dtype=torch.float32, device=DEV)
    temps = torch.full((N * T,), 0.7, device=DEV)
    topk = torch.zeros(N * T, dtype=torch.int32, device=DEV)
    topp = torch.full((N * T,), 0.95, device=DEV)
    target = torch.softmax(torch.randn(N, T, V, generator=g, device=DEV, dtype=torch.float32), -1)
    cand = torch.randint(0, V, (N, T - 1, C), generator=g, device=DEV)
    qp = torch.softmax(torch.randn(N, T - 1, C, generator=g, device=DEV), -1)
    drafts = cand[:, :, 0].contiguous()
    gen = torch.rand(N, T, generator=torch.Generator(device=DEV).manual_seed(5), device=DEV)   # the uniforms: an input (base/draws)

    print(f"  n={N} t={T} V={V:,} C={C}\n")
    print(f"  {'stage':<36} {'median':>10} {'min':>10}")
    for name, fn in (("distribution_batch (one kernel)",
                      lambda: sampler_rows(logits, temps, topk, topp, None, None, probs)),
                     ("block_verify_batch (torch ops)",
                      lambda: block_verify_batch(target, drafts, cand, qp, gen))):
        med, low = timed(fn)
        print(f"  {name:<36} {med:>8.1f}us {low:>8.1f}us")

    from torch.profiler import ProfilerActivity, profile
    call = lambda: block_verify_batch(target, drafts, cand, qp, gen)   # noqa: E731
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        for _ in range(10):
            call()
        torch.cuda.synchronize()
    rows = pr.key_averages()
    ops = sum(e.count for e in rows if e.key.startswith("void ") or "kernel" in e.key.lower() or "Memcpy" in e.key)
    gpu = sum(e.self_device_time_total for e in rows) / 10
    host = sum(e.self_cpu_time_total for e in rows) / 10
    print(f"\n  block_verify_batch per call: {gpu:.0f} us of GPU work, {host:.0f} us of host, ~{ops / 10:.0f} device ops")


if __name__ == "__main__":
    main()
