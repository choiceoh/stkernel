"""What the sampler costs at a serving shape, and what it used to (45차 §28).

    python3 probes/engine_sampler_bench.py [--vocab 154880] [--seqs 4] [--spec 5]

Reports the fused kernel against the two things it replaces: the batched sort-and-multinomial the
captured/pipeline path ran, and the same work done one row at a time, which is what the rich path
did. Also reports the pieces, so a later change can see which pass it moved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.base.sampler import rows


def timed(name, fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(stop) / iters * 1000
    print(f"  {name:48s} {us:9.1f} us")
    return us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--seqs", type=int, default=4)
    ap.add_argument("--spec", type=int, default=5)
    ap.add_argument("--top-p", type=float, default=0.9)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA: the sampler kernel has nothing to run on")
        return 1
    V, M = a.vocab, a.seqs * (a.spec + 1)
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(M, V, device=dev, generator=g) * 2
    narrow = logits.bfloat16()
    temp = torch.ones(M, device=dev)
    top_k = torch.zeros(M, dtype=torch.int32, device=dev)
    top_p = torch.full((M,), a.top_p, device=dev)
    u = torch.rand(M, device=dev, generator=g)
    probs = torch.empty(M, V, dtype=torch.float32, device=dev)
    print(f"[{M} rows x {V} vocabulary = {M * V * 4 / 2**20:.1f} MB fp32, top_p {a.top_p}]")

    print("the kernel")
    timed("ids only", lambda: rows(logits, temp, top_k, top_p, u))
    timed("ids + distributions", lambda: rows(logits, temp, top_k, top_p, u, None, probs))
    timed("distributions only (the pipeline's call)", lambda: rows(logits, temp, top_k, top_p, None, None, probs))
    timed("bf16 logits, distributions only", lambda: rows(narrow, temp, top_k, top_p, None, None, probs))
    timed("no truncation (top_p 1)", lambda: rows(logits, temp, top_k, torch.ones(M, device=dev), u, None, probs))
    timed("top_k 64 and top_p (both searches)",
          lambda: rows(logits, temp, torch.full((M,), 64, dtype=torch.int32, device=dev), top_p, u, None, probs))

    print("what it replaces")
    softmax = torch.softmax(logits.float(), -1)

    def batched_sort():
        srt, idx = softmax.sort(-1, descending=True)
        keep = (srt.cumsum(-1) - srt) < top_p.unsqueeze(-1)
        srt = srt * keep
        out = torch.zeros_like(softmax).scatter_(1, idx, srt / srt.sum(-1, keepdim=True))
        return torch.multinomial(out, 1)

    def row_at_a_time():
        out = []
        for i in range(M):
            srt, idx = softmax[i].sort(descending=True)
            keep = (srt.cumsum(-1) - srt) < a.top_p
            srt = srt * keep
            out.append(torch.zeros_like(softmax[i]).scatter_(0, idx, srt / srt.sum()))
        return torch.cat([torch.multinomial(d, 1) for d in out])

    timed("batched sort + multinomial", batched_sort)
    timed("one row at a time (the rich path)", row_at_a_time, 10)

    print("the pieces, for whoever moves a pass next")
    timed("sort(descending)", lambda: softmax.sort(-1, descending=True))
    timed("one streaming pass (max)", lambda: logits.max(-1))
    timed("cumsum", lambda: softmax.cumsum(-1))
    timed("multinomial", lambda: torch.multinomial(softmax, 1, generator=g))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
