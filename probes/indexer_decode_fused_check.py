#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""glm53_kpool_tail_select: the fused decode tail-select launch vs the stock
chain, bit for bit, plus the graph-replay time of both (fresh container).

The fused path (VLLM_GLM53_INDEXER_DECODE_FUSED=1, 34차) replaces, per
full-attention layer of a uniform spec-verify decode step:

    _decode_topk_seq_lens        positions[:n].to(int32) + 1        (2 launches)
    _force_tail_pool_into_logits int64 // pool - 1, clamp, gather,   (8 launches)
                                 full_like, >= 0, where, scatter_
    topk_indices_buffer[...] = expand_pools_and_append_tail(...)     (1 launch + copy)

with `_glm53_indexer_tail_select_kernel` (one launch) and
`expand_pools_and_append_tail_into` (the same expand kernel writing into the
persistent buffer). All of it is integer index arithmetic and one fp32
constant, so the gate is torch.equal on every output, over random and
edge-case rows (positions 0..pool*2, empty tails, pools past the logits
width, -1 pool ids), and on the untouched rest of the buffer.

    bash probes/run_indexer_fused_check.sh [--trials 40] [--iters 200]
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch

# The model package must initialise first: the kpool layer module imports
# back into vllm.models.glm5next (ops.kpool_compress), and importing the
# layer module cold trips the package __init__ -> attention -> layer cycle.
import vllm.models.glm5next.nvidia.ops.kpool_compress  # noqa: F401
import vllm.model_executor.layers.sparse_attn_indexer_kpool as K


def stock_chain(positions, logits, pool_ids, pool, buf, force):
    n = positions.shape[0]
    seq = K._decode_topk_seq_lens(positions, None, n, n, 1, False)
    if force:
        K._force_tail_pool_into_logits(logits, seq, pool)
    out = K.expand_pools_and_append_tail(pool_ids, seq, pool)
    buf[: out.shape[0], : out.shape[-1]] = out
    return seq


def fused_chain(positions, logits, pool_ids, pool, buf, force):
    n = positions.shape[0]
    seq = K.indexer_tail_select_fused(positions, logits, n, pool, force)
    w = K.expand_pools_and_append_tail_into(pool_ids, seq, pool, buf)
    assert w is not None, "direct write refused the buffer layout"
    return seq


def one_case(gen, n_rows, n_cols, n_groups, pool, force, edge):
    dev = "cuda"
    if edge:
        # positions around pool boundaries and 0, pools past the width, -1 ids
        pos = torch.randint(0, pool * 3, (n_rows,), generator=gen, device=dev)
        pos[0] = 0
        pos[-1] = (n_cols + 5) * pool
    else:
        pos = torch.randint(0, 5000, (n_rows,), generator=gen, device=dev)
    pos = pos.to(torch.int64)
    logits = torch.randn(n_rows, n_cols, generator=gen, device=dev, dtype=torch.float32)
    pool_ids = torch.randint(-1, max(1, n_cols), (n_rows, n_groups), generator=gen,
                             device=dev).to(torch.int32)
    width = n_groups * pool + pool - 1 + 7
    buf_s = torch.full((n_rows + 3, width), -7, dtype=torch.int32, device=dev)
    buf_f = buf_s.clone()
    la, lb = logits.clone(), logits.clone()
    seq_s = stock_chain(pos, la, pool_ids, pool, buf_s, force)
    seq_f = fused_chain(pos, lb, pool_ids, pool, buf_f, force)
    torch.cuda.synchronize()
    ok = (torch.equal(seq_s, seq_f) and torch.equal(la, lb) and torch.equal(buf_s, buf_f)
          and seq_f.dtype == torch.int32)
    return ok, (seq_s, seq_f, la, lb, buf_s, buf_f)


def replay_us(fn, iters):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000 / iters)
    return statistics.median(ts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(1234)
    pool, n_groups = 4, 512          # index_kpool 4, index_topk 2048 -> 512 pools
    fails = 0
    shapes = [(8, 1), (8, 40), (16, 40), (32, 40), (8, 1250), (24, 1250), (8, 1), (256, 300)]
    for t in range(a.trials):
        n_rows, n_cols = shapes[t % len(shapes)]
        force = (t % 3) != 2
        edge = (t % 4) == 1
        ok, parts = one_case(gen, n_rows, n_cols, n_groups, pool, force, edge)
        if not ok:
            fails += 1
            seq_s, seq_f, la, lb, bs, bf = parts
            print(f"MISMATCH trial {t} rows={n_rows} cols={n_cols} force={force} edge={edge}: "
                  f"seq {int((seq_s != seq_f).sum())} logits {int((la != lb).sum())} "
                  f"buf {int((bs != bf).sum())}")
    print(f"bit-exact: {a.trials - fails}/{a.trials} trials PASS (pool={pool}, groups={n_groups})")
    # timing: the serving shape, C=1 (8 rows) and C=4 (32 rows), logits width of a
    # 5K context (1250 pools); both chains captured in a CUDA graph
    for n_rows in (8, 32):
        n_cols = 1250
        pos = torch.randint(0, 5000, (n_rows,), generator=gen, device="cuda").to(torch.int64)
        logits = torch.randn(n_rows, n_cols, generator=gen, device="cuda")
        pool_ids = torch.randint(0, n_cols, (n_rows, n_groups), generator=gen,
                                 device="cuda").to(torch.int32)
        buf = torch.full((n_rows + 3, n_groups * pool + pool - 1 + 7), -1,
                         dtype=torch.int32, device="cuda")
        us_s = replay_us(lambda: stock_chain(pos, logits, pool_ids, pool, buf, True), a.iters)
        us_f = replay_us(lambda: fused_chain(pos, logits, pool_ids, pool, buf, True), a.iters)
        print(f"replay rows={n_rows} pools={n_cols}: stock chain {us_s:.1f} us -> fused "
              f"{us_f:.1f} us per layer ({11 * (us_s - us_f) / 1000:+.3f} ms/step x11 layers)")
    print("VERDICT:", "PASS" if fails == 0 else "FAIL")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
