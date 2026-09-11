"""What one DSv4.1 block's forward costs above its weights, per rank.

`engine/budget.py` carries this as its last estimated line, borrowed from GLM's
`profile-run +9.17 GiB`. dsv41 splits encoder/decoder, gathers engram rows and
carries three MTP layers, so there is no reason the two match.

An activation peak is a function of SHAPES, not of values, so this builds the
reference's own `Block` with uninitialized weights and runs it. That makes the
measurement cheap enough to sweep -- no 73 GiB load, no fleet, no boot -- and
it is the same object the model will have to allocate for.

world_size is forced to 4 with rank 0 so every ColumnParallelLinear takes the
shard a real rank would take; `dist.all_reduce` is stubbed because one process
cannot perform it and it does not allocate.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

GIB = 1 << 30
REF = Path("/home/choiceoh/models/DeepSeek-V4.1-Flash/inference")


def _kernel_stub():
    """A shape-faithful stand-in for inference/kernel.py.

    tilelang is not installed on this fleet, and installing it would not make
    the measurement better: a TileLang kernel's GLOBAL footprint is its output.
    Its scores, its running max, its gathered KV tiles live in shared memory
    and registers, which is the entire reason `sparse_attn` exists instead of a
    materialised [b, m, h, n] score matrix. So allocating exactly the output --
    and nothing else -- is what a real kernel does to the pool this measures.

    Values are garbage on purpose. An activation peak is a function of shapes.
    """
    import types
    import torch

    mod = types.ModuleType("kernel")

    def act_quant(x, block_size=128, scale_fmt=None,
                  scale_dtype=torch.float32, inplace=False):
        y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        s = x.new_empty(*x.shape[:-1], -(-x.size(-1) // block_size), dtype=scale_dtype)
        return y, s

    def fp4_act_quant(x, block_size=32, inplace=False,
                      scale_dtype=torch.float8_e8m0fnu):
        y = x.new_empty(*x.shape[:-1], x.size(-1) // 2, dtype=torch.float4_e2m1fn_x2)
        s = x.new_empty(*x.shape[:-1], -(-x.size(-1) // block_size), dtype=scale_dtype)
        return y, s

    def _gemm(a, a_s, b, b_s, *args, **kwargs):
        return torch.empty(*a.shape[:-1], b.shape[0],
                           dtype=torch.bfloat16, device=a.device)

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        return torch.empty_like(q)                 # the kernel's own line 398

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4,
                          sinkhorn_iters=20, eps=1e-6):
        n = mixes.shape[0]
        opts = dict(dtype=torch.float32, device=mixes.device)
        return (torch.empty(n, hc_mult, **opts), torch.empty(n, hc_mult, **opts),
                torch.empty(n, hc_mult, hc_mult, **opts))

    mod.act_quant = act_quant
    mod.fp4_act_quant = fp4_act_quant
    mod.fp8_gemm = _gemm
    mod.fp4_gemm = _gemm
    mod.sparse_attn = sparse_attn
    mod.hc_split_sinkhorn = hc_split_sinkhorn
    return mod


def build(layer: int, batch: int, seq: int, world: int = 4):
    sys.path.insert(0, str(REF))
    import torch
    import torch.distributed as dist

    dist.all_reduce = lambda *a, **k: None          # single process, no allocation
    sys.modules.setdefault("kernel", _kernel_stub())
    import model as ref

    ref.world_size, ref.rank = world, 0
    cfg = json.loads((REF / "config.json").read_text())
    cfg["max_batch_size"], cfg["max_seq_len"] = batch, seq
    args = ref.ModelArgs(**cfg)

    torch.cuda.set_device(0)
    with ref.set_dtype(torch.bfloat16), torch.device("cuda"):
        block = ref.Block(layer, args)
    return ref, args, block


def measure(layer: int, batch: int, seq: int, tokens: int, world: int = 4) -> dict:
    import torch

    ref, args, block = build(layer, batch, seq, world)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    hc = args.hc_mult
    x = torch.zeros(batch, tokens, hc, args.dim, dtype=torch.bfloat16, device="cuda")
    pre_mix = torch.zeros(batch, tokens, hc, dtype=torch.bfloat16, device="cuda")
    inputs = torch.cuda.memory_allocated() - weights

    started = time.perf_counter()
    # factory calls inside the forward (arange, empty for index buffers) take
    # the default device, and the reference runs with cuda already set.
    with torch.inference_mode(), torch.device("cuda"):
        out, _ = block(x, 0, pre_mix, None)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    peak = torch.cuda.max_memory_allocated()
    return {
        "layer": layer, "batch": batch, "seq": seq, "tokens": tokens,
        "weights_gib": weights / GIB,
        "inputs_gib": inputs / GIB,
        "activation_gib": (peak - weights - inputs) / GIB,
        "peak_gib": peak / GIB,
        "seconds": elapsed,
        "out_shape": tuple(out.shape),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=131072)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--world", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(measure(args.layer, args.batch, args.seq, args.tokens, args.world)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
