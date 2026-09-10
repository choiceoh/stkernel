#!/usr/bin/env python3
"""Does our KV compressor equal DeepSeek's? No GPU, no weights.

The compressor is the CED mechanism: four layers pool `compress_ratio` tokens
into one KV latent and everything downstream reads what they produce. Three
behaviours are checked, and the last two are the ones a shape test cannot see:

  1. prefill, exact multiple of the ratio
  2. RAGGED prefill -- a trailing partial group goes into the carry state and
     must not appear in the output
  3. DECODE across a group boundary -- the step that completes a group yields a
     latent and every other step yields None. A caller that assumes one KV
     entry per token is off by `compress_ratio` and keeps the wrong history.

    python3 probes/dsv41_compressor_diff.py --model-py .../inference/model.py

Bit equality. The pooling runs in fp32 above ratio 1 and the reference promotes
the weights to match while the checkpoint stores bf16; doing the softmax in
bf16 instead changes every group's weights by a little, which is exactly the
kind of difference a tolerance is chosen to permit.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_model"))

from dsv41_compressor import Compressor  # noqa: E402


@dataclass
class RefArgs:
    dim: int
    head_dim: int
    norm_eps: float
    compress_ratios: tuple
    max_batch_size: int = 2


def load_reference(model_py: Path):
    text = model_py.read_text()
    parts = []
    for start, end in (("def linear(", "class Linear(nn.Module):"),
                       ("class Linear(nn.Module):", "class ColumnParallelLinear"),
                       ("class RMSNorm(nn.Module):", "class ParallelEngramEmbedding"),
                       ("class Compressor(nn.Module):", "class Indexer(")):
        assert text.count(start) == 1, start
        i = text.index(start)
        parts.append(text[i:text.index(end, i)])
    ns = {"nn": nn, "torch": torch, "F": F, "default_dtype": torch.bfloat16,
          "fp8_block_size": 32, "fp4_block_size": 32, "scale_fmt": "ue8m0",
          "scale_dtype": torch.float8_e8m0fnu, "world_size": 1, "rank": 0}
    exec(compile("\n".join(parts), str(model_py), "exec"), ns)
    return ns["Compressor"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-py", required=True)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--eps", type=float, default=1e-20)
    args = ap.parse_args()

    torch.manual_seed(20260910)
    Ref = load_reference(Path(args.model_py))
    fail = 0

    for ratio in (1, 2):
        cfg = RefArgs(dim=args.dim, head_dim=args.head_dim, norm_eps=args.eps,
                      compress_ratios=(ratio,))
        ref = Ref(cfg, 0)
        with torch.no_grad():
            wkv = torch.randn(args.head_dim, args.dim) * 0.05
            ref.wkv.weight.copy_(wkv.to(ref.wkv.weight.dtype))
            ref.norm.weight.copy_(torch.randn(args.head_dim) * 0.3 + 1.0)
            wgate = None
            if ratio > 1:
                wgate = torch.randn(args.head_dim, args.dim) * 0.05
                ref.wgate.weight.copy_(wgate.to(ref.wgate.weight.dtype))
        nw = ref.norm.weight.detach()

        cases = [("prefill x{}".format(4 * ratio), 0, 4 * ratio),
                 ("ragged prefill", 0, 4 * ratio + 1)]
        for label, start, seqlen in cases:
            ours = Compressor(args.dim, args.head_dim, ratio, args.eps,
                              max_batch_size=2)
            x = (torch.randn(2, seqlen, args.dim) * 0.4).to(torch.bfloat16)
            r = ref(x, start)
            o = ours.forward(x, start, wkv, wgate, nw)
            ok = (r is None) == (o is None) and (
                r is None or torch.equal(r, o))
            fail |= not ok
            shape = tuple(r.shape) if r is not None else None
            print(f"  ratio {ratio}  {label:16s} {'OK' if ok else 'MISMATCH'}"
                  f"   out {shape}")
            if not ok and r is not None and o is not None:
                d = (r.float() - o.float()).abs()
                print(f"      max {d.max().item():.4g} over {d.numel()}")

        # Decode after a prefill. Two prefills, and the RAGGED one is the
        # case that matters: it leaves a partial group in the carry state, and
        # the first decode step has to complete it. A test that only ever
        # prefills an exact multiple never touches the carry at all -- dropping
        # it entirely passes such a test, because the tail was never in the
        # prefill's output either way.
        for pre_len, tag in ((3 * ratio, "after exact prefill"),
                             (3 * ratio + 1, "after ragged prefill")):
            fail |= run_decode(Ref, cfg, ratio, args, wkv, wgate, nw,
                               pre_len, tag)
    print("\n" + ("COMPRESSOR FAIL" if fail else "COMPRESSOR PASS"))
    return 1 if fail else 0


def run_decode(Ref, cfg, ratio, args, wkv, wgate, nw, pre_len, tag) -> int:
        import torch
        fail = 0
        ours = Compressor(args.dim, args.head_dim, ratio, args.eps,
                          max_batch_size=2)
        ref = Ref(cfg, 0)
        with torch.no_grad():
            ref.wkv.weight.copy_(wkv.to(ref.wkv.weight.dtype))
            ref.norm.weight.copy_(nw)
            if ratio > 1:
                ref.wgate.weight.copy_(wgate.to(ref.wgate.weight.dtype))
        pre = (torch.randn(2, pre_len, args.dim) * 0.4).to(torch.bfloat16)
        ref(pre, 0)
        ours.forward(pre, 0, wkv, wgate, nw)
        yielded_ref, yielded_our = [], []
        for step in range(pre_len, pre_len + 2 * ratio):
            tok = (torch.randn(2, 1, args.dim) * 0.4).to(torch.bfloat16)
            r, o = ref(tok, step), ours.forward(tok, step, wkv, wgate, nw)
            yielded_ref.append(r is not None)
            yielded_our.append(o is not None)
            if (r is None) != (o is None) or (r is not None
                                              and not torch.equal(r, o)):
                print(f"      decode step {step}: ref "
                      f"{'latent' if r is not None else 'None'}, ours "
                      f"{'latent' if o is not None else 'None'}")
                fail = 1
        ok = yielded_ref == yielded_our and any(yielded_ref)
        fail |= not ok
        print(f"  ratio {ratio}  {tag:20s} {'OK' if ok else 'MISMATCH'}"
              f"   yielded {yielded_ref}")
        return fail


if __name__ == "__main__":
    raise SystemExit(main())
