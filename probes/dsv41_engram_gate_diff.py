#!/usr/bin/env python3
"""Does our engram gate equal DeepSeek's `Engram.forward`? No GPU, no weights.

The lookup is already proven bit-identical (dsv41_engram_diff.py) and so are the
hashes (dsv41_engram_hash_diff.py). What is left is the arithmetic between the
readout and the residual stream, and it is the part most able to be wrong
quietly: a `rstd` taken jointly over the hc copies instead of per copy, an
unsigned sqrt, a missing `dim ** -0.5`, or a masked token skipped instead of
gated to zero all produce finite, plausible numbers.

So the reference class is run verbatim -- with `default_dtype` set to bfloat16
so its `Linear` falls through to `F.linear` and the triton fp8 path is never
touched -- and compared against `gate_and_write` on the same tensors.

    python3 probes/dsv41_engram_gate_diff.py --model-py .../inference/model.py

Bit equality, not a tolerance: both sides run the same ops in the same order on
the same inputs, so anything but equality is a difference in the formula.

The four mutations this was checked against, and what each costs -- the second
is why the bar is equality:

    unsigned sqrt (drop copysign)      45,846 of 98,304 differ, max 1.26
    rstd joint over the hc copies      34,751 differ, max 0.0156
    drop the dim ** -0.5 scale         89,546 differ, max 0.617
    ignore token_mask                   8,157 differ, first at token 5,
                                        the start of the masked span

A tolerance loose enough to be useful in bf16 passes the second one outright.
It is also the most likely thing to get wrong, because `mean(-1)` reads as
"normalize the last dim" whichever way the copies are laid out.
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
sys.path.insert(0, str(HERE / "overlay/modules/dsv41_engram"))

from dsv41_engram_gate import gate_and_write  # noqa: E402


@dataclass
class RefArgs:
    dim: int
    hc_mult: int
    norm_eps: float


class Layout:
    """Only the three attributes `Engram.__init__` reads."""

    def __init__(self, layer_ids, max_ngram_size, n_heads, head_dim,
                 num_embeddings):
        self.layer_ids = layer_ids
        self.max_ngram_size = max_ngram_size
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_embeddings = num_embeddings


def load_reference(model_py: Path):
    text = model_py.read_text()
    parts = []
    for start, end in (("def linear(", "class Linear(nn.Module):"),
                       ("class Linear(nn.Module):", "class ColumnParallelLinear"),
                       ("class ParallelEngramEmbedding(nn.Module):",
                        "@lru_cache(2)")):
        assert text.count(start) == 1, start
        i = text.index(start)
        parts.append(text[i:text.index(end, i)])
    source = "\n".join(parts)

    class _NoDist:
        @staticmethod
        def all_reduce(_t):
            return None

    ns = {"nn": nn, "torch": torch, "F": F, "dist": _NoDist,
          "world_size": 1, "rank": 0, "fp8_block_size": 32,
          "fp4_block_size": 32, "scale_fmt": "ue8m0",
          "scale_dtype": torch.float8_e8m0fnu,
          # bf16 storage keeps `linear()` on F.linear: no act_quant, no triton
          "default_dtype": torch.bfloat16}
    exec(compile(source, str(model_py), "exec"), ns)
    return ns["Engram"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-py", required=True)
    ap.add_argument("--dim", type=int, default=512, help="args.dim (5120 live)")
    ap.add_argument("--head-dim", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--hc-mult", type=int, default=4)
    ap.add_argument("--eps", type=float, default=1e-20,
                    help="rms_norm_eps; V4.1 ships 1e-20")
    args = ap.parse_args()

    torch.manual_seed(20260910)
    dim, hc, hd = args.dim, args.hc_mult, args.head_dim
    n_cols = (4 - 1) * 8                      # max_ngram_size 4, n_heads 8

    Engram = load_reference(Path(args.model_py))
    layout = Layout((1, 14), 4, 8, hd, (4096, 4096))
    ref = Engram(RefArgs(dim=dim, hc_mult=hc, norm_eps=args.eps), 1, layout)

    with torch.no_grad():
        ref.wkv.weight.copy_(torch.randn(dim * (hc + 1), n_cols * hd) * 0.02)
        ref.q_weight.copy_(torch.randn(hc, dim) * 0.5 + 1.0)
        ref.k_weight.copy_(torch.randn(hc, dim) * 0.5 + 1.0)

    x = (torch.randn(1, args.tokens, hc, dim) * 0.4).to(torch.bfloat16)
    readout = (torch.randn(1, args.tokens, n_cols, hd) * 0.3).to(torch.bfloat16)
    mask = torch.ones(1, args.tokens, dtype=torch.bool)
    mask[0, 5:9] = False                      # an image span: gate must be 0

    # -- reference, with its lookup replaced by the readout we already have.
    #    ParallelEngramEmbedding is proven separately; substituting it here is
    #    what isolates the arithmetic this probe is about.
    class _Fixed(nn.Module):
        def __init__(self, table):
            super().__init__()
            self.table = table

        def forward(self, _ids):
            return self.table

    ref.embed = _Fixed(readout)
    ids = torch.zeros(1, args.tokens, n_cols, dtype=torch.long)
    ref_out = ref(x, ids, mask)

    ours = gate_and_write(x, readout, ref.wkv.weight, ref.q_weight,
                          ref.k_weight, hc_mult=hc, dim=dim, eps=args.eps,
                          token_mask=mask)

    print(f"  shapes  x {tuple(x.shape)}  readout {tuple(readout.shape)}  "
          f"eps {args.eps:g}")
    same = torch.equal(ref_out, ours)
    if not same:
        d = (ref_out.float() - ours.float()).abs()
        bad = int((d > 0).sum())
        print(f"  FAIL: {bad} of {d.numel()} differ, max {d.max().item():.3g}, "
              f"first {(d > 0).nonzero()[:3].tolist()}")
        return 1
    # the masked span must have passed through untouched -- equality alone
    # would also hold if BOTH sides were wrong about it in the same way, so
    # check it against x directly
    untouched = torch.equal(ref_out[0, 5:9], x[0, 5:9])
    print(f"  gate    masked span passes through untouched: "
          f"{'yes' if untouched else 'NO'}")
    if not untouched:
        print("  FAIL: a masked token was modified; the gate is meant to be 0 "
              "there, leaving the residual exactly as it arrived")
        return 1
    moved = int((ref_out[0, 9:] != x[0, 9:]).sum())
    print(f"  gate    unmasked positions actually change: {moved:,} elements")
    print(f"  MATCH: bit-identical over {ref_out.numel():,} elements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
