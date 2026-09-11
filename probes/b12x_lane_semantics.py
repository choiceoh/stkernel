"""Which side is wrong at the MoE: the served b12x lane or the torch reference?
Layer-3 real experts, ONE token, the same call the engine makes, against
dequant + torch on the same bytes -- and variants that flip one assumption
each (gate/up halves swapped; activation quant off) so the mismatch names
its cause. Inside the ST image: bash probes/run_engine_probe.sh probes/b12x_lane_semantics.py

Verdict (45th ledger §15): the kernel gates on the SECOND half of w13 --
flashinfer's CuTe-DSL order is [up; gate], which vLLM reaches by swapping
its [gate; up] at load (reorder_w13_to_w31_for_flashinfer_cutedsl). Rank
files written before that finding held [gate; up]; specs.py writes
[up; gate] (main marks such files weight_layout=st-glm53-b12x-up-gate-v1)
and the reference lane gates on the second half, so on current files the
direct comparison is the right one and "halves swapped" is the wrong order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.profiles.glm53.weights import rank_loader
from engine.modules.moe import expert_gemm
from engine.modules.nvfp4_sf import unswizzle_sf, mma_sf_view
from engine.profiles.glm53 import facts, lanes as lane_tables
from engine.profiles.glm53.lanes import swiglu_clamped


def main() -> int:
    dev = "cuda"; torch.manual_seed(0)
    F = facts.load()
    ranks = Path(sys.argv[1]) if len(sys.argv) > 1 else facts.RANKS
    got = rank_loader(ranks / "rank0of4.safetensors").load([f"L3.moe.{n}" for n in ("w13", "w13_sf", "w2", "w2_sf")], device=dev)
    w13, w13_sf, w2, w2_sf = (got[f"L3.moe.{n}"] for n in ("w13", "w13_sf", "w2", "w2_sf"))
    E, two_i, half_h = w13.shape; hidden, i_local = half_h * 2, two_i // 2
    x = torch.randn(4, hidden, device=dev, dtype=torch.bfloat16) * 0.5
    sel = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]] * 4, device=dev, dtype=torch.int32)
    w = torch.full((4, 8), 1.0 / 8, device=dev, dtype=torch.float32)
    lanes = lane_tables.served()
    served = lanes.moe(x, sel, w, w13, w13_sf, w2, w2_sf, F.swiglu_limit).float()
    ref = lane_tables.reference().moe(x, sel, w, w13, w13_sf, w2, w2_sf, F.swiglu_limit).float()
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
    print(f"  served b12x vs reference (8 experts, 4 tokens): rel {rel(served, ref):.3e}  |served| {served.abs().mean():.4f} |ref| {ref.abs().mean():.4f}")
    # variants of the reference, one assumption flipped each
    one = torch.ones((), device=dev)
    def variant(swap_halves=False, act_quant=True):
        out = torch.zeros(4, hidden, device=dev)
        for e in range(8):
            s13 = unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16).view(torch.float8_e4m3fn)
            s2 = unswizzle_sf(w2_sf[e].view(torch.uint8), hidden, i_local // 16).view(torch.float8_e4m3fn)
            a, b = (slice(i_local, None), slice(0, i_local)) if swap_halves else (slice(0, i_local), slice(i_local, None))
            u = expert_gemm(x.float(), w13[e, a], s13[a], one, one, quantize_act=act_quant)
            g = expert_gemm(x.float(), w13[e, b], s13[b], one, one, quantize_act=act_quant)
            y = expert_gemm(swiglu_clamped(g, u, F.swiglu_limit), w2[e], s2, one, one, quantize_act=act_quant)
            out += y.float() * w[:, e][:, None]
        return out
    for label, kw in (("halves swapped", dict(swap_halves=True)), ("no activation quant", dict(act_quant=False))):
        print(f"  served vs reference[{label}]: rel {rel(served, variant(**kw)):.3e}")
    # served with the swapped weight halves handed to the kernel
    w13_swapped = torch.cat([w13[:, i_local:], w13[:, :i_local]], dim=1).contiguous()
    sf_swapped = torch.stack([torch.cat([unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16)[i_local:], unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16)[:i_local]]) for e in range(E)])
    from engine.modules.nvfp4_sf import swizzle_sf
    sf_swapped = torch.stack([swizzle_sf(sf_swapped[e]) for e in range(E)]).view(torch.float8_e4m3fn).contiguous()
    served_sw = lanes.moe(x, sel, w, w13_swapped, sf_swapped, w2, w2_sf, F.swiglu_limit).float()
    print(f"  served[halves swapped] vs reference: rel {rel(served_sw, ref):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
