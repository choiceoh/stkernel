"""b12x vs reference, with activations that quantise EXACTLY (fp4-representable
values, block amax 6 -> scale 1): whatever remains is not activation
quantisation. Single expert, weight 1, halves in the kernel's [up; gate]
order. Then the clamp: a limit no value reaches vs the model's 10.
    bash probes/run_mk_probe.sh probes/b12x_lane_semantics2.py   (models mounted)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.base.loader import RankLoader
from engine.modules.moe import dequant_nvfp4
from engine.modules.nvfp4_sf import unswizzle_sf
from engine.profiles.glm53 import facts, lanes as lane_tables
from engine.profiles.glm53.lanes import swiglu_clamped


def main() -> int:
    dev = "cuda"; torch.manual_seed(0)
    F = facts.load()
    got = RankLoader(facts.RANKS / "rank0of4.safetensors").load([f"L3.moe.{n}" for n in ("w13", "w13_sf", "w2", "w2_sf")], device=dev)
    w13, w13_sf, w2, w2_sf = (got[f"L3.moe.{n}"] for n in ("w13", "w13_sf", "w2", "w2_sf"))
    E, two_i, half_h = w13.shape; hidden, i_local = half_h * 2, two_i // 2
    lanes = lane_tables.served()
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max().clamp_min(1e-6)).item()
    e = 3
    s13 = unswizzle_sf(w13_sf[e].view(torch.uint8), two_i, hidden // 16).view(torch.float8_e4m3fn)
    s2 = unswizzle_sf(w2_sf[e].view(torch.uint8), hidden, i_local // 16).view(torch.float8_e4m3fn)
    one = torch.ones((), device=dev)
    W13 = dequant_nvfp4(w13[e], s13, one)                      # [2I, H] fp32, rows [up; gate] in the kernel's order
    W2 = dequant_nvfp4(w2[e], s2, one)                         # [H, I]
    fp4_vals = torch.tensor([0., 0.5, 1., 1.5, 2., 3., 4., 6.], device=dev)
    # every 16-block holds a 6 so the kernel's dynamic scale is exactly 1; all values exact in e2m1
    x = fp4_vals[torch.randint(0, 8, (4, hidden), device=dev)] * torch.where(torch.rand(4, hidden, device=dev) < 0.5, -1.0, 1.0)
    x = x.view(4, hidden // 16, 16); x[:, :, 0] = 6.0; x = x.view(4, hidden).to(torch.bfloat16)
    sel1 = torch.tensor([[e]] * 4, device=dev, dtype=torch.int32); w1 = torch.ones(4, 1, device=dev)
    served = lanes.moe(x, sel1, w1, w13, w13_sf, w2, w2_sf, F.swiglu_limit).float()
    fc1 = x.float() @ W13.T                                    # [4, 2I]
    u, g = fc1[:, :i_local], fc1[:, i_local:]                  # kernel order: first half up, second half gate
    h = swiglu_clamped(g, u, F.swiglu_limit).float()
    ref_exact_h = h @ W2.T                                     # no fc2 activation quant
    # fc2 input quantised the reference way (dynamic per 16, global 1)
    from engine.modules.moe import quant_nvfp4_act, dequant_nvfp4_act
    p_, s_ = quant_nvfp4_act(h.to(torch.bfloat16), one); hq = dequant_nvfp4_act(p_, s_, one)
    ref_q = hq @ W2.T
    print(f"  exact fc1 inputs: served vs ref (fc2 act quant): rel {rel(served, ref_q):.3e}; vs ref (no fc2 quant): rel {rel(served, ref_exact_h):.3e}; "
          f"|served| {served.abs().mean():.4f} |ref| {ref_q.abs().mean():.4f}")
    print(f"  fc1 magnitudes: |gate| max {g.abs().max():.2f} |up| max {u.abs().max():.2f} (limit {F.swiglu_limit}); clamp touched gate {(g > F.swiglu_limit).float().mean():.3e} up {(u.abs() > F.swiglu_limit).float().mean():.3e}")
    # the other naming: gate first
    h_alt = swiglu_clamped(u, g, F.swiglu_limit).float()
    print(f"  served vs ref with halves the other way (gate first): rel {rel(served, quant_nvfp4_act and (dequant_nvfp4_act(*quant_nvfp4_act(h_alt.to(torch.bfloat16), one), one) @ W2.T)):.3e}")
    # per-token and column structure of the residual
    d = served - ref_q
    print(f"  residual per token rel: {[(round(v, 3)) for v in (d.abs().amax(-1) / ref_q.abs().amax(-1)).tolist()]}; "
          f"corr(served, ref) {torch.corrcoef(torch.stack([served.flatten(), ref_q.flatten()]))[0, 1].item():.5f}; "
          f"scale ratio |served|/|ref| {(served.abs().mean() / ref_q.abs().mean()).item():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
