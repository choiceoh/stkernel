"""Hold the KDA reference (modules/linear_attention: kda_gate + per-channel
delta rule) to GLM-5.3's SERVED kernels -- our own kda.py, run inside the
glm53 image with the overlay file on its target path. D4 at kernel level:
the reference is the thing that serves.

    decode form   fused_recurrent_kda(compute_gate=True, sigmoid_beta=True)
    prefill form  chunk_kda_with_fused_gate(...)

Conventions (scale, q/k l2norm) are found by grid, not assumed -- the GDN
check needed that once already.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.linear_attention import gated_delta_rule, kda_gate, l2norm


def main() -> int:
    from vllm.third_party.flash_linear_attention.ops.kda import fused_recurrent_kda, chunk_kda_with_fused_gate
    torch.manual_seed(0); dev = "cuda"
    B, T, H, D = 1, 64, 4, 128                      # GLM: 16 local heads x 128; 4 is enough to judge
    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16)
    q, k, v = mk(B, T, H, D), mk(B, T, H, D), mk(B, T, H, D)
    raw_g = mk(B, T, H, D) * 0.5
    beta_raw = mk(B, T, H)
    A_log = (torch.randn(H, device=dev) * 0.3).to(torch.bfloat16)
    g_bias = (torch.randn(H * D, device=dev) * 0.1).to(torch.bfloat16)
    beta = torch.sigmoid(beta_raw.float()).to(torch.bfloat16)
    g = kda_gate(raw_g, A_log, g_bias, -5.0, True)       # [B, T, H, D] fp32, per channel
    init = torch.zeros(B, H, D, D, device=dev, dtype=torch.float32)

    def rel(a, b):
        return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
    cu = torch.tensor([0, T], device=dev, dtype=torch.int32)      # varlen packed, one sequence
    rel = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()
    # The served contract, verbatim (glm5next_kda.py:692-748): scale D^-0.5,
    # l2norm INSIDE the kernel, cu_seqlens given, chunk path takes fp32
    # pre-sigmoided beta, recurrent path sigmoids raw beta itself.
    scale = D ** -0.5
    o_r, s_r = gated_delta_rule(q, k, v, g, beta, initial_state=init, scale=scale, qk_l2norm=True,
                                decay_per_channel=True)
    o_c, s_c = chunk_kda_with_fused_gate(q, k, v, raw_g, torch.sigmoid(beta_raw.float()), A_log, g_bias, scale,
                                         initial_state=init.clone(), output_final_state=True,
                                         use_qk_l2norm_in_kernel=True, cu_seqlens=cu,
                                         safe_gate=True, lower_bound=-5.0)
    # plain decode is the DENSE [B, T, H, D] form; cu_seqlens on the recurrent
    # kernel selects the spec-verify path, which also wants ssm_state_indices
    # and num_accepted_tokens (glm5next_kda.py:692-707)
    o_d, s_d = fused_recurrent_kda(q, k, v, raw_g, beta_raw, scale, initial_state=init.clone(),
                                   inplace_final_state=False, use_qk_l2norm_in_kernel=True,
                                   sigmoid_beta=True, a_log=A_log, g_bias=g_bias, compute_gate=True,
                                   lower_bound=-5.0)
    r_c, r_d, r_cd = rel(o_r, o_c), rel(o_r, o_d), rel(o_c, o_d)
    s_layout = min(rel(s_r, s_c), rel(s_r.transpose(-1, -2), s_c)) if s_c is not None and s_c.numel() == s_r.numel() else float("nan")
    print(f"  prefill (chunk_kda_with_fused_gate)  vs reference: o rel {r_c:.2e}")
    print(f"  decode  (fused_recurrent_kda)        vs reference: o rel {r_d:.2e}")
    print(f"  chunk vs recurrent (kernel vs kernel):             o rel {r_cd:.2e}")
    print(f"  final state, best of [H,K,V]/[H,V,K] vs reference: rel {s_layout:.2e}")
    print(f"  magnitudes: |o_ref| max {o_r.float().abs().max().item():.4f}  |o_chunk| {o_c.float().abs().max().item():.4f}  |o_rec| {o_d.float().abs().max().item():.4f}")
    ok = r_c < 2e-2 and r_d < 2e-2
    print("\n  " + ("KDA reference == served kernels, prefill and decode forms" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
