"""Hold engine/modules/linear_attention.py to HF's gated delta rule.

Two judges, because the runner relies on their agreement: HF's chunked form
(what a prefill step runs) and HF's recurrent form (what a decode step runs).
The third check is the one the runner actually depends on -- prefill T tokens,
then continue ONE token from the final state, must equal running T+1 straight.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.modules.linear_attention import gated_delta_rule
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule)

TOL_O, TOL_S = 2e-3, 5e-3


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def main() -> int:
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, H, D = 2, 129, 4, 32                       # T not a multiple of the chunk (64)
    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.bfloat16)
    q, k, v = mk(B, T, H, D), mk(B, T, H, D), mk(B, T, H, D)
    g = (-torch.rand(B, T, H, device=dev) * 0.5).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(B, T, H, device=dev)).to(torch.bfloat16)
    good = True

    o, s = gated_delta_rule(q, k, v, g, beta)
    o_c, s_c = torch_chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True)
    o_r, s_r = torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    e = (o.float() - o_c.float()).abs().max().item(); es = (s - s_c.float()).abs().max().item()
    good &= check("== HF chunked (prefill form)", e < TOL_O and es < TOL_S, f"o {e:.2e} state {es:.2e}")
    e = (o.float() - o_r.float()).abs().max().item(); es = (s - s_r.float()).abs().max().item()
    good &= check("== HF recurrent (decode form)", e < TOL_O and es < TOL_S, f"o {e:.2e} state {es:.2e}")

    # prefill T-1 then decode 1 from the final state == T straight
    o_pre, s_pre = gated_delta_rule(q[:, :-1], k[:, :-1], v[:, :-1], g[:, :-1], beta[:, :-1])
    o_dec, s_dec = gated_delta_rule(q[:, -1:], k[:, -1:], v[:, -1:], g[:, -1:], beta[:, -1:], initial_state=s_pre)
    e = (o_dec.float()[:, 0] - o.float()[:, -1]).abs().max().item(); es = (s_dec - s).abs().max().item()
    good &= check("prefill state -> one decode step == straight run", e < 1e-5 and es < 1e-5, f"o {e:.2e} state {es:.2e}")

    # HF chunked from OUR final state continues identically (the state is interchangeable)
    o_hf_dec, _ = torch_recurrent_gated_delta_rule(q[:, -1:], k[:, -1:], v[:, -1:], g[:, -1:], beta[:, -1:],
                                                   initial_state=s_pre, output_final_state=False, use_qk_l2norm_in_kernel=True)
    e = (o_hf_dec.float()[:, 0] - o.float()[:, -1]).abs().max().item()
    good &= check("HF decode from OUR prefill state", e < TOL_O, f"o {e:.2e}")
    print("\n  " + ("ALL PASS" if good else "SOMETHING FAILED"))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
