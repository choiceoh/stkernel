"""Compile both KDA state policies and deferred commit for SM121 without a GPU."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from engine.kernels.kda.deferred import _commit
from engine.kernels.kda.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    variants = []
    for t in (1, 6, 7, 8):
        bv = 16 if t <= 6 else 8
        for deferred in (False, True):
            signature = {p:"*bf16" for p in ("q", "k", "v", "g", "beta", "o")}
            signature.update({p:"*fp32" for p in ("h0", "ht", "a_log", "g_bias")})
            signature.update(N="i64", T="i64", ring_slot="*i64", ring_context="*i64")
            constants = dict(cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
                             scale=128**-.5, B=1, H=16, HV=16, K=128, V=128, BK=128, BV=bv,
                             stride_init_state_token=16*128*128, stride_final_state_token=16*128*128,
                             stride_indices_seq=1, stride_indices_tok=1,
                             USE_INITIAL_STATE=True, INPLACE_FINAL_STATE=False, IS_BETA_HEADWISE=False,
                             USE_QK_L2NORM_IN_KERNEL=True, IS_VARLEN=False, IS_CONTINUOUS_BATCHING=False,
                             IS_SPEC_DECODING=False, IS_KDA=True, SIGMOID_BETA=True, COMPUTE_GATE=True,
                             SAFE_GATE=True, LOWER_BOUND=-5., STATE_KV=True, INPUT_STRIDES=None,
                             RING_SIZE=8, RING_SLOT_STRIDE=8*16*128*128, RING_DEVICE_INDICES=True,
                             DEFERRED_STATE=deferred)
            for p in ("deferred_keys", "deferred_decay", "deferred_updates"):
                if deferred:
                    signature[p] = "*fp32"
                else:
                    constants[p] = None
            variants.append((f"verify-t{t}-deferred{int(deferred)}",
                             fused_recurrent_gated_delta_rule_fwd_kernel.fn, signature, constants))
        variants.append((f"commit-t{t}", _commit,
                         {**{p:"*fp32" for p in ("KEY", "DECAY", "UPDATE", "RING")},
                          **{p:"*i64" for p in ("SLOT", "CONTEXT", "COUNT")}},
                         dict(T=t, H=16, K=128, V=128, R=8, SLOT_STRIDE=8*16*128*128,
                              BK=128, BV=bv, BLOCK=768)))
    report = dict(scope="SM121 compilation only", gpu_used=False, torch=torch.__version__,
                  triton=triton.__version__, variants=[])
    for name, fn, signature, constants in variants:
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                target=GPUTarget("cuda", 121, 32), options=dict(num_warps=1, num_stages=3))
        (args.output/(name+".ptx")).write_text(kernel.asm["ptx"])
        report["variants"].append(dict(name=name, shared_bytes=kernel.metadata.shared,
                                       cubin_sha256=hashlib.sha256(kernel.asm["cubin"]).hexdigest()))
    assert not torch.cuda.is_initialized()
    report["status"] = "PASS"
    (args.output/"result.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
