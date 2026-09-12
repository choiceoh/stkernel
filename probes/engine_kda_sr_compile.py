"""Compile the actual SR stores for SM121 without allocating or reserving a GPU."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from engine.kernels.kda.fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel
from engine.kernels.kda.rounding import _copy
from engine.kernels.state import _write_ring


def main():
    variants = [('prefill', _copy, dict(SRC='*fp32', DST='*fp16', POSITION='i64', SEED='i64'),
                 dict(H=16, K=128, V=128, S0=16384, S1=128, S2=1, BLOCK=256)),
                ('graph-writer', _write_ring, dict(SRC='*fp32', DST='*fp16', SLOT='*i64', CTX='*i64', ROUND_SEED='i64'),
                 dict(SS=262144, DS=7*262144, RS=262144, WIDTH=262144, RING=7, FIRST=0, BLOCK=256))]
    for bv in (8, 16):
        signature = {p: '*bf16' for p in ('q', 'k', 'v', 'g', 'beta', 'o')}
        signature.update({p: '*fp32' for p in ('a_log', 'g_bias')})
        signature.update(h0='*fp16', ht='*fp16', N='i64', T='i64', ring_slot='*i64', ring_context='*i64', ROUND_SEED='i64')
        constants = dict(cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
                         scale=128**-.5, B=1, H=16, HV=16, K=128, V=128, BK=128, BV=bv,
                         stride_init_state_token=262144, stride_final_state_token=262144,
                         stride_indices_seq=1, stride_indices_tok=1,
                         USE_INITIAL_STATE=True, INPLACE_FINAL_STATE=False, IS_BETA_HEADWISE=False,
                         USE_QK_L2NORM_IN_KERNEL=True, IS_VARLEN=False, IS_CONTINUOUS_BATCHING=False,
                         IS_SPEC_DECODING=False, IS_KDA=True, SIGMOID_BETA=True, COMPUTE_GATE=True,
                         SAFE_GATE=True, LOWER_BOUND=-5., STATE_KV=True, INPUT_STRIDES=None,
                         RING_SIZE=7, RING_SLOT_STRIDE=7*262144, RING_DEVICE_INDICES=True,
                         deferred_keys=None, deferred_decay=None, deferred_updates=None,
                         DEFERRED_STATE=False)
        variants.append((f'recurrent-bv{bv}', fused_recurrent_gated_delta_rule_fwd_kernel.fn, signature, constants))
    records = []
    for name, fn, signature, constants in variants:
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                target=GPUTarget('cuda', 121, 32), options=dict(num_warps=1, num_stages=3))
        ptx = kernel.asm['ptx']
        assert 'cvt.rs.f16x2.f32' not in ptx, name
        assert 'cvt.rz.f16' in ptx, name
        assert 'mul.hi.u32' in ptx, name  # Philox high products survive lowering
        records.append(dict(name=name, stochastic_conversion=True, shared_bytes=kernel.metadata.shared,
                            cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(scope='SM121 compilation only; no runtime quality verdict', gpu_used=False,
                          torch=torch.__version__, triton=triton.__version__, variants=records, status='PASS')))


if __name__ == '__main__':
    main()
