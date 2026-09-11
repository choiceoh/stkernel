"""The two DeepGEMM library entry points used by ST's served lanes.

Import failure is a boot failure. The installed library must include its
compiled extension and JIT headers; no framework namespace is consulted.
"""
from functools import cache

from deep_gemm import (
    fp8_fp4_mqa_logits as _mqa_logits,
    set_pdl,
    tf32_hc_prenorm_gemm as _prenorm_gemm,
)

# The fleet seed disables PDL on SM12x: its KDA state kernels race there.
# Match that tested policy, including DeepGEMM's process-global setting.
@cache
def _initialize():
    # DeepGEMM creates a CUDA/cuBLAS context here. Defer until first execution
    # so package imports work before device assignment and in CPU checks.
    set_pdl(False)


def fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits):
    _initialize()
    return _mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits=clean_logits)


def tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split):
    _initialize()
    return _prenorm_gemm(x, fn, out, sqrsum, num_split)

__all__ = ["fp8_fp4_mqa_logits", "tf32_hc_prenorm_gemm"]
