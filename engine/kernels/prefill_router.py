"""Long-prefill router GEMM: original BF16 operands, FP32 accumulation/output.

The existing sigmoid, bias, top-k and normalized route weights stay in Torch.
Tensor-core summation may change FP32 rounding; actual-weight route and full
consumer quality gates are required. Short prefill and decode use the old GEMM.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M'])
def _router_gemm(X, W, Out, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # Adjacent CTAs cover the five expert tiles of one input tile. Keep the
    # reused activation tile hot instead of walking the full input five times.
    rows = (tl.program_id(0)//tl.cdiv(288,BN))*BM + tl.arange(0,BM)
    cols = (tl.program_id(0)%tl.cdiv(288,BN))*BN + tl.arange(0,BN)
    kk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN),tl.float32)
    for block in range(4096//BK):
        k = block*BK + kk
        a = tl.load(X + rows[:,None]*4096 + k[None,:], mask=rows[:,None] < M, other=0.)
        b = tl.load(W + cols[None,:]*4096 + k[:,None], mask=cols[None,:] < 288, other=0.)
        acc = tl.dot(a,b,acc)
    tl.store(Out + rows[:,None]*288 + cols[None,:],acc,
             mask=(rows[:,None] < M) & (cols[None,:] < 288))


def router_logits(x, weight):
    if (x.ndim != 2 or weight.ndim != 2 or not 8192 < x.shape[0] <= 32768
            or x.shape[1] != 4096 or tuple(weight.shape) != (288,4096)
            or not x.is_cuda or not weight.is_cuda or x.device != weight.device
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_contiguous() or not weight.is_contiguous()):
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    out = torch.empty((x.shape[0],288),device=x.device,dtype=torch.float32)
    _router_gemm[(triton.cdiv(x.shape[0],64)*triton.cdiv(288,64),)](
        x,weight,out,x.shape[0],BM=64,BN=64,BK=64,num_warps=4,num_stages=3,
        enable_fp_fusion=False)
    return out
