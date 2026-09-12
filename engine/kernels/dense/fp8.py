"""One activation quantization launch for the DeepGEMM 128-column recipe."""
import torch
import triton
import triton.language as tl


@triton.jit
def _quantize(X,Q,S,K:tl.constexpr,G:tl.constexpr):
    row=tl.program_id(0)
    group=tl.program_id(1)*4+tl.arange(0,4)
    col=group[:,None]*128+tl.arange(0,128)[None,:]
    x=tl.load(X+row*K+col,col<K,other=0).to(tl.float32)
    amax=tl.maximum(tl.max(tl.abs(x),1),1e-4)
    scale=tl.exp2(tl.ceil(tl.log2(amax/448.)))
    tl.store(Q+row*K+col,(x/scale[:,None]).to(tl.float8e4nv),col<K)
    tl.store(S+row*G+group,scale,group<G)


def quantize(x):
    if x.ndim!=2 or x.shape[1]%128 or x.dtype!=torch.bfloat16 or not x.is_contiguous():
        raise ValueError("FP8 quantization requires contiguous BF16 rows with K aligned to 128")
    m,k=x.shape
    q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
    scales=torch.empty((m,k//128),device=x.device,dtype=torch.float32)
    _quantize[(m,triton.cdiv(k//128,4))](x,q,scales,k,k//128,num_warps=4)
    return q,scales
