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


def require_disjoint(*tensors):
    """Producer outputs must not clobber their input or one another."""
    intervals = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()) for t in tensors if t.numel())
    if any(a1 > b0 for (_, a1), (b0, _) in zip(intervals, intervals[1:])):
        raise ValueError('FP8 producer input/output buffers overlap')


def quantize(x, *, out=None):
    if (x.ndim!=2 or min(x.shape)<=0 or x.shape[1]%128 or x.dtype!=torch.bfloat16
            or not x.is_cuda or not x.is_contiguous()):
        raise ValueError("FP8 quantization requires contiguous BF16 rows with K aligned to 128")
    m,k=x.shape
    if out is None:
        q=torch.empty_like(x,dtype=torch.float8_e4m3fn)
        scales=torch.empty((m,k//128),device=x.device,dtype=torch.float32)
    else:
        q,scales=out
        if (q.shape!=x.shape or q.dtype!=torch.float8_e4m3fn or q.device!=x.device
                or scales.shape!=(m,k//128) or scales.dtype!=torch.float32 or scales.device!=x.device
                or not q.is_contiguous() or not scales.is_contiguous()):
            raise ValueError('FP8 output buffers do not match their input')
        require_disjoint(x, q, scales)
    _quantize[(m,triton.cdiv(k//128,4))](x,q,scales,k,k//128,num_warps=4)
    return q,scales
