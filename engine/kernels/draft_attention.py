"""DFlash block attention directly over its circular, grouped-query KV cache.

One query/head per CTA keeps 6 x 32 independent CTAs on GB10. The kernel
never expands KV heads, gathers the whole ring, or materializes scores.
Position is a device scalar so the same graph handles ring wrap and startup.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _attention(Q, K, V, R, P, O, Slot, SLOT_STRIDE: tl.constexpr, LAYER_OFFSET: tl.constexpr,
               B: tl.constexpr, H: tl.constexpr,
               HK: tl.constexpr, RHK: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
               RS: tl.constexpr, SCALE: tl.constexpr, BN: tl.constexpr):
    query = tl.program_id(0)
    if SLOT_STRIDE:
        R += tl.load(Slot).to(tl.int64)*SLOT_STRIDE + LAYER_OFFSET
    head = tl.program_id(1)
    kh = head // (H // HK)
    d = tl.arange(0, D)
    q = tl.load(Q + (query * H + head) * D + d).to(tl.float32)
    position = tl.load(P)
    maximum = tl.full((), -float("inf"), tl.float32)
    denominator = tl.full((), 0., tl.float32)
    accumulator = tl.full((D,), 0., tl.float32)
    for start in range(tl.cdiv(W + B, BN)):
        n = start * BN + tl.arange(0, BN)
        context = n < W
        absolute = position - W + n
        valid = (n < W + B) & (~context | (absolute >= 0))
        slot = (absolute + W) % W
        kr = tl.load(R + (slot[:, None] * RHK + kh) * D + d[None, :],
                     context[:, None] & valid[:, None], other=0)
        kb = tl.load(K + ((n[:, None] - W) * HK + kh) * D + d[None, :],
                     ~context[:, None] & valid[:, None], other=0)
        key = tl.where(context[:, None], kr, kb).to(tl.float32)
        score = tl.sum(key * q[None, :], 1) * SCALE
        score = tl.where(valid, score, -float("inf"))
        block_max = tl.max(score, 0)
        new_max = tl.maximum(maximum, block_max)
        # Empty initial blocks must not turn -inf - -inf into NaN.
        safe_max = tl.where(new_max == -float("inf"), 0., new_max)
        correction = tl.exp(maximum - safe_max)
        probability = tl.exp(score - safe_max)
        vr = tl.load(R + RS + (slot[:, None] * RHK + kh) * D + d[None, :],
                     context[:, None] & valid[:, None], other=0)
        vb = tl.load(V + ((n[:, None] - W) * HK + kh) * D + d[None, :],
                     ~context[:, None] & valid[:, None], other=0)
        value = tl.where(context[:, None], vr, vb).to(tl.float32)
        accumulator = accumulator * correction + tl.sum(probability[:, None] * value, 0)
        denominator = denominator * correction + tl.sum(probability, 0)
        maximum = new_max
    tl.store(O + (query * H + head) * D + d, accumulator / denominator)


def draft_attention(q, k, v, ring, position, *, slot=None, layer=0):
    """BF16 [B,H,128], [B,HK,128], ring [2,W,HK,128] -> [B,H,128]."""
    if slot is not None:
        if (ring.ndim != 6 or not 0 <= layer < ring.shape[1] or slot.numel()!=1
                or slot.dtype!=torch.int64 or slot.device!=q.device):
            raise ValueError("DFlash arena requires a device slot and an in-range layer")
        geometry=ring.shape[2:]
        stride,offset=ring.stride(0),layer*ring.stride(1)
    else:
        geometry=ring.shape
        stride,offset=0,0
    if (q.ndim != 3 or k.ndim != 3 or k.shape != v.shape or len(geometry) != 4
            or q.shape[0] != k.shape[0] or q.shape[2] != 128 or k.shape[2] != 128
            or geometry[0] != 2 or geometry[2] < k.shape[1] or geometry[3] != k.shape[2]
            or q.shape[1] % k.shape[1] or not 1 <= q.shape[0] <= 32
            or not all(t.is_cuda and t.device == q.device and t.dtype == torch.bfloat16
                       for t in (q, k, v, ring))
            or not all(t.is_contiguous() for t in (q,k,v))
            or not (ring[0].is_contiguous() if slot is not None else ring.is_contiguous())):
        raise ValueError("DFlash attention requires contiguous CUDA BF16 block and KV ring")
    if isinstance(position, int):
        if position < 0:
            raise ValueError("negative DFlash position")
        position = torch.tensor(position, device=q.device, dtype=torch.int64)
    if (position.device != q.device or position.dtype != torch.int64 or position.numel() != 1):
        raise ValueError("DFlash position must be a CUDA int64 scalar")
    out = torch.empty_like(q)
    b, h, d = q.shape
    _attention[(b, h)](q, k, v, ring, position, out, slot if slot is not None else position,
                       stride, offset, b, h, k.shape[1], geometry[2], d,
                       geometry[1], geometry[1]*geometry[2]*geometry[3], d**-.5, 32,
                       num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit
def _write_kv(K,V,R,Slot,Pos,Valid,HAS_VALID:tl.constexpr,N:tl.constexpr,WIDTH:tl.constexpr,ROW:tl.constexpr,W:tl.constexpr,
              STRIDE:tl.constexpr,OFFSET:tl.constexpr,BLOCK:tl.constexpr):
    token=tl.program_id(0)
    d=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(Slot).to(tl.int64)
    pos=tl.load(Pos+token).to(tl.int64)%W
    base=R+slot*STRIDE+OFFSET+pos*ROW+d
    accepted = token < tl.load(Valid) if HAS_VALID else True
    mask=(d<WIDTH)&accepted
    tl.store(base,tl.load(K+token*WIDTH+d,mask,other=0),mask)
    tl.store(base+W*ROW,tl.load(V+token*WIDTH+d,mask,other=0),mask)


def write_draft_kv(field,slot,layer,positions,k,v,*,valid=None):
    """Write only accepted positions in the arena, without copying a whole ring."""
    if (field.ndim!=6 or k.shape!=v.shape or k.shape[-1]!=field.shape[-1] or k.shape[-2]>field.shape[-2]
            or positions.numel()!=k.shape[0] or k.shape[0]>field.shape[3]
            or slot.numel()!=1 or slot.dtype!=torch.int64 or positions.dtype!=torch.int64
            or not 0<=layer<field.shape[1]):
        raise ValueError("invalid direct DFlash ring write")
    if valid is not None and (valid.numel()!=1 or valid.dtype!=torch.int64 or valid.device!=field.device):
        raise ValueError("accepted DFlash count must be an int64 device scalar")
    width=k.shape[-1]*k.shape[-2]
    _write_kv[(k.shape[0],triton.cdiv(width,256))](
        k.contiguous(),v.contiguous(),field,slot,positions,valid if valid is not None else slot,valid is not None,
        k.shape[0],width,field.shape[-1]*field.shape[-2],field.shape[3],
        field.stride(0),layer*field.stride(1),256)
