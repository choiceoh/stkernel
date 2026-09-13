"""Long-prefill router GEMM: original BF16 operands, FP32 accumulation/output.

The existing sigmoid, bias, top-k and normalized route weights stay in Torch.
Tensor-core summation may change FP32 rounding; actual-weight route and full
consumer quality gates are required. Short prefill and decode use the old GEMM.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['M', 'LOCAL_ROWS', 'PACKET_BYTES'])
def _router_gemm(X, W, Out, M, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 Scales=None, LOCAL_ROWS=0, PACKET_BYTES=0, PACKETS: tl.constexpr=False):
    # Adjacent CTAs cover the five expert tiles of one input tile. Keep the
    # reused activation tile hot instead of walking the full input five times.
    rows = (tl.program_id(0)//tl.cdiv(288,BN))*BM + tl.arange(0,BM)
    cols = (tl.program_id(0)%tl.cdiv(288,BN))*BN + tl.arange(0,BN)
    kk = tl.arange(0,BK)
    acc = tl.zeros((BM,BN),tl.float32)
    for block in range(4096//BK):
        k = block*BK + kk
        if PACKETS:
            rank, local_row = rows // LOCAL_ROWS, rows % LOCAL_ROWS
            offset = local_row[:,None]*4096 + k[None,:]
            v = tl.load(X + rank[:,None]*PACKET_BYTES + offset, rows[:,None] < M, other=0.).to(tl.float32)
            scale = tl.load(Scales + rank[:,None]*(PACKET_BYTES//4) + LOCAL_ROWS*1024
                            + offset//2048, rows[:,None] < M, other=0.)
            a = (v*scale).to(tl.bfloat16)
        else:
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


def router_packet_logits(batch, weight):
    from engine.modules.prefill_packets import PacketBatch, ffn_packet_rows
    if (not isinstance(batch, PacketBatch) or not ffn_packet_rows(batch.geometry.rows)
            or tuple(weight.shape) != (288,4096) or not weight.is_cuda
            or weight.device != batch.received.device or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()):
        raise ValueError('packet router requires the long-prefill TP4/H4096/E288 contract')
    g, x = batch.geometry, batch.received
    out = torch.empty((g.rows,288), device=x.device, dtype=torch.float32)
    _router_gemm[(triton.cdiv(g.rows,64)*triton.cdiv(288,64),)](
        x.view(torch.float8_e4m3fn), weight, out, g.rows,
        BM=64, BN=64, BK=64, Scales=x.view(torch.float32), LOCAL_ROWS=g.local_rows,
        PACKET_BYTES=g.stride, PACKETS=True, num_warps=4, num_stages=3, enable_fp_fusion=False)
    return out
