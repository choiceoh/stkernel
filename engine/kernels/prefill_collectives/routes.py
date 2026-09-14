"""Sender-owned routing in the same TP4 FP8 all-gather as the activations.

The replicated gate/bias run once per token, before communication. IDs are
lossless uint16 (288 experts); normalized weights retain all FP32 bits. The
source roundtrip buffer is local and released before the all-gather. No
global BF16 activation or second data collective is introduced.
"""
import torch
import triton
import triton.language as tl

from engine.modules.prefill_packets import PacketBatch, PacketGeometry, ffn_packet_rows


@triton.jit(do_not_specialize=['LOCAL_ROWS', 'ACTIVATION_BYTES', 'PACKET_BYTES'])
def _write_routes(Ids, Weights, Packet, LOCAL_ROWS, ACTIVATION_BYTES, PACKET_BYTES,
                  B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    ids = tl.load(Ids+i, i < LOCAL_ROWS*8, other=0)
    weights = tl.load(Weights+i, i < LOCAL_ROWS*8, other=0.)
    tl.store(Packet.to(tl.pointer_type(tl.uint16))+ACTIVATION_BYTES//2+i,
             ids.to(tl.uint16), i < LOCAL_ROWS*8)
    tl.store(Packet.to(tl.pointer_type(tl.float32))+ACTIVATION_BYTES//4+LOCAL_ROWS*4+i,
             weights, i < LOCAL_ROWS*8)
    if tl.program_id(0) == tl.cdiv(LOCAL_ROWS*8, B)-1:
        tail = ACTIVATION_BYTES+LOCAL_ROWS*48+tl.arange(0, 128)
        tl.store(Packet+tail, 0, tail < PACKET_BYTES)


@triton.jit(do_not_specialize=['ROWS', 'LOCAL_ROWS', 'ACTIVATION_BYTES', 'PACKET_BYTES'])
def _read_routes(Packet, Ids, Weights, ROWS, LOCAL_ROWS, ACTIVATION_BYTES, PACKET_BYTES,
                 B: tl.constexpr):
    i = tl.program_id(0)*B + tl.arange(0, B)
    rank, local = i//(LOCAL_ROWS*8), i % (LOCAL_ROWS*8)
    ids = tl.load(Packet.to(tl.pointer_type(tl.uint16))+rank*(PACKET_BYTES//2)
                  + ACTIVATION_BYTES//2+local, i < ROWS*8, other=0)
    weights = tl.load(Packet.to(tl.pointer_type(tl.float32))+rank*(PACKET_BYTES//4)
                      + ACTIVATION_BYTES//4+LOCAL_ROWS*4+local, i < ROWS*8, other=0.)
    tl.store(Ids+i, ids.to(tl.int32), i < ROWS*8)
    tl.store(Weights+i, weights, i < ROWS*8)


def pack_routed(x, geometry, route):
    """Pack one shard and append routes computed from its transport values.

    `route` is the bound model's existing selection, with replicated weights
    and bias. It receives the exact FP8-to-BF16 roundtrip, never the original
    unquantized activation. The caller already agreed all readers on TP4.
    """
    from .kernels import _pack_rs_payload
    g = geometry
    if (not isinstance(g, PacketGeometry) or not g.routed or not ffn_packet_rows(g.rows)
            or tuple(x.shape) != (g.local_rows, g.hidden) or not x.is_cuda
            or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or torch.cuda.is_current_stream_capturing()):
        raise ValueError('routed packet requires one eager CUDA BF16 shard of an eligible TP4 prefill')
    payload = torch.empty(g.stride, device=x.device, dtype=torch.uint8)
    roundtrip = torch.empty_like(x)
    _pack_rs_payload[(g.local_elements//g.block,)](
        x, payload.view(torch.float8_e4m3fn), payload.view(torch.float32),
        g.local_elements, g.local_elements, g.stride, BLOCK=g.block, Roundtrip=roundtrip)
    ids, weights = route(roundtrip)
    if (tuple(ids.shape) != (g.local_rows, 8) or tuple(weights.shape) != (g.local_rows, 8)
            or ids.dtype != torch.int32 or weights.dtype != torch.float32
            or ids.device != x.device or weights.device != x.device
            or not ids.is_contiguous() or not weights.is_contiguous()):
        raise ValueError('sender routes require contiguous local top-8 int32 IDs and FP32 weights')
    _write_routes[(triton.cdiv(g.local_rows*8, 256),)](
        ids, weights, payload, g.local_rows, g.activation_bytes, g.stride, B=256)
    return payload


def packet_routes(batch):
    """Restore rank-ordered routes, excluding every transport padding row."""
    if not isinstance(batch, PacketBatch) or not batch.geometry.routed:
        raise ValueError('received FFN routes require the v2 routed packet ABI')
    g, x = batch.geometry, batch.received
    ids = torch.empty((g.rows, 8), device=x.device, dtype=torch.int32)
    weights = torch.empty((g.rows, 8), device=x.device, dtype=torch.float32)
    _read_routes[(triton.cdiv(g.rows*8, 256),)](
        x, ids, weights, g.rows, g.local_rows, g.activation_bytes, g.stride, B=256)
    return ids, weights
