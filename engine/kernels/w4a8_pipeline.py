"""Statically scheduled stages over the existing GPTQ W4A8 planes.

Quantize each input group once, then share it across all FC1 output tiles.
The stream orders input/FC1 publication and FC2 consumption. Each FC2 CTA reduces
all K groups in registers, so there is no partial tensor, device queue,
cross-CTA polling or host completion read. The owner prepares/reuses storage
before graph capture; outputs are borrowed until its next invocation.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from engine.kernels.tile_dataflow import _w4a8_quantize, W4A8_MMA_ARCH as MMA_ARCH

# Triton 3.7.1 recognizes native FP8 mmav2 at capability == 120, but
# mistakenly promotes SM121's operands to FP16. Select the consumer Blackwell
# lowering for these two GEMMs. The backend still emits the actual device's
# sm_121a PTX/cubin; admission remains SM121. No precision knob is relaxed.
# triton-lang/triton v3.7.1 AccelerateMatmul.cpp::mmav2SupportsFp8Operands

@triton.jit
def _weight_tile(W, S, tile, block, K: tl.constexpr):
    # Read each packed byte once in its native contiguous [128,64] tile.
    # Adjacent K values share the scale/table decision as well as the byte.
    r, c = tl.arange(0, 128), tl.arange(0, 64)
    base = (tile*(K//128)+block)*128
    packed = tl.load(W+(base+r[:, None])*64+c[None, :]).to(tl.int32)
    d = tl.load(S+(base+r[:, None])*8+c[None, :]//8).to(tl.int32)
    table = tl.where(((d & 7) >= 1) & ((d & 7) <= 5),
                     0x4D4845403D383000, 0x4C4844403C383000).to(tl.uint64)
    lo, hi = packed & 15, packed >> 4
    lm, hm = lo & 7, hi & 7
    lv = ((table >> (lm*8)) & 255).to(tl.int32)
    hv = ((table >> (hm*8)) & 255).to(tl.int32)
    lv = tl.where(lm == 0, 0, lv+d) | ((lo & 8) << 4)
    hv = tl.where(hm == 0, 0, hv+d) | ((hi & 8) << 4)
    return tl.interleave(lv, hv).to(tl.uint8).to(tl.float8e4nv, bitcast=True).T


@triton.jit
def _input(X, XQ, XS, M: tl.constexpr, H: tl.constexpr):
    group = tl.program_id(0)*4+tl.arange(0, 4)
    col = tl.arange(0, 128)
    x = tl.load(X+group[:, None]*128+col[None, :], group[:, None] < M*(H//128), other=0)
    q, scale = _w4a8_quantize(x)
    tl.store(XQ+group[:, None]*128+col[None, :], q.to(tl.uint8, bitcast=True), group[:, None] < M*(H//128))
    tl.store(XS+group, scale, group < M*(H//128))


@triton.jit
def _gate_up(XQ, XS, W, S, RS, U, US, M: tl.constexpr, H: tl.constexpr,
             I: tl.constexpr, BM: tl.constexpr, LIMIT: tl.constexpr):
    r, t = tl.arange(0, BM), tl.arange(0, 128)
    for group in range(tl.program_id(0), I//128, tl.num_programs(0)):
        col = group*128+t
        g = tl.full((BM, 128), 0., tl.float32)
        u = tl.full((BM, 128), 0., tl.float32)
        for block in range(H//128):
            k = block*128+t
            xq = tl.load(XQ+r[:, None]*H+k[None, :], r[:, None] < M, other=0).to(tl.float8e4nv, bitcast=True)
            xs = tl.load(XS+r*(H//128)+block, r < M, other=0)
            gate = _weight_tile(W, S, group, block, H)
            g = tl.fma(tl.dot(xq, gate), xs[:, None], g)
            up = _weight_tile(W, S, I//128+group, block, H)
            u = tl.fma(tl.dot(xq, up), xs[:, None], u)
        gs, us = tl.load(RS+col), tl.load(RS+I+col)
        g = (g*gs[None, :]).to(tl.bfloat16).to(tl.float32)
        u = (u*us[None, :]).to(tl.bfloat16).to(tl.float32)
        g, u = tl.minimum(g, LIMIT), tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        value = (g*tl.div_rn(1., 1.+libdevice.exp(-g)))*u
        uq, scale = _w4a8_quantize(value.to(tl.bfloat16))
        tl.store(U+r[:, None]*I+col[None, :], uq.to(tl.uint8, bitcast=True), r[:, None] < M)
        tl.store(US+r*(I//128)+group, scale, r < M)


@triton.jit
def _down(U, US, W, S, RS, OUT, M: tl.constexpr, H: tl.constexpr,
          I: tl.constexpr, BM: tl.constexpr):
    r, t = tl.arange(0, BM), tl.arange(0, 128)
    for tile in range(tl.program_id(0), H//128, tl.num_programs(0)):
        col = tile*128+t
        acc = tl.full((BM, 128), 0., tl.float32)
        for group in range(I//128):
            k = group*128+t
            u = tl.load(U+r[:, None]*I+k[None, :], r[:, None] < M, other=0).to(tl.float8e4nv, bitcast=True)
            scale = tl.load(US+r*(I//128)+group, r < M, other=0)
            weight = _weight_tile(W, S, tile, group, I)
            # Keep the queued oracle's separately rounded scale multiply then
            # ordered sum; only the location of these FP32 partials changes.
            part = tl.dot(u, weight)*scale[:, None]
            acc += part
        out = acc*tl.load(RS+col)[None, :]
        tl.store(OUT+r[:, None]*H+col[None, :], out, r[:, None] < M)


class Workspace:
    """One reusable, stream-owned buffer; not safe for concurrent invocations."""
    def __init__(self, plans, device):
        plans = tuple(plans)
        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.cuda.get_device_capability(device) != (12, 1):
            raise ValueError("W4A8 pipeline requires SM121")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("prepare W4A8 workspace before graph capture")
        self.shapes = frozenset((p.rows, p.hidden, p.intermediate) for p in plans)
        size = max(p.scratch_bytes for p in plans)
        self.storage = torch.empty(size, dtype=torch.uint8, device=device)
        self.device = self.storage.device
        self.views = {}
        for p in plans:
            shapes = ((p.rows, p.intermediate), (p.rows, p.producers),
                      (p.rows, p.hidden), (p.rows, p.hidden), (p.rows, p.hidden//128))
            dtypes = (torch.uint8, torch.float32, torch.bfloat16, torch.uint8, torch.float32)
            self.views[p.rows, p.hidden, p.intermediate] = tuple(
                self.storage[start:end].view(dtype).view(shape)
                for (start, end), dtype, shape in zip(p.buffer_spans, dtypes, shapes))


def execute(plan, x, weights, limit, workspace):
    weights.validate_input(plan, x)
    if (workspace.device != x.device or (plan.rows, plan.hidden, plan.intermediate) not in workspace.shapes
            or not math.isfinite(limit) or limit <= 0):
        raise ValueError("W4A8 pipeline needs a matching prepared workspace and finite positive clamp")
    u, us, out, xq, xs = workspace.views[plan.rows, plan.hidden, plan.intermediate]
    lo, hi = x.data_ptr(), x.data_ptr()+x.numel()*x.element_size()
    if any(t.data_ptr() < hi and lo < t.data_ptr()+t.numel()*t.element_size() for t in (xq, xs)):
        raise ValueError("W4A8 input overlaps its quantization publication workspace")
    g, d = weights.gate_up, weights.down
    bm = max(16, triton.next_power_of_2(plan.rows))
    _input[(triton.cdiv(plan.rows*(plan.hidden//128), 4),)](x, xq, xs, plan.rows, plan.hidden,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _gate_up[(min(plan.workers, plan.producers),)](xq, xs, g.data, g.scale, g.rowscale, u, us,
        plan.rows, plan.hidden, plan.intermediate, bm, limit,
        num_warps=8, num_stages=1, enable_fp_fusion=False, arch=MMA_ARCH)
    _down[(min(plan.workers, plan.outputs),)](u, us, d.data, d.scale, d.rowscale, out,
        plan.rows, plan.hidden, plan.intermediate, bm, num_warps=8, num_stages=1, enable_fp_fusion=False, arch=MMA_ARCH)
    return out
