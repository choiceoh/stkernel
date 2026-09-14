"""Two statically scheduled stages over the existing GPTQ W4A8 planes.

The stream orders FC1 publication and FC2 consumption. Each FC2 CTA reduces
all K groups in registers, so there is no partial tensor, device queue,
cross-CTA polling or host completion read. The owner prepares/reuses storage
before graph capture; outputs are borrowed until its next invocation.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from engine.kernels.tile_dataflow import _w4a8_quantize, _w4a8_weight


@triton.jit
def _gate_up(X, W, S, RS, U, US, M: tl.constexpr, H: tl.constexpr,
             I: tl.constexpr, BM: tl.constexpr, LIMIT: tl.constexpr):
    r, t = tl.arange(0, BM), tl.arange(0, 128)
    for group in range(tl.program_id(0), I//128, tl.num_programs(0)):
        col = group*128+t
        g = tl.full((BM, 128), 0., tl.float32)
        u = tl.full((BM, 128), 0., tl.float32)
        for block in range(H//128):
            k = block*128+t
            x = tl.load(X+r[:, None]*H+k[None, :], r[:, None] < M, other=0)
            xq, xs = _w4a8_quantize(x)
            gate = _w4a8_weight(W, S, col[None, :], k[:, None], 2*I, H)
            up = _w4a8_weight(W, S, I+col[None, :], k[:, None], 2*I, H)
            g = tl.fma(tl.dot(xq, gate), xs[:, None], g)
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
            weight = _w4a8_weight(W, S, col[None, :], k[:, None], H, I)
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
            a, s = p.rows*p.intermediate, p.rows*p.producers*4
            self.views[p.rows, p.hidden, p.intermediate] = (
                self.storage[:a].view(p.rows, p.intermediate),
                self.storage[a:a+s].view(torch.float32).view(p.rows, p.producers),
                self.storage[a+s:a+s+p.rows*p.hidden*2].view(torch.bfloat16).view(p.rows, p.hidden))


def execute(plan, x, weights, limit, workspace):
    weights.validate_input(plan, x)
    if (workspace.device != x.device or (plan.rows, plan.hidden, plan.intermediate) not in workspace.shapes
            or not math.isfinite(limit) or limit <= 0):
        raise ValueError("W4A8 pipeline needs a matching prepared workspace and finite positive clamp")
    u, us, out = workspace.views[plan.rows, plan.hidden, plan.intermediate]
    g, d = weights.gate_up, weights.down
    bm = max(16, triton.next_power_of_2(plan.rows))
    _gate_up[(min(plan.workers, plan.producers),)](x, g.data, g.scale, g.rowscale, u, us,
        plan.rows, plan.hidden, plan.intermediate, bm, limit,
        num_warps=8, num_stages=1, enable_fp_fusion=False)
    _down[(min(plan.workers, plan.outputs),)](u, us, d.data, d.scale, d.rowscale, out,
        plan.rows, plan.hidden, plan.intermediate, bm, num_warps=8, num_stages=1, enable_fp_fusion=False)
    return out
