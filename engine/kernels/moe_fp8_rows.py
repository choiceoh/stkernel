"""A few rows' routed experts on block-scaled FP8 weights, BF16 activations (kernels): the MTP head's MoE in the
checkpoint's own precision.

Qwen3.8's NVIDIA export keeps the MTP head's 512 experts in FP8 -- e4m3 [out, in] under one BF16 scale a 128 x 128 tile
that multiplies -- where the target layers' are NVFP4. The b12x dispatcher serves NVFP4 (and W4A16) only, so the
preshard re-encoded the MTP experts as NVFP4 from the FP8 (quantised twice, activations to FP4 too). This serves them
as they are: every (row, route) pair of this rank's experts computes its expert's FFN from the FP8 weights and the
row's BF16 activation -- the weights widened in registers, FP32 sums -- in two launches:

    gate_up   a program a (pair, block of the intermediate): gate and up for the pair's row, each rounded to BF16 as
              the reference's GEMM outputs are, silu(gate) * up rounded to BF16 -> act [pairs, I]
    down      a program a (row, block of the hidden width): the row's routes in order, each act @ down.T times the
              route's weight, summed in FP32 and rounded once -> [rows, H] BF16, the routed partial

A pair reads its expert's weights on its own: the draft rows are few (the MTP head runs past its attention only the rows
a caller reads -- net.mtp_forward), so pairs rarely share an expert. Routes to another rank's experts carry weight 0 or
an id past the rank's experts (lanes.local_routes, route_local's sentinel) and are skipped. Launch-shape independent:
nothing is read back to the host, so a captured step replays it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 16
BLOCK = 128                                   # the checkpoint's scale tile


@triton.jit
def _gate_up(X, IDS, WTS, W13, S13, ACT, E, TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr,
             BLOCK_I: tl.constexpr):
    p, ib = tl.program_id(0), tl.program_id(1)
    e = tl.load(IDS + p)
    w = tl.load(WTS + p)
    if (w != 0) & (e >= 0) & (e < E):
        row = p // TOPK
        i = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        live = i < I
        gate = tl.zeros((BLOCK_I,), dtype=tl.float32)
        up = tl.zeros((BLOCK_I,), dtype=tl.float32)
        base = W13 + e.to(tl.int64) * (2 * I * H)
        sbase = S13 + e.to(tl.int64) * ((2 * I // 128) * (H // 128))
        for kb in range(H // 128):
            ks = kb * 128 + tl.arange(0, 128)
            x = tl.load(X + row * H + ks).to(tl.float32)
            g = tl.load(base + i[:, None] * H + ks[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            u = tl.load(base + (I + i)[:, None] * H + ks[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            sg = tl.load(sbase + (i // 128) * (H // 128) + kb, mask=live, other=0.0)
            su = tl.load(sbase + ((I + i) // 128) * (H // 128) + kb, mask=live, other=0.0)
            gate += tl.sum(g * x[None, :], 1) * sg
            up += tl.sum(u * x[None, :], 1) * su
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        act = (gate * tl.sigmoid(gate)) * up
        tl.store(ACT + p * I + i, act.to(tl.bfloat16), mask=live)


@triton.jit
def _down(ACT, IDS, WTS, W2, S2, OUT, E, TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr, BLOCK_H: tl.constexpr):
    row, hb = tl.program_id(0), tl.program_id(1)
    hs = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    live = hs < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    for r in range(TOPK):
        p = row * TOPK + r
        e = tl.load(IDS + p)
        w = tl.load(WTS + p)
        if (w != 0) & (e >= 0) & (e < E):
            base = W2 + e.to(tl.int64) * (H * I)
            sbase = S2 + e.to(tl.int64) * ((H // 128) * (I // 128))
            y = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for kb in range(I // 128):
                ks = kb * 128 + tl.arange(0, 128)
                a = tl.load(ACT + p * I + ks).to(tl.float32)
                d = tl.load(base + hs[:, None] * I + ks[None, :], mask=live[:, None], other=0.0).to(tl.float32)
                s = tl.load(sbase + (hs // 128) * (I // 128) + kb, mask=live, other=0.0)
                y += tl.sum(d * a[None, :], 1) * s
            acc += w * y
    tl.store(OUT + row * H + hs, acc.to(tl.bfloat16), mask=live)


def moe(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, w13: torch.Tensor, s13: torch.Tensor,
        w2: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """x [M <= 16, H] BF16, ids [M, k] int32 (this rank's expert ids; a route to skip has weight 0 or an id >= E),
    weights [M, k] FP32; w13 [E, 2I, H] e4m3 rows [gate; up] under s13 [E, 2I/128, H/128] FP32, w2 [E, H, I] under
    s2 [E, H/128, I/128] -> the routed partial [M, H] BF16."""
    m, h = x.shape
    e, two_i, h2 = w13.shape
    i = two_i // 2
    k = ids.shape[1]
    if (not 1 <= m <= MAX_ROWS or h2 != h or tuple(w2.shape) != (e, h, i) or h % BLOCK or i % BLOCK
            or tuple(s13.shape) != (e, two_i // BLOCK, h // BLOCK) or tuple(s2.shape) != (e, h // BLOCK, i // BLOCK)
            or tuple(ids.shape) != (m, k) or tuple(weights.shape) != (m, k) or x.dtype != torch.bfloat16
            or w13.dtype != torch.float8_e4m3fn or w2.dtype != torch.float8_e4m3fn):
        raise ValueError(f"moe_fp8_rows: x {tuple(x.shape)}, ids {tuple(ids.shape)}, w13 {tuple(w13.shape)}, "
                         f"w2 {tuple(w2.shape)}")
    x, ids = x.contiguous(), ids.to(torch.int32).contiguous()
    weights = weights.to(torch.float32).contiguous()
    act = torch.empty(m * k, i, dtype=torch.bfloat16, device=x.device)
    out = torch.empty(m, h, dtype=torch.bfloat16, device=x.device)
    block_i, block_h = 64, 64
    _gate_up[(m * k, triton.cdiv(i, block_i))](x, ids, weights, w13, s13, act, e, TOPK=k, H=h, I=i, BLOCK_I=block_i,
                                               num_warps=4)
    _down[(m, triton.cdiv(h, block_h))](act, ids, weights, w2, s2, out, e, TOPK=k, H=h, I=i, BLOCK_H=block_h,
                                        num_warps=4)
    return out


def reference(x, ids, weights, w13, s13, w2, s2) -> torch.Tensor:
    """The same FFN in torch: each live route's expert dequantised to FP32, gate and up rounded to BF16, silu(gate) *
    up rounded to BF16, the down product times the route's weight summed in FP32, rounded once."""
    m, h = x.shape
    e, two_i, _ = w13.shape
    i = two_i // 2
    out = torch.zeros(m, h, dtype=torch.float32, device=x.device)

    def dequant(w, s):
        return w.float() * s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)

    for row in range(m):
        for r in range(ids.shape[1]):
            ex, gain = int(ids[row, r]), float(weights[row, r])
            if gain == 0 or not 0 <= ex < e:
                continue
            w = dequant(w13[ex], s13[ex])
            xf = x[row].float()
            gate = (w[:i] @ xf).bfloat16().float()
            up = (w[i:] @ xf).bfloat16().float()
            act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            out[row] += gain * (dequant(w2[ex], s2[ex]) @ act)
    return out.bfloat16()


def qualify(device, *, experts: int = 4, hidden: int = 256, inter: int = 128, topk: int = 3, rows=(1, 4)) -> dict:
    """D3 before a boot serves: the kernel held to `reference` on random FP8 experts -> {rows: largest error over the
    largest magnitude}; raises past 2^-6 (two BF16 steps). Routes to skip (weight 0, an id past the experts) included."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    w13 = (torch.randn(experts, 2 * inter, hidden, generator=gen) * 20).clamp(-448, 448).to(torch.float8_e4m3fn)
    w2 = (torch.randn(experts, hidden, inter, generator=gen) * 20).clamp(-448, 448).to(torch.float8_e4m3fn)
    s13 = torch.rand(experts, 2 * inter // BLOCK, hidden // BLOCK, generator=gen) * 1e-3
    s2 = torch.rand(experts, hidden // BLOCK, inter // BLOCK, generator=gen) * 1e-3
    tensors = [t.to(device) for t in (w13, s13, w2, s2)]
    out = {}
    for m in rows:
        x = torch.randn(m, hidden, generator=gen).bfloat16().to(device)
        ids = torch.randint(0, experts + 1, (m, topk), generator=gen, dtype=torch.int32).to(device)   # E = skip
        weights = torch.rand(m, topk, generator=gen).to(device)
        weights[:, -1] = 0                                                                           # a foreign route
        got, want = moe(x, ids, weights, *tensors).float(), reference(x, ids, weights, *tensors).float()
        err = float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))
        if not err <= 2.0 ** -6:
            raise RuntimeError(f"moe_fp8_rows at {m} rows: error {err:.2e} of the largest magnitude")
        out[m] = round(err, 6)
    return out


__all__ = ["MAX_ROWS", "moe", "reference", "qualify"]
