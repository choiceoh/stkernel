"""A few rows' routed experts on BF16 or block-scaled FP8 weights, BF16 activations (kernels): the MTP head's MoE at
a precision the target layers' NVFP4 does not set.

Qwen3.8's MTP head drafts every token a speculative step proposes, so its error costs acceptance three times a K=3 step
while its experts are a sliver of the step's bytes (a draft row reads two or three of a rank's experts). The target
layers' experts are NVFP4 (the NVIDIA export's, calibrated); the export kept the MTP head's in FP8, and the rank files
re-encoded those as NVFP4 again (quantised twice, activations to FP4 too). The operator's rule of 2026-09-19 -- NVFP4 by
default, precision where it costs little and moves acceptance -- puts the MTP head's experts at the checkpoint's original
BF16 (the side files mtp_side.py writes); FP8 (the export's) stays a choice. This serves either, from the weights and the
row's BF16 activation -- FP8 widened in registers under its 128 x 128 tile scales, FP32 sums -- in two launches:

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
             BLOCK_I: tl.constexpr, SCALED: tl.constexpr):
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
            if SCALED:
                sg = tl.load(sbase + (i // 128) * (H // 128) + kb, mask=live, other=0.0)
                su = tl.load(sbase + ((I + i) // 128) * (H // 128) + kb, mask=live, other=0.0)
                gate += tl.sum(g * x[None, :], 1) * sg
                up += tl.sum(u * x[None, :], 1) * su
            else:
                gate += tl.sum(g * x[None, :], 1)
                up += tl.sum(u * x[None, :], 1)
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        act = (gate * tl.sigmoid(gate)) * up
        tl.store(ACT + p * I + i, act.to(tl.bfloat16), mask=live)


@triton.jit
def _down(ACT, IDS, WTS, W2, S2, OUT, E, TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr, BLOCK_H: tl.constexpr,
          SCALED: tl.constexpr):
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
                if SCALED:
                    s = tl.load(sbase + (hs // 128) * (I // 128) + kb, mask=live, other=0.0)
                    y += tl.sum(d * a[None, :], 1) * s
                else:
                    y += tl.sum(d * a[None, :], 1)
            acc += w * y
    tl.store(OUT + row * H + hs, acc.to(tl.bfloat16), mask=live)


def moe(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, w13: torch.Tensor, s13, w2: torch.Tensor,
        s2) -> torch.Tensor:
    """x [M <= 16, H] BF16, ids [M, k] int32 (this rank's expert ids; a route to skip has weight 0 or an id >= E),
    weights [M, k] FP32; w13 [E, 2I, H] rows [gate; up], w2 [E, H, I] -- BF16 (s13, s2 None), or e4m3 under s13
    [E, 2I/128, H/128] and s2 [E, H/128, I/128] FP32 -> the routed partial [M, H] BF16."""
    m, h = x.shape
    e, two_i, h2 = w13.shape
    i = two_i // 2
    k = ids.shape[1]
    scaled = w13.dtype == torch.float8_e4m3fn
    if scaled:
        shapes_ok = (tuple(s13.shape) == (e, two_i // BLOCK, h // BLOCK) and tuple(s2.shape) == (e, h // BLOCK, i // BLOCK)
                     and w2.dtype == torch.float8_e4m3fn)
    else:
        shapes_ok = s13 is None and s2 is None and w13.dtype == w2.dtype == torch.bfloat16
    if (not 1 <= m <= MAX_ROWS or h2 != h or tuple(w2.shape) != (e, h, i) or h % BLOCK or i % BLOCK or not shapes_ok
            or tuple(ids.shape) != (m, k) or tuple(weights.shape) != (m, k) or x.dtype != torch.bfloat16):
        raise ValueError(f"moe_rows: x {tuple(x.shape)}, ids {tuple(ids.shape)}, w13 {tuple(w13.shape)} {w13.dtype}, "
                         f"w2 {tuple(w2.shape)}, scales {'given' if s13 is not None else 'none'}")
    x, ids = x.contiguous(), ids.to(torch.int32).contiguous()
    weights = weights.to(torch.float32).contiguous()
    act = torch.empty(m * k, i, dtype=torch.bfloat16, device=x.device)
    out = torch.empty(m, h, dtype=torch.bfloat16, device=x.device)
    block_i, block_h = 64, 64
    _gate_up[(m * k, triton.cdiv(i, block_i))](x, ids, weights, w13, s13 if scaled else w13, act, e, TOPK=k, H=h, I=i,
                                               BLOCK_I=block_i, SCALED=scaled, num_warps=4)
    _down[(m, triton.cdiv(h, block_h))](act, ids, weights, w2, s2 if scaled else w2, out, e, TOPK=k, H=h, I=i,
                                        BLOCK_H=block_h, SCALED=scaled, num_warps=4)
    return out


def reference(x, ids, weights, w13, s13, w2, s2) -> torch.Tensor:
    """The same FFN in torch: each live route's expert in FP32 (FP8 dequantised under its tile scales), gate and up
    rounded to BF16, silu(gate) * up rounded to BF16, the down product times the route's weight summed in FP32, rounded
    once."""
    m, h = x.shape
    e, two_i, _ = w13.shape
    i = two_i // 2
    out = torch.zeros(m, h, dtype=torch.float32, device=x.device)

    def widen(w, s):
        if s is None:
            return w.float()
        return w.float() * s.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)

    for row in range(m):
        for r in range(ids.shape[1]):
            ex, gain = int(ids[row, r]), float(weights[row, r])
            if gain == 0 or not 0 <= ex < e:
                continue
            w = widen(w13[ex], None if s13 is None else s13[ex])
            xf = x[row].float()
            gate = (w[:i] @ xf).bfloat16().float()
            up = (w[i:] @ xf).bfloat16().float()
            act = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            out[row] += gain * (widen(w2[ex], None if s2 is None else s2[ex]) @ act)
    return out.bfloat16()


def qualify(device, *, precision: str = "bf16", experts: int = 4, hidden: int = 256, inter: int = 128, topk: int = 3,
            rows=(1, 4), band: float = 2.0 ** -6) -> dict:
    """D3 before a boot serves: the kernel held to `reference` on random experts at `precision` ("bf16" or "fp8") ->
    {rows: largest error over the largest magnitude}; raises past `band` (two BF16 steps; Triton's interpreter, which
    truncates a BF16 cast where a GPU rounds it, needs twice that). Routes to skip (weight 0, an id past the experts)
    included."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    if precision == "fp8":
        w13 = (torch.randn(experts, 2 * inter, hidden, generator=gen) * 20).clamp(-448, 448).to(torch.float8_e4m3fn)
        w2 = (torch.randn(experts, hidden, inter, generator=gen) * 20).clamp(-448, 448).to(torch.float8_e4m3fn)
        s13 = (torch.rand(experts, 2 * inter // BLOCK, hidden // BLOCK, generator=gen) * 1e-3).to(device)
        s2 = (torch.rand(experts, hidden // BLOCK, inter // BLOCK, generator=gen) * 1e-3).to(device)
    elif precision == "bf16":
        w13 = (torch.randn(experts, 2 * inter, hidden, generator=gen) * 0.02).bfloat16()
        w2 = (torch.randn(experts, hidden, inter, generator=gen) * 0.02).bfloat16()
        s13 = s2 = None
    else:
        raise ValueError(f"moe_rows.qualify: precision {precision!r}")
    w13, w2 = w13.to(device), w2.to(device)
    out = {}
    for m in rows:
        x = torch.randn(m, hidden, generator=gen).bfloat16().to(device)
        ids = torch.randint(0, experts + 1, (m, topk), generator=gen, dtype=torch.int32).to(device)   # E = skip
        weights = torch.rand(m, topk, generator=gen).to(device)
        weights[:, -1] = 0                                                                           # a foreign route
        got, want = moe(x, ids, weights, w13, s13, w2, s2).float(), reference(x, ids, weights, w13, s13, w2, s2).float()
        err = float((got - want).abs().max() / want.abs().max().clamp_min(1e-30))
        if not err <= band:
            raise RuntimeError(f"moe_rows ({precision}) at {m} rows: error {err:.2e} of the largest magnitude")
        out[m] = round(err, 6)
    return out


__all__ = ["MAX_ROWS", "moe", "reference", "qualify"]
