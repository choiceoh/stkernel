"""Persistent GB10 MLP workers: producer -> dependent partial -> output reduction.

All producer tickets are handed out before a worker can wait on a producer.
Likewise, all partial tickets are owned before reduction waiters appear.
Thus outstanding dependencies belong to resident, progressing workers; no
unlaunched CTA or reserved scheduler is needed to release a waiting CTA.
Acq/rel atomics publish data, each output has one writer, and bounded polling
returns an explicit error rather than silently using incomplete output.
"""
import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# Triton 3.7.1's mmav2SupportsFp8Operands admits 120 but omits 121.
# Use consumer Blackwell lowering for W4A8; the backend still emits the
# actual device's sm_121a PTX/cubin. Do not relax accumulation precision.
W4A8_MMA_ARCH = "sm120"


@triton.jit
def _quantize(x, global_scale, BM: tl.constexpr, B: tl.constexpr):
    # ModelOpt's input scale is a dequantization multiplier. Preserve the
    # divide, group-16 E4M3 rounding, and E2M1 round-to-nearest-even steps.
    z = tl.div_rn(x.to(tl.float32), global_scale).reshape(BM, B//16, 16)
    scale = tl.minimum(tl.max(tl.abs(z), 2) / 6., 448.).to(tl.float8e4nv)
    sf = tl.maximum(scale.to(tl.float32), 1.1754943508222875e-38)
    value = tl.div_rn(z, sf[:, :, None]).reshape(BM, B//2, 2)
    even, odd = tl.split(value)
    packed = tl.inline_asm_elementwise("{ .reg .b8 q; cvt.rn.satfinite.e2m1x2.f32 q, $2, $1; mov.b32 $0, {q, 0, 0, 0}; }", "=r,f,f", [even, odd],
                                       dtype=tl.int32, is_pure=True, pack=1).to(tl.uint8)
    return packed, scale


@triton.jit
def _weight(W, rows, kb, ROWS: tl.constexpr, K: tl.constexpr, TILE_BYTES: tl.constexpr):
    if TILE_BYTES:
        offset = (kb // TILE_BYTES * ROWS + rows) * TILE_BYTES + kb % TILE_BYTES
    else:
        offset = rows * (K//2) + kb
    return tl.load(W + offset, (rows < ROWS) & (kb < K//2), other=0)


@triton.jit
def _scale(S, rows, groups, ROWS: tl.constexpr, K: tl.constexpr, SF6: tl.constexpr, FC2: tl.constexpr):
    valid = (rows < ROWS) & (groups < K//16)
    if SF6:
        if FC2:
            stage = rows // 256 * (K//128) + groups // 8
            byte = (groups % 8 // 4)*1024 + (rows % 256 // 128)*512
        else:
            stage = rows // 128 * (K//256) + groups // 16
            byte = (groups % 16 // 4)*512
        byte += (rows % 32)*16 + (rows % 128 // 32)*4 + groups % 4
        base = stage * 1552
        low = (tl.load(S+base+byte//2, valid, other=0).to(tl.int32) >> (byte % 2 * 4)) & 15
        high = (tl.load(S+base+1024+byte//4, valid, other=0).to(tl.int32) >> (byte % 4 * 2)) & 3
        code = tl.load(S+base+1536, valid, other=0).to(tl.int32) + low + (high << 4)
        return code.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
    else:
        sp: tl.constexpr = triton.cdiv(K//16, 4)*4
        offset = (rows//128*(sp//4)+groups//4)*512 + (rows%32)*16 + (rows%128//32)*4 + groups%4
        return tl.load(S+offset, valid, other=0).to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _pause():
    tl.inline_asm_elementwise("nanosleep.u32 64; mov.u32 $0, 0;", "=r", [],
                             dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _w4a8_scaled_input(x):
    # Exactly the current DenseLinear activation group: 128 values, E4M3,
    # amax * fp32(1/448), rounded reciprocal, SATFINITE conversion.
    value = x.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 1) * (1. / 448.), 1.e-30)
    inverse = tl.div_rn(1., scale)
    return tl.minimum(tl.maximum(value * inverse[:, None], -448.), 448.), scale


@triton.jit
def _w4a8_quantize(x):
    value, scale = _w4a8_scaled_input(x)
    return value.to(tl.float8e4nv), scale


@triton.jit
def _w4a8_weight(W, S, rows, columns, ROWS: tl.constexpr, K: tl.constexpr):
    # Borrow W4Pack's [N/128,K/128,128,64/8] planes directly. Its signed
    # scale byte encodes exponent*8 + mantissa; it is NOT an NVFP4 SF byte.
    valid = (rows < ROWS) & (columns < K)
    byte = ((rows//128*(K//128)+columns//128)*128+rows%128)*64+columns%128//2
    code = (tl.load(W+byte, valid, other=0).to(tl.int32) >> (columns%2*4)) & 15
    sf = ((rows//128*(K//128)+columns//128)*128+rows%128)*8+columns%128//16
    d = tl.load(S+sf, valid, other=0).to(tl.int32)
    mag = code & 7
    # Same two byte tables as MK_E2M1_LUT64 / MK_E2M1_LUT64_B in the
    # deployed CUDA lane. Integer expansion preserves tie-to-even exactly
    # and avoids exponent arithmetic and a floating conversion per weight.
    table = tl.where(((d & 7) >= 1) & ((d & 7) <= 5),
                     0x4D4845403D383000, 0x4C4844403C383000).to(tl.uint64)
    value = ((table >> (mag*8)) & 255).to(tl.int32)
    value = tl.where(mag == 0, 0, value+d)
    value = value | ((code & 8) << 4)
    return value.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _workers(X, WG, WD, U, PART, OUT, CONTROL, S1, S2, US, A1, A2, ALPHA1, ALPHA2,
             M: tl.constexpr, H: tl.constexpr, I: tl.constexpr, P: tl.constexpr, O: tl.constexpr,
             BM: tl.constexpr, B: tl.constexpr, LIMIT: tl.constexpr, SPINS: tl.constexpr,
             NV4: tl.constexpr, TILE13: tl.constexpr, TILE2: tl.constexpr, SF6: tl.constexpr,
             W4A8: tl.constexpr = False):
    rows = tl.arange(0, BM)
    tile = tl.arange(0, B)
    job = tl.atomic_add(CONTROL, 1, sem="relaxed")
    while job < P:
        columns = job*B + tile
        g = tl.full((BM, B), 0., tl.float32)
        u = tl.full((BM, B), 0., tl.float32)
        for block in range(tl.cdiv(H, B)):
            kk = block*B + tile
            x = tl.load(X + rows[:, None]*H + kk[None, :], (rows[:, None] < M) & (kk[None, :] < H), other=0)
            if NV4:
                xq, xs = _quantize(x, tl.load(A1), BM, B)
                kb = block*(B//2) + tl.arange(0, B//2)
                groups = block*(B//16) + tl.arange(0, B//16)
                # NVFP4 checkpoint order is [UP | GATE], unlike BF16 gate_up.
                up = _weight(WG, columns[None, :], kb[:, None], 2*I, H, TILE13)
                gate = _weight(WG, I+columns[None, :], kb[:, None], 2*I, H, TILE13)
                su = _scale(S1, columns[:, None], groups[None, :], 2*I, H, SF6, False)
                sg = _scale(S1, I+columns[:, None], groups[None, :], 2*I, H, SF6, False)
                g = tl.dot_scaled(xq, xs, "e2m1", gate, sg, "e2m1", g)
                u = tl.dot_scaled(xq, xs, "e2m1", up, su, "e2m1", u)
            elif W4A8:
                xq, xs = _w4a8_quantize(x)
                gate = _w4a8_weight(WG, S1, columns[None, :], kk[:, None], 2*I, H)
                up = _w4a8_weight(WG, S1, I+columns[None, :], kk[:, None], 2*I, H)
                g = tl.fma(tl.dot(xq, gate), xs[:, None], g)
                u = tl.fma(tl.dot(xq, up), xs[:, None], u)
            else:
                gate = tl.load(WG + columns[None, :]*H + kk[:, None], (columns[None, :] < I) & (kk[:, None] < H), other=0)
                up = tl.load(WG + (I+columns[None, :])*H + kk[:, None], (columns[None, :] < I) & (kk[:, None] < H), other=0)
                g = tl.dot(x, gate, g)
                u = tl.dot(x, up, u)
        if NV4:
            # Keep FC1 FP32 through SwiGLU, matching the current b12x lane.
            alpha = tl.load(ALPHA1)
            g, u = g * alpha, u * alpha
        elif W4A8:
            gs = tl.load(ALPHA1+columns, columns < I, other=0)
            us = tl.load(ALPHA1+I+columns, columns < I, other=0)
            g = (g*gs[None, :]).to(tl.bfloat16).to(tl.float32)
            u = (u*us[None, :]).to(tl.bfloat16).to(tl.float32)
        else:
            g, u = g.to(tl.bfloat16).to(tl.float32), u.to(tl.bfloat16).to(tl.float32)
        g = tl.minimum(g, LIMIT)
        u = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        if W4A8:
            value = (g * tl.div_rn(1., 1. + libdevice.exp(-g))) * u
        else:
            value = g / (1 + tl.exp(-g)) * u
        if NV4:
            uq, us = _quantize(value.to(tl.bfloat16), tl.load(A2), BM, B)
            kb = job*(B//2) + tl.arange(0, B//2)
            groups = job*(B//16) + tl.arange(0, B//16)
            tl.store(U + rows[:, None]*(I//2) + kb[None, :], uq, (rows[:, None] < M) & (kb[None, :] < I//2))
            tl.store(US + rows[:, None]*(I//16) + groups[None, :], us.to(tl.uint8, bitcast=True),
                     (rows[:, None] < M) & (groups[None, :] < I//16))
        elif W4A8:
            uq, us = _w4a8_quantize(value.to(tl.bfloat16))
            tl.store(U + rows[:, None]*I + columns[None, :], uq.to(tl.uint8, bitcast=True),
                     (rows[:, None] < M) & (columns[None, :] < I))
            tl.store(US + rows*P + job, us, rows < M)
        else:
            tl.store(U + rows[:, None]*I + columns[None, :], value, (rows[:, None] < M) & (columns[None, :] < I))
        tl.debug_barrier()
        tl.atomic_xchg(CONTROL + 4 + job, 1, sem="release")
        job = tl.atomic_add(CONTROL, 1, sem="relaxed")
    job = tl.atomic_add(CONTROL + 1, 1, sem="relaxed")
    failed = False
    while job < P*O and not failed:
        producer, column = job // O, job % O
        ready = tl.atomic_add(CONTROL + 4 + producer, 0, sem="acquire")
        spins = 0
        while ready == 0 and spins < SPINS:
            _pause()
            ready = tl.atomic_add(CONTROL + 4 + producer, 0, sem="acquire")
            spins += 1
        if ready == 0:
            tl.atomic_max(CONTROL + 3, 1, sem="relaxed")
            failed = True
        else:
            kk, cols = producer*B + tile, column*B + tile
            if NV4:
                kb = producer*(B//2) + tl.arange(0, B//2)
                groups = producer*(B//16) + tl.arange(0, B//16)
                act = tl.load(U + rows[:, None]*(I//2) + kb[None, :], (rows[:, None] < M) & (kb[None, :] < I//2), other=0)
                asc = tl.load(US + rows[:, None]*(I//16) + groups[None, :],
                              (rows[:, None] < M) & (groups[None, :] < I//16), other=0).to(tl.float8e4nv, bitcast=True)
                weight = _weight(WD, cols[None, :], kb[:, None], H, I, TILE2)
                wsc = _scale(S2, cols[:, None], groups[None, :], H, I, SF6, True)
                part = tl.dot_scaled(act, asc, "e2m1", weight, wsc, "e2m1") * tl.load(ALPHA2)
            elif W4A8:
                act = tl.load(U + rows[:, None]*I + kk[None, :],
                              (rows[:, None] < M) & (kk[None, :] < I), other=0).to(tl.float8e4nv, bitcast=True)
                asc = tl.load(US + rows*P + producer, rows < M, other=0)
                weight = _w4a8_weight(WD, S2, cols[None, :], kk[:, None], H, I)
                part = tl.dot(act, weight) * asc[:, None]
            else:
                act = tl.load(U + rows[:, None]*I + kk[None, :], (rows[:, None] < M) & (kk[None, :] < I), other=0)
                weight = tl.load(WD + cols[None, :]*I + kk[:, None], (cols[None, :] < H) & (kk[:, None] < I), other=0)
                part = tl.dot(act, weight)
            tl.store(PART + (producer*M+rows[:, None])*H + cols[None, :], part,
                     (rows[:, None] < M) & (cols[None, :] < H))
            tl.debug_barrier()
            tl.atomic_add(CONTROL + 4 + P + column, 1, sem="acq_rel")
            job = tl.atomic_add(CONTROL + 1, 1, sem="relaxed")
    job = tl.atomic_add(CONTROL + 2, 1, sem="relaxed")
    while job < O and not failed:
        ready = tl.atomic_add(CONTROL + 4 + P + job, 0, sem="acquire")
        spins = 0
        while ready < P and spins < SPINS:
            _pause()
            ready = tl.atomic_add(CONTROL + 4 + P + job, 0, sem="acquire")
            spins += 1
        if ready != P:
            tl.atomic_max(CONTROL + 3, 2, sem="relaxed")
            failed = True
        else:
            cols = job*B + tile
            acc = tl.full((BM, B), 0., tl.float32)
            for producer in range(P):
                acc += tl.load(PART + (producer*M+rows[:, None])*H + cols[None, :],
                               (rows[:, None] < M) & (cols[None, :] < H), other=0)
            if W4A8:
                acc *= tl.load(ALPHA2+cols, cols < H, other=0)[None, :]
            tl.store(OUT + rows[:, None]*H + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < H))
            job = tl.atomic_add(CONTROL + 2, 1, sem="relaxed")


def execute(plan, x, gate_up, down, limit, *, max_spins=1_000_000):
    plan.validate_tensors(x, gate_up, down)
    if (not x.is_cuda or torch.cuda.get_device_capability(x.device) != (12, 1)
            or not math.isfinite(limit) or limit <= 0 or type(max_spins) is not int or max_spins <= 0):
        raise ValueError("persistent dataflow requires SM121 and finite positive execution bounds")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("experimental dataflow reports completion on the host; capture is not supported")
    activation = torch.empty((plan.rows, plan.intermediate), dtype=x.dtype, device=x.device)
    partial = torch.empty((plan.producers, plan.rows, plan.hidden), dtype=torch.float32, device=x.device)
    control = torch.zeros(4+plan.producers+plan.outputs, dtype=torch.int32, device=x.device)
    output = torch.empty_like(x)
    _workers[(plan.workers,)](x, gate_up, down, activation, partial, output, control, None, None, None, None, None, None, None,
        plan.rows, plan.hidden, plan.intermediate, plan.producers, plan.outputs,
        max(16, triton.next_power_of_2(plan.rows)), plan.tile, limit, max_spins, False, 0, 0, False, num_warps=4, num_stages=1)
    # Reading this also keeps all scratch alive until every worker has finished.
    error = int(control[3].item())
    return output, error


def execute_nvfp4(plan, x, weights, limit, *, max_spins=1_000_000):
    if (x.shape != (plan.rows, plan.hidden) or not x.is_cuda or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.device != weights.w13.device or torch.cuda.get_device_capability(x.device) != (12, 1)
            or not math.isfinite(limit) or limit <= 0 or type(max_spins) is not int or max_spins <= 0):
        raise ValueError("NVFP4 dataflow requires matching SM121 BF16 inputs and positive execution bounds")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("experimental dataflow completion requires an eager owner")
    activation = torch.empty((plan.rows, plan.intermediate//2), dtype=torch.uint8, device=x.device)
    scales = torch.empty((plan.rows, plan.intermediate//16), dtype=torch.uint8, device=x.device)
    partial = torch.empty((plan.producers, plan.rows, plan.hidden), dtype=torch.float32, device=x.device)
    control = torch.zeros(4+plan.producers+plan.outputs, dtype=torch.int32, device=x.device)
    output = torch.empty_like(x)
    s = weights.scales
    _workers[(plan.workers,)](x, weights.w13, weights.w2, activation, partial, output, control,
        weights.sf13, weights.sf2, scales, s.input13, s.input2, s.alpha13, s.alpha2,
        plan.rows, plan.hidden, plan.intermediate, plan.producers, plan.outputs,
        max(16, triton.next_power_of_2(plan.rows)), plan.tile, limit, max_spins,
        True, weights.tile13, weights.tile2, weights.sf6, num_warps=4, num_stages=1)
    return output, int(control[3].item())


def execute_w4a8(plan, x, weights, limit, *, max_spins=1_000_000):
    weights.validate_input(plan, x)
    if (not x.is_cuda or torch.cuda.get_device_capability(x.device) != (12, 1)
            or not math.isfinite(limit) or limit <= 0 or type(max_spins) is not int or max_spins <= 0):
        raise ValueError("W4A8 dataflow requires SM121 and finite positive execution bounds")
    if torch.cuda.is_current_stream_capturing():
        raise ValueError("experimental W4A8 dataflow completion requires an eager owner")
    activation = torch.empty((plan.rows, plan.intermediate), dtype=torch.uint8, device=x.device)
    scales = torch.empty((plan.rows, plan.producers), dtype=torch.float32, device=x.device)
    partial = torch.empty((plan.producers, plan.rows, plan.hidden), dtype=torch.float32, device=x.device)
    control = torch.zeros(4+plan.producers+plan.outputs, dtype=torch.int32, device=x.device)
    output = torch.empty_like(x)
    gate, down = weights.gate_up, weights.down
    _workers[(plan.workers,)](x, gate.data, down.data, activation, partial, output, control,
        gate.scale, down.scale, scales, None, None, gate.rowscale, down.rowscale,
        plan.rows, plan.hidden, plan.intermediate, plan.producers, plan.outputs,
        max(16, triton.next_power_of_2(plan.rows)), plan.tile, limit, max_spins,
        False, 0, 0, False, True, num_warps=4, num_stages=1, enable_fp_fusion=False, arch=W4A8_MMA_ARCH)
    return output, int(control[3].item())
