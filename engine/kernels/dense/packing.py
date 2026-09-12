# SPDX-License-Identifier: Apache-2.0
"""Native W4 pack arithmetic, extracted from the fleet-qualified MK packer."""
import torch

_E2M1_GRID = (0., .5, 1., 1.5, 2., 3., 4., 6.)
_E2M1_MIDS = (.25, .75, 1.25, 1.75, 2.5, 3.5, 5.)


def _w4_search(g, e0, mids, grid):
    """Best (code, d, scale) per 16-group of g [..., 16] fp32 (already
    shifted): the 2-octave x 8-mantissa e4m3-scale search of the RTN packer
    (it reaches the 72-candidate optimum exactly on the real weights), the
    error taken on what the kernel's expanded byte holds. e0 [..., 1] is the
    covering exponent ceil(log2(amax/6)) of the group, already shifted."""
    import torch

    sign = torch.sign(g)
    best = None
    for j in (0.0, 1.0):
        e = (e0 - j).clamp(-5, 5)
        for kk in range(8):
            sc = (1.0 + kk / 8.0) * torch.exp2(e)
            code = torch.bucketize((g / sc).abs(), mids)
            deq = (grid[code] * sign * sc).to(torch.float8_e4m3fn).float()
            err = (deq - g).pow(2).sum(-1, keepdim=True)
            d = (e * 8.0 + kk).to(torch.int8)
            if best is None:
                best = [err, code, d, sc]
            else:
                take = err < best[0]
                best = [torch.where(take, err, best[0]),
                        torch.where(take, code, best[1]),
                        torch.where(take, d, best[2]),
                        torch.where(take, sc, best[3])]
    return best[1], best[2], best[3]

def _w4_row_shift(weight, n_pad: int, kg: int, per_row: bool):
    """(need [n_pad, kg] covering exponents, shift [n_pad] int per row,
    clamped fraction). Per-tensor: one median shift on every row (wgs);
    per-row (33차 lever 3): each row centred on its own median, so a row
    living 2^6 from the tensor median no longer clamps its groups at the
    e4m3 exponent floor/ceiling -- the 1.5% of clamped groups on the
    production [6416, 4096] in_proj were exactly those rows."""
    import torch

    n, k = weight.shape
    need = torch.empty(n_pad, kg, dtype=torch.float32, device=weight.device)
    CH = max(128, (((1 << 20) // k) // 128) * 128)
    for r0 in range(0, n, CH):
        r1 = min(r0 + CH, n)
        a = weight[r0:r1].float().view(r1 - r0, kg, 16).abs().amax(-1)
        need[r0:r1] = torch.ceil(torch.log2((a / 6.0).clamp(min=1e-30)))
    need[n:] = 0.0
    live = need[:n]
    if per_row:
        shift = torch.zeros(n_pad, dtype=torch.float32, device=weight.device)
        shift[:n] = -torch.median(live, dim=1).values
    else:
        s = float(-torch.median(live).item()) if n else 0.0
        shift = torch.full((n_pad,), s, dtype=torch.float32,
                           device=weight.device)
    e_sh = live + shift[:n, None]
    clamped = float(((e_sh < -5) | (e_sh > 5)).float().mean()) if n else 0.0
    return need, shift, clamped

def _w4_rtn_codes(weight, shift, need, mids, grid):
    """Round-to-nearest packer (the 24차..32차 path): per chunk of rows,
    the e4m3-scale search per 16-group on the shifted weights."""
    import torch

    n, k = weight.shape
    n_pad = shift.shape[0]
    kg = k // 16
    dev = weight.device
    q_out = torch.zeros(n_pad, kg, 16, dtype=torch.uint8, device=dev)
    d_out = torch.zeros(n_pad, kg, dtype=torch.int8, device=dev)
    # ~1M elements per row chunk: the search makes a dozen temporaries per
    # candidate per chunk, and the GB10 allocator answered a whole-tensor
    # search by mapping new pages (14.5 GiB reserved on one [6416, 4096]
    # pack, 26차) -- a quarter chunk is a quarter of every temporary.
    CH = max(128, (((1 << 20) // k) // 128) * 128)
    for r0 in range(0, n_pad, CH):
        r1 = min(r0 + CH, n_pad)
        g = torch.zeros(r1 - r0, kg, 16, dtype=torch.float32, device=dev)
        if r0 < n:
            src = weight[r0:min(r1, n)].float()
            g[:src.shape[0]] = src.view(src.shape[0], kg, 16)
        g *= torch.exp2(shift[r0:r1])[:, None, None]
        sign = torch.signbit(g).to(torch.uint8) << 3
        e0 = need[r0:r1].unsqueeze(-1) + shift[r0:r1, None, None]
        code, d, _sc = _w4_search(g, e0, mids, grid)
        q_out[r0:r1] = code.to(torch.uint8) | sign
        d_out[r0:r1] = d.squeeze(-1)
        del g, sign, e0, code, d
    return q_out.view(n_pad, k), d_out

def mk_w4_dequant_rowmajor(wq4_rm, ws4_rm, wgs=1.0, rgs=None):
    """fp32 [n_pad, k] from the ROW-MAJOR (pre-tile) nibbles [n_pad, k/2]
    and scale bytes [n_pad, k/16]: the kernel's expanded e4m3 byte times
    the shift's undo (wgs, or rgs per row)."""
    import torch

    n_pad, k2 = wq4_rm.shape
    k = k2 * 2
    lo = wq4_rm & 0xF
    hi = wq4_rm >> 4
    q = torch.stack([lo, hi], dim=-1).reshape(n_pad, k)
    grid = torch.tensor(_E2M1_GRID, device=wq4_rm.device)
    mag = grid[(q & 7).long()]
    sign = torch.where((q & 8) != 0, -1.0, 1.0)
    d = ws4_rm.to(torch.int32)
    scale = (1.0 + (d & 7).float() / 8.0) * torch.exp2((d >> 3).float())
    w = mag * sign * scale.repeat_interleave(16, dim=1)
    w = w.to(torch.float8_e4m3fn).float() * wgs
    if rgs is not None:
        w = w * rgs.float()[:, None]
    return w

def mk_w4_dequant(wq4, ws4, n_rows, gscale=1.0, rgs=None):
    """fp32 [n_rows, k] the kernel's expansion reads: nibble -> e2m1 grid
    value x the group's e4m3 scale, rounded into e4m3 the way the kernel's
    table byte is, times the shift's undo (gscale, or rgs per row). Zero
    stays zero; every other code round-trips the grid bit-exactly, which is
    what the exact gate and the by-design gate both need."""
    import torch

    tn, tk, _, _ = wq4.shape
    n_pad, k = tn * 128, tk * 128
    # tile-major [n/128][k/128][128][64] -> row-major [n_pad, k/2]
    wq4_rm = wq4.permute(0, 2, 1, 3).reshape(n_pad, k // 2)
    ws4_rm = ws4.permute(0, 2, 1, 3).reshape(n_pad, k // 16)
    w = mk_w4_dequant_rowmajor(wq4_rm, ws4_rm, gscale, rgs)
    return w[:n_rows]

def _mk_quant_x_ref(x):
    """fp32 [m, k]: x after the kernel's activation quant (per row, per
    128-k group: the EXACT scale amax/448 (33차 lever 1; it was the pow2
    2^frexp_exp(amax/448) before, which wasted up to one bit of e4m3's
    three), e4m3 round-to-nearest at v * (1/scale), rescale). Pure twin of
    the prologue: the division, the reciprocal (__frcp_rn) and the product
    are the same IEEE fp32 operations in both, so with mk_w4_dequant it
    makes a torch fp32 matmul the kernel's exact reference (no fp8 MK arm
    exists to diff against any more)."""
    import torch

    m, k = x.shape
    g = x.float().view(m, k // 128, 128)
    amax = g.abs().amax(-1, keepdim=True)
    # amax * fp32(1/448): the kernel's form, and torch's own for a scalar
    # divisor; the floor is the kernel's fmaxf
    scale = (amax * (1.0 / 448.0)).clamp(min=1e-30)
    rsc = 1.0 / scale
    # the kernel converts with SATFINITE: a product one ulp over 448 (the
    # row's own amax times a rounded reciprocal) saturates instead of NaN
    q = (g * rsc).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return q.view(m, k)

import logging
logger = logging.getLogger(__name__)


def _w4_quant_cols(w, sc, mids, grid):
    """Quantize the columns w [R, c] with the fixed per-row scale sc [R, 1]:
    (code uint8 [R, c] without the sign bit, deq fp32 [R, c] = the kernel's
    expanded byte)."""
    import torch

    code = torch.bucketize((w / sc).abs(), mids)
    deq = (grid[code] * torch.sign(w) * sc).to(torch.float8_e4m3fn).float()
    return code, deq


def _w4_static_scales(weight, shift, need, mids, grid):
    """The RTN packer's per-16-group e4m3 scales of the shifted weights, for every group at once: (d int8 [n_pad,
    kg], sc fp32 [n_pad, kg]). What an act-ordered GPTQ quantises against -- its columns come in Hessian order, so a
    group's scale cannot be re-derived when the group "starts" (it never does); the groups stay the kernel's."""
    import torch

    n, k = weight.shape
    n_pad = shift.shape[0]
    kg = k // 16
    dev = weight.device
    d_out = torch.zeros(n_pad, kg, dtype=torch.int8, device=dev)
    sc_out = torch.ones(n_pad, kg, dtype=torch.float32, device=dev)
    CH = max(128, (((1 << 20) // k) // 128) * 128)
    for r0 in range(0, n_pad, CH):
        r1 = min(r0 + CH, n_pad)
        g = torch.zeros(r1 - r0, kg, 16, dtype=torch.float32, device=dev)
        if r0 < n:
            src = weight[r0:min(r1, n)].float()
            g[:src.shape[0]] = src.view(src.shape[0], kg, 16)
        g *= torch.exp2(shift[r0:r1])[:, None, None]
        e0 = need[r0:r1].unsqueeze(-1) + shift[r0:r1, None, None]
        _code, d, sc = _w4_search(g, e0, mids, grid)
        d_out[r0:r1] = d.squeeze(-1)
        sc_out[r0:r1] = sc.squeeze(-1)
        del g, e0, _code, d, sc
    return d_out, sc_out


def _gptq_inverse_factor(H, k, percdamp, factor_device, perm=None):
    """U = chol(H_damped^-1, upper) in fp32: symmetrised, columns permuted by `perm` when given, dead columns made
    unit, damped up a ladder until it factors. Two K x K fp64 matrices alive at most (the source is re-read on a
    retry instead of kept), on `factor_device` -- the CPU for a wide K: the GB10's fp64 is slow and a 20480^2 pair
    is 6.8 GiB of unified memory either way. Returns (U on that device, dead)."""
    import torch

    dev = H.device if factor_device is None else torch.device(factor_device)
    src = H.to(dev, torch.float32)                          # the symmetrised, permuted source in fp32: 1/2 the fp64 bytes
    if perm is not None:
        perm = perm.to(dev)
        src = src.index_select(0, perm)
        src = src.index_select(1, perm)
    src = 0.5 * (src + src.T)
    dead = torch.diagonal(src) <= 0
    src[dead, dead] = 1.0
    mean_diag = float(torch.mean(torch.diagonal(src)))
    diag = torch.arange(k, device=dev)
    U = None
    for damp_f in (percdamp, 10 * percdamp, 100 * percdamp, 1.0):
        Hd = src.to(torch.float64)
        Hd[diag, diag] += damp_f * mean_diag
        try:
            L = torch.linalg.cholesky(Hd)
            del Hd
            Hinv = torch.cholesky_inverse(L)
            del L
            U = torch.linalg.cholesky(Hinv, upper=True)
            del Hinv
            U = U.to(torch.float32)
            if damp_f != percdamp:
                logger.warning("[megakernel] w4 pack GPTQ: Hessian held at damping %.0f%% "
                               "(not at %.0f%%)", 100 * damp_f, 100 * percdamp)
            break
        except Exception:
            U = None
    del src
    if U is None:
        raise RuntimeError("Hessian not positive-definite at any damping")
    return U, dead


def gptq_factor(H, percdamp=0.01, act_order=True, factor_device=None):
    """The fp64 step both of a weight's GPTQ lanes share: the column order and the inverse factor of its input's
    Hessian. The W4 pack and the FP8 pack of one weight walk the same columns of the same H, so the factorisation --
    a fifth of a 4096-wide pack, a minute of a 20480-wide one -- is worth computing once and handing to both
    (kernels/dense/store._factor). Returns (perm or None, U, dead) as `_gptq_inverse_factor` does."""
    import torch

    perm = torch.argsort(torch.diagonal(H).to(torch.float64), descending=True) if act_order else None
    U, dead = _gptq_inverse_factor(H, H.shape[0], percdamp, factor_device, perm)
    return perm, U, dead


def _w4_gptq_codes(weight, shift, need, H, mids, grid, blocksize=128,
                   percdamp=0.01, act_order=False, factor_device=None, factor=None):
    """GPTQ (OBQ error feedback, Frantar et al. 2022) on the e2m1 x e4m3
    grid: columns are quantized in order; each column's rounding error is
    fed forward into the not-yet-quantized columns through the inverse
    Hessian of the layer's INPUT (H = sum x x^T over calibration tokens),
    so the rounding decisions minimise the OUTPUT error x @ (W - Q)^T, not
    the weight error. In the plain order the group scales are re-derived on
    the error-updated weights when the group starts (groups never cross a
    block: 16 | 128). With `act_order` the columns come in decreasing
    Hessian-diagonal order (the inputs that matter most are rounded first,
    their error absorbed by the rest) and the group scales are the RTN
    packer's, fixed before the walk (static groups: the kernel's 16-groups
    are untouched). Same bytes, same kernel: the accuracy is bought at pack
    time. `factor_device`: where the fp64 factorisation runs (see
    _gptq_inverse_factor).

    Returns (codes uint8 [n_pad, k] with the sign in bit 3, d int8 [n_pad,
    kg]) in the SHIFTED domain (weights x 2^shift_r), like the RTN path."""
    import torch

    n, k = weight.shape
    n_pad = shift.shape[0]
    kg = k // 16
    dev = weight.device
    W = torch.zeros(n_pad, k, dtype=torch.float32, device=dev)
    W[:n] = weight.float() * torch.exp2(shift[:n, None])
    codes = torch.zeros(n_pad, k, dtype=torch.uint8, device=dev)
    if act_order:
        d_out, sc_static = _w4_static_scales(weight, shift, need, mids, grid)
    else:
        d_out = torch.zeros(n_pad, kg, dtype=torch.int8, device=dev)
    if factor is None:
        factor = gptq_factor(H, percdamp, act_order, factor_device)
    perm, Hinv, dead = factor
    if (perm is not None) != bool(act_order):
        raise ValueError("a shared factor's column order must be the one this pack asked for (act_order)")
    Hinv = Hinv.to(dev)
    dead = dead.to(dev)
    if perm is not None:
        perm = perm.to(dev)
    if perm is not None:
        W = W[:, perm].contiguous()
    W[:, dead] = 0.0
    del H
    sc = None
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            col = i1 + i
            orig = int(perm[col]) if perm is not None else col
            if perm is not None:
                sc = sc_static[:, orig // 16:orig // 16 + 1]
            elif col % 16 == 0:
                g16 = W1[:, i:i + 16]
                amax = g16.abs().amax(-1, keepdim=True)
                e0 = torch.ceil(torch.log2((amax / 6.0).clamp(min=1e-30)))
                # a dead / all-zero group: any scale, keep the byte 0
                e0 = torch.where(amax > 0, e0, torch.zeros_like(e0))
                _c, d, sc = _w4_search(g16, e0, mids, grid)
                d_out[:, col // 16] = d.squeeze(-1)
            w = W1[:, i:i + 1]
            code, q = _w4_quant_cols(w, sc, mids, grid)
            sgn = torch.signbit(w).to(torch.uint8) << 3
            codes[:, orig] = (code.to(torch.uint8) | sgn).squeeze(-1)
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    del W, Hinv
    return codes, d_out


FP8_BLOCK = 128


def fp8_block_scales(weight):
    """DeepGEMM's per-128x128-block UE8M0 scale of `per_block_cast_to_fp8(use_ue8m0=True)`: 2^ceil(log2(amax/448)),
    fp32 [n_pad/128, k/128] for the 128-row-padded weight (the GPU judge holds it byte-equal to the library's)."""
    import torch

    n, k = weight.shape
    n_pad = (n + FP8_BLOCK - 1) // FP8_BLOCK * FP8_BLOCK
    w = torch.zeros(n_pad, k, dtype=torch.float32, device=weight.device)
    w[:n] = weight.float()
    b = w.view(n_pad // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    amax = b.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    return torch.exp2(torch.ceil(torch.log2(amax / 448.0)))


def fp8_rtn(weight):
    """(q e4m3 [n_pad, k], scale fp32 [n_pad/128, k/128]): the FP8 lane's round-to-nearest weights."""
    import torch

    n, k = weight.shape
    scale = fp8_block_scales(weight)
    n_pad = scale.shape[0] * FP8_BLOCK
    w = torch.zeros(n_pad, k, dtype=torch.float32, device=weight.device)
    w[:n] = weight.float()
    per = scale.repeat_interleave(FP8_BLOCK, dim=0).repeat_interleave(FP8_BLOCK, dim=1)
    return (w / per).to(torch.float8_e4m3fn), scale


def fp8_gptq(weight, H, blocksize=128, percdamp=0.01, act_order=True, factor_device=None, factor=None):
    """GPTQ on the FP8 lane's own grid: the served 128x128 UE8M0 block scales stay (static), each column is rounded to
    e4m3 under its block's scale in decreasing Hessian-diagonal order, the error fed forward through the inverse
    Hessian. The fp8 rounding is what the lane pays for every prefill row; compensating it costs the same pack time
    as the W4 packs (1..3 s a weight, cached by the store). Returns (q e4m3 [n_pad, k], scale fp32 [n_pad/128, k/128])."""
    import torch

    n, k = weight.shape
    dev = weight.device
    scale = fp8_block_scales(weight)
    n_pad = scale.shape[0] * FP8_BLOCK
    rows_scale = scale.repeat_interleave(FP8_BLOCK, dim=0)                 # [n_pad, k/128]
    if factor is None:
        factor = gptq_factor(H, percdamp, act_order, factor_device)
    perm, U, dead = factor
    if (perm is not None) != bool(act_order):
        raise ValueError("a shared factor's column order must be the one this pack asked for (act_order)")
    U, dead = U.to(dev), dead.to(dev)
    if perm is not None:
        perm = perm.to(dev)
    W = torch.zeros(n_pad, k, dtype=torch.float32, device=dev)
    W[:n] = weight.float()
    if perm is not None:
        W = W[:, perm].contiguous()
    W[:, dead] = 0.0
    Q = torch.zeros(n_pad, k, dtype=torch.float8_e4m3fn, device=dev)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            orig = int(perm[col]) if perm is not None else col
            s = rows_scale[:, orig // FP8_BLOCK:orig // FP8_BLOCK + 1]
            w = W1[:, i:i + 1]
            q8 = (w / s).to(torch.float8_e4m3fn)
            Q[:, orig:orig + 1] = q8
            err = (w - q8.float() * s) / U1[i, i]
            W1[:, i:] -= err @ U1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        W[:, i2:] -= Err1 @ U[i1:i2, i2:]
    del W, U
    return Q, scale

