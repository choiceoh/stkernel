"""The gated residual hyper-connection in five launches a site, three for a decode step's rows (Qwen3.8's residual form).

engine/modules/hyper_connection.gated_residual is the oracle: the hc streams laid end to end [N, hc*H] are RMS
normalised one by one (unit-offset weight), a low-rank mixer weights every channel of every stream --
sigmoid(up(silu(down(normed) / hc))) -- and the sublayer reads the streams' weighted mean [N, H]; the injection
2*sigmoid(inject(normed) / hc) [N, hc] is how strongly the sublayer's output is added back into each stream.

Composed from torch ops that is about a dozen launches a site, and the model has 97 sites (two a layer and the
closing mixer), over rows 10,240 wide. Here a site is:

    leave_norm   the previous sublayer's output added into the streams AND the streams normalised    one launch
    GEMM         down and inject in one BF16 matmul (the injection's 4 rows ride the mixer's 320)       one launch
    gates        silu(x / hc) on the mixer rows, 2*sigmoid(z / hc) on the injection rows              one launch
    GEMM         up                                                                                    one launch
    mix_mean     sigmoid(up) * normed, averaged over the streams                                       one launch

A decode step's rows (1..16, `DECODE_ROWS`) take `mix_rows` instead: the two products on the skinny GEMV
(engine/kernels/common/skinny_gemv -- one read of each weight tile for every row), each with the elementwise launch
after it folded into its store -- carry H2's two launches a site (the gates ride the down launch rather than the up
one), three launches a site:

    leave_norm   as above                                                                              one launch
    down_gates   down(+inject) for the rows, split over K; the last program of a column block sums     one launch
                 the split, rounds the product to BF16 where the GEMM's output did, and stores the gates
    up_mean      up for one block of hidden channels in each stream, sigmoid, times the streams, the   one launch
                 mean over them -- the up product never leaves the program

The arithmetic after each product is `_gates`' and `_mix_mean`'s, on the same BF16 product, so the site's outputs are
byte for byte the five-launch site's with the same products (probes/engine_qwen38_gemv, q38site-0919a). On a GB10 at
Qwen3.8's widths the mixer went from 66.5-70.5 us (its four launches on cuBLAS, plus an output copy the probe added)
to 60.3-61.8 us for 1-16 rows, 16 sites a graph over rotated weights: 13.2 MB of weights at about 218 GB/s.

`norm_streams` opens the first site (no output to add yet) and follows an injection feature that reads the
streams between two sites (Qwen3.8's PLE before layer 1, the config's one-indexed 2); `leave` adds an output without the norm for the same
case. The closing mixer is a site without an injection.

Rounding is the torch form's wherever the form rounds: the division by hc, silu, sigmoid, the gate product and
the residual product each round to the activations' dtype before the next operation reads them, and the norm
rounds once after its unit-offset weight, as rmsnorm_unit_offset does. Two reductions are the kernel's own and
can differ from torch's in the last bit: the norm's sum of squares and the mean over the streams (torch reduces
BF16 partial sums; this accumulates in fp32 and rounds once). The GEMMs are torch's matmul; the injection sharing
the mixer's matmul changes the matrix cuBLAS tiles, not the arithmetic of a row.

`qualify` holds the lane to the oracle on the device before it serves (D3): a boot calls it once and dies on a
mismatch rather than serving a residual stream that drifts. Its bounds are a few BF16 steps (a step is 2^-7 of a value)
because rounding order moves elements by a step; a wrong formula moves them by tenths.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels.common import skinny_gemv
from engine.kernels.common.skinny_gemv import rows_dot, split_span, split_sum

DECODE_ROWS = skinny_gemv.MAX_ROWS          # rows up to this take `mix_rows`
UP_TILE = (32, 64, 4, 3)                    # up_mean's BLOCK_D, BLOCK_K, warps, stages (the best of five, q38site-0919a)


@triton.jit
def _norm_streams(X, W, OUT, sX, sO, EPS, HID: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    x = tl.load(X + r * sX + off, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
    w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * sO + off, ((x * scale) * (1.0 + w)).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _leave_norm(H, OUT, INJ, W, NORMED, sH, sO, sI, sN, EPS, HID: tl.constexpr, BD: tl.constexpr,
                NORM: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    h = tl.load(H + r * sH + off, mask=m, other=0.0)
    o = tl.load(OUT + r * sO + d, mask=m, other=0.0).to(tl.float32)
    g = tl.load(INJ + r * sI + s).to(tl.float32)
    delta = (o * g).to(h.dtype)                                   # the product rounds, then the sum does
    new = (h.to(tl.float32) + delta.to(tl.float32)).to(h.dtype)
    tl.store(H + r * sH + off, new, mask=m)
    if NORM:
        x = new.to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
        w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
        tl.store(NORMED + r * sN + off, ((x * scale) * (1.0 + w)).to(NORMED.dtype.element_ty), mask=m)


@triton.jit
def _gates(DI, MIX, INJ, sD, sM, sI, HC_F, R: tl.constexpr, BR: tl.constexpr, HC: tl.constexpr,
           BH: tl.constexpr, WITH_INJECT: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, BR)
    m = i < R
    x = tl.load(DI + r * sD + i, mask=m, other=0.0)
    q = (x.to(tl.float32) / HC_F).to(x.dtype).to(tl.float32)     # the division rounds as the torch form's does
    tl.store(MIX + r * sM + i, (q * tl.sigmoid(q)).to(x.dtype), mask=m)
    if WITH_INJECT:
        j = tl.arange(0, BH)
        mj = j < HC
        z = tl.load(DI + r * sD + R + j, mask=mj, other=0.0)
        zq = (z.to(tl.float32) / HC_F).to(z.dtype).to(tl.float32)
        g = tl.sigmoid(zq).to(z.dtype).to(tl.float32)               # sigmoid rounds, then the doubling does
        tl.store(INJ + r * sI + j, (2.0 * g).to(z.dtype), mask=mj)


@triton.jit
def _mix_mean(UP, NORMED, OUT, sU, sN, sO, HC_F, HID: tl.constexpr, BD: tl.constexpr, HC: tl.constexpr):
    # a program is one BD-wide tile of a row's channels (carry H3): nothing here reduces along them, so the tiles are
    # the one-block launch's bytes whatever their width
    r = tl.program_id(0)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    m = d < HID
    acc = tl.zeros([BD], dtype=tl.float32)
    for s in tl.static_range(HC):
        off = s * HID + d
        u = tl.load(UP + r * sU + off, mask=m, other=0.0)
        g = tl.sigmoid(u.to(tl.float32)).to(u.dtype).to(tl.float32)
        n = tl.load(NORMED + r * sN + off, mask=m, other=0.0).to(tl.float32)
        acc += (g * n).to(u.dtype).to(tl.float32)
    tl.store(OUT + r * sO + d, (acc / HC_F).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gate_store(total, rows, cols, M, MIX, INJ, sM, sI, HC_F, R: tl.constexpr, HC: tl.constexpr,
                WITH_INJECT: tl.constexpr):
    # `_gates` over the product's columns: the product rounds to BF16 first, as the GEMM's output does
    x = total.to(MIX.dtype.element_ty)
    q = (x.to(tl.float32) / HC_F).to(x.dtype).to(tl.float32)
    live = rows[:, None] < M
    tl.store(MIX + rows[:, None] * sM + cols[None, :], (q * tl.sigmoid(q)).to(x.dtype), mask=live & (cols[None, :] < R))
    if WITH_INJECT:
        g = tl.sigmoid(q).to(x.dtype).to(tl.float32)
        tl.store(INJ + rows[:, None] * sI + (cols - R)[None, :], (2.0 * g).to(x.dtype),
                 mask=live & (cols[None, :] >= R) & (cols[None, :] < R + HC))


@triton.jit
def _down_gates(X, W, MIX, INJ, PART, LOCKS, M, N, K, sX, sW, sM, sI, HC_F, R: tl.constexpr, HC: tl.constexpr,
                WITH_INJECT: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT: tl.constexpr,
                FP32_DOT: tl.constexpr):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0, k1 = split_span(K, pid_k, SPLIT, BLOCK_K)
    acc = rows_dot(X, W, sX, sW, rows, cols, M, N, k0, k1, BLOCK_N, BLOCK_K, FP32_DOT)
    if SPLIT == 1:
        _gate_store(acc, rows, cols, M, MIX, INJ, sM, sI, HC_F, R, HC, WITH_INJECT)
    else:
        total, last = split_sum(acc, PART, LOCKS, pid_n, pid_k, rows, cols, M, N, SPLIT)
        if last:
            _gate_store(total, rows, cols, M, MIX, INJ, sM, sI, HC_F, R, HC, WITH_INJECT)


@triton.jit
def _up_mean(G, W, NORMED, OUT, M, sG, sW, sN, sO, HC_F, HID: tl.constexpr, R: tl.constexpr, HC: tl.constexpr,
             BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
    # `_mix_mean` over one block of hidden channels, the up product's four streams computed here
    rows = tl.arange(0, 16)
    d = tl.program_id(0) * BLOCK_D + tl.arange(0, BLOCK_D)
    live = (rows[:, None] < M) & (d[None, :] < HID)
    acc = tl.zeros((16, BLOCK_D), dtype=tl.float32)
    for s in tl.static_range(HC):
        u = rows_dot(G, W, sG, sW, rows, s * HID + d, M, s * HID + HID, 0, R, BLOCK_D, BLOCK_K, FP32_DOT)
        g = tl.sigmoid(u.to(OUT.dtype.element_ty).to(tl.float32)).to(OUT.dtype.element_ty).to(tl.float32)
        n = tl.load(NORMED + rows[:, None] * sN + (s * HID + d)[None, :], mask=live, other=0.0).to(tl.float32)
        acc += (g * n).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + rows[:, None] * sO + d[None, :], (acc / HC_F).to(OUT.dtype.element_ty), mask=live)


def _warps(width: int) -> int:
    return 4 if width <= 1024 else 8


# Probe hook (probes/engine_qwen38_mix_tiles.py, carry H3): mix_mean's (tile width, warps) forced when set, the rule when
# None. Read when `mix` launches, so a captured graph keeps the tile it was captured with. Nothing served sets it.
_MIX_TILE_OVERRIDE = None


def _mix_tile(hid: int) -> "tuple[int, int]":
    """(tile width, warps) of mix_mean's launch over `hid` channels: one block over the whole row, as it always was."""
    if _MIX_TILE_OVERRIDE is None:
        return triton.next_power_of_2(hid), _warps(hid)
    forced = _MIX_TILE_OVERRIDE
    if (type(forced) is not tuple or len(forced) != 2 or not all(type(n) is int and n > 0 and not n & (n - 1)
                                                                  for n in forced) or forced[0] < 16):
        raise ValueError("_MIX_TILE_OVERRIDE is (tile width of 16 or more, warps), powers of two")
    return forced


def _check_streams(h: torch.Tensor, hc: int, *, least: int = 2) -> int:
    if h.ndim != 2 or type(hc) is not int or hc < least or h.shape[1] % hc:
        raise ValueError(f"the streams are [N, hc*H] with hc >= {least}; got {tuple(h.shape)} for hc {hc}")
    if h.stride(1) != 1:
        raise ValueError("the streams must be packed along their channels")
    return h.shape[1] // hc


def pack_down_inject(down: torch.Tensor, inject: "torch.Tensor | None") -> torch.Tensor:
    """The mixer's down projection [r, hc*H] and the injection [hc, hc*H] as one weight [r + hc, hc*H]: both read
    the normalised streams, so one matmul computes both. Bind once when the weights are bound; `inject` None is the
    closing mixer, which has no injection."""
    if down.ndim != 2 or (inject is not None and (inject.ndim != 2 or inject.shape[1] != down.shape[1]
                                                  or inject.dtype != down.dtype)):
        raise ValueError("down [r, hc*H] and inject [hc, hc*H] share their input width and dtype")
    return (down if inject is None else torch.cat([down, inject], 0)).contiguous()


def norm_streams(h: torch.Tensor, w: torch.Tensor, eps: float, hc: int) -> torch.Tensor:
    """rmsnorm_unit_offset(h, w, eps, group=H): each of the hc streams normalised on its own, weight 1 + w. hc 1 is the
    plain unit-offset norm over the whole row (the MTP fuse's joint norm over 10,240 channels, its embedding norm)."""
    hid = _check_streams(h, hc, least=1)
    if w.shape != (h.shape[1],):
        raise ValueError("the stream norm's weight covers every stream's channels")
    if not h.is_cuda:
        from engine.modules.norm import rmsnorm_unit_offset
        return rmsnorm_unit_offset(h, w, eps, group=None if hc == 1 else hid)
    out = torch.empty_like(h)
    if h.shape[0]:
        _norm_streams[(h.shape[0], hc)](h, w, out, h.stride(0), out.stride(0), eps,
                                        HID=hid, BD=triton.next_power_of_2(hid), num_warps=_warps(hid))
    return out


def leave(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor, hc: int) -> torch.Tensor:
    """h + out (x) inject, in place: the sublayer's output [N, H] added into every stream with that stream's weight
    [N, hc]. For the site before an injection feature, which reads the streams un-normalised."""
    return _leave(h, out, inject, None, 0.0, hc, norm=False)[0]


def leave_norm(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor, w: torch.Tensor, eps: float,
               hc: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """The previous site's leave and this site's stream norm in one pass: h updated in place, and the normalised
    streams the mixer reads. Returns (h, normed)."""
    if w.shape != (h.shape[1],):
        raise ValueError("the stream norm's weight covers every stream's channels")
    return _leave(h, out, inject, w, eps, hc, norm=True)


def _leave(h, out, inject, w, eps, hc, *, norm):
    hid = _check_streams(h, hc)
    if out.shape != (h.shape[0], hid) or inject.shape != (h.shape[0], hc):
        raise ValueError(f"a leave takes the output [N, {hid}] and the injection [N, {hc}] for {h.shape[0]} rows")
    if out.dtype != h.dtype or inject.dtype != h.dtype or out.stride(1) != 1 or inject.stride(1) != 1:
        raise ValueError("the output and the injection are packed rows in the streams' dtype")
    if not h.is_cuda:
        h.add_((out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))
        if not norm:
            return h, None
        from engine.modules.norm import rmsnorm_unit_offset
        return h, rmsnorm_unit_offset(h, w, eps, group=hid)
    normed = torch.empty_like(h) if norm else h
    if h.shape[0]:
        _leave_norm[(h.shape[0], hc)](h, out, inject, w if norm else h, normed, h.stride(0), out.stride(0),
                                      inject.stride(0), normed.stride(0), eps, HID=hid,
                                      BD=triton.next_power_of_2(hid), NORM=norm, num_warps=_warps(hid))
    return h, (normed if norm else None)


def mix(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor, hc: int, *,
        inject: bool = True, project_down=None, project_up=None) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """The site's mixer over the normalised streams: (mixed [N, H], injection [N, hc] or None for the closing mixer).
    `down_inject` is pack_down_inject's weight; `up` [hc*H, r]. `project_down` / `project_up`, when given, compute the
    two matmuls instead of BF16 torch.mm over those weights -- (normed [N, hc*H]) -> [N, r(+hc)] and (gates [N, r]) ->
    [N, hc*H] BF16, rows packed along their columns (a quantised dense lane); the weights still name the shapes."""
    hid = _check_streams(normed, hc)
    rank = up.shape[1]
    if up.shape != (normed.shape[1], rank) or down_inject.shape != (rank + (hc if inject else 0), normed.shape[1]):
        raise ValueError(f"a site mixes through down(+inject) [{rank}{' + ' + str(hc) if inject else ''}, "
                         f"{normed.shape[1]}] and up [{normed.shape[1]}, {rank}]")
    if down_inject.dtype != normed.dtype or up.dtype != normed.dtype:
        raise ValueError("the hyper-connection weights are held in the activations' dtype (BF16 in the checkpoint)")
    if not normed.is_cuda:
        di = torch.nn.functional.linear(normed, down_inject) if project_down is None else project_down(normed)
        gates = torch.nn.functional.silu(di[:, :rank] / hc)
        up_rows = torch.nn.functional.linear(gates, up) if project_up is None else project_up(gates)
        weights = torch.sigmoid(up_rows).unflatten(-1, (hc, hid))
        mixed = (weights * normed.unflatten(-1, (hc, hid))).mean(dim=-2)
        return mixed, (2 * torch.sigmoid(di[:, rank:] / hc) if inject else None)
    rows = normed.shape[0]
    if project_down is None and project_up is None and folds(normed, down_inject, up):
        return mix_rows(normed, down_inject, up, hc, inject=inject)
    di = torch.mm(normed, down_inject.t()) if project_down is None else project_down(normed)
    if di.shape != (rows, down_inject.shape[0]) or di.dtype != normed.dtype or di.stride(1) != 1:
        raise ValueError("the down projection returns packed [N, r(+hc)] rows in the streams' dtype")
    gates = torch.empty(rows, rank, device=normed.device, dtype=normed.dtype)
    injection = torch.empty(rows, hc, device=normed.device, dtype=normed.dtype) if inject else None
    mixed = torch.empty(rows, hid, device=normed.device, dtype=normed.dtype)
    if rows:
        _gates[(rows,)](di, gates, gates if injection is None else injection, di.stride(0), gates.stride(0),
                        (gates if injection is None else injection).stride(0), float(hc), R=rank,
                        BR=triton.next_power_of_2(rank), HC=hc, BH=triton.next_power_of_2(hc), WITH_INJECT=inject,
                        num_warps=4)
    weights = torch.mm(gates, up.t()) if project_up is None else project_up(gates)
    if weights.shape != (rows, normed.shape[1]) or weights.dtype != normed.dtype or weights.stride(1) != 1:
        raise ValueError("the up projection returns packed [N, hc*H] rows in the streams' dtype")
    if rows:
        tile, warps = _mix_tile(hid)
        _mix_mean[(rows, triton.cdiv(hid, tile))](weights, normed, mixed, weights.stride(0), normed.stride(0),
                                                  mixed.stride(0), float(hc), HID=hid, BD=tile, HC=hc, num_warps=warps)
    return mixed, injection


def folds(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor) -> bool:
    """Whether `mix_rows` serves this site: 1..DECODE_ROWS rows in BF16, a down projection the skinny GEMV has a
    tile for, each operand packed along its last dimension."""
    return (1 <= normed.shape[0] <= DECODE_ROWS and tuple(down_inject.shape) in skinny_gemv.CONFIGS
            and normed.dtype == down_inject.dtype == up.dtype == torch.bfloat16
            and normed.stride(1) == 1 and down_inject.stride(1) == 1 and up.stride(1) == 1)


def mix_rows(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor, hc: int, *,
             inject: bool = True) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """`mix` for a decode step's rows in two launches (the module's docstring: carry H2): (mixed [N, H],
    injection [N, hc] or None). `mix` takes it on CUDA where `folds` says; the Triton interpreter runs it on the CPU."""
    hid = _check_streams(normed, hc)
    rank = up.shape[1]
    if up.shape != (normed.shape[1], rank) or down_inject.shape != (rank + (hc if inject else 0), normed.shape[1]):
        raise ValueError(f"a site mixes through down(+inject) [{rank}{' + ' + str(hc) if inject else ''}, "
                         f"{normed.shape[1]}] and up [{normed.shape[1]}, {rank}]")
    if not folds(normed, down_inject, up):
        raise ValueError(f"mix_rows takes 1..{DECODE_ROWS} BF16 rows of a down projection skinny_gemv tiles; got "
                         f"{tuple(normed.shape)} {normed.dtype} over {tuple(down_inject.shape)}")
    rows, width = normed.shape
    n = down_inject.shape[0]
    block_n, block_k, split, warps, stages = skinny_gemv.CONFIGS[tuple(down_inject.shape)]
    gates = torch.empty(rows, rank, device=normed.device, dtype=normed.dtype)
    injection = torch.empty(rows, hc, device=normed.device, dtype=normed.dtype) if inject else None
    inj = gates if injection is None else injection
    if split > 1:
        partial = torch.empty(split, rows, n, device=normed.device, dtype=torch.float32)
        locks = skinny_gemv.prepare(normed.device)
    else:
        partial = locks = gates
    interpreted = not normed.is_cuda
    _down_gates[(triton.cdiv(n, block_n), split)](
        normed, down_inject, gates, inj, partial, locks, rows, n, width, normed.stride(0), down_inject.stride(0),
        gates.stride(0), inj.stride(0), float(hc), R=rank, HC=hc, WITH_INJECT=inject, BLOCK_N=block_n,
        BLOCK_K=block_k, SPLIT=split, FP32_DOT=interpreted, num_warps=warps, num_stages=stages)
    mixed = torch.empty(rows, hid, device=normed.device, dtype=normed.dtype)
    block_d, block_k, warps, stages = UP_TILE
    _up_mean[(triton.cdiv(hid, block_d),)](
        gates, up, normed, mixed, rows, gates.stride(0), up.stride(0), normed.stride(0), mixed.stride(0), float(hc),
        HID=hid, R=rank, HC=hc, BLOCK_D=block_d, BLOCK_K=block_k, FP32_DOT=interpreted, num_warps=warps,
        num_stages=stages)
    return mixed, injection


def drift(ours: torch.Tensor, ref: torch.Tensor) -> "tuple[float, float]":
    """(largest error over the largest reference magnitude, error RMS over reference RMS), in fp32. A BF16 step is 2^-7
    of a value (7 mantissa bits): an operation that rounds where the torch form does not -- a reduction's partial
    sums -- moves an element by a step or two, far under the tenths that a wrong formula, stream or channel gives."""
    a, b = ours.float(), ref.float()
    err = _far((a - b).abs())
    return (float((err.max() / b.abs().max().clamp_min(1e-30)).item()),
            float((err.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)).item()))


def _far(err: torch.Tensor) -> torch.Tensor:
    """An element that is not a number is as far from its reference as an element gets. Left a NaN it would pass every
    hold: `nan > band` is false, and so is what `max(0.0, nan)` compares."""
    return torch.nan_to_num(err, nan=float("inf"), posinf=float("inf"))


def blame(ours: torch.Tensor, ref: torch.Tensor, ours_fn, ref_fn, host_fn=None, *, band_max: float = 5e-2) -> str:
    """What a failed hold says about itself, for its error -- `drift`'s two numbers cannot say where the error sits or
    whose it is, and a failure that does not come back leaves nothing else to read (measurements/qwen38_lane_20260919).
    `ours` and `ref` are the tensors that failed; `ours_fn()` and `ref_fn()` make the same two calls again. Says where
    the elements past `band_max` sit (how many, the span of each index, the worst one's two values), whether each side
    repeats itself -- a side that gives other bytes to the same call is the wrong one, and its fault is not its
    arithmetic -- and, with `host_fn` (the reference computed on the CPU), which side leaves it. The failing path only."""
    a, b = ours.float(), ref.float()
    err = _far((a - b).abs())
    past = (err > band_max * b.abs().max().clamp_min(1e-30)).nonzero()
    if past.shape[0]:
        spans = ", ".join(f"dim {d} {int(past[:, d].min())}..{int(past[:, d].max())} ({int(past[:, d].unique().numel())} "
                          f"distinct)" for d in range(past.shape[1]))
        at = tuple(int(i) for i in (err == err.max()).nonzero()[0])
        said = [f"{past.shape[0]} of {err.numel()} elements past the max band: {spans}; the worst at {at} is "
                f"{a[at].item():.6g} against the reference's {b[at].item():.6g}"]
    else:
        said = ["no element is past the max band (the rms band is what failed)"]
    for name, first, fn in (("ours", ours, ours_fn), ("the reference", ref, ref_fn)):
        again = fn()
        moved = int(((again != first) & ~(again.isnan() & first.isnan())).sum().item())
        said.append(f"{name} computed again gives the same bytes" if not moved else
                    f"{name} computed again differs in {moved} elements -- it does not repeat itself")
    if host_fn is not None:
        host = host_fn().to(ref.device)
        said.append(f"against the CPU's reference ours drifts {drift(ours, host)} and the device's reference "
                    f"{drift(ref, host)}")
    return "; ".join(said)


def qualify(device, *, hc: int, hidden: int, rank: int, eps: float, dtype=torch.bfloat16, rows=(1, 5, 64),
            band_max: float = 5e-2, band_rms: float = 2e-2, seed: int = 0) -> dict:
    """Hold the lane to engine/modules/hyper_connection.gated_residual on `device`, with random weights at the model's
    widths: two sites joined by a leave with its norm, a leave without one, and a closing mixer. Raises when an output
    drifts past `band_max` (largest error / largest magnitude) or `band_rms` (`drift`): bounds a few BF16 steps wide,
    so rounding order passes and arithmetic does not; returns the worst (max, rms) seen per output."""
    from engine.modules.hyper_connection import gated_residual
    gen = torch.Generator(device="cpu").manual_seed(seed)
    width = hc * hidden

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=dtype)

    worst = {k: (0.0, 0.0) for k in ("enter", "inject", "leave_norm", "leave", "close")}
    norm_w, down, up, inj = rand(width, scale=0.1), rand(rank, width, scale=0.02), rand(width, rank, scale=0.02), \
        rand(hc, width, scale=0.02)
    norm_c, down_c, up_c = rand(width, scale=0.1), rand(rank, width, scale=0.02), rand(width, rank, scale=0.02)
    di, dc = pack_down_inject(down, inj), pack_down_inject(down_c, None)

    def note(key, ours, ref):
        m, r = drift(ours, ref)
        worst[key] = (max(worst[key][0], m), max(worst[key][1], r))

    for n in rows:
        h = rand(n, width)
        ref_mixed, ref_inj = gated_residual(h, norm_w, down, up, inj, hc, eps)
        mixed, injection = mix(norm_streams(h, norm_w, eps, hc), di, up, hc)
        note("enter", mixed, ref_mixed)
        note("inject", injection, ref_inj)
        out = rand(n, hidden)
        ref_h = h + (out.unsqueeze(-2) * ref_inj.unsqueeze(-1)).flatten(-2)
        ours, normed = leave_norm(h.clone(), out, ref_inj, norm_c, eps, hc)
        note("leave_norm", ours, ref_h)
        note("leave", leave(h.clone(), out, ref_inj, hc), ref_h)
        closed, _ = mix(normed, dc, up_c, hc, inject=False)
        note("close", closed, gated_residual(ref_h, norm_c, down_c, up_c, None, hc, eps))
    bad = {k: v for k, v in worst.items() if v[0] > band_max or v[1] > band_rms}
    if bad:
        raise RuntimeError(f"gated residual lane drifts from engine/modules/hyper_connection.gated_residual beyond "
                           f"max {band_max:g} / rms {band_rms:g}: {bad}")
    return worst


__all__ = ["DECODE_ROWS", "pack_down_inject", "norm_streams", "leave", "leave_norm", "mix", "folds", "mix_rows", "drift",
           "blame", "qualify"]
