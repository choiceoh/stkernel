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

A prefill step's rows (PREFILL_ROWS or more) take `mix_block`: the same two folds over row blocks, on tensor-core tiles
instead of the skinny GEMV's padded 16 rows --

    down_gates_rows   down(+inject) for a block of rows and a block of the mixer's columns, the gates stored   one launch
                      from the product; the column blocks of a row block adjacent programs (the rows read from
                      DRAM once, the other blocks' reads L2's), the injection's 4 columns a narrow last block
                      rather than a padded one, and K split where the rows are too few to fill the device --
                      the last program of a tile sums the split, as `_down_gates` does (DOWN_TILES; cuBLAS and
                      `_gates` for rows short of the table)
    up_mean_rows      up for a block of rows and a block of hidden channels in each stream, sigmoid, times the
                      streams, the mean -- the [N, hc*H] up product (84 MB at 4,096 rows) never written      one launch

`site` serves a prefill step's site whole and never writes the normalised streams [N, hc*H] (84 MB at 4,096 rows, a
third of the leave's bytes): the leave stores each stream's scale [N, hc] FP32 instead (`stream_scales`), and both
launches normalise each tile of the streams as they read it -- the stream norm's own arithmetic on the same scale, so
the operand is the normalised streams' bytes and the outputs are leave_norm-then-mix_block's byte for byte.

On a GB10 (q38sitecmp-0919a, eight sites a graph, interleaved): up and the mean 1,703 -> 815 us a site at 4,096 rows
(x2.09), x1.66 at 2,048, x1.65 at 1,024, x1.39 at 512 -- and the output byte for byte cuBLAS's up with `_mix_mean`.
The down projection and the gates, before the column blocks were adjacent programs: 590 -> 557 us at 4,096 rows at
128 x 128 x 32 tiles (-5.7%), level with cuBLAS at 1,024 and 2,048 rows, behind it at 512 (+24%).

The arithmetic after each product is `_gates`' and `_mix_mean`'s, on the same BF16 product, so the site's outputs are
byte for byte the five-launch site's with the same products (probes/engine_qwen38_gemv, q38site-0919a). On a GB10 at
Qwen3.8's widths the mixer went from 66.5-70.5 us (its four launches on cuBLAS, plus an output copy the probe added)
to 60.3-61.8 us for 1-16 rows, 16 sites a graph over rotated weights: 13.2 MB of weights at about 218 GB/s.

`norm_streams` opens the first site (no output to add yet) and follows an injection feature that reads the
streams between two sites (Qwen3.8's PLE before layer 1, the config's one-indexed 2); `leave` adds an output without the norm for the same
case. The closing mixer is a site without an injection.

Every leave follows a sublayer's TP sum, and on the fleet a sum mostly waits: the one-shot consumer publishes this rank's
packet, then idles about 20 us until the other ranks' land, the memory idle with it. A leave launched with `pdl` (carry
H4, GLM-5.3's AR consumer overlap: the peer wait prepared the next MHC's immutable weights) is that consumer's
programmatic dependent -- resident through the wait, started when the sum lands -- and with `prefetch` it spends the
wait pulling the site's down projection into L2, the weight the skinny GEMV reads next and reads once. Neither changes
a byte the leave computes.

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
PREFILL_ROWS = 512                          # rows from this take `mix_block` (a prefill step's; q38sitecmp-0919a)
# mix_block's down tile (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, split) by the rows it serves from, the last entry the
# rows reach; rows short of every entry take cuBLAS and `_gates`. Its up tile (BLOCK_M, BLOCK_D, BLOCK_K, warps, stages)
# at every row.
DOWN_TILES = ((512, (128, 64, 64, 4, 3, 1)), (4096, (128, 128, 32, 8, 3, 1)))
UP_BLOCK_TILE = (64, 64, 32, 4, 4)
UP_TILE = (32, 64, 4, 3)                    # up_mean's BLOCK_D, BLOCK_K, warps, stages (the best of five, q38site-0919a)


@triton.jit
def _norm_streams(X, W, OUT, sX, sO, EPS, HID: tl.constexpr, BD: tl.constexpr, SCALE_ONLY: tl.constexpr):
    # SCALE_ONLY: OUT is [N, hc] FP32, the stream's scale and not the normalised stream (`stream_scales`)
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    x = tl.load(X + r * sX + off, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
    if SCALE_ONLY:
        tl.store(OUT + r * sO + s, scale)
    else:
        w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + r * sO + off, ((x * scale) * (1.0 + w)).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _prefetch_l2(NEXT, SECTORS, p, P, BLOCK: tl.constexpr):
    # program p of P's share of the next launch's weight, as 32-byte sectors (16 BF16) pulled into L2. A prefetch
    # returns nothing and writes nothing, so the share's clamped tail may name a sector twice
    per = tl.cdiv(SECTORS, P)
    lo = p * per
    hi = tl.minimum(lo + per, SECTORS)
    for i in range(lo, hi, BLOCK):
        sector = tl.minimum(i + tl.arange(0, BLOCK), hi - 1)
        tl.inline_asm_elementwise("prefetch.global.L2 [$1]; // $0", "=r,l", [NEXT + sector * 16], dtype=tl.int32,
                                  is_pure=False, pack=1)


@triton.jit
def _leave_norm(H, OUT, INJ, W, NORMED, NEXT, sH, sO, sI, sN, EPS, SECTORS, HID: tl.constexpr, BD: tl.constexpr,
                NORM: tl.constexpr, PDL: tl.constexpr, PREFETCH: tl.constexpr, SCALE_ONLY: tl.constexpr):
    # SCALE_ONLY: NORMED is [N, hc] FP32, the stream's scale and not the normalised stream (`stream_scales`)
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    if NORM and not SCALE_ONLY:
        w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)   # immutable: read while the sum is still in flight
    if PREFETCH:
        _prefetch_l2(NEXT, SECTORS, r * tl.num_programs(1) + s, tl.num_programs(0) * tl.num_programs(1), BD)
    if PDL:
        # the sum (OUT) is the primary's; the streams and the injection are read after the wait as well, so a leave
        # whose previous launch wrote one of them (the MTP head's row selection) reads what that launch wrote
        tl.extra.cuda.gdc_wait()
    h = tl.load(H + r * sH + off, mask=m, other=0.0)
    o = tl.load(OUT + r * sO + d, mask=m, other=0.0).to(tl.float32)
    g = tl.load(INJ + r * sI + s).to(tl.float32)
    delta = (o * g).to(h.dtype)                                   # the product rounds, then the sum does
    new = (h.to(tl.float32) + delta.to(tl.float32)).to(h.dtype)
    tl.store(H + r * sH + off, new, mask=m)
    if NORM:
        x = new.to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
        if SCALE_ONLY:
            tl.store(NORMED + r * sN + s, scale)
        else:
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
MIX_TILE_ROWS = 32                  # rows up to this take the narrow tile (measurements/qwen38_mix_tiles_20260919)


def _mix_tile(hid: int, rows: int) -> "tuple[int, int]":
    """(tile width, warps) of mix_mean's launch over `rows` rows of `hid` channels, as a GB10 measured it (carry H3):
    a step of up to MIX_TILE_ROWS rows -- a captured step past the folded mixer's 16, or any under --hc-fp8 -- in
    256-wide tiles at 4 warps, 30-51% less time than one block a row at 4, 17 and 32 rows; more rows in one block a row,
    as always (from 64 rows the launch is the bandwidth's: every geometry within 2%). A tile is the same bytes."""
    if _MIX_TILE_OVERRIDE is None:
        if rows <= MIX_TILE_ROWS:
            return min(256, triton.next_power_of_2(hid)), 4
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
                                        HID=hid, BD=triton.next_power_of_2(hid), SCALE_ONLY=False, num_warps=_warps(hid))
    return out


def leave(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor, hc: int, *, pdl: bool = False) -> torch.Tensor:
    """h + out (x) inject, in place: the sublayer's output [N, H] added into every stream with that stream's weight
    [N, hc]. For the site before an injection feature, which reads the streams un-normalised. `pdl`: as leave_norm's."""
    return _leave(h, out, inject, None, 0.0, hc, norm=False, pdl=pdl)[0]


def leave_norm(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor, w: torch.Tensor, eps: float,
               hc: int, *, pdl: bool = False, prefetch: "torch.Tensor | None" = None
               ) -> "tuple[torch.Tensor, torch.Tensor]":
    """The previous site's leave and this site's stream norm in one pass: h updated in place, and the normalised
    streams the mixer reads. Returns (h, normed).

    `pdl` (carry H4): launched as the programmatic dependent of the launch before it -- on the fleet the TP sum of `out`,
    whose one-shot consumer releases its dependents once it has published and then waits about 20 us for the other
    ranks' packets. The leave is resident through that wait and starts when the sum lands, not a launch later. Only the
    immutable norm weight is read before `griddepcontrol.wait`, so the bytes are the ordinary launch's whatever launch
    comes before. `prefetch`: the weight the next launch reads first (this site's down projection), pulled into L2 during
    the same wait, `PREFETCH_BYTES` of it -- only with `pdl` (without it the leave starts after the sum and there is no
    wait to fill) and only for a decode step's rows, which the skinny GEMV serves with one read of the weight."""
    if w.shape != (h.shape[1],):
        raise ValueError("the stream norm's weight covers every stream's channels")
    return _leave(h, out, inject, w, eps, hc, norm=True, pdl=pdl, prefetch=prefetch)


def stream_scales(h: torch.Tensor, out: "torch.Tensor | None", injection: "torch.Tensor | None", eps: float, hc: int, *,
                  pdl: bool = False) -> torch.Tensor:
    """Each stream's norm scale rsqrt(mean(x^2) + eps), [N, hc] FP32 -- after `out` leaves into the streams in place
    with `injection` (leave_norm's leave), or of the streams as they are when `out` is None (norm_streams'). The scale
    leave_norm and norm_streams multiply by, bit for bit; the normalised streams are not written (mix_block's `norm`
    normalises what it reads). CUDA, or the Triton interpreter. `pdl`: as leave_norm's."""
    hid = _check_streams(h, hc, least=1)
    if out is not None:
        return _leave(h, out, injection, h, eps, hc, norm=True, pdl=pdl, scale_only=True)[1]
    scale = torch.empty(h.shape[0], hc, device=h.device, dtype=torch.float32)
    if h.shape[0]:
        _norm_streams[(h.shape[0], hc)](h, h, scale, h.stride(0), scale.stride(0), eps, HID=hid,
                                        BD=triton.next_power_of_2(hid), SCALE_ONLY=True, num_warps=_warps(hid))
    return scale


# The bytes of the next weight a prefetching leave pulls into L2 (carry H4): None, all of it. The mixer's down projection
# is 6.6 MB; a sum's wait on the fleet is about 20 us, about 4 MB at the memory's bandwidth.
PREFETCH_BYTES = None
# Probe hook (probes/engine_qwen38_leave.py): the budget forced when set, bytes (0: none). Read when a leave launches, so
# a captured graph keeps the budget it was captured with. Nothing served sets it.
_PREFETCH_BYTES_OVERRIDE = None


def _prefetch_sectors(weight, rows: int) -> int:
    """The 32-byte sectors of `weight` a leave of `rows` rows prefetches: 0 without a weight, past a decode step's rows,
    or for a weight that is not packed BF16 on the leave's device."""
    if weight is None or rows > DECODE_ROWS:
        return 0
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous() or weight.device.type != "cuda":
        raise ValueError("a leave prefetches a packed BF16 weight on its own device")
    budget = PREFETCH_BYTES if _PREFETCH_BYTES_OVERRIDE is None else _PREFETCH_BYTES_OVERRIDE
    size = weight.numel() * weight.element_size()
    return (size if budget is None else min(size, budget)) // 32


def _leave(h, out, inject, w, eps, hc, *, norm, pdl=False, prefetch=None, scale_only=False):
    hid = _check_streams(h, hc)
    if out.shape != (h.shape[0], hid) or inject.shape != (h.shape[0], hc):
        raise ValueError(f"a leave takes the output [N, {hid}] and the injection [N, {hc}] for {h.shape[0]} rows")
    if out.dtype != h.dtype or inject.dtype != h.dtype or out.stride(1) != 1 or inject.stride(1) != 1:
        raise ValueError("the output and the injection are packed rows in the streams' dtype")
    if type(pdl) is not bool:
        raise ValueError("pdl is a declared boolean")
    if not h.is_cuda and not scale_only:
        h.add_((out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))
        if not norm:
            return h, None
        from engine.modules.norm import rmsnorm_unit_offset
        return h, rmsnorm_unit_offset(h, w, eps, group=hid)
    if scale_only:
        normed = torch.empty(h.shape[0], hc, device=h.device, dtype=torch.float32)
    else:
        normed = torch.empty_like(h) if norm else h
    # the interpreter (tests' is_cuda stand-in) has neither griddepcontrol nor a prefetch: the ordinary launch there
    pdl = pdl and h.device.type == "cuda"
    sectors = _prefetch_sectors(prefetch, h.shape[0]) if pdl else 0
    if h.shape[0]:
        _leave_norm[(h.shape[0], hc)](h, out, inject, w if norm else h, normed, prefetch if sectors else h,
                                      h.stride(0), out.stride(0), inject.stride(0), normed.stride(0), eps, sectors,
                                      HID=hid, BD=triton.next_power_of_2(hid), NORM=norm, PDL=pdl,
                                      PREFETCH=sectors > 0, SCALE_ONLY=scale_only, num_warps=_warps(hid),
                                      launch_pdl=pdl)
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
    if project_down is None and project_up is None and blocks_fold(normed, down_inject, up):
        return mix_block(normed, down_inject, up, hc, inject=inject)
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
        tile, warps = _mix_tile(hid, rows)
        _mix_mean[(rows, triton.cdiv(hid, tile))](weights, normed, mixed, weights.stride(0), normed.stride(0),
                                                  mixed.stride(0), float(hc), HID=hid, BD=tile, HC=hc, num_warps=warps)
    return mixed, injection


def site(h: torch.Tensor, out: "torch.Tensor | None", injection: "torch.Tensor | None", w: torch.Tensor, eps: float,
         hc: int, down_inject: torch.Tensor, up: torch.Tensor, *, inject: bool = True, pdl: bool = False,
         prefetch: "torch.Tensor | None" = None) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """A site whole: `out` (the previous sublayer's, with its `injection`; None at the first site) left into the streams
    h in place, the streams normalised with `w`, and the mixer -- (mixed [N, H], injection [N, hc] or None). A prefill
    step's rows (mix_block's, with a down tile) never write the normalised streams: the leave keeps each stream's scale
    (`stream_scales`) and mix_block's two launches normalise what they read -- leave_norm's and mix's bytes, and at
    4,096 rows 84 MB less written. Other rows: leave_norm (or norm_streams) and mix, `pdl` and `prefetch` as
    leave_norm's."""
    if w.shape != (h.shape[1],):
        raise ValueError("the stream norm's weight covers every stream's channels")
    tiles = block_tiles(h.shape[0])
    if (h.is_cuda and blocks_fold(h, down_inject, up) and tiles["down"] is not None
            and (h.shape[1] // hc) % tiles["down"][2] == 0):
        scale = stream_scales(h, out, injection, eps, hc, pdl=pdl)
        return mix_block(h, down_inject, up, hc, inject=inject, tiles=tiles, norm=(scale, w))
    if out is None:
        normed = norm_streams(h, w, eps, hc)
    else:
        h, normed = leave_norm(h, out, injection, w, eps, hc, pdl=pdl, prefetch=prefetch)
    return mix(normed, down_inject, up, hc, inject=inject)


def folds(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor) -> bool:
    """Whether `mix_rows` serves this site: 1..DECODE_ROWS rows in BF16, a down projection the skinny GEMV has a
    tile for, each operand packed along its last dimension."""
    return (1 <= normed.shape[0] <= DECODE_ROWS and tuple(down_inject.shape) in skinny_gemv.CONFIGS
            and normed.dtype == down_inject.dtype == up.dtype == torch.bfloat16
            and normed.stride(1) == 1 and down_inject.stride(1) == 1 and up.stride(1) == 1)


@triton.jit
def _tile_dot(X, W, sX, sW, rows, cols, M, N, k0, k1, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
    """[BLOCK_M, BLOCK_N] FP32: X's rows `rows` (< M) times W's rows `cols` (< N) over K in [k0, k1), BLOCK_K at a
    time."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k1, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        km = ks < k1
        x = tl.load(X + rows[:, None] * sX + ks[None, :], mask=(rows[:, None] < M) & km[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * sW + ks[None, :], mask=(cols[:, None] < N) & km[None, :], other=0.0)
        if FP32_DOT:                                              # the interpreter reads a BF16 dot's bits as integers
            x, w = x.to(tl.float32), w.to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
    return acc


@triton.jit
def _tile_dot_normed(X, W, SC, NW, sX, sW, sS, rows, cols, M, N, k0, k1, HID: tl.constexpr, BLOCK_M: tl.constexpr,
                     BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, FP32_DOT: tl.constexpr):
    """`_tile_dot` over the streams X normalised as each tile is read: row r's stream s times its scale SC[r, s], times
    1 + NW -- the stream norm's arithmetic (`_leave_norm`), so the operand is the normalised streams' bytes. A K tile
    lies in one stream (HID a multiple of BLOCK_K)."""
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    live = rows < M
    for k in range(k0, k1, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        km = ks < k1
        x = tl.load(X + rows[:, None] * sX + ks[None, :], mask=live[:, None] & km[None, :], other=0.0)
        sc = tl.load(SC + rows * sS + k // HID, mask=live, other=0.0)
        nw = tl.load(NW + ks, mask=km, other=0.0).to(tl.float32)
        x = ((x.to(tl.float32) * sc[:, None]) * (1.0 + nw[None, :])).to(X.dtype.element_ty)
        w = tl.load(W + cols[:, None] * sW + ks[None, :], mask=(cols[:, None] < N) & km[None, :], other=0.0)
        if FP32_DOT:
            x, w = x.to(tl.float32), w.to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
    return acc


@triton.jit
def _down_gates_tile(X, W, MIX, INJ, PART, LOCKS, SC, NW, tile, rows, cols, pid_k, M, N, K, sX, sW, sM, sI, sS, HC_F,
                     R: tl.constexpr, HC: tl.constexpr, WITH_INJECT: tl.constexpr, BLOCK_M: tl.constexpr,
                     WIDTH: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT: tl.constexpr, NORM_IN: tl.constexpr,
                     HID: tl.constexpr, FP32_DOT: tl.constexpr):
    # one [BLOCK_M, WIDTH] tile's share of K, and the gates from the whole product where this program has it
    k0, k1 = split_span(K, pid_k, SPLIT, BLOCK_K)
    if NORM_IN:
        acc = _tile_dot_normed(X, W, SC, NW, sX, sW, sS, rows, cols, M, N, k0, k1, HID, BLOCK_M, WIDTH, BLOCK_K,
                               FP32_DOT)
    else:
        acc = _tile_dot(X, W, sX, sW, rows, cols, M, N, k0, k1, BLOCK_M, WIDTH, BLOCK_K, FP32_DOT)
    if SPLIT == 1:
        _gate_store(acc, rows, cols, M, MIX, INJ, sM, sI, HC_F, R, HC, WITH_INJECT)
    else:
        total, last = split_sum(acc, PART, LOCKS, tile, pid_k, rows, cols, M, N, SPLIT)
        if last:
            _gate_store(total, rows, cols, M, MIX, INJ, sM, sI, HC_F, R, HC, WITH_INJECT)


@triton.jit
def _down_gates_rows(X, W, MIX, INJ, PART, LOCKS, SC, NW, M, N, K, sX, sW, sM, sI, sS, HC_F, R: tl.constexpr,
                     HC: tl.constexpr, WITH_INJECT: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     BLOCK_K: tl.constexpr, TAIL: tl.constexpr, SPLIT: tl.constexpr, NORM_IN: tl.constexpr,
                     HID: tl.constexpr, FP32_DOT: tl.constexpr):
    # `_down_gates` over a block of rows. The column block is the fastest grid index, so the blocks of one row block run
    # side by side and share its rows through L2; TAIL > 0 is the last column block's narrower width (its live columns
    # rounded up to a dot's 16)
    pid_n, pid_m, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    tile = pid_m * tl.num_programs(0) + pid_n
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if TAIL > 0:
        if pid_n == tl.num_programs(0) - 1:
            _down_gates_tile(X, W, MIX, INJ, PART, LOCKS, SC, NW, tile, rows, pid_n * BLOCK_N + tl.arange(0, TAIL),
                             pid_k, M, N, K, sX, sW, sM, sI, sS, HC_F, R, HC, WITH_INJECT, BLOCK_M, TAIL, BLOCK_K, SPLIT,
                             NORM_IN, HID, FP32_DOT)
        else:
            _down_gates_tile(X, W, MIX, INJ, PART, LOCKS, SC, NW, tile, rows, pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
                             pid_k, M, N, K, sX, sW, sM, sI, sS, HC_F, R, HC, WITH_INJECT, BLOCK_M, BLOCK_N, BLOCK_K,
                             SPLIT, NORM_IN, HID, FP32_DOT)
    else:
        _down_gates_tile(X, W, MIX, INJ, PART, LOCKS, SC, NW, tile, rows, pid_n * BLOCK_N + tl.arange(0, BLOCK_N),
                         pid_k, M, N, K, sX, sW, sM, sI, sS, HC_F, R, HC, WITH_INJECT, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT,
                         NORM_IN, HID, FP32_DOT)


@triton.jit
def _up_mean_rows(G, W, NORMED, OUT, SC, NW, M, sG, sW, sN, sO, sS, HC_F, HID: tl.constexpr, R: tl.constexpr,
                  HC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr,
                  NORM_IN: tl.constexpr, FP32_DOT: tl.constexpr):
    # `_up_mean` over a block of rows: each stream's up product for this block of channels, rounded to BF16 where the
    # GEMM's output was, then `_mix_mean`'s arithmetic -- the product never leaves the program. NORM_IN: NORMED is the
    # streams, normalised here as `_tile_dot_normed` does
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    live = (rows[:, None] < M) & (d[None, :] < HID)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for s in tl.static_range(HC):
        u = _tile_dot(G, W, sG, sW, rows, s * HID + d, M, s * HID + HID, 0, R, BLOCK_M, BLOCK_D, BLOCK_K, FP32_DOT)
        g = tl.sigmoid(u.to(OUT.dtype.element_ty).to(tl.float32)).to(OUT.dtype.element_ty).to(tl.float32)
        n = tl.load(NORMED + rows[:, None] * sN + (s * HID + d)[None, :], mask=live, other=0.0).to(tl.float32)
        if NORM_IN:
            sc = tl.load(SC + rows * sS + s, mask=rows < M, other=0.0)
            nw = tl.load(NW + s * HID + d, mask=d < HID, other=0.0).to(tl.float32)
            n = ((n * sc[:, None]) * (1.0 + nw[None, :])).to(OUT.dtype.element_ty).to(tl.float32)
        acc += (g * n).to(OUT.dtype.element_ty).to(tl.float32)
    tl.store(OUT + rows[:, None] * sO + d[None, :], (acc / HC_F).to(OUT.dtype.element_ty), mask=live)


def blocks_fold(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor) -> bool:
    """Whether `mix_block` serves this site: PREFILL_ROWS rows or more in BF16, each operand packed along its last
    dimension."""
    return (normed.shape[0] >= PREFILL_ROWS and normed.dtype == down_inject.dtype == up.dtype == torch.bfloat16
            and normed.stride(1) == 1 and down_inject.stride(1) == 1 and up.stride(1) == 1)


def block_tiles(rows: int) -> dict:
    """mix_block's tiles for `rows` rows: {"down": DOWN_TILES' entry, or None for cuBLAS and `_gates`; "up": ...}."""
    down = None
    for least, tile in DOWN_TILES:
        if rows >= least:
            down = tile
    return {"down": down, "up": UP_BLOCK_TILE}


def narrow_tail(n: int, block_n: int) -> int:
    """The last column block's width when it is narrower than `block_n` (its live columns rounded up to a power of two,
    16 at least -- the injection's 4 past the mixer's 320 in 64-wide blocks: 16, not 64); 0 when it is not."""
    tail = max(16, triton.next_power_of_2(n - (triton.cdiv(n, block_n) - 1) * block_n))
    return tail if tail < block_n else 0


def down_gates_block(normed, down_inject, gates, inj, hc: int, *, inject: bool, tile, norm=None) -> None:
    """`_down_gates_rows` at `tile` (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, split): the gates [N, r] and, with
    `inject`, the injection [N, hc] (`inj`; `gates` again without one) stored from down(+inject) of the rows. `norm`:
    (scale [N, hc] FP32, w [hc*H]) -- `normed` is then the streams, normalised as the tiles are read (mix_block)."""
    rows, width = normed.shape
    n = down_inject.shape[0]
    bm, bn, bk, warps, stages, split = tile
    grid = (triton.cdiv(n, bn), triton.cdiv(rows, bm), split)
    if split > 1:
        if grid[0] * grid[1] > skinny_gemv.MAX_BLOCKS:
            raise ValueError(f"a split down tile over {rows} rows needs {grid[0] * grid[1]} arrival words; a device has "
                             f"{skinny_gemv.MAX_BLOCKS}")
        part = torch.empty(split, rows, n, device=normed.device, dtype=torch.float32)
        locks = skinny_gemv.prepare(normed.device)
    else:
        part = locks = gates
    scale, w = (gates, gates) if norm is None else norm
    hid = width // hc
    if norm is not None and hid % bk:
        raise ValueError(f"a down tile that normalises the streams reads K tiles inside one stream: {bk} into {hid}")
    _down_gates_rows[grid](normed, down_inject, gates, inj, part, locks, scale, w, rows, n, width, normed.stride(0),
                           down_inject.stride(0), gates.stride(0), inj.stride(0), scale.stride(0), float(hc),
                           R=gates.shape[1], HC=hc, WITH_INJECT=inject, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                           TAIL=narrow_tail(n, bn), SPLIT=split, NORM_IN=norm is not None, HID=hid,
                           FP32_DOT=not normed.is_cuda, num_warps=warps, num_stages=stages)


def mix_block(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor, hc: int, *,
              inject: bool = True, tiles=None, norm=None) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """`mix` for a prefill step's rows (the module docstring): (mixed [N, H], injection [N, hc] or None). `tiles`:
    `block_tiles`' form, for a probe or a test; the table's for the rows otherwise. `norm`: (scale [N, hc] FP32 from
    `stream_scales`, the norm's weight [hc*H]) -- `normed` is then the streams themselves, and both launches normalise
    what they read with the stream norm's arithmetic: the same bytes, and the normalised streams never written."""
    hid = _check_streams(normed, hc)
    rank = up.shape[1]
    if up.shape != (normed.shape[1], rank) or down_inject.shape != (rank + (hc if inject else 0), normed.shape[1]):
        raise ValueError(f"a site mixes through down(+inject) [{rank}{' + ' + str(hc) if inject else ''}, "
                         f"{normed.shape[1]}] and up [{normed.shape[1]}, {rank}]")
    if not (normed.dtype == down_inject.dtype == up.dtype == torch.bfloat16) or normed.stride(1) != 1:
        raise ValueError("mix_block takes BF16 streams and weights, packed along their channels")
    rows = normed.shape[0]
    tiles = block_tiles(rows) if tiles is None else tiles
    gates = torch.empty(rows, rank, device=normed.device, dtype=normed.dtype)
    injection = torch.empty(rows, hc, device=normed.device, dtype=normed.dtype) if inject else None
    mixed = torch.empty(rows, hid, device=normed.device, dtype=normed.dtype)
    if not rows:
        return mixed, injection
    inj = gates if injection is None else injection
    if norm is not None:
        scale, w = norm
        if (scale.shape != (rows, hc) or scale.dtype != torch.float32 or scale.stride(1) != 1
                or w.shape != (normed.shape[1],)):
            raise ValueError("mix_block normalises through scale [N, hc] FP32 and the norm's weight [hc*H]")
        if tiles["down"] is None:
            raise ValueError("the streams normalise inside the down fold: a down tile, not cuBLAS's product")
    if tiles["down"] is None:
        di = torch.mm(normed, down_inject.t())
        _gates[(rows,)](di, gates, inj, di.stride(0), gates.stride(0), inj.stride(0), float(hc), R=rank,
                        BR=triton.next_power_of_2(rank), HC=hc, BH=triton.next_power_of_2(hc), WITH_INJECT=inject,
                        num_warps=4)
    else:
        down_gates_block(normed, down_inject, gates, inj, hc, inject=inject, tile=tiles["down"], norm=norm)
    up_mean_block(gates, up, normed, mixed, hc, tile=tiles["up"], norm=norm)
    return mixed, injection


def up_mean_block(gates, up, normed, mixed, hc: int, *, tile, norm=None) -> None:
    """`_up_mean_rows` at `tile` (BLOCK_M, BLOCK_D, BLOCK_K, warps, stages): the streams' mean [N, H] weighted by
    sigmoid(up(gates)) stored into `mixed`. `norm`: as down_gates_block's."""
    rows, hid = normed.shape[0], mixed.shape[1]
    scale, w = (gates, gates) if norm is None else norm
    bm, bd, bk, warps, stages = tile
    _up_mean_rows[(triton.cdiv(rows, bm), triton.cdiv(hid, bd))](
        gates, up, normed, mixed, scale, w, rows, gates.stride(0), up.stride(0), normed.stride(0), mixed.stride(0),
        scale.stride(0), float(hc), HID=hid, R=gates.shape[1], HC=hc, BLOCK_M=bm, BLOCK_D=bd, BLOCK_K=bk,
        NORM_IN=norm is not None, FP32_DOT=not normed.is_cuda, num_warps=warps, num_stages=stages)


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


def qualify(device, *, hc: int, hidden: int, rank: int, eps: float, dtype=torch.bfloat16,
            rows=(1, 5, 64, PREFILL_ROWS), band_max: float = 5e-2, band_rms: float = 2e-2, seed: int = 0) -> dict:
    """Hold the lane to engine/modules/hyper_connection.gated_residual on `device`, with random weights at the model's
    widths: two sites joined by a leave with its norm, a leave without one, and a closing mixer -- and the same two
    sites whole through `site` (at PREFILL_ROWS the streams normalised inside mix_block's launches). Raises when an output
    drifts past `band_max` (largest error / largest magnitude) or `band_rms` (`drift`): bounds a few BF16 steps wide,
    so rounding order passes and arithmetic does not; returns the worst (max, rms) seen per output."""
    from engine.modules.hyper_connection import gated_residual
    gen = torch.Generator(device="cpu").manual_seed(seed)
    width = hc * hidden

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=dtype)

    worst = {k: (0.0, 0.0) for k in ("enter", "inject", "leave_norm", "leave", "close", "site", "site_inject",
                                     "site_close")}
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
        ref_close = gated_residual(ref_h, norm_c, down_c, up_c, None, hc, eps)
        closed, _ = mix(normed, dc, up_c, hc, inject=False)
        note("close", closed, ref_close)
        mixed, injection = site(h.clone(), None, None, norm_w, eps, hc, di, up)
        note("site", mixed, ref_mixed)
        note("site_inject", injection, ref_inj)
        note("site_close", site(h.clone(), out, ref_inj, norm_c, eps, hc, dc, up_c, inject=False)[0], ref_close)
    bad = {k: v for k, v in worst.items() if v[0] > band_max or v[1] > band_rms}
    if bad:
        raise RuntimeError(f"gated residual lane drifts from engine/modules/hyper_connection.gated_residual beyond "
                           f"max {band_max:g} / rms {band_rms:g}: {bad}")
    return worst


__all__ = ["DECODE_ROWS", "pack_down_inject", "norm_streams", "leave", "leave_norm", "stream_scales", "site", "mix",
           "folds", "mix_rows", "mix_block", "block_tiles", "drift", "blame", "qualify"]
