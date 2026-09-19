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

From LEAVE_DOWN_TILES' rows the leave and the down fold are ONE launch (`leave_down_block`, `_leave_down_rows`): a
program is a block of rows of one stream, a row block's hc programs adjacent. It leaves the output into its rows and
sums their squares (`stream_scales`' arithmetic, BLOCK_K channels at a time -- the scale to a few FP32 ulps), then reads
its rows back out of L2, normalises them ONCE and multiplies them into every column block of its stream's K slice of
down(+inject) at the same time (five 64-wide blocks and a 16-wide tail for the mixer's 324 columns); the stream's FP32
partial goes to a scratch, and the last of the row block's hc programs to arrive sums the four in stream order and
stores the gates. Against `stream_scales` and then the down fold: the streams read from DRAM once instead of twice
(84 MB at 4,096 rows), the A tile transformed once instead of once a column block, and the MMA overlapped with the
leave's memory traffic across the programs resident on an SM. The gates come out within the oracle's band, not byte
for byte the two launches' (the sum of squares and the K sum associate differently); the leave's rows are byte for byte
the leave's. `site` then finishes with `up_mean_block` over the streams and the scales.

The stream launches (`_leave_norm`, `_norm_streams`) run one program a (row, stream) over the grid (hc, rows): a row's
hc programs back to back, so the output row they all add (21 MB at 4,096 rows) is read from DRAM once and the row's
streams (20 KB, contiguous) are walked in one go. Over (rows, hc) -- the grid until 2026-09-20 -- the four programs of
a row were `rows` programs apart, about 60 MB of streams between them at 4,096 rows, past the L2, and the output was
read from DRAM once a stream: a quarter of the leave's traffic (252 MB against 189 at 4,096 rows). Either order runs
the same programs on the same elements, so the bytes are the same (`_stream_grid`; probes/engine_qwen38_stream_order).
On a GB10 (q38streamorder-0920a, the minima of 21 interleaved rounds beside a training job) the prefill site's leave
(`stream_scales`) went 1,138 -> 792 us at 4,096 rows (239 GB/s, 87% of the memory's), 545 -> 396 at 2,048, 151 -> 126
at 1,024, 52 -> 49 at 512; `leave_norm` 1,678 -> 1,159 at 4,096 rows; `norm_streams`, which adds no output, was
unchanged (measurements/qwen38_stream_order_20260920).

On a GB10 (eight sites a graph, interleaved; beside production, so the minima of nine rounds): up and the mean 1,703 ->
815 us a site at 4,096 rows (x2.09), x1.66 at 2,048, x1.65 at 1,024, x1.39 at 512 (q38sitecmp-0919a, an idle GPU) -- and
the output byte for byte cuBLAS's up with `_mix_mean`. The down projection and the gates over the normalised streams
(q38sitecmp-0919d): 584 -> 532 us at 4,096 rows at 128 x 128 x 64, 298 -> 268 at 2,048, 168 -> 164 at 1,024 at 128 x 64,
86 -> 102 at 512 -- at 4,096 rows the MMA's, 27.2 GFLOP padded to 384 columns at about 51 TFLOPS. Normalising the
streams inside costs the down fold its A tile's transform in every column block's program -- two fp32 products and two
conversions an element, the MMA's own order of time: 567 -> 849 us at 128 x 128 x 64, 570 -> 776 at the table's
256 x 64 x 64 (q38sitenorm-0919a) -- and the up fold 8%. The leave saves more: 1,554 -> 1,148 us at 4,096 rows, 916 ->
532 at 2,048 (q38sitewhole-0919a/b), so the site is still ahead at every row count: about 17 us at 512 rows, 130 at
1,024 and 4,096, 260 at 2,048.

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
# at every row. The fastest with the streams normalised inside (`site`'s way, q38sitenorm-0919a): a tile the normalised
# streams favour (128 x 128 x 64: 567 us at 4,096 rows) pays the most for normalising three column blocks' A over (849).
DOWN_TILES = ((512, (64, 128, 64, 4, 3, 2)), (2048, (256, 64, 64, 8, 3, 1)))
UP_BLOCK_TILE = (64, 64, 32, 4, 4)
# The fused leave + down fold's tile (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages) by the rows it serves from, as DOWN_TILES
# (`leave_down_block`; probes/engine_qwen38_leave_down). Rows short of every entry leave with `stream_scales` and fold
# with mix_block's two launches.
LEAVE_DOWN_TILES = ((512, (32, 64, 32, 8, 3)),)     # a stage holds the A tile and six W tiles: 64-deep at 3 stages is 111 KB
LEAVE_DOWN_BLOCKS = 5                       # full column blocks the fused kernel unrolls at most (the mixer's 320 in 64s), plus a tail
UP_TILE = (32, 64, 4, 3)                    # up_mean's BLOCK_D, BLOCK_K, warps, stages (the best of five, q38site-0919a)


@triton.jit
def _norm_streams(X, W, OUT, sX, sO, EPS, HID: tl.constexpr, BD: tl.constexpr, SCALE_ONLY: tl.constexpr,
                  ROWS_FIRST: tl.constexpr):
    # SCALE_ONLY: OUT is [N, hc] FP32, the stream's scale and not the normalised stream (`stream_scales`)
    # ROWS_FIRST: the grid is (rows, hc), a stream's rows adjacent; else (hc, rows), a row's streams (`_stream_grid`)
    if ROWS_FIRST:
        r = tl.program_id(0)
        s = tl.program_id(1)
    else:
        s = tl.program_id(0)
        r = tl.program_id(1)
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
                NORM: tl.constexpr, PDL: tl.constexpr, PREFETCH: tl.constexpr, SCALE_ONLY: tl.constexpr,
                ROWS_FIRST: tl.constexpr):
    # SCALE_ONLY: NORMED is [N, hc] FP32, the stream's scale and not the normalised stream (`stream_scales`)
    # ROWS_FIRST: the grid is (rows, hc), a stream's rows adjacent; else (hc, rows), a row's streams (`_stream_grid`)
    if ROWS_FIRST:
        r = tl.program_id(0)
        s = tl.program_id(1)
    else:
        s = tl.program_id(0)
        r = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    if NORM and not SCALE_ONLY:
        w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)   # immutable: read while the sum is still in flight
    if PREFETCH:
        # this program's index in the launch, whichever axis is the rows'
        if ROWS_FIRST:
            p = r * tl.num_programs(1) + s
        else:
            p = r * tl.num_programs(0) + s
        _prefetch_l2(NEXT, SECTORS, p, tl.num_programs(0) * tl.num_programs(1), BD)
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


# Probe hook (probes/engine_qwen38_stream_order.py): the stream launches' grid order forced when set -- "rows", a stream's
# rows adjacent, the grid until 2026-09-20 -- the rule when None. Read when a launch is made, so a captured graph keeps
# the order it was captured with. Nothing served sets it.
_STREAM_GRID_OVERRIDE = None


def _stream_grid(rows: int, hc: int) -> "tuple[tuple[int, int], bool]":
    """(grid, ROWS_FIRST) of a launch of one program a (row, stream) -- `_leave_norm`, `_norm_streams`. The rule is
    (hc, rows): program p of the launch order is stream p % hc of row p // hc, so a row's hc programs run back to back,
    the output row they share read from DRAM once and its streams (hc*H contiguous) walked in one go. Over (rows, hc)
    the programs of one row are `rows` programs apart -- at 4,096 rows about 60 MB of streams between them, past the
    L2 -- and the output is read from DRAM once a stream. The same programs on the same elements either way: the same
    bytes."""
    forced = _STREAM_GRID_OVERRIDE
    if forced is None:
        return (hc, rows), False
    if forced != "rows":
        raise ValueError('_STREAM_GRID_OVERRIDE is None (a row\'s streams adjacent) or "rows" (a stream\'s rows adjacent)')
    return (rows, hc), True


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
        grid, rows_first = _stream_grid(h.shape[0], hc)
        _norm_streams[grid](h, w, out, h.stride(0), out.stride(0), eps, HID=hid, BD=triton.next_power_of_2(hid),
                            SCALE_ONLY=False, ROWS_FIRST=rows_first, num_warps=_warps(hid))
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
        grid, rows_first = _stream_grid(h.shape[0], hc)
        _norm_streams[grid](h, h, scale, h.stride(0), scale.stride(0), eps, HID=hid, BD=triton.next_power_of_2(hid),
                            SCALE_ONLY=True, ROWS_FIRST=rows_first, num_warps=_warps(hid))
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
        grid, rows_first = _stream_grid(h.shape[0], hc)
        _leave_norm[grid](h, out, inject, w if norm else h, normed, prefetch if sectors else h,
                          h.stride(0), out.stride(0), inject.stride(0), normed.stride(0), eps, sectors,
                          HID=hid, BD=triton.next_power_of_2(hid), NORM=norm, PDL=pdl, PREFETCH=sectors > 0,
                          SCALE_ONLY=scale_only, ROWS_FIRST=rows_first, num_warps=_warps(hid), launch_pdl=pdl)
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
    hid = h.shape[1] // hc
    if h.is_cuda and blocks_fold(h, down_inject, up):
        if tiles["leave_down"] is not None and hid % tiles["leave_down"][2] == 0:
            # the leave and the down fold one launch, the scales and the gates out of it (leave_mix_block)
            return leave_mix_block(h, out, injection, w, eps, hc, down_inject, up, inject=inject, tiles=tiles, pdl=pdl)
        if tiles["down"] is not None and hid % tiles["down"][2] == 0:
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
    down = leave_down = None
    for least, tile in DOWN_TILES:
        if rows >= least:
            down = tile
    for least, tile in LEAVE_DOWN_TILES:
        if rows >= least:
            leave_down = tile
    return {"down": down, "up": UP_BLOCK_TILE, "leave_down": leave_down}


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
    rows, rank = normed.shape[0], up.shape[1]
    tiles = block_tiles(rows) if tiles is None else tiles
    gates, injection, inj, mixed = _mix_block_buffers(normed, down_inject, up, hc, inject)
    if not rows:
        return mixed, injection
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


def _mix_block_buffers(normed, down_inject, up, hc: int, inject: bool):
    """(gates [N, r], injection [N, hc] or None, `inj` -- the injection, or the gates again without one -- and
    mixed [N, H]) for a block mixer over `normed`, the operands' shapes and dtypes checked."""
    hid = _check_streams(normed, hc)
    rank = up.shape[1]
    if up.shape != (normed.shape[1], rank) or down_inject.shape != (rank + (hc if inject else 0), normed.shape[1]):
        raise ValueError(f"a site mixes through down(+inject) [{rank}{' + ' + str(hc) if inject else ''}, "
                         f"{normed.shape[1]}] and up [{normed.shape[1]}, {rank}]")
    if not (normed.dtype == down_inject.dtype == up.dtype == torch.bfloat16) or normed.stride(1) != 1:
        raise ValueError("mix_block takes BF16 streams and weights, packed along their channels")
    rows = normed.shape[0]
    gates = torch.empty(rows, rank, device=normed.device, dtype=normed.dtype)
    injection = torch.empty(rows, hc, device=normed.device, dtype=normed.dtype) if inject else None
    mixed = torch.empty(rows, hid, device=normed.device, dtype=normed.dtype)
    return gates, injection, (gates if injection is None else injection), mixed


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


@triton.jit
def _w_tile(DW, sW, cols, kk, N, FP32_DOT: tl.constexpr):
    # DW's rows `cols` (< N) at the K positions kk, transposed for the dot
    w = tl.load(DW + cols[:, None] * sW + kk[None, :], mask=cols[:, None] < N, other=0.0)
    if FP32_DOT:
        w = w.to(tl.float32)
    return tl.trans(w)


@triton.jit
def _partial_store(PART, acc, rows, cols, live, N):
    tl.store(PART + rows[:, None] * N + cols[None, :], acc, mask=live[:, None] & (cols[None, :] < N))


@triton.jit
def _partials_sum(PART, MN, rows, cols, live, N, HC: tl.constexpr, BLOCK_M: tl.constexpr, WIDTH: tl.constexpr):
    # the HC streams' partials of one column block, summed in stream order from zero
    total = tl.zeros((BLOCK_M, WIDTH), dtype=tl.float32)
    keep = live[:, None] & (cols[None, :] < N)
    for j in range(HC):
        total += tl.load(PART + j * MN + rows[:, None] * N + cols[None, :], mask=keep, other=0.0, cache_modifier=".cg")
    return total


@triton.jit
def _leave_down_rows(H, OUT, INJ, W, SC, DW, MIX, IJ, PART, LOCKS, M, N, sH, sO, sI, sS, sM, sIJ, sW, EPS, HC_F,
                     HID: tl.constexpr, HC: tl.constexpr, R: tl.constexpr, WITH_INJECT: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, FULL: tl.constexpr,
                     TAIL: tl.constexpr, TAIL_W: tl.constexpr, LEAVE: tl.constexpr, PDL: tl.constexpr,
                     FP32_DOT: tl.constexpr):
    # One program: BLOCK_M rows of stream s, a row block's HC programs adjacent (the sum's rows shared through L2).
    # Pass 1 -- the leave (OUT x INJ[:, s] into the stream's channels in place, `_leave_norm`'s rounding) and the
    # stream's sum of squares, BLOCK_K channels at a time; its scale to SC[:, s]. Pass 2 -- the rows read back (L2's:
    # this program just wrote them), normalised with that scale and 1 + W as `_tile_dot_normed` does -- once, for every
    # column block at the same time: FULL blocks of BLOCK_N and a TAIL-wide last one -- times this stream's K slice of
    # DW. The stream's FP32 partial [BLOCK_M, N] goes to PART[s]; the last of a row block's HC programs to arrive sums
    # the partials in stream order and stores the gates (`_gate_store`).
    s = tl.program_id(0)
    pid_m = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    live = rows < M
    if PDL:
        # the sum (OUT) is the primary's; the streams and the injection are read after the wait as well
        tl.extra.cuda.gdc_wait()
    if LEAVE:
        g = tl.load(INJ + rows * sI + s, mask=live, other=0.0).to(tl.float32)
    sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k in range(0, HID, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        at = rows[:, None] * sH + (s * HID + ks)[None, :]
        h = tl.load(H + at, mask=live[:, None], other=0.0)
        if LEAVE:
            o = tl.load(OUT + rows[:, None] * sO + ks[None, :], mask=live[:, None], other=0.0).to(tl.float32)
            delta = (o * g[:, None]).to(h.dtype)                  # the product rounds, then the sum does
            h = (h.to(tl.float32) + delta.to(tl.float32)).to(h.dtype)
            tl.store(H + at, h, mask=live[:, None])
        x = h.to(tl.float32)
        sq += tl.sum(x * x, axis=1)
    scale = tl.rsqrt(sq / HID + EPS)
    tl.store(SC + rows * sS + s, scale, mask=live)
    if LEAVE:
        tl.debug_barrier()                                        # the rows stored above are read back by other threads
    c0 = tl.arange(0, BLOCK_N)
    c1 = BLOCK_N + c0
    c2 = 2 * BLOCK_N + c0
    c3 = 3 * BLOCK_N + c0
    c4 = 4 * BLOCK_N + c0
    ct = FULL * BLOCK_N + tl.arange(0, TAIL_W)
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc4 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acct = tl.zeros((BLOCK_M, TAIL_W), dtype=tl.float32)
    for k in range(0, HID, BLOCK_K):
        kk = s * HID + k + tl.arange(0, BLOCK_K)
        x = tl.load(H + rows[:, None] * sH + kk[None, :], mask=live[:, None], other=0.0)
        nw = tl.load(W + kk).to(tl.float32)
        x = ((x.to(tl.float32) * scale[:, None]) * (1.0 + nw[None, :])).to(H.dtype.element_ty)
        if FP32_DOT:                                              # the interpreter reads a BF16 dot's bits as integers
            x = x.to(tl.float32)
        acc0 += tl.dot(x, _w_tile(DW, sW, c0, kk, N, FP32_DOT))
        if FULL > 1:
            acc1 += tl.dot(x, _w_tile(DW, sW, c1, kk, N, FP32_DOT))
        if FULL > 2:
            acc2 += tl.dot(x, _w_tile(DW, sW, c2, kk, N, FP32_DOT))
        if FULL > 3:
            acc3 += tl.dot(x, _w_tile(DW, sW, c3, kk, N, FP32_DOT))
        if FULL > 4:
            acc4 += tl.dot(x, _w_tile(DW, sW, c4, kk, N, FP32_DOT))
        if TAIL > 0:
            acct += tl.dot(x, _w_tile(DW, sW, ct, kk, N, FP32_DOT))
    MN = M * N
    mine = PART + s * MN
    _partial_store(mine, acc0, rows, c0, live, N)
    if FULL > 1:
        _partial_store(mine, acc1, rows, c1, live, N)
    if FULL > 2:
        _partial_store(mine, acc2, rows, c2, live, N)
    if FULL > 3:
        _partial_store(mine, acc3, rows, c3, live, N)
    if FULL > 4:
        _partial_store(mine, acc4, rows, c4, live, N)
    if TAIL > 0:
        _partial_store(mine, acct, rows, ct, live, N)
    tl.debug_barrier()                                            # every thread's partial is stored before one arrives
    arrived = tl.atomic_add(LOCKS + pid_m, 1, sem="acq_rel")
    if arrived == HC - 1:
        _gate_store(_partials_sum(PART, MN, rows, c0, live, N, HC, BLOCK_M, BLOCK_N), rows, c0, M, MIX, IJ, sM, sIJ,
                    HC_F, R, HC, WITH_INJECT)
        if FULL > 1:
            _gate_store(_partials_sum(PART, MN, rows, c1, live, N, HC, BLOCK_M, BLOCK_N), rows, c1, M, MIX, IJ, sM,
                        sIJ, HC_F, R, HC, WITH_INJECT)
        if FULL > 2:
            _gate_store(_partials_sum(PART, MN, rows, c2, live, N, HC, BLOCK_M, BLOCK_N), rows, c2, M, MIX, IJ, sM,
                        sIJ, HC_F, R, HC, WITH_INJECT)
        if FULL > 3:
            _gate_store(_partials_sum(PART, MN, rows, c3, live, N, HC, BLOCK_M, BLOCK_N), rows, c3, M, MIX, IJ, sM,
                        sIJ, HC_F, R, HC, WITH_INJECT)
        if FULL > 4:
            _gate_store(_partials_sum(PART, MN, rows, c4, live, N, HC, BLOCK_M, BLOCK_N), rows, c4, M, MIX, IJ, sM,
                        sIJ, HC_F, R, HC, WITH_INJECT)
        if TAIL > 0:
            _gate_store(_partials_sum(PART, MN, rows, ct, live, N, HC, BLOCK_M, TAIL_W), rows, ct, M, MIX, IJ, sM,
                        sIJ, HC_F, R, HC, WITH_INJECT)
        tl.atomic_xchg(LOCKS + pid_m, 0)


def leave_down_block(h, out, injection, w, eps: float, hc: int, down_inject, gates, inj, *, inject: bool, tile,
                     pdl: bool = False) -> torch.Tensor:
    """`_leave_down_rows` at `tile` (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages): `out` (with `injection`, leave_norm's)
    left into the streams h in place -- or, `out` None, the streams as they are -- and the gates [N, r] and, with
    `inject`, the injection [N, hc] (`inj`; `gates` again without one) stored from down(+inject) of the streams
    normalised with `w`; returns each stream's scale [N, hc] FP32 (`stream_scales`' value to a few FP32 ulps: the sum of
    squares is BLOCK_K channels at a time), which the up fold normalises with. `pdl`: as leave_norm's."""
    hid = _check_streams(h, hc)
    rows, width = h.shape
    n = down_inject.shape[0]
    bm, bn, bk, warps, stages = tile
    if w.shape != (width,):
        raise ValueError("the stream norm's weight covers every stream's channels")
    if hid % bk:
        raise ValueError(f"the fused leave reads K tiles inside one stream: {bk} into {hid}")
    if down_inject.dtype != h.dtype or down_inject.stride(1) != 1 or down_inject.shape[1] != width:
        raise ValueError(f"down(+inject) is [{n}, {width}] packed in the streams' dtype")
    if out is not None:
        if out.shape != (rows, hid) or injection.shape != (rows, hc):
            raise ValueError(f"a leave takes the output [N, {hid}] and the injection [N, {hc}] for {rows} rows")
        if out.dtype != h.dtype or injection.dtype != h.dtype or out.stride(1) != 1 or injection.stride(1) != 1:
            raise ValueError("the output and the injection are packed rows in the streams' dtype")
    if type(pdl) is not bool:
        raise ValueError("pdl is a declared boolean")
    tail = narrow_tail(n, bn)
    full = triton.cdiv(n, bn) - (1 if tail else 0)
    if not 1 <= full <= LEAVE_DOWN_BLOCKS:
        raise ValueError(f"the fused leave unrolls up to {LEAVE_DOWN_BLOCKS} column blocks of {bn} and a tail; "
                         f"{n} columns need {full}")
    blocks = triton.cdiv(rows, bm)
    if blocks > skinny_gemv.MAX_BLOCKS:
        raise ValueError(f"a fused leave over {rows} rows needs {blocks} arrival words; a device has "
                         f"{skinny_gemv.MAX_BLOCKS}")
    scale = torch.empty(rows, hc, device=h.device, dtype=torch.float32)
    if not rows:
        return scale
    part = torch.empty(hc, rows, n, device=h.device, dtype=torch.float32)
    locks = skinny_gemv.prepare(h.device)
    leave = out is not None
    pdl = pdl and h.device.type == "cuda"
    _leave_down_rows[(hc, blocks)](h, out if leave else h, injection if leave else h, w, scale, down_inject, gates,
                                   inj, part, locks, rows, n, h.stride(0), out.stride(0) if leave else 0,
                                   injection.stride(0) if leave else 0, scale.stride(0), gates.stride(0),
                                   inj.stride(0), down_inject.stride(0), eps, float(hc), HID=hid, HC=hc,
                                   R=gates.shape[1], WITH_INJECT=inject, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, FULL=full,
                                   TAIL=tail, TAIL_W=tail or bn, LEAVE=leave, PDL=pdl, FP32_DOT=not h.is_cuda,
                                   num_warps=warps, num_stages=stages, launch_pdl=pdl)
    return scale


def leave_mix_block(h: torch.Tensor, out: "torch.Tensor | None", injection: "torch.Tensor | None", w: torch.Tensor,
                    eps: float, hc: int, down_inject: torch.Tensor, up: torch.Tensor, *, inject: bool = True,
                    tiles=None, pdl: bool = False) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """A prefill step's site in two launches (the module docstring): `leave_down_block` -- the leave, the scales, the
    gates -- then `up_mean_block` over the streams and those scales. (mixed [N, H], injection [N, hc] or None); h left
    into in place. `tiles`: `block_tiles`' form with a "leave_down" entry."""
    tiles = block_tiles(h.shape[0]) if tiles is None else tiles
    if tiles.get("leave_down") is None:
        raise ValueError("leave_mix_block needs a fused leave tile for its rows (LEAVE_DOWN_TILES)")
    gates, injection_out, inj, mixed = _mix_block_buffers(h, down_inject, up, hc, inject)
    if not h.shape[0]:
        return mixed, injection_out
    scale = leave_down_block(h, out, injection, w, eps, hc, down_inject, gates, inj, inject=inject,
                             tile=tiles["leave_down"], pdl=pdl)
    up_mean_block(gates, up, h, mixed, hc, tile=tiles["up"], norm=(scale, w))
    return mixed, injection_out


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
           "folds", "mix_rows", "mix_block", "leave_down_block", "leave_mix_block", "block_tiles", "drift", "blame",
           "qualify"]
