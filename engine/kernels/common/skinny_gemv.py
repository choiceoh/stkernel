"""A BF16 x @ W.T for a handful of rows that reads W once (kernels, common).

A decode step multiplies 1-16 rows by weights it reads once a step, so such a product's floor is the weight's bytes
over the memory's bandwidth (273 GB/s on a GB10). cuBLAS reaches it for one row (a gemv) and not for more: at Qwen3.8's
router [513, 2560] it took 27.9 / 33.6 / 35.7 us for 4 / 8 / 16 rows (94-74 GB/s) where this kernel takes 14.3 / 14.2 /
14.1 (184 GB/s), and at the mixers' down projection [324, 10240] 34.5-35.1 us against 31.1-31.7 (213 GB/s). At one row
the two tie (13.8 against 14.0, 31.0 against 31.1), and at the mixers' up projection [10240, 320] they tie at every row
count (30.3-31.3 against 29.1-30.8) -- so `linear_rows` takes 2..16 rows of the shapes in CONFIGS and hands everything
else to torch.mm: the same product in BF16 with FP32 accumulation, rounded once; only the order of the sums differs.
The MTP head's BF16 projections are the exception to "one row ties": there cuBLAS's gemv reads 161-172 GB/s against this
kernel's 213-257 (in_proj 132.0 against 96.8 us, fc 81.5 against 58.2, o_proj 48.1 against 36.8, shared down 4.8 against
3.2) -- and one row is what a C=1 draft multiplies -- so those shapes take one row too (ONE_ROW); the shared gate_up,
whose one row cuBLAS still wins (8.7 against 9.6), does not (ticket q38gemv-0919e).
The mixers do not call it: engine/kernels/gated_residual.mix_rows folds both of a site's products, with the launch after
each, into two kernels over `rows_dot` and `split_sum` here, at the down projection's tile in CONFIGS.
(probes/engine_qwen38_gemv, ticket q38gemv-0919c, 2026-09-19: CUDA graphs of 16 calls, interleaved, weights rotated
over 64 MB, production idle beside it; medians of nine.)

One launch: the rows pad to 16 for a tensor-core dot, a program keeps one [16, BLOCK_N] FP32 accumulator and walks its
share of K in BLOCK_K tiles. A narrow output over a long K splits K into SPLIT programs a column block; each stores its
FP32 partial and counts itself in on the block's arrival word, and the last to arrive sums the partials in split order
(the same order every call: a result does not depend on which program came last), rounds once, and resets the word to
zero for the next call. The arrival words are one int32 buffer a device, zeroed once outside any capture (`prepare`);
calls on one stream are ordered, so they share it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MIN_ROWS, MAX_ROWS = 2, 16         # one row is cuBLAS's gemv, as fast (but ONE_ROW's); past 16 a GEMM's tiles serve
MAX_BLOCKS = 4096                  # arrival words a device: column blocks of the widest split output

# (N, K) of W -> (BLOCK_N, BLOCK_K, SPLIT, warps, stages), the fastest of twelve summed over 1/4/8/16 rows; absent ->
# torch.mm. The sweep's median is 5-9% behind its best (15.0 against 14.3 us at the router's 4 rows, 32.4 against 31.1
# at the down projection's): past the tile, the weight's read is the time.
CONFIGS = {
    (513, 2560): (16, 256, 1, 4, 3),      # Qwen3.8's router and shared gate (512 experts + 1): 33 programs
    (324, 10240): (16, 256, 4, 4, 3),     # its mixers' down + inject (rank 320 + hc 4): 21 blocks x 4 splits
    (320, 10240): (16, 256, 4, 4, 3),     # its closing mixers' down (no inject)
    # the MTP head's dense projections at its BF16 precision (net.linear with no dense lane), a rank's shard
    (4224, 2560): (16, 256, 1, 4, 3),     # attention in: the rank's query+gate, k, v, index
    (2560, 1536): (32, 256, 1, 4, 3),     # attention out (split by heads along its input)
    (2560, 2560): (64, 256, 1, 8, 3),     # fc_embedding and fc_hidden (whole)
    (320, 2560): (16, 128, 4, 4, 3),      # shared expert gate | up: 1.16-1.19x at 4-16 rows
    (2560, 160): (64, 256, 1, 4, 1),      # shared expert down: one masked K tile
}

# Shapes whose ONE row this kernel takes as well: where cuBLAS's one-row gemv is behind it (the MTP head's, 1.31-1.49x).
ONE_ROW = frozenset({(4224, 2560), (2560, 1536), (2560, 2560), (2560, 160)})

_LOCKS: "dict[torch.device, torch.Tensor]" = {}


@triton.jit
def rows_dot(X, W, sx, sw, rows, cols, M, N, k0, k1, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             FP32_DOT: tl.constexpr):
    """[16, BLOCK_N] FP32: X's rows (< M, padded to 16) times W's rows `cols` (< N) over K in [k0, k1), BLOCK_K at a
    time -- one read of each weight tile for every row. The kernels that fold a product into their store call it."""
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k1, BLOCK_K):
        ks = k + tl.arange(0, BLOCK_K)
        kmask = ks < k1
        x = tl.load(X + rows[:, None] * sx + ks[None, :], mask=(rows[:, None] < M) & kmask[None, :], other=0.0)
        w = tl.load(W + cols[:, None] * sw + ks[None, :], mask=(cols[:, None] < N) & kmask[None, :], other=0.0)
        if FP32_DOT:                                              # the interpreter reads a BF16 dot's bits as integers
            x, w = x.to(tl.float32), w.to(tl.float32)
        acc += tl.dot(x, tl.trans(w))
    return acc


@triton.jit
def split_span(K, pid_k, SPLIT: tl.constexpr, BLOCK_K: tl.constexpr):
    """[k0, k1) of split `pid_k`: whole tiles a split, so none straddles two programs."""
    span = tl.cdiv(tl.cdiv(K, SPLIT), BLOCK_K) * BLOCK_K
    k0 = pid_k * span
    return k0, tl.minimum(k0 + span, K)


@triton.jit
def split_sum(acc, PART, LOCKS, pid_n, pid_k, rows, cols, M, N, SPLIT: tl.constexpr):
    """(total, last): this program's partial stored and counted in on its column block's arrival word; for the last
    to arrive, the block's partials summed in split order and the word reset -- `total` is meaningful where `last`."""
    keep = (rows[:, None] < M) & (cols[None, :] < N)
    at = rows[:, None] * N + cols[None, :]
    tl.store(PART + pid_k * M * N + at, acc, mask=keep)
    arrived = tl.atomic_add(LOCKS + pid_n, 1, sem="acq_rel")      # the partial above is visible to whoever sums
    last = arrived == SPLIT - 1
    total = tl.zeros_like(acc)
    if last:
        for s in range(SPLIT):
            total += tl.load(PART + s * M * N + at, mask=keep, other=0.0, cache_modifier=".cg")
        tl.atomic_xchg(LOCKS + pid_n, 0)
    return total, last


@triton.jit
def _skinny_gemv_kernel(X, W, OUT, PART, LOCKS, M, N, K, sx, sw, so, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        SPLIT: tl.constexpr, FP32_DOT: tl.constexpr):
    pid_n, pid_k = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, 16)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0, k1 = split_span(K, pid_k, SPLIT, BLOCK_K)
    acc = rows_dot(X, W, sx, sw, rows, cols, M, N, k0, k1, BLOCK_N, BLOCK_K, FP32_DOT)
    keep = (rows[:, None] < M) & (cols[None, :] < N)
    if SPLIT == 1:
        tl.store(OUT + rows[:, None] * so + cols[None, :], acc.to(OUT.dtype.element_ty), mask=keep)
    else:
        total, last = split_sum(acc, PART, LOCKS, pid_n, pid_k, rows, cols, M, N, SPLIT)
        if last:
            tl.store(OUT + rows[:, None] * so + cols[None, :], total.to(OUT.dtype.element_ty), mask=keep)


def prepare(device) -> torch.Tensor:
    """The device's arrival words, zeroed -- before a graph captures a split call (a buffer made inside a capture
    would live in that graph's pool)."""
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    locks = _LOCKS.get(device)
    if locks is None:
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("skinny_gemv.prepare(device) before a graph captures a split GEMV")
        locks = _LOCKS[device] = torch.zeros(MAX_BLOCKS, dtype=torch.int32, device=device)
    return locks


def gemv(x: torch.Tensor, w: torch.Tensor, cfg: "tuple[int, int, int, int, int]",
         out: "torch.Tensor | None" = None) -> torch.Tensor:
    """x [M <= 16, K] BF16 @ w [N, K].T -> [M, N] BF16 under `cfg` (BLOCK_N, BLOCK_K, SPLIT, warps, stages); the last
    dimension of each operand contiguous."""
    m, k = x.shape
    n = w.shape[0]
    block_n, block_k, split, warps, stages = cfg
    if not 1 <= m <= MAX_ROWS or w.shape[1] != k:
        raise ValueError(f"skinny_gemv takes 1..{MAX_ROWS} rows of W's K: x {tuple(x.shape)}, w {tuple(w.shape)}")
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    blocks = triton.cdiv(n, block_n)
    if split > 1:
        if blocks > MAX_BLOCKS:
            raise ValueError(f"a split GEMV of {blocks} column blocks is past the {MAX_BLOCKS} arrival words")
        locks = prepare(x.device)
        partial = torch.empty(split, m, n, dtype=torch.float32, device=x.device)
    else:
        locks = partial = out
    _skinny_gemv_kernel[(blocks, split)](x, w, out, partial, locks, m, n, k, x.stride(0), w.stride(0), out.stride(0),
                                         BLOCK_N=block_n, BLOCK_K=block_k, SPLIT=split, FP32_DOT=not x.is_cuda,
                                         num_warps=warps, num_stages=stages)
    return out


def linear_rows(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T in BF16: this kernel for 2..16 rows of a CUDA shape `CONFIGS` names (1..16 of one in ONE_ROW),
    torch.mm otherwise."""
    shape = tuple(w.shape)
    cfg = CONFIGS.get(shape)
    least = 1 if shape in ONE_ROW else MIN_ROWS
    if (cfg is None or not x.is_cuda or not least <= x.shape[0] <= MAX_ROWS or x.dtype != torch.bfloat16
            or w.dtype != torch.bfloat16 or x.stride(-1) != 1 or w.stride(-1) != 1):
        return torch.mm(x, w.t())
    return gemv(x, w, cfg)


def qualify(device, rows=(1, 4, 16)) -> dict:
    """D3 before a boot serves: every shape in CONFIGS on `device`, held to the FP32 product -> {shape: largest error
    over the largest magnitude}; raises past 2^-6 (two BF16 steps), which a wrong tile, split or sum passes by orders.
    Twice a shape, so a split's arrival words are seen reset."""
    prepare(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    out = {}
    for (n, k), cfg in CONFIGS.items():
        w = (torch.randn(n, k, generator=gen) * 0.02).to(torch.bfloat16).to(device)
        worst = 0.0
        for m in rows:
            x = torch.randn(m, k, generator=gen).to(torch.bfloat16).to(device)
            ref = x.float() @ w.float().t()
            for _ in range(2):
                got = gemv(x, w, cfg).float()
                err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
                worst = max(worst, err) if err == err else float("inf")     # max(0.0, nan) is 0.0: a NaN would pass
        if not worst <= 2.0 ** -6:
            raise RuntimeError(f"skinny_gemv [{n}, {k}] {cfg}: error {worst:.2e} of the largest magnitude")
        out[f"{n}x{k}"] = round(worst, 6)
    return out


__all__ = ["CONFIGS", "MAX_ROWS", "MIN_ROWS", "ONE_ROW", "gemv", "linear_rows", "prepare", "qualify", "rows_dot",
           "split_span", "split_sum"]
