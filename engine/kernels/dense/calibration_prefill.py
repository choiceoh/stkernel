"""Unmasked BF16 calibration, without materializing the full input in FP32.

Products use the input's original BF16 values and FP32 Tensor Core accumulation.
Only the order of the Gram sum changes. These statistics prepare a later boot's
packs; this boot's inference weights and all masked/decode observations stay put.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _gram(X, H, Armed, N, SX: tl.constexpr, SK: tl.constexpr,
          START: tl.constexpr, K: tl.constexpr,
          B: tl.constexpr = 32, R: tl.constexpr = 64):
    # One triangular tile computes both symmetric Gram contributions.
    if tl.load(Armed) != 0 and tl.program_id(0) >= tl.program_id(1):
        i = tl.program_id(0) * B + tl.arange(0, B)
        j = tl.program_id(1) * B + tl.arange(0, B)
        r = tl.arange(0, R)
        total = tl.zeros((B, B), tl.float32)
        # Reset the MMA accumulator every 256 rows. Carrying the large diagonal
        # through all 9K products loses low bits; add bounded partial sums in FP32.
        for group in range(tl.cdiv(N, 4 * R)):
            partial = tl.zeros((B, B), tl.float32)
            for sub in range(4):
                rows = (group * 4 + sub) * R + r
                a = tl.load(X + rows[None, :] * SX + (START + i[:, None]) * SK,
                            (rows[None, :] < N) & (i[:, None] < K), other=0)
                b = tl.load(X + rows[:, None] * SX + (START + j[None, :]) * SK,
                            (rows[:, None] < N) & (j[None, :] < K), other=0)
                partial = tl.dot(a, b, partial)
            total += partial
        ptr = H + i[:, None] * K + j[None, :]
        mask = (i[:, None] < K) & (j[None, :] < K)
        tl.store(ptr, tl.load(ptr, mask, other=0) + total, mask)
        if tl.program_id(0) != tl.program_id(1):
            mirror = H + j[:, None] * K + i[None, :]
            other_mask = (j[:, None] < K) & (i[None, :] < K)
            tl.store(mirror, tl.load(mirror, other_mask, other=0) + tl.trans(total), other_mask)


@triton.jit(do_not_specialize=["N"])
def _partial_peaks(X, Partial, Armed, N, SX: tl.constexpr, SK: tl.constexpr,
                   START: tl.constexpr, K: tl.constexpr,
                   C: tl.constexpr = 64, R: tl.constexpr = 128):
    if tl.load(Armed) != 0:
        cols = tl.program_id(0) * C + tl.arange(0, C)
        rows = tl.program_id(1) * R + tl.arange(0, R)
        values = tl.load(X + rows[:, None] * SX + (START + cols[None, :]) * SK,
                         (rows[:, None] < N) & (cols[None, :] < K), other=0).to(tl.float32)
        # Triton's max reduction may drop NaNs; Torch amax preserves them.
        bad = tl.sum((values != values).to(tl.int32), axis=0) != 0
        peaks = tl.where(bad, float("nan"), tl.max(tl.abs(values), axis=0))
        tl.store(Partial + tl.program_id(1) * K + cols, peaks, cols < K)


@triton.jit(do_not_specialize=["N", "NR"])
def _finish_peaks(Partial, Peaks, Rows, Armed, N, K: tl.constexpr,
                  NR, BR: tl.constexpr, C: tl.constexpr = 64):
    # All column blocks must make the same decision before block zero changes
    # Rows. The arm decision is materialized by the caller into one device flag.
    if tl.load(Armed) != 0:
        cols = tl.program_id(0) * C + tl.arange(0, C)
        groups = tl.arange(0, BR)
        partial = tl.load(Partial + groups[:, None] * K + cols[None, :],
                          (groups[:, None] < NR) & (cols[None, :] < K), other=0)
        old = tl.load(Peaks + cols, cols < K, other=0)
        bad = (old != old) | (tl.sum((partial != partial).to(tl.int32), axis=0) != 0)
        value = tl.where(bad, float("nan"), tl.maximum(old, tl.max(partial, axis=0)))
        tl.store(Peaks + cols, value, cols < K)
        if tl.program_id(0) == 0:
            tl.store(Rows, tl.load(Rows) + N)


@triton.jit
def _enabled(Armed, Rows, Enabled, TARGET: tl.constexpr):
    tl.store(Enabled, (tl.load(Armed) != 0) & (tl.load(Rows) < TARGET))


def observe(flat, armed, tiles, hessians, peaks, rows, target):
    """Called only for unmasked CUDA BF16 prefill. The arm is a device 0/1 scalar."""
    n = flat.shape[0]
    nr = triton.cdiv(n, 128)
    for key, start, width, hessian in tiles:
        # Stop a completed tile independently: an unvisited tile must not keep
        # every other matrix summing indefinitely while global complete waits.
        enabled = torch.empty((), device=flat.device, dtype=torch.int32)
        _enabled[(1,)](armed, rows[key], enabled, target)
        if hessian:
            # Larger output tiles reuse each BF16 input across twice as many
            # columns. Preserve the 256-row FP32 partial-sum boundaries.
            block = 64 if width >= 512 else 32
            _gram[(triton.cdiv(width, block), triton.cdiv(width, block))](
                flat, hessians[key], enabled, n, *flat.stride(), start, width,
                B=block, num_warps=4, num_stages=3)
        partial = torch.empty((nr, width), device=flat.device, dtype=torch.float32)
        _partial_peaks[(triton.cdiv(width, 64), nr)](
            flat, partial, enabled, n, *flat.stride(), start, width, num_warps=4)
        _finish_peaks[(triton.cdiv(width, 64),)](
            partial, peaks[key], rows[key], enabled, n, width, nr,
            # All supported long-prefill chunks share the reduction prepared
            # at boot. NR still masks the actual partial rows; padded rows are
            # zero, so maxima and NaN propagation are unchanged. Specializing
            # every tail's reduction width caused first-request compilation.
            max(256, triton.next_power_of_2(nr)), num_warps=4)
