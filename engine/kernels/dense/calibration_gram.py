"""Amortize small-row Gram updates without a host read or a lost input row.

Append, accumulate, advance are ordered on the caller's stream. The staging
plane has room for the threshold plus one maximum-sized call, so crossing the
threshold never overwrites an unconsumed row. A fixed persistent grid exits
immediately between flushes instead of launching one CTA per Hessian tile.
"""
import triton as tr
import triton.language as tl

from .calibration import GRAM_ROWS as ROWS


@tr.jit
def _observe(X, Mask, Buffer, Cursor, Armed, Count, Peaks,
             K: tl.constexpr, M: tl.constexpr, SX: tl.constexpr,
             MASKED: tl.constexpr, STAGE: tl.constexpr,
             BM: tl.constexpr, BC: tl.constexpr):
    if tl.load(Armed) != 0:
        r = tl.arange(0, BM)
        c = tl.program_id(0) * BC + tl.arange(0, BC)
        live = (r[:, None] < M) & (c[None, :] < K)
        if MASKED:
            weight = tl.load(Mask + r, r < M, 0).to(tl.float32)
        else:
            weight = (r < M).to(tl.float32)
        x = tl.load(X + r[:, None] * SX + c[None, :], live, 0).to(tl.float32)
        x = x * weight[:, None]
        if STAGE:
            offset = tl.load(Cursor)
            tl.store(Buffer + (offset + r[:, None]) * K + c[None, :], x, live)
        peak = tl.max(tl.abs(x), 0)
        old = tl.load(Peaks + c, c < K, 0)
        tl.store(Peaks + c, tl.maximum(old, peak), c < K)
        if tl.program_id(0) == 0:
            tl.store(Count, tl.load(Count) + tl.sum(weight, 0))


@tr.jit
def _gram(Buffer, H, Cursor, Armed, K: tl.constexpr, M: tl.constexpr,
          CAPACITY: tl.constexpr, FORCE: tl.constexpr, PROGRAMS: tl.constexpr, THRESHOLD: tl.constexpr,
          B: tl.constexpr = 32, R: tl.constexpr = 32):
    n = tl.load(Cursor) + M
    if (tl.load(Armed) != 0) & (n > 0) & (FORCE | (n >= THRESHOLD)):
        tiles: tl.constexpr = tr.cdiv(K, B)
        for tile in range(tl.program_id(0), tiles * tiles, PROGRAMS):
            i = (tile // tiles) * B + tl.arange(0, B)
            j = (tile % tiles) * B + tl.arange(0, B)
            total = tl.full((B, B), 0, tl.float32)
            for start in range(tr.cdiv(CAPACITY, R)):
                r = start * R + tl.arange(0, R)
                a = tl.load(Buffer + r[None, :] * K + i[:, None],
                            (r[None, :] < n) & (i[:, None] < K), 0)
                b = tl.load(Buffer + r[:, None] * K + j[None, :],
                            (r[:, None] < n) & (j[None, :] < K), 0)
                total += tl.dot(a, b, input_precision="tf32x3")
            addr = H + i[:, None] * K + j[None, :]
            mask = (i[:, None] < K) & (j[None, :] < K)
            old = tl.load(addr, mask, 0)
            tl.store(addr, old + total, mask)


@tr.jit
def _advance(Cursor, Armed, M: tl.constexpr, FORCE: tl.constexpr, THRESHOLD: tl.constexpr):
    if tl.load(Armed) != 0:
        n = tl.load(Cursor) + M
        tl.store(Cursor, tl.where(FORCE | (n >= THRESHOLD), 0, n))


def observe(x, mask, buffer, hessian, cursor, armed, count, peaks):
    m, k = x.shape
    if buffer is not None and not 0 < m <= buffer.shape[0] - ROWS + 1:
        raise ValueError("calibration append exceeds its declared staging capacity")
    _observe[(tr.cdiv(k, 128),)](x, mask, buffer, cursor, armed, count, peaks,
                                k, m, x.stride(0), mask is not None, buffer is not None,
                                tr.next_power_of_2(m), 128, enable_fp_fusion=False)
    if buffer is not None:
        _accumulate(buffer, hessian, cursor, armed, m, False)


def _accumulate(buffer, hessian, cursor, armed, rows, force):
    k = buffer.shape[1]
    programs = min(96, tr.cdiv(k, 32) ** 2)
    _gram[(programs,)](buffer, hessian, cursor, armed, k, rows,
                       buffer.shape[0], force, programs, ROWS)
    _advance[(1,)](cursor, armed, rows, force, ROWS)


def flush(buffer, hessian, cursor, armed):
    _accumulate(buffer, hessian, cursor, armed, 0, True)
