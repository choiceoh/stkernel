"""Amortize small-row Gram updates without a host read or a lost input row.

Append, accumulate, advance are ordered on the caller's stream. The staging
plane has room for the threshold plus one maximum-sized call, so crossing the
threshold never overwrites an unconsumed row. A fixed persistent grid exits
immediately between flushes instead of launching one CTA per Hessian tile.
"""
import triton as tr
import triton.language as tl

ROWS: tl.constexpr = 256


@tr.jit
def _append(X, Buffer, Cursor, Armed, K: tl.constexpr, M: tl.constexpr,
            SX: tl.constexpr, BLOCK: tl.constexpr):
    if tl.load(Armed) != 0:
        offset = tl.load(Cursor)
        ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, col = ix // K, ix % K
        x = tl.load(X + row * SX + col, row < M, 0)
        tl.store(Buffer + (offset + row) * K + col, x, row < M)


@tr.jit
def _gram(Buffer, H, Cursor, Armed, K: tl.constexpr, M: tl.constexpr,
          CAPACITY: tl.constexpr, FORCE: tl.constexpr, PROGRAMS: tl.constexpr,
          B: tl.constexpr = 32, R: tl.constexpr = 32):
    n = tl.load(Cursor) + M
    if (tl.load(Armed) != 0) & (n > 0) & (FORCE | (n >= ROWS)):
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
def _advance(Cursor, Armed, M: tl.constexpr, FORCE: tl.constexpr):
    if tl.load(Armed) != 0:
        n = tl.load(Cursor) + M
        tl.store(Cursor, tl.where(FORCE | (n >= ROWS), 0, n))


def update(x, buffer, hessian, cursor, armed):
    m, k = x.shape
    if not 0 < m <= buffer.shape[0] - ROWS + 1:
        raise ValueError("calibration append exceeds its declared staging capacity")
    _append[(tr.cdiv(m * k, 1024),)](x, buffer, cursor, armed, k, m, x.stride(0), 1024)
    _accumulate(buffer, hessian, cursor, armed, m, False)


def _accumulate(buffer, hessian, cursor, armed, rows, force):
    k = buffer.shape[1]
    programs = min(96, tr.cdiv(k, 32) ** 2)
    _gram[(programs,)](buffer, hessian, cursor, armed, k, rows,
                       buffer.shape[0], force, programs)
    _advance[(1,)](cursor, armed, rows, force)


def flush(buffer, hessian, cursor, armed):
    _accumulate(buffer, hessian, cursor, armed, 0, True)
