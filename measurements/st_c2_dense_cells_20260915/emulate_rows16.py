"""Byte-offset emulation of mk_gemm_rows16_body (kernels.cu) against the ordinary lane's arithmetic.

Exact integer-valued stand-ins replace the e4m3 values: a W element is (magnitude+1)*sign*(exponent+1)
decoded from its nibble and its group's exponent byte; an X element is its pack byte. Every address the
kernel computes (tile-major W pack, ring rows with the XOR swizzle, halfword exponents, the natural X
pack, partial layout, epilogue) is followed literally, and the cooperative MMA delivers out[a][b] to
lane (a & 7, b >> 1), element 2*(a >= 8) + (b & 1), summing A[a] and B[b] over the lanes' k windows.
Reference: out[r][c] = rgs[c] * sum_slices sum_{kb in slice} s[kb][r] * sum_e X[r][kb,e] W[c][kb,e].
"""
import numpy as np

rng = np.random.default_rng(915)
KSTEP, PITCH = 128, 64
NIB, RAW = 64 * PITCH, 64 * PITCH + 64 * 8


def value(code, exp):
    mag = (code & 7).astype(np.int64) + 1
    sign = np.where(code & 8, -1, 1)
    return mag * sign * (exp.astype(np.int64) + 1)


def run(n_orig, k, slices, blocks_to_check):
    kblks = k // KSTEP
    npad = (n_orig + 127) // 128 * 128
    tiles = npad // 128
    codes = rng.integers(0, 16, size=(npad, k), dtype=np.int64)
    exps = rng.integers(0, 16, size=(npad, k // 16), dtype=np.int64)
    # DenseLinear._tile_pack: byte j = code[2j] | code[2j+1] << 4, view [tiles, kblk, 128, 64]
    pairs = codes.reshape(npad, k // 2, 2)
    data = (pairs[..., 0] | (pairs[..., 1] << 4)).astype(np.uint8)
    wq4 = data.reshape(tiles, 128, kblks, 64).transpose(0, 2, 1, 3).copy().reshape(-1)
    ws4 = exps.astype(np.uint8).reshape(tiles, 128, kblks, 8).transpose(0, 2, 1, 3).copy().reshape(-1)
    W = value(codes, np.repeat(exps, 16, axis=1))                         # [npad, k]
    X = rng.integers(0, 16, size=(16, k), dtype=np.int64)                  # pack bytes = values
    S = rng.integers(1, 9, size=(kblks, 16), dtype=np.int64)               # row scales per k block
    rgs = rng.integers(1, 5, size=npad, dtype=np.int64)
    # wide pack, natural order: aq[(kb*32 + row)*128 + e] (32 rows reserved per k block)
    aq = np.zeros(kblks * 32 * KSTEP, dtype=np.int64)
    for kb in range(kblks):
        for row in range(16):
            aq[(kb * 32 + row) * KSTEP:(kb * 32 + row + 1) * KSTEP] = X[row, kb * KSTEP:(kb + 1) * KSTEP]
    sliced = slices == 8
    ref = np.zeros((16, n_orig), dtype=np.int64)
    for s in range(slices):
        for kb in range(kblks * s // slices, kblks * (s + 1) // slices):
            e = slice(kb * KSTEP, (kb + 1) * KSTEP)
            ref += (S[kb][:, None] * (X[:, e] @ W[:n_orig, e].T))
    ref *= rgs[None, :n_orig]
    out = np.full((16, n_orig), -999999, dtype=np.int64)
    words = lambda u32: [(u32 >> (4 * i)) & 15 for i in range(8)]    # eight nibbles, element order
    for nt in blocks_to_check:
        partial = np.zeros((kblks if not sliced else 8) * 128, dtype=np.int64)
        for warp in range(8):
            kb0, kbn = kblks * warp // 8, kblks * (warp + 1) // 8
            acc = np.zeros((8, 4), dtype=np.int64)            # per g: SLICED accumulators, lane (g, q) -> [q]
            accq = np.zeros((32, 4), dtype=np.int64)
            for kb in range(kb0, kbn):
                ring = np.zeros(RAW, dtype=np.int64)
                w0 = ((nt // 16) * kblks + kb) * 8192 + (nt % 16) * 512
                s0 = ((nt // 16) * kblks + kb) * 1024 + (nt % 16) * 64
                for lane in range(32):
                    g, q = lane >> 2, lane & 3
                    r = warp * 8 + g
                    dst = r * PITCH + ((q ^ ((r >> 1) & 3)) << 4)
                    ring[dst:dst + 16] = wq4[w0 + lane * 16:w0 + lane * 16 + 16]
                    if lane < 4:
                        d = NIB + (warp * 4 + lane) * 16
                        ring[d:d + 16] = ws4[s0 + lane * 16:s0 + lane * 16 + 16]
                ka = np.zeros((32, 4), dtype=np.int64)
                for ks in range(4):
                    A = np.zeros((16, 32), dtype=np.int64)    # A[a][q*8 + j]
                    B = np.zeros((8, 32), dtype=np.int64)
                    for lane in range(32):
                        g, q = lane >> 2, lane & 3
                        r = warp * 8 + g
                        ea, eb = ring[NIB + r * 8 + 2 * q], ring[NIB + r * 8 + 2 * q + 1]
                        slot = r * PITCH + ((q ^ ((r >> 1) & 3)) << 4)
                        wsel = (ks + q) & 3
                        koff = 32 * q + 8 * wsel
                        b = ring[slot + 4 * wsel:slot + 4 * wsel + 4]
                        u32 = int(b[0]) | int(b[1]) << 8 | int(b[2]) << 16 | int(b[3]) << 24
                        exp = np.full(8, ea if wsel < 2 else eb)
                        B[g, q * 8:q * 8 + 8] = value(np.array(words(u32)), exp)
                        xr0 = (kb * 32 + g) * KSTEP
                        A[g, q * 8:q * 8 + 8] = aq[xr0 + koff:xr0 + koff + 8]
                        A[g + 8, q * 8:q * 8 + 8] = aq[xr0 + 8 * KSTEP + koff:xr0 + 8 * KSTEP + koff + 8]
                    P = A @ B.T                                   # [16 X rows][8 W rows]
                    for lane in range(32):
                        g, q = lane >> 2, lane & 3
                        for i in range(4):
                            ka[lane, i] += P[g + 8 * (i >> 1), 2 * q + (i & 1)]
                for lane in range(32):
                    g, q = lane >> 2, lane & 3
                    if sliced:
                        accq[lane, 0] += ka[lane, 0] * S[kb, g]
                        accq[lane, 1] += ka[lane, 1] * S[kb, g]
                        accq[lane, 2] += ka[lane, 2] * S[kb, g + 8]
                        accq[lane, 3] += ka[lane, 3] * S[kb, g + 8]
                    else:
                        for i in range(4):
                            partial[kb * 128 + (g + 8 * (i >> 1)) * 8 + 2 * q + (i & 1)] = ka[lane, i]
            if sliced:
                for lane in range(32):
                    g, q = lane >> 2, lane & 3
                    for i in range(4):
                        partial[(warp * 16 + g + 8 * (i >> 1)) * 8 + 2 * q + (i & 1)] = accq[lane, i]
        for t in range(128):
            row, col = t >> 3, nt * 8 + (t & 7)
            if sliced:
                v = sum(partial[s * 128 + t] for s in range(8))
            else:
                v = 0
                for s in range(slices):
                    o = 0
                    for kb in range(kblks * s // slices, kblks * (s + 1) // slices):
                        o = partial[kb * 128 + t] * S[kb, row] + o
                    v += o
            out[row, col] = v * rgs[col]
    cols = np.concatenate([np.arange(nt * 8, nt * 8 + 8) for nt in blocks_to_check])
    bad = np.argwhere(out[:, cols] != ref[:, cols])
    return len(cols), len(bad)


if __name__ == '__main__':
    for n, k, slices in ((6416, 4096, 8), (4096, 2048, 3), (4096, 3072, 3), (4096, 4096, 3),
                         (6144, 4096, 2), (2048, 4096, 6), (4096, 1536, 3), (8192, 1536, 3)):
        blocks = n // 8
        pick = sorted({0, 1, 15, 16, 17, blocks // 2, blocks - 17, blocks - 16, blocks - 2, blocks - 1})
        checked, bad = run(n, k, slices, [b for b in pick if 0 <= b < blocks])
        print(f'n={n} k={k} slices={slices}: {checked} columns x 16 rows checked, {bad} mismatches')
