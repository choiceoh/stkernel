"""Byte-layout emulation: the KDA norm's WIDE o_proj pack (engine/kernels/kda/output.py _output_norm_pack) against
mk_wide_input_pack_kernel's layout (engine/kernels/dense/kernels.cu), from the same random BF16 norm output.

Both writers are followed literally at the level that decides bytes and places: the norm writes per program
(token, head) from Y's 128 values -- 32 lanes of four, amax over all 128, scale max(amax * f32(1/448), 1e-30),
reciprocal, SATFINITE e4m3 with x0 in the low byte -- the int32 word at (head*32 + token)*32 + lane and the
scale at head*32 + token. The pack kernel reads x = Y.reshape(16, heads*128) per (k block, row) and writes
the four bytes of lane l at (kb*32 + row)*128 + 4l and the scale at kb*32 + row. Both use one e4m3 encoder
(round to nearest, ties to even, saturating), so this checks layout, reductions and byte order; the
device conversion itself is checked on the GPU by probes/engine_producer_pack.py.
"""
import numpy as np

F32 = np.float32
INV_448 = F32(1.0 / 448.0)
FLOOR = F32(1e-30)
# Every non-negative finite e4m3fn value (bias 7, no infinities, 0x7F is NaN), in code order.
_E4M3 = np.array([(2.0 ** (e - 7)) * (1 + m / 8) if e else (2.0 ** -6) * (m / 8)
                  for e in range(16) for m in range(8)][:127], dtype=np.float64)


def e4m3(values):
    v = values.astype(np.float64)
    mag = np.minimum(np.abs(v), _E4M3[-1])
    idx = np.searchsorted(_E4M3, mag)
    lo, hi = np.clip(idx - 1, 0, 126), np.clip(idx, 0, 126)
    pick_hi = (np.abs(_E4M3[hi] - mag) < np.abs(mag - _E4M3[lo])) | (
        (np.abs(_E4M3[hi] - mag) == np.abs(mag - _E4M3[lo])) & (hi % 2 == 0))
    code = np.where(pick_hi, hi, lo).astype(np.uint8)
    return code | np.where(np.signbit(v) & (code != 0), 0x80, 0).astype(np.uint8)


def norm_writer(Y, heads, *, mutate=None):
    tokens = Y.shape[0]
    words = np.zeros(heads * 32 * 32, dtype=np.uint32)
    scales = np.zeros(heads * 32, dtype=F32)
    for pid in range(tokens * heads):
        token, block = pid // heads, pid % heads
        v = Y[token, block].astype(F32).reshape(32, 4)
        amax = F32(np.max(np.abs(v)))
        sc = np.maximum(F32(amax * INV_448), FLOOR)
        rcp = F32(F32(1.0) / sc)
        b = e4m3(v * rcp).astype(np.uint32)                  # [32 lanes][c0 c1 c2 c3]
        word = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
        if mutate == 'byte order':
            word = b[:, 1] | (b[:, 0] << 8) | (b[:, 2] << 16) | (b[:, 3] << 24)
        base = (block * 32 + token) * 32
        if mutate == 'token/head swap':
            base = (token * 32 + block) * 32
        words[base:base + 32] = word
        scales[block * 32 + token + (1 if mutate == 'scale index' else 0)] = sc
    return words.view(np.uint8), scales


def pack_kernel(Y, heads):
    tokens, d = Y.shape[0], Y.shape[2]
    x = Y.reshape(tokens, heads * d)
    aq = np.zeros(heads * 32 * 128, dtype=np.uint8)
    scales = np.zeros(heads * 32, dtype=F32)
    for kb in range(heads):
        for row in range(tokens):
            v = x[row, kb * 128:kb * 128 + 128].astype(F32).reshape(32, 4)
            mx = F32(np.max(np.abs(v)))                          # the warp's reduce_max over its 32 lanes
            scale = np.maximum(F32(mx * INV_448), FLOOR)
            inv = F32(F32(1.0) / scale)
            aq[(kb * 32 + row) * 128:(kb * 32 + row + 1) * 128] = e4m3(v * inv).reshape(-1)
            scales[kb * 32 + row] = scale
    return aq, scales


def compare(Y, heads, mutate=None):
    words, s_norm = norm_writer(Y, heads, mutate=mutate)
    aq, s_pack = pack_kernel(Y, heads)
    used_bytes = aq.reshape(heads, 32, 128)[:, :Y.shape[0]]
    got_bytes = words.reshape(heads, 32, 128)[:, :Y.shape[0]]
    bad_bytes = int(np.count_nonzero(used_bytes != got_bytes))
    bad_scales = int(np.count_nonzero(s_pack.reshape(heads, 32)[:, :Y.shape[0]] != s_norm.reshape(heads, 32)[:, :Y.shape[0]]))
    return bad_bytes, bad_scales


def bf16_like(rng, shape, scale):
    """Random values representable in BF16 (the norm output's storage) with zeros, tiny and huge rows mixed in."""
    y = rng.standard_normal(shape).astype(np.float32) * scale
    y[..., ::17] = 0.0
    y[0, 0] = 0.0                                           # an all-zero (token, head): the 1e-30 floor
    y[-1, -1, :4] = 3.0e38 if scale > 1 else y[-1, -1, :4]  # a saturating lane when the scale allows it
    return y


if __name__ == '__main__':
    rng = np.random.default_rng(9150)
    for heads in (1, 3, 16, 32):
        for scale in (1e-3, 1.0, 50.0, 1e4):
            Y = bf16_like(rng, (16, heads, 128), scale)
            print(f'heads={heads:2d} scale={scale:g}: mismatched bytes {compare(Y, heads)[0]}, scales {compare(Y, heads)[1]}')
    Y = bf16_like(rng, (16, 16, 128), 1.0)
    for mutation in ('token/head swap', 'byte order', 'scale index'):
        print(f'mutation {mutation!r}: mismatched bytes/scales {compare(Y, 16, mutation)}')
