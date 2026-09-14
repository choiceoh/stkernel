"""Integer-only draft preparation and the existing keyed draw recipe in one launch each."""
import torch
import triton
import triton.language as tl


@triton.jit
def _draft_inputs(A, P, IDS, POS, AS: tl.constexpr, PS: tl.constexpr, T: tl.constexpr,
                  MASK: tl.constexpr, DEVICE_POSITION: tl.constexpr, POSITION: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B)
    anchor = tl.load(A + row * AS)
    if DEVICE_POSITION:
        position = tl.load(P + row * PS)
    else:
        position = tl.full((), POSITION, tl.int64)
    tl.store(IDS + row * T + i, tl.where(i == 0, anchor, tl.full((), MASK, tl.int64)), i < T)
    tl.store(POS + row * T + i, position + i.to(tl.int64), i < T)


def draft_inputs(anchors, positions, k, mask_id):
    n, t = anchors.numel(), k + 1
    storage = torch.empty((2, n * t), device=anchors.device, dtype=torch.int64)
    ids, pos = storage.unbind(0)
    device_position = torch.is_tensor(positions)
    _draft_inputs[(n,)](anchors, positions if device_position else anchors, ids, pos,
                        anchors.stride(0), positions.stride(0) if device_position else 0, t, mask_id,
                        device_position, 0 if device_position else positions, triton.next_power_of_2(t), num_warps=1)
    return ids, pos


@triton.jit
def _mix(x):
    # Unsigned arithmetic supplies the same modulo-2^64 operations as base/draws.
    x = x + tl.full((), 0x9E3779B97F4A7C15, tl.uint64)
    x = (x ^ (x >> 30)) * tl.full((), 0xBF58476D1CE4E5B9, tl.uint64)
    x = (x ^ (x >> 27)) * tl.full((), 0x94D049BB133111EB, tl.uint64)
    return x ^ (x >> 31)


@triton.jit
def _step_block(NONCE, GEN, OUT, NS: tl.constexpr, GS: tl.constexpr, SEED_KEY: tl.constexpr,
                K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, B).to(tl.uint64)
    key = _mix(tl.load(NONCE + row * NS).to(tl.uint64) ^ tl.full((), SEED_KEY, tl.uint64))
    key = _mix(key ^ tl.load(GEN + row * GS).to(tl.uint64))
    word = tl.where(i < K, (1 << 32) + i, tl.where(i < 2 * K, (3 << 32) + i - K, 4 << 32))
    z = _mix(key ^ word)
    # The reference converts the top 53 bits to FP64, scales by 2^-53, then rounds once to FP32.
    uniform = (z >> 11).to(tl.float64) * tl.full((), 2.0 ** -53, tl.float64)
    tl.store(OUT + row * (2 * K + 1) + i, uniform.to(tl.float32), i < 2 * K + 1)


def step_block(seed, nonces, generations, k):
    from engine.base.draws import mix
    out = torch.empty(nonces.numel(), 2 * k + 1, device=nonces.device, dtype=torch.float32)
    _step_block[(nonces.numel(),)](nonces, generations, out, nonces.stride(0), generations.stride(0),
                                   mix(int(seed)), k, triton.next_power_of_2(2 * k + 1),
                                   num_warps=1, enable_fp_fusion=False)
    return out
