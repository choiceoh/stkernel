"""Dense W4 GEMM for a handful of rows: y = x @ dequant(W)^T with x [M, K] bf16 and W int4 per group along K.

Written for the DFlash2 drafter's proposal (45차 §23 GPU 판정 5차): its 47 GEMMs read 2.03 GiB of bf16 weights on
every rank every decode step, for M = rows x (k+1) = 6..24 activations -- pure weight streaming. Packed as symmetric
int4 with a bf16 scale per group of 32 the step reads 0.64 GiB. The kernel streams the packed weights once, unpacks in
registers to bf16 (levels -7..7 times the group's scale) and runs the tensor-core dot against the L2-resident
activation tile with fp32 accumulation. Split-K partials land in an fp32 workspace and are summed in a fixed order:
every rank drafts the same tokens, so nothing here is atomic.

Packing is offline (profiles/glm53/preshard.py --drafter-w4 -> drafter-w4.safetensors, once, one file for every
rank): levels -7..7 stored as q+8 in a nibble; the scale of a group is clip * amax / 7 with the clip picked per
group by least squared error over CLIP_GRID, then rounded to bf16 BEFORE quantising so the stored scale is the one
the kernel multiplies by. A K block of 128 columns is 64 bytes: the block's first 64 columns in the low nibbles, its
last 64 in the high nibbles, so a tile unpacks into two K=64 halves with no interleaving.

Small M only: the whole weight is streamed once per 64-row activation tile.
"""
import torch
import torch.nn.functional as Fn
import triton
import triton.language as tl

GROUP = 32                       # columns per scale
KBLOCK = 128                     # columns per 64 packed bytes (two halves of 64)
LEVELS = 7                       # symmetric int4: -7..7 (the code 0 is never written)
CLIP_GRID = (1.0, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8, 0.75, 0.7)


def pack_w4(weight: torch.Tensor, group: int = GROUP, clip_grid=CLIP_GRID) -> "tuple[torch.Tensor, torch.Tensor]":
    """(packed uint8 [N, K/2], scales bf16 [N, K/group]) for a [N, K] weight. K must be whole 128-blocks and the group must
    divide a 64-column half. Deterministic for a given input on a given device; the fleet packs once and ships the file."""
    if weight.dim() != 2:
        raise ValueError("pack_w4 takes a [N, K] weight")
    N, K = weight.shape
    if K % KBLOCK or group <= 0 or 64 % group:
        raise ValueError(f"pack_w4: K={K} must be a multiple of {KBLOCK} and the group ({group}) must divide 64")
    w = weight.detach().float().reshape(N, K // group, group)
    amax = w.abs().amax(-1, keepdim=True)
    best_err = best_scale = None
    for clip in clip_grid:
        scale = (amax * (clip / LEVELS)).to(torch.bfloat16).float().clamp_min(2.0 ** -24)   # the stored (bf16) scale
        q = (w / scale).round().clamp_(-LEVELS, LEVELS)
        err = ((q * scale - w) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best_scale = err, scale
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, scale, best_scale)
    q = (w / best_scale).round().clamp_(-LEVELS, LEVELS)
    codes = (q + 8).to(torch.uint8).reshape(N, K // KBLOCK, KBLOCK)
    packed = (codes[:, :, :64] | (codes[:, :, 64:] << 4)).reshape(N, K // 2).contiguous()
    return packed, best_scale.reshape(N, K // group).to(torch.bfloat16).contiguous()


def unpack_w4(packed: torch.Tensor, scales: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """The bf16 [N, K] weight the kernel multiplies by: (q - 8) as bf16 times the group's bf16 scale, rounded as in-register."""
    N, half = packed.shape
    K = half * 2
    bytes_ = packed.reshape(N, K // KBLOCK, 64)
    codes = torch.cat([bytes_ & 0xF, bytes_ >> 4], dim=-1).reshape(N, K)
    q = (codes.to(torch.int16) - 8).to(torch.bfloat16)
    return (q.reshape(N, K // group, group) * scales.unsqueeze(-1)).reshape(N, K)


@triton.jit
def _w4_gemm(X, W, S, Y, M, N, stride_xm, stride_wn, stride_sn, stride_ym, stride_ys, blocks_per_split,
             GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, KB: tl.constexpr, F32_OUT: tl.constexpr):
    """KB 128-column blocks per loop step: a row's packed bytes are read KB*64 contiguous at a time."""
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    j = tl.arange(0, 64)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    first = pid_s * blocks_per_split
    for kb in range(first, first + blocks_per_split, KB):
        for u in tl.static_range(KB):
            k0 = (kb + u) * 128
            x_lo = tl.load(X + rm[:, None] * stride_xm + (k0 + j)[None, :], mask=m_ok[:, None], other=0.0)
            x_hi = tl.load(X + rm[:, None] * stride_xm + (k0 + 64 + j)[None, :], mask=m_ok[:, None], other=0.0)
            packed = tl.load(W + rn[:, None] * stride_wn + ((kb + u) * 64 + j)[None, :], mask=n_ok[:, None], other=0).to(tl.int32)
            q_lo = (packed & 0xF) - 8
            q_hi = (packed >> 4) - 8
            gp = (k0 // GROUP) + tl.arange(0, 64 // GROUP)                                   # the groups of the low half
            s_lo = tl.load(S + rn[:, None] * stride_sn + gp[None, :], mask=n_ok[:, None], other=0.0)
            s_hi = tl.load(S + rn[:, None] * stride_sn + (gp + 64 // GROUP)[None, :], mask=n_ok[:, None], other=0.0)
            s_lo = tl.reshape(tl.broadcast_to(tl.expand_dims(s_lo, 2), [BLOCK_N, 64 // GROUP, GROUP]), [BLOCK_N, 64])
            s_hi = tl.reshape(tl.broadcast_to(tl.expand_dims(s_hi, 2), [BLOCK_N, 64 // GROUP, GROUP]), [BLOCK_N, 64])
            w_lo = q_lo.to(tl.bfloat16) * s_lo
            w_hi = q_hi.to(tl.bfloat16) * s_hi
            acc = tl.dot(x_lo, tl.trans(w_lo), acc)
            acc = tl.dot(x_hi, tl.trans(w_hi), acc)
    out = Y + pid_s * stride_ys + rm[:, None] * stride_ym + rn[None, :]
    mask = m_ok[:, None] & n_ok[None, :]
    if F32_OUT:
        tl.store(out, acc, mask=mask)
    else:
        tl.store(out, acc.to(tl.bfloat16), mask=mask)


def plan(M: int, N: int, K: int) -> "tuple[int, int, int, int, int]":
    """(BLOCK_M, BLOCK_N, SPLIT_K, KB, num_warps), from the GB10 sweep beside production (45차 §23 GPU 판정 5차).
    Up to 16 rows: 32 output columns per program and two 128-blocks per loop step (a row's packed bytes read 128
    contiguous). More rows: 128 columns per program, one block per step -- the activation tile is fetched per program
    per step, so a wider tile amortises it over more weight columns. Four warps, two pipeline stages, and the K range
    split until the grid holds ~256 programs (48 SMs streaming weights want the parallelism; the split's partials are
    summed in a fixed order), always whole steps."""
    block_m = 16 if M <= 16 else 32 if M <= 32 else 64
    block_n, kb = (32, 2) if M <= 16 else (128, 1)
    warps = 4
    tiles = triton.cdiv(N, block_n)
    steps = K // (KBLOCK * kb)
    split = 1
    if tiles < 256:
        for candidate in (2, 4, 8):
            if steps % candidate == 0:
                split = candidate
                if tiles * candidate >= 256:
                    break
    return block_m, block_n, split, kb, warps


def w4_linear(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """x [M, K] bf16 -> [M, N] bf16 = x @ unpack(packed, scales)^T. On CUDA the kernel; elsewhere the reference
    (unpack, then Fn.linear), which is what the tests compare against."""
    if x.dtype != torch.bfloat16 or x.dim() != 2:
        raise TypeError("w4_linear takes a [M, K] bf16 activation")
    if not x.is_cuda:
        return Fn.linear(x, unpack_w4(packed, scales, group))
    M, K = x.shape
    N, half = packed.shape
    if half * 2 != K or scales.shape != (N, K // group) or packed.dtype != torch.uint8 or scales.dtype != torch.bfloat16:
        raise ValueError(f"w4_linear: x {tuple(x.shape)}, packed {tuple(packed.shape)}, scales {tuple(scales.shape)} disagree")
    if K % KBLOCK or x.stride(1) != 1 or packed.stride(1) != 1 or scales.stride(1) != 1:
        raise ValueError("w4_linear: K must be whole 128-blocks and the operands row-major")
    return _launch(x, packed, scales, group, *plan(M, N, K))


def _launch(x, packed, scales, group, block_m, block_n, split, kb, warps, stages=2):
    M, K = x.shape
    N = packed.shape[0]
    if (K // KBLOCK) % kb or (K // KBLOCK // kb) % split:
        raise ValueError(f"w4_linear: K={K} is not whole steps of {kb} blocks over {split} splits")
    grid = (triton.cdiv(N, block_n), triton.cdiv(M, block_m), split)
    if split == 1:
        y = torch.empty(M, N, dtype=torch.bfloat16, device=x.device)
        _w4_gemm[grid](x, packed, scales, y, M, N, x.stride(0), packed.stride(0), scales.stride(0), y.stride(0), 0,
                       K // KBLOCK, GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n, KB=kb, F32_OUT=False, num_warps=warps, num_stages=stages)
        return y
    ws = torch.empty(split, M, N, dtype=torch.float32, device=x.device)
    _w4_gemm[grid](x, packed, scales, ws, M, N, x.stride(0), packed.stride(0), scales.stride(0), ws.stride(1), ws.stride(0),
                   K // KBLOCK // split, GROUP=group, BLOCK_M=block_m, BLOCK_N=block_n, KB=kb, F32_OUT=True, num_warps=warps, num_stages=stages)
    return ws.sum(0).to(torch.bfloat16)                                     # a fixed-order reduction: the same on every rank


def _selfcheck() -> None:
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    for M, N, K in ((6, 1024, 4096), (24, 4096, 12288), (12, 300, 20480)):
        w = (torch.randn(N, K, device=dev, generator=g) * 0.02).to(torch.bfloat16)
        x = torch.randn(M, K, device=dev, generator=g).to(torch.bfloat16)
        packed, scales = pack_w4(w)
        got = w4_linear(x, packed, scales).float()
        ref = Fn.linear(x, unpack_w4(packed, scales)).float()
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        assert err < 2e-2, (M, N, K, err)
        q = ((unpack_w4(packed, scales).float() - w.float()).pow(2).mean().sqrt() / w.float().pow(2).mean().sqrt()).item()
        print(f"  w4_gemm: M={M} N={N} K={K} kernel vs reference max rel {err:.2e}, quantisation rms rel {q:.3f} OK")


if __name__ == "__main__":
    _selfcheck()
