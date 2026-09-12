"""Durable KDA rounding identity and eager state stores.

FP16 uses adjacent-value stochastic conversion, FP32 arithmetic, and Philox10.
Copies of already rounded FP16 bits (rollback, snapshots, stage, tier) must
remain exact copies. CPU execution is a reference for the storage boundary.
"""
import torch

ROUNDING = "sr-philox10-v1"
FP16_FORMAT = "glm53-kda-fp16-sr-philox10-v2"


def rounding_seed(layer, rank):
    if not 0 <= layer < 65536 or not 0 <= rank < 256:
        raise ValueError("KDA rounding domain exceeds layer/rank encoding")
    return 0x4B44410000000000 | (layer << 8) | rank


def philox_words(counter, seed):
    """CPU Philox4x32-10 reference, including the high counter word."""
    mask = 0xffffffff
    c0, c1 = counter & mask, (counter >> 32) & mask
    c2, c3 = torch.zeros_like(c0), torch.zeros_like(c0)
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        p0, p1 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = ((p1 >> 32) & mask) ^ c1 ^ k0, p1 & mask, ((p0 >> 32) & mask) ^ c3 ^ k1, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0


def store_state(dst, src, position, seed):
    if dst.dtype != torch.float16 or src.dtype == torch.float16:
        dst.copy_(src)
    elif src.is_cuda:
        from engine.kernels.kda.rounding import store
        store(dst, src, position, seed)
    else:
        # Independent adjacent-value reference for CPU composition tests.
        x = src.float()
        magnitude = x.abs()
        nearest = magnitude.half()
        lo = torch.where(nearest.float() > magnitude, torch.nextafter(nearest, torch.zeros_like(nearest)), nearest)
        hi = torch.nextafter(lo, torch.full_like(lo, float('inf')))
        counter = torch.arange(x.numel(), dtype=torch.int64).reshape(x.shape) + position * x.numel()
        probability = (magnitude.double() - lo.double()) / (hi.double() - lo.double())
        threshold = (torch.where(torch.isfinite(probability), probability, 0) * 2**32).floor().long()
        value = torch.where(philox_words(counter, seed) < threshold, hi, lo)
        value = torch.copysign(value, x)
        dst.copy_(torch.where(magnitude <= 65504., value, x.half()))
