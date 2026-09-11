"""The NVFP4 scale-factor layout the SM120/SM121 MoE kernels eat (module).

A block-scale matrix [M, K/16] (one e4m3 per 16 elements along K) is not
stored row-major for the tensor cores: TRT-LLM / flashinfer interleave it in
128 x 4 tiles ("block_scale_interleave", what
`flashinfer_convert_sf_to_mma_layout` wraps for the b12x lane):

    padded to [round_up(M, 128), round_up(K/16, 4)]
    element (m, s) lives at  ((m // 128) * (Kp // 4) + s // 4) * 512
                             + (m % 32) * 16 + ((m % 128) // 32) * 4 + s % 4

The served path builds this at load for 42 x 288 experts (4.75 GiB of
scales per rank) from the row-major checkpoint; the ST engine's preshard
writes the rank file in this layout once (D1: the loader never repacks),
with the per-expert global scale folded in as the served layer does
(`process_weights_after_loading`: block_scale *= 1/w_gs, alpha := 1).
`unswizzle_sf` is the inverse, for the reference lane and the judge.
probes/sf_swizzle_check.py holds both to the flashinfer op inside the image.
"""
from __future__ import annotations

import torch


def _pad(n: int, to: int) -> int:
    return -(-n // to) * to


def swizzle_sf(sf: torch.Tensor) -> torch.Tensor:
    """[M, S] (any dtype) -> [Mp * Sp] in tile-interleaved order, zero padded."""
    m, s = sf.shape
    mp, sp = _pad(m, 128), _pad(s, 4)
    x = torch.zeros(mp, sp, dtype=sf.dtype, device=sf.device)
    x[:m, :s] = sf
    # [mp//128, 128, sp//4, 4] -> [mp//128, sp//4, 32, 4(m//32), 4(s)]
    t = x.view(mp // 128, 4, 32, sp // 4, 4)            # m = a*128 + b*32 + c ; s = d*4 + e
    return t.permute(0, 3, 2, 1, 4).reshape(-1)          # [a, d, c, b, e]


def unswizzle_sf(packed: torch.Tensor, m: int, s: int) -> torch.Tensor:
    mp, sp = _pad(m, 128), _pad(s, 4)
    t = packed.view(mp // 128, sp // 4, 32, 4, 4).permute(0, 3, 2, 1, 4)     # back to [a, b, c, d, e]
    return t.reshape(mp, sp)[:m, :s]


def mma_sf_view(packed: torch.Tensor, m: int, k: int) -> torch.Tensor:
    """Expose presharded [E, bytes] scales in b12x's six-dimensional layout.

    The bytes are already interleaved; this only changes shape and strides.
    Keep the view alive alongside its weights because b12x caches by pointer
    and registers the scale tensor's lifetime with its cache entry.
    """
    experts = packed.shape[0]
    return packed.view(experts, _pad(m, 128) // 128, _pad(_pad(k, 16) // 16, 4) // 4,
                       32, 4, 4).permute(3, 4, 1, 5, 2, 0)


def _selfcheck() -> None:
    torch.manual_seed(0)
    for m, s in ((1024, 256), (4096, 32), (100, 6)):
        sf = torch.randn(m, s)
        assert torch.equal(unswizzle_sf(swizzle_sf(sf), m, s), sf)
    sf = torch.arange(128 * 4, dtype=torch.float32).view(128, 4)     # one tile: (m, s) -> (m%32)*16 + (m//32)*4 + s
    sw = swizzle_sf(sf)
    for m_, s_ in ((0, 0), (1, 0), (32, 0), (0, 1), (127, 3)):
        assert sw[(m_ % 32) * 16 + (m_ // 32) * 4 + s_] == sf[m_, s_]
    print("  nvfp4_sf: 128x4 tile interleave, inverse exact, index law holds OK")


if __name__ == "__main__":
    _selfcheck()
