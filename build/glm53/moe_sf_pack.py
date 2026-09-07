"""6-bit packing of the NVFP4 block scales (39차 §4c, spec cell ``q``).

Every 16 fp4 values (8 packed bytes) carry one e4m3 scale byte, so the scales
are 1/9 = 11.1% of an expert's packed bytes. The static MoE kernel is
bandwidth-bound at 93-95% of the linear DRAM ceiling (39차 §3e/§3g), so bytes
removed here convert to time almost 1:1 -- and the expansion this costs rides
in the shadow of the DMA, where the MMA warps are already waiting.

What makes it lossless AND uniform is the packing unit. Measured over 516
expert scale tensors spanning every layer (``probes/nvfp4_scale_alphabet.py``):

    global alphabet            66 of 256 codes   -> a global LUT needs 7 bits
    per tensor, code span      median 28, max 239 -> 5 bits covers only 95%
    per 4096 B block, span     median 21, max 45  -> 6 bits covers 100.00%

4096 B is exactly the (128 rows x 512 k) scale block the kernel's DMA already
reads per stage, so the base is one scalar per stage. Within a block every
code is ``base + index`` with ``index < 64``: the kernel side is an add, not a
table read, and no block needs an escape.

Layout, for a block of ``SF_PACK_BLOCK`` scales (i = the scale's byte offset
inside the stock 4 KB block, so the packed order is the stock order):

    plane A   2048 B   low 4 bits    byte i >> 1, nibble (i & 1) * 4
    plane B   1024 B   high 2 bits   byte i >> 2, field  (i & 3) * 2
    base      1 B in a side table, one per block

    scale[i] = base + (planeA_nibble | planeB_field << 4)

Both planes are byte-aligned, so a lane that wants four consecutive scales
(one 32-bit read in the stock layout) reads two bytes of plane A and one of
plane B. 4096 -> 3072 B is -25% of the scale bytes and -2.78% of an expert's
packed bytes.
"""

from typing import Tuple

import torch

SF_PACK_BLOCK = 4096          # FC1 stages a (128 rows x 512 k) block: 4096 B
SF_PACK_BLOCK_FC2 = 1024      # FC2 stages (128 rows x 128 k): 1024 B
SF_PACK_BITS = 6
SF_PACK_MAX_INDEX = (1 << SF_PACK_BITS) - 1           # 63


def sf_packed_block_bytes(block: int = SF_PACK_BLOCK) -> int:
    """Packed bytes for one block: plane A (4 bits) + plane B (2 bits)."""
    return block * SF_PACK_BITS // 8


SF_PACK_BYTES = sf_packed_block_bytes(SF_PACK_BLOCK)          # 3072
SF_STAGE_BYTES = SF_PACK_BYTES + 16                           # 3088 (base tail)
SF_PACK_PLANE_A = SF_PACK_BLOCK // 2                          # 2048
SF_PACK_PLANE_B = SF_PACK_BLOCK // 4                          # 1024


def _blocks(sf: torch.Tensor, block: int) -> torch.Tensor:
    if sf.dtype != torch.uint8:
        raise TypeError(f"block scales are packed e4m3 bytes (uint8), got {sf.dtype}")
    if block % 4:
        raise ValueError(f"the packing block must be a multiple of 4 B, got {block}")
    flat = sf.reshape(-1)
    if flat.numel() % block:
        raise ValueError(
            f"scale bytes {flat.numel()} is not a multiple of the {block} B "
            "block the kernel stages"
        )
    return flat.view(-1, block)


def sf_pack_span(sf: torch.Tensor, block: int = SF_PACK_BLOCK) -> torch.Tensor:
    """Per-block ``max - min + 1`` over the e4m3 codes: the gate for packing
    (every block must fit ``SF_PACK_MAX_INDEX + 1`` codes)."""
    blk = _blocks(sf, block).to(torch.int16)
    return (blk.amax(dim=1) - blk.amin(dim=1) + 1).to(torch.int32)


def pack_sf(sf: torch.Tensor, block: int = SF_PACK_BLOCK) -> Tuple[torch.Tensor, torch.Tensor]:
    """(packed [blocks, block * 6 // 8] uint8, bases [blocks] uint8). Lossless:
    the codes of a block are ``base + index`` with a 6-bit index, split into a
    4-bit and a 2-bit byte-aligned plane."""
    blk = _blocks(sf, block)
    wide = blk.to(torch.int16)
    base = wide.amin(dim=1)
    idx = wide - base[:, None]
    top = int(idx.amax()) if idx.numel() else 0
    if top > SF_PACK_MAX_INDEX:
        raise ValueError(
            f"a {block} B scale block spans {top + 1} codes, over the "
            f"{SF_PACK_MAX_INDEX + 1} a {SF_PACK_BITS}-bit index holds"
        )
    idx = idx.to(torch.uint8)
    lo = idx & 0x0F
    hi = (idx >> 4) & 0x03
    plane_a = lo[:, 0::2] | (lo[:, 1::2] << 4)
    plane_b = (hi[:, 0::4] | (hi[:, 1::4] << 2)
               | (hi[:, 2::4] << 4) | (hi[:, 3::4] << 6))
    packed = torch.cat((plane_a, plane_b), dim=1).contiguous()
    want = sf_packed_block_bytes(block)
    if packed.shape[1] != want:
        raise AssertionError(f"packed block is {packed.shape[1]} B, expected {want}")
    return packed, base.to(torch.uint8)


def unpack_sf(packed: torch.Tensor, bases: torch.Tensor,
              block: int = SF_PACK_BLOCK) -> torch.Tensor:
    """The inverse of :func:`pack_sf` (the kernel does this per scale, in
    registers); returns the flat e4m3 bytes."""
    blocks = packed.reshape(-1, sf_packed_block_bytes(block))
    plane_a = blocks[:, : block // 2]
    plane_b = blocks[:, block // 2:]
    n = blocks.shape[0]
    lo = torch.empty((n, block), dtype=torch.uint8, device=packed.device)
    lo[:, 0::2] = plane_a & 0x0F
    lo[:, 1::2] = (plane_a >> 4) & 0x0F
    hi = torch.empty((n, block), dtype=torch.uint8, device=packed.device)
    for f in range(4):
        hi[:, f::4] = (plane_b >> (2 * f)) & 0x03
    idx = lo | (hi << 4)
    return (idx.to(torch.int16) + bases.reshape(-1, 1).to(torch.int16)).to(torch.uint8).reshape(-1)


# The kernel reads one packed block per stage and needs that block's base. It
# rides in the block itself (16 B tail, the copy's alignment unit) so the DMA
# is a single contiguous copy and the expanding threads read the base from
# smem -- no side table, no block-index arithmetic on the consumer side.
SF_PACK_BASE_TAIL = 16


def sf_stage_bytes(block: int = SF_PACK_BLOCK) -> int:
    """Bytes the DMA moves per stage: the packed planes plus the base tail."""
    return sf_packed_block_bytes(block) + SF_PACK_BASE_TAIL


def pack_sf_inline(sf: torch.Tensor, block: int = SF_PACK_BLOCK) -> torch.Tensor:
    """[blocks, sf_stage_bytes(block)] uint8: plane A, plane B, then the base
    byte in the 16 B tail (byte 0 of the tail; the rest is zero padding so the
    stage stays 16 B aligned)."""
    packed, bases = pack_sf(sf, block)
    tail = torch.zeros((packed.shape[0], SF_PACK_BASE_TAIL), dtype=torch.uint8,
                       device=packed.device)
    tail[:, 0] = bases
    return torch.cat((packed, tail), dim=1).contiguous()


def unpack_sf_inline(staged: torch.Tensor, block: int = SF_PACK_BLOCK) -> torch.Tensor:
    """The inverse of :func:`pack_sf_inline` (what the kernel's expansion must
    reproduce byte for byte)."""
    rows = staged.reshape(-1, sf_stage_bytes(block))
    body = sf_packed_block_bytes(block)
    return unpack_sf(rows[:, :body].contiguous(), rows[:, body], block)


def sf_packed_bytes(scale_bytes: int, block: int = SF_PACK_BLOCK) -> int:
    """Packed size of ``scale_bytes`` e4m3 bytes, bases included."""
    blocks = scale_bytes // block
    return blocks * (sf_packed_block_bytes(block) + 1)


__all__ = [
    "SF_PACK_BLOCK", "SF_PACK_BLOCK_FC2", "SF_PACK_BITS", "sf_packed_block_bytes",
    "SF_PACK_BASE_TAIL", "SF_STAGE_BYTES", "sf_stage_bytes", "pack_sf_inline",
    "unpack_sf_inline", "SF_PACK_BYTES", "SF_PACK_PLANE_A",
    "SF_PACK_PLANE_B", "SF_PACK_MAX_INDEX",
    "sf_pack_span", "pack_sf", "unpack_sf", "sf_packed_bytes",
]
