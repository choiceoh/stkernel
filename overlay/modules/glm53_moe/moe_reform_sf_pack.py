"""Lossless scale storage for the optional ``t,r,sf6`` decode lane.

FC1 consumes 128 rows x K256, FC2 consumes 256 rows x K128: both scale
stages contain 2048 bytes. FC2 joins two separated 128-row storage blocks
and interleaves their 512-byte K64 groups: [K64][row128][512 bytes].
Flattening its original tensor into 2048-byte rows is NOT the stage order.
Original scale storage is never modified and continues to serve prefill.

A plane whose stage contains a byte span above 64 is returned as unsupported.
The dispatcher then retains the ordinary kernel for the whole layer. This
covers all 256 byte codes without lossy clipping or a mixed-format DMA ring.
"""
from __future__ import annotations

from dataclasses import dataclass

REFORM_SF_BLOCK = 2048
REFORM_SF_STAGE = 1552  # low nibbles 1024 + high pairs 512 + base/alignment 16
REFORM_SF_CHUNK = 1024


def stage_shape(rows: int, k: int, kind: str) -> tuple[int, int]:
    if kind not in ("fc1", "fc2"):
        raise ValueError(f"unknown scale plane {kind!r}")
    nr, nk = (128, 256) if kind == "fc1" else (256, 128)
    if rows <= 0 or k <= 0 or rows % nr or k % nk:
        raise ValueError(f"{kind} requires rows%{nr}=0 and k%{nk}=0")
    return rows // nr, k // nk


def stage_source_offset(rows: int, k: int, kind: str,
                        expert: int, row_tile: int, k_tile: int,
                        stage_byte: int) -> int:
    """Independent integer byte map used by the layout/device correctness gate."""
    nr, nk = stage_shape(rows, k, kind)
    if expert < 0 or not 0 <= row_tile < nr or not 0 <= k_tile < nk:
        raise ValueError("scale tile coordinate out of bounds")
    if not 0 <= stage_byte < REFORM_SF_BLOCK:
        raise ValueError("scale byte coordinate out of bounds")
    base = expert * rows * (k // 16)
    if kind == "fc1":
        return base + (row_tile * nk + k_tile) * REFORM_SF_BLOCK + stage_byte
    k64, rest = divmod(stage_byte, 1024)
    row_block, byte = divmod(rest, 512)
    return (base + ((row_tile * 2 + row_block) * nk + k_tile) * 1024
            + k64 * 512 + byte)


def pack_stage_bytes(raw: bytes) -> bytes | None:
    """CPU oracle for one stage; None is an exact raw-format fallback."""
    if len(raw) != REFORM_SF_BLOCK:
        raise ValueError("an sf6 stage is exactly 2048 bytes")
    base = min(raw)
    if max(raw) - base > 63:
        return None
    out = bytearray(REFORM_SF_STAGE)
    for i, code in enumerate(raw):
        value = code - base
        out[i // 2] |= (value & 15) << ((i % 2) * 4)
        out[1024 + i // 4] |= (value >> 4) << ((i % 4) * 2)
    out[1536] = base
    return bytes(out)


def unpack_stage_bytes(packed: bytes) -> bytes:
    if len(packed) != REFORM_SF_STAGE or any(packed[1537:]):
        raise ValueError("invalid sf6 stage size or reserved bytes")
    base = packed[1536]
    out = bytearray(REFORM_SF_BLOCK)
    for i in range(REFORM_SF_BLOCK):
        lo = (packed[i // 2] >> ((i % 2) * 4)) & 15
        hi = (packed[1024 + i // 4] >> ((i % 4) * 2)) & 3
        code = base + lo + (hi << 4)
        if code > 255:
            raise ValueError("sf6 index overflows the original byte code")
        out[i] = code
    return bytes(out)


def _stage_rows(sf, *, experts: int, rows: int, k: int, kind: str,
                first: int, last: int):
    """Gather only this bounded chunk in the actual MMA stage byte order."""
    import torch
    nr, nk = stage_shape(rows, k, kind)
    if sf.dtype != torch.uint8 or not sf.is_contiguous():
        raise ValueError("scale storage must be contiguous uint8 bytes")
    if experts <= 0 or sf.numel() != experts * rows * k // 16:
        raise ValueError("scale storage does not match the expert geometry")
    count = experts * nr * nk
    if not 0 <= first <= last <= count:
        raise ValueError("scale stage slice out of bounds")
    if kind == "fc1":
        return sf.reshape(count, REFORM_SF_BLOCK)[first:last]
    idx = torch.arange(first, last, device=sf.device, dtype=torch.int64)
    expert = idx // (nr * nk)
    rt, kt = (idx // nk) % nr, idx % nk
    src = sf.reshape(experts, nr * 2, nk, 2, 512)
    # CuTe's FC2 shared scale layout interleaves the row blocks within
    # each K64 group; global storage keeps the two row blocks separated.
    return torch.stack((src[expert, rt * 2, kt],
                        src[expert, rt * 2 + 1, kt]), dim=2).reshape(-1, 2048)


def pack_plane(sf, *, experts: int, rows: int, k: int, kind: str):
    """Pack with bounded temporary storage and a mandatory on-device roundtrip.

    Called only before capture. Both source and packed storage remain live in
    the layer's WeightViews. All chunks are checked for representability before
    allocating the result; a span >64 never leaves a partial packed plane.
    """
    import torch
    from .moe_sf_pack import pack_sf_inline, unpack_sf_inline
    nr, nk = stage_shape(rows, k, kind)
    if experts <= 0:
        raise ValueError("scale plane must contain at least one expert")
    count = experts * nr * nk
    if sf.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("sf6 weights must be prepared before CUDA graph capture")
    widest = torch.zeros((), dtype=torch.int16, device=sf.device)
    for first in range(0, count, REFORM_SF_CHUNK):
        raw = _stage_rows(sf, experts=experts, rows=rows, k=k, kind=kind,
                          first=first, last=min(count, first + REFORM_SF_CHUNK))
        wide = raw.to(torch.int16)
        widest = torch.maximum(widest, (wide.amax(1) - wide.amin(1)).amax())
    span = int(widest.item()) + 1
    if span > 64:
        return None, f"{kind} stage spans {span} byte codes (limit 64)"
    packed = torch.empty((experts, nr * nk, REFORM_SF_STAGE),
                         device=sf.device, dtype=torch.uint8)
    target = packed.view(count, REFORM_SF_STAGE)
    valid = torch.ones((), dtype=torch.bool, device=sf.device)
    for first in range(0, count, REFORM_SF_CHUNK):
        last = min(count, first + REFORM_SF_CHUNK)
        raw = _stage_rows(sf, experts=experts, rows=rows, k=k, kind=kind,
                          first=first, last=last)
        stage = pack_sf_inline(raw, REFORM_SF_BLOCK)
        target[first:last].copy_(stage)
        back = unpack_sf_inline(target[first:last], REFORM_SF_BLOCK).view_as(raw)
        valid &= (back == raw).all()
    if not bool(valid.item()):
        raise RuntimeError(f"sf6 {kind} device pack roundtrip failed")
    return packed, None


@dataclass(frozen=True)
class ReformScales:
    fc1: object | None
    fc2: object | None
    reason: str | None = None

    @property
    def enabled(self) -> bool:
        return self.fc1 is not None and self.fc2 is not None


def prepare_reform_scales(fc1, fc2, *, experts: int, n: int, k: int) -> ReformScales:
    """Atomically admit both planes, retaining exact raw scales on fallback."""
    first, reason = pack_plane(fc1, experts=experts, rows=2*n, k=k, kind="fc1")
    if reason:
        return ReformScales(None, None, reason)
    second, reason = pack_plane(fc2, experts=experts, rows=k, k=n, kind="fc2")
    if reason:
        return ReformScales(None, None, reason)
    return ReformScales(first, second)
