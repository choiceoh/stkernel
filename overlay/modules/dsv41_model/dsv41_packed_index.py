"""Packed E2M1/E8M0 index-key storage for the official V4.1 reference.

Only the ORIGINAL quantizer's packed output is accepted by the writer. There
is no re-quantizer here: quantizing an already fake-quantized BF16 cache again
is not a proven lossless operation, particularly after overflow to infinity.
Each D128 key occupies 64 packed bytes and four E8M0 scale bytes, versus 256
BF16 bytes. Even features are in the low nibble, odd features in the high one.

The score paths restore BF16 only in bounded position/query tiles. They retain
the eager reference's BF16 dot output, BF16 weighted product and BF16 head sum.
The Torch backend is an oracle/default, not a performance claim. Triton is
explicit opt-in and requires real-device differential validation.
"""
from __future__ import annotations

import torch


_INT32_MAX = 2**31 - 1
_MAX_TILE_VALUES = 262144


def _integer(value, name, minimum=0, maximum=_INT32_MAX):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _identity(tensor):
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device)


def _version(tensor):
    try:
        return tensor._version
    except RuntimeError:
        # Inference tensors have no counter. External raw/inference writes
        # cannot be detected by this metadata guard and are unsupported.
        return None


class PackedIndexCache:
    """One owner allocation; publication into the shared slot is adapter-owned.

    `generation` advances only after a nonempty official-byte write. Validation
    detects descriptor/storage/shape changes and ordinary Torch mutations made
    outside the writer. This is not a lock: concurrent forwards or external raw
    writes, including writes to inference tensors, are unsupported.
    """

    def __init__(self, packed, scales, owner_layer, ratio, generation=0):
        _integer(owner_layer, "owner_layer")
        _integer(ratio, "ratio", minimum=1)
        _integer(generation, "generation", maximum=2**63 - 1)
        if not isinstance(packed, torch.Tensor) or not isinstance(scales, torch.Tensor):
            raise TypeError("packed and scales must be tensors")
        if (packed.ndim != 3 or scales.ndim != 3 or packed.shape[-1] != 64
                or scales.shape[-1] != 4 or packed.shape[:2] != scales.shape[:2]):
            raise ValueError("cache must be packed[B,capacity,64] and scales[B,capacity,4]")
        if packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or packed.device != scales.device:
            raise ValueError("cache planes must be uint8 on the same device")
        if not packed.is_contiguous() or not scales.is_contiguous():
            raise ValueError("owner cache allocations must be contiguous")
        _integer(packed.shape[0], "batch_size", minimum=1)
        _integer(packed.shape[1], "capacity", minimum=1)
        if packed.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr():
            raise ValueError("packed and scale owners must have disjoint storage")
        self.packed, self.scales = packed, scales
        self.owner_layer, self.ratio, self.generation = owner_layer, ratio, generation
        self.batch_size, self.capacity = packed.shape[:2]
        self._descriptor = (owner_layer, ratio, self.batch_size, self.capacity)
        self._planes = (_identity(packed), _identity(scales))
        self._versions = (_version(packed), _version(scales))
        self._expected_generation = generation

    @property
    def storage_bytes(self):
        self.validate()
        return self.packed.numel() + self.scales.numel()

    def validate(self):
        descriptor = (self.owner_layer, self.ratio, self.batch_size, self.capacity)
        if any(type(value) is not int for value in descriptor) or descriptor != self._descriptor:
            raise RuntimeError("packed cache descriptor changed")
        if self.generation != self._expected_generation or type(self.generation) is not int:
            raise RuntimeError("packed cache generation changed outside the writer")
        if not isinstance(self.packed, torch.Tensor) or not isinstance(self.scales, torch.Tensor):
            raise RuntimeError("packed cache planes changed")
        if (_identity(self.packed), _identity(self.scales)) != self._planes:
            raise RuntimeError("packed cache storage metadata changed")
        if (_version(self.packed), _version(self.scales)) != self._versions:
            raise RuntimeError("packed cache changed outside the official-byte writer")
        return self

    def _written(self):
        self.generation += 1
        self._expected_generation = self.generation
        self._versions = (_version(self.packed), _version(self.scales))


def allocate_packed_index_cache(batch_size, capacity, *, device, owner_layer, ratio):
    """Zero keys, scale=1, matching the reference's initially zero BF16 cache."""
    _integer(batch_size, "batch_size", minimum=1)
    _integer(capacity, "capacity", minimum=1)
    packed = torch.zeros((batch_size, capacity, 64), dtype=torch.uint8, device=device)
    scales = torch.full((batch_size, capacity, 4), 127, dtype=torch.uint8, device=device)
    return PackedIndexCache(packed, scales, owner_layer, ratio)


def write_packed_index(cache, official_y, official_sf, *, start_slot, rows=None):
    """Copy official fp4_act_quant(k,32,False,E8M0) output, without conversion.

    Accept its packed FP4x2 / E8M0 dtypes or their uint8 byte views. A pair of
    ordinary same-current-stream copies precedes publication by the adapter.
    The caller must not publish on another stream without an explicit event.
    Nonfinite quantizer input is not repaired here; its policy belongs at the
    producer. Scale 255 decodes as canonical NaN, not as a finite substitute.
    """
    if not isinstance(cache, PackedIndexCache):
        raise TypeError("cache must be PackedIndexCache")
    cache.validate()
    _integer(start_slot, "start_slot", maximum=cache.capacity)
    if not isinstance(official_y, torch.Tensor) or not isinstance(official_sf, torch.Tensor):
        raise TypeError("official quantizer outputs must be tensors")
    y_types = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", None))
    sf_types = (torch.uint8, getattr(torch, "float8_e8m0fnu", None))
    if official_y.dtype not in y_types or official_sf.dtype not in sf_types:
        raise TypeError("writer accepts only packed E2M1x2/E8M0 output or raw uint8 views")
    if (official_y.ndim != 3 or official_sf.ndim != 3 or official_y.shape[-1] != 64
            or official_sf.shape[-1] != 4 or official_y.shape[:2] != official_sf.shape[:2]):
        raise ValueError("official outputs must be [batch,rows,64] and [batch,rows,4]")
    batch, available = official_y.shape[:2]
    if batch < 1 or batch > cache.batch_size or official_y.device != cache.packed.device or official_sf.device != cache.packed.device:
        raise ValueError("quantizer outputs have incompatible batch/device")
    if rows is None:
        rows = available
    _integer(rows, "rows", maximum=available)
    if start_slot + rows > cache.capacity:
        raise ValueError("packed write exceeds cache capacity")
    owners = {cache.packed.untyped_storage().data_ptr(), cache.scales.untyped_storage().data_ptr()}
    if any(t.untyped_storage().data_ptr() in owners for t in (official_y, official_sf)):
        raise ValueError("quantizer output must not alias either cache owner")
    if rows:
        _integer(cache.generation + 1, "next generation", maximum=2**63 - 1)
        cache.packed[:batch, start_slot:start_slot + rows].copy_(official_y[:, :rows].view(torch.uint8))
        cache.scales[:batch, start_slot:start_slot + rows].copy_(official_sf[:, :rows].view(torch.uint8))
        cache._written()


def _e2m1_e8m0_bits(codes, scales):
    """BF16 bit construction, avoiding FP32 multiply/FTZ in dequantization.

    Raw E8M0 codes 0..254 denote 2**(code-127); 255 is NaN. This includes
    subnormals, overflow to infinity, and signed zero. Finite official BF16
    producer inputs use scale codes 1..253, a strict subset of this decoder.
    NaN payload/sign identity is deliberately not promised for scale 255.
    """
    codes, scales = codes.to(torch.int32), scales.to(torch.int32)
    magnitude = codes & 7
    exponent = scales + (magnitude >> 1) - 1
    mantissa = torch.where(magnitude > 1, (magnitude & 1) << 6, 0)
    normal = (exponent << 7) | mantissa
    subnormal = (128 | mantissa) >> (1 - exponent).clamp(min=0, max=8)
    bits = torch.where(exponent > 0, normal, subnormal)
    bits = torch.where(exponent >= 255, 0x7F80, bits)
    bits = torch.where(magnitude == 0, 0, bits) | ((codes & 8) << 12)
    return torch.where(scales == 255, 0x7FC0, bits).to(torch.int16)


def unpack_index_tile(packed, scales):
    """Decode a SMALL uint8 [...,64]/[...,4] tile to BF16 [...,128]."""
    if not isinstance(packed, torch.Tensor) or not isinstance(scales, torch.Tensor):
        raise TypeError("packed tile and scales must be tensors")
    if (packed.ndim < 1 or scales.ndim != packed.ndim or packed.shape[-1] != 64
            or scales.shape[-1] != 4 or packed.shape[:-1] != scales.shape[:-1]):
        raise ValueError("tile shapes must be [...,64] and [...,4]")
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8 or packed.device != scales.device:
        raise ValueError("tile planes must be uint8 on the same device")
    if packed.numel() * 2 > _MAX_TILE_VALUES:
        raise ValueError("unpack_index_tile exceeds the bounded scratch limit")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    expanded_scales = scales.repeat_interleave(32, dim=-1)
    return _e2m1_e8m0_bits(codes, expanded_scales).view(torch.bfloat16)


def _score_contract(q, cache, weights, width, ids, query_chunk_size, position_chunk_size):
    if not isinstance(cache, PackedIndexCache):
        raise TypeError("cache must be PackedIndexCache")
    cache.validate()
    _integer(width, "width", maximum=cache.capacity)
    _integer(query_chunk_size, "query_chunk_size", minimum=1)
    _integer(position_chunk_size, "position_chunk_size", minimum=1)
    if query_chunk_size * position_chunk_size * 128 > _MAX_TILE_VALUES:
        raise ValueError("query/position tile exceeds bounded scratch limit")
    if not isinstance(q, torch.Tensor) or not isinstance(weights, torch.Tensor):
        raise TypeError("q and weights must be tensors")
    if q.ndim != 4 or q.shape[-1] != 128 or min(q.shape[:3]) < 1 or weights.shape != q.shape[:3]:
        raise ValueError("expected q[B,Q,H,128] and weights[B,Q,H]")
    if q.shape[0] > cache.batch_size:
        raise ValueError("query batch exceeds cache allocation")
    if (q.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16
            or q.device != cache.packed.device or weights.device != q.device):
        raise ValueError("q and weights must be BF16 on the cache device")
    if ids is not None:
        if (not isinstance(ids, torch.Tensor) or ids.ndim != 3 or ids.shape[:2] != q.shape[:2]
                or ids.dtype != torch.int32 or ids.device != q.device or ids.shape[-1] > width):
            raise ValueError("ids must be int32[B,Q,C<=width] on the query device")
    return q.shape[0], q.shape[1], q.shape[2], width if ids is None else ids.shape[-1]


def packed_index_scores(
    q, cache, weights, *, width, ids=None, reduce_fn=None,
    query_chunk_size=4, position_chunk_size=256, backend="torch",
):
    """Return BF16 [B,Q,width] dense or [B,Q,C] compact head-reduced scores.

    No entire-cache BF16 materialization. Scratch is bounded independently of
    context capacity and total query count; returned scores retain the original
    dense domain when ids=None. Causal masking, candidate selection and final
    full-width top-k remain outside this function, AFTER the collective.
    The callback is called once, in place, and must return None or the SAME
    tensor while preserving dtype/device/shape/strides/storage.
    """
    batch, queries, heads, columns = _score_contract(
        q, cache, weights, width, ids, query_chunk_size, position_chunk_size)
    if backend not in ("torch", "triton") or (reduce_fn is not None and not callable(reduce_fn)):
        raise ValueError("invalid backend or reduction callback")
    if backend == "triton":
        try:
            from .dsv41_packed_index_triton import packed_scores_triton
        except ImportError:
            from dsv41_packed_index_triton import packed_scores_triton
        output = packed_scores_triton(q, cache, weights, width=width, ids=ids)
    else:
        output = torch.empty((batch, queries, columns), dtype=torch.bfloat16, device=q.device)
        for start in range(0, batch * queries, query_chunk_size):
            rows = torch.arange(start, min(start + query_chunk_size, batch * queries), device=q.device)
            b, query = rows // queries, rows % queries
            for col in range(0, columns, position_chunk_size):
                stop = min(col + position_chunk_size, columns)
                if ids is None:
                    selected = torch.arange(col, stop, device=q.device).expand(rows.numel(), -1)
                else:
                    selected = ids[b, query, col:stop].to(torch.int64)
                valid = (selected >= 0) & (selected < width)
                # Even invalid IDs only gather an in-bounds initialized row.
                safe = selected.clamp(0, cache.capacity - 1)
                keys = unpack_index_tile(cache.packed[b[:, None], safe], cache.scales[b[:, None], safe])
                dot = torch.einsum("nhd,ncd->nhc", q[b, query], keys)
                product = dot.relu() * weights[b, query].unsqueeze(-1)
                values = product.sum(dim=1)
                output[b, query, col:stop] = values.masked_fill(~valid, 0)
    if reduce_fn is not None:
        identity = _identity(output)
        returned = reduce_fn(output)
        if returned is not None and returned is not output:
            raise TypeError("reduce_fn must return None or the same score tensor")
        if _identity(output) != identity or not output.is_contiguous():
            raise ValueError("reduce_fn changed score metadata/storage")
    cache.validate()
    return output
