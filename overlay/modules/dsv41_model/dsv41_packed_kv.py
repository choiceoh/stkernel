"""Packed E2M1/E4M3 compressed KV for the official V4.1 reference.

Store only the ORIGINAL fp4_act_quant(latent,16,False,E4M3) output. There is
no re-quantizer. D512 rows occupy 256 payload + 32 scale bytes, versus 1024
BF16 bytes. Even features occupy the low nibble. Finite raw products are
exactly representable in BF16, including signed zeros; NaN classification is
preserved but NaN payload/sign identity is not promised.

Sparse attention reconstructs only bounded 64-slot tiles and retains the
separate BF16 window pool. Its reference-ordered FP32 online softmax includes
BF16 probability rounding before PV and the final sink denominator. CPU
oracle/AOT evidence is not real-device numerical or performance validation.
"""
from __future__ import annotations

import torch

try:
    from . import dsv41_dual_sparse as _dual
except ImportError:
    import dsv41_dual_sparse as _dual

_INT32_MAX = 2**31 - 1
_MAX_TILE_VALUES = 32 * 64 * 512
_integer = _dual._integer


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


class PackedKVCache:
    """One owner allocation; publication into the shared slot is adapter-owned.

    `generation` advances only after a nonempty official-byte write. Validation
    detects descriptor/storage/shape changes and ordinary Torch mutations made
    outside the writer. This is not a lock: concurrent forwards or external raw
    writes, including writes to inference tensors, are unsupported.
    """

    def __init__(self, batch_size, capacity, device, owner_layer, ratio):
        _integer(batch_size, "batch_size", minimum=1)
        _integer(capacity, "capacity", minimum=1)
        _integer(owner_layer, "owner_layer")
        _integer(ratio, "ratio", minimum=1)
        packed = torch.zeros((batch_size, capacity, 256), dtype=torch.uint8, device=device)
        # Positive scale 1.0, raw E4M3 0x38: untouched slots reconstruct +0.
        scales = torch.full((batch_size, capacity, 32), 0x38, dtype=torch.uint8, device=device)
        generation = 0
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


def write_packed_kv(cache, official_y, official_sf, *, start_slot, rows=None):
    """Copy official fp4_act_quant(latent,16,False,E4M3) output, without conversion.

    Accept its packed FP4x2 / E4M3 dtypes or their uint8 byte views. Writes finish
    in stream order before attention reads the initialized prefix. The adapter
    may publish the descriptor before writing, matching the original source.
    Cross-stream consumers require an explicit event; concurrent use is unsupported.
    Nonfinite quantizer input is not repaired here; its policy belongs at the
    producer. Scale bytes 0x7f/0xff decode as canonical NaN, not a finite substitute.
    """
    if not isinstance(cache, PackedKVCache):
        raise TypeError("cache must be PackedKVCache")
    cache.validate()
    _integer(start_slot, "start_slot", maximum=cache.capacity)
    if not isinstance(official_y, torch.Tensor) or not isinstance(official_sf, torch.Tensor):
        raise TypeError("official quantizer outputs must be tensors")
    y_types = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", None))
    sf_types = (torch.uint8, getattr(torch, "float8_e4m3fn", None))
    if official_y.dtype not in y_types or official_sf.dtype not in sf_types:
        raise TypeError("writer accepts only packed E2M1x2/E4M3 output or raw uint8 views")
    if official_y.layout != torch.strided or official_sf.layout != torch.strided:
        raise ValueError("official quantizer outputs must be strided tensors")
    if (official_y.ndim != 3 or official_sf.ndim != 3 or official_y.shape[-1] != 256
            or official_sf.shape[-1] != 32 or official_y.shape[:2] != official_sf.shape[:2]):
        raise ValueError("official outputs must be [batch,rows,256] and [batch,rows,32]")
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


def _e2m1_e4m3_bits(codes, scales):
    """Exact finite-product BF16 bits, with canonical NaN for E4M3 NaN.

    E2M1 magnitudes in half units are 0,1,2,3,4,6,8,12. E4M3 is an integer
    significand times a power of two. Their product integer fits in eight
    bits, so its FP32->BF16 high bits are exact; exponent adjustment needs
    no floating multiply or subnormal/FTZ assumption. Raw signed scales are
    supported, although official finite producer scales are positive.
    """
    codes, scales = codes.to(torch.int32), scales.to(torch.int32)
    mag = codes & 7
    half_units = torch.where(mag < 2, mag, (2 + (mag & 1)) << ((mag >> 1) - 1).clamp(min=0))
    exponent, mantissa = (scales >> 3) & 15, scales & 7
    scale_units = torch.where(exponent == 0, mantissa, mantissa + 8)
    product = half_units * scale_units
    power = torch.where(exponent == 0, -10, exponent - 11)
    bits = (product.float().view(torch.int32) >> 16) + (power << 7)
    bits = torch.where(product == 0, 0, bits)
    bits = bits | (((codes & 8) << 12) ^ ((scales & 128) << 8))
    return torch.where((scales & 127) == 127, 0x7FC0, bits).to(torch.int16)


def unpack_kv_tile(packed, scales):
    """Decode a SMALL uint8 [...,256]/[...,32] tile to BF16 [...,512]."""
    if not isinstance(packed, torch.Tensor) or not isinstance(scales, torch.Tensor):
        raise TypeError("packed tile and scales must be tensors")
    if (packed.ndim < 1 or scales.ndim != packed.ndim or packed.shape[-1] != 256
            or scales.shape[-1] != 32 or packed.shape[:-1] != scales.shape[:-1]):
        raise ValueError("tile shapes must be [...,256] and [...,32]")
    if (packed.dtype != torch.uint8 or scales.dtype != torch.uint8
            or packed.device != scales.device or packed.layout != torch.strided
            or scales.layout != torch.strided):
        raise ValueError("tile planes must be strided uint8 on the same device")
    if packed.numel() * 2 > _MAX_TILE_VALUES:
        raise ValueError("unpack_kv_tile exceeds the bounded scratch limit")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    expanded_scales = scales.repeat_interleave(16, dim=-1)
    return _e2m1_e4m3_bits(codes, expanded_scales).view(torch.bfloat16)


def _contract(q, window_kv, cache, attn_sink, topk_idxs, softmax_scale, width, query_chunk_size):
    if not isinstance(cache, PackedKVCache):
        raise TypeError("cache must be PackedKVCache")
    cache.validate()
    batch, queries, heads, width_window, _, scale = _dual._contract(
        q, window_kv, None, attn_sink, topk_idxs, softmax_scale, query_chunk_size)
    _integer(width, "compressed width", maximum=cache.capacity)
    _integer(width_window + width, "merged KV width")
    if cache.batch_size < batch or cache.packed.device != q.device:
        raise ValueError("packed cache batch/device does not match q")
    return batch, queries, heads, width_window, width, scale


def _gather_tile(window_kv, cache, width, batch_ids, selected):
    """Only bounded 64-slot tiles are unpacked; no full-prefix BF16 copy."""
    width_window = window_kv.shape[1]
    valid_window = (selected >= 0) & (selected < width_window)
    valid_comp = (selected >= width_window) & (selected < width_window + width)
    keys = torch.zeros((*selected.shape, 512), dtype=torch.bfloat16, device=selected.device)
    if width_window:
        gathered = window_kv[batch_ids[:, None], selected.clamp(0, width_window - 1)]
        keys = torch.where(valid_window[..., None], gathered, keys)
    if width:
        local = (selected - width_window).clamp(0, width - 1)
        packed = cache.packed[batch_ids[:, None], local]
        scales = cache.scales[batch_ids[:, None], local]
        gathered = unpack_kv_tile(packed, scales)
        keys = torch.where(valid_comp[..., None], gathered, keys)
    return keys, valid_window | valid_comp


def packed_sparse_attn(
    q, window_kv, cache, attn_sink, topk_idxs, softmax_scale,
    *, width, backend="torch", query_chunk_size=1,
):
    """Return BF16[B,Q,H,512], preserving the merged window+compressed IDs.

    `width` is the initialized compressed prefix, not the allocation capacity.
    The adapter must establish its freshness before calling. Inputs/cache may
    not be concurrently mutated; inference/raw writes cannot be detected.
    Duplicate indices keep order. -1 is the reference invalid slot; other OOB
    IDs are safely invalidated but lie outside the reference's defined domain.
    K=0/all-invalid still execute the sink denominator, without zero shortcut.
    """
    batch, queries, heads, _, _, scale = _contract(
        q, window_kv, cache, attn_sink, topk_idxs, softmax_scale, width, query_chunk_size)
    if backend not in ("torch", "triton"):
        raise ValueError("backend must be 'torch' or explicit opt-in 'triton'")
    if backend == "triton":
        try:
            from .dsv41_packed_kv_triton import packed_sparse_triton
        except ImportError:
            from dsv41_packed_kv_triton import packed_sparse_triton
        return packed_sparse_triton(q, window_kv, cache, attn_sink, topk_idxs, scale, width=width)

    result = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    slots = topk_idxs.shape[-1]
    for first in range(0, batch * queries, query_chunk_size):
        rows = torch.arange(first, min(first + query_chunk_size, batch * queries), device=q.device)
        batch_ids, query_ids = rows // queries, rows % queries
        q_float = q[batch_ids, query_ids].float()
        maximum = torch.full((rows.numel(), heads), -1e30, dtype=torch.float32, device=q.device)
        denominator = torch.zeros_like(maximum)
        accumulated = torch.zeros_like(q_float)
        for start in range(0, slots, _dual._SLOTS):
            selected = torch.full((rows.numel(), _dual._SLOTS), -1, dtype=torch.int64, device=q.device)
            count = min(_dual._SLOTS, slots - start)
            selected[:, :count] = topk_idxs[batch_ids, query_ids, start:start + count]
            keys, valid = _gather_tile(window_kv, cache, width, batch_ids, selected)
            keys_float = keys.float()
            initial = torch.full((rows.numel(), heads, _dual._SLOTS), -torch.inf,
                                 dtype=torch.float32, device=q.device)
            initial.masked_fill_(valid[:, None, :], 0.0)
            # Seed QK with -inf before GEMM, preserving NaN*0 behavior of
            # invalid slots. A post-GEMM masked_fill would not be equivalent.
            scores = torch.baddbmm(initial, q_float, keys_float.transpose(1, 2)) * scale
            previous = maximum
            maximum = torch.maximum(previous, scores.amax(dim=-1))
            correction = torch.exp(previous - maximum)
            probability = torch.exp(scores - maximum[..., None])
            denominator = denominator * correction + probability.sum(dim=-1)
            accumulated = torch.baddbmm(
                accumulated * correction[..., None],
                probability.to(torch.bfloat16).float(), keys_float,
            )
        denominator = denominator + torch.exp(attn_sink[None, :] - maximum)
        result[batch_ids, query_ids] = (accumulated / denominator[..., None]).to(torch.bfloat16)
    cache.validate()
    return result
