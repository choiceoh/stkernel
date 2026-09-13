"""Bounded, lossless SF6 expansion for an offline prefill experiment.

The packed model remains sealed. Returned raw planes belong to one launch,
not the model or a process-wide cache; decode keeps its packed-scale reader.
No production dispatcher enables this experiment by default.
"""
from .moe_reform_sf_pack import REFORM_SF_BLOCK, REFORM_SF_STAGE, stage_shape


def plane_geometry(experts, rows, k, kind):
    if any(type(v) is not int or v <= 0 for v in (experts, rows, k)):
        raise ValueError("scale plane geometry must contain positive integers")
    row_tiles, k_tiles = stage_shape(rows, k, kind)
    return (experts, row_tiles * k_tiles, REFORM_SF_STAGE), experts * rows * k // 16


def expand_plane(packed, *, experts, rows, k, kind):
    """Restore the original TMA scale byte layout, including FC2 interleaving."""
    import torch
    from .moe_sf6_prefill_scales_kernel import expand

    shape, size = plane_geometry(experts, rows, k, kind)
    if (packed.dtype != torch.uint8 or not packed.is_cuda or not packed.is_contiguous()
            or tuple(packed.shape) != shape):
        raise ValueError("expected the exact CUDA SF6 packed plane")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("temporary SF6 expansion is an eager prefill experiment")
    out = torch.empty(size, dtype=torch.uint8, device=packed.device)
    expand[(size // REFORM_SF_BLOCK,)](
        packed, out, k // (256 if kind == 'fc1' else 128),
        FC2=kind == 'fc2', num_warps=4)
    return out


def expand_scales(scales, *, experts, hidden, intermediate):
    """Two owned planes; never install raw aliases on the immutable weight owner."""
    if not scales.enabled:
        raise ValueError("temporary expansion requires both admitted SF6 planes")
    first = expand_plane(scales.fc1, experts=experts, rows=2*intermediate,
                         k=hidden, kind='fc1')
    second = expand_plane(scales.fc2, experts=experts, rows=hidden,
                          k=intermediate, kind='fc2')
    return first, second
