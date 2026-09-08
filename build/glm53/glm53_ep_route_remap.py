"""One-launch remap for the opt-in eager E72/top8 prefill wrapper.

The caller has already admitted the expert-local prefill geometry and capture
state. Unsupported tensor metadata returns False so the existing Torch remap
keeps handling it. Launch failures propagate; they must not be hidden by a
second remap after a potentially failed device operation.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _remap_ep_local_kernel(
    IDS, WEIGHTS, EXPERT_MAP, OUT_IDS, OUT_WEIGHTS, N_PAIRS, LOCAL_OFFSET,
    MAP_LEN: tl.constexpr, HAS_MAP: tl.constexpr, BLOCK: tl.constexpr,
):
    slot = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = slot < N_PAIRS
    # Treat weights as bits: even local NaN payloads and signed zero must
    # survive the copy exactly, including fp16/bf16 storage.
    if WEIGHTS.dtype.element_ty == tl.float32:
        weight_bits = WEIGHTS.to(tl.pointer_type(tl.uint32))
        output_bits = OUT_WEIGHTS.to(tl.pointer_type(tl.uint32))
    else:
        weight_bits = WEIGHTS.to(tl.pointer_type(tl.uint16))
        output_bits = OUT_WEIGHTS.to(tl.pointer_type(tl.uint16))
    if HAS_MAP and MAP_LEN == 0:
        # The empty-map contract is independent of both input tensors. An
        # explicit specialization avoids dead-but-retained masked loads.
        tl.store(OUT_IDS + slot, 72, mask=live)
        tl.store(output_bits + slot, 0, mask=live)
    else:
        expert = tl.load(IDS + slot, mask=live, other=-1).to(tl.int64)
        if HAS_MAP:
            in_range = (expert >= 0) & (expert < MAP_LEN)
            local = tl.load(EXPERT_MAP + expert, mask=live & in_range, other=-1)
            remote = ~in_range | (local < 0)
        else:
            # Match the existing out_ids.copy_(ids); out_ids.sub_(offset)
            # order, including int32 conversion before offset subtraction.
            local = (expert.to(tl.int32) - LOCAL_OFFSET).to(tl.int32)
            remote = (expert < 0) | (local < 0) | (local >= 72)
        # Remote weights become positive zero without reading their storage.
        # Load local weights as bits so NaN payloads and signed zero survive.
        weight = tl.load(weight_bits + slot, mask=live & ~remote, other=0)
        tl.store(OUT_IDS + slot, tl.where(remote, 72, local).to(tl.int32), mask=live)
        tl.store(output_bits + slot, weight, mask=live)


def _ep_route_remap_metadata(
    topk_ids, topk_weights, *, expert_map, num_local_experts,
    local_expert_offset, out_ids, out_scales,
):
    """Validate one call and return its launch sizes without caching tensors."""
    if (num_local_experts != 72 or type(local_expert_offset) is not int
            or not 0 <= local_expert_offset <= (1 << 31) - 1):
        return None
    tensors = (topk_ids, topk_weights, out_ids, out_scales)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return None
    shape = topk_ids.shape
    if len(shape) != 2 or not 4096 <= shape[0] <= 16384 or shape[1] != 8:
        return None
    weight_dtype = topk_weights.dtype
    if (topk_ids.dtype not in (torch.int32, torch.int64)
            or weight_dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or out_ids.dtype != torch.int32 or out_scales.dtype != weight_dtype):
        return None
    if not topk_ids.is_cuda or not topk_ids.is_contiguous():
        return None
    device = topk_ids.device
    if any(t.shape != shape or t.device != device or not t.is_contiguous()
           for t in (topk_weights, out_ids, out_scales)):
        return None
    map_len = 0
    if expert_map is not None:
        if (not isinstance(expert_map, torch.Tensor) or expert_map.ndim != 1
                or expert_map.dtype not in (torch.int32, torch.int64)
                or expert_map.device != device or not expert_map.is_contiguous()):
            return None
        map_len = expert_map.numel()
        if map_len > (1 << 31) - 1:
            return None
    return shape[0] * shape[1], map_len


def ep_route_remap_supported(
    topk_ids, topk_weights, *, expert_map, num_local_experts,
    local_expert_offset, out_ids, out_scales,
):
    """Check metadata without copying data, synchronizing, or probing CUDA."""
    return _ep_route_remap_metadata(
        topk_ids, topk_weights, expert_map=expert_map,
        num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,
        out_ids=out_ids, out_scales=out_scales,
    ) is not None


def try_remap_ep_local(
    topk_ids, topk_weights, *, expert_map, num_local_experts,
    local_expert_offset, out_ids, out_scales,
):
    """Write existing route scratch in one launch; False requests Torch fallback."""
    metadata = _ep_route_remap_metadata(
        topk_ids, topk_weights, expert_map=expert_map,
        num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,
        out_ids=out_ids, out_scales=out_scales,
    )
    if metadata is None:
        return False
    num_pairs, map_len = metadata
    has_map = expert_map is not None
    # Empty-map and offset specializations never need a map load. Give them a
    # valid typed pointer instead of relying on a zero-sized tensor's nullptr.
    map_tensor = expert_map if map_len else out_ids
    _remap_ep_local_kernel[(triton.cdiv(num_pairs, 256),)](
        topk_ids, topk_weights, map_tensor, out_ids, out_scales,
        num_pairs, local_expert_offset,
        MAP_LEN=map_len, HAS_MAP=has_map, BLOCK=256, num_warps=4,
    )
    return True


@triton.jit
def _prepare_ep_short_decode_kernel(
    X, IDS, WEIGHTS, EXPERT_MAP, PAD_X, PAD_IDS, PAD_WEIGHTS, LOCAL_OFFSET,
    MAP_LEN: tl.constexpr, HAS_MAP: tl.constexpr,
):
    # Keep the established M8 input exactly: six real rows, then two copies
    # of row zero with positive-zero router weights. Each CTA owns one row.
    row = tl.program_id(0)
    source_row = tl.where(row < 6, row, 0)
    columns = tl.arange(0, 4096)
    x_bits = X.to(tl.pointer_type(tl.uint16))
    pad_x_bits = PAD_X.to(tl.pointer_type(tl.uint16))
    bits = tl.load(x_bits + source_row * 4096 + columns)
    tl.store(pad_x_bits + row * 4096 + columns, bits)

    slots = tl.arange(0, 8)
    source_slot = source_row * 8 + slots
    destination = row * 8 + slots
    if WEIGHTS.dtype.element_ty == tl.float32:
        weight_bits = WEIGHTS.to(tl.pointer_type(tl.uint32))
        pad_weight_bits = PAD_WEIGHTS.to(tl.pointer_type(tl.uint32))
    else:
        weight_bits = WEIGHTS.to(tl.pointer_type(tl.uint16))
        pad_weight_bits = PAD_WEIGHTS.to(tl.pointer_type(tl.uint16))
    if HAS_MAP and MAP_LEN == 0:
        tl.store(PAD_IDS + destination, 72)
        tl.store(pad_weight_bits + destination, 0)
    else:
        expert = tl.load(IDS + source_slot).to(tl.int64)
        if HAS_MAP:
            in_range = (expert >= 0) & (expert < MAP_LEN)
            local = tl.load(EXPERT_MAP + expert, mask=in_range, other=-1)
            remote = ~in_range | (local < 0)
        else:
            local = (expert.to(tl.int32) - LOCAL_OFFSET).to(tl.int32)
            remote = (expert < 0) | (local < 0) | (local >= 72)
        weights = tl.load(weight_bits + source_slot,
                          mask=(row < 6) & ~remote, other=0)
        tl.store(PAD_IDS + destination, tl.where(remote, 72, local).to(tl.int32))
        tl.store(pad_weight_bits + destination, weights)


def _ep_short_decode_metadata(
    hidden_states, topk_ids, topk_weights, *, expert_map,
    num_local_experts, local_expert_offset,
):
    """Input-only admission before allocating the existing padded scratch."""
    if (num_local_experts != 72 or type(local_expert_offset) is not int
            or not 0 <= local_expert_offset <= (1 << 31) - 1):
        return None
    tensors = (hidden_states, topk_ids, topk_weights)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return None
    if (hidden_states.shape != (6, 4096) or hidden_states.dtype != torch.bfloat16
            or topk_ids.shape != (6, 8) or topk_weights.shape != (6, 8)
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_weights.dtype not in (torch.float32, torch.float16, torch.bfloat16)):
        return None
    device = hidden_states.device
    if not hidden_states.is_cuda or any(
            t.device != device or not t.is_contiguous() for t in tensors):
        return None
    map_len = 0
    if expert_map is not None:
        if (not isinstance(expert_map, torch.Tensor) or expert_map.ndim != 1
                or expert_map.dtype not in (torch.int32, torch.int64)
                or expert_map.device != device or not expert_map.is_contiguous()):
            return None
        map_len = expert_map.numel()
        if map_len > (1 << 31) - 1:
            return None
    return map_len


def ep_short_decode_prepare_supported(
    hidden_states, topk_ids, topk_weights, *, expert_map,
    num_local_experts, local_expert_offset,
):
    return _ep_short_decode_metadata(
        hidden_states, topk_ids, topk_weights, expert_map=expert_map,
        num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,
    ) is not None


def try_prepare_ep_short_decode(
    hidden_states, topk_ids, topk_weights, *, expert_map,
    num_local_experts, local_expert_offset, pad_x, pad_ids, pad_weights,
):
    """Fuse the exact six-row remap and existing M8 staging, without caching."""
    map_len = _ep_short_decode_metadata(
        hidden_states, topk_ids, topk_weights, expert_map=expert_map,
        num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,
    )
    if map_len is None:
        return False
    outputs = (pad_x, pad_ids, pad_weights)
    for tensor, shape, dtype in (
        (pad_x, (8, 4096), torch.bfloat16),
        (pad_ids, (8, 8), torch.int32),
        (pad_weights, (8, 8), topk_weights.dtype),
    ):
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                or tensor.dtype != dtype or tensor.device != hidden_states.device
                or not tensor.is_contiguous()):
            return False
    # Reject partial aliases too. All admitted tensors are dense, so byte
    # intervals include storage offsets without inspecting any device value.
    inputs = (hidden_states, topk_ids, topk_weights)
    if expert_map is not None:
        inputs += (expert_map,)
    for index, output in enumerate(outputs):
        start = output.data_ptr()
        end = start + output.numel() * output.element_size()
        for other in inputs + outputs[:index]:
            other_start = other.data_ptr()
            other_end = other_start + other.numel() * other.element_size()
            if other_start < other_end and start < other_end and other_start < end:
                return False
    map_tensor = expert_map if map_len else pad_ids
    _prepare_ep_short_decode_kernel[(8,)](
        hidden_states, topk_ids, topk_weights, map_tensor,
        pad_x, pad_ids, pad_weights, local_expert_offset,
        MAP_LEN=map_len, HAS_MAP=expert_map is not None, num_warps=4,
    )
    return True
