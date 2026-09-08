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


def ep_route_remap_supported(
    topk_ids, topk_weights, *, expert_map, num_local_experts,
    local_expert_offset, out_ids, out_scales,
):
    """Check metadata without copying data, synchronizing, or probing CUDA."""
    if (num_local_experts != 72 or type(local_expert_offset) is not int
            or not 0 <= local_expert_offset <= (1 << 31) - 1):
        return False
    tensors = (topk_ids, topk_weights, out_ids, out_scales)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return False
    shape = tuple(topk_ids.shape)
    if len(shape) != 2 or not 4096 <= shape[0] <= 16384 or shape[1] != 8:
        return False
    if (topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_weights.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or out_ids.dtype != torch.int32 or out_scales.dtype != topk_weights.dtype):
        return False
    if not topk_ids.is_cuda:
        return False
    if any(tuple(t.shape) != shape or t.device != topk_ids.device
           or not t.is_contiguous() for t in tensors):
        return False
    if expert_map is not None:
        if (not isinstance(expert_map, torch.Tensor) or expert_map.ndim != 1
                or expert_map.dtype not in (torch.int32, torch.int64)
                or expert_map.device != topk_ids.device or not expert_map.is_contiguous()
                or expert_map.numel() > (1 << 31) - 1):
            return False
    return True


def try_remap_ep_local(
    topk_ids, topk_weights, *, expert_map, num_local_experts,
    local_expert_offset, out_ids, out_scales,
):
    """Write existing route scratch in one launch; False requests Torch fallback."""
    if not ep_route_remap_supported(
        topk_ids, topk_weights, expert_map=expert_map,
        num_local_experts=num_local_experts, local_expert_offset=local_expert_offset,
        out_ids=out_ids, out_scales=out_scales,
    ):
        return False
    has_map = expert_map is not None
    map_len = expert_map.numel() if has_map else 0
    # Empty-map and offset specializations never need a map load. Give them a
    # valid typed pointer instead of relying on a zero-sized tensor's nullptr.
    map_tensor = expert_map if map_len else out_ids
    _remap_ep_local_kernel[(triton.cdiv(topk_ids.numel(), 256),)](
        topk_ids, topk_weights, map_tensor, out_ids, out_scales,
        topk_ids.numel(), local_expert_offset,
        MAP_LEN=map_len, HAS_MAP=has_map, BLOCK=256, num_warps=4,
    )
    return True
