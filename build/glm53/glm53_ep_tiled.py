"""Opt-in EP tile-major owner shared by static decode and local prefill.

The only weight copy is permuted once during loading. Every inference call
uses an explicit tiled entry; no row-major backend may consume this owner.
Imports are inert so admission and lifetime contracts can be checked on CPU.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from threading import RLock
from weakref import WeakValueDictionary


KNOB = "VLLM_GLM53_EP_TILED"
STATIC_MAX_TOKENS = 32
MAX_TOKENS = 16384
_WORKSPACES = WeakValueDictionary()
_LOCK = RLock()
_LAUNCHED = set()
_LAUNCH_PREFIXES = {
    "decode": "[ep-tiled] LAUNCHED decode E72/H4096/I2048/top8 T=",
    "prefill": "[ep-tiled] LAUNCHED prefill E72/H4096/I2048/top8 T=",
}


def validate_configuration(owner):
    expected = (True, True, 288, 72, 4096, 2048, 8,
                "swigluoai_uninterleave", 1., 0., 10.)
    actual = (owner._use_ep, owner._ep_no_dummy, owner.global_num_experts,
              owner.num_local_experts, owner.hidden_dim,
              owner.intermediate_size_per_partition, owner.topk,
              owner._activation_str, owner._swiglu_alpha,
              owner._swiglu_beta, owner._swiglu_limit)
    if actual != expected:
        raise ValueError(f"{KNOB}=1 requires GLM EP4 {expected}; got {actual}")
    if type(owner.max_num_tokens) is not int or not 8192 <= owner.max_num_tokens <= MAX_TOKENS:
        raise ValueError(f"{KNOB}=1 requires max_num_tokens in 8192..{MAX_TOKENS}")
    if owner._ep_stock_topk_micro or owner._ep_disable_micro:
        raise ValueError(f"{KNOB}=1 cannot combine with alternate EP micro modes")


def weight_generation(w1, w2, owner):
    tensors = [w1, w2, owner.w1_scale, owner.w2_scale,
               owner.w1_sf_mma, owner.w2_sf_mma, owner.g1_alphas,
               owner.g2_alphas, owner._fc2_input_scale]
    views = getattr(owner, "_ep_tiled_weight_views", None)
    if views is not None:
        tensors.extend((views._w13_sf_storage, views._down_sf_storage,
                        views.w1_alpha, views.w2_alpha))
    result = []
    for tensor in tensors:
        try:
            version = tensor._version
        except RuntimeError:
            version = -1  # Inference tensors are immutable after loading.
        result.append((tensor.data_ptr(), version, tuple(tensor.shape),
                       tuple(tensor.stride()), tensor.dtype, tensor.device))
    return tuple(result)


def _require_output_disjoint(output, tensors):
    begin = output.data_ptr()
    end = begin + output.numel() * output.element_size()
    for tensor in tensors:
        if tensor is None or tensor.device != output.device:
            continue
        other = tensor.data_ptr()
        if begin < other + tensor.numel() * tensor.element_size() and other < end:
            raise ValueError("tiled EP output overlaps input, weights, or owned scratch")


@dataclass
class _Workspace:
    static: object
    dynamic: object
    scratch: object


def _shared_workspace(device, capacity):
    import torch
    from . import moe_dispatch as md
    from .moe_static_ep_tiled import allocate_ep_tiled_decode_scratch, warm_ep_tiled_decode

    key = (str(device), capacity)
    with _LOCK:
        workspace = _WORKSPACES.get(key)
        if workspace is None:
            static = md.allocate_sm120_static_workspace(
                state_E=72, weight_E=72, max_rows=STATIC_MAX_TOKENS * 8,
                k=4096, n=2048, num_topk=8, device=device)
            dynamic = md.allocate_sm120_dynamic_workspace(
                state_E=72, weight_E=72, routed_rows=capacity * 8,
                k=4096, n=2048, num_topk=8, device=device,
                activation="swigluoai_uninterleave", tile_m=128)
            dynamic.ep_tiled = True
            dynamic.ep_scatter_fp32 = torch.empty(
                (capacity, 4096), dtype=torch.float32, device=device)
            scratch = allocate_ep_tiled_decode_scratch(device=device)
            warm_ep_tiled_decode()
            md._get_dynamic_kernel(
                72, capacity, 4096, 2048, 8, dynamic.max_rows,
                activation="swigluoai_uninterleave", swiglu_alpha=1.,
                swiglu_beta=0., swiglu_limit=10., tile_m=128, tiled=True)
            workspace = _Workspace(static, dynamic, scratch)
            _WORKSPACES[key] = workspace
        return workspace


def prepare_ep_tiled(owner, layer):
    """Seal one loaded layer and run the first actual-weight startup canary."""
    import torch
    from . import moe_dispatch as md
    from .glm53_ep_tiled_selftest import before_relayout, after_relayout

    validate_configuration(owner)
    owner._ep_tiled_ready = False
    w1, w2 = layer.w13_weight, layer.w2_weight
    if (torch.cuda.get_device_capability(w1.device) != (12, 1)
            or owner.out_dtype != torch.bfloat16):
        raise ValueError("tiled EP requires SM121 and BF16 activations")
    if (tuple(w1.shape), tuple(w2.shape)) != ((72, 4096, 2048), (72, 4096, 1024)):
        raise ValueError("tiled EP loaded weight geometry differs from E72/H4096/I2048")
    if w1.dtype != torch.uint8 or w2.dtype != torch.uint8:
        raise ValueError("tiled EP requires packed NVFP4 uint8 weights")
    if (md._FORCED_BACKEND is not None
            or md._GLM53_B12X_FORCE_BACKEND is not None
            or md._STATIC_V2_OVERRIDE is not None
            or os.environ.get(md._FORCE_MOE_W4A16_ENV, "0") == "1"):
        raise ValueError("tiled EP owns its static/dynamic dispatch; unset forced backend")
    md.invalidate_tile_major_if_reloaded(w1, w2)
    reference = before_relayout(owner, layer)
    md.tile_expert_weights_inplace(w1, w2)
    owner._ep_tiled_weight_views = md._get_weight_views(
        w1, owner.w1_sf_mma, w2, owner.w2_sf_mma,
        owner.g1_alphas, owner.g2_alphas, n=2048, k=4096,
        tiled=True, reform_sf_pack=False)
    views = owner._ep_tiled_weight_views
    if (views.w13_tiled_storage.data_ptr() != w1.data_ptr()
            or views.w2_tiled_storage.data_ptr() != w2.data_ptr()):
        raise RuntimeError("tiled EP must alias the single loaded weight storage")
    owner._ep_tiled_workspace = _shared_workspace(w1.device, owner.max_num_tokens)
    owner._ep_tiled_generation = weight_generation(w1, w2, owner)
    # All serving remap planes are allocated before capture. int32 and FP32
    # are the explicit ABI of both new kernels, independent of router dtype.
    owner._ensure_ep_scratch(w1.device, torch.float32, torch.int32)
    owner._ep_tiled_canary_active = True
    try:
        after_relayout(owner, layer, reference)
    finally:
        owner._ep_tiled_canary_active = False
    owner._ep_tiled_ready = True


def launch_ep_tiled(owner, output, x, w1, w2, ids, scales, expert_map):
    import torch
    from . import moe_dispatch as md
    from .moe_static_ep_tiled import launch_ep_tiled_decode
    from .glm53_ep_route_remap import try_remap_ep_local

    if not (getattr(owner, "_ep_tiled_ready", False)
            or getattr(owner, "_ep_tiled_canary_active", False)):
        raise RuntimeError("tiled EP startup validation has not completed")
    tokens = x.shape[0]
    if (type(tokens) is not int or not 1 <= tokens <= owner.max_num_tokens
            or tuple(x.shape) != (tokens, 4096) or tuple(output.shape) != tuple(x.shape)
            or x.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
            or not x.is_contiguous() or not output.is_contiguous()
            or x.device != w1.device or output.device != x.device
            or tuple(ids.shape) != (tokens, 8) or tuple(scales.shape) != (tokens, 8)):
        raise ValueError("tiled EP requires bounded contiguous CUDA BF16 [T,4096] and top8 routes")
    if (weight_generation(w1, w2, owner) != owner._ep_tiled_generation
            or getattr(w1, "_b12x_tile_major", None) != "plain"
            or getattr(w2, "_b12x_tile_major", None) != "plain"):
        raise RuntimeError("tiled EP weights changed after preparation; reload the model")
    if ids.dtype not in (torch.int32, torch.int64) or scales.dtype != torch.float32:
        raise ValueError("tiled EP requires int32/int64 route IDs and FP32 route weights")
    if (ids.device != x.device or scales.device != x.device
            or not ids.is_contiguous() or not scales.is_contiguous()):
        raise ValueError("tiled EP routes must be contiguous on the activation device")
    workspace = owner._ep_tiled_workspace
    _require_output_disjoint(output, (
        x, ids, scales, expert_map, w1, w2, owner._ep_ids, owner._ep_scales,
        workspace.scratch.scatter_fp32, workspace.dynamic.ep_scatter_fp32,
        owner.w1_scale, owner.w2_scale, owner.w1_sf_mma, owner.w2_sf_mma,
        owner.g1_alphas, owner.g2_alphas, owner._fc2_input_scale,
        owner._ep_tiled_weight_views._w13_sf_storage,
        owner._ep_tiled_weight_views._down_sf_storage,
        owner._ep_tiled_weight_views.w1_alpha, owner._ep_tiled_weight_views.w2_alpha))
    local_ids, local_scales = owner._ep_ids[:tokens], owner._ep_scales[:tokens]
    if not try_remap_ep_local(
            ids, scales, expert_map=expert_map, num_local_experts=72,
            local_expert_offset=owner.local_expert_offset,
            out_ids=local_ids, out_scales=local_scales, _tiled_owner=True):
        raise ValueError("tiled EP requires its prepared one-launch remap contract")
    weights = owner._ep_tiled_weight_views
    if tokens <= STATIC_MAX_TOKENS:
        launch_ep_tiled_decode(
            workspace=workspace.static, weights=weights, a=x,
            topk_ids=local_ids, topk_weights=local_scales,
            input_gs=owner.g1_alphas, down_input_scale=owner._fc2_input_scale,
            output=output, scratch=workspace.scratch)
        lane = "decode"
    else:
        md.launch_sm120_dynamic_moe(
            workspace=workspace.dynamic, weights=weights, a=x,
            topk_ids=local_ids, topk_weights=local_scales,
            input_gs=owner.g1_alphas, down_input_scale=owner._fc2_input_scale,
            scatter_output=output, num_experts=72, num_tokens=tokens,
            k=4096, n=2048, top_k=8, activation="swigluoai_uninterleave",
            swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        lane = "prefill"
    if (lane not in _LAUNCHED and not getattr(owner, "_ep_tiled_canary_active", False)
            and not torch.cuda.is_current_stream_capturing()):
        print(_LAUNCH_PREFIXES[lane] + str(tokens), flush=True)
        _LAUNCHED.add(lane)
    return output
