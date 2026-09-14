"""Explicit FP8-v3 input for the ordinary long-prefill SF6 MoE body."""
import torch

from engine.modules.prefill_packets import PacketBatch, ffn_packet_rows


def supported(weights, input_gs, down_scale, rows):
    """Inspect the actual prepared owner before agreeing the FFN plan."""
    if (not ffn_packet_rows(rows) or not weights.tiled
            or weights.reform_scales is None or not weights.reform_scales.enabled
            or input_gs is None or input_gs.numel() != 288
            or down_scale is None or down_scale.numel() not in (1, 288)
            or tuple(weights.w1_storage.shape) != (288, 1024, 2048)
            or tuple(weights.w2_storage.shape) != (288, 4096, 256)):
        return False
    tensors = (input_gs, down_scale, weights.w1_alpha, weights.w2_alpha,
               weights.w1_storage, weights.w2_storage, weights.sfb1_packed, weights.sfb2_packed)
    device = input_gs.device
    if any(t is None or not t.is_cuda or t.device != device or not t.is_contiguous() for t in tensors):
        return False
    if torch.cuda.is_current_stream_capturing() or torch.cuda.get_device_capability(device) != (12, 1):
        return False
    from . import moe_dispatch as md
    from .moe_dynamic_prefill_packets import stock_contract_matches
    if md._GLM53_B12X_PREFILL_REUSE or md._GLM53_B12X_PREFILL_FC1_N128 or not stock_contract_matches():
        return False
    geometry = dict(num_experts=288, num_local_experts=288, hidden_size=4096,
                    intermediate_size=512, num_topk=8, quant_mode='nvfp4',
                    activation='swigluoai_uninterleave', swiglu_limit=10.)
    return (md.select_sm120_moe_backend(num_tokens=rows, **geometry) == 'dynamic'
            and md._dynamic_workspace_tile_m(routed_rows=rows*8, state_E=288, weight_E=288,
                    k=4096, n=512, num_topk=8, quant_mode='nvfp4',
                    activation='swigluoai_uninterleave', swiglu_limit=10.) == 128)


def launch(batch, ids, route_weights, weights, input_gs, down_scale, *, workspace=None):
    if not isinstance(batch, PacketBatch) or not supported(weights, input_gs, down_scale, batch.geometry.rows):
        raise ValueError('packet MoE must be agreed with supported prepared SF6 weights before transport')
    from . import moe_dispatch as md
    rows, device = batch.geometry.rows, batch.received.device
    if device != input_gs.device:
        raise ValueError('packet and prepared MoE weights must share a device')
    if workspace is None:
        # Same eager workspace as the ordinary BF16 path. The two variants
        # are serialized on the current stream and never keep its outputs.
        workspace = md._get_cached_workspace(backend='dynamic', state_E=288, weight_E=288,
            routed_rows=rows*8, k=4096, n=512, num_topk=8, device=device,
            activation_precision='fp4', quant_mode='nvfp4',
            activation='swigluoai_uninterleave', swiglu_limit=10.)
    out = torch.empty((rows, 4096), device=device, dtype=torch.bfloat16)
    return md.launch_sm120_dynamic_moe(workspace=workspace, weights=weights, a=None,
        topk_ids=ids.contiguous(), topk_weights=route_weights.contiguous(),
        input_gs=input_gs, down_input_scale=down_scale, scatter_output=out,
        num_experts=288, num_tokens=rows, k=4096, n=512, top_k=8,
        activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.,
        _packet_input=batch)
