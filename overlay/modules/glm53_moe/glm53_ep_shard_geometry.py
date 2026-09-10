"""Exact routed-expert shard geometry without loader or CUDA side effects.

E144/I1024 is a MoE-only logical shard. Attention/shared experts and the
terminal output sum still belong to the original four-rank group.
"""

import os

HYBRID_KNOB = "VLLM_GLM53_EP_HYBRID_TP2"

HYBRID_TAG = "glm53_ep2tp2_tiled_e144_i1024_v1"


def ep_shard_geometry(num_local_experts=72, intermediate_size=2048):
    if (type(num_local_experts) is not int or type(intermediate_size) is not int
            or (num_local_experts, intermediate_size) not in ((72, 2048), (144, 1024))):
        raise ValueError("EP tiled shard requires exactly E72/I2048 or E144/I1024")
    e, i, h = num_local_experts, intermediate_size, 4096
    return dict(E=e, I=i, H=h, top_k=8, hybrid=e == 144,
                sentinel=e, native_slices=i // 128, dynamic_groups=i // 512,
                raw_w13=(e, 2*i, h//2), raw_down=(e, h, i//2),
                cute_w13=(2*i, 512, h//512, e),
                cute_down=(h, 128, i//128, e),
                torch_w13=(2*i, 256, h//512, e),
                torch_down=(h, 64, i//128, e),
                torch_w13_stride=(256, 1, 2*i*256, 2*i*h//2),
                torch_down_stride=(64, 1, h*64, h*i//2),
                sf6_fc1=(e, (2*i//128)*(h//256), 1552),
                sf6_fc2=(e, (h//256)*(i//128), 1552),
                raw_sf1_bytes=e*2*i*h//16,
                raw_sf2_bytes=e*h*i//16)


def ep_shard_cache_suffix(shard):
    return (HYBRID_TAG, shard["E"], shard["I"]) if shard["hybrid"] else ()


def require_hybrid_mode(shard, reform_sf_pack, decode_opt):
    # Do not silently inherit the failed v5 optimization or broaden raw scales.
    if shard["hybrid"] and (reform_sf_pack is not True or decode_opt is not False):
        raise ValueError("Hybrid EP requires SF6 and explicit decode_opt=False")


def owner_contract(owner):
    """Geometry only, not a replacement for loader/canary/finalization proof."""
    shard = ep_shard_geometry(owner.num_local_experts,
                              owner.intermediate_size_per_partition)
    if (owner._use_ep is not True or owner._ep_no_dummy is not True
            or owner.global_num_experts != 288 or owner.hidden_dim != 4096
            or owner.topk != 8):
        raise ValueError("Owner is not the declared global288/H4096/top8 EP shard")
    enabled = os.environ.get(HYBRID_KNOB, "0")
    if enabled not in ("0", "1"):
        raise ValueError("Hybrid flag must be exactly 0 or 1")
    identity = getattr(owner, "_glm53_hybrid_loader_identity", None)
    if shard["hybrid"]:
        if enabled != "1" or os.environ.get("VLLM_GLM53_EP_DECODE_OPT", "0") != "0":
            raise ValueError("Hybrid requires explicit selection and baseline decode implementation")
        hybrid_rank_contract(owner)
    elif enabled != "0" or identity is not None:
        raise ValueError("Baseline E72 owner cannot carry hybrid selection or loader identity")
    return shard

def hybrid_rank_contract(owner):
    """Validate a loader seal, never infer topology from the local device index.

    The factory authenticates the physical rank. This owner checks the exact
    forwarded identity and logical groups; it cannot prove collective execution.
    """
    identity = getattr(owner, "_glm53_hybrid_loader_identity", None)
    if (type(identity) is not tuple or len(identity) != 13
            or identity[0] != "glm53_ep2_tp2_loader_v1"
            or any(type(value) is not int for value in identity[1:])):
        raise ValueError("Hybrid requires the exact immutable loader identity")
    p = identity[6]
    if not 0 <= p < 4:
        raise ValueError("Hybrid physical rank must be in 0..3")
    ep, tp = p // 2, p % 2
    expected = ("glm53_ep2_tp2_loader_v1", 288, 144, 4096, 2048, 1024,
                p, ep, tp, 144*ep, 144*(ep+1), 1024*tp, 1024*(tp+1))
    parallel = getattr(owner, "_glm53_ep_parallelism", None)
    if (identity != expected or type(parallel) is not tuple
            or any(type(value) is not int for value in parallel)
            or parallel != (2, tp, 2, ep)
            or type(owner.local_expert_offset) is not int
            or owner.local_expert_offset != 144*ep):
        raise ValueError("Hybrid loader/rank/offset geometry differs")
    return dict(physical_rank=p, world_size=4, ep_size=2, ep_rank=ep,
                tp_size=2, tp_rank=tp, expert_start=144*ep, expert_stop=144*(ep+1),
                intermediate_start=1024*tp, intermediate_stop=1024*(tp+1),
                attention_shared_tp_size=4, terminal_output_sum_size=4,
                collective_execution_verified=False)

def memory_contract(shard, max_rows=256):
    if type(max_rows) is not int or max_rows <= 0 or max_rows % 128:
        raise ValueError("Static rows must remain positive and 128-aligned")
    e, i, h = shard["E"], shard["I"], shard["H"]
    weight = e * 3 * i * h // 2
    sf6 = (shard["sf6_fc1"][0] * shard["sf6_fc1"][1]
           + shard["sf6_fc2"][0] * shard["sf6_fc2"][1]) * 1552
    return dict(routed_weight_bytes=weight, sf6_bytes=sf6,
                static_q0_bytes=e*max_rows*(h//2+h//16),
                static_route_table_bytes=e*max_rows*8,
                static_expert_vector_bytes=e*4,
                workspace_shared_across_layers=True,
                excludes_dynamic_padding_alphas_and_allocator=True)
