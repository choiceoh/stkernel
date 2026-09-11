"""Re-hosting meter (profile): every vLLM symbol the served GLM-5.3 files
import, against what the engine does instead -- so "얼마나 남았나" is a
number, not a feeling.

The served path is three files (glm5next_model, glm5next_kda,
glm5next_attention; mtp is not served -- the fleet drafts with DFlash2 --
and multimodal is text-only dropped). The engine did not re-host them line
by line: `net.py` is a fresh composition on `base/` and `modules/`, so each
served import maps to the engine construct that replaced it (`ours`) or to
generality the engine does not carry (`drop`). `real` is what is still
neither: kernels whose served lane has no engine binding yet.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = ["glm5next_model.py", "glm5next_kda.py", "glm5next_attention.py"]

# vllm symbol -> what the engine has instead (None = not yet ours)
MAP = {
    # config / context / logging: facts, arguments, instruments
    "VllmConfig": "profiles/glm53/facts.py (Facts, asserted at load)", "CacheConfig": "facts.py + base/cache_spec",
    "get_current_vllm_config": "facts.py", "Glm5NextConfig": "facts.py", "KimiLinearConfig": "facts.py",
    "ParallelConfig": "base/comm (TP=4 fixed)", "get_forward_context": "net.py: a step's metadata is its arguments",
    "is_forward_context_available": "net.py: no context", "GDNAttentionMetadata": "net.py prefill args (ctx, slot, caches)",
    "init_logger": "base/instruments.Recorder",
    # distributed
    "get_tensor_model_parallel_world_size": "base/comm.Comm.world_size", "get_tensor_model_parallel_rank": "base/comm.Comm.rank",
    "tensor_model_parallel_all_reduce": "base/comm.Comm.all_reduce", "tensor_model_parallel_all_gather": "base/comm.Comm.all_gather",
    # layers the engine composes from views
    "ColumnParallelLinear": "specs.py row split + F.linear on a view", "RowParallelLinear": "specs.py col split + F.linear + all_reduce",
    "MergedColumnParallelLinear": "specs.py merges at preshard (in_proj, gate_up, qkv_a)", "ReplicatedLinear": "specs.py whole",
    "QKVParallelLinear": "specs.py", "DeepSeekV2FusedQkvAProjLinear": "specs.py L*.mla.qkv_a",
    "RMSNorm": "net.rmsnorm", "LayerNorm": "net._indexer (F.layer_norm fp32)", "FusedRMSNormGated": "net._kda o_norm",
    "get_rope": "facts: nope (rope 0), nothing to build", "RotaryEmbedding": "facts: nope",
    "VocabParallelEmbedding": "net.embed", "ParallelLMHead": "net.head", "LogitsProcessor": "net.head + base/sampler",
    "Fp8HeadLogitsProcessor": "net.head (bf16; fp8 head is a lane to bind)", "decodable_vocab_size": "base/sampler (mask, later)",
    "SiluAndMul": "lanes.swiglu_clamped", "SiluAndMulWithClamp": "lanes.swiglu_clamped",
    "GateLinear": "net.route", "DenebGateLinear": "net.route", "_get_moe_router_dtype": "facts (fp32 asserted)",
    "FusedMoEFactory": "net._moe + lanes.expert", "FusedMoE": "net._moe", "SharedFusedMoE": "net._moe (shared inline)",
    "FusedMoEConfig": "facts", "fused_moe_make_expert_params_mapping": "specs.py w13/w2 stacks",
    "MHCPreOp": "lanes.mhc_pre", "MHCPostOp": "lanes.mhc_post", "MHCFusedPostPreOp": "net.prefill (post then pre)",
    "hc_contract": "net.prefill (mean over streams)", "hc_expand": "net.prefill (expand)",
    "GatedDeltaNetAttention": "net._kda", "causal_conv1d_fn": "lanes.conv_prefill", "causal_conv1d_update": "lanes (decode, next)",
    "chunk_kda": "lanes.kda_chunk", "chunk_kda_with_fused_gate": "lanes.kda_chunk", "fused_recurrent_kda": "lanes (decode, next)",
    "fused_kda_gate": "modules/linear_attention.kda_gate", "divide": "net (// W)",
    "MLAModules": "net._dsa", "MultiHeadLatentAttentionWrapper": "net._dsa (absorbed MQA on the latent)",
    "SparseAttnIndexerKpool": "net._indexer", "fwht128_quant_fp8": "modules/sparse_indexer.fwht128_quant", "head_gate": "net._indexer (fp32 matmul)",
    "DeepseekV32IndexerCache": "net.Caches pool_keys/pool_scales", "KpoolTailSpec": "net.Caches tail", "MLAAttentionSpec": "net.Caches latent",
    "AttentionSpec": "base/cache_spec", "MambaStateShapeCalculator": "net.Caches kda", "MambaStateDtypeCalculator": "net.Caches kda (bf16 conv, f32 state)",
    "MambaStateCopyFunc": "base/tiered_kv (park/resume)", "MambaStateCopyFuncCalculator": "base/tiered_kv",
    "gather_initial_states": "net._kda (state view per slot)", "scatter_states": "net._kda (rec.copy_)", "is_conv_state_dim_first": "net.Caches ([C, K-1])",
    # loading
    "AutoWeightsLoader": "base/loader.RankLoader + base/params.bind", "WeightsMapper": "specs.py", "default_weight_loader": "base/preshard",
    "sharded_weight_loader": "specs.py _rows/_cols", "set_weight_attrs": "base/params.Spec", "maybe_remap_kv_scale_name": "facts (no kv scales)",
    "GroupShape": "not needed: no fp8 dense in this checkpoint", "scaled_dequantize": "not needed", "yarn_get_mscale": "facts: no yarn",
    "QuantizationConfig": "specs.py (NVFP4 experts are the only quantised tensors)",
    "maybe_prefix": "specs.py names", "extract_layer_index": "net.layers",
    "eager_break_during_capture": "base/graphs (capture is the runner's, not the layer's)",
}

KIND = {
    "drop": {"get_ep_group", "PPMissingLayer", "is_pp_missing_parameter", "make_layers", "SupportsPP", "IntermediateTensors",
             "MULTIMODAL_REGISTRY", "Glm4vDummyInputsBuilder", "Glm4vForConditionalGeneration", "init_vllm_registered_model",
             "sequence_parallel_chunk", "sp_all_gather", "sp_reduce_scatter", "sp_shard", "EagleModelMixin", "SupportsEagle3",
             "HasInnerState", "IsHybrid", "MixtureOfExperts", "current_platform", "maybe_disable_graph_partition",
             "get_pp_group", "SharedHead", "DeepseekV2MixtureOfExperts"},
}


def kind_of(symbol: str) -> str:
    if MAP.get(symbol):
        return "ours"
    for k, names in KIND.items():
        if symbol in names:
            return k
    return "real"


def edges(path: Path):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("vllm"):
            for a in node.names:
                yield node.module, a.name


def meter() -> str:
    out, total, counts, real = [], 0, {"ours": 0, "drop": 0, "real": 0}, {}
    for f in FILES:
        es = list(edges(ROOT / "overlay/modules/glm53_model" / f))
        kinds = [kind_of(s) for _m, s in es]
        for (m, s), k in zip(es, kinds):
            counts[k] += 1
            if k == "real":
                real.setdefault(f"{m}.{s}", []).append(f)
        total += len(es)
        out.append(f"  {f:<24} {len(es):>3} vLLM symbols: {kinds.count('ours'):>3} ours, {kinds.count('drop'):>2} dropped, {kinds.count('real'):>2} real")
    out.append(f"  {'TOTAL':<24} {total:>3} symbols: {counts['ours']} ours ({counts['ours'] / total:.0%}), "
               f"{counts['drop']} dropped (PP/SP/multimodal/platform/MTP), {counts['real']} real")
    for k in sorted(real):
        out.append(f"    real: {k:<70} {','.join(sorted(set(real[k])))}")
    return "\n".join(out)


if __name__ == "__main__":
    print(meter())
