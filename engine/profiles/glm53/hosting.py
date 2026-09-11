"""Re-hosting meter: what glm53_model still imports from vLLM, symbol by
symbol, against what the engine already provides (profile).

The served path is four files (glm5next_model, glm5next_kda,
glm5next_attention, mtp; multimodal is dropped -- text only). Each
`from vllm.X import Y` is one edge to cut. This parses them, maps every
symbol to an engine module or to TODO, and prints the count -- so "얼마나
남았나" is a number that goes down, not a feeling.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = ["glm5next_model.py", "glm5next_kda.py", "glm5next_attention.py", "mtp.py"]

# vllm symbol -> engine home (None = not yet ours)
MAP = {
    "ColumnParallelLinear": "modules.linear", "RowParallelLinear": "modules.linear",
    "MergedColumnParallelLinear": "modules.linear", "ReplicatedLinear": "modules.linear",
    "QKVParallelLinear": "modules.linear", "RMSNorm": "modules.norm", "get_rope": "modules.rotary",
    "RotaryEmbedding": "modules.rotary", "VocabParallelEmbedding": "modules.logits",
    "ParallelLMHead": "modules.logits", "LogitsProcessor": "modules.logits",
    "divide": "modules.linear", "get_tensor_model_parallel_world_size": "base.comm",
    "get_tensor_model_parallel_rank": "base.comm", "tensor_model_parallel_all_reduce": "base.comm",
    "tensor_model_parallel_all_gather": "base.comm", "get_pp_group": "base.comm",
    "sharded_weight_loader": "base.loader", "default_weight_loader": "base.loader",
    "maybe_remap_kv_scale_name": "base.loader", "set_weight_attrs": "base.loader",
    "MambaStateShapeCalculator": "base.cache_spec", "MambaStateDtypeCalculator": "base.cache_spec",
    "get_forward_context": "base.step_meta", "is_forward_context_available": "base.step_meta",
    "get_current_vllm_config": "base.config", "CacheConfig": "base.config", "VllmConfig": "base.config",
    "QuantizationConfig": "modules.nvfp4_linear", "init_logger": "base.instruments",
    "current_platform": None, "IntermediateTensors": None, "PPMissingLayer": None,
    "maybe_prefix": None, "extract_layer_index": None, "make_layers": None, "is_pp_missing_parameter": None,
    "AutoWeightsLoader": "base.loader", "WeightsMapper": "base.loader",
    "GatedDeltaNetAttention": "modules.linear_attention", "causal_conv1d_fn": "modules.causal_conv (judged vs served op)", "causal_conv1d_update": "modules.causal_conv (judged)",
    "MHCPreOp": "modules.hyper_connection (judged vs served tilelang)", "MHCPostOp": "modules.hyper_connection (judged)",
    "MHCFusedPostPreOp": "modules.hyper_connection (judged)", "hc_contract": "modules.hyper_connection", "hc_expand": "modules.hyper_connection",
    "chunk_kda": "glm53_kernels/kda.py (ours, judged)", "fused_recurrent_kda": "glm53_kernels/kda.py (ours, judged)",
    "chunk_kda_with_fused_gate": "glm53_kernels/kda.py (ours, judged)", "FusedRMSNormGated": "modules.norm",
    "fused_kda_gate": "modules.linear_attention.kda_gate", "GDNAttentionMetadata": "base.step_meta",
    "Glm5NextConfig": "base.config", "KimiLinearConfig": "base.config",
    "KpoolTailSpec": "base.cache_spec", "MLAAttentionSpec": "base.cache_spec",
    "gather_initial_states": "base.kv", "scatter_states": "base.kv", "eager_break_during_capture": "base.graphs",
    "FusedMoE": "modules.moe", "SharedFusedMoE": "modules.moe", "FusedMoEConfig": "modules.moe",
    "FusedMoEFactory": "modules.moe.moe_reference (ours; the b12x lane is judged through the served layer, D4)",
    "SiluAndMul": None, "SiluAndMulWithClamp": None,
    "MLAModules": "modules.linear/norm + modules.sparse_attention.mla_sparse_mqa (judged vs mk lane)",
    "MultiHeadLatentAttentionWrapper": "profiles.glm53 (plain torch composition on the layer library)",
    "SparseAttnIndexerKpool": "modules.sparse_indexer (logits/quantiser/pooling/tail judged vs served)",
    "fwht128_quant_fp8": "modules.sparse_indexer.fwht128_quant (byte-identical)",
    "head_gate": "modules.sparse_indexer (the per-head weights; glm53_indexer_gate.py is ours)",
    "DeepseekV32IndexerCache": "base.cache_spec / base.kv (the indexer's paged key cache)",
    "DeepSeekV2FusedQkvAProjLinear": "modules.linear.MergedColumnParallelLinear (q_a + kv_a fused)", "AttentionSpec": "base.cache_spec", "SharedHead": "modules.logits",
    "DeepseekV2MixtureOfExperts": "modules.moe", "_get_moe_router_dtype": None,
}


# What a still-vLLM symbol turns into when re-hosted:
#   drop  generality this engine does not carry (PP, SP, multimodal, platform
#         dispatch, interface mixins) -- the import line disappears
#   shim  ours under another name, or a few lines (activation, prefix helpers)
#   real  kernel or feature work: attention/MLA/indexer, KDA conv+kernels,
#         the MoE factory and router lane, mHC
KIND = {
    "drop": {"ParallelConfig", "get_ep_group", "PPMissingLayer", "is_pp_missing_parameter", "make_layers",
             "SupportsPP", "IntermediateTensors", "MULTIMODAL_REGISTRY", "Glm4vDummyInputsBuilder",
             "Glm4vForConditionalGeneration", "init_vllm_registered_model", "sequence_parallel_chunk",
             "sp_all_gather", "sp_reduce_scatter", "sp_shard", "EagleModelMixin", "SupportsEagle3",
             "HasInnerState", "IsHybrid", "MixtureOfExperts", "current_platform", "maybe_disable_graph_partition"},
    "shim": {"maybe_prefix", "extract_layer_index", "LayerNorm", "SiluAndMul", "SiluAndMulWithClamp",
             "GroupShape", "scaled_dequantize", "yarn_get_mscale", "is_conv_state_dim_first",
             "MambaStateCopyFunc", "MambaStateCopyFuncCalculator", "fused_moe_make_expert_params_mapping",
             "Fp8HeadLogitsProcessor", "decodable_vocab_size", "GateLinear", "DenebGateLinear",
             "_get_moe_router_dtype"},
}


def kind_of(symbol: str) -> str:
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
    out, total, ours, todo = [], 0, 0, {}
    for f in FILES:
        path = ROOT / "overlay/modules/glm53_model" / f
        es = list(edges(path)); n_ours = sum(1 for _m, s in es if MAP.get(s))
        from engine.profiles.glm53.shims import SHIMS
        n_ours = sum(1 for _m, s in es if MAP.get(s) or s in SHIMS)
        for m, s in es:
            if not MAP.get(s) and s not in SHIMS:
                todo.setdefault(f"{m}.{s}", []).append(f)
        total += len(es); ours += n_ours
        out.append(f"  {f:<24} {len(es):>3} vLLM symbols, {n_ours:>3} already ours")
    out.append(f"  {'TOTAL':<24} {total:>3} symbols, {ours:>3} ours = {ours / total:.0%} -- {total - ours} edges to cut")
    kinds = {}
    for k in todo:
        kinds.setdefault(kind_of(k.rsplit(".", 1)[1]), []).append(k)
    out.append(f"  of which: drop {len(kinds.get('drop', []))} (generality we do not carry), "
               f"shim {len(kinds.get('shim', []))} (ours under another name), "
               f"REAL {len(kinds.get('real', []))} (kernel/feature work) -- "
               f"after drop+shim the meter reads {(ours + len(kinds.get('drop', [])) + len(kinds.get('shim', []))) / total:.0%}")
    out.append("  real work, symbol: files")
    for k in sorted(kinds.get("real", [])):
        out.append(f"    {k:<70} {','.join(sorted(set(todo[k])))}")
    return "\n".join(out)


if __name__ == "__main__":
    print(meter())
