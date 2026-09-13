"""Qwen3.8-Flash-Next as a composition (profile): the plan, the residual form and the features, bound by name.

Nothing here computes. The layer loop is engine/base/composition's; the math is the features' (engine/modules:
hyper_connection.GatedResidualStreams, linear_attention.GatedDeltaNet, sparse_attention.GatedSparseAttention,
ngram_embedding.NGramInjection, moe.SharedExpertMoE). What is Qwen3.8's -- and so lives here -- is which layer runs
which feature (config `layer_types`, `ple_layer_ids`), the hyperparameters read off its text config, and the
checkpoint names the features' weights sit under (transformers qwen4_exp, pinned in plan.py).

The builder takes `tensor(name)` -> a float tensor, so the same wiring binds a transformers model's state dict (the
oracle: tests/test_engine_composition.py holds the composition to Qwen4ExpForCausalLM on the CPU) and, later, the
checkpoint's tensors. Experts come through `expert(layer, e)` so NVFP4 experts can be dequantised where they are read.
"""
from __future__ import annotations

from engine.base.composition import Composition, Layer, Plan

FULL_ATTENTION = ("full_attention", "qwen_sparse_attention")        # the checkpoint's name, and transformers' mapping of it


def plan(cfg: dict) -> Plan:
    """One layer per `layer_types` entry: GDN or QSA attention, the MoE, and PLE before layers `ple_layer_ids`
    (one-indexed)."""
    ple = set(cfg.get("ple_layer_ids") or ())
    layers = []
    for i, kind in enumerate(cfg["layer_types"]):
        if kind != "linear_attention" and kind not in FULL_ATTENTION:
            raise ValueError(f"layer {i}: Qwen3.8 has linear and full attention layers, not {kind!r}")
        layers.append(Layer("linear_attention" if kind == "linear_attention" else "sparse_attention", "moe",
                            ("ple",) if i + 1 in ple else ()))
    return Plan(tuple(layers))


def _eos(cfg: dict) -> int:
    eos = cfg.get("eos_token_id")
    return eos[0] if isinstance(eos, list) else eos


def build(cfg: dict, tensor, *, prefix: str = "model.", expert=None) -> Composition:
    """The composition for text config `cfg` over `tensor(name)`. `prefix` is where the text model's names start
    ("model." in transformers' Qwen4ExpForCausalLM). `expert(layer, e)` -> (gate_up [2I, H], down [H, I]); by default
    the transformers fused tensors `mlp.experts.gate_up_proj` [E, 2I, H] and `mlp.experts.down_proj` [E, H, I]."""
    from engine.modules.hyper_connection import GatedResidualStreams
    from engine.modules.linear_attention import GatedDeltaNet
    from engine.modules.moe import SharedExpertMoE
    from engine.modules.ngram_embedding import NGramInjection
    from engine.modules.sparse_attention import GatedSparseAttention

    def layer_name(layer, part, name):
        # a module's matrix is "<name>.weight"; a bare parameter (dt_bias, A_log, the fused experts) is "<name>"
        full = f"{prefix}layers.{layer}.{part}.{name}"
        try:
            return tensor(f"{full}.weight")
        except KeyError:
            return tensor(full)
    eps, hc, hidden = cfg["rms_norm_eps"], cfg["hc_count"], cfg["hidden_size"]
    rope = cfg.get("rope_parameters") or {}
    rotary = int(cfg["head_dim"] * rope.get("partial_rotary_factor", cfg.get("partial_rotary_factor", 1.0)))
    section = rope.get("mrope_section")
    ple_layers = list(cfg.get("ple_layer_ids") or ())
    if expert is None:
        expert = lambda layer, e: (layer_name(layer, "mlp", "experts.gate_up_proj")[e],
                                   layer_name(layer, "mlp", "experts.down_proj")[e])
    features = {
        "linear_attention": GatedDeltaNet(
            k_heads=cfg["linear_num_key_heads"], v_heads=cfg["linear_num_value_heads"], k_dim=cfg["linear_key_head_dim"],
            v_dim=cfg["linear_value_head_dim"], conv=cfg["linear_conv_kernel_dim"], eps=eps,
            gate_activation=cfg.get("output_gate_type") or cfg["hidden_act"], activation=cfg["hidden_act"],
            weights=lambda layer, name: layer_name(layer, "linear_attn", name)),
        "sparse_attention": GatedSparseAttention(
            heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
            rotary_dim=rotary, theta=rope.get("rope_theta", cfg.get("rope_theta")), eps=eps,
            index_heads=cfg["indexer_n_heads"], index_head_dim=cfg["indexer_head_dim"], budget=cfg["indexer_budget"],
            ratio=cfg["indexer_compress_ratio"], mrope_section=tuple(section) if section else None,
            weights=lambda layer, name: layer_name(layer, "self_attn", name)),
        "moe": SharedExpertMoE(
            experts=cfg["num_experts"], topk=cfg["num_experts_per_tok"], normalize=cfg.get("norm_topk_prob", True),
            activation=cfg["hidden_act"], weights=lambda layer, name: layer_name(layer, "mlp", name), expert=expert),
    }
    if ple_layers:
        features["ple"] = NGramInjection(
            hidden=hidden, hc=hc, ngram_size=cfg["ngram_size"], heads_per_ngram=cfg["heads_per_ngram"],
            unigram_vocab=cfg["vocab_size"], ngram_vocab_base=cfg["ngram_vocab_size_base"], seed=cfg.get("seed", 1234),
            eos=_eos(cfg), conv=cfg["ple_conv_kernel_size"], eps=eps,
            table_index=lambda layer: ple_layers.index(layer + 1),
            weights=lambda layer, name: layer_name(layer, "ple", name),
            table=lambda layer, rows: layer_name(layer, "ple", "ple_embedding.ngram_embedding.weight")[rows])
    residual = GatedResidualStreams(
        hc, eps, weights=lambda layer, site, name: layer_name(
            layer, "attn_hyper_connection" if site == "mixer" else "mlp_hyper_connection", name),
        final=lambda name: tensor(f"{prefix}hyper_connection_mixer.{name}.weight"))
    # weights are read when a step runs, never at construction: cache_specs() needs only the config
    return Composition(plan(cfg), embed=lambda ids: tensor(f"{prefix}embed_tokens.weight")[ids], residual=residual,
                       features=features, head=lambda h: h @ tensor("lm_head.weight").T)


__all__ = ["plan", "build"]
