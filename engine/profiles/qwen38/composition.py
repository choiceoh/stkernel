"""Qwen3.8-Flash-Next as a composition (profile): the plan, the residual form and the features, bound by name.

Nothing here computes. The layer loop is engine/base/composition's; the math is the features' (engine/modules:
hyper_connection.GatedResidualStreams, linear_attention.GatedDeltaNet, attention.Attention (+ attention.QSA),
ngram_embedding.NGramInjection, moe.MoE). What is Qwen3.8's -- and so lives here -- is which layer runs
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


def build(cfg: dict, tensor, *, prefix: str = "model.", expert=None, dtype: "str | None" = None, table=None) -> Composition:
    """The composition for text config `cfg` over `tensor(name)`. `prefix` is where the text model's names start
    ("model." in transformers' Qwen4ExpForCausalLM). `expert(layer, e)` -> (gate_up [2I, H], down [H, I]); by default
    the transformers fused tensors `mlp.experts.gate_up_proj` [E, 2I, H] and `mlp.experts.down_proj` [E, H, I].
    `dtype` is the activations' (the cache rows a store keeps): the config's `dtype`, else bfloat16. `table(name,
    rows)` -> [..., heads, width] gathers PLE rows by index; by default the whole table tensor is indexed."""
    from engine.modules.hyper_connection import GatedResidualStreams
    from engine.modules.linear_attention import VARIANTS, GatedDeltaNet, named
    from engine.modules.moe import MoE, shared_of
    from engine.modules.moe import named as moe_named
    from engine.modules.ngram_embedding import NGramHash, NGramInjection
    from engine.modules.ngram_embedding import VARIANTS as NGRAM_VARIANTS
    from engine.modules.ngram_embedding import named as ngram_named
    from engine.modules.attention import QSA, Attention
    from engine.modules.attention import named as attention_named

    def layer_name(layer, part, name):
        # a module's matrix is "<name>.weight"; a bare parameter (dt_bias, A_log, the fused experts) is "<name>"
        full = f"{prefix}layers.{layer}.{part}.{name}"
        try:
            return tensor(f"{full}.weight")
        except KeyError:
            return tensor(full)
    eps, hc, hidden = cfg["rms_norm_eps"], cfg["hc_count"], cfg["hidden_size"]
    dtype = dtype or str(cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16").replace("torch.", "")
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
            v_dim=cfg["linear_value_head_dim"], conv=cfg["linear_conv_kernel_dim"], eps=eps, **VARIANTS["gdn"],
            gate_activation=cfg.get("output_gate_type") or cfg["hidden_act"], activation=cfg["hidden_act"],
            weights=lambda layer, name: named("qwen4_exp", lambda hf: layer_name(layer, "linear_attn", hf))(name),
            dtype=dtype),
        "sparse_attention": Attention(
            form="gqa", heads=cfg["num_attention_heads"], kv_heads=cfg["num_key_value_heads"], head_dim=cfg["head_dim"],
            rotary_dim=rotary, theta=rope.get("rope_theta", cfg.get("rope_theta")), eps=eps, qk_norm="rms_unit_offset",
            gate="channel", mrope_section=tuple(section) if section else None,
            select=QSA(index_heads=cfg["indexer_n_heads"], index_head_dim=cfg["indexer_head_dim"],
                       budget=cfg["indexer_budget"], ratio=cfg["indexer_compress_ratio"]),
            weights=lambda layer, name: attention_named("qwen4_exp", lambda hf: layer_name(layer, "self_attn", hf),
                                                        heads=cfg["num_attention_heads"], head_dim=cfg["head_dim"])(name),
            dtype=dtype),
        "moe": MoE(
            experts=cfg["num_experts"], topk=cfg["num_experts_per_tok"], score="softmax", normalize=cfg.get("norm_topk_prob", True),
            router_fp32=False, shared=1, shared_mode="sigmoid", activation=cfg["hidden_act"],
            weights=lambda layer, name: moe_named("qwen4_exp", lambda hf: layer_name(layer, "mlp", hf))(name),
            expert=expert, shared_expert=lambda layer, i: shared_of("qwen4_exp", lambda hf: layer_name(layer, "mlp", hf))(i)),
    }
    if ple_layers:
        features["ple"] = NGramInjection(
            hidden=hidden, hc=hc, ngram_size=cfg["ngram_size"], conv=cfg["ple_conv_kernel_size"], eps=eps,
            **NGRAM_VARIANTS["ple"],
            hash=lambda layer: NGramHash.splitmix(
                ngram_size=cfg["ngram_size"], heads=cfg["heads_per_ngram"], unigram_vocab=cfg["vocab_size"],
                base=cfg["ngram_vocab_size_base"], table_index=ple_layers.index(layer + 1), seed=cfg.get("seed", 1234),
                eos=_eos(cfg)),
            weights=lambda layer, name: ngram_named("qwen4_exp", lambda hf: layer_name(layer, "ple", hf))(name),
            table=(lambda layer, rows: layer_name(layer, "ple", "ple_embedding.ngram_embedding.weight")[rows]) if table is None
            else (lambda layer, rows: table(f"{prefix}layers.{layer}.ple.ple_embedding.ngram_embedding", rows)),
            dtype=dtype)
    residual = GatedResidualStreams(
        hc, eps, weights=lambda layer, site, name: layer_name(
            layer, "attn_hyper_connection" if site == "mixer" else "mlp_hyper_connection", name),
        final=lambda name: tensor(f"{prefix}hyper_connection_mixer.{name}.weight"))
    # weights are read when a step runs, never at construction: cache_specs() needs only the config
    return Composition(plan(cfg), embed=lambda ids: tensor(f"{prefix}embed_tokens.weight")[ids], residual=residual,
                       features=features, head=lambda h: h @ tensor("lm_head.weight").T)


__all__ = ["plan", "build"]
