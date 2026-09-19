"""The front door: what would it take to serve THIS checkpoint (base).

CHARTER D5 names no model since 2026-09-19 -- the engine's forms are the hardware's (four GB10s, TP=4, NVFP4 as the
base weight form, no legacy) and a model is attached, not built in. This module is the other half of that sentence:
a checkpoint **no profile has been written for** can still be read here, and what its config does not say comes back
as a **blank** -- the field, and what would settle it -- never as a guess. A wrong guess in any of these fields is a
model that boots and is wrong, which is the failure this repository spends its tests on.

    from engine.base import onboard
    reading = onboard.read_config(cfg, placement="ep")   # cfg: the checkpoint's `text_config`
    reading.shape                                        # a KernelShape, or None when a blank blocks it
    reading.blanks                                       # every field the config did not settle, with why
    onboard.judge(reading)                               # + the lane table and the work it asks for (cells)

Three kinds of field, and the difference matters:

  hardware       tp, the device, the collectives' world. Not read from the model at all -- D5's forms.
  read           taken from a config key, and the key is kept beside the value (`Reading.sources`).
  not settled    either the config is silent or two different models write the same keys. Two of these the shape
                 itself can carry as "not established" (`Attention.sink`, `KernelShape.hc_variant`): the shape
                 builds and `cells.admission` refuses the lane by name, which is the outcome we want. The rest
                 (the attention kind, an indexer's compression, the expert placement, a quantisation the lane
                 must match) cannot be expressed as unknown, so they block the shape and appear as blanks.

What fills a blank is always the same short list: the checkpoint's own reference implementation, a profile written
for it under engine/profiles/, or an argument the operator passes (placement). `tools/onboard.py` prints this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.base.kernel_shape import Attention, Comm, Device, Indexer, KernelShape, LinearAttention, MoE

#: the engine's forms, not the model's (CHARTER D5)
TP = 4


@dataclass(frozen=True)
class Blank:
    """A field the config did not settle: what it is, and what would settle it."""
    field: str
    why: str

    def __str__(self) -> str:
        return f"{self.field}: {self.why}"


@dataclass(frozen=True)
class Reading:
    """What a config said, what it did not, and the shape when nothing blocks it."""
    model_type: "str | None"
    sources: dict = field(default_factory=dict)     # field -> the config key it was read from
    blanks: tuple = ()                              # Blank, in the order they were met
    shape: "KernelShape | None" = None
    #: blanks the SHAPE can carry (sink, hc_variant): stated as "not established", the lane refuses by name
    unsettled: tuple = ()

    @property
    def complete(self) -> bool:
        return self.shape is not None

    def describe(self) -> str:
        head = f"model_type {self.model_type or '(none declared)'}"
        if self.shape is not None:
            lines = [f"  {head}", f"  {self.shape.describe()}"]
            if self.unsettled:
                lines.append("  not established (the lane refuses it by name): "
                             + ", ".join(b.field for b in self.unsettled))
            return "\n".join(lines)
        return "\n".join([f"  {head}", "  no shape: the config does not settle"]
                         + [f"    {b}" for b in self.blanks])


def _int(cfg: dict, *keys) -> "tuple[int | None, str]":
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value, key
    return None, ""


def read_config(cfg: dict, *, tp: int = TP, placement: "str | None" = None) -> Reading:
    """Read a checkpoint's text config into a `Reading`. `placement` is the operator's, not the config's: "ep" gives
    each rank whole experts, "tp" slices every expert's intermediate. A routed model with neither is a blank."""
    if placement not in (None, "ep", "tp"):
        raise ValueError("placement is 'ep', 'tp' or None (not chosen)")
    sources, blanks, unsettled = {"tp": "hardware (CHARTER D5)", "device": "hardware (CHARTER D5)"}, [], []

    def blank(name, why):
        blanks.append(Blank(name, why))

    hidden, key = _int(cfg, "hidden_size")
    if hidden is None:
        blank("hidden", "no `hidden_size`; every lane's width comes from it")
        return Reading(cfg.get("model_type"), sources, tuple(blanks))
    sources["hidden"] = key

    # --- the residual streams ------------------------------------------------------------------------------------
    hc, key = _int(cfg, "hc_mult", "hc_count")
    if hc is None:
        hc, sources["hc"], hc_variant = 1, "no hyper-connection key: one residual stream", None
    else:
        sources["hc"] = key
        hc_variant = None
        if hc > 1:
            unsettled.append(Blank("hc_variant", f"{key} says {hc} streams but no key says HOW they mix (mhc, "
                                                 "split_sinkhorn, gated_residual): the model's reference does"))

    # --- attention -----------------------------------------------------------------------------------------------
    heads, heads_key = _int(cfg, "num_attention_heads")
    kv_heads, kv_key = _int(cfg, "num_key_value_heads")
    latent, latent_key = _int(cfg, "kv_lora_rank")
    head_dim, dim_key = _int(cfg, "head_dim")
    if heads is None:
        blank("attention.heads", "no `num_attention_heads`")
    kv_heads = 1 if kv_heads is None else kv_heads
    if latent is not None:
        kind, head_dim, dim_key, kv_heads = "mla", latent, latent_key, 1
        sources["attention.kind"] = latent_key
    elif kv_heads > 1:
        kind = "gqa"
        sources["attention.kind"] = kv_key
    else:
        kind = None
        blank("attention.kind", "one KV head and no `kv_lora_rank`: a latent attention and a GQA with a single KV "
                                "head declare the same numbers. The model's reference or a profile settles it")
    if head_dim is None and kind is not None:
        blank("attention.head_dim", "no `head_dim` and no `kv_lora_rank`")
    if kind is not None and head_dim is not None:
        sources["attention.head_dim"] = dim_key
        unsettled.append(Blank("attention.sink", "no key says whether the softmax denominator carries a learned "
                                                 "per-head sink; the model's reference does"))

    # --- linear attention ----------------------------------------------------------------------------------------
    linear_cfg = cfg.get("linear_attn_config") if isinstance(cfg.get("linear_attn_config"), dict) else None
    k_heads, k_key = _int(cfg, "linear_num_key_heads")
    if linear_cfg is None and k_heads is None:
        linear, sources["linear"] = None, "no linear-attention key: the KDA lanes do not apply"
    else:
        linear = None
        blank("linear.decay", "the config gives the linear-attention widths but not whether the decay is per key "
                              "channel (KDA) or per head (GDN); modules/linear_attention's axis, from the reference")

    # --- the sparse indexer --------------------------------------------------------------------------------------
    topk, topk_key = _int(cfg, "index_topk", "indexer_budget")
    index_heads, _ = _int(cfg, "index_n_heads", "indexer_n_heads")
    if topk is None and index_heads is None:
        indexer, sources["indexer"] = None, "no indexer key: the attention is dense over its window"
    else:
        indexer = None
        blank("indexer.compress", f"`{topk_key or 'the indexer keys'}` declares a sparse indexer, but how it "
                                  "compresses keys (kpool, ced, qsa) is the model's own; modules/sparse_indexer "
                                  "shares the scoring, not the compression")

    # --- the routed experts --------------------------------------------------------------------------------------
    experts, experts_key = _int(cfg, "n_routed_experts", "num_experts")
    inter, inter_key = _int(cfg, "moe_intermediate_size")
    topk_experts, topk_experts_key = _int(cfg, "num_experts_per_tok")
    dense_inter, dense_key = _int(cfg, "intermediate_size")
    shared, _ = _int(cfg, "n_shared_experts")
    quant_cfg = cfg.get("quantization_config")
    if quant_cfg is None:
        quant, sources["moe.quant"] = "bf16", "no `quantization_config`: the weights are the checkpoint's dtype"
    else:
        quant = None
        method = quant_cfg.get("quant_method") if isinstance(quant_cfg, dict) else quant_cfg
        blank("moe.quant", f"`quantization_config` says {method!r}; the lane is admitted against a CELL name "
                           "(nvfp4, mxfp4-a8), which the preshard or a b12x cell establishes (kernels/cells.py)")
    if experts is None or inter is None or topk_experts is None:
        blank("moe", "no routed experts in this config (`n_routed_experts`/`moe_intermediate_size`/"
                     "`num_experts_per_tok`): a dense model is served through the E=1 lane, which a profile declares")
    elif placement is None:
        blank("moe.experts_local", "expert placement is the operator's, not the config's: 'ep' gives a rank whole "
                                   "experts, 'tp' slices every expert's intermediate")
    else:
        sources["moe.experts"], sources["moe.inter"], sources["moe.topk"] = experts_key, inter_key, topk_experts_key
        sources["moe.placement"] = f"operator: {placement}"
    if dense_inter is None and shared and inter:
        dense_inter, dense_key = shared * inter, "n_shared_experts x moe_intermediate_size"
    sources["moe.dense_inter_local"] = dense_key or "no dense MLP width in the config"

    spec_k, spec_key = _int(cfg, "num_nextn_predict_layers", "mtp_num_hidden_layers")
    sources["spec_k"] = spec_key or "no MTP key: one token a step"

    if blanks:
        return Reading(cfg.get("model_type"), sources, tuple(blanks), None, tuple(unsettled))

    shape = KernelShape(
        comm=Comm(world=tp, hidden=hidden), hidden=hidden, hc=hc, tp=tp,
        attention=Attention(kind=kind, heads=heads // tp, head_dim=head_dim,
                            kv_heads=max(1, kv_heads // tp) if kind == "gqa" else 1, sink=None),
        linear=linear, indexer=indexer,
        moe=MoE(experts=experts, experts_local=experts // tp if placement == "ep" else experts, hidden=hidden,
                inter=inter, inter_local=inter if placement == "ep" else inter // tp, topk=topk_experts,
                quant=quant, activation=cfg.get("hidden_act", "silu"), swiglu_limit=cfg.get("swiglu_limit"),
                dense_inter_local=(dense_inter or 0) // tp),
        spec_k=spec_k or 1, device=Device(), hc_variant=hc_variant)
    return Reading(cfg.get("model_type"), sources, (), shape, tuple(unsettled))


def judge(reading: Reading) -> dict:
    """A reading plus the lane table it implies: {"reading", "admission", "plan"} -- empty lists when a blank blocked
    the shape, because a lane verdict on a guessed shape is worth less than no verdict."""
    from engine.kernels import cells
    if reading.shape is None:
        return {"reading": reading, "admission": [], "plan": []}
    verdicts = cells.admission(reading.shape)
    return {"reading": reading, "admission": verdicts, "plan": cells.plan(verdicts)}


__all__ = ["TP", "Blank", "Reading", "read_config", "judge"]
