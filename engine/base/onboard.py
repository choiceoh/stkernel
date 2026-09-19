"""The front door: what would it take to serve THIS checkpoint (base).

CHARTER D5 names no model since 2026-09-19 -- the engine's forms are the hardware's (four GB10s, TP=4, NVFP4 as the
base weight form, no legacy) and a model is attached, not built in. This module is the other half of that sentence:
a checkpoint **no profile has been written for** can still be read here, and what its config does not say comes back
as a **blank** -- the field, and what would settle it -- never as a guess. A wrong guess in any of these fields is a
model that boots and is wrong, which is the failure this repository spends its tests on.

    from engine.base import onboard
    reading = onboard.read_config(cfg, placement="ep")   # cfg: the checkpoint's `text_config`
    reading.shape                                        # a KernelShape, or None when a blank blocks it
    reading.values / reading.sources                     # every field it DID read, and the key each came from
    reading.blanks                                       # every field the config did not settle, with why
    onboard.judge(reading)                               # + the lane table and the work it asks for (cells)

Four kinds of field, and the difference matters:

  hardware       tp, the device, the collectives' world. Not read from the model at all -- D5's forms.
  read           taken from a config key, and the key is kept beside the value (`Reading.sources`). This includes
                 the axes a config NAMES: `index_kpool_compress` names the indexer's compression, a `kda_layers`
                 key names the linear attention's per-channel decay, `mhc: true` names the mixer.
  stated         the operator's answer for a field no config settles, passed as `states={field: value}` and marked
                 "operator" in the sources. D11 allows exactly this shape of input -- a FACT about the model, not a
                 performance axis -- and it may only fill a blank: stating a field the config settles raises.
  not settled    neither given nor stated. Two of these the shape itself can carry as "not established"
                 (`Attention.sink`, `KernelShape.hc_variant`): the shape builds and `cells.admission` refuses the
                 lane by name, which is the outcome we want. The rest (the attention kind, an indexer's
                 compression, the expert placement, a quantisation or a gate the lane must match) cannot be
                 expressed as unknown, so they block the shape and appear as blanks.

What fills a blank is always the same short list: the checkpoint's own reference implementation, a profile written
for it under engine/profiles/, or an argument the operator passes (`placement`, `states`). `tools/onboard.py`
prints this. `tests/test_engine_onboard.py` holds the door to the three profiles the repo serves: their own
derivations and this one agree field for field once the operator states what their references told them -- and the
list of what that is, per model, is exactly the door's blanks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.base.kernel_shape import (HC_VARIANTS, INDEXER_COMPRESS, Attention, Comm, Device, Indexer, KernelShape,
                                      LinearAttention, MoE)

#: the engine's forms, not the model's (CHARTER D5)
TP = 4

#: what an operator may STATE, and the values each field takes (None: any name). Every one of these is a fact about
#: the model that a config can leave unsaid -- never a performance axis, which D11 keeps out of the inputs.
STATEABLE = {
    "attention.kind": ("mla", "gqa"),
    "attention.sink": (True, False),
    "indexer.compress": INDEXER_COMPRESS,
    "linear.decay": ("channel", "head"),
    "hc_variant": HC_VARIANTS,
    "moe.quant": None,
    "moe.activation": None,
}


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
    values: dict = field(default_factory=dict)      # field -> what was read (per rank, as the shape holds it)

    @property
    def complete(self) -> bool:
        return self.shape is not None

    def read(self) -> "list[tuple]":
        """(field, value, source) for every field the reading settled -- the table `tools/onboard.py` prints."""
        return [(name, self.values.get(name, ""), source) for name, source in self.sources.items()]

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
    """The first of `keys` the config states as a positive int. A width of 0 (GLM-5.3's `head_dim`, which its
    `kv_lora_rank` replaces) is not a width: it reads as unstated, like a missing key or a null."""
    for key in keys:
        value = cfg.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value, key
    return None, ""


def _checked(states: "dict | None") -> dict:
    states = dict(states or {})
    for name, value in states.items():
        if name not in STATEABLE:
            raise ValueError(f"{name!r} is not a field an operator states; one of {sorted(STATEABLE)}")
        allowed = STATEABLE[name]
        if allowed is None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} is a name, got {value!r}")
        elif allowed == (True, False):
            if not isinstance(value, bool):                 # 1 == True, and a shape must not carry an int here
                raise ValueError(f"{name} is True or False, got {value!r}")
        elif value not in allowed:
            raise ValueError(f"{name} is one of {allowed}, got {value!r}")
    return states


def read_config(cfg: dict, *, tp: int = TP, placement: "str | None" = None, states: "dict | None" = None) -> Reading:
    """Read a checkpoint's text config into a `Reading`.

    `placement` is the operator's, not the config's: "ep" gives each rank whole experts, "tp" slices every expert's
    intermediate. A routed model with neither is a blank. `states` is the operator's too -- {field: value} out of
    `STATEABLE`, for the axes a config cannot settle. It only ever FILLS a blank: stating a field the config settles
    raises, so no argument to this door can serve something other than what the checkpoint says.

    A config with no expert key at all is read as the E=1 cell -- the one MLP every token passes, which is what this
    engine serves a dense or shared MLP through -- and asks for no placement. One that DOES name experts in a
    spelling this door cannot read is a blank instead: reading it as dense would serve every token through one MLP.
    """
    if placement not in (None, "ep", "tp"):
        raise ValueError("placement is 'ep', 'tp' or None (not chosen)")
    states = _checked(states)
    sources = {"tp": "hardware (CHARTER D5)", "device": "hardware (CHARTER D5)"}
    values, blanks, unsettled = {"tp": tp, "device": "GB10 sm_121a"}, [], []

    def read(name, key, value):
        """A field the CONFIG states. An operator state for it is a contradiction, not an override."""
        if name in states:
            raise ValueError(f"the config settles {name}: {key}; `states` fills a blank, it does not overrule it")
        sources[name], values[name] = key, value
        return value

    def absent(name, why):
        """A field the config settles by saying nothing (no indexer keys, no MTP head): the absence IS the fact."""
        sources[name] = why
        return None

    def ask(name, why, *, carried=False):
        """A field the config does not settle: the operator's answer, or a blank. `carried`: the shape can hold it
        as "not established" and the lane refuses it by name, so it does not block."""
        if name in states:
            sources[name], values[name] = f"operator: stated {name}", states[name]
            return states[name]
        (unsettled if carried else blanks).append(Blank(name, why))
        return None

    def built(name, make):
        """A piece of the shape from numbers the config states -- as a blank, never a traceback, when its own
        numbers contradict the descriptor (a head count that does not divide, a width that is not a power of two)."""
        try:
            return make()
        except ValueError as exc:
            blanks.append(Blank(name, f"the config's own numbers do not make a {name}: {exc}"))
            return None

    hidden, key = _int(cfg, "hidden_size")
    if hidden is None:
        blanks.append(Blank("hidden", "no `hidden_size`; every lane's width comes from it"))
        return Reading(cfg.get("model_type"), sources, tuple(blanks), None, (), values)
    read("hidden", key, hidden)

    # --- the residual streams ------------------------------------------------------------------------------------
    hc, key = _int(cfg, "hc_mult", "hc_count")
    hc_variant = None
    if hc is None:
        hc = 1
        absent("hc", "no hyper-connection key: one residual stream")
    else:
        read("hc", key, hc)
        if hc > 1 and cfg.get("mhc") is True:
            hc_variant = read("hc_variant", "mhc", "mhc")               # the key names the mixer
        elif hc > 1:
            hc_variant = ask("hc_variant", f"`{key}` says {hc} streams but no key says HOW they mix (mhc, "
                                           "split_sinkhorn, gated_residual): the model's reference does", carried=True)

    # --- attention -----------------------------------------------------------------------------------------------
    heads, heads_key = _int(cfg, "num_attention_heads")
    kv_heads, kv_key = _int(cfg, "num_key_value_heads")
    latent, latent_key = _int(cfg, "kv_lora_rank")
    head_dim, dim_key = _int(cfg, "head_dim")
    sink = None
    if heads is None:
        blanks.append(Blank("attention.heads", "no `num_attention_heads`"))
    else:
        read("attention.heads", heads_key, heads // tp)
    kv_heads = 1 if kv_heads is None else kv_heads
    if latent is not None:
        kind = read("attention.kind", latent_key, "mla")                # a declared latent: one key per position
        head_dim, dim_key, kv_heads = latent, latent_key, 1
    elif kv_heads > 1:
        kind = read("attention.kind", kv_key, "gqa")
    else:
        kind = ask("attention.kind", "one KV head and no `kv_lora_rank`: a latent attention and a GQA with a single "
                                     "KV head declare the same numbers. The model's reference or a profile settles it")
    if kind is not None and head_dim is None:
        blanks.append(Blank("attention.head_dim", "no `head_dim` and no `kv_lora_rank`"))
    elif kind is not None:
        read("attention.head_dim", dim_key, head_dim)
    if heads is not None:
        # The sink belongs to the softmax, not to the kind: it is asked of every attention, mla or gqa.
        sink = ask("attention.sink", "no key says whether the softmax denominator carries a learned per-head sink; "
                                     "the model's reference does", carried=True)

    # --- linear attention ----------------------------------------------------------------------------------------
    nested = cfg.get("linear_attn_config") if isinstance(cfg.get("linear_attn_config"), dict) else None
    k_heads, k_key = _int(cfg, "linear_num_key_heads")
    linear = None
    if nested is None and k_heads is None:
        absent("linear", "no linear-attention key: the KDA lanes do not apply")
    else:
        if nested is not None:      # one head count and one width for both halves (GLM-5.3's spelling)
            n, n_key = _int(nested, "num_heads")
            dim, dim_k = _int(nested, "head_dim")
            conv, conv_k = _int(nested, "short_conv_kernel_size")
            widths, names = (n, n, dim, dim, conv), f"linear_attn_config.{n_key}/{dim_k}/{conv_k}"
            named = next((f"linear_attn_config.{k}" for k in nested if "kda" in k.lower()), "")
        else:                       # key and value heads counted apart (Qwen3.8's spelling)
            v_heads, v_key = _int(cfg, "linear_num_value_heads")
            k_dim, k_dim_key = _int(cfg, "linear_key_head_dim")
            v_dim, v_dim_key = _int(cfg, "linear_value_head_dim")
            conv, conv_k = _int(cfg, "linear_conv_kernel_dim")
            widths = (k_heads, v_heads, k_dim, v_dim, conv)
            names = f"{k_key}/{v_key}/{k_dim_key}/{v_dim_key}/{conv_k}"
            named = next((k for k in cfg if "kda" in k.lower()), "")
        # The decay is the family's axis, not a width: "channel" is KDA's per key channel, "head" is GDN's per head
        # (engine/base/kernel_shape.LinearAttention, engine/modules/linear_attention). A key that NAMES kda settles it.
        decay = (read("linear.decay", named, "channel") if named else
                 ask("linear.decay", "the config gives the linear-attention widths but not whether the decay is per "
                                     "key channel (KDA) or per head (GDN); modules/linear_attention's axis, from the "
                                     "reference"))
        if any(w is None for w in widths):
            blanks.append(Blank("linear", f"a linear attention is declared but one of its widths is not ({names})"))
        elif decay is not None:
            heads_l, v_heads_l = max(1, widths[0] // tp), max(1, widths[1] // tp)
            linear = built("linear", lambda: LinearAttention(heads=heads_l, v_heads=v_heads_l, k_dim=widths[2],
                                                             v_dim=widths[3], conv=widths[4], decay=decay))
            if linear is not None:
                read("linear", names, f"{heads_l}/{v_heads_l}x{widths[2]}x{widths[3]} conv {widths[4]}")

    # --- the sparse indexer --------------------------------------------------------------------------------------
    topk, topk_key = _int(cfg, "index_topk", "indexer_budget")
    idx_heads, idx_heads_key = _int(cfg, "index_n_heads", "indexer_n_heads")
    idx_dim, idx_dim_key = _int(cfg, "index_head_dim", "indexer_head_dim")
    pool, pool_key = _int(cfg, "index_kpool", "indexer_compress_ratio")
    if pool is None and isinstance(cfg.get("compress_ratios"), list):
        # a ratio per layer: one pooling group above 1 is the indexer's; several, and the config does not say which
        ratios = {r for r in cfg["compress_ratios"] if isinstance(r, int) and not isinstance(r, bool) and r > 1}
        pool, pool_key = (ratios.pop(), "compress_ratios") if len(ratios) == 1 else (None, "")
    indexer = None
    if topk is None and idx_heads is None:
        absent("indexer", "no indexer key: the attention is dense over its window")
    else:
        compress = (read("indexer.compress", "index_kpool_compress", "kpool")
                    if cfg.get("index_kpool_compress") is True else
                    ask("indexer.compress", f"`{topk_key or idx_heads_key}` declares a sparse indexer, but how it "
                                            "compresses keys (kpool, ced, qsa) is the model's own; "
                                            "modules/sparse_indexer shares the scoring, not the compression"))
        missing = [n for n, v in (("heads", idx_heads), ("head_dim", idx_dim), ("pool", pool), ("topk", topk))
                   if v is None]
        if missing:
            blanks.append(Blank("indexer." + "/".join(missing), "a sparse indexer is declared but the config does not "
                                f"state its {', '.join(missing)} (index_n_heads/index_head_dim/index_kpool|"
                                "indexer_compress_ratio|compress_ratios/index_topk|indexer_budget)"))
        elif compress is not None:
            indexer = built("indexer", lambda: Indexer(heads=idx_heads, head_dim=idx_dim, pool=pool, topk=topk,
                                                       compress=compress))
            if indexer is not None:
                read("indexer", f"{idx_heads_key}/{idx_dim_key}/{pool_key}/{topk_key}",
                     f"{compress} {idx_heads}x{idx_dim} pool {pool} top {topk}")

    # --- the routed experts --------------------------------------------------------------------------------------
    # any key that names experts or a router, in any spelling -- a config that states one is a routed model whose
    # numbers this door may simply not know how to read (Mixtral counts its experts in `num_local_experts`). The
    # vocabulary is deliberately wide: a false blank costs a question, and reading a routed model as dense does not.
    named_experts = sorted(k for k, v in cfg.items() if v is not None
                           and (any(w in k.lower() for w in ("expert", "router"))
                                or k.lower().startswith(("moe", "n_routed"))))
    experts, experts_key = _int(cfg, "n_routed_experts", "num_experts")
    inter, inter_key = _int(cfg, "moe_intermediate_size")
    topk_experts, topk_experts_key = _int(cfg, "num_experts_per_tok")
    dense_inter, dense_key = _int(cfg, "intermediate_size", "shared_expert_intermediate_size")
    shared, _ = _int(cfg, "n_shared_experts")
    quant_cfg = cfg.get("quantization_config")
    if quant_cfg is None:
        quant = read("moe.quant", "no `quantization_config`: the weights are the checkpoint's dtype", "bf16")
    else:
        stated = (", ".join(f"{k}={v!r}" for k, v in quant_cfg.items() if not isinstance(v, dict))
                  if isinstance(quant_cfg, dict) else repr(quant_cfg)) or "a nested encoding"
        quant = ask("moe.quant", f"`quantization_config` says {stated}; the lane is admitted against a CELL name "
                                 "(nvfp4, mxfp4-a8), which the preshard or a b12x cell establishes (kernels/cells.py)")
    limit = cfg.get("swiglu_limit")
    if limit is not None:
        read("moe.swiglu_limit", "swiglu_limit", limit)
    if not isinstance(cfg.get("hidden_act"), str):
        activation = None
        blanks.append(Blank("moe.activation", "no `hidden_act`: the gate the expert GEMM is admitted for"))
    elif limit is None:
        activation = read("moe.activation", "hidden_act", cfg["hidden_act"])
    else:
        # Two checkpoints write exactly `hidden_act` silu + `swiglu_limit`, and are served with different gates:
        # GLM-5.3's is swigluoai_uninterleave, DSv4.1's is silu. The MoE cell is compared by name (kernels/cells.py).
        activation = ask("moe.activation", f"`hidden_act` says {cfg['hidden_act']!r} and `swiglu_limit` {limit} -- a "
                                           "clamped gate, whose spelling is not the same in the two checkpoints that "
                                           "write these keys (swigluoai_uninterleave, silu); the reference settles it")
    if dense_inter is None and shared and inter:
        dense_inter, dense_key = shared * inter, "n_shared_experts x moe_intermediate_size"
    sources["moe.dense_inter_local"] = dense_key or "no dense MLP width in the config"
    if dense_inter is not None:
        values["moe.dense_inter_local"] = dense_inter // tp

    experts_local = inter_local = None
    if experts is not None and inter is not None and topk_experts is not None:
        if placement is None:
            blanks.append(Blank("moe.experts_local", "expert placement is the operator's, not the config's: 'ep' "
                                                     "gives a rank whole experts, 'tp' slices every expert's "
                                                     "intermediate"))
        else:
            experts_local = experts // tp if placement == "ep" else experts
            inter_local = inter if placement == "ep" else inter // tp
            read("moe.experts", experts_key, experts)
            read("moe.inter", inter_key, inter_local)
            read("moe.topk", topk_experts_key, topk_experts)
            sources["moe.placement"], values["moe.placement"] = f"operator: {placement}", placement
    elif named_experts:
        missing = [k for k, v in (("`n_routed_experts`|`num_experts`", experts), ("`moe_intermediate_size`", inter),
                                  ("`num_experts_per_tok`", topk_experts)) if v is None]
        blanks.append(Blank("moe", f"this config names experts ({', '.join(named_experts)}) but not "
                                   f"{', '.join(missing)}: a spelling this door does not read is a blank, never a "
                                   "dense model -- reading it as one would serve every token through a single MLP"))
    elif dense_inter is not None:
        # No routed experts anywhere in the config. The one MLP every token passes is what this engine serves
        # through the E=1 cell -- b12x's gate admits (1, hidden, dense_inter_local, 1) beside the routed tuple
        # (engine/kernels/b12x/moe_dispatch._glm_tp_scatter_shape) -- and it is TP-sharded, like every dense and
        # shared MLP here, so there is no expert placement to choose and none is asked for.
        experts, experts_local, topk_experts = 1, 1, 1
        inter, inter_local = dense_inter, dense_inter // tp
        sources["moe"] = f"no expert key: the one MLP ({dense_key}) as the E=1 cell b12x serves it"
        values["moe"] = f"1 expert, I{inter_local}, top1"
    else:
        blanks.append(Blank("moe", "no experts (`n_routed_experts`/`moe_intermediate_size`/`num_experts_per_tok`) "
                                   "and no MLP width (`intermediate_size`) either: this config declares no MLP"))

    spec_k, spec_key = _int(cfg, "num_nextn_predict_layers", "mtp_num_hidden_layers")
    sources["spec_k"], values["spec_k"] = spec_key or "no MTP key: one token a step", spec_k or 1

    left = [name for name in states if name not in values]
    if left:      # a state for a part the model does not have would sit in the reading doing nothing
        raise ValueError(f"nothing for {', '.join(sorted(left))} to fill: this config declares no such part of the "
                         "model (no indexer, no linear attention, no hyper-connection...)")

    if blanks:
        return Reading(cfg.get("model_type"), sources, tuple(blanks), None, tuple(unsettled), values)

    shape = built("shape", lambda: KernelShape(
        comm=Comm(world=tp, hidden=hidden), hidden=hidden, hc=hc, tp=tp,
        attention=Attention(kind=kind, heads=heads // tp, head_dim=head_dim,
                            kv_heads=max(1, kv_heads // tp) if kind == "gqa" else 1, sink=sink),
        linear=linear, indexer=indexer,
        moe=MoE(experts=experts, experts_local=experts_local, hidden=hidden, inter=inter, inter_local=inter_local,
                topk=topk_experts, quant=quant, activation=activation, swiglu_limit=limit,
                dense_inter_local=(dense_inter or 0) // tp),
        spec_k=spec_k or 1, device=Device(), hc_variant=hc_variant))
    return Reading(cfg.get("model_type"), sources, tuple(blanks), shape, tuple(unsettled), values)


def judge(reading: Reading) -> dict:
    """A reading plus the lane table it implies: {"reading", "admission", "plan"} -- empty lists when a blank blocked
    the shape, because a lane verdict on a guessed shape is worth less than no verdict."""
    from engine.kernels import cells
    if reading.shape is None:
        return {"reading": reading, "admission": [], "plan": []}
    verdicts = cells.admission(reading.shape)
    return {"reading": reading, "admission": verdicts, "plan": cells.plan(verdicts)}


__all__ = ["TP", "STATEABLE", "Blank", "Reading", "read_config", "judge"]
