"""Qwen3.8-Flash-Next's weights as one TEP=4 rank holds them (profile): the layout, written down once, that both the
preshard and the served net read.

Every tensor the net touches is a Spec (base/params.py): its rank-local shape, dtype, the checkpoint tensors it is
built from and HOW. Placement is facts.py's: GDN and attention TP by heads, routed experts EP (whole experts
[128r, 128(r+1)) on rank r), the shared expert TP by intermediate, embed/head/PLE table vocabulary-parallel,
hyper-connection, router, indexer and norm weights replicated.

The projections that read one input are merged here, once, so the served step issues one GEMM where the checkpoint
names several (what the served lanes fuse; nothing is repacked at boot, D1):

    gdn.in_proj        q | k | v | z | b | a       one input, six outputs -> one matmul        [4120, 2560] a rank
    attn.in_proj       query+gate | k | v | index   one input, the rank's heads, its KV head    [4224, 2560] a rank
    hc.<site>.down_inject   down | block_inject     both read the normalised streams            [324, 10240]
    moe.sh_gate_up     gate | up                    the shared expert's first projection         [320, 2560]
    moe.gates          router | shared gate         the router's 512 rows and the shared gate's one [513, 2560]
    ple.kv_proj        key | value                  PLE's two projections of its looked-up rows  [12800, 2560]

Merged rows keep each part's own row order, so a slice of the merged matrix is the part exactly. The routed experts
are ModelOpt's lossless layout (engine/profiles/glm53/modelopt_weights.quant_specs, whole experts instead of an
intermediate split): packed nibbles as they are with rows [up; gate] (the b12x kernel's order), E4M3 block scales
tile-interleaved (modules/nvfp4_sf), the FP32 `weight_scale_2` multipliers and `input_scale` kept beside them.

The MTP head's experts are BF16 in the checkpoint (a fused [512, 1280, 2560] and [512, 2560, 640]); they are written
as NVFP4 in the same layout so the drafter's MoE runs on the target's lane. A drafter changes how many tokens a step
yields, never which (engine/base/composed: verification picks), so the drafter's quantisation costs acceptance, not
correctness.

The PLE table (128 shards of [2,500,012, 160] e4m3, one scalar scale) is written as 32 parts a rank, one group each,
so presharding never holds more than four shards (1.5 GiB) of it.
"""
from __future__ import annotations

import torch

from engine.base.params import Spec
from engine.modules.nvfp4_sf import swizzle_sf_batch
from engine.profiles.qwen38.facts import TP, Facts

CK = "model.language_model."
BF, F32, U8, E4, I64 = torch.bfloat16, torch.float32, torch.uint8, torch.float8_e4m3fn, torch.int64
PLE_SHARD_ROWS = 2_500_012                     # rows of one ngram_embedding shard (the checkpoint's header)


def _split(t, dim, r, W):
    n = t.shape[dim] // W
    if n * W != t.shape[dim]:
        raise ValueError(f"dim {dim} of {tuple(t.shape)} does not split {W} ways")
    return t.narrow(dim, r * n, n)


def _rows(key, dtype=None):
    return lambda s, r, W: (_split(s[key], 0, r, W) if dtype is None else _split(s[key], 0, r, W).to(dtype)).contiguous()


def _cols(key):
    return lambda s, r, W: _split(s[key], 1, r, W).contiguous()


def _whole(key, dtype=None):
    return lambda s, r, W: s[key].contiguous() if dtype is None else s[key].to(dtype).contiguous()


def _hc_specs(n: str, p: str, F: Facts, sites) -> "list[Spec]":
    """The gated residual's weights per site: the stream norm, down and injection as one matmul, up."""
    width = F.hc * F.hidden
    out = []
    for site, name in sites:
        norm, down, up = (p + f"{name}.hc_norm.weight", p + f"{name}.input_mix_weight_down.weight",
                          p + f"{name}.input_mix_weight_up.weight")
        inject = p + f"{name}.block_inject_weight.weight"
        if site == "close":
            out += [Spec(n + "close.norm", (width,), BF, (norm,), _whole(norm)),
                    Spec(n + "close.down", (F.hc_rank, width), BF, (down,), _whole(down)),
                    Spec(n + "close.up", (width, F.hc_rank), BF, (up,), _whole(up))]
            continue
        di = (down, inject)
        out += [Spec(n + f"hc.{site}.norm", (width,), BF, (norm,), _whole(norm)),
                Spec(n + f"hc.{site}.down_inject", (F.hc_rank + F.hc, width), BF, di,
                     lambda s, r, W, k=di: torch.cat([s[k[0]], s[k[1]]], 0).contiguous()),
                Spec(n + f"hc.{site}.up", (width, F.hc_rank), BF, (up,), _whole(up))]
    return out


def _attention_specs(n: str, a: str, F: Facts) -> "list[Spec]":
    """Gated GQA with the QSA indexer: the rank's query heads (each [query | gate]), its one KV head, the index
    projection, merged; norms replicated; o_proj split by heads along its input."""
    H, Hq, D = F.hidden, F.heads_local, F.head_dim
    q, k, v, idx = a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight", a + "indexer.index_qk_proj.weight"
    width = Hq * 2 * D + 2 * F.kv_heads_local * D + F.idx_heads * F.idx_dim + F.idx_dim

    def in_proj(s, r, W, q=q, k=k, v=v, idx=idx):
        per_kv = s[k].shape[0] // F.kv_heads
        h = F.kv_head_of(r)
        return torch.cat([_split(s[q], 0, r, W), s[k].narrow(0, h * per_kv, per_kv),
                          s[v].narrow(0, h * per_kv, per_kv), s[idx]], 0).contiguous()

    return [
        Spec(n + "attn.in_proj", (width, H), BF, (q, k, v, idx), in_proj),
        Spec(n + "attn.q_norm", (D,), BF, (a + "q_norm.weight",), _whole(a + "q_norm.weight")),
        Spec(n + "attn.k_norm", (D,), BF, (a + "k_norm.weight",), _whole(a + "k_norm.weight")),
        Spec(n + "attn.idx_q_norm", (F.idx_dim,), BF, (a + "indexer.q_layernorm.weight",),
             _whole(a + "indexer.q_layernorm.weight")),
        Spec(n + "attn.idx_k_norm", (F.idx_dim,), BF, (a + "indexer.k_layernorm.weight",),
             _whole(a + "indexer.k_layernorm.weight")),
        Spec(n + "attn.o_proj", (H, Hq * D), BF, (a + "o_proj.weight",), _cols(a + "o_proj.weight")),
    ]


def _gdn_specs(n: str, g: str, F: Facts) -> "list[Spec]":
    """GatedDeltaNet: q|k|v|z|b|a merged with each part split by its heads; conv over the rank's q|k|v channels
    (fp32: the conv computes in fp32, as the reference does); A_log and dt_bias per value head in fp32."""
    H = F.hidden
    qk, v = F.k_heads * F.k_dim, F.v_heads * F.v_dim
    qkv, z, b, a_ = g + "in_proj_qkv.weight", g + "in_proj_z.weight", g + "in_proj_b.weight", g + "in_proj_a.weight"
    rows = F.qkv_local + F.v_heads_local * F.v_dim + 2 * F.v_heads_local

    def qkv_parts(t, r, W):
        return [_split(t.narrow(0, 0, qk), 0, r, W), _split(t.narrow(0, qk, qk), 0, r, W),
                _split(t.narrow(0, 2 * qk, v), 0, r, W)]

    def in_proj(s, r, W):
        return torch.cat(qkv_parts(s[qkv], r, W) + [_split(s[z], 0, r, W), _split(s[b], 0, r, W),
                                                    _split(s[a_], 0, r, W)], 0).contiguous()

    def conv(s, r, W, key=g + "conv1d.weight"):
        return torch.cat(qkv_parts(s[key][:, 0, :], r, W), 0).to(F32).contiguous()

    return [
        Spec(n + "gdn.in_proj", (rows, H), BF, (qkv, z, b, a_), in_proj),
        Spec(n + "gdn.conv", (F.qkv_local, F.conv), F32, (g + "conv1d.weight",), conv),
        Spec(n + "gdn.A_log", (F.v_heads_local,), F32, (g + "A_log",), _rows(g + "A_log", F32)),
        Spec(n + "gdn.dt_bias", (F.v_heads_local,), F32, (g + "dt_bias",), _rows(g + "dt_bias", F32)),
        Spec(n + "gdn.norm", (F.v_dim,), BF, (g + "norm.weight",), _whole(g + "norm.weight")),
        Spec(n + "gdn.out_proj", (H, F.v_heads_local * F.v_dim), BF, (g + "out_proj.weight",),
             _cols(g + "out_proj.weight")),
    ]


def _expert_ids(F: Facts, r: int):
    lo, hi = F.expert_range(r)
    return range(lo, hi)


def _routed_specs(n: str, m: str, F: Facts) -> "list[Spec]":
    """The rank's 128 whole NVFP4 experts in the ModelOpt lossless layout, rows [up; gate]."""
    E, I, H = F.experts_local, F.moe_inter, F.hidden
    names = [m + f"experts.{e}." for e in range(F.experts)]
    suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    first = tuple(p + proj + "_proj." + suf for p in names for proj in ("up", "gate") for suf in suffixes)
    second = tuple(p + "down_proj." + suf for p in names for suf in suffixes)

    def local(r):
        return [names[e] for e in _expert_ids(F, r)]

    def packed_first(s, r, W):
        return torch.stack([torch.cat([s[p + "up_proj.weight"], s[p + "gate_proj.weight"]], 0) for p in local(r)]).contiguous()

    def packed_second(s, r, W):
        return torch.stack([s[p + "down_proj.weight"] for p in local(r)]).contiguous()

    def scale_first(s, r, W):
        rows = torch.stack([torch.cat([s[p + "up_proj.weight_scale"], s[p + "gate_proj.weight_scale"]], 0)
                            for p in local(r)])
        return swizzle_sf_batch(rows.view(torch.uint8)).view(E4).contiguous()

    def scale_second(s, r, W):
        rows = torch.stack([s[p + "down_proj.weight_scale"] for p in local(r)])
        return swizzle_sf_batch(rows.view(torch.uint8)).view(E4).contiguous()

    def scalars_first(suffix):
        def build(s, r, W):
            values = []
            for p in local(r):
                up, gate = (s[p + g + "_proj." + suffix].float().reshape(()) for g in ("up", "gate"))
                if not torch.equal(up, gate):
                    raise ValueError(f"{p}: gate/up {suffix} must match for the fused first projection")
                values.append(up)
            return _positive(torch.stack(values))
        return build

    def scalars_second(suffix):
        return lambda s, r, W: _positive(torch.stack([s[p + "down_proj." + suffix].float().reshape(()) for p in local(r)]))

    return [
        Spec(n + "moe.w13", (E, 2 * I, H // 2), U8, first, packed_first),
        Spec(n + "moe.w13_sf", (E, 2 * I * (H // 16)), E4, first, scale_first),
        Spec(n + "moe.w13_alpha", (E,), F32, first, scalars_first("weight_scale_2")),
        Spec(n + "moe.a13_scale", (E,), F32, first, scalars_first("input_scale")),
        Spec(n + "moe.w2", (E, H, I // 2), U8, second, packed_second),
        Spec(n + "moe.w2_sf", (E, H * (I // 16)), E4, second, scale_second),
        Spec(n + "moe.w2_alpha", (E,), F32, second, scalars_second("weight_scale_2")),
        Spec(n + "moe.a2_scale", (E,), F32, second, scalars_second("input_scale")),
    ]


def _positive(t):
    if not torch.isfinite(t).all() or not (t > 0).all():
        raise ValueError("NVFP4 global scales must be positive finite multipliers")
    return t


def nvfp4_from_bf16(w: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """A BF16 matrix [out, in] as ModelOpt NVFP4: (packed U8 [out, in/2], E4M3 block scales [out, in/16], FP32 global
    multiplier) with the global chosen so the largest block scale is E4M3's largest. engine/modules/moe
    .quant_nvfp4_act is the encoder (per-16 amax / 6, nearest e2m1, ties to even)."""
    from engine.modules.moe import FP4_MAX, FP8_MAX, quant_nvfp4_act
    wf = w.float()
    amax = wf.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    global_scale = (amax / (FP4_MAX * FP8_MAX)).reshape(())
    packed, scale = quant_nvfp4_act(wf, global_scale)
    return packed.view(torch.uint8).contiguous(), scale.contiguous(), global_scale.to(F32)


def _mtp_routed_specs(n: str, m: str, F: Facts) -> "list[Spec]":
    """The MTP head's BF16 experts written as NVFP4 in the routed layout (see the module docstring)."""
    E, I, H = F.experts_local, F.moe_inter, F.hidden
    gate_up, down = m + "experts.gate_up_proj", m + "experts.down_proj"

    def encoded(s, r, part):
        cache = s.setdefault(("_mtp_nvfp4", r), {})
        if part in cache:
            return cache[part]
        ids = list(_expert_ids(F, r))
        w13, sf13, g13, w2, sf2, g2 = [], [], [], [], [], []
        for e in ids:
            gu = s[gate_up][e]                                           # [2I, H] rows [gate; up]
            first = torch.cat([gu[I:], gu[:I]], 0)                       # rows [up; gate], the kernel's order
            p, sc, gs = nvfp4_from_bf16(first)
            w13.append(p); sf13.append(sc); g13.append(gs)
            p, sc, gs = nvfp4_from_bf16(s[down][e])
            w2.append(p); sf2.append(sc); g2.append(gs)
        cache.update({
            "w13": torch.stack(w13).contiguous(),
            "w13_sf": swizzle_sf_batch(torch.stack(sf13).view(torch.uint8)).view(E4).contiguous(),
            "w13_alpha": torch.stack(g13), "a13_scale": torch.ones(E, dtype=F32),
            "w2": torch.stack(w2).contiguous(),
            "w2_sf": swizzle_sf_batch(torch.stack(sf2).view(torch.uint8)).view(E4).contiguous(),
            "w2_alpha": torch.stack(g2), "a2_scale": torch.ones(E, dtype=F32),
        })
        return cache[part]

    src = (gate_up, down)
    return [
        Spec(n + "moe.w13", (E, 2 * I, H // 2), U8, src, lambda s, r, W: encoded(s, r, "w13")),
        Spec(n + "moe.w13_sf", (E, 2 * I * (H // 16)), E4, src, lambda s, r, W: encoded(s, r, "w13_sf")),
        Spec(n + "moe.w13_alpha", (E,), F32, src, lambda s, r, W: encoded(s, r, "w13_alpha")),
        Spec(n + "moe.a13_scale", (E,), F32, src, lambda s, r, W: encoded(s, r, "a13_scale")),
        Spec(n + "moe.w2", (E, H, I // 2), U8, src, lambda s, r, W: encoded(s, r, "w2")),
        Spec(n + "moe.w2_sf", (E, H * (I // 16)), E4, src, lambda s, r, W: encoded(s, r, "w2_sf")),
        Spec(n + "moe.w2_alpha", (E,), F32, src, lambda s, r, W: encoded(s, r, "w2_alpha")),
        Spec(n + "moe.a2_scale", (E,), F32, src, lambda s, r, W: encoded(s, r, "a2_scale")),
    ]


def _moe_common_specs(n: str, m: str, F: Facts) -> "list[Spec]":
    H, Is = F.hidden, F.shared_inter_local
    gu = (m + "shared_expert.gate_proj.weight", m + "shared_expert.up_proj.weight")
    gates = (m + "gate.weight", m + "shared_expert_gate.weight")
    return [
        # the router's rows and the shared expert's gate row read the same input: one matmul, rows [experts; shared]
        Spec(n + "moe.gates", (F.experts + 1, H), BF, gates,
             lambda s, r, W, k=gates: torch.cat([s[k[0]], s[k[1]]], 0).contiguous()),
        Spec(n + "moe.sh_gate_up", (2 * Is, H), BF, gu,
             lambda s, r, W, k=gu: torch.cat([_split(s[k[0]], 0, r, W), _split(s[k[1]], 0, r, W)], 0).contiguous()),
        Spec(n + "moe.sh_down", (H, Is), BF, (m + "shared_expert.down_proj.weight",),
             _cols(m + "shared_expert.down_proj.weight")),
    ]


def top_specs(F: Facts) -> "list[Spec]":
    vp = F.vocab_local
    return ([Spec("embed", (vp, F.hidden), BF, (CK + "embed_tokens.weight",), _rows(CK + "embed_tokens.weight")),
             Spec("head", (vp, F.hidden), BF, ("lm_head.weight",), _rows("lm_head.weight"))]
            + _hc_specs("", CK, F, (("close", "hyper_connection_mixer"),)))


def layer_specs(F: Facts, L: int) -> "list[Spec]":
    p, n = f"{CK}layers.{L}.", f"L{L}."
    out = _hc_specs(n, p, F, (("attn", "attn_hyper_connection"), ("mlp", "mlp_hyper_connection")))
    out += _attention_specs(n, p + "self_attn.", F) if F.is_qsa(L) else _gdn_specs(n, p + "linear_attn.", F)
    out += _moe_common_specs(n, p + "mlp.", F) + _routed_specs(n, p + "mlp.", F)
    if L in F.ple_layers:
        out += ple_specs(F, L)
    return out


def ple_specs(F: Facts, L: int) -> "list[Spec]":
    """The injection's small tensors; the table itself is `ple_table_specs` (parts)."""
    p, n = f"{CK}layers.{L}.ple.", f"L{L}.ple."
    width = F.hc * F.hidden
    e = p + "ple_embedding."
    return [
        Spec(n + "scale", (1,), F32, (e + "ngram_embedding.weight_scale",), _whole(e + "ngram_embedding.weight_scale", F32)),
        # key [hc*H] and value [H] read the same looked-up rows: one matmul, rows [key; value] (NGramInjection's "kv")
        Spec(n + "kv_proj", (width + F.hidden, F.ple_dim), BF, (p + "key_proj.weight", p + "value_proj.weight"),
             lambda s, r, W, k=(p + "key_proj.weight", p + "value_proj.weight"): torch.cat([s[k[0]], s[k[1]]], 0).contiguous()),
        Spec(n + "conv", (width, F.ple_conv), F32, (p + "conv1d.weight",),
             lambda s, r, W, k=p + "conv1d.weight": s[k][:, 0, :].to(F32).contiguous()),
        Spec(n + "norm_conv", (width,), BF, (p + "norm_conv.weight",), _whole(p + "norm_conv.weight")),
        Spec(n + "norm_key", (width,), BF, (p + "norm_key.weight",), _whole(p + "norm_key.weight")),
        Spec(n + "norm_query", (width,), BF, (p + "norm_query.weight",), _whole(p + "norm_query.weight")),
        Spec(n + "heads_offsets", (F.ple_heads,), I64, (e + "ngram_heads_offsets",), _whole(e + "ngram_heads_offsets")),
        Spec(n + "heads_vocab", (F.ple_heads,), I64, (e + "ngram_heads_vocab_sizes",), _whole(e + "ngram_heads_vocab_sizes")),
        Spec(n + "layer_multipliers", (F.ngram_size,), I64, (e + "layer_multipliers",), _whole(e + "layer_multipliers")),
    ]


def ple_table_specs(F: Facts, L: int, part: int) -> "list[Spec]":
    """Part `part` of the rank's vocabulary range of the table: checkpoint shard (parts_per_rank * r + part)."""
    per_rank = F.ngram_parts // TP
    key = lambda shard: f"{CK}layers.{L}.ple.ple_embedding.ngram_embedding.shard_{shard}.weight"
    sources = tuple(key(per_rank * r + part) for r in range(TP))
    return [Spec(f"L{L}.ple.table.{part}", (PLE_SHARD_ROWS, F.ple_head_dim), E4, sources,
                 lambda s, r, W: s[key(per_rank * r + part)].contiguous())]


def mtp_specs(F: Facts) -> "list[Spec]":
    """The MTP head: its fuse, one QSA + MoE layer at model layer F.layers, and its closing mixer."""
    m, n = "mtp.", "mtp."
    L = f"{m}layers.0."
    out = [
        Spec(n + "fc_embedding", (F.hidden, F.hidden), BF, (m + "fc_embedding.weight",), _whole(m + "fc_embedding.weight")),
        Spec(n + "fc_hidden", (F.hidden, F.hidden), BF, (m + "fc_hidden.weight",), _whole(m + "fc_hidden.weight")),
        Spec(n + "pre_fc_norm_embedding", (F.hidden,), BF, (m + "pre_fc_norm_embedding.weight",),
             _whole(m + "pre_fc_norm_embedding.weight")),
        Spec(n + "pre_fc_norm_hidden", (F.hc * F.hidden,), BF, (m + "pre_fc_norm_hidden.weight",),
             _whole(m + "pre_fc_norm_hidden.weight")),
    ]
    out += _hc_specs(n, m, F, (("close", "hyper_connection_mixer"),))
    out += _hc_specs(n + "L0.", L, F, (("attn", "attn_hyper_connection"), ("mlp", "mlp_hyper_connection")))
    out += _attention_specs(n + "L0.", L + "self_attn.", F)
    out += _moe_common_specs(n + "L0.", L + "mlp.", F) + _mtp_routed_specs(n + "L0.", L + "mlp.", F)
    return out


def all_specs(F: Facts, layers=None, *, mtp: bool = True) -> "list[Spec]":
    return [s for _, _, specs_of in groups(F, layers, mtp=mtp) for s in specs_of(0)]


def groups(F: Facts, layers=None, *, mtp: bool = True):
    """(label, source keys, specs_of(rank)) per preshard group: the top, one group a layer (a layer's sources read
    once), the PLE table a part at a time, then the MTP head."""
    layers = range(F.layers) if layers is None else list(layers)
    yield "top", sorted({k for s in top_specs(F) for k in s.sources}), lambda r: top_specs(F)
    for L in layers:
        keys = sorted({k for s in layer_specs(F, L) for k in s.sources})
        yield f"layer {L}", keys, (lambda r, L=L: layer_specs(F, L))
        if L in F.ple_layers:
            for part in range(F.ngram_parts // TP):
                specs = ple_table_specs(F, L, part)
                yield f"layer {L} ple part {part}", sorted({k for s in specs for k in s.sources}), (lambda r, s=specs: s)
    if mtp and F.mtp_layers:
        keys = sorted({k for s in mtp_specs(F) for k in s.sources})
        yield "mtp", keys, lambda r: mtp_specs(F)


__all__ = ["CK", "PLE_SHARD_ROWS", "top_specs", "layer_specs", "ple_specs", "ple_table_specs", "mtp_specs",
           "all_specs", "groups", "nvfp4_from_bf16"]
