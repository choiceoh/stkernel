"""GLM-5.3-Flash's weights as this rank holds them (profile): the layout,
written down once, that both the preshard and the model read.

Every tensor the model touches is a Spec (base/params.py): its rank-local
shape, its dtype, the checkpoint tensors it is built from and HOW. Placement
follows the launcher (facts.py): TP=4 by heads for KDA and MLA, replicated
low-rank/indexer/router/mHC/norm tensors, vocab-parallel embed and head,
routed experts TP-split on the intermediate dim as packed NVFP4 bytes
(modules/nvfp4_linear: a row split takes rows of packed and scale alike; a
column split takes in/2 packed columns and in/16 scale columns -- exact
because 16 divides every boundary here).

Merges the served path does at load are done here instead, once: q|k|v|b|
f_a|g_a into one KDA input projection, q_a|kv_a into one MLA down-projection,
the three KDA convs into one [3HD, K] bank, gate|up into one dense/shared
GEMM, and per-expert up|gate into `w13` (b12x's row order). Dtype promotions likewise (A_log,
dt_bias, mHC, indexer head-gate and k_norm to fp32: what the kernels read).
The routed experts are written the way the served b12x lane eats them
(flashinfer_b12x_moe.process_weights_after_loading): packed nibbles as is,
block scales with the per-expert global scale FOLDED in (block *= 1/w_gs,
so alpha = 1) and INTERLEAVED in 128x4 tiles (modules/nvfp4_sf). The
calibrated input_global_scale is not written: the SM12x kernel quantises
activations dynamically per block (it forces fc2's to 1.0 and passes none
for fc1), and the reference lane does the same. Nothing is repacked at
boot (D1).
"""
from __future__ import annotations

import torch

from engine.base.params import Spec
from engine.modules.nvfp4_sf import swizzle_sf
from engine.profiles.glm53.facts import TP, Facts

CK = "model.language_model."
BF, F32, U8, E4 = torch.bfloat16, torch.float32, torch.uint8, torch.float8_e4m3fn


def _split(t, dim, r, W):
    n = t.shape[dim] // W
    if n * W != t.shape[dim]:
        raise ValueError(f"dim {dim} of {tuple(t.shape)} does not split {W} ways")
    return t.narrow(dim, r * n, n)


def _rows(key):
    return lambda s, r, W: _split(s[key], 0, r, W).contiguous()


def _cols(key):
    return lambda s, r, W: _split(s[key], 1, r, W).contiguous()


def _whole(key, dtype=None):
    return lambda s, r, W: s[key].contiguous() if dtype is None else s[key].to(dtype).contiguous()


def _cat_rows(keys, dtype=None):
    def build(s, r, W):
        parts = [_split(s[k], 0, r, W) for k in keys]
        t = torch.cat(parts, 0)
        return (t if dtype is None else t.to(dtype)).contiguous()
    return build


def top_specs(F: Facts) -> "list[Spec]":
    vp = F.vocab_local
    return [
        Spec("embed", (vp, F.hidden), BF, (CK + "embed_tokens.weight",), _rows(CK + "embed_tokens.weight")),
        Spec("norm", (F.hidden,), BF, (CK + "norm.weight",), _whole(CK + "norm.weight")),
        Spec("head", (vp, F.hidden), BF, ("lm_head.weight",), _rows("lm_head.weight")),
    ]


def layer_specs(F: Facts, L: int) -> "list[Spec]":
    p = f"{CK}layers.{L}."
    n = f"L{L}."
    H = F.hidden
    out = [
        Spec(n + "in_norm", (H,), BF, (p + "input_layernorm.weight",), _whole(p + "input_layernorm.weight")),
        Spec(n + "post_norm", (H,), BF, (p + "post_attention_layernorm.weight",), _whole(p + "post_attention_layernorm.weight")),
    ]
    mix = (2 + F.hc) * F.hc
    for side in ("attn", "ffn"):
        out += [
            Spec(n + f"hc.{side}_fn", (mix, F.hc * H), F32, (p + f"hc_{side}_fn",), _whole(p + f"hc_{side}_fn", F32)),
            Spec(n + f"hc.{side}_base", (mix,), F32, (p + f"hc_{side}_base",), _whole(p + f"hc_{side}_base", F32)),
            Spec(n + f"hc.{side}_scale", (3,), F32, (p + f"hc_{side}_scale",), _whole(p + f"hc_{side}_scale", F32)),
        ]
    a = p + "self_attn."
    if F.is_dsa(L):
        Hl = F.heads_local
        qkv_a = (a + "q_a_proj.weight", a + "kv_a_proj_with_mqa.weight")
        out += [
            Spec(n + "mla.qkv_a", (F.q_lora + F.kv_lora, H), BF, qkv_a, lambda s, r, W, k=qkv_a: torch.cat([s[k[0]], s[k[1]]], 0).contiguous()),
            Spec(n + "mla.q_a_norm", (F.q_lora,), BF, (a + "q_a_layernorm.weight",), _whole(a + "q_a_layernorm.weight")),
            Spec(n + "mla.kv_a_norm", (F.kv_lora,), BF, (a + "kv_a_layernorm.weight",), _whole(a + "kv_a_layernorm.weight")),
            Spec(n + "mla.q_b", (Hl * F.qk_nope, F.q_lora), BF, (a + "q_b_proj.weight",), _rows(a + "q_b_proj.weight")),
            Spec(n + "mla.kv_b", (Hl * (F.qk_nope + F.v_dim), F.kv_lora), BF, (a + "kv_b_proj.weight",), _rows(a + "kv_b_proj.weight")),
            Spec(n + "mla.o_proj", (H, Hl * F.v_dim), BF, (a + "o_proj.weight",), _cols(a + "o_proj.weight")),
        ]
        i = a + "indexer."
        out += [
            Spec(n + "idx.wq_b", (F.idx_heads * F.idx_dim, F.q_lora), BF, (i + "wq_b.weight",), _whole(i + "wq_b.weight")),
            Spec(n + "idx.wk", (F.idx_dim, H), BF, (i + "wk.weight",), _whole(i + "wk.weight")),
            Spec(n + "idx.w_heads", (F.idx_heads, H), F32, (i + "weights_proj.weight",), _whole(i + "weights_proj.weight", F32)),
            Spec(n + "idx.k_norm_w", (F.idx_dim,), F32, (i + "k_norm.weight",), _whole(i + "k_norm.weight", F32)),
            Spec(n + "idx.k_norm_b", (F.idx_dim,), F32, (i + "k_norm.bias",), _whole(i + "k_norm.bias", F32)),
            Spec(n + "idx.gate", (F.idx_dim, H), BF, (i + "index_kpool_compress_gate",), _whole(i + "index_kpool_compress_gate")),
            Spec(n + "idx.ape", (F.kpool, F.idx_dim), F32, (i + "index_kpool_compress_ape",), _whole(i + "index_kpool_compress_ape", F32)),
        ]
    else:
        Hl, D, K = F.kda_heads_local, F.kda_dim, F.conv
        proj = tuple(a + s for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "b_proj.weight"))
        rep = (a + "f_a_proj.weight", a + "g_a_proj.weight")
        convs = tuple(a + s for s in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"))

        def in_proj(s, r, W, proj=proj, rep=rep):
            return torch.cat([_split(s[k], 0, r, W) for k in proj] + [s[k] for k in rep], 0).contiguous()

        def conv(s, r, W, convs=convs):
            return torch.cat([_split(s[k][:, 0, :], 0, r, W) for k in convs], 0).to(F32).contiguous()

        out += [
            Spec(n + "kda.in_proj", (3 * Hl * D + Hl + 2 * D, H), BF, proj + rep, in_proj),
            Spec(n + "kda.f_b", (Hl * D, D), BF, (a + "f_b_proj.weight",), _rows(a + "f_b_proj.weight")),
            Spec(n + "kda.g_b", (Hl * D, D), BF, (a + "g_b_proj.weight",), _rows(a + "g_b_proj.weight")),
            Spec(n + "kda.conv", (3 * Hl * D, K), F32, convs, conv),
            Spec(n + "kda.A_log", (Hl,), F32, (a + "A_log",), lambda s, r, W, k=a + "A_log": _split(s[k], 0, r, W).to(F32).contiguous()),
            Spec(n + "kda.dt_bias", (Hl * D,), F32, (a + "dt_bias",), lambda s, r, W, k=a + "dt_bias": _split(s[k], 0, r, W).to(F32).contiguous()),
            Spec(n + "kda.o_norm", (D,), BF, (a + "o_norm.weight",), _whole(a + "o_norm.weight")),
            Spec(n + "kda.o_proj", (H, Hl * D), BF, (a + "o_proj.weight",), _cols(a + "o_proj.weight")),
        ]
    m = p + "mlp."
    if not F.is_moe(L):
        Il = F.dense_inter_local
        gu = (m + "gate_proj.weight", m + "up_proj.weight")
        out += [
            Spec(n + "mlp.gate_up", (2 * Il, H), BF, gu, _cat_rows(gu)),
            Spec(n + "mlp.down", (H, Il), BF, (m + "down_proj.weight",), _cols(m + "down_proj.weight")),
        ]
    else:
        Is = F.moe_inter_local
        E = F.experts
        gu = (m + "shared_experts.gate_proj.weight", m + "shared_experts.up_proj.weight")
        ex = [m + f"experts.{e}." for e in range(E)]
        w13_src = tuple(x + f"{g}_proj.{t}" for x in ex for g in ("gate", "up") for t in ("weight_packed", "weight_scale", "weight_global_scale"))
        w2_src = tuple(x + f"down_proj.{t}" for x in ex for t in ("weight_packed", "weight_scale", "weight_global_scale"))

        # w13 rows are [up; gate]: the order the b12x (flashinfer CuTe-DSL) kernel gates on -- vLLM swaps its [gate; up]
        # to that at load (reorder_w13_to_w31_for_flashinfer_cutedsl); here the file is written that way once
        def w13_packed(s, r, W):
            return torch.stack([torch.cat([_split(s[x + "up_proj.weight_packed"], 0, r, W),
                                           _split(s[x + "gate_proj.weight_packed"], 0, r, W)], 0) for x in ex]).contiguous()

        def w2_packed(s, r, W):
            return torch.stack([_split(s[x + "down_proj.weight_packed"], 1, r, W) for x in ex]).contiguous()

        def fold(scale, global_scale):                     # compressed-tensors: the global is a DIVISOR; the served layer bakes 1/w_gs in
            return (scale.float() / global_scale.float().reshape(())).to(E4)

        def w13_sf(s, r, W):
            rows = []
            for x in ex:
                g = fold(_split(s[x + "gate_proj.weight_scale"], 0, r, W), s[x + "gate_proj.weight_global_scale"])
                u = fold(_split(s[x + "up_proj.weight_scale"], 0, r, W), s[x + "up_proj.weight_global_scale"])
                rows.append(swizzle_sf(torch.cat([u, g], 0).view(torch.uint8)).view(E4))
            return torch.stack(rows).contiguous()

        def w2_sf(s, r, W):
            return torch.stack([swizzle_sf(fold(_split(s[x + "down_proj.weight_scale"], 1, r, W), s[x + "down_proj.weight_global_scale"]).view(torch.uint8)).view(E4)
                                for x in ex]).contiguous()

        out += [
            Spec(n + "moe.gate", (E, H), BF, (m + "gate.weight",), _whole(m + "gate.weight")),
            Spec(n + "moe.bias", (E,), F32, (m + "gate.e_score_correction_bias",), _whole(m + "gate.e_score_correction_bias", F32)),
            Spec(n + "moe.sh_gate_up", (2 * Is, H), BF, gu, _cat_rows(gu)),
            Spec(n + "moe.sh_down", (H, Is), BF, (m + "shared_experts.down_proj.weight",), _cols(m + "shared_experts.down_proj.weight")),
            Spec(n + "moe.w13", (E, 2 * Is, H // 2), U8, w13_src, w13_packed),          # rows [up; gate] (the kernel's order)
            Spec(n + "moe.w13_sf", (E, 2 * Is * (H // 16)), E4, w13_src, w13_sf),        # folded + 128x4 interleaved (b12x layout), same order
            Spec(n + "moe.w2", (E, H, Is // 2), U8, w2_src, w2_packed),
            Spec(n + "moe.w2_sf", (E, H * (Is // 16)), E4, w2_src, w2_sf),
        ]
    return out


def all_specs(F: Facts, layers=None) -> "list[Spec]":
    layers = range(F.layers) if layers is None else layers
    out = top_specs(F)
    for L in layers:
        out += layer_specs(F, L)
    return out


def groups(F: Facts, layers=None):
    """(label, source keys, specs_of(rank)) per preshard group: the top-level
    trio, then one group per layer, so a layer's sources are read once."""
    layers = range(F.layers) if layers is None else list(layers)
    yield "top", sorted({k for s in top_specs(F) for k in s.sources}), lambda r: top_specs(F)
    for L in layers:
        keys = sorted({k for s in layer_specs(F, L) for k in s.sources})
        yield f"layer {L}", keys, (lambda r, L=L: layer_specs(F, L))


def _selfcheck() -> None:
    from engine.base.params import total_bytes
    from engine.profiles.glm53 import facts, plan
    F = facts.load()
    W = TP
    specs = all_specs(F)
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), "duplicate spec names"
    gib = total_bytes(specs) / 2**30
    # plan.py's census of the same bytes, minus the MTP layer it counts (its experts fell into the expert
    # rule there) -- the only differences left are this file's fp32 promotions and replicated f_a/g_a
    import json, struct
    wm = json.loads((facts.CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    mtp_experts = 0
    for shard in sorted(set(wm.values())):
        with (facts.CKPT / shard).open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]; h = json.loads(fh.read(n)); h.pop("__metadata__", None)
        mtp_experts += sum(e["data_offsets"][1] - e["data_offsets"][0] for k, e in h.items() if k.startswith(f"{CK}layers.{F.layers}.mlp.experts."))
    census = sum(r[2] for r in plan.census() if not r[0].startswith("MTP")) - mtp_experts / W / 2**30
    kda = [s for s in specs if s.name == "L0.kda.in_proj"][0]
    assert kda.shape == (3 * 16 * 128 + 16 + 256, 4096)
    moe = [s for s in specs if s.name == "L3.moe.w13"][0]
    assert moe.shape == (288, 1024, 2048) and [s for s in specs if s.name == "L3.moe.w2"][0].shape == (288, 4096, 256)
    assert [s for s in specs if s.name == "L3.moe.w13_sf"][0].shape == (288, 1024 * 256)
    # every checkpoint tensor of the text model is a source of exactly the specs that need it, except the calibrated
    # input_global_scale the served lane never reads (dynamic activation quant)
    text = {k for k in wm if not k.startswith("model.visual.") and not k.startswith(f"{CK}layers.{F.layers}.") and not k.endswith("input_global_scale")}
    used = {k for s in specs for k in s.sources}
    missing = sorted(text - used)
    assert not missing, f"checkpoint tensors no spec reads: {missing[:5]} (+{len(missing)})"
    assert not (used - set(wm)), "spec reads a tensor the checkpoint does not have"
    print(f"  specs: {len(specs):,} tensors/rank at TP={W}, {gib:.2f} GiB (plan census {census:.2f}: "
          f"{'agrees' if 0 <= gib - census < 0.15 else 'DISAGREES'}); every text tensor of the checkpoint is consumed OK")


if __name__ == "__main__":
    _selfcheck()
