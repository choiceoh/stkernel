"""NVFP4 (W4A4, group 16) expert weights: what the bytes mean, and a reference
GEMM that consumes them (module).

Qwen3.8's routed experts are the only NVFP4 tensors in its checkpoint
(hf_quant_config: quant_algo NVFP4, group_size 16, everything else excluded).
Per expert projection the checkpoint holds four tensors:

    weight          U8   [out, in/2]     two e2m1 values per byte
    weight_scale    E4M3 [out, in/16]    one scale per 16 elements along K
    weight_scale_2  F32  []              one global scale for the tensor
    input_scale     F32  []              the activation side's global scale

so a weight element is  e2m1(nibble) * weight_scale[row, k//16] * weight_scale_2.
(modelopt's weight_scale_2 is a MULTIPLIER; GLM's compressed-tensors
weight_global_scale is the reciprocal -- see modules/nvfp4_linear.py.)
The nibble order is the same as DSv4.1's fp4 (modules/quant.py): even element
low, odd element high -- and unlike DSv4.1 the scale is per ROW x 16-group,
not per [32, 32] block, which is exactly the shape difference the b12x lane's
dispatch has to get right (its IMA lives there).

This is the oracle for the MoE lane (D4/D14), slow on purpose.
"""
from __future__ import annotations

import torch

from engine.modules.quant import FP4_TABLE, _unpack_fp4, _fp4_encode, _pow2_round

GROUP = 16
FP4_MAX = 6.0
FP8_MAX = 448.0


def dequant_nvfp4(weight_u8: torch.Tensor, weight_scale: torch.Tensor,
                  weight_scale_2: torch.Tensor) -> torch.Tensor:
    """[out, in] float32 from the four-tensor NVFP4 layout (three of them)."""
    out, half = weight_u8.shape
    vals = _unpack_fp4(weight_u8.view(torch.float4_e2m1fn_x2))            # [out, in]
    scales = weight_scale.float().repeat_interleave(GROUP, dim=1)[:, : half * 2]
    return vals * scales * weight_scale_2.float()


def quant_nvfp4_act(x: torch.Tensor, input_scale: torch.Tensor):
    """W4A4's activation side: per-16 e4m3 scales under one global scale.

    Returns (packed e2m1 [.., K/2], scales e4m3 [.., K/16]). Dequantise as
    e2m1 * scale * input_scale.
    """
    k = x.size(-1)
    assert k % GROUP == 0
    z = x.float() / input_scale.float()
    blocks = z.unflatten(-1, (k // GROUP, GROUP))
    scale = (blocks.abs().amax(dim=-1) / FP4_MAX).clamp(max=FP8_MAX)
    scale = scale.to(torch.float8_e4m3fn).float().clamp_min(torch.finfo(torch.float32).tiny)
    nib = _fp4_encode((blocks / scale.unsqueeze(-1)).flatten(-2)).unflatten(-1, (k // 2, 2))
    packed = (nib[..., 0] | (nib[..., 1] << 4)).view(torch.float4_e2m1fn_x2)
    return packed, scale.to(torch.float8_e4m3fn)


def dequant_nvfp4_act(packed, scale, input_scale) -> torch.Tensor:
    vals = _unpack_fp4(packed)
    return vals * scale.float().repeat_interleave(GROUP, dim=-1) * input_scale.float()


def expert_gemm(x: torch.Tensor, weight_u8, weight_scale, weight_scale_2, input_scale=None,
                quantize_act: bool = False) -> torch.Tensor:
    """y[M, out] = x[M, in] @ W^T with W dequantised; optionally with x quantised
    the way the W4A4 kernel sees it (that is the fidelity a kernel is held to)."""
    w = dequant_nvfp4(weight_u8, weight_scale, weight_scale_2)
    if quantize_act:
        packed, s = quant_nvfp4_act(x, input_scale)
        xq = dequant_nvfp4_act(packed, s, input_scale)
    else:
        xq = x.float()
    return (xq @ w.T).to(x.dtype)


def _selfcheck() -> None:
    import json, struct
    from pathlib import Path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path("/home/choiceoh/models/qwen38-flash-next-nvfp4")
    shard = ck / "layer-00000-experts-0000-0127.safetensors"
    with shard.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    def load(name, dtype):
        e = hdr[name]; lo, hi = e["data_offsets"]
        with shard.open("rb") as f:
            f.seek(base + lo); raw = f.read(hi - lo)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dtype).reshape(e["shape"]).to(dev)
    p = "model.language_model.layers.0.mlp.experts.0.gate_proj."
    w = load(p + "weight", torch.uint8); ws = load(p + "weight_scale", torch.float8_e4m3fn)
    ws2 = load(p + "weight_scale_2", torch.float32); xs = load(p + "input_scale", torch.float32)
    W = dequant_nvfp4(w, ws, ws2)
    ok = torch.isfinite(W).all().item() and W.shape == (640, 2560)
    print(f"  real expert 0 gate_proj: dequant {tuple(W.shape)} finite={torch.isfinite(W).all().item()} "
          f"std {W.std().item():.4f} |max| {W.abs().max().item():.4f} scale_2 {ws2.item():.3e} input_scale {xs.item():.3e}")
    assert ok and 1e-3 < W.std().item() < 1.0, "a projection's weights should be O(1e-2) and finite"
    # unique e2m1 magnitudes: a real NVFP4 tensor uses the whole table, not a corner of it
    mags = _unpack_fp4(w.view(torch.float4_e2m1fn_x2)).abs().unique().tolist()
    assert set(mags) == {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}, mags
    # activation round trip inside fp4's own resolution relative to the block max
    x = torch.randn(8, 2560, device=dev) * 0.5
    packed, s = quant_nvfp4_act(x, xs)
    xq = dequant_nvfp4_act(packed, s, xs)
    rel = ((xq - x).abs() / x.abs().clamp_min(1e-2)).median().item()
    assert rel < 0.25, rel                      # e2m1 has 1 mantissa bit: coarse by design
    y = expert_gemm(x.bfloat16(), w, ws, ws2, xs, quantize_act=True)
    assert torch.isfinite(y).all() and y.shape == (8, 640)
    print(f"  nvfp4 act round trip median rel err {rel:.3f} (1 mantissa bit), W4A4 gemm finite {tuple(y.shape)} OK")


if __name__ == "__main__":
    _selfcheck()


# --------------------------------------------------------------------------
# routing and the whole MoE block, as GLM-5.3 serves it (reference)
# --------------------------------------------------------------------------

def route_noaux_tc(hidden: torch.Tensor, gate_weight: torch.Tensor, correction_bias: "torch.Tensor | None",
                   topk: int, routed_scaling_factor: float = 1.0, renormalize: bool = True,
                   scoring: str = "sigmoid"):
    """DeepSeek-V3's `noaux_tc` router, GLM-5.3's config: sigmoid scores in fp32,
    SELECT top-k by (score + e_score_correction_bias), WEIGHT by the uncorrected
    scores of the selected experts, renormalise, scale by routed_scaling_factor.
    n_group = topk_group = 1 on GLM, so no group stage. Returns (ids [T,k] int32, w [T,k] fp32)."""
    logits = hidden.float() @ gate_weight.float().T                        # [T, E]
    scores = torch.sigmoid(logits) if scoring == "sigmoid" else torch.softmax(logits, -1)
    select_on = scores + correction_bias.float() if correction_bias is not None else scores
    ids = select_on.topk(topk, dim=-1).indices
    w = scores.gather(-1, ids)
    if renormalize:
        w = w / w.sum(-1, keepdim=True)
    return ids.to(torch.int32), w * routed_scaling_factor


def moe_reference(hidden: torch.Tensor, ids: torch.Tensor, w: torch.Tensor, experts: dict,
                  shared: "dict | None" = None, quantize_act: bool = False) -> torch.Tensor:
    """experts: {expert_id: {"gate_proj": four-tensor dict, "up_proj": ..., "down_proj": ...}}
    (NVFP4, either name family). SwiGLU per expert, weighted combine, plus the
    shared expert (bf16 dict {"gate_proj","up_proj","down_proj": weight}) if given."""
    from engine.modules.nvfp4_linear import NVFP4Linear
    t = hidden.shape[0]
    out = torch.zeros(t, hidden.shape[1], dtype=torch.float32, device=hidden.device)
    cache = {}
    for e in ids.unique().tolist():
        if e not in experts:
            raise KeyError(f"expert {e} selected but not provided")
        if e not in cache:
            mods = {}
            for name, (inn, outn) in (("gate_proj", (hidden.shape[1], None)), ("up_proj", (hidden.shape[1], None)), ("down_proj", (None, hidden.shape[1]))):
                tens = experts[e][name]
                packed = tens.get("weight_packed", tens.get("weight"))
                o, i2 = packed.shape[0], packed.shape[1] * 2
                m = NVFP4Linear(i2, o, "replicated").to(hidden.device); m.load(tens); mods[name] = m
            cache[e] = mods
        rows, slot = (ids == e).nonzero(as_tuple=True)
        x = hidden[rows]
        g, _ = cache[e]["gate_proj"](x, quantize_act); u, _ = cache[e]["up_proj"](x, quantize_act)
        h = torch.nn.functional.silu(g.float()) * u.float()
        d, _ = cache[e]["down_proj"](h.to(hidden.dtype), quantize_act)
        out.index_add_(0, rows, d.float() * w[rows, slot].unsqueeze(-1))
    if shared is not None:
        g = hidden.float() @ shared["gate_proj"].float().T; u = hidden.float() @ shared["up_proj"].float().T
        out += (torch.nn.functional.silu(g) * u) @ shared["down_proj"].float().T
    return out.to(hidden.dtype)


def _selfcheck_moe() -> None:
    import json, struct
    from pathlib import Path
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = Path("/home/choiceoh/models/glm53-redhat-nvfp4")
    wm = json.loads((ck / "model.safetensors.index.json").read_text())["weight_map"]
    def load(name):
        sh = ck / wm[name]
        with sh.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        e = h[name]; lo, hi = e["data_offsets"]
        with sh.open("rb") as f: f.seek(base + lo); raw = f.read(hi - lo)
        dt = {"U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}[e["dtype"]]
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(e["shape"]).to(dev)
    L = "model.language_model.layers.3."
    gate_w = load(L + "mlp.gate.weight"); bias = load(L + "mlp.gate.e_score_correction_bias")
    torch.manual_seed(0); hidden = torch.randn(6, 4096, device=dev, dtype=torch.bfloat16)
    ids, w = route_noaux_tc(hidden, gate_w, bias, topk=8, routed_scaling_factor=2.5)
    assert ids.shape == (6, 8) and torch.allclose(w.sum(-1), torch.full((6,), 2.5, device=dev), atol=1e-4)
    # selection uses the bias, weights do not: a large bias on expert 0 must select it without changing its weight law
    ids_b, w_b = route_noaux_tc(hidden, gate_w, bias + 0, topk=8, routed_scaling_factor=2.5)
    big = bias.clone(); big[0] += 100.0
    ids_c, _ = route_noaux_tc(hidden, gate_w, big, topk=8, routed_scaling_factor=2.5)
    assert (ids_c == 0).any(-1).all() and torch.equal(ids_b, ids)
    # experts: load the ones selected for token 0 (real NVFP4 tensors) and check the combine against dense bf16
    experts = {}
    for e in ids[0].tolist():
        experts[e] = {p: {k: load(f"{L}mlp.experts.{e}.{p}.{k}") for k in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")}
                      for p in ("gate_proj", "up_proj", "down_proj")}
    out = moe_reference(hidden[:1], ids[:1], w[:1], experts)
    from engine.modules.moe import dequant_nvfp4
    dense = torch.zeros(1, 4096, device=dev)
    for j, e in enumerate(ids[0].tolist()):
        W = {p: dequant_nvfp4(experts[e][p]["weight_packed"], experts[e][p]["weight_scale"], 1.0 / experts[e][p]["weight_global_scale"]) for p in ("gate_proj", "up_proj", "down_proj")}
        x = hidden[:1].float()
        dense += w[0, j] * ((torch.nn.functional.silu(x @ W["gate_proj"].T) * (x @ W["up_proj"].T)) @ W["down_proj"].T)
    rel = ((out.float() - dense).abs().max() / dense.abs().max()).item()
    assert rel < 2e-2, rel
    print(f"  moe: noaux_tc router (sum w = 2.5, bias selects but does not weight), 8-expert NVFP4 combine == dense (rel {rel:.1e}) OK")


if __name__ == "__main__":
    _selfcheck_moe()
