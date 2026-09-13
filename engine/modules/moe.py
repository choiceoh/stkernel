"""The MoE family (module): one router, one expert loop, three activations and three ways shared experts join --
the channel mixer Qwen3.8, GLM-5.3, DeepSeek-V3 / Ling-3.0, MiniMax-M3, Inkling and Kimi K3 share (`MoE`, `Dense`,
`route`, `apply_experts`, `gated_mlp`, `named`/`experts_of`/`shared_of`; held to transformers on the CPU in
tests/test_engine_moe_family.py). First, NVFP4 (W4A4, group 16) expert weights: what the bytes mean, and a reference
GEMM that consumes them.

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
not per 32-group under an E8M0 scale (modules/quant.fp4_gemm), which is exactly the shape difference the b12x lane's
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


def route_softmax_topk(logits: torch.Tensor, k: int, normalize: bool) -> "tuple[torch.Tensor, torch.Tensor]":
    """The softmax top-k router (Qwen3-Next's, Qwen3.8's: transformers qwen4_exp Qwen4ExpTextTopKRouter): probabilities
    in fp32 over every expert, the k largest, renormalised to sum to one when `normalize` -> (ids [N, k], weights [N, k]
    in the logits' dtype)."""
    probs = torch.nn.functional.softmax(logits, dtype=torch.float, dim=-1)
    top, ids = torch.topk(probs, k, dim=-1)
    if normalize:
        top = top / top.sum(dim=-1, keepdim=True)
    return ids, top.to(logits.dtype)


# --------------------------------------------------------------------------
# the MoE family: one router, one expert loop, the axes the seven models differ on
# --------------------------------------------------------------------------

def gated_mlp(gate: torch.Tensor, up: torch.Tensor, activation) -> torch.Tensor:
    """The expert nonlinearity on (gate, up) -- the family's three:

        "silu"                       silu(gate) * up                                     (Qwen3.8, DeepSeek, Ling, Inkling)
        ("swiglu_clamped", limit)    silu(min(gate, limit)) * clamp(up, -limit, limit)   (GLM-5.3: swiglu_limit 10)
        ("swigluoai", alpha, limit)  (clamp(up) + 1) * g * sigmoid(alpha * g), g = min(gate, limit)   (MiniMax-M3)
    """
    if activation == "silu":
        return torch.nn.functional.silu(gate) * up
    kind = activation[0]
    if kind == "swiglu_clamped":
        limit = activation[1]
        return torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
    if kind == "swigluoai":
        alpha, limit = activation[1], activation[2]
        gate, up = gate.clamp(max=limit), up.clamp(min=-limit, max=limit)
        return (up + 1.0) * (gate * torch.sigmoid(gate * alpha))
    raise ValueError(f"activation is 'silu', ('swiglu_clamped', limit) or ('swigluoai', alpha, limit), not {activation!r}")


def route(x: torch.Tensor, weight: torch.Tensor, *, score: str, topk: int, bias: "torch.Tensor | None" = None,
          groups: "tuple[int, int] | None" = None, normalize: bool = True, scaling: float = 1.0, fp32: bool = True,
          sink: int = 0, sink_scale: "torch.Tensor | None" = None):
    """The family's router: (ids [N, k] int64, weights [N, k], gammas [N, sink] or None).

    score "softmax": probabilities in fp32 over every expert, the k largest, renormalised when `normalize`, in the
        logits' dtype (transformers qwen4_exp Qwen4ExpTextTopKRouter).
    score "sigmoid": sigmoid scores; the CHOICE by score + bias (the correction bias never weights); with `groups`
        (n_group, topk_group) the choice is confined to the topk_group groups whose top-2 sum is largest
        (DeepSeek-V3's noaux_tc: deepseek_v3, glm5_next, Ling-3.0); the chosen experts' scores, renormalised when
        `normalize` (+1e-20 in the denominator as the models write it), times `scaling`. `fp32`: logits and scores
        in fp32 (deepseek_v3, glm5_next); else the logits in the model dtype and the sigmoid in fp32 (minimax_m3_vl).
    `sink` > 0: Inkling's shared experts are scored by the router too -- the weight has E + sink rows; the chosen
        experts' logits and the shared logits normalise together, exp(logsigmoid - logsumexp), times `scaling` and
        `sink_scale` (the router's global_scale); the last `sink` weights are the shared experts' gammas."""
    if sink:
        logits = torch.nn.functional.linear(x, weight)                                             # [N, E + S]
        scores = torch.sigmoid(logits)
        routed_scores, routed_logits = scores[:, :-sink], logits[:, :-sink]
        choice = routed_scores + bias if bias is not None else routed_scores
        ids = torch.topk(choice, topk, dim=-1, sorted=False).indices
        chosen = torch.cat([routed_logits.gather(-1, ids), logits[:, -sink:]], dim=-1)
        log_probs = torch.nn.functional.logsigmoid(chosen)
        weights = torch.exp(log_probs - torch.logsumexp(log_probs, dim=-1, keepdim=True)) * scaling
        if sink_scale is not None:
            weights = weights * sink_scale
        return ids, weights[:, :topk], weights[:, topk:]
    if score == "softmax":
        ids, weights = route_softmax_topk(torch.nn.functional.linear(x, weight), topk, normalize)
        return ids, weights * scaling if scaling != 1.0 else weights, None
    if score != "sigmoid":
        raise ValueError(f"score is 'softmax' or 'sigmoid', not {score!r}")
    if fp32:
        logits = torch.nn.functional.linear(x.float(), weight.float())
    else:
        logits = torch.nn.functional.linear(x.to(weight.dtype), weight).float()
    scores = torch.sigmoid(logits)
    choice = scores + bias.float() if bias is not None else scores
    if groups is not None:
        n_group, topk_group = groups
        experts = weight.shape[0]
        group_scores = choice.view(-1, n_group, experts // n_group).topk(2, dim=-1).values.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
        score_mask = group_mask[:, :, None].expand(-1, n_group, experts // n_group).reshape(-1, experts)
        choice = choice.masked_fill(~score_mask.bool(), float("-inf"))
    ids = torch.topk(choice, k=topk, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return ids, weights * scaling, None


def apply_experts(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, experts: int, expert, activation):
    """sum_k w_k expert_k(x) over the chosen experts, each a gated MLP `expert(e)` -> (gate_up [2I, H], down [H, I]),
    experts adding their tokens in ascending id (the reference loop of every transformers MoE)."""
    linear = torch.nn.functional.linear
    out = torch.zeros_like(x)
    mask = torch.nn.functional.one_hot(ids, num_classes=experts).permute(2, 1, 0)
    for e in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
        e = int(e[0])
        slot, tokens = torch.where(mask[e])
        gate_up, down = expert(e)
        gate, up = linear(x[tokens], gate_up).chunk(2, dim=-1)
        y = linear(gated_mlp(gate, up, activation), down) * weights[tokens, slot, None]
        out.index_add_(0, tokens, y.to(out.dtype))
    return out


class Dense:
    """A gated MLP as a channel mixer (engine/base/composition.Feature): the dense layers before the MoE ones (GLM-5.3
    3, DeepSeek 3, Ling-3.0 2, MiniMax-M3 3, Inkling 2), with the family's activation; `scale` (Inkling's global_scale)
    multiplies the output when the checkpoint has one.

    `weights(layer, name)`: gate_up [2I, H], down [H, I], scale (optional scalar)."""

    def __init__(self, *, activation="silu", weights):
        gated_mlp(torch.zeros(1), torch.zeros(1), activation)                       # the activation is checked here
        self.activation, self.weights = activation, weights

    def __call__(self, layer, x, step=None, state=None):
        linear = torch.nn.functional.linear
        gate, up = linear(x, self.weights(layer, "gate_up")).chunk(2, dim=-1)
        y = linear(gated_mlp(gate, up, self.activation), self.weights(layer, "down"))
        try:
            scale = self.weights(layer, "scale")
        except KeyError:
            return y
        return y * scale


class MoE:
    """Routed experts and shared experts as a channel mixer (engine/base/composition.Feature): the family Qwen3.8,
    GLM-5.3, DeepSeek-V3 / Ling-3.0, MiniMax-M3 and Inkling share (`route`, `apply_experts`, `gated_mlp`). Axes:

        score, bias, groups, normalize, scaling, router_fp32     the router (`route`)
        scaling_on   "weights" (the weights carry routed_scaling_factor: DeepSeek, GLM, Ling, Inkling's route_scale)
                     | "output" (the routed sum is scaled, in the model dtype: MiniMax-M3)
        shared, shared_mode   how many shared experts and how they join: None | "plain" (added: GLM, DeepSeek, Ling,
                     M3, Kimi K3) | "sigmoid" (times sigmoid(shared_gate(x)): Qwen3.8) | "sink" (scored by the router,
                     each shared expert's gamma from the joint normalisation, summed in fp32: Inkling)
        activation   "silu" | ("swiglu_clamped", limit) | ("swigluoai", alpha, limit)

    `weights(layer, name)`: router [E (+ shared, "sink"), H], router_bias [E] (optional), router_scale (scalar,
    "sink", optional), shared_gate [1, H] ("sigmoid"). `expert(layer, e)` -> (gate_up [2I, H], down [H, I]) as
    floats: the transformers fused tensors sliced, or a checkpoint's NVFP4 experts dequantised (`dequant_nvfp4`),
    whichever the profile holds; `shared_expert(layer, i)` the same for shared expert i (a "plain" model's one wide
    MLP is its shared expert 0)."""

    def __init__(self, *, experts: int, topk: int, score: str = "sigmoid", bias: bool = False, groups=None,
                 normalize: bool = True, scaling: float = 1.0, scaling_on: str = "weights", router_fp32: bool = True,
                 shared: int = 0, shared_mode: "str | None" = None, activation="silu", weights, expert, shared_expert=None):
        if not 0 < topk <= experts:
            raise ValueError(f"top-{topk} of {experts} experts")
        if score not in ("softmax", "sigmoid"):
            raise ValueError(f"score is 'softmax' or 'sigmoid', not {score!r}")
        if groups is not None and (score != "sigmoid" or experts % groups[0] or groups[1] > groups[0]):
            raise ValueError("groups (n_group, topk_group): a sigmoid router, n_group dividing the experts, topk_group <= n_group")
        if scaling_on not in ("weights", "output"):
            raise ValueError(f"scaling_on is 'weights' or 'output', not {scaling_on!r}")
        if shared_mode not in (None, "plain", "sigmoid", "sink") or (shared > 0) != (shared_mode is not None):
            raise ValueError("shared experts come with a mode: None, 'plain', 'sigmoid' or 'sink'")
        if shared_mode == "sink" and score != "sigmoid":
            raise ValueError("Inkling's shared-expert sink is a sigmoid router's")
        if shared and shared_expert is None:
            raise ValueError("shared experts need `shared_expert(layer, i)`")
        gated_mlp(torch.zeros(1), torch.zeros(1), activation)
        self.experts, self.topk, self.score, self.bias, self.groups = experts, topk, score, bias, groups
        self.normalize, self.scaling, self.scaling_on, self.router_fp32 = normalize, scaling, scaling_on, router_fp32
        self.shared, self.shared_mode, self.activation = shared, shared_mode, activation
        self.weights, self.expert, self.shared_expert = weights, expert, shared_expert

    def __call__(self, layer, x, step=None, state=None):
        linear = torch.nn.functional.linear
        w = lambda name: self.weights(layer, name)
        sink = self.shared if self.shared_mode == "sink" else 0
        ids, weights, gammas = route(
            x, w("router"), score=self.score, topk=self.topk, bias=w("router_bias") if self.bias else None,
            groups=self.groups, normalize=self.normalize, scaling=self.scaling if self.scaling_on == "weights" else 1.0,
            fp32=self.router_fp32, sink=sink, sink_scale=self._optional(w, "router_scale") if sink else None)
        out = apply_experts(x, ids, weights, self.experts, lambda e: self.expert(layer, e), self.activation)
        if self.scaling_on == "output":
            out = out * self.scaling
        if not self.shared:
            return out
        if self.shared_mode == "sink":
            total = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
            for i in range(self.shared):
                gate_up, down = self.shared_expert(layer, i)
                gate, up = linear(x, gate_up).chunk(2, dim=-1)
                total = total + linear(gated_mlp(gate, up, self.activation) * gammas[:, i, None], down).float()
            return out + total.to(x.dtype)
        gate_up, down = self.shared_expert(layer, 0)
        gate, up = linear(x, gate_up).chunk(2, dim=-1)
        shared = linear(gated_mlp(gate, up, self.activation), down)
        if self.shared_mode == "sigmoid":
            shared = torch.sigmoid(linear(x, w("shared_gate"))) * shared
        return out + shared

    @staticmethod
    def _optional(w, name):
        try:
            return w(name)
        except KeyError:
            return None


# The family's canonical names on each checkpoint's, and the fused expert layouts every transformers MoE keeps
# (experts.gate_up_proj [E, 2I, H], experts.down_proj [E, H, I]); shared experts differ in layout per model.
SCHEMES = {
    "qwen4_exp": {"router": "gate", "shared_gate": "shared_expert_gate",
                  "shared": ("shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj")},
    "glm5_next": {"router": "gate", "router_bias": "gate.e_score_correction_bias",
                  "shared": ("shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj")},
    "deepseek_v3": {"router": "gate", "router_bias": "gate.e_score_correction_bias",
                    "shared": ("shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj")},
    "minimax_m3_vl": {"router": "gate", "router_bias": "gate.e_score_correction_bias",
                      "shared_fused": ("shared_experts.gate_up_proj", "shared_experts.down_proj")},
    "inkling": {"router": "gate", "router_bias": "gate.e_score_correction_bias", "router_scale": "gate.global_scale",
                "shared_stacked": ("shared_experts.gate_proj", "shared_experts.up_proj", "shared_experts.down_proj")},
}


def named(scheme: str, source):
    """canonical name -> tensor over `source(checkpoint name)`; KeyError for a name the scheme or the checkpoint lacks."""
    table = SCHEMES[scheme]

    def get(name: str) -> torch.Tensor:
        if name not in table or not isinstance(table[name], str):
            raise KeyError(name)
        return source(table[name])
    return get


def experts_of(source, prefix: str = "experts."):
    """expert(e) -> (gate_up [2I, H], down [H, I]) from the fused 3D tensors."""
    return lambda e: (source(f"{prefix}gate_up_proj")[e], source(f"{prefix}down_proj")[e])


def shared_of(scheme: str, source):
    """shared(i) -> (gate_up [2I, H], down [H, I]) for the scheme's shared-expert layout: separate gate/up/down
    matrices (one wide expert), a fused gate_up (MiniMax-M3), or stacked [S, ...] tensors (Inkling)."""
    table = SCHEMES[scheme]
    if "shared" in table:
        g, u, d = table["shared"]
        return lambda i: (torch.cat([source(g), source(u)], dim=0), source(d))
    if "shared_fused" in table:
        gu, d = table["shared_fused"]
        return lambda i: (source(gu), source(d))
    g, u, d = table["shared_stacked"]
    return lambda i: (torch.cat([source(g)[i], source(u)[i]], dim=0), source(d)[i])
