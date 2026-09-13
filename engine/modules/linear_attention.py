"""The gated delta rule (module): the linear attention GLM-5.3 (KDA) and
Qwen3.8 (GDN) share, as a torch reference.

One recurrence, written once, in the form every kernel in this family
implements (fla's fused_recurrent_gated_delta_rule; vLLM's vendored copy;
this repo's KDA Triton):

    h_t  = h_{t-1} * exp(g_t)                    per-head scalar decay, g <= 0
    u_t  = beta_t * (v_t - k_t^T h_{t-1}')        the delta: what the state got wrong
    h_t  = h_t + k_t u_t^T                        rank-1 correction
    o_t  = q_t h_t * scale

KDA differs from GDN in where the decay lives (per channel vs per head) and
in how g and beta are produced; those are the profile's projections, not
this recurrence. `decay_per_channel` covers the KDA shape when it is needed.

This file is the ORACLE for the family (CHARTER D4/D14): slow, exact, and the
thing the Triton lanes are judged against. Its own judge is HF's torch
implementation (transformers 5.16.1, modeling_qwen4_exp.py), which
probes/linear_attention_check.py holds it to. The convention that matched,
found by grid rather than assumed (2026-09-11):

    q, k l2-normalised before the recurrence   (use_qk_l2norm_in_kernel=True)
    scale = Dk ** -0.5, applied to q            (scale=1.0 misses by 0.98)
    decay per head                              (GDN; KDA's per-channel is a flag)

    max |o - HF|  9.8e-4   max |state - HF|  2.3e-3   at bf16 inputs, T=96

and the same numbers against BOTH HF forms, chunked and recurrent -- which is
the property the runner leans on: a prefill computed in chunks and a decode
continued one token at a time from its final state are the same recurrence.
"""
from __future__ import annotations

import torch


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def kda_output_norm(core: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """Per-head RMS norm, weight and sigmoid gate; round to BF16 only at output."""
    x = core.float()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
            * torch.sigmoid(gate.float())).to(core.dtype)


def kda_gate(raw_g: torch.Tensor, A_log: torch.Tensor, g_bias: "torch.Tensor | None",
             lower_bound: float = -5.0, safe_gate: bool = True) -> torch.Tensor:
    """GLM-5.3's per-channel log-decay, as its served kernel computes it
    (glm53_kernels/kda.py:1589-1608, the fused gate):

        safe_gate:  g = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + g_bias))   in (lower_bound, 0)
        else:       g = -exp(A_log[h]) * softplus(raw_g + g_bias)

    raw_g is [B, T, H, Dk] (f_b_proj), A_log [H], g_bias [H*Dk] (dt_bias).
    GLM-5.3 checkpoints take the safe branch (linear_attn_config has no
    safe_gate key and the model defaults it True, lower bound -5.0).
    """
    g = raw_g.float()
    if g_bias is not None:
        g = g + g_bias.float().view(1, 1, *raw_g.shape[-2:])
    a = torch.exp(A_log.float()).view(1, 1, -1, 1)
    if safe_gate:
        return lower_bound * torch.sigmoid(a * g)
    return -a * torch.nn.functional.softplus(g)


def gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     g: torch.Tensor, beta: torch.Tensor, initial_state=None,
                     scale: "float | None" = None, qk_l2norm: bool = True,
                     decay_per_channel: bool = False):
    """[B, T, H, Dk] q/k, [B, T, H, Dv] v, [B, T, H] g (log decay) and beta.

    Returns (o [B, T, H, Dv], final state [B, H, Dk, Dv]). Pure recurrence,
    fp32 inside, the caller's dtype outside.
    """
    b, t, h, dk = query.shape
    dv = value.shape[-1]
    q, k, v = (x.float() for x in (query, key, value))
    if qk_l2norm:
        q, k = l2norm(q), l2norm(k)
    if scale is None:
        scale = dk ** -0.5
    g, beta = g.float(), beta.float()
    state = (torch.zeros(b, h, dk, dv, dtype=torch.float32, device=q.device)
             if initial_state is None else initial_state.float().clone())
    out = torch.empty(b, t, h, dv, dtype=torch.float32, device=q.device)
    for i in range(t):
        decay = torch.exp(g[:, i])                              # [B, H] or [B, H, Dk] per channel
        state = state * (decay.unsqueeze(-1) if decay_per_channel else decay[..., None, None])
        k_i, v_i, q_i = k[:, i], v[:, i], q[:, i]               # [B, H, D]
        pred = torch.einsum("bhk,bhkv->bhv", k_i, state)         # k^T h
        u = (v_i - pred) * beta[:, i].unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k_i, u)
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state) * scale
    return out.to(query.dtype), state


def gdn_decay(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """GDN's per-head log-decay: -exp(A_log) * softplus(a + dt_bias), in fp32 (transformers qwen4_exp
    Qwen4ExpTextGatedDeltaNet: "if the model is loaded in fp16, without the .float() here, A might be -inf")."""
    return -A_log.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)


class GatedDeltaNet:
    """Gated DeltaNet as a token mixer (engine/base/composition.Feature): Qwen3.8's linear-attention layer
    (transformers qwen4_exp Qwen4ExpTextGatedDeltaNet).

    x -> in_proj_qkv -> causal conv (kernel `conv`, silu) -> q, k, v heads; beta = sigmoid(in_proj_b(x)); decay per head
    = gdn_decay(in_proj_a(x)); key heads repeated to the value heads; the gated delta rule (`gated_delta_rule`, q/k
    l2-normalised, scale Dk^-0.5); a gated RMS norm with z = in_proj_z(x); out_proj. Per sequence it carries the conv's
    last kernel-1 inputs and the fp32 recurrent state [HV, Dk, Dv].

    `weights(layer, name)`: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, conv1d ([C, K] or [C, 1, K]), dt_bias, A_log,
    norm, out_proj -- the transformers names under `linear_attn.`."""

    def __init__(self, *, k_heads: int, v_heads: int, k_dim: int, v_dim: int, conv: int, eps: float,
                 gate_activation: str, weights, activation: str = "silu"):
        if v_heads % k_heads:
            raise ValueError(f"{v_heads} value heads are not a multiple of {k_heads} key heads")
        self.k_heads, self.v_heads, self.k_dim, self.v_dim, self.conv = k_heads, v_heads, k_dim, v_dim, conv
        self.eps, self.gate_activation, self.activation, self.weights = eps, gate_activation, activation, weights

    @property
    def conv_dim(self) -> int:
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim

    def __call__(self, layer, x, step, state):
        from engine.modules.causal_conv import causal_conv1d
        from engine.modules.norm import rmsnorm_gated
        w = lambda name: self.weights(layer, name)
        out = None
        for s in step.segments:
            xs = x[s.start:s.start + s.length]
            t = xs.shape[0]
            qkv = torch.nn.functional.linear(xs, w("in_proj_qkv"))
            z = torch.nn.functional.linear(xs, w("in_proj_z")).reshape(t, self.v_heads, self.v_dim)
            beta = torch.nn.functional.linear(xs, w("in_proj_b")).sigmoid()
            g = gdn_decay(torch.nn.functional.linear(xs, w("in_proj_a")), w("A_log"), w("dt_bias"))
            conv_w = w("conv1d")
            qkv, conv_state = causal_conv1d(qkv, conv_w.reshape(conv_w.shape[0], -1), None,
                                            state.get(layer, "gdn_conv", s.seq), self.activation)
            q, k, v = torch.split(qkv, [self.k_heads * self.k_dim, self.k_heads * self.k_dim, self.v_heads * self.v_dim], -1)
            q = q.reshape(1, t, self.k_heads, self.k_dim)
            k = k.reshape(1, t, self.k_heads, self.k_dim)
            v = v.reshape(1, t, self.v_heads, self.v_dim)
            if self.v_heads != self.k_heads:
                q = q.repeat_interleave(self.v_heads // self.k_heads, dim=2)
                k = k.repeat_interleave(self.v_heads // self.k_heads, dim=2)
            core, recurrent = gated_delta_rule(q, k, v, g[None], beta[None], state.get(layer, "gdn_state", s.seq),
                                               scale=self.k_dim ** -0.5, qk_l2norm=True)
            core = rmsnorm_gated(core.reshape(-1, self.v_dim), z.reshape(-1, self.v_dim), w("norm"), self.eps,
                                 self.gate_activation).reshape(t, -1)
            ys = torch.nn.functional.linear(core, w("out_proj"))
            if out is None:
                out = xs.new_empty(x.shape[0], ys.shape[-1])
            out[s.start:s.start + s.length] = ys
            state.put(layer, "gdn_conv", s.seq, conv_state)
            state.put(layer, "gdn_state", s.seq, recurrent)
        return out

    def cache_specs(self, layers):
        from engine.base.cache_spec import SlotSpec
        return [SlotSpec("gdn conv state", len(layers), self.conv_dim * (self.conv - 1) * 2,
                         "[conv_dim, kernel-1] bf16: the conv's last inputs (modules/causal_conv)"),
                SlotSpec("gdn recurrent state", len(layers), self.v_heads * self.k_dim * self.v_dim * 4,
                         "[HV, Dk, Dv] fp32 (mamba_ssm_dtype float32)")]
