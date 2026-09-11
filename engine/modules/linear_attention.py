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
