"""The gated delta rule (module): the linear-attention family GLM-5.3 (KDA), Qwen3.8 (GDN), Kimi K3 (KDA, full-rank
gate) and Ling-3.0 (KDA without LoRA) share -- one recurrence, one feature, and the axes the four differ on.

One recurrence, written once, in the form every kernel in this family
implements (fla's fused_recurrent_gated_delta_rule; vLLM's vendored copy;
this repo's KDA Triton):

    h_t  = h_{t-1} * exp(g_t)                    log-decay g <= 0, per head or per key channel
    u_t  = beta_t * (v_t - k_t^T h_{t-1}')        the delta: what the state got wrong
    h_t  = h_t + k_t u_t^T                        rank-1 correction
    o_t  = q_t h_t * scale

Around it the four models are one layer (`GatedDeltaNet`, the feature) with six axes; `VARIANTS` names the four:

    decay          "head"     GDN: one log-decay per value head, -exp(A_log) * softplus(a + dt_bias)
                   "channel"  KDA: one per key channel; A_log per head, dt_bias per channel
    lower_bound    None       the softplus form above (GDN; KDA with safe_gate off)
                   -5.0       KDA's safe gate: lower_bound * sigmoid(exp(A_log) * (f + dt_bias)), in (lower_bound, 0)
    lowrank_decay  the decay projection as a rank-Dk pair f_a, f_b (GLM-5.3, Kimi Linear, Kimi K3) or one matrix
                   (GDN's in_proj_a; Ling-3.0's no_kda_lora)
    lowrank_gate   the output-gate projection as a pair g_a, g_b (GLM-5.3, Kimi Linear) or one matrix (GDN's
                   in_proj_z; Kimi K3's use_full_rank_gate; Ling-3.0)
    gate_activation  "silu" | "sigmoid" on the output gate inside the gated RMS norm (GDN: config output_gate_type;
                   every KDA: sigmoid)
    norm_cast      "rounded"  the norm rounds to the activation dtype before the weight (transformers qwen4_exp)
                   "strict"   weight and gate in fp32, one rounding at the end (glm5_next; fla's FusedRMSNormGated)

What is NOT an axis: q and k l2-normalised, scale Dk^-0.5, beta = sigmoid(b(x)), a depthwise causal conv (silu) over
q|k|v, key heads repeated to the value heads -- all four do these. Nor is the weight layout: fused or separate q/k/v
projections and convs are the same arithmetic on concatenated weights, so `named(scheme, source)` maps the family's
canonical names onto each checkpoint's names and concatenates. The conv and recurrent states are the same two slots for
every variant (`cache_specs`: linear_conv, linear_state).

This file is the ORACLE for the family (CHARTER D4/D14): slow, exact, and the
thing the Triton lanes are judged against. Its own judges are the HF torch
implementations: the GDN variant is held to transformers qwen4_exp
(tests/test_engine_composition.py, probes/linear_attention_check.py), the KDA
variant to transformers glm5_next (tests/test_engine_linear_family.py). The
convention that matched, found by grid rather than assumed (2026-09-11):

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


def output_norm(core: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float,
                activation: str = "sigmoid", cast: str = "strict") -> torch.Tensor:
    """weight * rmsnorm(core) * act(gate) over the last dim -- the family's gated output norm, rounding as `cast` says
    (the two conventions in the module docstring)."""
    if cast == "rounded":
        from engine.modules.norm import rmsnorm_gated
        return rmsnorm_gated(core, gate, weight, eps, activation)
    if cast != "strict":
        raise ValueError(f"norm_cast is 'strict' or 'rounded', not {cast!r}")
    if activation not in ("silu", "swish", "sigmoid"):
        raise ValueError(f"the output gate is silu or sigmoid, not {activation!r}")
    x = core.float()
    g = gate.float()
    act = torch.sigmoid(g) if activation == "sigmoid" else torch.nn.functional.silu(g)
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float() * act).to(core.dtype)


def kda_output_norm(core: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor,
                    eps: float = 1e-6) -> torch.Tensor:
    """Per-head RMS norm, weight and sigmoid gate; round to BF16 only at output (GLM-5.3's o_norm)."""
    return output_norm(core, gate, weight, eps, "sigmoid", "strict")


def log_decay(raw: torch.Tensor, A_log: torch.Tensor, dt_bias: "torch.Tensor | None",
              lower_bound: "float | None" = None, per_channel: bool = False) -> torch.Tensor:
    """The family's log-decay, fp32: raw [..., H] (per head) or [..., H, Dk] (per channel), A_log [H] per head in
    both, dt_bias [H] or [H*Dk] to match.

        lower_bound None:  -exp(A_log) * softplus(raw + dt_bias)              (GDN; transformers: "if the model is
                                                                              loaded in fp16, without the .float()
                                                                              here, A might be -inf")
        lower_bound b:     b * sigmoid(exp(A_log) * (raw + dt_bias))          in (b, 0): KDA's safe gate
    """
    g = raw.float()
    if dt_bias is not None:
        g = g + dt_bias.float().reshape(raw.shape[-2:] if per_channel else raw.shape[-1:])
    a = A_log.float().exp()
    if per_channel:
        a = a.view(-1, 1)
    if lower_bound is None:
        return -a * torch.nn.functional.softplus(g)
    return lower_bound * torch.sigmoid(a * g)


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
    return log_decay(raw_g, A_log, g_bias, lower_bound if safe_gate else None, per_channel=True)


def gdn_decay(a: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """GDN's per-head log-decay: -exp(A_log) * softplus(a + dt_bias), in fp32 (transformers qwen4_exp
    Qwen4ExpTextGatedDeltaNet)."""
    return log_decay(a, A_log, dt_bias, None, per_channel=False)


def gated_delta_rule(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                     g: torch.Tensor, beta: torch.Tensor, initial_state=None,
                     scale: "float | None" = None, qk_l2norm: bool = True,
                     decay_per_channel: bool = False, all_states: bool = False):
    """[B, T, H, Dk] q/k, [B, T, H, Dv] v, [B, T, H] g (log decay) and beta.

    Returns (o [B, T, H, Dv], final state [B, H, Dk, Dv]) -- and with `all_states` also the state after every
    token [B, T, H, Dk, Dv] (a verify step keeps them). Pure recurrence, fp32 inside, the caller's dtype outside.
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
    kept = []
    for i in range(t):
        decay = torch.exp(g[:, i])                              # [B, H] or [B, H, Dk] per channel
        state = state * (decay.unsqueeze(-1) if decay_per_channel else decay[..., None, None])
        k_i, v_i, q_i = k[:, i], v[:, i], q[:, i]               # [B, H, D]
        pred = torch.einsum("bhk,bhkv->bhv", k_i, state)         # k^T h
        u = (v_i - pred) * beta[:, i].unsqueeze(-1)
        state = state + torch.einsum("bhk,bhv->bhkv", k_i, u)
        out[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, state) * scale
        if all_states:
            kept.append(state)
    if all_states:
        return out.to(query.dtype), state, torch.stack(kept, dim=1)
    return out.to(query.dtype), state


# The four models as settings of the axes (gate_activation comes with the model's config where it is one).
_KDA = dict(decay="channel", lower_bound=-5.0, lowrank_decay=True, lowrank_gate=True, gate_activation="sigmoid",
            norm_cast="strict")
VARIANTS = {
    "gdn": dict(decay="head", lower_bound=None, lowrank_decay=False, lowrank_gate=False, norm_cast="rounded"),   # Qwen3.8
    "kda": dict(_KDA),                                                                # GLM-5.3, Kimi Linear
    "kda_full_gate": dict(_KDA, lowrank_gate=False),                                  # Kimi K3 (use_full_rank_gate)
    "kda_full": dict(_KDA, lowrank_decay=False, lowrank_gate=False),                  # Ling-3.0 (no_kda_lora)
}

# The family's canonical weight names on each checkpoint's: a tuple is concatenated along dim 0 (separate projections
# or convs are the fused one). The feature asks only for what its axes need (decay or decay_a/decay_b, gate or
# gate_a/gate_b; conv_bias when the checkpoint has one).
SCHEMES = {
    "qwen4_exp": {"qkv": "in_proj_qkv", "conv": "conv1d", "conv_bias": "conv1d.bias", "beta": "in_proj_b",
                  "decay": "in_proj_a", "A_log": "A_log", "dt_bias": "dt_bias", "gate": "in_proj_z", "norm": "norm",
                  "out": "out_proj"},
    "glm5_next": {"qkv": ("q_proj", "k_proj", "v_proj"), "conv": "conv1d", "conv_bias": "conv1d.bias", "beta": "b_proj",
                  "decay_a": "forget_gate.f_a_proj", "decay_b": "forget_gate.f_b_proj", "A_log": "forget_gate.A_log",
                  "dt_bias": "forget_gate.dt_bias", "gate_a": "g_a_proj", "gate_b": "g_b_proj", "norm": "o_norm",
                  "out": "o_proj"},
    "kimi": {"qkv": ("q_proj", "k_proj", "v_proj"), "conv": ("q_conv1d", "k_conv1d", "v_conv1d"),
             "conv_bias": ("q_conv1d.bias", "k_conv1d.bias", "v_conv1d.bias"), "beta": "b_proj", "decay": "f_proj",
             "decay_a": "f_a_proj", "decay_b": "f_b_proj", "A_log": "A_log", "dt_bias": "dt_bias", "gate": "g_proj",
             "gate_a": "g_a_proj", "gate_b": "g_b_proj", "norm": "o_norm", "out": "o_proj"},    # Kimi Linear/K3, Ling-3.0
}


MARK_CHUNK = 64                 # a mark is a chunk index; the served chunk kernels cut the sequence 64 tokens a chunk


def gated_delta_rule_marked(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            g: torch.Tensor, beta: torch.Tensor, initial_state=None, *,
                            scale: "float | None" = None, qk_l2norm: bool = True,
                            decay_per_channel: bool = False, marks=None):
    """`gated_delta_rule`, and the state at each of `marks` (chunk indices, MARK_CHUNK tokens a chunk): what a prefill
    step that saves block boundaries asks of the reference.

    The recurrence is token by token, so the state at a chunk's start is exactly the state after the piece before
    it: the pieces run one after another and each piece's final state is that mark's. A mark at or before the first
    token (no piece yet) is the initial state, or zeros [H, Dk, Dv] fp32 without one.

    Without marks this is `gated_delta_rule`: (o, final state). With them: (o, final state, [len(marks), H, Dk, Dv]).
    Every profile with a delta-rule layer asks the same (KDA's per-channel decay, GDN's per-head one).
    """
    if not marks:
        return gated_delta_rule(query, key, value, g, beta, initial_state, scale=scale, qk_l2norm=qk_l2norm,
                                decay_per_channel=decay_per_channel)
    outs, states, state, lo = [], [], initial_state, 0
    for hi in [c * MARK_CHUNK for c in marks] + [query.shape[1]]:
        if hi > lo:
            o, state = gated_delta_rule(query[:, lo:hi], key[:, lo:hi], value[:, lo:hi], g[:, lo:hi], beta[:, lo:hi],
                                        state, scale=scale, qk_l2norm=qk_l2norm, decay_per_channel=decay_per_channel)
            outs.append(o)
        if len(states) < len(marks):
            states.append(state[0] if state is not None else
                          torch.zeros(value.shape[2], key.shape[-1], value.shape[-1], device=query.device,
                                      dtype=torch.float32))
        lo = hi
    return torch.cat(outs, dim=1), state, torch.stack(states)


def named(scheme: str, source):
    """canonical name -> tensor over `source(checkpoint name)` (a module's matrix by its module name, a bare parameter
    by its own); KeyError for a name the scheme or the checkpoint lacks."""
    table = SCHEMES[scheme]

    def get(name: str) -> torch.Tensor:
        if name not in table:
            raise KeyError(name)
        at = table[name]
        if isinstance(at, tuple):
            return torch.cat([source(n) for n in at], dim=0)
        return source(at)
    return get


class GatedDeltaNet:
    """The family's layer as a token mixer (engine/base/composition.Feature): the six axes of the module docstring
    over one recurrence.

    x -> qkv -> causal conv (kernel `conv`, silu) -> q, k, v heads; beta = sigmoid(beta(x)); the log-decay from
    decay(x) (or decay_b(decay_a(x))) with A_log and dt_bias; key heads repeated to the value heads; the gated delta
    rule (`gated_delta_rule`, q/k l2-normalised, scale Dk^-0.5); a gated RMS norm with the gate from gate(x) (or
    gate_b(gate_a(x))); out. Per sequence it carries the conv's last kernel-1 inputs and the fp32 recurrent state
    [HV, Dk, Dv].

    `weights(layer, name)` answers the canonical names (`named` maps a checkpoint's): qkv, conv, conv_bias (optional),
    beta, decay | decay_a, decay_b, A_log, dt_bias, gate | gate_a, gate_b, norm, out."""

    def __init__(self, *, k_heads: int, v_heads: int, k_dim: int, v_dim: int, conv: int, eps: float, weights,
                 decay: str = "head", lower_bound: "float | None" = None, lowrank_decay: bool = False,
                 lowrank_gate: bool = False, gate_activation: str = "silu", norm_cast: str = "rounded",
                 activation: str = "silu", dtype: str = "bfloat16"):
        if v_heads % k_heads:
            raise ValueError(f"{v_heads} value heads are not a multiple of {k_heads} key heads")
        if decay not in ("head", "channel"):
            raise ValueError(f"decay is 'head' or 'channel', not {decay!r}")
        if lower_bound is not None and lower_bound >= 0:
            raise ValueError(f"the safe gate's lower bound is negative, not {lower_bound}")
        if norm_cast not in ("strict", "rounded"):
            raise ValueError(f"norm_cast is 'strict' or 'rounded', not {norm_cast!r}")
        if gate_activation not in ("silu", "swish", "sigmoid"):
            raise ValueError(f"the output gate is silu or sigmoid, not {gate_activation!r}")
        self.k_heads, self.v_heads, self.k_dim, self.v_dim, self.conv = k_heads, v_heads, k_dim, v_dim, conv
        self.eps, self.weights, self.activation, self.dtype = eps, weights, activation, dtype
        self.decay, self.lower_bound = decay, lower_bound
        self.lowrank_decay, self.lowrank_gate = lowrank_decay, lowrank_gate
        self.gate_activation, self.norm_cast = gate_activation, norm_cast

    @property
    def conv_dim(self) -> int:
        return 2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim

    @staticmethod
    def _proj(xs, w, name: str, lowrank: bool) -> torch.Tensor:
        if lowrank:
            return torch.nn.functional.linear(torch.nn.functional.linear(xs, w(f"{name}_a")), w(f"{name}_b"))
        return torch.nn.functional.linear(xs, w(name))

    @staticmethod
    def _optional(w, name: str):
        try:
            return w(name)
        except KeyError:
            return None

    def __call__(self, layer, x, step, state):
        from engine.base.composition import put_state
        from engine.modules.causal_conv import causal_conv1d, conv_states
        w = lambda name: self.weights(layer, name)
        per_channel = self.decay == "channel"
        out = None
        for s in step.segments:
            xs = x[s.start:s.start + s.length]
            t = xs.shape[0]
            qkv = torch.nn.functional.linear(xs, w("qkv"))
            gate = self._proj(xs, w, "gate", self.lowrank_gate).reshape(t, self.v_heads, self.v_dim)
            beta = torch.nn.functional.linear(xs, w("beta")).sigmoid()                      # [T, HV]
            raw = self._proj(xs, w, "decay", self.lowrank_decay)                              # [T, HV] or [T, HV*Dk]
            if per_channel:
                raw = raw.reshape(t, self.v_heads, self.k_dim)
            g = log_decay(raw, w("A_log"), w("dt_bias"), self.lower_bound, per_channel)
            conv_w = w("conv")
            held_conv = state.get(layer, "linear_conv", s.seq)
            conv_each = conv_states(qkv, held_conv, self.conv - 1) if s.verify else None
            qkv, conv_state = causal_conv1d(qkv, conv_w.reshape(conv_w.shape[0], -1), self._optional(w, "conv_bias"),
                                            held_conv, self.activation)
            q, k, v = torch.split(qkv, [self.k_heads * self.k_dim, self.k_heads * self.k_dim, self.v_heads * self.v_dim], -1)
            q = q.reshape(1, t, self.k_heads, self.k_dim)
            k = k.reshape(1, t, self.k_heads, self.k_dim)
            v = v.reshape(1, t, self.v_heads, self.v_dim)
            if self.v_heads != self.k_heads:
                q = q.repeat_interleave(self.v_heads // self.k_heads, dim=2)
                k = k.repeat_interleave(self.v_heads // self.k_heads, dim=2)
            ran = gated_delta_rule(q, k, v, g[None], beta[None], state.get(layer, "linear_state", s.seq),
                                   scale=self.k_dim ** -0.5, qk_l2norm=True, decay_per_channel=per_channel,
                                   all_states=s.verify)
            core, recurrent = ran[0], ran[1]
            state_each = ran[2].transpose(0, 1) if s.verify else None          # [T, 1, HV, Dk, Dv]
            core = output_norm(core.reshape(-1, self.v_dim), gate.reshape(-1, self.v_dim), w("norm"), self.eps,
                               self.gate_activation, self.norm_cast).reshape(t, -1)
            ys = torch.nn.functional.linear(core, w("out"))
            if out is None:
                out = xs.new_empty(x.shape[0], ys.shape[-1])
            out[s.start:s.start + s.length] = ys
            put_state(state, layer, "linear_conv", s, conv_state, conv_each)
            put_state(state, layer, "linear_state", s, recurrent, state_each)
        return out

    def cache_specs(self, layers):
        from engine.base.cache_spec import SlotSpec, _ITEMSIZE
        return [SlotSpec("linear conv state", len(layers), self.conv_dim * (self.conv - 1) * _ITEMSIZE[self.dtype],
                         f"[conv_dim, kernel-1] {self.dtype}: the conv's last inputs (modules/causal_conv)",
                         key="linear_conv", dtype=self.dtype, shape=(self.conv_dim, self.conv - 1)),
                SlotSpec("linear recurrent state", len(layers), self.v_heads * self.k_dim * self.v_dim * 4,
                         "[HV, Dk, Dv] fp32 (mamba_ssm_dtype float32)",
                         key="linear_state", dtype="float32", shape=(1, self.v_heads, self.k_dim, self.v_dim))]
