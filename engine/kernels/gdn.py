"""GatedDeltaNet's per-token arithmetic around the delta rule, in one launch each (Qwen3.8's linear attention).

The delta rule itself runs on engine/kernels/kda: ring.recurrent_gdn_ring(_rows) for decode and verify, which computes
`gates`' decay and beta from the raw projection inside its own launch (HEAD_GATE in kda/fused_recurrent.py, this file's
arithmetic), and chunk_decay.chunk_kda_with_decay for prefill, with the decay computed here -- the wizard's glue for a
per-head decay. What those kernels do not compute is the model's own arithmetic before and after them:

    gates        decay = -exp(A_log) * softplus(a + dt_bias) per value head, in fp32
                 (engine/modules/linear_attention.gdn_decay), and beta = sigmoid(b) when the consumer wants it applied
                 (the chunk kernel; the ring kernels apply their own sigmoid to the raw logits)
    gated_norm   the output norm with GDN's rounding: RMS over each head in fp32 rounded to the activations' dtype,
                 times the plain weight (rounds), times sigmoid(z) in fp32, rounded once
                 (engine/modules/norm.rmsnorm_gated, "rounded"; GLM's KDA output norm rounds only at the end,
                 engine/kernels/kda/output.py, so it cannot serve this one)

softplus follows torch's: log1p(exp(x)), and x above its threshold 20 passes through. `qualify` holds both to the
modules on the device.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gates(A, B, A_LOG, DT, DECAY, BETA, sA, sB, sD, sE, HV: tl.constexpr, BH: tl.constexpr,
           SIGMOID_BETA: tl.constexpr):
    r = tl.program_id(0)
    h = tl.arange(0, BH)
    m = h < HV
    g = tl.load(A + r * sA + h, mask=m, other=0.0).to(tl.float32) + tl.load(DT + h, mask=m, other=0.0).to(tl.float32)
    # log1p, not log(1 + .): a strongly negative gate's decay is exp(g), which 1 + exp(g) rounds away in fp32
    softplus = tl.where(g > 20.0, g, libdevice.log1p(tl.exp(tl.minimum(g, 20.0))))
    decay = -tl.exp(tl.load(A_LOG + h, mask=m, other=0.0).to(tl.float32)) * softplus
    tl.store(DECAY + r * sD + h, decay, mask=m)
    b = tl.load(B + r * sB + h, mask=m, other=0.0)
    if SIGMOID_BETA:
        tl.store(BETA + r * sE + h, tl.sigmoid(b.to(tl.float32)).to(b.dtype), mask=m)
    else:
        tl.store(BETA + r * sE + h, b, mask=m)


@triton.jit
def _gated_norm(X, Z, W, OUT, sX, sZr, sZh, sO, EPS, HV: tl.constexpr, D: tl.constexpr, BD: tl.constexpr):
    p = tl.program_id(0)                                   # one program a (row, head): X [rows*heads, D] flat
    d = tl.arange(0, BD)
    m = d < D
    xr = tl.load(X + p * sX + d, mask=m, other=0.0)
    x = xr.to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    normed = (x * scale).to(xr.dtype).to(tl.float32)               # the norm rounds to the activations' dtype
    weighted = (normed * tl.load(W + d, mask=m, other=0.0).to(tl.float32)).to(xr.dtype).to(tl.float32)
    # z is a column slice of the in_proj row: read through its row and head strides rather than a packed copy
    gate = tl.sigmoid(tl.load(Z + (p // HV) * sZr + (p % HV) * sZh + d, mask=m, other=0.0).to(tl.float32))
    tl.store(OUT + p * sO + d, (weighted * gate).to(OUT.dtype.element_ty), mask=m)


def gates(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, *,
          sigmoid_beta: bool) -> "tuple[torch.Tensor, torch.Tensor]":
    """(decay fp32 [rows, HV], beta [rows, HV]) from the projections' a and b rows [rows, HV]; beta is sigmoid(b) in
    b's dtype when `sigmoid_beta`, else the raw logits (a contiguous copy the ring kernel reads)."""
    if (a.ndim != 2 or b.shape != a.shape or A_log.shape != (a.shape[1],) or dt_bias.shape != A_log.shape
            or A_log.dtype != torch.float32 or dt_bias.dtype != torch.float32):
        raise ValueError("GDN gates take a and b [rows, HV] and fp32 A_log, dt_bias [HV]")
    if not a.is_cuda:
        from engine.modules.linear_attention import gdn_decay
        return gdn_decay(a, A_log, dt_bias), (torch.sigmoid(b) if sigmoid_beta else b.clone())
    rows, hv = a.shape
    decay = torch.empty(rows, hv, device=a.device, dtype=torch.float32)
    beta = torch.empty(rows, hv, device=a.device, dtype=b.dtype)
    if rows:
        _gates[(rows,)](a, b, A_log, dt_bias, decay, beta, a.stride(0), b.stride(0), decay.stride(0), beta.stride(0),
                        HV=hv, BH=triton.next_power_of_2(hv), SIGMOID_BETA=sigmoid_beta, num_warps=1)
    return decay, beta


def gated_norm(core: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """rmsnorm_gated(core, z, weight, eps, "sigmoid") per head: core and z [rows, HV, D] -> [rows, HV * D]."""
    if core.ndim != 3 or z.shape != core.shape or weight.shape != (core.shape[-1],):
        raise ValueError("the GDN output norm takes core and z [rows, HV, D] and a weight [D]")
    rows, hv, dim = core.shape
    if not core.is_cuda:
        from engine.modules.norm import rmsnorm_gated
        return rmsnorm_gated(core, z, weight, eps, "sigmoid").reshape(rows, hv * dim)
    x = core.reshape(rows * hv, dim)
    if x.stride(1) != 1 or z.stride(2) != 1:
        raise ValueError("the GDN output norm reads packed heads")
    out = torch.empty(rows * hv, dim, device=core.device, dtype=core.dtype)
    if rows:
        # z is not reshaped: at more than one row its [rows, HV, D] view over the in_proj split has no flat view, and
        # the reshape would copy it on every GDN layer
        _gated_norm[(rows * hv,)](x, z, weight, out, x.stride(0), z.stride(0), z.stride(1), out.stride(0), eps,
                                  HV=hv, D=dim, BD=triton.next_power_of_2(dim), num_warps=2)
    return out.view(rows, hv * dim)


def qualify(device, *, heads: int, dim: int, eps: float, dtype=torch.bfloat16, rows=(1, 7, 300),
            band_max: float = 5e-2, band_rms: float = 2e-2, seed: int = 0) -> dict:
    """Hold `gates` and `gated_norm` to engine/modules (gdn_decay, rmsnorm_gated) on `device` with random inputs, within
    a few BF16 steps (engine/kernels/gated_residual.drift); returns the worst (max, rms) per output."""
    from engine.kernels.gated_residual import drift
    from engine.modules.linear_attention import gdn_decay
    from engine.modules.norm import rmsnorm_gated
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=dt)

    worst = {k: (0.0, 0.0) for k in ("decay", "beta", "raw_beta", "norm")}

    def note(key, ours, ref):
        m, r = drift(ours, ref)
        worst[key] = (max(worst[key][0], m), max(worst[key][1], r))

    A_log, dt_bias = rand(heads, dt=torch.float32), rand(heads, dt=torch.float32)
    weight = rand(dim, scale=0.1)
    for n in rows:
        a, b = rand(n, heads, scale=4.0), rand(n, heads, scale=4.0)
        decay, beta = gates(a, b, A_log, dt_bias, sigmoid_beta=True)
        note("decay", decay, gdn_decay(a, A_log, dt_bias))
        note("beta", beta, torch.sigmoid(b))
        _, raw = gates(a, b, A_log, dt_bias, sigmoid_beta=False)
        note("raw_beta", raw, b)
        core, z = rand(n, heads, dim), rand(n, heads, dim)
        note("norm", gated_norm(core, z, weight, eps), rmsnorm_gated(core, z, weight, eps, "sigmoid").reshape(n, heads * dim))
    bad = {k: v for k, v in worst.items() if v[0] > band_max or v[1] > band_rms}
    if bad:
        raise RuntimeError(f"GDN lane arithmetic drifts from engine/modules beyond max {band_max:g} / rms {band_rms:g}: {bad}")
    return worst


__all__ = ["gates", "gated_norm", "qualify"]
