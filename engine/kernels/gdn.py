"""GatedDeltaNet's per-token arithmetic around the delta rule, in one launch each (Qwen3.8's linear attention).

The delta rule itself runs on engine/kernels/kda: ring.recurrent_decay_ring(_rows) for decode and verify and
chunk_decay.chunk_kda_with_decay for prefill, both with the decay computed outside the kernel -- the wizard's glue for a
per-head decay. What those kernels do not compute is the model's own arithmetic before and after them:

    gates        decay = -exp(A_log) * softplus(a + dt_bias) per value head, in fp32
                 (engine/modules/linear_attention.gdn_decay), and beta = sigmoid(b) when the consumer wants it applied
                 (the chunk kernel; the ring kernel applies its own sigmoid to the raw logits)
    gated_norm   the output norm with GDN's rounding: RMS over each head in fp32 rounded to the activations' dtype,
                 times the plain weight (rounds), times sigmoid(z) in fp32, rounded once
                 (engine/modules/norm.rmsnorm_gated, "rounded"; GLM's KDA output norm rounds only at the end,
                 engine/kernels/kda/output.py, so it cannot serve this one)

softplus follows torch's: x above its threshold 20 passes through. `qualify` holds both to the modules on the device.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gates(A, B, A_LOG, DT, DECAY, BETA, sA, sB, sD, sE, HV: tl.constexpr, BH: tl.constexpr,
           SIGMOID_BETA: tl.constexpr):
    r = tl.program_id(0)
    h = tl.arange(0, BH)
    m = h < HV
    g = tl.load(A + r * sA + h, mask=m, other=0.0).to(tl.float32) + tl.load(DT + h, mask=m, other=0.0).to(tl.float32)
    softplus = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(tl.minimum(g, 20.0))))
    decay = -tl.exp(tl.load(A_LOG + h, mask=m, other=0.0).to(tl.float32)) * softplus
    tl.store(DECAY + r * sD + h, decay, mask=m)
    b = tl.load(B + r * sB + h, mask=m, other=0.0)
    if SIGMOID_BETA:
        tl.store(BETA + r * sE + h, tl.sigmoid(b.to(tl.float32)).to(b.dtype), mask=m)
    else:
        tl.store(BETA + r * sE + h, b, mask=m)


@triton.jit
def _gated_norm(X, Z, W, OUT, sX, sZ, sO, EPS, D: tl.constexpr, BD: tl.constexpr):
    p = tl.program_id(0)                                   # one program a (row, head): X [rows*heads, D] flat
    d = tl.arange(0, BD)
    m = d < D
    xr = tl.load(X + p * sX + d, mask=m, other=0.0)
    x = xr.to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / D + EPS)
    normed = (x * scale).to(xr.dtype).to(tl.float32)               # the norm rounds to the activations' dtype
    weighted = (normed * tl.load(W + d, mask=m, other=0.0).to(tl.float32)).to(xr.dtype).to(tl.float32)
    gate = tl.sigmoid(tl.load(Z + p * sZ + d, mask=m, other=0.0).to(tl.float32))
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
    g = z.reshape(rows * hv, dim)
    if x.stride(1) != 1 or g.stride(1) != 1:
        raise ValueError("the GDN output norm reads packed heads")
    out = torch.empty(rows * hv, dim, device=core.device, dtype=core.dtype)
    if rows:
        _gated_norm[(rows * hv,)](x, g, weight, out, x.stride(0), g.stride(0), out.stride(0), eps,
                                  D=dim, BD=triton.next_power_of_2(dim), num_warps=2)
    return out.view(rows, hv * dim)


def qualify(device, *, heads: int, dim: int, eps: float, dtype=torch.bfloat16, rows=(1, 7, 300), band: float = 2e-3,
            seed: int = 0) -> dict:
    """Hold `gates` and `gated_norm` to engine/modules (gdn_decay, rmsnorm_gated) on `device` with random inputs."""
    from engine.modules.linear_attention import gdn_decay
    from engine.modules.norm import rmsnorm_gated
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape, scale=1.0, dt=dtype):
        return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=dt)

    def rel(x, y):
        return float(((x.float() - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-12)).item())

    worst = {"decay": 0.0, "beta": 0.0, "norm": 0.0}
    A_log, dt_bias = rand(heads, dt=torch.float32), rand(heads, dt=torch.float32)
    weight = rand(dim, scale=0.1)
    for n in rows:
        a, b = rand(n, heads, scale=4.0), rand(n, heads, scale=4.0)
        decay, beta = gates(a, b, A_log, dt_bias, sigmoid_beta=True)
        worst["decay"] = max(worst["decay"], rel(decay, gdn_decay(a, A_log, dt_bias)))
        worst["beta"] = max(worst["beta"], rel(beta, torch.sigmoid(b)))
        core, z = rand(n, heads, dim), rand(n, heads, dim)
        ref = rmsnorm_gated(core, z, weight, eps, "sigmoid").reshape(n, heads * dim)
        worst["norm"] = max(worst["norm"], rel(gated_norm(core, z, weight, eps), ref))
    bad = {k: v for k, v in worst.items() if v > band}
    if bad:
        raise RuntimeError(f"GDN lane arithmetic drifts from engine/modules beyond {band:g}: {bad}")
    return worst


__all__ = ["gates", "gated_norm", "qualify"]
