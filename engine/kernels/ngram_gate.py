"""The n-gram memory's gate and its conv norm in one launch (kernels): engine/modules/ngram_embedding's "separate" form --
Qwen3.8's PLE injection -- over a prefill step's rows.

NGramInjection._gated and ._norm are about fifteen torch launches over [N, hc*H] rows (10,240 wide on Qwen3.8), each
reading and writing the rows again: two unit-offset RMS norms (the streams as the query, the key), their product, its sum
over each stream, the division by sqrt(H), the signed square root, the sigmoid, the gate times the shared value, and the
conv's unit-offset norm of the result. On a GB10 a 4,096-token prefill chunk spent 48.8 ms there (measurements/
qwen38_prefill_census_20260919: PLE 54.44 ms, 5.4% of the chunk) for about 350 MB of necessary traffic -- 1.3 ms at the
memory's rate.

One program is one stream of one row: it reads the row's stream, the key's stream and the shared value once, and writes the
gated stream and its conv norm.

`conv_add` is the rest of the injection: the dilated causal conv over the normed rows (engine/modules/causal_conv's torch
form: four taps accumulated in fp32 in tap order, silu, rounded to BF16) and the gated rows added back -- another dozen torch
launches (a cat, the taps' products and sums in fp32 over [hc*H, T], the silu, the casts, the add) as one. A program is one
row and a block of channels; a tap before the step reads the history the caller gathered from its ring. The products and
sums may contract into fused multiply-adds, so an element can differ from torch's in the fp32 last bit before it rounds. Every elementwise step rounds to BF16 where the torch form's does (a BF16 tensor op
computes in fp32 and rounds once); the three reductions are the kernel's own -- the norms' sums of squares and the
query-key dot -- and can differ from torch's in the last bit, which rounding then carries or drops. `qualify` holds it to
the torch form within a few BF16 steps (D3).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _unit_offset_norm(x, w, EPS, HID: tl.constexpr, OUT_DTYPE: tl.constexpr):
    # rmsnorm_unit_offset over one stream: fp32 throughout, rounded once after the (1 + w) weight
    scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
    return ((x * scale) * (1.0 + w)).to(OUT_DTYPE).to(tl.float32)


@triton.jit
def _gate(H, KEY, VALUE, QW, KW, CW, GATED, NORMED, sH, sK, sV, sG, sN, EPS, INV_SQRT_H, CLAMP,
          HID: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    dt = GATED.dtype.element_ty
    h = tl.load(H + r * sH + off, mask=m, other=0.0).to(tl.float32)
    k = tl.load(KEY + r * sK + off, mask=m, other=0.0).to(tl.float32)
    q = _unit_offset_norm(h, tl.load(QW + off, mask=m, other=0.0).to(tl.float32), EPS, HID, dt)
    kn = _unit_offset_norm(k, tl.load(KW + off, mask=m, other=0.0).to(tl.float32), EPS, HID, dt)
    prod = (kn * q).to(dt).to(tl.float32)                          # key * query rounds, then the sum does
    dot = tl.sum(tl.where(m, prod, 0.0)).to(dt).to(tl.float32)
    x = (dot * INV_SQRT_H).to(dt).to(tl.float32)                   # / sqrt(H)
    root = tl.sqrt(tl.maximum(tl.abs(x), CLAMP).to(dt).to(tl.float32)).to(dt).to(tl.float32)
    sign = tl.where(x > 0, 1.0, tl.where(x < 0, -1.0, 0.0))
    g = tl.sigmoid(root * sign).to(dt).to(tl.float32)
    v = tl.load(VALUE + r * sV + d, mask=m, other=0.0).to(tl.float32)
    gated = (g * v).to(dt)
    tl.store(GATED + r * sG + off, gated, mask=m)
    normed = _unit_offset_norm(gated.to(tl.float32), tl.load(CW + off, mask=m, other=0.0).to(tl.float32), EPS, HID, dt)
    tl.store(NORMED + r * sN + off, normed.to(dt), mask=m)


@triton.jit
def _conv_add(X, G, W, HELD, OUT, C, sX, sG, sW, sH, sO, K: tl.constexpr, DIL: tl.constexpr, SPAN: tl.constexpr,
              BC: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = c < C
    acc = tl.zeros((BC,), dtype=tl.float32)
    for i in tl.static_range(K):
        j = t + i * DIL                                              # the tap's index into [history, this step's rows]
        before = j < SPAN
        xh = tl.load(HELD + c * sH + j, mask=m & before, other=0.0)
        xs = tl.load(X + (j - SPAN) * sX + c, mask=m & (j >= SPAN), other=0.0)
        x = tl.where(before, xh.to(tl.float32), xs.to(tl.float32))
        w = tl.load(W + c * sW + i, mask=m, other=0.0).to(tl.float32)
        acc += w * x
    local = (acc / (1.0 + tl.exp(-acc))).to(OUT.dtype.element_ty)   # silu in fp32, then the rounding
    g = tl.load(G + t * sG + c, mask=m, other=0.0)
    tl.store(OUT + t * sO + c, (g.to(tl.float32) + local.to(tl.float32)).to(OUT.dtype.element_ty), mask=m)


def conv_add(normed: torch.Tensor, gated: torch.Tensor, weight: torch.Tensor, held: torch.Tensor,
             dilation: int, *, out: "torch.Tensor | None" = None) -> torch.Tensor:
    """gated + causal_conv1d(normed, weight, None, held, "silu", dilation)[0]: normed, gated [T, C] BF16, weight [C, K],
    held [C, (K-1)*dilation] -- the inputs before the step (zeros before the sequence) -> [T, C] BF16, into `out` (rows
    packed along C) when given."""
    t, c = normed.shape
    k = weight.shape[1] if weight.ndim == 2 else 0
    span = (k - 1) * dilation
    if (gated.shape != (t, c) or weight.shape != (c, k) or k < 1 or held.shape != (c, span) or dilation < 1
            or normed.stride(1) != 1 or gated.stride(1) != 1 or weight.stride(1) != 1 or held.stride(1) != 1):
        raise ValueError(f"conv_add takes normed and gated [T, C], weight [C, K] and held [C, (K-1)*dilation], packed "
                         f"along their last dimension; got {tuple(normed.shape)} {tuple(gated.shape)} "
                         f"{tuple(weight.shape)} {tuple(held.shape)}")
    if out is None:
        out = torch.empty_like(gated)
    elif out.shape != (t, c) or out.dtype != gated.dtype or out.stride(1) != 1:
        raise ValueError("conv_add writes [T, C] rows of the gated rows' dtype, packed along C")
    if t:
        block = 1024
        _conv_add[(t, triton.cdiv(c, block))](normed, gated, weight, held, out, c, normed.stride(0), gated.stride(0),
                                              weight.stride(0), held.stride(0), out.stride(0), K=k, DIL=dilation,
                                              SPAN=span, BC=block, num_warps=4)
    return out


def gate(h: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_norm: torch.Tensor, k_norm: torch.Tensor,
         conv_norm: torch.Tensor, eps: float, hc: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """(gated, normed) [N, hc*H] BF16: NGramInjection._gated(h, ...).flatten(-2) and ._norm(gated, conv_norm) for the
    "separate" unit-offset form with the sign square root. h, key [N, hc*H] and value [N, H] packed along their channels
    (key and value may be column slices of the kv projection); the norm weights [hc*H]."""
    rows, width = h.shape
    if hc < 1 or width % hc:
        raise ValueError(f"the streams are [N, hc*H]; got {tuple(h.shape)} for hc {hc}")
    hid = width // hc
    if (key.shape != (rows, width) or value.shape != (rows, hid)
            or any(t.shape != (width,) for t in (q_norm, k_norm, conv_norm))):
        raise ValueError(f"the gate takes key [N, {width}], value [N, {hid}] and norm weights [{width}]")
    if any(t.stride(-1) != 1 for t in (h, key, value, q_norm, k_norm, conv_norm)):
        raise ValueError("the gate reads rows packed along their channels")
    if not (h.dtype == key.dtype == value.dtype == torch.bfloat16):
        raise ValueError("the gate's rows are BF16")
    gated = torch.empty(rows, width, device=h.device, dtype=h.dtype)
    normed = torch.empty_like(gated)
    if rows:
        clamp = float(torch.tensor(1e-6, dtype=h.dtype))              # clamp_min(1e-6) on a BF16 tensor lands here
        _gate[(rows, hc)](h, key, value, q_norm, k_norm, conv_norm, gated, normed, h.stride(0), key.stride(0),
                          value.stride(0), gated.stride(0), normed.stride(0), eps, 1.0 / math.sqrt(hid), clamp,
                          HID=hid, BD=triton.next_power_of_2(hid), num_warps=8 if hid > 1024 else 4)
    return gated, normed


def reference(feature, h, embeddings, w):
    """The torch form this launch stands for: (gated, normed) exactly as NGramInjection computes them."""
    gated = feature._gated(h, embeddings, w).flatten(-2)
    return gated, feature._norm(gated, w("conv_norm"))


def qualify(device, *, hc: int, hidden: int, eps: float, rows=(3, 64)) -> dict:
    """D3 before a boot serves: the launch against the torch form's arithmetic (written here in torch ops, cast for
    cast) -> {rows: (largest error of gated, of normed) over the largest magnitude}; raises past 2^-6."""
    gen = torch.Generator(device="cpu").manual_seed(0)
    width = hc * hidden
    out = {}
    for n in rows:
        h = torch.randn(n, width, generator=gen).to(torch.bfloat16).to(device)
        key = torch.randn(n, width, generator=gen).to(torch.bfloat16).to(device)
        value = torch.randn(n, hidden, generator=gen).to(torch.bfloat16).to(device)
        qw, kw, cw = ((torch.randn(width, generator=gen) * 0.1).to(torch.bfloat16).to(device) for _ in range(3))
        got = gate(h, key, value, qw, kw, cw, eps, hc)
        want = torch_form(h, key, value, qw, kw, cw, eps, hc)
        errs = []
        for a, b in zip(got, want):
            err = float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))
            if not err <= 2.0 ** -6:
                raise RuntimeError(f"ngram_gate {n} rows: error {err:.2e} of the largest magnitude")
            errs.append(round(err, 6))
        out[f"{n}_rows"] = tuple(errs)
    return out


def torch_form(h, key, value, q_norm, k_norm, conv_norm, eps, hc):
    """NGramInjection's separate form in torch ops, cast for cast, from the kv projection's key and value."""
    from engine.modules.norm import rmsnorm_unit_offset
    hid = h.shape[1] // hc
    k = rmsnorm_unit_offset(key, k_norm, eps, group=hid).unflatten(-1, (hc, hid))
    q = rmsnorm_unit_offset(h, q_norm, eps, group=hid).unflatten(-1, (hc, hid))
    dot = (k * q).sum(dim=-1, keepdim=True) / math.sqrt(hid)
    root = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
    gated = (torch.sigmoid(root) * value.unsqueeze(-2)).flatten(-2)
    return gated, rmsnorm_unit_offset(gated, conv_norm, eps, group=hid)


__all__ = ["conv_add", "gate", "qualify", "reference", "torch_form"]
