"""The gated residual hyper-connection in five launches a site (Qwen3.8's residual form).

engine/modules/hyper_connection.gated_residual is the oracle: the hc streams laid end to end [N, hc*H] are RMS
normalised one by one (unit-offset weight), a low-rank mixer weights every channel of every stream --
sigmoid(up(silu(down(normed) / hc))) -- and the sublayer reads the streams' weighted mean [N, H]; the injection
2*sigmoid(inject(normed) / hc) [N, hc] is how strongly the sublayer's output is added back into each stream.

Composed from torch ops that is about a dozen launches a site, and the model has 97 sites (two a layer and the
closing mixer), over rows 10,240 wide. Here a site is:

    leave_norm   the previous sublayer's output added into the streams AND the streams normalised    one launch
    GEMM         down and inject in one BF16 matmul (the injection's 4 rows ride the mixer's 320)       one launch
    gates        silu(x / hc) on the mixer rows, 2*sigmoid(z / hc) on the injection rows              one launch
    GEMM         up                                                                                    one launch
    mix_mean     sigmoid(up) * normed, averaged over the streams                                       one launch

`norm_streams` opens the first site (no output to add yet) and follows an injection feature that reads the
streams between two sites (Qwen3.8's PLE before layer 1, the config's one-indexed 2); `leave` adds an output without the norm for the same
case. The closing mixer is a site without an injection.

Rounding is the torch form's wherever the form rounds: the division by hc, silu, sigmoid, the gate product and
the residual product each round to the activations' dtype before the next operation reads them, and the norm
rounds once after its unit-offset weight, as rmsnorm_unit_offset does. Two reductions are the kernel's own and
can differ from torch's in the last bit: the norm's sum of squares and the mean over the streams (torch reduces
BF16 partial sums; this accumulates in fp32 and rounds once). The GEMMs are torch's matmul; the injection sharing
the mixer's matmul changes the matrix cuBLAS tiles, not the arithmetic of a row.

`qualify` holds the lane to the oracle on the device before it serves (D3): a boot calls it once and dies on a
mismatch rather than serving a residual stream that drifts. Its bounds are a few BF16 steps (a step is 2^-7 of a value)
because rounding order moves elements by a step; a wrong formula moves them by tenths.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_streams(X, W, OUT, sX, sO, EPS, HID: tl.constexpr, BD: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    x = tl.load(X + r * sX + off, mask=m, other=0.0).to(tl.float32)
    scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
    w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r * sO + off, ((x * scale) * (1.0 + w)).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _leave_norm(H, OUT, INJ, W, NORMED, sH, sO, sI, sN, EPS, HID: tl.constexpr, BD: tl.constexpr,
                NORM: tl.constexpr):
    r = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < HID
    off = s * HID + d
    h = tl.load(H + r * sH + off, mask=m, other=0.0)
    o = tl.load(OUT + r * sO + d, mask=m, other=0.0).to(tl.float32)
    g = tl.load(INJ + r * sI + s).to(tl.float32)
    delta = (o * g).to(h.dtype)                                   # the product rounds, then the sum does
    new = (h.to(tl.float32) + delta.to(tl.float32)).to(h.dtype)
    tl.store(H + r * sH + off, new, mask=m)
    if NORM:
        x = new.to(tl.float32)
        scale = tl.rsqrt(tl.sum(x * x) / HID + EPS)
        w = tl.load(W + off, mask=m, other=0.0).to(tl.float32)
        tl.store(NORMED + r * sN + off, ((x * scale) * (1.0 + w)).to(NORMED.dtype.element_ty), mask=m)


@triton.jit
def _gates(DI, MIX, INJ, sD, sM, sI, HC_F, R: tl.constexpr, BR: tl.constexpr, HC: tl.constexpr,
           BH: tl.constexpr, WITH_INJECT: tl.constexpr):
    r = tl.program_id(0)
    i = tl.arange(0, BR)
    m = i < R
    x = tl.load(DI + r * sD + i, mask=m, other=0.0)
    q = (x.to(tl.float32) / HC_F).to(x.dtype).to(tl.float32)     # the division rounds as the torch form's does
    tl.store(MIX + r * sM + i, (q * tl.sigmoid(q)).to(x.dtype), mask=m)
    if WITH_INJECT:
        j = tl.arange(0, BH)
        mj = j < HC
        z = tl.load(DI + r * sD + R + j, mask=mj, other=0.0)
        zq = (z.to(tl.float32) / HC_F).to(z.dtype).to(tl.float32)
        g = tl.sigmoid(zq).to(z.dtype).to(tl.float32)               # sigmoid rounds, then the doubling does
        tl.store(INJ + r * sI + j, (2.0 * g).to(z.dtype), mask=mj)


@triton.jit
def _mix_mean(UP, NORMED, OUT, sU, sN, sO, HC_F, HID: tl.constexpr, BD: tl.constexpr, HC: tl.constexpr):
    r = tl.program_id(0)
    d = tl.arange(0, BD)
    m = d < HID
    acc = tl.zeros([BD], dtype=tl.float32)
    for s in tl.static_range(HC):
        off = s * HID + d
        u = tl.load(UP + r * sU + off, mask=m, other=0.0)
        g = tl.sigmoid(u.to(tl.float32)).to(u.dtype).to(tl.float32)
        n = tl.load(NORMED + r * sN + off, mask=m, other=0.0).to(tl.float32)
        acc += (g * n).to(u.dtype).to(tl.float32)
    tl.store(OUT + r * sO + d, (acc / HC_F).to(OUT.dtype.element_ty), mask=m)


def _warps(width: int) -> int:
    return 4 if width <= 1024 else 8


def _check_streams(h: torch.Tensor, hc: int, *, least: int = 2) -> int:
    if h.ndim != 2 or type(hc) is not int or hc < least or h.shape[1] % hc:
        raise ValueError(f"the streams are [N, hc*H] with hc >= {least}; got {tuple(h.shape)} for hc {hc}")
    if h.stride(1) != 1:
        raise ValueError("the streams must be packed along their channels")
    return h.shape[1] // hc


def pack_down_inject(down: torch.Tensor, inject: "torch.Tensor | None") -> torch.Tensor:
    """The mixer's down projection [r, hc*H] and the injection [hc, hc*H] as one weight [r + hc, hc*H]: both read
    the normalised streams, so one matmul computes both. Bind once when the weights are bound; `inject` None is the
    closing mixer, which has no injection."""
    if down.ndim != 2 or (inject is not None and (inject.ndim != 2 or inject.shape[1] != down.shape[1]
                                                  or inject.dtype != down.dtype)):
        raise ValueError("down [r, hc*H] and inject [hc, hc*H] share their input width and dtype")
    return (down if inject is None else torch.cat([down, inject], 0)).contiguous()


def norm_streams(h: torch.Tensor, w: torch.Tensor, eps: float, hc: int) -> torch.Tensor:
    """rmsnorm_unit_offset(h, w, eps, group=H): each of the hc streams normalised on its own, weight 1 + w. hc 1 is the
    plain unit-offset norm over the whole row (the MTP fuse's joint norm over 10,240 channels, its embedding norm)."""
    hid = _check_streams(h, hc, least=1)
    if w.shape != (h.shape[1],):
        raise ValueError("the stream norm's weight covers every stream's channels")
    if not h.is_cuda:
        from engine.modules.norm import rmsnorm_unit_offset
        return rmsnorm_unit_offset(h, w, eps, group=None if hc == 1 else hid)
    out = torch.empty_like(h)
    if h.shape[0]:
        _norm_streams[(h.shape[0], hc)](h, w, out, h.stride(0), out.stride(0), eps,
                                        HID=hid, BD=triton.next_power_of_2(hid), num_warps=_warps(hid))
    return out


def leave(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor, hc: int) -> torch.Tensor:
    """h + out (x) inject, in place: the sublayer's output [N, H] added into every stream with that stream's weight
    [N, hc]. For the site before an injection feature, which reads the streams un-normalised."""
    return _leave(h, out, inject, None, 0.0, hc, norm=False)[0]


def leave_norm(h: torch.Tensor, out: torch.Tensor, inject: torch.Tensor, w: torch.Tensor, eps: float,
               hc: int) -> "tuple[torch.Tensor, torch.Tensor]":
    """The previous site's leave and this site's stream norm in one pass: h updated in place, and the normalised
    streams the mixer reads. Returns (h, normed)."""
    if w.shape != (h.shape[1],):
        raise ValueError("the stream norm's weight covers every stream's channels")
    return _leave(h, out, inject, w, eps, hc, norm=True)


def _leave(h, out, inject, w, eps, hc, *, norm):
    hid = _check_streams(h, hc)
    if out.shape != (h.shape[0], hid) or inject.shape != (h.shape[0], hc):
        raise ValueError(f"a leave takes the output [N, {hid}] and the injection [N, {hc}] for {h.shape[0]} rows")
    if out.dtype != h.dtype or inject.dtype != h.dtype or out.stride(1) != 1 or inject.stride(1) != 1:
        raise ValueError("the output and the injection are packed rows in the streams' dtype")
    if not h.is_cuda:
        h.add_((out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2))
        if not norm:
            return h, None
        from engine.modules.norm import rmsnorm_unit_offset
        return h, rmsnorm_unit_offset(h, w, eps, group=hid)
    normed = torch.empty_like(h) if norm else h
    if h.shape[0]:
        _leave_norm[(h.shape[0], hc)](h, out, inject, w if norm else h, normed, h.stride(0), out.stride(0),
                                      inject.stride(0), normed.stride(0), eps, HID=hid,
                                      BD=triton.next_power_of_2(hid), NORM=norm, num_warps=_warps(hid))
    return h, (normed if norm else None)


def mix(normed: torch.Tensor, down_inject: torch.Tensor, up: torch.Tensor, hc: int, *,
        inject: bool = True) -> "tuple[torch.Tensor, torch.Tensor | None]":
    """The site's mixer over the normalised streams: (mixed [N, H], injection [N, hc] or None for the closing mixer).
    `down_inject` is pack_down_inject's weight; `up` [hc*H, r]."""
    hid = _check_streams(normed, hc)
    rank = up.shape[1]
    if up.shape != (normed.shape[1], rank) or down_inject.shape != (rank + (hc if inject else 0), normed.shape[1]):
        raise ValueError(f"a site mixes through down(+inject) [{rank}{' + ' + str(hc) if inject else ''}, "
                         f"{normed.shape[1]}] and up [{normed.shape[1]}, {rank}]")
    if down_inject.dtype != normed.dtype or up.dtype != normed.dtype:
        raise ValueError("the hyper-connection weights are held in the activations' dtype (BF16 in the checkpoint)")
    if not normed.is_cuda:
        di = torch.nn.functional.linear(normed, down_inject)
        gates = torch.nn.functional.silu(di[:, :rank] / hc)
        weights = torch.sigmoid(torch.nn.functional.linear(gates, up)).unflatten(-1, (hc, hid))
        mixed = (weights * normed.unflatten(-1, (hc, hid))).mean(dim=-2)
        return mixed, (2 * torch.sigmoid(di[:, rank:] / hc) if inject else None)
    rows = normed.shape[0]
    di = torch.mm(normed, down_inject.t())
    gates = torch.empty(rows, rank, device=normed.device, dtype=normed.dtype)
    injection = torch.empty(rows, hc, device=normed.device, dtype=normed.dtype) if inject else None
    mixed = torch.empty(rows, hid, device=normed.device, dtype=normed.dtype)
    if rows:
        _gates[(rows,)](di, gates, gates if injection is None else injection, di.stride(0), gates.stride(0),
                        (gates if injection is None else injection).stride(0), float(hc), R=rank,
                        BR=triton.next_power_of_2(rank), HC=hc, BH=triton.next_power_of_2(hc), WITH_INJECT=inject,
                        num_warps=4)
    weights = torch.mm(gates, up.t())
    if rows:
        _mix_mean[(rows,)](weights, normed, mixed, weights.stride(0), normed.stride(0), mixed.stride(0), float(hc),
                           HID=hid, BD=triton.next_power_of_2(hid), HC=hc, num_warps=_warps(hid))
    return mixed, injection


def drift(ours: torch.Tensor, ref: torch.Tensor) -> "tuple[float, float]":
    """(largest error over the largest reference magnitude, error RMS over reference RMS), in fp32. A BF16 step is 2^-7
    of a value (7 mantissa bits): an operation that rounds where the torch form does not -- a reduction's partial
    sums -- moves an element by a step or two, far under the tenths that a wrong formula, stream or channel gives."""
    a, b = ours.float(), ref.float()
    err = (a - b).abs()
    return (float((err.max() / b.abs().max().clamp_min(1e-30)).item()),
            float((err.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-30)).item()))


def qualify(device, *, hc: int, hidden: int, rank: int, eps: float, dtype=torch.bfloat16, rows=(1, 5, 64),
            band_max: float = 5e-2, band_rms: float = 2e-2, seed: int = 0) -> dict:
    """Hold the lane to engine/modules/hyper_connection.gated_residual on `device`, with random weights at the model's
    widths: two sites joined by a leave with its norm, a leave without one, and a closing mixer. Raises when an output
    drifts past `band_max` (largest error / largest magnitude) or `band_rms` (`drift`): bounds a few BF16 steps wide,
    so rounding order passes and arithmetic does not; returns the worst (max, rms) seen per output."""
    from engine.modules.hyper_connection import gated_residual
    gen = torch.Generator(device="cpu").manual_seed(seed)
    width = hc * hidden

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=dtype)

    worst = {k: (0.0, 0.0) for k in ("enter", "inject", "leave_norm", "leave", "close")}
    norm_w, down, up, inj = rand(width, scale=0.1), rand(rank, width, scale=0.02), rand(width, rank, scale=0.02), \
        rand(hc, width, scale=0.02)
    norm_c, down_c, up_c = rand(width, scale=0.1), rand(rank, width, scale=0.02), rand(width, rank, scale=0.02)
    di, dc = pack_down_inject(down, inj), pack_down_inject(down_c, None)

    def note(key, ours, ref):
        m, r = drift(ours, ref)
        worst[key] = (max(worst[key][0], m), max(worst[key][1], r))

    for n in rows:
        h = rand(n, width)
        ref_mixed, ref_inj = gated_residual(h, norm_w, down, up, inj, hc, eps)
        mixed, injection = mix(norm_streams(h, norm_w, eps, hc), di, up, hc)
        note("enter", mixed, ref_mixed)
        note("inject", injection, ref_inj)
        out = rand(n, hidden)
        ref_h = h + (out.unsqueeze(-2) * ref_inj.unsqueeze(-1)).flatten(-2)
        ours, normed = leave_norm(h.clone(), out, ref_inj, norm_c, eps, hc)
        note("leave_norm", ours, ref_h)
        note("leave", leave(h.clone(), out, ref_inj, hc), ref_h)
        closed, _ = mix(normed, dc, up_c, hc, inject=False)
        note("close", closed, gated_residual(ref_h, norm_c, down_c, up_c, None, hc, eps))
    bad = {k: v for k, v in worst.items() if v[0] > band_max or v[1] > band_rms}
    if bad:
        raise RuntimeError(f"gated residual lane drifts from engine/modules/hyper_connection.gated_residual beyond "
                           f"max {band_max:g} / rms {band_rms:g}: {bad}")
    return worst


__all__ = ["pack_down_inject", "norm_streams", "leave", "leave_norm", "mix", "drift", "qualify"]
