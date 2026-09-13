"""Exact adapters that put other attention cells on the compiled MLA kernel (cells.GLUE).

The megakernel computes, for every query head h of row t over that row's selected latent rows c_j
(e4m3 x ckv_scale; `mla_decode_ref` in this package is its torch twin):

    out[t, h] = sum_j softmax_j(q[t, h] . c_j * sm_scale) c_j

and it is compiled for MLA_HEADS query heads over an MLA_LATENT latent. Heads never interact and a zero coordinate
adds nothing to a dot product, so three reshapes of the tensors reach other cells with the kernel's own arithmetic:

- **heads** (`grouped`): any head count per rank runs in groups of MLA_HEADS. Zero-query heads fill a partial group
  and their outputs are dropped: a zero query is a uniform softmax over the same rows, wasted work with no effect on
  the real heads. engine/profiles/glm53/lanes.py already groups GLM-5.3's 64 heads this way at world 1.
- **latent** (`grouped` over rows from `pad_rows`): a latent narrower than MLA_LATENT is zero-extended -- the cache rows
  once, when they are written, the queries per call -- and the output's first coordinates are the attention. The
  cache pays MLA_LATENT/latent of its memory.
- **GQA** (`gqa` over rows from `pack_kv`): a KV head's key and value sit side by side in one latent row,
  [k * k_gain ; v * v_gain ; 0], and the query is [q / k_gain ; 0], so q'.c = q.k and the value half of the output,
  divided by v_gain, is sum_j softmax_j(q . k_j * scale) v_j. Query head h reads KV head g = h // (heads / kv_heads),
  whose row for position s is s * kv_heads + g: the cache viewed as [positions, kv_heads, MLA_LATENT]. The gains are
  powers of two, exact in BF16 and e4m3, and balance the two halves under the latent's one scale.

What is not exact: the KV cache is the latent's one-scale e4m3, so a model whose reference keeps K and V in BF16 is
served FP8 KV (the quality gate decides). A softmax with a sink term is not this kernel's math; `check` refuses it by
the rule the wizard's table uses (cells.mla_glue_refusal).

`ckv_scale` keeps the kernel's meaning: the cache row times it is the latent (`mla_decode_ref`). `pad_rows` and
`pack_kv` produce the latent itself, what a writer stores at ckv_scale 1 (as engine/profiles/glm53/net.py stores its
latent); a writer that stores rows divided by a scale passes that scale.

`attend` is the kernel call, `mla_decode` once `arm()` has armed it; a test hands a torch twin instead. It receives
[T, MLA_HEADS, MLA_LATENT] contiguous queries and the caller's cache, slots, lens, scales and a contiguous `out`, and
writes `out`; its return value is not read.
"""
from __future__ import annotations

import math

from engine.kernels.cells import MLA_HEADS, MLA_LATENT, mla_glue_refusal


def check(attention=None) -> None:
    """Refuse, by name, an attention these adapters cannot put on the compiled cell (default: the bound shape's)."""
    if attention is None:
        from engine.base.kernel_shape import bound
        attention = bound().attention
    why = mla_glue_refusal(attention)
    if why is not None:
        raise RuntimeError(f"MLA glue: {why}; the attention asks for {attention}")


def arm() -> None:
    """Compile and self-test the MLA kernel at its own cell once, before graph capture, admitting the bound attention
    by `check` -- the lane's own `_check_cell` admits only the compiled cell."""
    from engine.kernels import mla
    mla.maybe_arm(check=check)


def _kernel():
    from engine.kernels import mla
    if not mla._ARMED["mla"]:
        raise RuntimeError("MLA glue: the kernel is not armed; call glue.arm() before the first call and before capture")
    return mla.mla_decode


def _gain(value, name: str) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0 and math.frexp(value)[0] == 0.5):
        raise ValueError(f"{name} must be a positive power of two (exact in BF16 and e4m3), got {value!r}")
    return value


def pad_rows(rows):
    """Latent rows [..., D], D <= MLA_LATENT, zero-extended to the compiled latent: what the cache writer stores."""
    import torch
    d = rows.shape[-1]
    if d > MLA_LATENT:
        raise ValueError(f"MLA glue: a {d} latent does not fit the compiled {MLA_LATENT}")
    return rows if d == MLA_LATENT else torch.nn.functional.pad(rows, (0, MLA_LATENT - d))


def pack_kv(k, v, *, k_gain=1.0, v_gain=1.0):
    """Keys and values [..., G, D] (after RoPE), 2 * D <= MLA_LATENT -> latent rows [..., G, MLA_LATENT]
    [k * k_gain ; v * v_gain ; 0]: what the cache writer stores at row position * G + g."""
    if k.shape != v.shape or k.dtype != v.dtype or k.device != v.device:
        raise ValueError("MLA glue: keys and values must match in shape, dtype and device")
    d = k.shape[-1]
    if 2 * d > MLA_LATENT:
        raise ValueError(f"MLA glue: a {d}-wide key and value do not fit side by side in the {MLA_LATENT} latent")
    k_gain, v_gain = _gain(k_gain, "k_gain"), _gain(v_gain, "v_gain")
    rows = k.new_zeros(*k.shape[:-1], MLA_LATENT)
    rows[..., :d] = k if k_gain == 1.0 else k * k_gain
    rows[..., d:2 * d] = v if v_gain == 1.0 else v * v_gain
    return rows


def _groups(q, ckv, slots, lens, sm_scale, ckv_scale, attend):
    """[T, H, W] queries, W <= MLA_LATENT and any H -> [T, H, MLA_LATENT] kernel outputs, one launch per group."""
    t, h, w = q.shape
    if (h, w) == (MLA_HEADS, MLA_LATENT):
        out = q.new_empty(t, MLA_HEADS, MLA_LATENT)
        attend(q.contiguous(), ckv, slots, lens, sm_scale, ckv_scale, out=out)
        return out
    groups = -(-h // MLA_HEADS)
    wide = q.new_zeros(groups, t, MLA_HEADS, MLA_LATENT)     # each [g] is a contiguous [T, MLA_HEADS, MLA_LATENT]
    for g in range(groups):
        lo, hi = g * MLA_HEADS, min(h, (g + 1) * MLA_HEADS)
        wide[g, :, :hi - lo, :w] = q[:, lo:hi]
    out = q.new_empty(groups, t, MLA_HEADS, MLA_LATENT)
    for g in range(groups):
        attend(wide[g], ckv, slots, lens, sm_scale, ckv_scale, out=out[g])
    return out.transpose(0, 1).reshape(t, groups * MLA_HEADS, MLA_LATENT)[:, :h]


def grouped(q, ckv, slots, lens, sm_scale, ckv_scale, *, attend=None):
    """MLA attention for any head count over a latent up to MLA_LATENT: q [T, H, D] bf16 over latent rows written
    MLA_LATENT wide (`pad_rows`) -> [T, H, D] bf16, contiguous. Otherwise the kernel's contract: e4m3 rows as a uint8
    view, int32 slots [T, W] with the valid prefix first, int32 lens [T] on the device."""
    t, h, d = q.shape
    if d > MLA_LATENT:
        raise ValueError(f"MLA glue: a {d} latent does not fit the compiled {MLA_LATENT}")
    out = _groups(q, ckv, slots, lens, sm_scale, ckv_scale, attend or _kernel())
    return out if (h, d) == (MLA_HEADS, MLA_LATENT) else out[..., :d].contiguous()


def gqa(q, ckv, slots, lens, sm_scale, ckv_scale, *, kv_heads, k_gain=1.0, v_gain=1.0, attend=None):
    """Grouped-query attention on the MLA kernel: q [T, H, D] bf16 (after RoPE, H a multiple of `kv_heads`) over rows
    written by `pack_kv` with the same gains -> [T, H, D] bf16, contiguous: sum_j softmax_j(q . k_j * sm_scale) v_j
    over each row's selected positions. `slots` are positions; KV head g reads rows position * kv_heads + g."""
    import torch
    t, h, d = q.shape
    if type(kv_heads) is not int or kv_heads <= 0 or h % kv_heads:
        raise ValueError(f"MLA glue: {h} query heads are not a multiple of {kv_heads!r} KV heads")
    if 2 * d > MLA_LATENT:
        raise ValueError(f"MLA glue: a {d}-wide key and value do not fit side by side in the {MLA_LATENT} latent")
    if kv_heads > 1 and ckv.shape[0] > 2 ** 31 - 1:
        raise ValueError("MLA glue: the latent rows must stay addressable by int32 slot ids")
    k_gain, v_gain = _gain(k_gain, "k_gain"), _gain(v_gain, "v_gain")
    attend = attend or _kernel()
    queries = q if k_gain == 1.0 else q / k_gain
    per = h // kv_heads
    parts = [_groups(queries[:, g * per:(g + 1) * per], ckv, slots if kv_heads == 1 else slots * kv_heads + g, lens,
                     sm_scale, ckv_scale, attend)[..., d:2 * d] for g in range(kv_heads)]
    out = parts[0] if kv_heads == 1 else torch.cat(parts, dim=1)
    return (out if v_gain == 1.0 else out / v_gain).contiguous()


__all__ = ["check", "arm", "pad_rows", "pack_kv", "grouped", "gqa"]
