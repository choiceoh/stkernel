"""The rotary tables, and the rotation that is undone again.

Two things here are easy to get wrong in a way that runs.

TWO TABLES. A layer that compresses uses `compress_rope_theta` (160,000 here)
with YaRN over `original_max_position_embeddings`; a layer that does not --
compress_ratio 0, which is layers 0 and 1 -- disables YaRN and uses the base
`rope_theta` (10,000). Handing a layer the other table produces positions that
are wrong by a smooth, plausible amount, which is the failure mode long context
is made of. `LayerPlan.rope` picks; this builds.

THE ROTATION COMES BACK OFF. `apply_rotary_emb(..., inverse=True)` conjugates,
and the reference uses it on the attention OUTPUT to remove the query's
rotation -- which is what lets the KV cache hold one shared rotated form
instead of a per-query one. Skipping the inverse leaves every output rotated by
its own position: no error, no NaN, and an answer that drifts with position.

YaRN itself is a ramp, not a switch. Dimensions whose wavelength already fits
in the training context keep their frequency; those far past it are divided by
`factor`; the `beta_fast`..`beta_slow` band between is faded linearly. The two
corner dimensions come from `dim * log(orig / (rot * 2pi)) / (2 * log(base))`,
and getting the log base wrong tilts the whole ramp by a factor nothing
downstream can notice.
"""

from __future__ import annotations

import math


def precompute_freqs_cis(dim: int, seqlen: int, original_seq_len: int,
                         base: float, factor: float, beta_fast: float,
                         beta_slow: float):
    """[seqlen, dim // 2] complex. `original_seq_len` 0 disables YaRN."""
    import torch

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations: float) -> float:
            return (dim * math.log(original_seq_len / (rotations * 2 * math.pi))
                    / (2 * math.log(base)))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    # arange without a dtype, as the reference has it: int64 promoted by the
    # outer product. Exact up to 2**24 positions, which covers the 1M context.
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(x, freqs_cis, inverse: bool = False):
    """Rotate IN PLACE, adjacent element pairs as one complex number.

    Accepts [b, s, d] and [b, s, h, d]. `inverse` conjugates, which is how the
    output gets the query's rotation removed so the cache can stay in one
    shared rotated form.
    """
    import torch

    out = x
    z = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if z.ndim == 3:
        freqs_cis = freqs_cis.view(1, z.size(1), z.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, z.size(1), 1, z.size(-1))
    out.copy_(torch.view_as_real(z * freqs_cis).flatten(-2).type_as(out))
    return out
