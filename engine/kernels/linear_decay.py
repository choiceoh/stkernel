"""Decay layouts for the gated delta rule lanes (torch only: importable without Triton).

The recurrent kernels in kda/ read the log-decay as ONE VALUE PER KEY CHANNEL,
`[B, T, HV, K]` -- KDA's form. GDN's decay is one value per head, `[B, T, HV]`
(modules/linear_attention: "decay per head (GDN; KDA's per-channel is a flag)"). The two
recurrences are the same, GDN's being KDA's with every channel of a head sharing the
decay, so a per-head decay reaches the same kernels as a stride-0 channel axis: no copy,
and the strided loaders (INPUT_STRIDES) read the one value for every channel.

A profile whose kernel shape says `linear.decay == "head"` (engine/base/kernel_shape)
calls `per_channel` on its precomputed log-decay before `fused_recurrent_kda(...,
compute_gate=False)`. The fused-gate paths (ring KDA, chunk KDA) compute KDA's own
per-channel gate inside the kernel and take no decay at all; they refuse a per-head
cell instead of silently running KDA's gate on GDN's projections.
"""
import torch


def per_channel(g: torch.Tensor, k_dim: int) -> torch.Tensor:
    """[B, T, HV] per-head log-decay -> the [B, T, HV, K] per-channel view the kernels read (stride 0 along K)."""
    if not isinstance(g, torch.Tensor) or g.ndim != 3:
        raise ValueError("per_channel takes a per-head log-decay [B, T, HV]")
    if type(k_dim) is not int or k_dim <= 0:
        raise ValueError("per_channel needs the key width as a positive int")
    return g.unsqueeze(-1).expand(*g.shape, k_dim)


def is_per_channel(g: torch.Tensor, k_dim: int) -> bool:
    """True when `g` already carries a decay per key channel (the KDA form the kernels read)."""
    return isinstance(g, torch.Tensor) and g.ndim == 4 and g.shape[-1] == k_dim
