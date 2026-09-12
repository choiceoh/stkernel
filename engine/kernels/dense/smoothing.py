"""Channel smoothing from the calibration (SmoothQuant, Xiao et al. 2022), folded into the norm before the weight.

The NVFP4 prefill lane quantises its activations to e2m1 under one e4m3 scale per 16 channels: an outlier channel
(the 33차 Hessians had them 20x) sets its group's scale and the fifteen neighbours lose their bits. The calibration
knows every input channel's peak (kernels/dense/calibration sums it beside the Hessian), and the weight column can
carry what the activation gives up: with s_k = amax_x[k]^a / amax_w[k]^(1-a) the product x @ W^T equals
(x / s) @ (W diag s)^T exactly, the activation quantises evenly, and the pack takes the scaled weight through the
same GPTQ (its Hessian scaled alike). The division by s is free when the input is a norm's output: the norm's
weight is divided instead, once, at preparation (45차 §23 조사 9차: -28% of the NVFP4 lane's error, nothing lost on
the FP8 and W4A8 lanes whose e4m3 mantissa is the floor). Every weight fed by the same norm output takes the same s,
computed over all their columns' peaks -- a consumer left out would read a scaled input it was not prepared for.
"""
import torch

ALPHA = 0.5
CLAMP = (2.0 ** -6, 2.0 ** 6)      # a bf16 norm weight divided by s must stay in range; so must the folded columns
POW2 = True                        # factors rounded to powers of two: dividing the norm and scaling the columns is then
                                   # EXACT in bf16 (no product changes), the fold is bit-identical end to end, and the
                                   # activation balance is within sqrt(2) of the continuous optimum


def scales(amax_x: torch.Tensor, weights, alpha: float = ALPHA, pow2: bool = POW2) -> torch.Tensor:
    """s [K] fp32 for an input whose channel peaks are `amax_x` [K], read by `weights` ([N_i, K] each): the
    migration strength `alpha` (0.5 balances activation and weight ranges); channels nobody uses keep 1."""
    amax_w = None
    for w in weights:
        peak = w.detach().float().abs().amax(0)
        amax_w = peak if amax_w is None else torch.maximum(amax_w, peak)
    ax = amax_x.detach().float().to(amax_w.device)
    s = ax.clamp_min(1e-5).pow(alpha) / amax_w.clamp_min(1e-5).pow(1 - alpha)
    s = torch.where((ax > 0) & (amax_w > 0), s, torch.ones_like(s))
    s = s.clamp(*CLAMP)
    return torch.exp2(torch.round(torch.log2(s))) if pow2 else s


def fold(norm_w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Divide the norm's weight by s in place (bf16), and return the EXACT factor the weights must be multiplied by:
    the ratio of the old weight to the rounded new one, so that (x * norm_w') @ (W * s_eff)^T == (x * norm_w) @ W^T
    up to the weights' own bf16 rounding -- the reciprocal of a rounded quotient is not the quotient."""
    old = norm_w.detach().float().clone()
    new = (old / s.to(old.device)).to(norm_w.dtype)
    s_eff = torch.where(new.float() != 0, old / new.float().clamp_min(torch.finfo(torch.float32).tiny), torch.ones_like(old))
    s_eff = torch.where(new.float() != 0, s_eff, torch.ones_like(old))
    norm_w.copy_(new)
    return s_eff


def smooth_weight(w: torch.Tensor, s_eff: torch.Tensor) -> torch.Tensor:
    """W diag(s_eff) in W's dtype (a new tensor; the caller decides what to do with the source region)."""
    return (w.detach().float() * s_eff.to(w.device)[None, :]).to(w.dtype)


def smooth_hessian(H: torch.Tensor, s_eff: torch.Tensor) -> torch.Tensor:
    """The Hessian of the scaled input x / s: diag(1/s) H diag(1/s)."""
    inv = (1.0 / s_eff.to(H.device, H.dtype))
    return (H * inv[:, None]) * inv[None, :]
