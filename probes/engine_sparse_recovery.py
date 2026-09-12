# SPDX-License-Identifier: Apache-2.0
"""Calibration-only sparse FP4 experiments, independent of serving dispatch.

SparseGPT-style inverse-Hessian reconstruction, with adjacent-pair masks
chosen per K32 block and fixed E4M3 scales while propagating FP4 rounding
error. This is an adaptation, not an invocation of stock scalar-2:4 SparseGPT.
Algorithm reference: Frantar and Alistarh, https://arxiv.org/abs/2301.00774.
"""
import torch

from probes.engine_sparse_nvfp4 import dequant, validate_sparse
from probes.engine_sparse_nvfp4_prune import quantize32


def pair_mask(score):
    if score.shape[-1] % 8 or not torch.isfinite(score).all():
        raise ValueError('finite scores with K divisible by eight required')
    grouped = score.reshape(*score.shape[:-1], -1, 4, 2).sum(-1)
    selected = grouped.argsort(dim=-1, descending=True, stable=True)[..., :2]
    keep = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, selected, True)
    return keep.unsqueeze(-1).expand(*keep.shape, 2).reshape_as(score)


def quantized32(weight):
    packed, scale = quantize32(weight)
    return dequant(packed, scale), packed, scale


def magnitude(weight):
    return quantized32(weight * pair_mask(weight.square()))


def wanda_pair(weight, inputs):
    second_moment = inputs.float().square().mean(0)
    return quantized32(weight * pair_mask(weight.square() * second_moment))


def hessian(inputs):
    if inputs.ndim != 2 or len(inputs) == 0 or not torch.isfinite(inputs).all():
        raise ValueError('nonempty finite calibration matrix required')
    x = inputs.float()
    return (x.T @ x) / len(x)


def inverse_factor(moment, damping):
    """U.T @ U = inverse(H + lambda I), with unseen columns isolated."""
    if not 0 < damping <= 1 or moment.ndim != 2 or moment.shape[0] != moment.shape[1]:
        raise ValueError('positive damping <= 1 and square Hessian required')
    if not torch.isfinite(moment).all():
        raise ValueError('nonfinite Hessian')
    h = moment.float().clone()
    diagonal = h.diagonal()
    dead = diagonal == 0
    diagonal[dead] = 1
    diagonal.add_(damping * diagonal.mean())
    chol = torch.linalg.cholesky(h)
    return torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True), dead


def _encode_scaled(values, scales):
    """Nearest-even E2M1 under fixed E4M3 scales; return values and nibbles."""
    positive = scales.float()
    z = values * torch.where(positive > 0, positive.reciprocal(), 0)
    edges = values.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    absolute = z.abs().contiguous()
    code = torch.bucketize(absolute, edges)
    tie = (code < 7) & (absolute == edges[code.clamp(max=6)])
    code += ((code % 2 == 1) & tie)
    code = code.to(torch.uint8) | ((z < 0).to(torch.uint8) << 3)
    table = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                              -0., -.5, -1., -1.5, -2., -3., -4., -6.])
    return table[code.long()] * positive, code


@torch.inference_mode()
def sparsegpt_pair(weight, moment, damping=.1):
    """Sequential OBS error compensation with pair masks and exportable K32 SF.

    Decisions use calibration inputs only. Every group fixes its scale before
    column-wise reconstruction; both pruning and FP4 rounding error propagate
    into remaining columns. Blocks of 128 bound the per-column update cost.
    """
    if weight.ndim != 2 or weight.shape[1] % 32 or not torch.isfinite(weight).all():
        raise ValueError('finite 2D weight with K divisible by 32 required')
    if moment.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError('Hessian does not match weight input dimension')
    upper, dead = inverse_factor(moment, damping)
    work = weight.float().clone()
    work[:, dead] = 0
    rows, k = work.shape
    codes = torch.empty_like(work, dtype=torch.uint8)
    scales = torch.empty((rows, k//32), device=work.device, dtype=torch.float8_e4m3fn)
    reconstructed = torch.empty_like(work)
    for start in range(0, k, 128):
        end = min(start+128, k)
        current = work[:, start:end].clone()
        errors = torch.zeros_like(current)
        block_u = upper[start:end, start:end]
        for local in range(end-start):
            if local % 32 == 0:
                group = current[:, local:local+32]
                saliency = group.square() / block_u.diagonal()[local:local+32].square()
                keep = pair_mask(saliency)
                sf = ((group * keep).abs().amax(-1)/6).clamp(max=448).to(torch.float8_e4m3fn)
                scales[:, (start+local)//32] = sf
            column = current[:, local]
            value, nibble = _encode_scaled(column * keep[:, local % 32], sf)
            codes[:, start+local] = nibble
            reconstructed[:, start+local] = value
            error = (column-value) / block_u[local, local]
            current[:, local:] -= error[:, None] * block_u[local, local:][None, :]
            errors[:, local] = error
        work[:, end:] -= errors @ upper[start:end, end:]
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    raw_scales = scales.view(torch.uint8).contiguous()
    validate_sparse(packed)
    if not torch.equal(dequant(packed, raw_scales), reconstructed):
        raise AssertionError('export differs from reconstructed weights')
    return reconstructed, packed, raw_scales


def metrics(actual, reference):
    actual, reference = actual.float(), reference.float()
    if actual.shape != reference.shape or not actual.numel():
        raise ValueError('matching nonempty outputs required')
    delta = actual-reference
    row_error = delta.norm(dim=-1) / reference.norm(dim=-1).clamp_min(1e-12)
    return dict(relative_l2=(delta.norm()/reference.norm().clamp_min(1e-12)).item(),
                row_relative_l2_p50=row_error.median().item(),
                row_relative_l2_p95=torch.quantile(row_error, .95).item(),
                mean_cosine=torch.nn.functional.cosine_similarity(actual, reference, dim=-1).mean().item(),
                finite=bool(torch.isfinite(actual).all()), rows=len(actual))


def choose_validation(candidates):
    """No test metrics accepted: select using the explicitly named validation key."""
    if not candidates:
        raise ValueError('no candidates')
    return min(candidates, key=lambda name: (candidates[name]['validation']['relative_l2'], name))
