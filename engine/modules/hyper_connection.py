"""Hyper-connection mixes (module): the family DSv4.1, GLM-5.3 and Qwen3.8 share.

    hc_split_sinkhorn   DSv4.1's kernel, term for term (sigmoid gates + Sinkhorn comb)
    mhc_pre / mhc_post  GLM-5.3's mHC, term for term from vLLM's kernels/mhc/torch.py --
                        the reference its served TileLang fork (ours) is held to.
                        probes/mhc_check.py judges these against torch.ops.vllm.mhc_*_tilelang
                        inside the glm53 image.

The two differ in where the RMS normalisation sits (mHC normalises the mix
logits by the flattened stream's RMS) and in the pre-mix epsilon; the comb is
the same softmax + Sinkhorn in both.
"""
from __future__ import annotations

import torch


def hc_split_sinkhorn(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
                      hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """kernel.py:407, term for term."""
    b, s, _ = mixes.shape
    hc = hc_mult
    m = mixes.float().view(-1, (2 + hc) * hc)
    base, scale = hc_base.float(), hc_scale.float()
    pre = torch.sigmoid(m[:, :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(m[:, hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = (m[:, 2 * hc:] * scale[2] + base[2 * hc:]).view(-1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return (pre.view(b, s, hc), post.view(b, s, hc), comb.view(b, s, hc, hc))


def mhc_pre(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
            rms_eps: float, hc_pre_eps: float, hc_sinkhorn_eps: float, hc_post_mult_value: float,
            sinkhorn_repeat: int):
    """residual [.., hc, hidden] bf16; fn [(2+hc)*hc, hc*hidden] fp32; returns
    (post_mix [.., hc, 1], comb_mix [.., hc, hc], layer_input [.., hidden])."""
    hc, hidden = residual.shape[-2], residual.shape[-1]
    outer = residual.shape[:-2]
    r = residual.reshape(-1, hc, hidden)
    x = r.reshape(r.shape[0], hc * hidden).float()
    mixes = x @ fn.float().t()
    mixes = mixes * torch.rsqrt(x.square().sum(-1, keepdim=True) / (hc * hidden) + rms_eps)
    pre = torch.sigmoid(mixes[:, :hc] * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post = torch.sigmoid(mixes[:, hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]) * hc_post_mult_value
    comb = mixes[:, 2 * hc:].view(-1, hc, hc) * hc_scale[2] + hc_base[2 * hc:].view(1, hc, hc)
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (pre.unsqueeze(-1) * r.float()).sum(1).to(torch.bfloat16)
    return post.view(*outer, hc, 1), comb.view(*outer, hc, hc), layer_input.view(*outer, hidden)


def mhc_post(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor,
             comb_res_mix: torch.Tensor) -> torch.Tensor:
    """residual_new[j] = sum_i comb[i, j] * residual[i] + post[j] * x."""
    mixed = torch.einsum("...ij,...ih->...jh", comb_res_mix.float(), residual.float())
    return (mixed + post_layer_mix.float() * x.unsqueeze(-2).float()).to(residual.dtype)


def gated_residual(hyper: torch.Tensor, norm_weight: torch.Tensor, down: torch.Tensor, up: torch.Tensor,
                   inject: "torch.Tensor | None", hc: int, eps: float):
    """Qwen3.8's gated residual hyper-connection (transformers qwen4_exp Qwen4ExpTextGatedResidual, op for op).

    hyper [N, hc*H] (the hc streams laid end to end); the streams are RMS-normalised one by one (unit-offset weight),
    a low-rank mixer (down [r, hc*H], silu over down/hc, up [hc*H, r], sigmoid) weights every channel of every stream,
    and the sublayer input is the streams' weighted mean [N, H]. With `inject` [hc, hc*H] it also returns the injection
    weights 2*sigmoid(inject(normed)/hc) [N, hc] the sublayer's output is added back with; without, the mixed input only
    (the final mixer, before the head)."""
    from engine.modules.norm import rmsnorm_unit_offset
    n, width = hyper.shape
    if width % hc:
        raise ValueError(f"{width} hyper-connection features are not {hc} streams")
    hidden = width // hc
    normed = rmsnorm_unit_offset(hyper, norm_weight, eps, group=hidden)
    mix = torch.nn.functional.silu(torch.nn.functional.linear(normed, down) / hc)
    mix = torch.sigmoid(torch.nn.functional.linear(mix, up)).unflatten(-1, (hc, hidden))
    mixed = (mix * normed.unflatten(-1, (hc, hidden))).mean(dim=-2)
    if inject is None:
        return mixed
    return mixed, 2 * torch.sigmoid(torch.nn.functional.linear(normed, inject) / hc)


class GatedResidualStreams:
    """The gated residual hyper-connection as a residual form (engine/base/composition.Residual): the embedding copied
    into hc streams; each sublayer reads the gated mix of the streams and writes its output into every stream with that
    stream's injection weight, h + (out x inject) (transformers qwen4_exp Qwen4ExpTextDecoderLayer); the final mixer,
    without injection, is the hidden state the head reads.

    `weights(layer, site, name)` returns "hc_norm", "input_mix_weight_down", "input_mix_weight_up" and
    "block_inject_weight" for site "mixer" or "mlp"; `final(name)` the same names but the last for the closing mixer."""

    def __init__(self, hc: int, eps: float, weights, final):
        if type(hc) is not int or hc <= 1:
            raise ValueError("gated residual streams need hc > 1")
        self.hc, self.eps, self.weights, self.final = hc, eps, weights, final

    def open(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, self.hc)

    def enter(self, layer: int, site: str, h: torch.Tensor, step=None, state=None):
        w = lambda name: self.weights(layer, site, name)
        mixed, inject = gated_residual(h, w("hc_norm"), w("input_mix_weight_down"), w("input_mix_weight_up"),
                                       w("block_inject_weight"), self.hc, self.eps)
        return mixed, (h, inject)

    def leave(self, layer: int, site: str, out: torch.Tensor, carry, step=None, state=None) -> torch.Tensor:
        h, inject = carry
        return h + (out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

    def close(self, h: torch.Tensor) -> torch.Tensor:
        return gated_residual(h, self.final("hc_norm"), self.final("input_mix_weight_down"),
                              self.final("input_mix_weight_up"), None, self.hc, self.eps)
