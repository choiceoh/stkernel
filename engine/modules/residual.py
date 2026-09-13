"""Residual forms (module): how sublayers read and write the residual state -- the family the seven models' decoders
share (engine/base/composition.Residual: open / enter / leave / close), in five forms:

    PreNorm               x = norm(h); h = h + out. DeepSeek-V3, Ling-3.0, MiniMax-M3, Kimi K3 (without AttnRes). With
                          `out_conv` (Inkling): h = h + sconv(out), a fp32 depthwise causal conv over positions plus
                          out, its history per sequence (a slot per sublayer site). `norm` "rms" (T5) | "rms_unit_offset".
    GatedResidualStreams  Qwen3.8's gated streams -- engine/modules/hyper_connection.GatedResidualStreams.
    HyperStreams          mHC (Xie et al. 2026): the embedding copied into hc streams; per sublayer one linear map of the
                          RMS-normalised streams gives pre (stream collapse weights), post (where the output goes) and
                          comb (an hc x hc stream mixer, Sinkhorn-projected toward doubly stochastic); x = norm(sum_i
                          pre_i h_i); h' = post (x) out + comb^T h. GLM-5.3 (head: the streams' mean), DeepSeek-V4 (head:
                          a learned weighted collapse) -- transformers glm5_next / deepseek_v4 HyperConnection.
    AttnRes               Kimi K3's attention residuals: the residual is the running sum of the current block of `block`
                          layers; at each block boundary that sum is stored; a sublayer's input is a softmax over depth
                          -- every stored block sum and the running one, each scored by rmsnorm(v) . (norm_w * proj) --
                          of those vectors; the head does the same once more. modeling_kimi_linear.py's
                          `_apply_attn_res` and its decoder layer, quoted; there is no local oracle for it.

What every form shares: the sublayer sees a [N, H] input and returns a [N, H] output; the state between sublayers is the
form's (a tensor the composition carries opaquely: [N, H], [N, hc*H], or AttnRes's [N, (1 + blocks)*H]).

Held to transformers on the CPU (tests/test_engine_residual_family.py): HyperStreams to glm5_next's decoder layer
and deepseek_v4's head, PreNorm to deepseek_v3's decoder layer and to inkling's (with the output convs), the Qwen form
by the composition tests.
"""
from __future__ import annotations

import torch

from engine.base.composition import SITES
from engine.modules.norm import rmsnorm, rmsnorm_unit_offset


def _norm(kind: str, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if kind == "rms":
        return rmsnorm(x, weight, eps)
    if kind == "rms_unit_offset":
        return rmsnorm_unit_offset(x, weight, eps)
    raise ValueError(f"norm is 'rms' or 'rms_unit_offset', not {kind!r}")


def unweighted_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """x * rsqrt(mean(x^2) + eps) in x's dtype (glm5_next / deepseek_v4 UnweightedRMSNorm on fp32 streams)."""
    return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps).to(x.dtype)


class PreNorm:
    """`weights(layer, site, name)`: norm [H]; conv [H, 1, K] or [H, K] with `out_conv`. `final(name)`: norm, or None
    when the model has no final norm."""

    def __init__(self, *, eps: float, weights, final=None, norm: str = "rms", out_conv: int = 0, hidden: int = 0):
        _norm(norm, torch.zeros(1), torch.ones(1), eps)
        if out_conv and hidden <= 0:
            raise ValueError("output convs need the hidden width for their state")
        self.eps, self.weights, self.final, self.norm, self.out_conv, self.hidden = eps, weights, final, norm, out_conv, hidden

    def open(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def enter(self, layer: int, site: str, h: torch.Tensor, step=None, state=None):
        return _norm(self.norm, h, self.weights(layer, site, "norm"), self.eps), h

    def leave(self, layer: int, site: str, out: torch.Tensor, carry, step=None, state=None) -> torch.Tensor:
        if self.out_conv:
            out = self._conv(layer, site, out, step, state)
        return carry + out

    def close(self, h: torch.Tensor) -> torch.Tensor:
        return h if self.final is None else _norm(self.norm, h, self.final("norm"), self.eps)

    def _conv(self, layer, site, out, step, state):
        """Inkling's short conv on a sublayer's output: fp32 depthwise causal conv (no activation) plus the output,
        rounded once; the conv's last inputs kept per sequence."""
        from engine.base.composition import put_state
        from engine.modules.causal_conv import causal_conv1d, conv_states
        weight = self.weights(layer, site, "conv")
        weight = weight.reshape(weight.shape[0], -1).float()
        key = f"residual_{site}_conv"
        result = torch.empty_like(out)
        for s in step.segments:
            xs = out[s.start:s.start + s.length].float()
            before = state.get(layer, key, s.seq)
            y, held = causal_conv1d(xs, weight, None, before, None)
            put_state(state, layer, key, s, held, conv_states(xs, before, self.out_conv - 1) if s.verify else None)
            result[s.start:s.start + s.length] = (y.float() + xs).to(out.dtype)
        return result

    def cache_specs(self, layers):
        from engine.base.cache_spec import SlotSpec
        if not self.out_conv:
            return []
        taps = self.out_conv - 1
        return [SlotSpec(f"residual {site} conv", len(layers), self.hidden * taps * 4,
                         f"[hidden, kernel-1] fp32: the {site} output conv's last inputs",
                         key=f"residual_{site}_conv", dtype="float32", shape=(self.hidden, taps)) for site in SITES]


def sinkhorn_mix(streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor, hc: int,
                 rms_eps: float, hc_eps: float, iters: int):
    """(pre [N, hc], post [N, hc], comb [N, hc, hc]) from the flattened streams [N, hc*H] -- transformers glm5_next /
    deepseek_v4 HyperConnection, op for op: the streams RMS-normalised (no weight) in fp32, one linear map, sigmoid
    (+ hc_eps) for pre, 2 sigmoid for post, a softmax for comb then `iters` rounds of column/row normalisation."""
    flat = unweighted_rmsnorm(streams.float(), rms_eps)
    pre_w, post_w, comb_w = torch.nn.functional.linear(flat, fn.float()).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = base.split([hc, hc, hc * hc])
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + hc_eps
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = torch.softmax(comb_w.view(-1, hc, hc) * scale[2] + comb_b.view(hc, hc), dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, comb


class HyperStreams:
    """`weights(layer, site, name)`: fn [(2+hc)*hc, hc*H], base [(2+hc)*hc], scale [3], norm [H] (the sublayer's input
    norm). `final(name)`: norm; with head "weighted" also fn [hc, hc*H], base [hc], scale [1]."""

    def __init__(self, *, hc: int, eps: float, hc_eps: float, sinkhorn: int, weights, final, head: str = "mean",
                 norm: str = "rms"):
        if type(hc) is not int or hc <= 1 or sinkhorn < 1:
            raise ValueError("hyper streams: hc > 1 and at least one Sinkhorn round")
        if head not in ("mean", "weighted"):
            raise ValueError(f"head is 'mean' (GLM-5.3) or 'weighted' (DeepSeek-V4), not {head!r}")
        _norm(norm, torch.zeros(1), torch.ones(1), eps)
        self.hc, self.eps, self.hc_eps, self.sinkhorn = hc, eps, hc_eps, sinkhorn
        self.weights, self.final, self.head, self.norm = weights, final, head, norm

    def open(self, x: torch.Tensor) -> torch.Tensor:
        return x.repeat(1, self.hc)

    def enter(self, layer: int, site: str, h: torch.Tensor, step=None, state=None):
        w = lambda name: self.weights(layer, site, name)
        n, hc = h.shape[0], self.hc
        r = h.view(n, hc, -1)
        pre, post, comb = sinkhorn_mix(h, w("fn"), w("base"), w("scale"), hc, self.eps, self.hc_eps, self.sinkhorn)
        collapsed = (pre.unsqueeze(-1) * r).sum(dim=1).to(h.dtype)
        return _norm(self.norm, collapsed, w("norm"), self.eps), (post, comb, r)

    def leave(self, layer: int, site: str, out: torch.Tensor, carry, step=None, state=None) -> torch.Tensor:
        post, comb, r = carry
        streams = post.to(out.dtype).unsqueeze(-1) * out.unsqueeze(-2) + torch.matmul(comb.to(out.dtype).transpose(-1, -2), r)
        return streams.flatten(1)

    def close(self, h: torch.Tensor) -> torch.Tensor:
        r = h.view(h.shape[0], self.hc, -1)
        if self.head == "mean":
            collapsed = r.mean(dim=1)
        else:
            flat = unweighted_rmsnorm(h.float(), self.eps)
            mixes = torch.nn.functional.linear(flat, self.final("fn").float())
            pre = torch.sigmoid(mixes * self.final("scale").float() + self.final("base").float()) + self.hc_eps
            collapsed = (pre.unsqueeze(-1) * r).sum(dim=1).to(h.dtype)
        return _norm(self.norm, collapsed, self.final("norm"), self.eps)


def attn_res(prefix: torch.Tensor, blocks: torch.Tensor, proj: torch.Tensor, norm_weight: torch.Tensor,
             eps: float) -> torch.Tensor:
    """Kimi K3's `_apply_attn_res`: over v = [blocks..., prefix] ([N, B+1, H]) a softmax of rmsnorm(v) . (norm_w * proj)
    picks the mix of the raw vectors."""
    v = torch.cat([blocks, prefix.unsqueeze(1)], dim=1)
    vf = v.float()
    k = vf * torch.rsqrt(vf.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * (norm_weight.float() * proj.float().reshape(-1))).sum(-1)                       # [N, B+1]
    probs = torch.softmax(scores, dim=-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(v.dtype)


class AttnRes:
    """`weights(layer, site, name)`: norm [H] (the sublayer's input norm), res_proj [1, H], res_norm [H]. `final(name)`:
    norm, res_proj, res_norm. The state is [prefix | block sums] flattened, [N, (1 + B) * H]."""

    def __init__(self, *, block: int, eps: float, weights, final, norm: str = "rms"):
        if block <= 0:
            raise ValueError(f"a block is a positive number of layers, not {block}")
        _norm(norm, torch.zeros(1), torch.ones(1), eps)
        self.block, self.eps, self.weights, self.final, self.norm = block, eps, weights, final, norm

    def open(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def _split(h: torch.Tensor, hidden: int):
        n = h.shape[0]
        parts = h.view(n, -1, hidden)
        return parts[:, 0], parts[:, 1:]                                          # prefix [N, H], blocks [N, B, H]

    def enter(self, layer: int, site: str, h: torch.Tensor, step=None, state=None):
        w = lambda name: self.weights(layer, site, name)
        hidden = self.weights(layer, site, "norm").shape[-1]
        prefix, blocks = self._split(h, hidden)
        if site == "mixer":
            x = attn_res(prefix, blocks, w("res_proj"), w("res_norm"), self.eps) if blocks.shape[1] else prefix
            if layer % self.block == 0:
                blocks = torch.cat([blocks, prefix.unsqueeze(1)], dim=1)
                prefix = None
        else:
            x = attn_res(prefix, blocks, w("res_proj"), w("res_norm"), self.eps)
        return _norm(self.norm, x, w("norm"), self.eps), (prefix, blocks)

    def leave(self, layer: int, site: str, out: torch.Tensor, carry, step=None, state=None) -> torch.Tensor:
        prefix, blocks = carry
        prefix = out if prefix is None else prefix + out
        return torch.cat([prefix.unsqueeze(1), blocks], dim=1).flatten(1)

    def close(self, h: torch.Tensor) -> torch.Tensor:
        hidden = self.final("norm").shape[-1]
        prefix, blocks = self._split(h, hidden)
        x = attn_res(prefix, blocks, self.final("res_proj"), self.final("res_norm"), self.eps)
        return _norm(self.norm, x, self.final("norm"), self.eps)
