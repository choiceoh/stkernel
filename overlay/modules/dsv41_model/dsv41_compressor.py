"""The KV compressor: `compress_ratio` tokens pooled into one latent.

This is the CED mechanism itself. Four layers carry one -- kv_source_layer_ids
[2, 8, 14, 20] -- and everything downstream reads what they produce: the
decoder attends to it, and the indexer keys against it.

Two behaviours matter more than the arithmetic.

It returns None while a group is filling. At ratio 2 a decode step yields a
latent only every other step, and the caller has to handle that rather than
assume one KV entry per token. A KV cache written as if every step produced one
is off by a factor of `compress_ratio` and silently keeps the wrong history.

It is pre-RoPE on purpose. The indexer needs the unrotated latent, so the
rotation happens in attention afterwards. Rotating here would be invisible in
any shape check and wrong in every score.

Ratio 1 is a different path, not a special case of the same one: no gate, no
fp32, no state. The reference builds `wgate` only above ratio 1, which is why
layer 20 has none -- it is the decoder's ratio, not its position.

The fp32 is load-bearing. Above ratio 1 the pooling runs in fp32 and the
reference promotes wkv/wgate to fp32 to match, while the checkpoint stores them
bf16. That promotion is a load step. Doing the softmax in bf16 instead changes
the pooling weights of every group.

probes/dsv41_compressor_diff.py holds this to bit equality against the
reference class over prefill, ragged prefill, and decode across a group
boundary.
"""

from __future__ import annotations


class Compressor:
    """Stateless over prefill, stateful across decode steps.

    `kv_state` / `score_state` carry the tail of an incomplete group. They are
    per-batch-row, so a caller that reuses a row for a different sequence has
    to clear them -- there is no sequence id here to notice.
    """

    def __init__(self, hidden: int, head_dim: int, compress_ratio: int,
                 norm_eps: float, max_batch_size: int = 1) -> None:
        import torch

        self.ratio = int(compress_ratio)
        self.head_dim = head_dim
        self.eps = norm_eps
        self.hidden = hidden
        if self.ratio > 1:
            shape = (max_batch_size, self.ratio, head_dim)
            self.kv_state = torch.zeros(shape, dtype=torch.float32)
            self.score_state = torch.full(shape, -torch.inf, dtype=torch.float32)
        else:
            self.kv_state = self.score_state = None

    def _rms(self, x, weight):
        import torch

        out = x.float()
        out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * weight.float()).type_as(x)

    def forward(self, x, start_pos: int, wkv, wgate, norm_weight):
        """x: [B, L, hidden] -> [B, groups, head_dim], or None if none completed.

        `wkv`/`wgate` are the raw checkpoint weights; above ratio 1 they are
        used in fp32, which is the promotion the reference does at construction.
        """
        import torch
        import torch.nn.functional as F

        bsz, seqlen, _ = x.shape
        ratio, dtype = self.ratio, x.dtype
        if ratio == 1:
            return self._rms(F.linear(x, wkv.to(x.dtype)), norm_weight)

        xf = x.float()
        kv = F.linear(xf, wkv.float())
        score = F.linear(xf, wgate.float())
        if start_pos == 0:
            should = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            if remainder:
                kv, tail_kv = kv.split([cutoff, remainder], dim=1)
                score, tail_score = score.split([cutoff, remainder], dim=1)
                self.kv_state[:bsz, :remainder] = tail_kv
                self.score_state[:bsz, :remainder] = tail_score
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should = (start_pos + 1) % ratio == 0
            slot = start_pos % ratio
            self.kv_state[:bsz, slot] = kv.squeeze(1)
            self.score_state[:bsz, slot] = score.squeeze(1)
            if should:
                kv = (self.kv_state[:bsz]
                      * self.score_state[:bsz].softmax(dim=1)).sum(dim=1,
                                                                   keepdim=True)
        if not should:
            return None
        return self._rms(kv.to(dtype), norm_weight)
