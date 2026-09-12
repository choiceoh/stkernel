"""Sampling over flat logits (base): greedy, temperature, top-k, top-p -- and a
seed that makes a step replayable (D12).

Inputs are flat: logits [N, vocab] for the N sequences of a decode step, and
per-row arrays for temperature, top-k and top-p. No per-request objects
(CHARTER I2). The generator is explicit so a recorded step can be re-run
with the same seed and produce the same tokens -- that is what makes the
death dump a replay and not a log.

**One sampler, one shape.** Every path -- the captured decode sampler, the
device pipeline, and the rich rows with options -- hands its whole block of
rows to `rows()` and gets ids (and, when the speculative path needs them, the
truncated distributions) back from a single launch. There is no per-row loop
and no per-position loop anywhere below this line, because at a 154,880-token
vocabulary a decode step is 24 rows and doing them one at a time cost 11.8 ms.

**A nucleus is a value threshold, not a prefix.** top-k keeps every token whose
weight reaches the k-th largest; top-p keeps every token whose weight reaches
the smallest one the cumulative mass needed. Ties are kept on both -- which is
what top-k already did here, and what top-p only appeared to do differently
because a sort had to break its ties somehow. torch does not specify that tie
order (`modules/vocab.topk` says so); a threshold does not have to.

`engine.kernels.sampler` searches for that threshold without a sort. The torch
code below is the same definition written the obvious way, by sorting: it is
what the CUDA kernel is judged against, not a fallback it drops to.
"""
from __future__ import annotations

import torch

from engine.base.constants import iota


def rows(logits: torch.Tensor, temperature: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
         uniform: "torch.Tensor | None", valid: "int | None" = None,
         probs: "torch.Tensor | None" = None) -> "torch.Tensor | None":
    """[M] token ids for a whole block of rows, one launch, nothing read back.

    `temperature` (0 = greedy), `top_k` (0 = off) and `top_p` (1 = off) are per row, so a mixed
    batch is one call and no policy is decided by a host predicate. `uniform` is one draw a row,
    made by the caller from the row's generator -- that is where seeds and rank agreement live; a
    caller that wants only `probs` passes None and gets None. `probs`, when given, receives each
    row's sampling distribution (the speculative path picks with those, not from them).
    """
    if logits.is_cuda:
        from engine.kernels.sampler import sample_rows
        return sample_rows(logits, temperature, top_k, top_p, uniform, valid, probs)
    return _rows_by_sorting(logits, temperature, top_k, top_p, uniform, valid, probs)


def threshold(weights: torch.Tensor, top_k: "int | None", top_p: "float | None") -> float:
    """The smallest weight the truncations keep: every weight at or above it is in the nucleus.

    `weights` is one row of exp((logit - max)/T), so its largest entry is 1 and the row's mass is
    its sum. Written with a sort, which is the definition; the kernel searches for the same value.
    """
    thr = 0.0
    if top_k is not None and 0 < top_k < weights.numel():
        thr = float(weights.topk(top_k).values[-1])
    if top_p is not None and top_p < 1:
        srt = weights[weights >= thr].sort(descending=True).values
        cum = srt.cumsum(0)
        reached = cum >= float(srt.sum()) * top_p
        # the first position whose cumulative mass reaches the target is the last one kept
        thr = float(srt[int(reached.to(torch.uint8).argmax()) if bool(reached.any()) else srt.numel() - 1])
    return thr


def _rows_by_sorting(logits, temperature, top_k, top_p, uniform, valid, probs):
    """`rows` written the obvious way: the reference the kernel is judged against."""
    M, V = logits.shape
    N = V if valid is None else min(V, valid)
    out = torch.empty(M, dtype=torch.int64, device=logits.device)
    if probs is not None:
        probs.zero_()
    for i in range(M):
        raw = logits[i, :N].float()
        if float(temperature[i]) <= 0:
            out[i] = int(raw.argmax())
            if probs is not None:
                probs[i, int(out[i])] = 1.0
            continue
        w = (raw - raw.max()).div(float(temperature[i])).exp()
        w = torch.where(raw > float("-inf"), w, torch.zeros((), device=w.device))
        keep = w >= threshold(w, int(top_k[i]), float(top_p[i]))
        w = torch.where(keep, w, torch.zeros((), device=w.device))
        total = w.sum()
        if uniform is not None:
            out[i] = int(torch.searchsorted(w.cumsum(0).contiguous(), (float(uniform[i]) * total).reshape(1),
                                            right=True).clamp_max(N - 1))
        if probs is not None:
            probs[i, :N] = w / total
    return out if uniform is not None else None


def sample(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor,
           generator: "torch.Generator | None" = None, *, top_k: "torch.Tensor | None" = None,
           valid: "int | None" = None) -> torch.Tensor:
    """[N] token ids. temperature 0 means greedy for that row.

    One uniform a row comes off `generator`, in row order, on every rank alike -- greedy rows draw
    one too, so a row's stream does not shift when its neighbour's temperature does.
    """
    n = logits.shape[0]
    if top_k is None:
        top_k = torch.zeros(n, dtype=torch.int32, device=logits.device)
    u = torch.rand(n, generator=generator, device=logits.device)
    return rows(logits, temperature, top_k, top_p, u, valid)


def _selfcheck() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    logits = torch.randn(6, 1000, device=dev) * 3
    t = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.7, 0.7], device=dev)
    p = torch.tensor([1.0, 0.5, 1.0, 0.9, 1.0, 0.1], device=dev)
    g = torch.Generator(device=dev).manual_seed(1234)
    a = sample(logits, t, p, g)
    g2 = torch.Generator(device=dev).manual_seed(1234)
    b = sample(logits, t, p, g2)
    assert torch.equal(a, b), "same seed, same tokens: the replay property"
    assert (a[:2] == logits[:2].argmax(-1)).all(), "temperature 0 is greedy regardless of top_p"
    # top-p 0.1 on a peaked row must land inside the nucleus: check over many draws
    row = logits[5:6]; tt = t[5:6]; pp = p[5:6]
    srt, idx = torch.softmax(row / 0.7, -1).sort(descending=True)
    nucleus = set(idx[0, :int(((srt.cumsum(-1) - srt) < 0.1).sum())].tolist())
    draws = {sample(row, tt, pp, torch.Generator(device=dev).manual_seed(s)).item() for s in range(64)}
    assert draws <= nucleus, (draws - nucleus)
    # temperature 1, top-p 1 must reproduce the softmax itself: the draw's law, not one draw of it
    row = logits[2:3].expand(20000, -1).contiguous()
    g3 = torch.Generator(device=dev).manual_seed(7)
    got = sample(row, t[2:3].expand(20000).contiguous(), p[2:3].expand(20000).contiguous(), g3)
    seen = torch.bincount(got, minlength=1000).float() / 20000
    want = torch.softmax(logits[2].float(), -1)
    assert float((seen - want).abs().max()) < 0.01, float((seen - want).abs().max())
    print("  sampler: replayable by seed, greedy at T=0, nucleus respected, draws the softmax OK")


if __name__ == "__main__":
    _selfcheck()


# ---- the OpenAI-dialect options a row may carry (45차 §23 A3/A4/B4) ----------------------------------------------
# A row with any of these leaves the captured sampler: its logits are gathered whole and processed here, on every
# rank identically (same generator seeds, same order), so the picks agree without a message.

OPTION_KEYS = ("top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty",
               "logit_bias", "stop_token_ids", "logprobs", "grammar", "grammar_after")


def validate_options(options: dict) -> None:
    """The engine's verdict on a request's options: unknown keys and out-of-range values are refused (D3)."""
    unknown = sorted(set(options) - set(OPTION_KEYS))
    if unknown:
        raise ValueError(f"unknown sampling options {unknown}")
    p = options.get("top_p")
    if p is not None and (type(p) not in (int, float) or not 0 < p <= 1):
        raise ValueError("top_p must be in (0, 1]")
    k = options.get("top_k")
    if k is not None and (type(k) is not int or k < 1):
        raise ValueError("top_k must be a positive integer")
    for key in ("presence_penalty", "frequency_penalty"):
        v = options.get(key)
        if v is not None and (type(v) not in (int, float) or not -2 <= v <= 2):
            raise ValueError(f"{key} must be between -2 and 2")
    rp = options.get("repetition_penalty")
    if rp is not None and (type(rp) not in (int, float) or not 0 < rp <= 2):
        raise ValueError("repetition_penalty must be in (0, 2]")
    seed = options.get("seed")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    bias = options.get("logit_bias")
    if bias is not None and (not isinstance(bias, dict) or any(type(k) is not int or k < 0 or type(v) not in (int, float) for k, v in bias.items())):
        raise ValueError("logit_bias must map token ids to numbers")
    stop = options.get("stop_token_ids")
    if stop is not None and (not isinstance(stop, list) or any(type(t) is not int or t < 0 for t in stop)):
        raise ValueError("stop_token_ids must be a list of token ids")
    lp = options.get("logprobs")
    if lp is not None and (type(lp) is not int or not 0 <= lp <= 20):
        raise ValueError("logprobs must be an integer between 0 and 20")
    g = options.get("grammar")
    if g is not None and (not isinstance(g, dict) or g.get("type") not in ("json_object", "json_schema")):
        raise ValueError("grammar must be a json_object or json_schema spec")
    after = options.get("grammar_after")
    if after is not None and (type(after) is not int or after < 0):
        raise ValueError("grammar_after must be a token id")
    if after is not None and g is None:
        raise ValueError("grammar_after is the token a grammar waits for: there is no grammar")


def needs_rich_sampler(options: dict, temperature: float, drafts: bool) -> bool:
    """Whether a row's logits must be processed here rather than by the captured sampler.

    top-k and top-p are NOT on this list: the captured sampler takes both as per-row tensors now,
    so a plain `temperature + top_p` request -- the common one -- stays on the fast path and can
    run ahead of the host. What is left needs something the captured sampler cannot be handed:
    logits rewritten per row (penalties, bias, grammar), a draw from a generator that is not the
    engine's (seed), the raw logits kept for logprobs, or the probabilities themselves (a
    stochastic row with drafts, for the rejection sampling).
    """
    if any(options.get(k) is not None for k in ("seed", "presence_penalty", "frequency_penalty",
                                                 "repetition_penalty", "logit_bias", "logprobs", "grammar")):
        return True
    return drafts and temperature > 0


class History:
    """Each row's token history as device tensors, grown one token at a time.

    The penalties need two things: which ids appeared at all, prompt or output, for the
    repetition penalty; and how often each output id appeared, for the presence and
    frequency ones. Deriving those from the token list walks the whole prompt, and the
    rich sampler asks once per row per draft position -- measured at 20.4 ms a row on a
    128K prompt, 456 ms in one decode step, before a logit is touched. Here the prompt is
    walked once when the row starts and every later token is a single scatter.

    A row is rebuilt only when it is not an append away from what is held: a new row, a
    prompt length that moved (a turn that continued), or a list that shrank (a rejected
    draft taken back off the end).
    """

    def __init__(self, vocab: int, device):
        if not isinstance(vocab, int) or vocab <= 0:
            raise ValueError("history needs a positive vocabulary size")
        self.vocab, self.device = vocab, device
        self.rows = {}                 # seq -> [seen bool [V], counts f32 [V], covered, prompt_len]

    def of(self, seq: int, tokens, prompt_len: int):
        """(seen, counts) for `tokens`, without walking them again when nothing but the end moved."""
        row = self.rows.get(seq)
        if row is None or row[3] != prompt_len or row[2] > len(tokens):
            row = self._build(seq, tokens, prompt_len)
        elif row[2] < len(tokens):
            self._grow(row, tokens[row[2]:], len(tokens))
        return row[0], row[1]

    def forget(self, seq: int) -> None:
        self.rows.pop(seq, None)

    def _ids(self, ids):
        return torch.tensor(ids, dtype=torch.int64, device=self.device)

    def _build(self, seq: int, tokens, prompt_len: int):
        seen = torch.zeros(self.vocab, dtype=torch.bool, device=self.device)
        counts = torch.zeros(self.vocab, dtype=torch.float32, device=self.device)
        if tokens:
            seen[self._ids(tokens)] = True
        output = tokens[prompt_len:]
        if output:
            counts.index_add_(0, self._ids(output), torch.ones(len(output), device=self.device))
        row = [seen, counts, len(tokens), prompt_len]
        self.rows[seq] = row
        return row

    def _grow(self, row, fresh, covered: int) -> None:
        ids = self._ids(fresh)
        row[0][ids] = True
        row[1].index_add_(0, ids, torch.ones(len(fresh), device=self.device))
        row[2] = covered


def process_logits(logits: torch.Tensor, options: dict, seen: torch.Tensor, counts: torch.Tensor,
                   extra=(), decodable: "int | None" = None, forbid: "torch.Tensor | None" = None,
                   out: "torch.Tensor | None" = None) -> torch.Tensor:
    """One row's raw logits [V] -> the logits the pick is made from: logit_bias, repetition/presence/frequency
    penalties over the row's tokens, the decodable cut and min_tokens' forbidden ids.

    `seen` and `counts` come from `History`. `extra` is this step's drafts before this position: they belong to both
    and are applied as a correction, because rebuilding either for five ids would cost the whole prompt.

    `forbid` is a handful of end tokens, written one by one: a vocabulary of True to say so would cost more than
    the writes. A grammar's mask is the size of the vocabulary and is not applied here at all -- it lands on the
    finished row as packed words, by xgrammar's kernel (base/grammar.StepMasks.apply).

    `out` [V] fp32 receives the result instead of a fresh tensor. A row's positions are written into consecutive
    rows of one buffer that way, which is what lets the grammar mask cross the whole row in a single launch.
    """
    out = logits.to(torch.float32, copy=True) if out is None else out.copy_(logits)
    bias = options.get("logit_bias")
    if bias:
        ids = torch.tensor(list(bias.keys()), device=out.device, dtype=torch.int64)
        out.index_add_(0, ids, torch.tensor(list(bias.values()), device=out.device, dtype=torch.float32))
    drafted = torch.tensor(list(extra), device=out.device, dtype=torch.int64) if len(extra) else None
    rp = options.get("repetition_penalty")
    if rp and rp != 1:
        hit = seen
        if drafted is not None:
            hit = seen.clone()
            hit[drafted] = True
        torch.where(hit, torch.where(out > 0, out / rp, out * rp), out, out=out)
    pres, freq = options.get("presence_penalty"), options.get("frequency_penalty")
    if pres or freq:
        counted = counts
        if drafted is not None:
            counted = counts.clone().index_add_(0, drafted, torch.ones(len(extra), device=out.device))
        out -= (freq or 0.0) * counted + (pres or 0.0) * (counted > 0).float()
    if decodable is not None and out.shape[-1] > decodable:
        out[decodable:] = float("-inf")
    if forbid is not None:
        out[forbid] = float("-inf")
    return out


def distribution(logits: torch.Tensor, temperature: float, top_k: "int | None", top_p: "float | None") -> torch.Tensor:
    """The row's sampling distribution [V] at `temperature` under top-k / top-p (temperature 0 = one-hot argmax).

    One row through `rows`, so there is one definition of what top-k and top-p mean and this is
    not a second one. Callers with a whole step's worth of rows should not come here row by row.
    """
    dev = logits.device
    out = torch.empty(1, logits.shape[-1], dtype=torch.float32, device=dev)
    rows(logits.reshape(1, -1),
         torch.tensor([temperature], dtype=torch.float32, device=dev),
         torch.tensor([top_k or 0], dtype=torch.int32, device=dev),
         torch.tensor([1.0 if top_p is None else top_p], dtype=torch.float32, device=dev),
         torch.zeros(1, dtype=torch.float32, device=dev), None, out)
    return out[0]


def _inverse_cdf(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """[n] picks from [n, V] weights and [n] uniforms: the cumulative walk, no multinomial.

    `torch.multinomial` is a whole kernel to do this and arrives with two host-side sanity checks
    bolted on; vLLM avoids it with V exponentials a row (Gumbel-max). The walk needs one uniform.
    """
    flat = probs.reshape(-1, probs.shape[-1])
    aim = (u.reshape(-1) * flat.sum(-1)).unsqueeze(1)
    return torch.searchsorted(flat.cumsum(-1).contiguous(), aim.contiguous(), right=True) \
        .squeeze(1).clamp_max(flat.shape[-1] - 1)


def draw(probs: torch.Tensor, generator: "torch.Generator | None") -> int:
    u = torch.rand(1, generator=generator, device=probs.device)
    return int(_inverse_cdf(probs.reshape(1, -1), u)[0])


def pick_each(dists, temperature: float, generator: "torch.Generator | None") -> "list[int]":
    """One pick per row of `dists`: the argmax at temperature zero, a draw otherwise.

    The uniforms are drawn together and the walk runs over the whole block, so the crossing to the
    host is one for the step rather than one per draft position per sequence.
    """
    if temperature <= 0:
        return torch.stack([d.argmax() for d in dists]).tolist()
    block = torch.stack(dists)
    return _inverse_cdf(block, torch.rand(len(dists), generator=generator, device=block.device)).tolist()


def speculative_pick(target_probs, draft_ids, draft_probs, generator) -> "tuple[int, list[int]]":
    """Rejection sampling over K drafts (Leviathan/Chen; vLLM's rejection_sample): returns (accepted count, the
    committed tokens = accepted drafts + one recovered or bonus token).

    The engine verifies with `block_verify`, which accepts a longer prefix for the same output
    distribution. This stays as the reference that claim is measured against, the way the lane
    tables keep a reference implementation of every served kernel.

    target_probs: K+1 rows [V] -- the target's distribution at each draft position and the bonus position.
    draft_ids: the K proposed tokens; draft_probs: K rows [V] -- the drafter's distribution each was drawn from
    (zero outside its candidates). u ~ U(0,1) per position from `generator`, identical on every rank."""
    k = len(draft_ids)
    accepted = 0
    # Everything the K comparisons need crosses to the host once: the two probabilities and the K
    # uniforms, in one list. Drawing them all up front is what `speculative_pick_batch` does, so
    # the scalar path and the batch path advance the generator the same way for the same row.
    chosen = torch.tensor(list(draft_ids), device=target_probs.device, dtype=torch.int64)
    at = iota(k, target_probs.device)
    gathered = torch.cat([target_probs[at, chosen], draft_probs[at, chosen],
                          torch.rand(k, generator=generator, device=target_probs.device)]).tolist()
    ps, qs, us = gathered[:k], gathered[k:2 * k], gathered[2 * k:]
    for i, d in enumerate(draft_ids):
        p, q, u = ps[i], qs[i], us[i]
        if q > 0 and u < min(1.0, p / q):
            accepted += 1
            continue
        recovered = (target_probs[i] - draft_probs[i]).clamp_min(0)
        total = float(recovered.sum())
        if total <= 0:
            recovered = target_probs[i]
            total = float(recovered.sum())
        return accepted, list(draft_ids[:accepted]) + [draw(recovered / total, generator)]
    return accepted, list(draft_ids) + [draw(target_probs[k], generator)]


def block_verify(target_probs, draft_ids, draft_probs, generator) -> "tuple[int, list[int]]":
    """Block verification (Sun et al. 2024, arXiv 2403.10444): the same output distribution as
    `speculative_pick`, accepting a longer prefix on average.

    Token-level rejection decides each draft against its own position and stops at the first
    failure. Block verification decides the LENGTH of the accepted prefix instead, using how much
    target mass the whole accepted run has carried so far, so a draft the token-level rule would
    have thrown away can still be kept when the run behind it was good.

    The running quantity is the joint ratio of the accepted prefix, capped at one at every step:

        P_0 = 1 ,  P_i = min(P_{i-1} * p_i(x_i) / q_i(x_i), 1)

    and the threshold at position i is built from the residual mass the next position would leave:

        R_i = sum_x max(P_i * p_{i+1}(x) - q_{i+1}(x), 0)
        h_i = R_i / (R_i + 1 - P_i)      and h_K = P_K at the last position

    The accepted length is the longest prefix whose uniform falls under its threshold. What follows
    it is drawn from the residual max(P_{L-1} * p_L(x) - q_L(x), 0), or from the target itself when
    every draft was accepted and the extra token is the bonus.

    target_probs: K+1 rows [V]; draft_ids: the K proposals; draft_probs: K rows [V].
    """
    k = len(draft_ids)
    if k == 0:
        return 0, [draw(target_probs[0], generator)]
    device = target_probs.device
    chosen = torch.tensor(list(draft_ids), device=device, dtype=torch.int64)
    at = iota(k, device)
    ps = target_probs[at, chosen].tolist()
    qs = draft_probs[at, chosen].tolist()
    carried, ratio = [], 1.0
    for i in range(k):
        ratio = min(ratio * ps[i] / qs[i], 1.0) if qs[i] > 0 else 0.0
        carried.append(ratio)
    thresholds = list(carried)                      # the last position's threshold is P_K itself
    if k > 1:
        ahead = torch.tensor(carried[:-1], device=device, dtype=target_probs.dtype).unsqueeze(-1)
        residual = (ahead * target_probs[1:k] - draft_probs[1:k]).clamp_min(0).sum(-1).tolist()
        for i, mass in enumerate(residual):
            denominator = mass + 1.0 - carried[i]
            thresholds[i] = mass / denominator if denominator > 0 else 1.0
    uniform = torch.rand(k, generator=generator, device=device).tolist()
    accepted = 0
    for i in range(k):
        if uniform[i] <= thresholds[i]:
            accepted = i + 1
    if accepted == k:
        return accepted, list(draft_ids) + [draw(target_probs[k], generator)]
    before = carried[accepted - 1] if accepted else 1.0
    rest = (before * target_probs[accepted] - draft_probs[accepted]).clamp_min(0)
    total = float(rest.sum())
    if total <= 0:
        rest, total = target_probs[accepted], float(target_probs[accepted].sum())
    return accepted, list(draft_ids[:accepted]) + [draw(rest / total, generator)]


def speculative_pick_batch(target_probs: torch.Tensor, drafts: torch.Tensor, draft_probs: torch.Tensor, generator):
    """`speculative_pick` for a whole decode batch on the device, with no host round trip (45차 §23 B3): rows run
    ahead of the host, so their picks must be tensors. target_probs [n, K+1, V]; drafts [n, K]; draft_probs [n, K, V].
    Returns (accepted [n], tokens [n, K+1] with the committed ones first, count [n] = accepted + 1). Draws K uniforms
    per row then one more per row from `generator`, in that order, identically on every rank."""
    n, k1, V = target_probs.shape
    K = k1 - 1
    device = target_probs.device
    rows = torch.arange(n, device=device)
    u = torch.rand(n, K, generator=generator, device=device)
    p_d = target_probs[:, :K].gather(2, drafts.unsqueeze(2)).squeeze(2)                 # the target's mass on each draft
    q_d = draft_probs.gather(2, drafts.unsqueeze(2)).squeeze(2)                          # the drafter's
    accept = (q_d > 0) & (u < (p_d / q_d.clamp_min(1e-30)).clamp_max(1.0))
    accepted = accept.long().cumprod(1).sum(1)                                           # leading accepts only
    at = accepted.clamp_max(K)                                                           # the position drawn afresh: recovered or bonus
    row_p = target_probs[rows, at]
    row_q = torch.where((at < K).unsqueeze(1), draft_probs[rows, at.clamp_max(K - 1)], torch.zeros_like(row_p))
    recovered = (row_p - row_q).clamp_min(0)
    total = recovered.sum(1, keepdim=True)
    recovered = torch.where(total > 0, recovered / total.clamp_min(1e-30), row_p / row_p.sum(1, keepdim=True).clamp_min(1e-30))
    fresh = _inverse_cdf(recovered, torch.rand(n, generator=generator, device=device))
    tokens = torch.cat([drafts, torch.zeros(n, 1, dtype=drafts.dtype, device=device)], 1)
    tokens.scatter_(1, at.unsqueeze(1), fresh.unsqueeze(1))
    return accepted, tokens, accepted + 1


def block_verify_batch(target_probs: torch.Tensor, drafts: torch.Tensor, draft_probs: torch.Tensor, generator):
    """`block_verify` for a whole decode batch on the device, with no host round trip.

    target_probs [n, K+1, V]; drafts [n, K]; draft_probs [n, K, V]. Returns (accepted [n],
    tokens [n, K+1] with the committed ones first, count [n] = accepted + 1). Draws K uniforms per
    row then one more per row, in that order, identically on every rank -- the same shape of
    stream `speculative_pick_batch` drew, so the ranks stay in step.
    """
    n, k1, _ = target_probs.shape
    K = k1 - 1
    device = target_probs.device
    rows = iota(n, device)
    on_draft = target_probs[:, :K].gather(2, drafts.unsqueeze(2)).squeeze(2)
    by_draft = draft_probs.gather(2, drafts.unsqueeze(2)).squeeze(2)
    step = torch.where(by_draft > 0, on_draft / by_draft.clamp_min(1e-30), torch.zeros_like(on_draft))
    # The cap lands at every step, so this scan is not a cumprod. K is the draft width, five here.
    carried = torch.empty_like(step)
    running = torch.ones(n, device=device, dtype=step.dtype)
    for i in range(K):
        running = (running * step[:, i]).clamp_max(1.0)
        carried[:, i] = running
    thresholds = carried.clone()                     # the last position's threshold is its own P_K
    if K > 1:
        ahead = carried[:, : K - 1].unsqueeze(-1)
        mass = (ahead * target_probs[:, 1:K] - draft_probs[:, 1:K]).clamp_min(0).sum(-1)
        denominator = mass + 1.0 - carried[:, : K - 1]
        thresholds[:, : K - 1] = torch.where(denominator > 0, mass / denominator.clamp_min(1e-30),
                                             torch.ones_like(mass))
    u = torch.rand(n, K, generator=generator, device=device)
    reach = iota(K, device).add(1).expand(n, K)
    accepted = torch.where(u <= thresholds, reach, torch.zeros_like(reach)).max(1).values
    at = accepted.clamp_max(K)
    before = torch.where(accepted > 0, carried.gather(1, (accepted - 1).clamp_min(0).unsqueeze(1)).squeeze(1),
                         torch.ones(n, device=device, dtype=carried.dtype))
    row_p = target_probs[rows, at]
    row_q = torch.where((at < K).unsqueeze(1), draft_probs[rows, at.clamp_max(K - 1)], torch.zeros_like(row_p))
    rest = (before.unsqueeze(1) * row_p - row_q).clamp_min(0)
    total = rest.sum(1, keepdim=True)
    rest = torch.where(total > 0, rest / total.clamp_min(1e-30),
                       row_p / row_p.sum(1, keepdim=True).clamp_min(1e-30))
    fresh = _inverse_cdf(rest, torch.rand(n, generator=generator, device=device))
    tokens = torch.cat([drafts, torch.zeros(n, 1, dtype=drafts.dtype, device=device)], 1)
    tokens.scatter_(1, at.unsqueeze(1), fresh.unsqueeze(1))
    return accepted, tokens, accepted + 1


def draft_ceilings(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> "tuple[float, float]":
    """What this draft could have reached, and what its candidate set allowed.

    Acceptance has three ceilings and only the first is closed. The verification rule is now the
    optimal block coupling (Sun et al.), so it cannot leave anything on the table. Above it sits
    `sum_x min(p, q)` -- the most ANY rule can accept from this draft distribution -- and above
    that the target mass the drafter's candidates cover at all, which is what `sel_top_k` fixes.

    Measured acceptance against the first says whether the verifier is doing its job; the first
    against the second says whether the draft distribution is shaped wrong inside its candidates;
    the second says whether the candidate set is too narrow. Without the split, "raise the
    acceptance rate" has three answers and no way to choose.

    Returns (reachable, covered), summed over the positions given. Two passes over the vocabulary,
    so the caller samples rather than asking every step.
    """
    k = draft_probs.shape[-2]
    p = target_probs[..., :k, :]
    return float(torch.minimum(p, draft_probs).sum()), float((p * (draft_probs > 0)).sum())


def commit_batch(picks: torch.Tensor, drafts: torch.Tensor, alive: torch.Tensor, generated: torch.Tensor, limit: torch.Tensor,
                 ends: torch.Tensor, accepted: "torch.Tensor | None" = None):
    """The device half of adapter._commit for a decode batch running ahead of the host (45차 §23 B3).
    picks [n, K+1]: the tokens chosen at each position (greedy: the sampler's; stochastic: speculative_pick_batch's, with
    `accepted` given); drafts [n, K]; alive [n] bool; generated/limit [n]; ends [n, E] end-token ids padded with -1.
    Returns (count [n] tokens committed, done [n], accepted [n], tokens [n, K+1] = picks). A row that is not alive
    commits nothing; the row's remaining limit and its first end token clip the run, as the host does."""
    n, k1 = picks.shape
    K = k1 - 1
    if accepted is None:
        accepted = (picks[:, :K] == drafts).long().cumprod(1).sum(1)
    count = accepted + 1
    room = (limit - generated).clamp_min(0)
    count = torch.minimum(count, room)
    is_end = (picks.unsqueeze(2) == ends.unsqueeze(1)).any(2)                            # [n, K+1]
    positions = torch.arange(k1, device=picks.device).unsqueeze(0)
    first_end = torch.where(is_end & (positions < count.unsqueeze(1)), positions, torch.full_like(positions, k1)).min(1).values
    count = torch.minimum(count, first_end + 1)
    count = torch.where(alive, count, torch.zeros_like(count))
    hit_end = (first_end < k1) & alive
    done = alive & (hit_end | (generated + count >= limit))
    return count, done, torch.minimum(accepted, (count - 1).clamp_min(0)), picks


def top_logprobs(logits: torch.Tensor, chosen: int, k: int) -> "tuple[float, list[tuple[int, float]]]":
    """(log-probability of `chosen`, the k most likely (id, logprob)) from raw logits, vLLM's raw_logprobs mode."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    top = lp.topk(k).indices.tolist() if k > 0 else []
    return float(lp[chosen]), [(i, float(lp[i])) for i in top]
