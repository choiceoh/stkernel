"""Sampling over flat logits (base): greedy, temperature, top-p -- and a seed
that makes a step replayable (D12).

Inputs are flat: logits [N, vocab] for the N sequences of a decode step, and
per-row float arrays for temperature and top-p. No per-request objects
(CHARTER I2). The generator is explicit so a recorded step can be re-run
with the same seed and produce the same tokens -- that is what makes the
death dump a replay and not a log.
"""
from __future__ import annotations

import torch


def sample(logits: torch.Tensor, temperature: torch.Tensor, top_p: torch.Tensor,
           generator: "torch.Generator | None" = None, *, top_p_enabled=None) -> torch.Tensor:
    """[N] token ids. temperature 0 means greedy for that row.

    A graph caller may supply the known top-p policy without reading a
    device predicate; ordinary callers retain the per-row tensor policy.
    """
    greedy = temperature <= 0
    scaled = logits.float() / temperature.clamp_min(1e-5).unsqueeze(-1)
    probs = torch.softmax(scaled, dim=-1)
    use_nucleus = bool((top_p < 1).any()) if top_p_enabled is None else top_p_enabled
    if use_nucleus:
        srt, idx = probs.sort(dim=-1, descending=True)
        cum = srt.cumsum(dim=-1)
        # keep the smallest prefix whose mass reaches top_p (always at least one)
        keep = (cum - srt) < top_p.unsqueeze(-1)
        srt = srt * keep
        srt = srt / srt.sum(dim=-1, keepdim=True)
        picked = torch.multinomial(srt, 1, generator=generator).squeeze(-1)
        sampled = idx.gather(-1, picked.unsqueeze(-1)).squeeze(-1)
    else:
        sampled = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return torch.where(greedy, logits.argmax(dim=-1), sampled)


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
    # temperature 1, top-p 1 must match torch.multinomial on the softmax exactly for the same generator state
    g3 = torch.Generator(device=dev).manual_seed(7); g4 = torch.Generator(device=dev).manual_seed(7)
    ours = sample(logits[2:3], t[2:3], p[2:3], g3)
    ref = torch.multinomial(torch.softmax(logits[2:3].float(), -1), 1, generator=g4).squeeze(-1)
    assert torch.equal(ours, ref)
    print("  sampler: replayable by seed, greedy at T=0, nucleus respected, == torch.multinomial OK")


if __name__ == "__main__":
    _selfcheck()


# ---- the OpenAI-dialect options a row may carry (45차 §23 A3/A4/B4) ----------------------------------------------
# A row with any of these leaves the captured sampler: its logits are gathered whole and processed here, on every
# rank identically (same generator seeds, same order), so the picks agree without a message.

OPTION_KEYS = ("top_p", "top_k", "seed", "presence_penalty", "frequency_penalty", "repetition_penalty",
               "logit_bias", "stop_token_ids", "logprobs", "grammar")


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


def needs_rich_sampler(options: dict, temperature: float, drafts: bool) -> bool:
    """Whether a row's logits must be processed here rather than by the captured greedy/top-p sampler:
    any option beyond temperature/top_p, or a stochastic row with drafts (rejection sampling needs the probabilities)."""
    if any(options.get(k) is not None for k in ("top_p", "top_k", "seed", "presence_penalty", "frequency_penalty",
                                                 "repetition_penalty", "logit_bias", "logprobs", "grammar")):
        return True                                  # (the captured sampler was recorded without a nucleus branch: top_p is rich)
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
                   extra=(), decodable: "int | None" = None, mask: "torch.Tensor | None" = None) -> torch.Tensor:
    """One row's raw logits [V] fp32 -> the logits the pick is made from: logit_bias, repetition/presence/frequency
    penalties over the row's tokens, the decodable cut and an optional grammar mask (True = allowed).

    `seen` and `counts` come from `History`. `extra` is this step's drafts before this position: they belong to both
    and are applied as a correction, because rebuilding either for five ids would cost the whole prompt.
    """
    out = logits.float().clone()
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
        out = torch.where(hit, torch.where(out > 0, out / rp, out * rp), out)
    pres, freq = options.get("presence_penalty"), options.get("frequency_penalty")
    if pres or freq:
        counted = counts
        if drafted is not None:
            counted = counts.clone().index_add_(0, drafted, torch.ones(len(extra), device=out.device))
        out -= (freq or 0.0) * counted + (pres or 0.0) * (counted > 0).float()
    if decodable is not None and out.shape[-1] > decodable:
        out[decodable:] = float("-inf")
    if mask is not None:
        out = out.masked_fill(~mask, float("-inf"))
    return out


def distribution(logits: torch.Tensor, temperature: float, top_k: "int | None", top_p: "float | None") -> torch.Tensor:
    """The row's sampling distribution [V] at `temperature` under top-k / top-p (temperature 0 = one-hot argmax)."""
    if temperature <= 0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    scaled = logits / temperature
    if top_k is not None and 0 < top_k < scaled.shape[-1]:
        kth = scaled.topk(top_k).values[-1]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    probs = torch.softmax(scaled, dim=-1)
    if top_p is not None and top_p < 1:
        srt, idx = probs.sort(descending=True)
        cum = srt.cumsum(-1)
        keep = (cum - srt) < top_p
        srt = srt * keep
        probs = torch.zeros_like(probs).scatter_(0, idx, srt / srt.sum())
    return probs


def draw(probs: torch.Tensor, generator: "torch.Generator | None") -> int:
    return int(torch.multinomial(probs, 1, generator=generator).item())


def speculative_pick(target_probs, draft_ids, draft_probs, generator) -> "tuple[int, list[int]]":
    """Rejection sampling over K drafts (Leviathan/Chen; vLLM's rejection_sample): returns (accepted count, the
    committed tokens = accepted drafts + one recovered or bonus token).

    target_probs: K+1 rows [V] -- the target's distribution at each draft position and the bonus position.
    draft_ids: the K proposed tokens; draft_probs: K rows [V] -- the drafter's distribution each was drawn from
    (zero outside its candidates). u ~ U(0,1) per position from `generator`, identical on every rank."""
    k = len(draft_ids)
    accepted = 0
    for i, d in enumerate(draft_ids):
        p, q = float(target_probs[i][d]), float(draft_probs[i][d])
        u = float(torch.rand((), generator=generator, device=target_probs.device))
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


def speculative_pick_batch(target_probs: torch.Tensor, drafts: torch.Tensor, draft_probs: torch.Tensor, generator):
    """`speculative_pick` for a whole decode batch on the device, with no host round trip (45차 §23 B3): rows run
    ahead of the host, so their picks must be tensors. target_probs [n, K+1, V]; drafts [n, K]; draft_probs [n, K, V].
    Returns (accepted [n], tokens [n, K+1] with the committed ones first, count [n] = accepted + 1). Draws K uniforms
    per row then one multinomial per row from `generator`, in that order, identically on every rank."""
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
    fresh = torch.multinomial(recovered, 1, generator=generator).squeeze(1)
    tokens = torch.cat([drafts, torch.zeros(n, 1, dtype=drafts.dtype, device=device)], 1)
    tokens.scatter_(1, at.unsqueeze(1), fresh.unsqueeze(1))
    return accepted, tokens, accepted + 1


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
