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


def process_logits(logits: torch.Tensor, options: dict, prompt_ids, generated_ids, decodable: "int | None" = None,
                   mask: "torch.Tensor | None" = None) -> torch.Tensor:
    """One row's raw logits [V] fp32 -> the logits the pick is made from: logit_bias, repetition/presence/frequency
    penalties over the row's tokens, the decodable cut and an optional grammar mask (True = allowed)."""
    out = logits.float().clone()
    bias = options.get("logit_bias")
    if bias:
        ids = torch.tensor(list(bias.keys()), device=out.device, dtype=torch.int64)
        out.index_add_(0, ids, torch.tensor(list(bias.values()), device=out.device, dtype=torch.float32))
    rp = options.get("repetition_penalty")
    if rp and rp != 1 and (prompt_ids or generated_ids):
        seen = torch.tensor(sorted(set(prompt_ids) | set(generated_ids)), device=out.device, dtype=torch.int64)
        vals = out[seen]
        out[seen] = torch.where(vals > 0, vals / rp, vals * rp)
    pres, freq = options.get("presence_penalty"), options.get("frequency_penalty")
    if (pres or freq) and generated_ids:
        gen = torch.tensor(generated_ids, device=out.device, dtype=torch.int64)
        counts = torch.zeros_like(out).index_add_(0, gen, torch.ones(len(generated_ids), device=out.device))
        out -= (freq or 0.0) * counts + (pres or 0.0) * (counts > 0).float()
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


def top_logprobs(logits: torch.Tensor, chosen: int, k: int) -> "tuple[float, list[tuple[int, float]]]":
    """(log-probability of `chosen`, the k most likely (id, logprob)) from raw logits, vLLM's raw_logprobs mode."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    top = lp.topk(k).indices.tolist() if k > 0 else []
    return float(lp[chosen]), [(i, float(lp[i])) for i in top]
