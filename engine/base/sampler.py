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
           generator: "torch.Generator | None" = None) -> torch.Tensor:
    """[N] token ids. temperature 0 means greedy for that row."""
    greedy = temperature <= 0
    scaled = logits.float() / temperature.clamp_min(1e-5).unsqueeze(-1)
    probs = torch.softmax(scaled, dim=-1)
    if (top_p < 1).any():
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
