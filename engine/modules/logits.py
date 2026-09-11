"""Vocab-parallel embedding, LM head, and the logits processor (module).

    VocabParallelEmbedding(vocab, hidden, prefix="")      rows split by rank; out-of-range ids
                                                          contribute zeros and the all-reduce sums
    ParallelLMHead(vocab, hidden, quant_config=None, prefix="")   same rows, used as a projection
    LogitsProcessor(vocab, scale=1.0)   forward(lm_head, hidden) -> full-vocab logits (all-gather)

glm53_model's `Fp8HeadLogitsProcessor` subclasses the last with an fp8 head
option; it plugs in here by overriding `_project`.
"""
from __future__ import annotations

import torch
from torch import nn

from engine.modules.linear import _Identity


class VocabParallelEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden, prefix="", comm=None):
        super().__init__()
        self.comm = comm or _Identity()
        self.tp, self.rank = self.comm.world_size, self.comm.rank
        self.vocab_size, self.hidden = vocab_size, hidden
        self.part = -(-vocab_size // self.tp)
        self.start = self.rank * self.part
        self.end = min(self.start + self.part, vocab_size)
        self.weight = nn.Parameter(torch.empty(self.part, hidden, dtype=torch.get_default_dtype()), requires_grad=False)

    def load(self, full):
        n = self.end - self.start
        self.weight.data[:n].copy_(full[self.start:self.end])
        if n < self.part:
            self.weight.data[n:].zero_()

    def forward(self, ids):
        mask = (ids < self.start) | (ids >= self.end)
        local = (ids - self.start).masked_fill(mask, 0)
        out = torch.nn.functional.embedding(local, self.weight)
        out = out.masked_fill(mask.unsqueeze(-1), 0)
        return self.comm.all_reduce(out) if self.tp > 1 else out


class ParallelLMHead(VocabParallelEmbedding):
    def __init__(self, vocab_size, hidden, quant_config=None, prefix="", comm=None):
        super().__init__(vocab_size, hidden, prefix, comm)
        self.quant_config = quant_config

    def forward(self, *_):
        raise RuntimeError("an LM head is applied through LogitsProcessor, not called")


class LogitsProcessor(nn.Module):
    def __init__(self, vocab_size: int, scale: float = 1.0):
        super().__init__()
        self.vocab_size, self.scale = vocab_size, scale

    def _project(self, lm_head, hidden):
        return torch.nn.functional.linear(hidden, lm_head.weight)          # [.., part]

    def forward(self, lm_head, hidden):
        local = self._project(lm_head, hidden)
        full = lm_head.comm.all_gather(local, dim=-1) if lm_head.tp > 1 else local
        full = full[..., : self.vocab_size]
        return full * self.scale if self.scale != 1.0 else full


def _selfcheck() -> None:
    torch.manual_seed(0); dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.bfloat16)

    class FakeComm:
        def __init__(self, rank): self.world_size, self.rank = 2, rank
        def all_reduce(self, t): return t
        def all_gather(self, t, dim=-1): return t

    V, H = 100, 32
    E = torch.randn(V, H, device=dev)
    ids = torch.tensor([[3, 57, 99], [0, 50, 49]], device=dev)
    with torch.device(dev):
        embs = [VocabParallelEmbedding(V, H, comm=FakeComm(r)) for r in (0, 1)]
    for e in embs: e.load(E)
    got = embs[0](ids) + embs[1](ids)                    # the all-reduce, done by hand
    assert torch.equal(got, torch.nn.functional.embedding(ids, E)), "vocab halves sum to the full lookup"
    with torch.device(dev):
        heads = [ParallelLMHead(V, H, comm=FakeComm(r)) for r in (0, 1)]; lp = LogitsProcessor(V)
    for h in heads: h.load(E)
    hid = torch.randn(2, H, device=dev)
    logits = torch.cat([lp(h, hid) for h in heads], dim=-1)[..., :V]   # the all-gather, by hand
    assert torch.allclose(logits, torch.nn.functional.linear(hid, E), atol=1e-2, rtol=1e-2)
    print("  logits: vocab-parallel embedding sums, LM head halves gather to full logits OK")


if __name__ == "__main__":
    _selfcheck()
