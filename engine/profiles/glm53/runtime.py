"""GLM composition bound to the common runner, with greedy token generation.

The profile owns token inputs and sampling; the runner owns scheduling and
KV/slot lifetimes. Prefill produces the first token, including immediate
completion at a one-token limit or EOS. Decode batches remaining sequences
into one net step. Drafting is deliberately rejected until a drafter is bound.
This is an eager integration/validation path; it does not claim graph serving.
"""
from __future__ import annotations

import torch

from engine.base.record import Ring
from engine.base.runner import Runner, STEP_RECORD
from engine.profiles.glm53.net import Step


class Glm53Runtime:
    def __init__(self, net, caches, contract, *, eos_ids=(), ring_capacity=64):
        if contract.draft_slots:
            raise ValueError("this runtime has no bound drafter; draft_slots must be zero")
        if net.F != caches.F or tuple(net.layers) != caches.layers:
            raise ValueError("model and caches must declare the same facts and layers")
        if contract.max_running > caches.pool.max_seqs:
            raise ValueError("decode width exceeds cache sequence capacity")
        self.net, self.caches = net, caches
        self.eos_ids = frozenset(eos_ids)
        self._prompts, self._limits, self._slots, self._next = {}, {}, {}, {}
        self._generated = {}
        self.runner = Runner(self, contract, caches.pool, caches.slots,
                             Ring(ring_capacity, STEP_RECORD.size))

    def submit(self, seq: int, ids: torch.Tensor, max_new_tokens: int, *, now=None):
        if seq in self._generated:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        if ids.ndim != 1 or ids.dtype != torch.int64 or ids.numel() == 0:
            raise ValueError("prompt must be a nonempty int64 token vector")
        if ids.device != self.caches.paged.device:
            raise ValueError("prompt and caches must be on the same device")
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if bool(((ids < 0) | (ids >= self.net.F.vocab)).any()):
            raise ValueError("prompt token id is outside the model vocabulary")
        # Detect an impossible context before doing any prefill work.
        if ids.numel() + max_new_tokens - 1 > self.caches.pool.max_blocks_per_seq * self.net.F.block:
            raise ValueError("prompt and generation limit exceed the cache row capacity")
        self._prompts[seq], self._limits[seq], self._generated[seq] = ids, max_new_tokens, []
        try:
            self.runner.submit(seq, ids.numel(), now=now)
        except BaseException:
            for rows in (self._prompts, self._limits, self._generated):
                rows.pop(seq, None)
            raise

    def open(self, seq, slot):
        self.caches.reset_slot(slot)
        self._slots[seq] = slot

    def close(self, seq):
        for rows in (self._prompts, self._limits, self._slots, self._next):
            rows.pop(seq, None)

    def _sample(self, hidden, seqs):
        tokens = self.net.head(hidden).argmax(-1)
        values = tokens.tolist()                   # one host transfer for the entire batch
        done = []
        for i, (seq, token) in enumerate(zip(seqs, values)):
            self._generated[seq].append(token)
            self._next[seq] = tokens[i:i + 1]
            done.append(token in self.eos_ids or len(self._generated[seq]) >= self._limits[seq])
        return done

    def horizon(self, seq):
        return self.caches.pool.tokens[seq] + 1

    def prefill(self, seq, start, tokens, blocks, slot):
        ids = self._prompts[seq][start:start + tokens]
        step = Step.prefill(ids, start, seq, self._slots[seq])
        self.caches.prepare(step)
        hidden = self.net.forward(step, self.caches)
        if start + tokens == self._prompts[seq].numel():
            return self._sample(hidden[-1:], (seq,))[0]
        return False

    def decode(self, seqs, blocks, slots):
        step = Step.decode([(self._next[s], self.caches.pool.tokens[s] - 1, s, slot)
                            for s, slot in zip(seqs, slots)])
        self.caches.prepare(step)
        return self._sample(self.net.forward(step, self.caches), seqs)

    def step(self, *, now=None):
        with torch.inference_mode():
            return self.runner.step(now=now)

    def take_result(self, seq):
        if seq in self._prompts:
            raise ValueError(f"seq {seq} has not finished")
        return tuple(self._generated.pop(seq))
