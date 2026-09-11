"""GLM-5.3 behind the runner (profile): tokens in, tokens out.

base/runner knows a four-method model and nothing else; this is that model
for GLM-5.3. It owns the token buffers (prompt + generated per sequence),
turns the runner's calls into `net.Step`s over the profile's caches, and
samples (base/sampler, seeded -> replayable, D12). Draft tokens come from a
`Drafter` -- `NullDrafter` proposes nothing (K=0); DFlash2 is the fleet's
drafter and plugs in here with the same two calls.

Verification is rejection by position: the step feeds [last token] + K
drafts at ctx..ctx+K, samples at every position, accepts drafts while the
sample agrees, and appends accepted+1 tokens. The caches were written for
all K+1 positions; the next step overwrites what was rejected (net.py).
"""
from __future__ import annotations

import torch

from engine.base.sampler import sample
from engine.profiles.glm53.caches import Glm53Caches
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53.net import Glm53Net, Step


class NullDrafter:
    k = 0

    def propose(self, seq: int, tokens: "list[int]") -> "list[int]":
        return []


class Glm53Engine:
    def __init__(self, net: Glm53Net, caches: Glm53Caches, F: Facts, drafter=None, max_new: int = 256,
                 eos_ids=(), temperature: float = 0.0, top_p: float = 1.0, seed: int = 0):
        self.net, self.caches, self.F = net, caches, F
        self.drafter = drafter or NullDrafter()
        if self.drafter.k > F.spec_k:
            raise ValueError(f"drafter proposes {self.drafter.k} > spec_k {F.spec_k}: the rings are sized for {F.spec_k}")
        self.max_new, self.eos = max_new, set(eos_ids)
        self.temperature, self.top_p = temperature, top_p
        self.gen = torch.Generator(device=caches.device).manual_seed(seed)
        self.tokens, self.prompt_len, self.ctx, self.slot, self.limits = {}, {}, {}, {}, {}
        self.accepted_total = 0
        self.steps = 0

    # -- the runner's protocol -------------------------------------------------------
    def add(self, seq: int, ids: "list[int]", max_new: "int | None" = None, temperature: "float | None" = None) -> None:
        self.tokens[seq] = list(ids); self.prompt_len[seq] = len(ids)
        self.limits[seq] = (self.max_new if max_new is None else max_new, self.temperature if temperature is None else temperature)

    def open(self, seq: int, slot: int) -> None:
        self.slot[seq] = slot; self.ctx[seq] = 0
        self.caches.clear_slot(slot)

    def close(self, seq: int) -> None:
        for d in (self.ctx, self.slot):
            d.pop(seq, None)

    def horizon(self, seq: int) -> int:
        return self.ctx[seq] + 1 + self.drafter.k

    def generated(self, seq: int) -> "list[int]":
        return self.tokens[seq][self.prompt_len[seq]:]

    def _sample(self, logits: torch.Tensor, temps: "list[float]") -> torch.Tensor:
        t = torch.tensor(temps, dtype=torch.float32, device=logits.device)
        p = torch.full((logits.shape[0],), self.top_p, device=logits.device)
        return sample(logits, t, p, self.gen)

    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int) -> None:
        self.caches.sync_row(seq)
        ids = torch.tensor(self.tokens[seq][start: start + tokens], dtype=torch.int64, device=self.caches.device)
        h = self.net.forward(Step.prefill(ids, start, seq, slot), self.caches)
        self.ctx[seq] = start + tokens
        if self.ctx[seq] == self.prompt_len[seq]:                         # the prompt is in: the first token comes from its last position
            first = self._sample(self.net.head(h[-1:]), [self.limits[seq][1]])
            self.tokens[seq].append(int(first.item()))
        self.steps += 1

    def decode(self, seqs, blocks, slots) -> "list[bool]":
        chunks, drafts = [], {}
        for seq, slot in zip(seqs, slots):
            self.caches.sync_row(seq)
            drafts[seq] = self.drafter.propose(seq, self.tokens[seq])
            ids = [self.tokens[seq][-1]] + drafts[seq]
            chunks.append((torch.tensor(ids, dtype=torch.int64, device=self.caches.device), self.ctx[seq], seq, slot))
        step = Step.decode(chunks)
        h = self.net.forward(step, self.caches)
        temps = [self.limits[s.seq][1] for s in step.segments for _ in range(s.length)]
        sampled = self._sample(self.net.head(h), temps).tolist()
        finished = []
        for s in step.segments:
            picks = sampled[s.start: s.start + s.length]
            accepted = 0
            for d, got in zip(drafts[s.seq], picks):
                if d != got:
                    break
                accepted += 1
            new = picks[: accepted + 1]                                    # the accepted drafts' confirmations, then the correction
            self.tokens[s.seq] += new
            self.ctx[s.seq] += accepted + 1
            self.accepted_total += accepted
            done = any(t in self.eos for t in new) or len(self.generated(s.seq)) >= self.limits[s.seq][0]
            finished.append(done)
        self.steps += 1
        return finished
