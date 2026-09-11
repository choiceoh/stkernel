"""GLM-5.3 behind the runner (profile, the adapter): tokens in, tokens out.

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
from math import isfinite

from engine.base.sampler import sample
from engine.profiles.glm53.caches import Glm53Caches
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53.net import Glm53Net, Step


class NullDrafter:
    """No drafts (K=0): a decode step is one token per sequence."""
    k = 0
    aux_layers = ()

    def observe(self, ring, positions, aux) -> None:
        pass

    def propose(self, anchor: int, position: int, ring) -> "list[int]":
        return []


class Glm53Engine:
    def __init__(self, net: Glm53Net, caches: Glm53Caches, F: Facts, drafter=None, max_new: int = 256,
                 eos_ids=(), temperature: float = 0.0, top_p: float = 1.0, seed: int = 0, decodable: "int | None" = None,
                 aux_layers=None):
        self.net, self.caches, self.F = net, caches, F
        self.drafter = drafter or NullDrafter()
        if self.drafter.k > F.spec_k:
            raise ValueError(f"drafter proposes {self.drafter.k} > spec_k {F.spec_k}: the rings are sized for {F.spec_k}")
        self.aux_layers = list(aux_layers) if aux_layers is not None else list(self.drafter.aux_layers)
        self.max_new, self.eos = max_new, set(eos_ids)
        self.temperature, self.top_p = temperature, top_p
        self.decodable = decodable                          # logits past this id are the tokenizer's orphans: masked (as served)
        self.gen = torch.Generator(device=caches.device).manual_seed(seed)
        self.tokens, self.prompt_len, self.ctx, self.slot, self.limits = {}, {}, {}, {}, {}
        self.accepted_total = 0
        self.drafted_total = 0
        self.steps = 0

    # -- the runner's protocol -------------------------------------------------------
    def validate(self, ids, max_new, temperature) -> None:
        if not ids or any(type(t) is not int or not 0 <= t < self.F.vocab for t in ids):
            raise ValueError("prompt token id is outside the model vocabulary")
        if type(max_new) is not int or max_new <= 0:
            raise ValueError("generation limit must be a positive integer")
        if type(temperature) not in (int, float):
            raise ValueError("temperature must be finite and nonnegative")
        try:
            valid = isfinite(temperature) and temperature >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("temperature must be finite and nonnegative")

    def add(self, seq: int, ids: "list[int]", max_new: "int | None" = None, temperature: "float | None" = None) -> None:
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        self.validate(ids, max_new, temperature)
        if seq in self.tokens:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        self.tokens[seq] = list(ids); self.prompt_len[seq] = len(ids)
        self.limits[seq] = (max_new, temperature)

    def forget(self, seq: int) -> None:
        if seq in self.slot:
            raise ValueError(f"seq {seq} is still live")
        for rows in (self.tokens, self.prompt_len, self.limits):
            rows.pop(seq, None)

    def open(self, seq: int, slot: int) -> None:
        self.slot[seq] = slot; self.ctx[seq] = 0
        self.caches.reset_slot(slot)

    def close(self, seq: int) -> None:
        for d in (self.ctx, self.slot):
            d.pop(seq, None)

    def extend(self, seq: int, ids: "list[int]", max_new: "int | None" = None, temperature: "float | None" = None) -> int:
        """A new turn: more prompt tokens on a conversation the caches still hold.
        Returns the tokens to prefill -- the last sampled token (never fed) and
        the new ones -- so `generated` counts this turn only from here on."""
        max_new = self.max_new if max_new is None else max_new
        temperature = self.temperature if temperature is None else temperature
        self.validate(ids, max_new, temperature)
        self.tokens[seq] += list(ids); self.prompt_len[seq] = len(self.tokens[seq])
        self.limits[seq] = (max_new, temperature)
        return len(self.tokens[seq]) - self.ctx[seq]

    def extension_tokens(self, seq: int, ids) -> int:
        """Inspect the next turn's prefill size before reserving its budget."""
        return len(self.tokens[seq]) + len(ids) - self.ctx[seq]

    def horizon(self, seq: int) -> int:
        return self.ctx[seq] + 1 + self.drafter.k

    def context(self, seq: int) -> int:
        return self.ctx[seq]

    def generated(self, seq: int) -> "list[int]":
        return self.tokens[seq][self.prompt_len[seq]:]

    def _sample(self, logits: torch.Tensor, temps: "list[float]") -> torch.Tensor:
        if self.decodable is not None and logits.shape[-1] > self.decodable:
            logits = logits.clone(); logits[:, self.decodable:] = float("-inf")
        t = torch.tensor(temps, dtype=torch.float32, device=logits.device)
        p = torch.full((logits.shape[0],), self.top_p, device=logits.device)
        return sample(logits, t, p, self.gen)

    def _forward(self, step: Step):
        self.caches.prepare(step)
        if self.drafter.k:
            return self.net.forward(step, self.caches, aux_layers=self.aux_layers)
        return self.net.forward(step, self.caches), None

    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int) -> bool:
        ids = torch.tensor(self.tokens[seq][start: start + tokens], dtype=torch.int64, device=self.caches.device)
        h, aux = self._forward(Step.prefill(ids, start, seq, slot))
        self.ctx[seq] = start + tokens
        if aux is not None:                                                 # every prompt token is context for the drafter
            self.drafter.observe(self.caches.draft_ring(slot), torch.arange(start, start + tokens, device=ids.device), aux)
        if self.ctx[seq] == self.prompt_len[seq]:                         # the prompt is in: the first token comes from its last position
            first = self._sample(self.net.head(h[-1:]), [self.limits[seq][1]])
            self.tokens[seq].append(int(first.item()))
        self.steps += 1
        generated = self.generated(seq)
        return bool(generated) and (generated[-1] in self.eos or len(generated) >= self.limits[seq][0])

    def decode(self, seqs, blocks, slots) -> "list[bool]":
        chunks, drafts = [], {}
        for seq, slot in zip(seqs, slots):
            drafts[seq] = self.drafter.propose(self.tokens[seq][-1], self.ctx[seq], self.caches.draft_ring(slot) if self.drafter.k else None)
            ids = [self.tokens[seq][-1]] + drafts[seq]
            chunks.append((torch.tensor(ids, dtype=torch.int64, device=self.caches.device), self.ctx[seq], seq, slot))
        step = Step.decode(chunks)
        h, aux = self._forward(step)
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
            new = new[:max(0, self.limits[s.seq][0] - len(self.generated(s.seq)))]
            for i, token in enumerate(new):
                if token in self.eos:
                    new = new[:i + 1]
                    break
            committed = len(new)                            # clipped tokens must not enter the next turn's context
            if aux is not None:
                rows = slice(s.start, s.start + committed)
                self.drafter.observe(self.caches.draft_ring(s.slot), torch.arange(s.ctx, s.ctx + committed, device=h.device), aux[rows])
            self.tokens[s.seq] += new
            self.ctx[s.seq] += committed
            self.accepted_total += min(accepted, committed)
            self.drafted_total += len(drafts[s.seq])
            done = any(t in self.eos for t in new) or len(self.generated(s.seq)) >= self.limits[s.seq][0]
            finished.append(done)
        self.steps += 1
        return finished
