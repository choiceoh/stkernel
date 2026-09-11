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
from engine.profiles.glm53.net import Glm53Net, Segment, Step


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
        self.decode_graphs = None
        self.sampling_graphs = None
        self.memory = None
        self.prefill_chunk = None

    def capture_decode(self, max_seqs: int) -> None:
        """Bind the fleet's finite target decode graphs before admitting work."""
        from engine.profiles.glm53.decode_graphs import Glm53DecodeGraphs
        if self.tokens:
            raise ValueError("capture must finish before requests are admitted")
        if self.decode_graphs is not None:
            raise ValueError("decode graphs are already prepared")
        try:
            if self.memory is not None:
                self._warmup_prefill_memory()
            self.decode_graphs = Glm53DecodeGraphs(self.net, self.caches, max_seqs,
                                                  self.drafter.k + 1, self.aux_layers, memory=self.memory)
            if self.drafter.k:
                self.drafter.capture_decode(self.caches, memory=self.memory)
            from engine.profiles.glm53.decode_graphs import SamplingGraphs
            self.sampling_graphs = SamplingGraphs(self.decode_graphs, self.gen, self.decodable, self.top_p)
            if self.memory is not None:
                self.memory.checkpoint("ready")
                self.memory.ready = True
        except BaseException:
            self.close_decode()
            raise

    def _warmup_prefill_memory(self):
        """Exercise the largest legal prefill at both ends of the KV capacity.

        Inputs are synthetic; this qualifies memory preparation, not quality.
        Unseen tail shapes remain subject to the same allocator byte ceiling.
        """
        if self.prefill_chunk is None:
            raise ValueError("full-model memory preparation requires the scheduler's prefill chunk")
        caches = self.caches
        if caches.pool.rows_in_use or any(owner >= 0 for owner in caches.slots.owner[1:]):
            raise ValueError("memory preparation requires empty request and state slots")
        capacity = caches.pool.num_blocks * self.F.block
        length = min(self.prefill_chunk, capacity)
        slot = caches.slots.take(0)
        try:
            caches.pool.reserve(0, capacity)
            for context in sorted({0, capacity-length}):
                self.memory.checkpoint(f"prefill/{length}/{context}/before")
                ids = torch.zeros(length, device=caches.device, dtype=torch.int64)
                step = Step.prefill(ids, context, 0, slot)
                h, aux = self._forward(step)
                self.net.head(h[-1:])
                if aux is not None:
                    self.drafter.observe(caches.draft_ring(slot),
                                         torch.arange(context, context+length, device=caches.device), aux)
                del h, aux, step, ids
                self.memory.checkpoint(f"prefill/{length}/{context}/prepared")
        finally:
            caches.pool.release(0)
            caches.slots.give(slot)
            caches.reset()

    def close_decode(self):
        if self.sampling_graphs is not None:
            self.sampling_graphs.close()
            self.sampling_graphs = None
        if self.decode_graphs is not None:
            self.decode_graphs.graphs.close()
            self.decode_graphs = None
        if self.drafter.k and self.drafter.decode_graphs is not None:
            self.drafter.decode_graphs.proposals.close()
            self.drafter.decode_graphs.observations.close()
            self.drafter.decode_graphs = None
        if self.memory is not None:
            self.memory.close()

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

    # -- parking (D16): the host side of a conversation travels as a record, the slot's bytes with the tier --
    def park(self, seq: int) -> dict:
        """Close the row and hand back everything the host held for it."""
        if seq not in self.slot:
            raise ValueError(f"seq {seq} is not open")
        record = {"context": self.ctx[seq], "pending": len(self.tokens[seq]) - self.ctx[seq],
                  "tokens": list(self.tokens[seq]), "prompt_len": self.prompt_len[seq],
                  "limits": [self.limits[seq][0], self.limits[seq][1]]}
        self.close(seq)
        self.forget(seq)
        return record

    def resume(self, seq: int, slot: int, record: dict) -> None:
        """Reopen the row in `slot` from a record; the slot's bytes were restored by the tier, so no reset."""
        if seq in self.tokens or seq in self.slot:
            raise ValueError(f"seq {seq} is live or has an uncollected result")
        self.tokens[seq] = list(record["tokens"]); self.prompt_len[seq] = int(record["prompt_len"])
        self.limits[seq] = (int(record["limits"][0]), float(record["limits"][1]))
        self.slot[seq] = slot; self.ctx[seq] = int(record["context"])

    def state_bytes(self, slot: int):
        return self.caches.slot_bytes(slot)

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

    def _generated_count(self, seq: int) -> int:
        return len(self.tokens[seq]) - self.prompt_len[seq]

    def _sample(self, logits: torch.Tensor, temps: "list[float]") -> torch.Tensor:
        # Temperatures already live on the host: no device predicate or random
        # draw is needed for an entirely greedy step. Such steps leave the RNG
        # untouched; stochastic/mixed steps retain the base sampler's draws.
        if all(t <= 0 for t in temps):
            return logits[:, :self.decodable].argmax(dim=-1)
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

    def _sample_hidden(self, hidden, temps):
        if all(t <= 0 for t in temps):
            return self.net.head_tokens(hidden, self.decodable)
        return self._sample(self.net.head(hidden), temps)

    def prefill(self, seq: int, start: int, tokens: int, blocks, slot: int) -> bool:
        ids = torch.tensor(self.tokens[seq][start: start + tokens], dtype=torch.int64, device=self.caches.device)
        h, aux = self._forward(Step.prefill(ids, start, seq, slot))
        self.ctx[seq] = start + tokens
        if aux is not None:                                                 # every prompt token is context for the drafter
            self.drafter.observe(self.caches.draft_ring(slot), torch.arange(start, start + tokens, device=ids.device), aux)
        if self.ctx[seq] == self.prompt_len[seq]:                         # the prompt is in: the first token comes from its last position
            first = self._sample_hidden(h[-1:], [self.limits[seq][1]])
            self.tokens[seq].append(int(first.item()))
        self.steps += 1
        generated = self._generated_count(seq)
        return generated > 0 and (self.tokens[seq][-1] in self.eos or generated >= self.limits[seq][0])

    def decode(self, seqs, blocks, slots) -> "list[bool]":
        flat, segments, drafts = [], [], {}
        for seq, slot in zip(seqs, slots):
            drafts[seq] = self.drafter.propose(self.tokens[seq][-1], self.ctx[seq], self.caches.draft_ring(slot) if self.drafter.k else None)
            ids = [self.tokens[seq][-1]] + drafts[seq]
            segments.append(Segment(seq, slot, self.ctx[seq], len(flat), len(ids)))
            flat.extend(ids)
        step = Step(torch.tensor(flat, dtype=torch.int64, device=self.caches.device), tuple(segments))
        temps = [self.limits[s.seq][1] for s in step.segments for _ in range(s.length)]
        if self.decode_graphs is None:
            h, aux = self._forward(step)
            sampled = self._sample_hidden(h, temps).tolist()
        else:
            h, aux, logits = self.decode_graphs.run(step)
            sampled = self.sampling_graphs.run(self.decode_graphs.shape(step), temps).tolist()
        finished = []
        for s in step.segments:
            picks = sampled[s.start: s.start + s.length]
            accepted = 0
            for d, got in zip(drafts[s.seq], picks):
                if d != got:
                    break
                accepted += 1
            new = picks[: accepted + 1]                                    # the accepted drafts' confirmations, then the correction
            new = new[:max(0, self.limits[s.seq][0] - self._generated_count(s.seq))]
            for i, token in enumerate(new):
                if token in self.eos:
                    new = new[:i + 1]
                    break
            committed = len(new)                            # clipped tokens must not enter the next turn's context
            if aux is not None:
                rows = slice(s.start, s.start + committed)
                observe = self.drafter.observe_decode if self.decode_graphs is not None else self.drafter.observe
                observe(self.caches.draft_ring(s.slot), torch.arange(s.ctx, s.ctx + committed, device=h.device), aux[rows])
            self.tokens[s.seq] += new
            self.ctx[s.seq] += committed
            self.accepted_total += min(accepted, committed)
            self.drafted_total += len(drafts[s.seq])
            done = any(t in self.eos for t in new) or self._generated_count(s.seq) >= self.limits[s.seq][0]
            finished.append(done)
        self.steps += 1
        return finished
