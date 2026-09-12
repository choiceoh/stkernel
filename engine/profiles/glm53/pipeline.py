"""Decode steps ahead of the host (profile): the device commits, observes and proposes; the host reads back later.

45차 §23 B3 -- what vLLM's async scheduling buys. The synchronous step (adapter.decode) reads the sampler back
before it can build the next step's inputs, so the device idles through the host's commit, the drafter's proposal
launch, the loop's broadcast and the scheduler. Here the chain that feeds one decode step from the last stays on
the device:

    target graph -> sampler -> commit (base/sampler.commit_batch) -> masked observe -> proposal -> next step's ids

and only the OUTCOME crosses to the host, into a pinned buffer behind a CUDA event, read at `Pending.resolve`
(base/runner) while the next step already runs. Rows finish on the device too: a finished row's later steps are
inert -- its state-slot input is redirected to the null slot so nothing it writes reaches its rings (the runner
sees it finish one step late and drops that ghost's result).

Which rows may run ahead: greedy rows and rows with only a temperature / top_p (the batch's device rejection
sampling, base/sampler.speculative_pick_batch); rows with penalties, logit_bias, seeds, logprobs, grammars or a
pending min_tokens keep the synchronous path, and the runner drains this one before them (adapter.async_ready).
Every device-side draw comes from the engine's generator in the same order on every rank.
"""
from __future__ import annotations

import torch

from engine.base.sampler import commit_batch, speculative_pick_batch
from engine.profiles.glm53.net import Segment, Step


def distribution_batch(logits: torch.Tensor, temps: torch.Tensor, top_p: torch.Tensor, nucleus: bool) -> torch.Tensor:
    """base/sampler.distribution for every row at once: [m, V] fp32 probabilities -- a one-hot argmax where the row's
    temperature is 0, the nucleus-truncated softmax otherwise. `nucleus` is decided on the host (no device predicate)."""
    greedy = temps <= 0
    scaled = logits / temps.clamp_min(1e-5).unsqueeze(1)
    probs = torch.softmax(scaled, dim=-1)
    if nucleus:
        srt, idx = probs.sort(dim=-1, descending=True)
        cum = srt.cumsum(dim=-1)
        keep = (cum - srt) < top_p.unsqueeze(1)
        srt = srt * keep
        probs = torch.zeros_like(probs).scatter_(1, idx, srt / srt.sum(dim=-1, keepdim=True).clamp_min(1e-30))
    onehot = torch.zeros_like(probs).scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
    return torch.where(greedy.unsqueeze(1), onehot, probs)


class Pending:
    def __init__(self, pipeline, slot: int, seqs, event):
        self.pipeline, self.slot, self.seqs, self.event = pipeline, slot, tuple(seqs), event

    def resolve(self) -> "list[bool]":
        return self.pipeline.resolve(self)


class AsyncDecode:
    def __init__(self, engine, depth: int = 2):
        self.e = engine
        self.depth = depth
        self.batch = ()                                  # the rows the device buffers describe, in order
        self.stale = True                                # host state moved without this chain: rebuild from the host
        self.pending = []                                # Pending, oldest first
        self.buf = None
        K = engine.drafter.k
        self.t = K + 1
        n_max = engine.caches.pool.max_seqs
        pin = engine.caches.device.type == "cuda"
        # the readback lanes: one pinned row set per step that may be in flight
        self.host = [dict(tokens=torch.empty(n_max, self.t, dtype=torch.int64, pin_memory=pin),
                          count=torch.empty(n_max, dtype=torch.int64, pin_memory=pin),
                          done=torch.empty(n_max, dtype=torch.bool, pin_memory=pin),
                          accepted=torch.empty(n_max, dtype=torch.int64, pin_memory=pin)) for _ in range(depth)]
        self.free = list(range(depth))
        self._slots = []

    # -- building the device view of a batch --------------------------------------------------------
    def _build(self, seqs, slots) -> None:
        """From the host's view, which is exact when nothing is in flight: every row's last token, context, limit,
        end tokens and sampling temperature, plus the first proposals (the synchronous step's first half)."""
        e, K, t = self.e, self.e.drafter.k, self.t
        dev = e.caches.device
        n = len(seqs)
        temps = [e.limits[s][1] for s in seqs]
        top_p = [float(e.options.get(s, {}).get("top_p", e.top_p)) for s in seqs]
        ends = [sorted(e.ends.get(s, e.eos)) for s in seqs]
        width = max(1, max(len(x) for x in ends))
        b = dict(
            seqs=torch.tensor(seqs, dtype=torch.int64, device=dev),
            real_slot=torch.tensor(slots, dtype=torch.int64, device=dev),
            ctx=torch.tensor([e.ctx[s] for s in seqs], dtype=torch.int64, device=dev),
            generated=torch.tensor([e._generated_count(s) for s in seqs], dtype=torch.int64, device=dev),
            limit=torch.tensor([e.limits[s][0] for s in seqs], dtype=torch.int64, device=dev),
            ends=torch.tensor([x + [-1] * (width - len(x)) for x in ends], dtype=torch.int64, device=dev),
            temps=torch.tensor(temps, dtype=torch.float32, device=dev),
            top_p=torch.tensor(top_p, dtype=torch.float32, device=dev),
            alive=torch.ones(n, dtype=torch.bool, device=dev),
            anchor=torch.tensor([e.tokens[s][-1] for s in seqs], dtype=torch.int64, device=dev),
            ids=torch.zeros(n * t, dtype=torch.int64, device=dev),
            drafts=torch.zeros(n, K, dtype=torch.int64, device=dev),
            stochastic=any(x > 0 for x in temps),
            nucleus=any(p < 1.0 for p in top_p),
        )
        b["slot"] = b["real_slot"].clone()
        b["dists"] = torch.zeros(n, K, e.F.vocab, dtype=torch.float32, device=dev) if b["stochastic"] else None
        self.buf, self.batch, self.stale = b, tuple(seqs), False
        for i in range(n):
            self._propose(i, temps[i])

    def _shrink(self, seqs) -> None:
        """Rows left the batch (they finished, the host learned it a step late): keep the device view of the rest.
        Every tensor is re-indexed on the device, after the steps in flight, so nothing is read back."""
        keep = [self.batch.index(s) for s in seqs]
        idx = torch.tensor(keep, dtype=torch.int64, device=self.e.caches.device)
        b = self.buf
        for name in ("seqs", "real_slot", "slot", "ctx", "generated", "limit", "ends", "temps", "top_p", "alive", "anchor", "drafts"):
            b[name] = b[name].index_select(0, idx)
        b["ids"] = b["ids"].view(-1, self.t).index_select(0, idx).reshape(-1)
        if b["dists"] is not None:
            b["dists"] = b["dists"].index_select(0, idx)
        self.batch = tuple(seqs)

    def _propose(self, i: int, temperature: float) -> None:
        """Row i's next proposal from its device anchor at its device context, into the next step's ids."""
        e, b, K, t = self.e, self.buf, self.e.drafter.k, self.t
        ring = e.caches.draft_ring(self._slots[i])
        anchor, position = b["anchor"][i:i + 1], b["ctx"][i]
        if b["stochastic"]:
            drafts, dists = e.drafter.propose_sampled_tensor(anchor, position, ring, temperature, e.gen, e.F.vocab) if temperature > 0 \
                else (self._greedy_drafts(anchor, position, ring, b["real_slot"][i:i + 1]), None)
            if dists is None:                                             # a greedy row among stochastic ones: one-hot draft distributions
                dists = torch.zeros(K, e.F.vocab, dtype=torch.float32, device=ring.device).scatter_(1, drafts.view(K, 1), 1.0)
            b["dists"][i] = dists
        else:
            drafts = self._greedy_drafts(anchor, position, ring, b["real_slot"][i:i + 1])
        b["drafts"][i] = drafts
        b["ids"][i * t] = anchor[0]
        b["ids"][i * t + 1:(i + 1) * t] = drafts

    def _greedy_drafts(self, anchor, position, ring, slot):
        e = self.e
        if e.drafter.decode_graphs is not None:
            return e.drafter.decode_graphs.propose_from(anchor, position, slot)
        return e.drafter.propose_tensor(anchor, position, ring)

    # -- one step ---------------------------------------------------------------------------------------
    def launch(self, seqs, slots) -> Pending:
        e, K, t = self.e, self.e.drafter.k, self.t
        seqs, slots = list(seqs), list(slots)
        if len(self.pending) >= self.depth or not self.free:
            raise RuntimeError("the decode pipeline is full: resolve a step before launching another")
        self._slots = slots
        if self.stale or tuple(seqs) != self.batch:
            if self.pending or (self.batch and set(seqs) - set(self.batch)):
                if set(seqs) - set(self.batch) or self.stale:
                    raise RuntimeError("rows joined the batch while steps were in flight: the runner must drain first")
                self._shrink(seqs)
            else:
                self._build(seqs, slots)
        b = self.buf
        n = len(seqs)
        # the host's step: its contexts may lag the device's by the steps in flight; the reservation covers that lag
        ahead = max(e.inflight.get(s, 0) for s in seqs) + 1
        host_step = Step(torch.zeros(n * t, dtype=torch.int64, device=e.caches.device),
                         tuple(Segment(s, slot, e.ctx[s], i * t, t) for i, (s, slot) in enumerate(zip(seqs, slots))))
        end = max(e.ctx[s] + t * ahead for s in seqs)
        shape = e.decode_graphs.shape_for(n, end)
        ctx_before = b["ctx"].clone()
        h, aux, local = e.decode_graphs.run_device(shape, host_step, b["ids"], b["ctx"], b["seqs"], b["slot"])
        if b["stochastic"]:
            full = e.net.comm.all_gather(local, dim=-1).float()
            if e.decodable is not None and full.shape[-1] > e.decodable:
                full[:, e.decodable:] = float("-inf")
            probs = distribution_batch(full, b["temps"].repeat_interleave(t), b["top_p"].repeat_interleave(t), b["nucleus"]).view(n, t, -1)
            accepted, picks, _ = speculative_pick_batch(probs, b["drafts"], b["dists"], e.gen)
        else:
            picks = e.sampling_graphs.greedy.run(shape[:2], lambda inputs: None).view(n, t)
            accepted = None
        count, done, accepted, tokens = commit_batch(picks, b["drafts"], b["alive"], b["generated"], b["limit"], b["ends"], accepted)
        b["generated"] += count
        b["ctx"] += count
        last = tokens.gather(1, (count - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
        b["anchor"] = torch.where(count > 0, last, b["anchor"])
        b["alive"] = b["alive"] & ~done
        b["slot"] = torch.where(b["alive"], b["real_slot"], torch.zeros_like(b["real_slot"]))
        if aux is not None:
            for i, slot in enumerate(slots):
                ring = e.caches.draft_ring(slot)
                positions = ctx_before[i] + torch.arange(t, device=ring.device)
                rows = aux[i * t:(i + 1) * t]
                if e.drafter.decode_graphs is not None:
                    e.drafter.decode_graphs.observe_masked(ring, positions, rows, count[i])
                else:
                    e.drafter.observe_masked(ring, positions, rows, count[i])
        for i, s in enumerate(seqs):
            self._propose(i, e.limits[s][1])
        # the outcome crosses to the host behind an event; the next step is already queued when it is read
        lane = self.free.pop(0)
        host = self.host[lane]
        host["tokens"][:n].copy_(tokens, non_blocking=True)
        host["count"][:n].copy_(count, non_blocking=True)
        host["done"][:n].copy_(done, non_blocking=True)
        host["accepted"][:n].copy_(accepted, non_blocking=True)
        event = torch.cuda.Event() if tokens.is_cuda else None
        if event is not None:
            event.record()
        for s in seqs:
            e.inflight[s] = e.inflight.get(s, 0) + 1
        pending = Pending(self, lane, seqs, event)
        self.pending.append(pending)
        return pending

    def resolve(self, pending: Pending) -> "list[bool]":
        if not self.pending or self.pending[0] is not pending:
            raise RuntimeError("decode steps resolve in launch order")
        self.pending.pop(0)
        if pending.event is not None:
            pending.event.synchronize()
        e, K = self.e, self.e.drafter.k
        host = self.host[pending.slot]
        n = len(pending.seqs)
        counts, dones, accepted = host["count"][:n].tolist(), host["done"][:n].tolist(), host["accepted"][:n].tolist()
        tokens = host["tokens"][:n].tolist()
        for i, seq in enumerate(pending.seqs):
            e.inflight[seq] = max(0, e.inflight.get(seq, 0) - 1)
            if seq not in e.tokens:                                        # released meanwhile: nothing to apply
                continue
            c = counts[i]
            if c > 0:
                e.tokens[seq] += tokens[i][:c]
                e.ctx[seq] += c
                e.accepted_total += accepted[i]
                e.drafted_total += K
        e.steps += 1
        self.free.append(pending.slot)
        return [bool(d) for d in dones]
