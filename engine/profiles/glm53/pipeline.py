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
sampling, base/sampler.block_verify_batch); rows with penalties, logit_bias, seeds, logprobs, grammars or a
pending min_tokens keep the synchronous path, and the runner drains this one before them (adapter.async_ready).
Every device-side draw is a keyed uniform (base/draws): the same on every rank whatever came before.
"""
from __future__ import annotations

import torch

from engine.base.sampler import block_verify_batch, commit_batch, rows as sampler_rows
from engine.base.stage_clock import StageClock
from engine.profiles.glm53.net import Segment, Step


def distribution_batch(logits: torch.Tensor, temps: torch.Tensor, top_k: torch.Tensor, top_p: torch.Tensor,
                       valid: "int | None" = None, into: "torch.Tensor | None" = None) -> torch.Tensor:
    """base/sampler.rows for every row at once, distributions only: [m, V] fp32 -- a one-hot argmax where the row's
    temperature is 0, the truncated softmax otherwise.

    There is no `nucleus` argument and no list of which rows to sort, because nothing here sorts: each row carries its
    own top-k and top-p as numbers the sampler reads on the device (45차 §32). A host predicate had to exist while the
    truncation was a sort -- the sort of a [24, 155k] batch was 2.9 ms and picking which rows to pay it for was worth
    the bookkeeping. The threshold search is 467 us for the same block whether one row asks for a nucleus or all of
    them do. No draw either: the speculative pick works from these distributions, not from a token drawn out of them.
    """
    out = torch.empty(logits.shape[0], logits.shape[-1], dtype=torch.float32,
                      device=logits.device) if into is None else into
    sampler_rows(logits, temps, top_k, top_p, None, valid, out)
    return out


class Pending:
    def __init__(self, pipeline, slot: int, seqs, event):
        self.pipeline, self.slot, self.seqs, self.event = pipeline, slot, tuple(seqs), event

    def resolve(self) -> "list[bool]":
        return self.pipeline.resolve(self)


class AsyncDecode:
    def __init__(self, engine, depth: int = 2):
        self.e = engine
        self.depth = depth
        self.clock = StageClock(device=getattr(getattr(engine, "caches", None), "device", None))
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
        self._zeros = {}                                 # n -> the host step's placeholder ids (prepare reads segments, never these)
        self._staged = []                                # pinned index tensors of recent shrinks, alive until their copies land
        self._probs = None                               # the step's target distributions, kept: 15 MB the step stops reallocating
        self.dirty = set()                               # only rows moved by synchronous host work
        self.slots = {}                                  # host identities; never read device slots to compare batches
        self.merges = 0

    def invalidate(self, seqs=None):
        if seqs is None:
            self.stale = True
        else:
            self.dirty.update(seqs)

    def ready_for(self, seqs, slots=None):
        if not self.pending:
            return True
        if self.stale:
            return False
        # A reused/extended row cannot inherit the old turn's outstanding
        # readback. Independent new rows may join without draining survivors.
        changing = self.dirty | (set(seqs) - set(self.batch))
        if slots is not None:
            changing.update(s for s, slot in zip(seqs, slots) if self.slots.get(s) != slot)
        return not any(changing.intersection(p.seqs) for p in self.pending)

    def _upload(self, values, dtype):
        if self.e.caches.device.type != "cuda":
            return torch.tensor(values, dtype=dtype, device=self.e.caches.device)
        host = torch.tensor(values, dtype=dtype, pin_memory=True)
        self._staged.append(host)
        return host.to(self.e.caches.device, non_blocking=True)

    def _dists(self, rows: int, vocab: int):
        """The block the sampler writes the step's distributions into."""
        if self._probs is None or self._probs.shape[1] != vocab:
            room = self.e.caches.pool.max_seqs * self.t
            self._probs = torch.empty(room, vocab, dtype=torch.float32, device=self.e.caches.device)
        return self._probs[:rows]

    # -- building the device view of a batch --------------------------------------------------------
    def _new_rows(self, seqs, slots):
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
            seqs=self._upload(seqs, torch.int64),
            real_slot=self._upload(slots, torch.int64),
            ctx=self._upload([e.ctx[s] for s in seqs], torch.int64),
            generated=self._upload([e._generated_count(s) for s in seqs], torch.int64),
            limit=self._upload([e.limits[s][0] for s in seqs], torch.int64),
            ends=self._upload([x + [-1] * (width - len(x)) for x in ends], torch.int64),
            nonce=self._upload([e.nonces[s] for s in seqs], torch.int64),      # the draws' key, with `generated` (base/draws)
            temps=self._upload(temps, torch.float32),
            top_k=self._upload([int(e.options.get(s, {}).get("top_k") or 0) for s in seqs], torch.int32),
            top_p=self._upload(top_p, torch.float32),
            alive=torch.ones(n, dtype=torch.bool, device=dev),
            anchor=self._upload([e.tokens[s][-1] for s in seqs], torch.int64),
            ids=torch.zeros(n * t, dtype=torch.int64, device=dev),
            drafts=torch.zeros(n, K, dtype=torch.int64, device=dev),
            stochastic=any(x > 0 for x in temps),
        )
        b["slot"] = b["real_slot"].clone()
        # the draft distribution is its candidates and their mass, not a vocabulary-wide row (base/sampler)
        b["qcand"] = torch.zeros(n, K, e.drafter.F.sel_top_k, dtype=torch.int64, device=dev) if b["stochastic"] else None
        b["qprob"] = torch.zeros(n, K, e.drafter.F.sel_top_k, dtype=torch.float32, device=dev) if b["stochastic"] else None
        # the K+1 uniforms the verification of the current proposals takes: keyed with them (base/draws.step_block)
        b["draws"] = torch.zeros(n, K + 1, dtype=torch.float32, device=dev) if b["stochastic"] else None
        self._propose_rows(b)
        return b

    def _build(self, seqs, slots):
        self.buf = self._new_rows(seqs, slots)
        self.batch, self.slots, self.stale = tuple(seqs), dict(zip(seqs, slots)), False
        self.dirty.difference_update(seqs)

    def _merge(self, seqs, slots):
        """Preserve survivor progress/proposals on device, initialize only joining rows."""
        keep = [s for s, slot in zip(seqs, slots)
                if s in self.batch and s not in self.dirty and self.slots.get(s) == slot]
        added = [s for s in seqs if s not in keep]
        if not keep:
            self._build(seqs, slots)
            return
        if tuple(keep) != self.batch:
            self._shrink(keep)
        if added:
            fresh = self._new_rows(added, [slots[seqs.index(s)] for s in added])
            b = self.buf
            width = max(b["ends"].shape[1], fresh["ends"].shape[1])
            for rows in (b, fresh):
                if rows["ends"].shape[1] < width:
                    rows["ends"] = torch.nn.functional.pad(rows["ends"], (0, width - rows["ends"].shape[1]), value=-1)
            stochastic = b["stochastic"] or fresh["stochastic"]
            if stochastic:
                for rows in (b, fresh):
                    if rows["qprob"] is None:
                        # Greedy proposals are point masses when a sampled row
                        # joins; regenerating survivors would discard progress
                        # and consume an extra set of random draws. A point mass
                        # is one candidate carrying all of it.
                        c = self.e.drafter.F.sel_top_k
                        rows["qcand"] = rows["drafts"].unsqueeze(2).expand(*rows["drafts"].shape, c).contiguous()
                        rows["qprob"] = torch.zeros((*rows["drafts"].shape, c), dtype=torch.float32,
                                                    device=self.e.caches.device)
                        rows["qprob"][..., 0] = 1.0
                    if rows.get("draws") is None:
                        # their verification's uniforms, keyed as the proposals they already hold would have been:
                        # the generation count has not moved since (a row at temperature 0 decides without them)
                        from engine.base import draws
                        rows["draws"] = draws.step_block(self.e.seed, rows["nonce"], rows["generated"],
                                                         self.e.drafter.k)[:, self.e.drafter.k:].contiguous()
            for name in b:
                if name == "stochastic":
                    continue
                if b[name] is not None:
                    b[name] = torch.cat((b[name], fresh[name]), 0)
            b["stochastic"] = stochastic
            self.batch = tuple(keep + added)
            if self.batch != tuple(seqs):
                self._shrink(seqs)
        self.slots = dict(zip(seqs, slots))
        self.dirty.difference_update(seqs)
        self.merges += 1

    def _shrink(self, seqs) -> None:
        """Rows left the batch (they finished, the host learned it a step late): keep the device view of the rest.
        Every tensor is re-indexed on the device, after the steps in flight, so nothing is read back."""
        keep = [self.batch.index(s) for s in seqs]
        dev = self.e.caches.device
        if dev.type == "cuda":
            host = torch.tensor(keep, dtype=torch.int64, pin_memory=True)     # a pageable copy would wait for the steps in flight
            idx = host.to(dev, non_blocking=True)
            self._staged.append(host)
        else:
            idx = torch.tensor(keep, dtype=torch.int64, device=dev)
        b = self.buf
        for name in ("seqs", "real_slot", "slot", "ctx", "generated", "limit", "ends", "nonce", "temps", "top_k", "top_p",
                     "alive", "anchor", "drafts"):
            b[name] = b[name].index_select(0, idx)
        b["ids"] = b["ids"].view(-1, self.t).index_select(0, idx).reshape(-1)
        for name in ("qcand", "qprob", "draws"):
            if b.get(name) is not None:
                b[name] = b[name].index_select(0, idx)
        if "stochastic" in b:
            b["stochastic"] = any(self.e.limits[s][1] > 0 for s in seqs)
            if not b["stochastic"]:
                b["qcand"] = b["qprob"] = b["draws"] = None
        # nothing else to re-index: the truncations ride in `top_k` and `top_p` above, and the list of
        # which rows to sort went away with the sort (45차 §32)
        self.batch = tuple(seqs)

    def _propose_rows(self, rows=None) -> None:
        """Every row's next proposal at once, from its device anchor at its device context, into the next step's ids
        (45차 §23 GPU 판정 4차: one drafter replay a step, not one a row)."""
        e, b, t = self.e, self.buf if rows is None else rows, self.t
        n = b["ctx"].shape[0]
        graphs = e.drafter.decode_graphs
        if b["stochastic"]:
            # Everything these proposals draw, keyed by the rows' nonce and how many tokens they have generated
            # (base/draws.step_block): the walk's K now, the verification's K+1 at the next step, when the count
            # has not moved. Device state in, device tensors out -- the chain stays ahead of the host, and in
            # the captured graph the hash is part of the replay, not a kernel launch each.
            if graphs is not None:
                drafts, qcand, qprob, verify = graphs.propose_rows_sampled(b["anchor"], b["ctx"], b["real_slot"], b["temps"],
                                                                           b["alive"], b["nonce"], b["generated"])
            else:
                from engine.base import draws
                K = e.drafter.k
                block = draws.step_block(e.seed, b["nonce"], b["generated"], K)
                drafts, qcand, qprob = e.drafter.propose_rows(e.caches.draft_field(), b["real_slot"], b["anchor"], b["ctx"],
                                                              temps=b["temps"], uniforms=block[:, :K], vocab=e.F.vocab, alive=b["alive"])
                verify = block[:, K:]
            b["qcand"].copy_(qcand)
            b["qprob"].copy_(qprob)
            b["draws"].copy_(verify)
        elif graphs is not None:
            drafts = graphs.propose_rows(b["anchor"], b["ctx"], b["real_slot"], b["alive"])
        else:
            drafts = e.drafter.propose_rows(e.caches.draft_field(), b["real_slot"], b["anchor"], b["ctx"], alive=b["alive"])
        b["drafts"].copy_(drafts)
        ids = b["ids"].view(n, t)
        ids[:, 0] = b["anchor"]
        ids[:, 1:] = b["drafts"]

    def iterate(self, shape, b, host_step=None, *, measured=False, stage_clock=None):
        """One shared target/commit/observe/propose chain; no host readback.

        With no host_step, block mappings were already published before launch.
        Bounded capture calls the same chain with stage events disabled.
        """
        from contextlib import nullcontext
        e, t, n = self.e, self.t, len(b["ctx"])
        mark = stage_clock.mark if stage_clock is not None else self.clock.mark if measured else lambda name: nullcontext()
        with mark("forward"):
            if host_step is None:
                h, aux, local = e.decode_graphs.run_inputs(shape, b["ids"], b["ctx"], b["seqs"], b["slot"])
            else:
                h, aux, local = e.decode_graphs.run_device(shape, host_step, b["ids"], b["ctx"], b["seqs"], b["slot"])
        if b["stochastic"]:
            # the model's dtype, not fp32: the sampler converts as it reads, and the undecodable tail
            # is a width it stops at rather than a copy of the block with minus infinity in its end
            with mark("all_gather"):
                full = e.net.comm.all_gather(local, dim=-1)
            with mark("sample"):
                probs = distribution_batch(full, b["temps"].repeat_interleave(t), b["top_k"].repeat_interleave(t),
                                           b["top_p"].repeat_interleave(t), e.decodable,
                                           self._dists(n * t, full.shape[-1])).view(n, t, -1)
            e.note_ceilings(probs, b["qprob"], b["qcand"])
            with mark("verify"):
                accepted, picks, _ = block_verify_batch(probs, b["drafts"], b["qcand"], b["qprob"], b["draws"])
        else:
            with mark("sample"):
                picks = e.sampling_graphs.greedy.run(shape[:2], lambda inputs: None).view(n, t)
            accepted = None
        if picks.is_cuda:
            from engine.kernels.decode_commit import advance
            with mark("commit"):
                count, done, accepted, tokens, ctx_before = advance(picks, b, accepted)
        else:
            ctx_before = b["ctx"].clone()
            count, done, accepted, tokens = commit_batch(picks, b["drafts"], b["alive"], b["generated"], b["limit"], b["ends"], accepted)
            b["generated"] += count
            b["ctx"] += count
            last = tokens.gather(1, (count - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
            b["anchor"] = torch.where(count > 0, last, b["anchor"])
            b["alive"] = b["alive"] & ~done
            b["slot"] = torch.where(b["alive"], b["real_slot"], torch.zeros_like(b["real_slot"]))
        with mark("boundaries"):
            e.caches.stage_boundaries(b["real_slot"], ctx_before, count)   # a block boundary crossed: parked for the host
        if aux is not None:
            positions = ctx_before.view(n, 1) + torch.arange(t, device=aux.device)
            with mark("observe"):
                prepared = getattr(e.decode_graphs, "observations", {}).get(shape)
                if prepared is not None:
                    positions, context = prepared
                    e.drafter.decode_graphs.observe_prepared_rows(b["real_slot"], positions, context, count, aux)
                elif e.drafter.decode_graphs is not None:
                    e.drafter.decode_graphs.observe_rows(b["real_slot"], positions, aux, count)
                else:
                    e.drafter.observe_rows(e.caches.draft_field(), b["real_slot"], positions, aux, count)
        with mark("propose"):
            self._propose_rows(b)
        return dict(tokens=tokens, count=count, done=done, accepted=accepted, before=ctx_before)

    # -- one step ---------------------------------------------------------------------------------------
    def launch(self, seqs, slots) -> Pending:
        e, K, t = self.e, self.e.drafter.k, self.t
        seqs, slots = list(seqs), list(slots)
        if len(self.pending) >= self.depth or not self.free:
            raise RuntimeError("the decode pipeline is full: resolve a step before launching another")
        if not self.ready_for(seqs, slots):
            raise RuntimeError("invalidated rows have steps in flight: the runner must drain first")
        if self.stale:
            self._build(seqs, slots)
        elif (tuple(seqs) != self.batch or self.dirty.intersection(seqs)
              or any(self.slots.get(s) != slot for s, slot in zip(seqs, slots))):
            self._merge(seqs, slots)
        b = self.buf
        n = len(seqs)
        # the host's step: its contexts may lag the device's by the steps in flight; the reservation covers that lag
        ahead = max(e.inflight.get(s, 0) for s in seqs) + 1
        zeros = self._zeros.get(n)
        if zeros is None:
            zeros = self._zeros[n] = torch.zeros(n * t, dtype=torch.int64, device=e.caches.device)
        host_step = Step(zeros, tuple(Segment(s, slot, e.ctx[s], i * t, t) for i, (s, slot) in enumerate(zip(seqs, slots))))
        end = max(e.ctx[s] + t * ahead for s in seqs)
        shape = e.decode_graphs.shape_for(n, end)
        self.clock.step()
        result = self.iterate(shape, b, host_step, measured=True)
        tokens, count, done, accepted = (result[k] for k in ("tokens", "count", "done", "accepted"))
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
        pending.staged, self._staged = self._staged, []
        self.pending.append(pending)
        return pending

    def resolve(self, pending: Pending) -> "list[bool]":
        if not self.pending or self.pending[0] is not pending:
            raise RuntimeError("decode steps resolve in launch order")
        self.pending.pop(0)
        if pending.event is not None:
            pending.event.synchronize()
        pending.staged = []                               # these uploads precede this exact event
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
                before = e.ctx[seq]
                e.tokens[seq] += tokens[i][:c]
                e.ctx[seq] += c
                e.accepted_total += accepted[i]
                e.drafted_total += K
                boundary = (e.ctx[seq] // e.F.block) * e.F.block
                if boundary > before:
                    e.staged[seq] = boundary                                  # the runner may checkpoint it from the stage
        e.steps += 1
        self.free.append(pending.slot)
        return [bool(d) for d in dones]
