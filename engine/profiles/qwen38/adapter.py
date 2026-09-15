"""Qwen3.8-Flash-Next's served net behind the runner and the door (profile).

base/composed.ComposedModel already is the runner's Model and the door's engine for any composition -- sampling with
keyed draws, penalties, grammars, logprobs, reasoning budgets, parking and resuming, verification by position -- over a
Composition and a PositionStore. The served model differs from the reference in two places only, so this file supplies
exactly those two:

    ServedStore         the store's contract over caches.Qwen38Caches: open sequences, contexts, verify and accept.
                        Every served value is addressed by position, so accepting n tokens of a verify step moves a
                        context and copies nothing; checkpoints and restores are the caches' block-boundary copies.
    ServedComposition   `forward(step, store, logits=, hidden=)` over net.Qwen38Net at TP=4: the base step's segments
                        get their state slots, the block table is published, the net runs, the head is gathered, and
                        `hidden` is the residual streams before the closing mixer (what the MTP head fuses). A decode or
                        verify step replays its captured graph (decode_graphs.TargetGraphs) once `capture` has run;
                        prefill runs eagerly.

and ServedMTP, base/composed's Drafter over the net's MTP head. ComposedModel observes one row at a time after a verify
step; the head's rows for those observations wait and run together in one captured draft step when the next step asks
for proposals (decode_graphs.DraftGraphs) -- one replay a step, not one eager head a row. The ranks agree on every pick
because the logits a pick reads are gathered (identical on every rank) and the draws are keyed (base/draws), as in
GLM-5.3's engine.

What this is not yet: the asynchronous pipeline (engine/profiles/glm53/pipeline.py is GLM's): the host reads each
step's picks before the next step is built.
"""
from __future__ import annotations

import torch

from engine.profiles.qwen38.net import Segment, Step


class ServedStore:
    """base/composed.PositionStore's surface that ComposedModel reads, over the served caches."""

    def __init__(self, caches):
        self.caches, self.pool, self.device = caches, caches.pool, caches.device
        self.ring = caches.F.spec_k + 1
        self.contexts: dict = {}
        self.slot_of: dict = {}
        self._verifying: dict = {}

    def open(self, seq: int, slot: int) -> None:
        if not 0 < slot < self.caches.slots.num_slots:
            raise ValueError(f"slot {slot} is outside the pool's rows (slot 0 is the null slot)")
        self.slot_of[seq] = slot
        self.contexts[seq] = 0
        self.caches.reset_slot(slot)

    def close(self, seq: int) -> None:
        self.slot_of.pop(seq, None)
        self.contexts.pop(seq, None)
        self._verifying.pop(seq, None)

    drop = close

    def slot_bytes(self, slot: int):
        return self.caches.slot_bytes(slot)

    def snapshot_bytes(self, snap: int):
        return self.caches.snapshot_bytes(snap)

    def check(self, step) -> None:
        for s in step.segments:
            if s.seq not in self.slot_of:
                raise ValueError(f"sequence {s.seq} is not open")
            if s.seq in self._verifying:
                raise ValueError(f"sequence {s.seq} has a verify step waiting for accept")
            if self.contexts[s.seq] != s.ctx:
                raise ValueError(f"sequence {s.seq} is at {self.contexts[s.seq]} tokens, the step says {s.ctx}")
            if getattr(s, "verify", False) and s.length > self.ring:
                raise ValueError(f"a verify segment of {s.length} tokens needs a ring of {s.length}; the caches keep {self.ring}")

    def commit(self, step) -> None:
        for s in step.segments:
            if getattr(s, "verify", False):
                self._verifying[s.seq] = (s.ctx, s.length)
            else:
                self.contexts[s.seq] = s.ctx + s.length

    def accept(self, seq: int, n: int) -> None:
        if seq not in self._verifying:
            raise ValueError(f"sequence {seq} has no verify step to accept")
        ctx, length = self._verifying.pop(seq)
        if not 1 <= n <= length:
            raise ValueError(f"accept keeps 1..{length} tokens of sequence {seq}'s verify step, not {n}")
        self.contexts[seq] = ctx + n

    def checkpoint(self, seq: int, position: int, snap: int) -> None:
        # the rings still hold every position of the last verify step, so a boundary inside it reads the same way
        self.caches.checkpoint(self.slot_of[seq], position, snap)

    def restore(self, seq: int, position: int, snap: int) -> None:
        self.caches.restore(self.slot_of[seq], position, snap)
        self.contexts[seq] = position

    def resume(self, seq: int, slot: int, context: int) -> None:
        self.slot_of[seq] = slot
        self.contexts[seq] = context


class ServedComposition:
    """The composition surface ComposedModel drives, over the served net."""

    def __init__(self, net, caches):
        self.net, self.caches = net, caches
        self.graphs = None

    def capture(self, max_seqs: int, tokens: int, *, ceiling: int, memory=None) -> None:
        """The target's decode (tokens 1) or verify (tokens k+1) graphs, before any request is admitted."""
        from engine.profiles.qwen38.decode_graphs import TargetGraphs
        if self.graphs is not None:
            raise ValueError("the target graphs are already captured")
        self.graphs = TargetGraphs(self.net, self.caches, max_seqs, tokens, ceiling=ceiling, memory=memory)

    def close(self) -> None:
        if self.graphs is not None:
            self.graphs.close()
            self.graphs = None

    def served_step(self, step, store) -> Step:
        return Step(step.ids, tuple(Segment(s.seq, store.slot_of[s.seq], s.ctx, s.start, s.length) for s in step.segments))

    def forward(self, step, store, *, logits: str = "last", hidden: bool = False, given=None):
        if given is not None:
            raise ValueError("the target composition opens from its embeddings")
        if logits not in ("last", "all"):
            raise ValueError("logits are 'last' or 'all'")
        store.check(step)
        served = self.served_step(step, store)
        if self.graphs is not None and self.graphs.admits(served, self.caches.pool):
            # rows: the graph's output row of each of the step's tokens (None: the same rows, nothing was padded)
            scores, streams, rows = self.graphs.run(served)
            t, device = self.graphs.tokens, scores.device
            if logits == "last":
                if t != 1:                        # one token a row: every row is its segment's last already
                    last = [i * t + s.length - 1 for i, s in enumerate(served.segments)]
                    scores = scores.index_select(0, torch.tensor(last, device=device))
            elif rows is not None:
                scores = scores.index_select(0, torch.tensor(rows, device=device))
            if hidden and rows is not None:
                streams = streams.index_select(0, torch.tensor(rows, device=device))
            store.commit(step)
            return (scores, streams) if hidden else scores
        self.caches.prepare(served)
        out, streams = self.net.forward(served, self.caches, streams=True)
        store.commit(step)
        if logits == "last":
            last = torch.tensor([s.start + s.length - 1 for s in step.segments], device=out.device)
            out = out.index_select(0, last)
        scores = self.net.head(out)
        return (scores, streams) if hidden else scores


class ServedMTP:
    """base/composed.Drafter over the net's MTP head (engine/modules/mtp.MTPDrafter's rule): observing the positions the
    target just kept -- each position's token is the next one, its given state the target's streams there -- leaves the
    last position's draft; `propose` hands it out (and, eagerly, chains the head from its own streams for k > 1). The
    head's rows sit in the target's blocks (caches region F.layers) at the positions they describe, so a rejected chain
    is overwritten like any draft.

    With the draft graphs captured, an observation of up to k+1 positions waits, holding its own copy of the streams
    rows (the target graph's outputs are overwritten by its next replay), until `propose` runs every waiting row in one
    replay; a second observation of the same row first runs the one waiting. Longer observations (a prompt) run the
    head eagerly at once."""

    def __init__(self, net, caches, store, k: int):
        if k <= 0:
            raise ValueError("a drafter proposes at least one token")
        self.net, self.caches, self.store, self.k = net, caches, store, k
        self.graphs = None
        self._next: dict = {}                     # seq -> (draft token, head streams [1, hc*H] | None, its position)
        self._waiting: dict = {}                  # seq -> (ctx, next ids, the target's streams rows [m, hc*H])

    def capture(self, max_seqs: int, *, ceiling: int, memory=None) -> None:
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        if self.k != 1:
            raise ValueError("the captured MTP head drafts one token a row (k=1): a chain would replay it k times")
        if self.graphs is not None:
            raise ValueError("the draft graphs are already captured")
        self.graphs = DraftGraphs(self.net, self.caches, max_seqs, self.k + 1, ceiling=ceiling, memory=memory)

    def close(self) -> None:
        if self.graphs is not None:
            self.graphs.close()
            self.graphs = None

    def _head(self, seq: int, ctx: int, ids, given):
        slot = self.store.slot_of[seq]
        step = Step(ids, (Segment(seq, slot, ctx, 0, ids.numel()),))
        self.caches.prepare(step)
        hidden, streams = self.net.mtp_forward(step, given, self.caches, last_hidden_only=True)
        token = int(self.net.head_tokens(hidden)[0])
        return token, streams

    def _run_waiting(self, seqs) -> None:
        rows = []
        for seq in seqs:
            ctx, ids, streams = self._waiting.pop(seq)
            rows.append((seq, self.store.slot_of[seq], ctx, ids, streams))
        for (seq, _slot, ctx, ids, _streams), token in zip(rows, self.graphs.run(rows)):
            self._next[seq] = (token, None, ctx + len(ids))

    def observe(self, seq: int, ctx: int, next_ids, hidden) -> None:
        n = min(len(next_ids), hidden.shape[0])
        if n == 0:
            return
        if seq in self._waiting:
            self._run_waiting([seq])              # its rows are positions before these
        self._next.pop(seq, None)
        if self.graphs is not None and n <= self.k + 1:
            self._waiting[seq] = (ctx, [int(t) for t in next_ids[:n]], hidden[:n].clone())
            return
        ids = torch.tensor([int(t) for t in next_ids[:n]], dtype=torch.int64, device=hidden.device)
        token, streams = self._head(seq, ctx, ids, hidden[:n])
        self._next[seq] = (token, streams, ctx + n)

    def propose(self, seqs) -> "list[list[int]]":
        waiting = [seq for seq in seqs if seq in self._waiting]
        if waiting:
            self._run_waiting(waiting)
        out = []
        for seq in seqs:
            got = self._next.get(seq)
            if got is None:
                out.append([])
                continue
            token, streams, position = got
            chain = [token]
            for _ in range(1, self.k):
                ids = torch.tensor([chain[-1]], dtype=torch.int64, device=streams.device)
                token, streams = self._head(seq, position, ids, streams)
                chain.append(token)
                position += 1
            out.append(chain)
        return out

    def forget(self, seq: int) -> None:
        self._next.pop(seq, None)
        self._waiting.pop(seq, None)


def build_model(net, caches, F, *, eos_ids, max_new: int, temperature: float, top_p: float, seed: int = 0,
                drafter: bool = True, grammars=None):
    """ComposedModel over the served net and caches (and the MTP drafter when `drafter`)."""
    from engine.base.composed import ComposedModel
    store = ServedStore(caches)
    composition = ServedComposition(net, caches)
    mtp = ServedMTP(net, caches, store, F.spec_k) if drafter and F.spec_k else None
    model = ComposedModel(composition, store, vocab=F.vocab, eos_ids=eos_ids, max_new=max_new, temperature=temperature,
                          top_p=top_p, seed=seed, max_context=F.max_position, drafter=mtp, grammars=grammars)
    return model, store


def capture(model, max_seqs: int, *, memory=None) -> None:
    """The fleet's decode graphs, before the door admits work: the target's at the verify width (or one token without a
    drafter), then the draft head's. The served ceiling is the model's context limit."""
    k = model.k if model.drafter is not None else 0
    try:
        model.composition.capture(max_seqs, k + 1, ceiling=model.max_context, memory=memory)
        if model.drafter is not None:
            model.drafter.capture(max_seqs, ceiling=model.max_context, memory=memory)
    except BaseException as exc:
        from engine.base.graphs import cleanup_after_error
        cleanup_after_error(exc, lambda: close(model), "close decode graphs after a capture failure")
        raise


def close(model) -> None:
    """Release the captured graphs (their NCCL references) before the process group goes."""
    if model.drafter is not None:
        model.drafter.close()
    model.composition.close()


__all__ = ["ServedStore", "ServedComposition", "ServedMTP", "build_model", "capture", "close"]
