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
for proposals (decode_graphs.DraftGraphs) -- one replay a step, not one eager head a row, and for K > 1 the chain of
K picks inside that replay (decode_graphs.draft_chain). The ranks agree on every pick because the logits a pick reads
are gathered (identical on every rank) and the draws are keyed (base/draws), as in GLM-5.3's engine.

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

    def takes_mark(self, start: int, position: int) -> bool:
        """base/composed.ComposedModel.prefill asks before it cuts a step at a prefix boundary: the net takes a
        boundary on its kernels' grid out of the one uncut forward (net.Step.marks -> caches.mark_gdn, mark_ple), so a
        chunk of whole blocks is one forward and one MTP observation, not one a block."""
        return self.net.takes_mark(position - start)

    def served_step(self, step, store, marks=()) -> Step:
        """`marks`: ((absolute position, snapshot) ...) inside the step's one prefill segment, as net.Step counts them
        -- from the segment's start."""
        ctx = step.segments[0].ctx
        return Step(step.ids, tuple(Segment(s.seq, store.slot_of[s.seq], s.ctx, s.start, s.length) for s in step.segments),
                    tuple((int(p) - ctx, int(snap)) for p, snap in marks))

    def forward(self, step, store, *, logits: str = "last", hidden: bool = False, given=None, marks=(), host=None):
        if given is not None:
            raise ValueError("the target composition opens from its embeddings")
        if logits not in ("last", "all"):
            raise ValueError("logits are 'last' or 'all'")
        store.check(step)
        served = self.served_step(step, store, marks)
        if not served.marks and self.graphs is not None and self.graphs.admits(served, self.caches.pool):
            # rows: the graph's output row of each of the step's tokens (None: the same rows, nothing was padded)
            # host: the step's ids and carried context as the model holds them (decode_graphs.TargetGraphs.run)
            scores, streams, rows = self.graphs.run(served, known=host)
            t, device = self.graphs.tokens, scores.device
            if logits == "last":
                if t != 1:                        # one token a row: every row is its segment's last already
                    last = [i * t + s.length - 1 for i, s in enumerate(served.segments)]
                    scores = scores.index_select(0, torch.tensor(last, device=device))
            elif rows is not None:
                scores = scores.index_select(0, torch.tensor(rows, device=device))
            if hidden:
                # never the graph's own tensor: a prefill split at a mark keeps each piece's streams until the prompt
                # is in, and the next piece's replay of the same shape would overwrite them
                streams = streams.clone() if rows is None else streams.index_select(0, torch.tensor(rows, device=device))
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
    last position's draft; `propose` hands it out, and for k > 1 the chain: the head at each next position from its own
    pick and its own streams. The head's rows sit in the target's blocks (caches region F.layers) at the positions they
    describe, so a rejected chain is overwritten like any draft.

    With the draft graphs captured, an observation of up to k+1 positions waits -- its streams rows are the
    composition's own copy, never a graph's output -- until `propose` runs every waiting row in one replay, the chain
    included (decode_graphs.draft_chain); a second observation of the same row first runs the one waiting, and `forget`
    runs it while the row still holds its slot and blocks (a park: the head's rows belong in blocks the tier keeps) and
    drops it once they are released. A waiting row whose reservation does not hold the replay's positions (a parked
    row holds its kept positions, not the horizon the next step reserves) runs the head eagerly over its observed
    positions alone, as longer observations (a prompt) do at once; the chain then runs eagerly at `propose`."""

    def __init__(self, net, caches, store, k: int):
        if k <= 0:
            raise ValueError("a drafter proposes at least one token")
        self.net, self.caches, self.store, self.k = net, caches, store, k
        self.graphs = None
        self._next: dict = {}                     # seq -> (picks, head streams [1, hc*H] | None, the chain's position)
        self._waiting: dict = {}                  # seq -> (slot, ctx, next ids, the target's streams rows [m, hc*H])

    def capture(self, max_seqs: int, *, ceiling: int, memory=None) -> None:
        from engine.profiles.qwen38.decode_graphs import DraftGraphs
        if self.graphs is not None:
            raise ValueError("the draft graphs are already captured")
        self.graphs = DraftGraphs(self.net, self.caches, max_seqs, self.k + 1, k=self.k, ceiling=ceiling,
                                  memory=memory)

    def close(self) -> None:
        if self.graphs is not None:
            self.graphs.close()
            self.graphs = None

    def _head(self, seq: int, ctx: int, ids, given):
        slot = self.store.slot_of[seq]
        step = Step(ids, (Segment(seq, slot, ctx, 0, ids.numel()),))
        self.caches.prepare(step)
        hidden, streams = self.net.mtp_forward(step, given, self.caches, last_hidden_only=True)
        token = int(self.net.draft_tokens(hidden)[0])
        return token, streams

    def _run_waiting(self, seqs) -> None:
        """The waiting rows of `seqs`: in one replay when each row's reservation holds the replay's positions (its
        observation padded to the verify width, then the chain), else the head eagerly over its observed positions."""
        pool, graphs = self.caches.pool, self.graphs
        rows, eager = [], []
        for seq in seqs:
            slot, ctx, ids, streams = self._waiting.pop(seq)
            if ctx + graphs.extent(len(ids)) <= pool.tokens[seq]:
                rows.append((seq, slot, ctx, ids, streams))
            else:
                eager.append((seq, ctx, ids, streams))
        if rows:
            for (seq, _slot, ctx, ids, _streams), picks in zip(rows, graphs.run(rows)):
                self._next[seq] = (picks, None, ctx + len(ids))
        for seq, ctx, ids, streams in eager:
            token, head = self._head(seq, ctx, torch.tensor(ids, dtype=torch.int64, device=streams.device), streams)
            self._next[seq] = ([token], head, ctx + len(ids))

    def observe(self, seq: int, ctx: int, next_ids, hidden) -> None:
        n = min(len(next_ids), hidden.shape[0])
        if n == 0:
            return
        if seq in self._waiting:
            self._run_waiting([seq])              # its rows are positions before these
        self._next.pop(seq, None)
        if self.graphs is not None and n <= self.k + 1:
            self._waiting[seq] = (self.store.slot_of[seq], ctx, [int(t) for t in next_ids[:n]], hidden[:n])
            return
        ids = torch.tensor([int(t) for t in next_ids[:n]], dtype=torch.int64, device=hidden.device)
        token, streams = self._head(seq, ctx, ids, hidden[:n])
        self._next[seq] = ([token], streams, ctx + n)

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
            picks, streams, position = got
            chain = [int(t) for t in picks[:self.k]]
            while streams is not None and len(chain) < self.k:
                # an eager head's chain: the next position from the row's last pick and the head's own streams
                ids = torch.tensor([chain[-1]], dtype=torch.int64, device=streams.device)
                token, streams = self._head(seq, position, ids, streams)
                chain.append(token)
                position += 1
            out.append(chain)
        return out

    def forget(self, seq: int) -> None:
        self._next.pop(seq, None)
        waiting = self._waiting.get(seq)
        if waiting is not None:
            slot, ctx, ids = waiting[0], waiting[1], waiting[2]
            # a park forgets the row before its slot and blocks go (base/runner.park_begin): run the head's rows into
            # them; a released row's blocks are no longer its own, so its rows are dropped
            if self.caches.slots.owner[slot] == seq and ctx + len(ids) <= self.caches.pool.tokens[seq]:
                self._run_waiting([seq])
                self._next.pop(seq, None)
            else:
                self._waiting.pop(seq)


def _served_model_class():
    from engine.base import draws
    from engine.base.composed import ComposedModel
    from engine.base.composition import Segment as BaseSegment, Step as BaseStep
    from engine.base.sampler import sample

    class ServedModel(ComposedModel):
        """base/composed.ComposedModel with a verify step that reads the host once for its picks.

        The base verify picks one position at a time -- a device read, and for a sampled row four small uploads, per
        position -- because a rich row's pick reads the tokens the picks before it committed. A plain row's pick does
        not: it reads its logits row, its temperature, top-p and top-k, and the uniform keyed at its generation count,
        and at position j that count is the row's count now plus j (the picks before j commit before the pick at j is
        read). So every position of every plain row is drawn in one call with those keys, and the base loop consumes
        them in its own order -- the same tokens, commits, accepts and observations; a position after the first
        rejection is drawn and never read (a draw is a pure function of its key). A row that is rich at its first
        position (options, a grammar, logprobs, min_tokens still holding its end) keeps the base's per-position pick.
        The step's ids go up in one upload rather than one a row."""

        def _draw_ahead(self, seqs, segments, logits):
            ahead = [None] * len(seqs)
            index, temps, top_ps, top_ks, keys, plain = [], [], [], [], [], []
            for i, (seq, s) in enumerate(zip(seqs, segments)):
                if self._rich(seq):                 # min_tokens only relaxes as the row grows: rich now or never
                    continue
                opts = self.options.get(seq, {})
                count = self.generated_count(seq)
                for j in range(s.length):
                    index.append(s.start + j)
                    temps.append(self.limits[seq][1])
                    top_ps.append(float(opts.get("top_p", self.top_p)))
                    top_ks.append(int(opts.get("top_k") or 0))
                    keys.append((seq, count + j))
                plain.append(i)
            if not index:
                return ahead
            dev = logits.device
            rows = logits.index_select(0, torch.tensor(index, device=dev))
            if all(t <= 0 for t in temps):
                drawn = rows[:, :self.vocab].argmax(dim=-1).tolist()
            else:
                uniforms = [draws.uniform(draws.row_key(self.seeds.get(seq, self.seed), self.nonces[seq], count),
                                          draws.PICK, 0) for seq, count in keys]
                drawn = sample(rows, torch.tensor(temps, dtype=torch.float32, device=dev),
                               torch.tensor(top_ps, dtype=torch.float32, device=dev),
                               torch.tensor(uniforms, dtype=torch.float32, device=dev),
                               top_k=torch.tensor(top_ks, dtype=torch.int32, device=dev), valid=self.vocab).tolist()
            at = 0
            for i in plain:
                n = segments[i].length
                ahead[i] = [int(t) for t in drawn[at:at + n]]
                at += n
            return ahead

        def _carried(self, seq, ctx):
            """The ngram_size - 1 tokens before position `ctx` of a row, DEAD before the sequence: what its slot's PLE
            ids ring holds there -- the tokens fed at those positions, which are the row's history."""
            from engine.modules.ngram_embedding import DEAD
            width = self.composition.net.F.ngram_size - 1
            tokens = self.tokens[seq]
            return [tokens[p] if p >= 0 else DEAD for p in range(ctx - width, ctx)]

        def _verify(self, seqs):
            proposals = self.drafter.propose(seqs)
            drafts = []                             # no more drafts than the row can still take after its next token
            for seq, proposal in zip(seqs, proposals):
                room = min(self.limits[seq][0] - self.generated_count(seq), self.max_context - self.context(seq)) - 1
                drafts.append([int(t) for t in proposal[:max(0, min(self.k, room))]])
            flat, segments = [], []
            for seq, d in zip(seqs, drafts):
                segments.append(BaseSegment(seq, self.context(seq), len(flat), 1 + len(d), True))
                flat += [self.tokens[seq][-1]] + d
            step = BaseStep(torch.tensor(flat, dtype=torch.int64, device=self.store.device), tuple(segments))
            # the PLE staging hashes the step's ids and each row's tokens before its context: this model holds both, so
            # the replay's host half reads neither back off the device (two reads and a launch sequence a step)
            carried = [self._carried(seq, segment.ctx) for seq, segment in zip(seqs, segments)]
            logits, hidden = self.composition.forward(step, self.store, logits="all", hidden=True, host=(flat, carried))
            self.steps += 1
            ahead = self._draw_ahead(seqs, step.segments, logits)
            finished = []
            for seq, d, segment, drawn in zip(seqs, drafts, step.segments, ahead):
                fed, done, matched = 0, False, 0
                for j in range(segment.length):
                    if drawn is not None:
                        pick = drawn[j]
                    else:
                        pick = self._pick([seq], logits[segment.start + j:segment.start + j + 1])[0]
                    done = self._commit(seq, pick)
                    fed = j + 1
                    if j < len(d) and pick == d[j]:
                        matched += 1
                    if done or j == len(d) or pick != d[j]:
                        break
                self.store.accept(seq, fed)
                if d:
                    self.drafts_total += 1
                    self.drafted_total += len(d)
                    self.accepted_total += matched
                self.drafter.observe(seq, segment.ctx, self.tokens[seq][segment.ctx + 1:segment.ctx + fed + 1],
                                     hidden[segment.start:segment.start + fed])
                finished.append(done)
            return finished

    return ServedModel


def build_model(net, caches, F, *, eos_ids, max_new: int, temperature: float, top_p: float, seed: int = 0,
                drafter: bool = True, grammars=None):
    """The served model (ServedModel: base/composed.ComposedModel with the one-read verify) over the served net and
    caches, and the MTP drafter when `drafter`."""
    store = ServedStore(caches)
    composition = ServedComposition(net, caches)
    mtp = ServedMTP(net, caches, store, F.spec_k) if drafter and F.spec_k else None
    model = _served_model_class()(composition, store, vocab=F.vocab, eos_ids=eos_ids, max_new=max_new,
                                  temperature=temperature, top_p=top_p, seed=seed, max_context=F.max_position,
                                  drafter=mtp, grammars=grammars)
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
