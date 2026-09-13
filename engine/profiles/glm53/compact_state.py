"""Eager transactions for the compact KDA cache.

CUDA uses the same factor ABI as captured decode. CPU reference execution
retains the reference lane's intermediate states only until the clipped
commit, so lifecycle tests do not need Triton or a device. Owners are bounded
by segment index and verification width, never by request/context identity.
"""
import torch


class EagerCommit:
    def __init__(self, caches):
        self.caches = caches
        self.layers = tuple(L for L in caches.layers if not caches.F.is_dsa(L))
        self.layer_index = {L: i for i, L in enumerate(self.layers)}
        self.owners = {}
        self.clear()

    def clear(self):
        self.segments = ()
        self.indices, self.states, self.seen = {}, {}, set()

    @property
    def nbytes(self):
        return sum(owner.nbytes for owner in self.owners.values())

    def check_reusable(self, slot):
        if any(s.slot == slot for s in self.segments):
            raise RuntimeError("compact verification must commit before slot reuse")

    def begin(self, step, *, commit_all):
        if self.segments:
            raise RuntimeError("compact verification must commit before the next forward")
        c, segments = self.caches, tuple(step.segments)
        if (not segments or len(segments) > c.pool.max_seqs
                or len({s.slot for s in segments}) != len(segments)
                or any(not 0 < s.slot < c.slots.num_slots or s.ctx < 0 or s.length <= 0 for s in segments)):
            raise ValueError("compact eager step requires distinct real slots and valid segments")
        if not commit_all and any(s.length > c.F.spec_k + 1 for s in segments):
            raise ValueError("compact speculative eager steps must fit the verification width")
        self.commit_all = commit_all
        self.segments = segments
        self.indices = {s.start: i for i, s in enumerate(segments)}
        for s in segments:
            if s.ctx:
                c._expect_position(s.slot, s.ctx)

    def complete_layer(self, layer, segment):
        key = (self.indices[segment.start], layer)
        if key in self.seen:
            raise RuntimeError("compact layer verified twice in the same transaction")
        self.seen.add(key)

    def verify(self, layer, segment, args, recurrent):
        c, s = self.caches, segment
        index = self.indices[s.start]
        if s.length > min(c.F.spec_k + 1, c.F.block):
            raise ValueError("compact eager verification exceeds its declared width")
        if c.device.type == "cuda":
            key = (index, s.length)
            if key not in self.owners:
                self.owners[key] = c.deferred_batch(1, s.length)
            slots = torch.tensor([s.slot], device=c.device, dtype=torch.int64)
            contexts = torch.tensor([s.ctx], device=c.device, dtype=torch.int64)
            out = self.owners[key].verify(self.layer_index[layer], *args, slots, contexts, c.F.lower_bound)
        else:
            state = c.kda(layer, s.slot)[1][0:1] if s.ctx else None
            out, states = recurrent(*args, state, c.F.lower_bound)
            self.states[index, layer] = states
        self.complete_layer(layer, s)
        return out

    def commit(self, counts=None):
        if not self.segments:
            raise RuntimeError("no compact eager verification to commit")
        if counts is None:
            if not self.commit_all:
                raise ValueError("speculative compact verification needs clipped counts")
            counts = [s.length for s in self.segments]
        counts = tuple(counts)
        if len(counts) != len(self.segments) or any(
                type(n) is not int or not 0 <= n <= s.length
                or (s.length > self.caches.F.spec_k + 1 and n != s.length)
                for s, n in zip(self.segments, counts)):
            raise ValueError("compact commit needs one valid clipped count per segment")
        if self.seen != {(i, L) for i in range(len(self.segments)) for L in self.layers}:
            raise RuntimeError("all KDA layers must finish before compact commit")
        c = self.caches
        for i, (s, count) in enumerate(zip(self.segments, counts)):
            if s.length > c.F.spec_k + 1:
                # Chunk recurrence stored the final state and all internal
                # prefix marks. A long prefill cannot use one boundary slot.
                c._fields["rec_meta", -1][s.slot, 0] = s.ctx + count
                c._fields["rec_meta", -1][s.slot, 1] = 0
                continue
            slots, contexts, accepted = (torch.tensor([v], device=c.device, dtype=torch.int64)
                                          for v in (s.slot, s.ctx, count))
            if c.device.type == "cuda":
                self.owners[i, s.length].commit(slots, contexts, accepted)
            elif count:
                boundary = ((s.ctx + count) // c.F.block) * c.F.block
                for L in self.layers:
                    states = self.states[i, L]
                    c.kda(L, s.slot)[1][0].copy_(states[count - 1])
                    if boundary > s.ctx:
                        c._fields["rec_boundary", L][s.slot].copy_(states[boundary - s.ctx - 1])
            c.stage_boundaries(slots, contexts, accepted)
        self.clear()
