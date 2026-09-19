"""A delta-rule layer's state in a slot, and what a prefix boundary keeps of it (module: linear attention).

A layer with a short causal conv and a recurrent state (KDA, GDN -- any delta-rule linear attention) keeps two rings in
each sequence's state slot, both addressed BY POSITION so a rejected draft is overwritten by the next step's writes:

    ("conv", L)   [C, conv-1 + K]   the conv's inputs by position: a boundary needs the conv-1 before it
    ("rec", L)    [K + 1, ...]      the recurrent state after each of the last K+1 positions: a boundary needs one

and a prefix snapshot keeps the same two per layer: ("conv", L) [C, conv-1] and ("rec", L) [...]. Saving a boundary
copies those cells out, restoring copies them back, and a prefill step that cuts its recurrence at a block boundary
writes its state and conv taps straight in (`mark_state`). The ring widths are read off the rings themselves, so a
model's speculative width (K) never has to be passed.

A conversation that leaves its slot for the NVMe tier keeps the same one recurrent cell (`live_bytes`): the other K
are draft positions' states, and most of a slot -- 238 of a GLM-5.3 slot's 286 MiB, 81 of a Qwen3.8 one's 109.

`StateRings` is a mixin for an engine/base/slot_caches.SlotCaches subclass whose layout carries those fields. The
subclass names its ring layers (`ring_layers`: the delta-rule ones, in layer order) and, if its conv is not
`F.conv` wide, `conv_history`. What else a boundary keeps -- a drafter's window, a lookup table's id ring -- is the
subclass's, around these calls.
"""
from __future__ import annotations


class StateRings:
    def ring_layers(self) -> tuple:
        raise NotImplementedError("the caches name their delta-rule layers")

    def conv_history(self) -> int:
        """Inputs before a position that the conv reads: its kernel width less one."""
        return self.F.conv - 1

    def rings(self, layer: int, slot: int):
        """(conv ring [C, W], recurrent-state ring [K+1, ...]) of `layer` in `slot`: views, not copies."""
        return self._fields["conv", layer][slot], self._fields["rec", layer][slot]

    def _boundary(self, slot: int, position: int, snap: int, what: str) -> None:
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError(f"{what} needs a real state slot and a declared snapshot")
        if position <= 0 or position % self.F.block:
            raise ValueError(f"a {what} sits at a block boundary")

    def _cells(self, position: int):
        """The conv ring cells before `position`, built once a ring width: one index tensor (one host-to-device copy)
        for every layer of a boundary, not one a layer."""
        taps, built = self.conv_history(), {}

        def of(width: int):
            if width not in built:
                built[width] = self._ring_cells(position, taps, width)
            return built[width]
        return of

    def save_rings(self, slot: int, position: int, snap: int) -> None:
        """Each ring layer's state at block boundary `position` of `slot` into snapshot `snap`."""
        self._boundary(slot, position, snap, "checkpoint")
        cells = self._cells(position)
        for L in self.ring_layers():
            conv, rec = self.rings(L, slot)
            self._snap["conv", L][snap].copy_(conv.index_select(1, cells(conv.shape[1])))
            self._snap["rec", L][snap].copy_(rec[(position - 1) % rec.shape[0]])

    def load_rings(self, slot: int, position: int, snap: int) -> None:
        """The inverse: `slot`'s rings continue from `position` with the snapshot's state."""
        self._boundary(slot, position, snap, "restore")
        cells = self._cells(position)
        for L in self.ring_layers():
            conv, rec = self.rings(L, slot)
            conv.index_copy_(1, cells(conv.shape[1]), self._snap["conv", L][snap])
            rec[(position - 1) % rec.shape[0]].copy_(self._snap["rec", L][snap])

    def live_bytes(self, slot: int, position: int):
        """What a conversation stopped after `position` tokens needs of `slot` to continue, as a tier moves it
        (engine/base/kv_tier.Segments): every field of the slot whole, but of each ring layer's recurrent ring only the
        state after position-1. A step reads (context-1) % (K+1) alone and writes every position it computes before
        anything reads it (engine/kernels/state `_read_rec`, engine/kernels/kda/fused_recurrent) -- the rule a restored
        prefix boundary already relies on -- so the other K cells, draft positions' states, are not the conversation's.
        A GLM-5.3 slot is 286 MiB and this 48; a Qwen3.8 one (K=3) 109 and 28. Reads the layout's `fields` and
        `slot_bytes`, and `state`."""
        from math import prod

        from engine.base.kv_tier import Segments
        from engine.base.slot_caches import SIZES

        if not 0 < slot < self.slots.num_slots:
            raise IndexError("only a real state slot has bytes to move")
        if position <= 0:
            raise ValueError("a conversation with nothing computed has no state to keep")
        rings = {("rec", L) for L in self.ring_layers()}
        base = slot * self.layout.slot_bytes
        views = []
        for f in sorted(self.layout.fields, key=lambda f: f.offset):
            at, size = base + f.offset, prod(f.shape) * SIZES[f.dtype]
            if (f.name, f.layer) in rings:
                size //= f.shape[0]                                  # one cell of the ring: [K + 1, ...]
                at += (position - 1) % f.shape[0] * size
            views.append(self.state[at:at + size])
        return Segments(views)

    def mark_state(self, layer: int, snap: int, state, taps) -> None:
        """A block boundary inside a prefill step: the layer's recurrent state there and the conv inputs of the conv-1
        positions before it [conv-1, C], straight into snapshot `snap` (the net cuts the recurrence at the mark)."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("a mark needs a declared snapshot")
        self._snap["rec", layer][snap].copy_(state)
        self._snap["conv", layer][snap].copy_(taps.T)


__all__ = ["StateRings"]
