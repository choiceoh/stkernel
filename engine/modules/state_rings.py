"""A delta-rule layer's state in a slot, and what a prefix boundary keeps of it (module: linear attention).

A layer with a short causal conv and a recurrent state (KDA, GDN -- any delta-rule linear attention) keeps two rings in
each sequence's state slot, both addressed BY POSITION so a rejected draft is overwritten by the next step's writes:

    ("conv", L)   [C, conv-1 + K]   the conv's inputs by position: a boundary needs the conv-1 before it
    ("rec", L)    [K + 1, ...]      the recurrent state after each of the last K+1 positions: a boundary needs one

and a prefix snapshot keeps the same two per layer: ("conv", L) [C, conv-1] and ("rec", L) [...]. Saving a boundary
copies those cells out, restoring copies them back, and a prefill step that cuts its recurrence at a block boundary
writes its state and conv taps straight in (`mark_state`). The ring widths are read off the rings themselves, so a
model's speculative width (K) never has to be passed.

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

    def mark_state(self, layer: int, snap: int, state, taps) -> None:
        """A block boundary inside a prefill step: the layer's recurrent state there and the conv inputs of the conv-1
        positions before it [conv-1, C], straight into snapshot `snap` (the net cuts the recurrence at the mark)."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("a mark needs a declared snapshot")
        self._snap["rec", layer][snap].copy_(state)
        self._snap["conv", layer][snap].copy_(taps.T)


__all__ = ["StateRings"]
