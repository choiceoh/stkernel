"""GLM-5.3's caches in the arena (profile): what a sequence's state is, byte
for byte, and how a block table addresses it.

Two allocators from base/kv, one arena (D16):

  paged, per DSA layer (11), grows with context, block = 2304 tokens:
      latent       [num_blocks * 2304, 512]  e4m3   the MLA latent, nope-only, scale 1
      pool keys    [num_blocks * 576, 128]   e4m3   kpool-compressed, FWHT-rotated keys (4 tokens -> 1)
      pool scales  [num_blocks * 576]        f32
  per slot, fixed per live sequence:
      KDA conv ring      [C=3*16*128, 3+K]    bf16   per KDA layer (34), by position % width
      KDA recurrent ring [K+1, 16, 128, 128]  f32    per KDA layer, one state per draft position
      indexer tail ring  [4, 2, 128]          bf16   per DSA layer: raw k and gate score of the in-progress pool

Every region is per LAYER and flat, so a block id means the same offset in
each layer's region (that is the layout the served kernels take: a flat
latent with global slot ids) and the NVMe tier demotes a block as 33 slices.
Block b of layer L costs 1,255,680 B; a slot 207 MiB (K=5). The numbers
plan.state_bytes reports are these.

The address translation is here and nowhere else: a position `p` of a
sequence whose block row is `row` lives at slot `row[p // 2304] * 2304 + p %
2304`; pool `j` at `row[j // 576] * 576 + j % 576`. The model asks through
the `Caches` protocol (net.py) and never sees a block table.
"""
from __future__ import annotations

from array import array

import torch

from engine.base.arena import Arena
from engine.base.kv import BlockPool, EMPTY, SlotPool
from engine.profiles.glm53.facts import Facts
from engine.profiles.glm53.net import BF16, E4M3, F32

GIB = 1 << 30


def block_bytes(F: Facts, layers) -> int:
    """One block, all paged layers of `layers` together."""
    n_dsa = sum(1 for L in layers if F.is_dsa(L))
    return n_dsa * (F.block * F.kv_lora + (F.block // F.kpool) * (F.idx_dim + 4))


def slot_bytes(F: Facts, layers, hk: int) -> int:
    wc, wr = F.conv - 1 + F.spec_k, F.spec_k + 1
    n_kda = sum(1 for L in layers if not F.is_dsa(L)); n_dsa = len(layers) - n_kda
    return (n_kda * (3 * hk * F.kda_dim * wc * 2 + wr * hk * F.kda_dim * F.kda_dim * 4)
            + n_dsa * F.kpool * 2 * F.idx_dim * 2)


class Glm53Caches:
    """The `Caches` protocol over base/kv pools, carved from the arena."""

    def __init__(self, arena: Arena, F: Facts, layers, hk: int, num_blocks: int, num_slots: int, max_seqs: int, draft=None):
        """`draft` = (layers, window, kv_heads, head_dim) adds the drafter's context K/V ring per slot (drafter.py)."""
        self.F, self.layers, self.B, self.kp = F, list(layers), F.block, F.kpool
        self._draft = None
        if draft is not None:
            dl, dw, dkv, dd = draft
            self._draft = arena.carve(num_slots * dl * 2 * dw * dkv * dd * 2, "drafter context rings").view(BF16).view(num_slots, dl, 2, dw, dkv, dd)
        self.bp = F.block // F.kpool                          # pools per block
        wc, wr = F.conv - 1 + F.spec_k, F.spec_k + 1
        self._lat, self._pk, self._ps, self._tail, self._kda = {}, {}, {}, {}, {}
        for L in self.layers:
            if F.is_dsa(L):
                self._lat[L] = arena.carve(num_blocks * self.B * F.kv_lora, f"L{L} latent").view(E4M3).view(num_blocks * self.B, F.kv_lora)
                self._pk[L] = arena.carve(num_blocks * self.bp * F.idx_dim, f"L{L} pool keys").view(E4M3).view(num_blocks * self.bp, F.idx_dim)
                self._ps[L] = arena.carve(num_blocks * self.bp * 4, f"L{L} pool scales").view(F32)
                self._tail[L] = arena.carve(num_slots * self.kp * 2 * F.idx_dim * 2, f"L{L} tail ring").view(BF16).view(num_slots, self.kp, 2, F.idx_dim)
            else:
                c = 3 * hk * F.kda_dim
                conv = arena.carve(num_slots * c * wc * 2, f"L{L} conv ring").view(BF16).view(num_slots, c, wc)
                rec = arena.carve(num_slots * wr * hk * F.kda_dim * F.kda_dim * 4, f"L{L} recurrent ring").view(F32).view(num_slots, wr, hk, F.kda_dim, F.kda_dim)
                self._kda[L] = (conv, rec)
        self.blocks = BlockPool(num_blocks, self.B, max_seqs=max_seqs, max_blocks_per_seq=num_blocks)
        self.slots = SlotPool(num_slots)
        self._rows = {}                                        # seq -> device int32 block row (refreshed by sync_row)
        self.device = (next(iter(self._lat.values())) if self._lat else next(iter(self._kda.values()))[0]).device

    # -- block tables ------------------------------------------------------------
    def sync_row(self, seq: int) -> None:
        """Copy the pool's block row for `seq` to the device (after reserve/release)."""
        row = self.blocks.row(seq)
        n = 0
        for b in row:
            if b == EMPTY:
                break
            n += 1
        self._rows[seq] = torch.tensor(list(row[:n]), dtype=torch.int32, device=self.device)

    def reserve(self, seq: int, tokens: int) -> None:
        self.blocks.reserve(seq, tokens); self.sync_row(seq)

    def release(self, seq: int) -> None:
        self.blocks.release(seq); self._rows.pop(seq, None)

    def token_slots(self, seq: int, positions: torch.Tensor) -> torch.Tensor:
        row = self._rows[seq]
        return (row[positions // self.B] * self.B + positions % self.B).to(torch.int32)

    def pool_slots(self, seq: int, pool_ids: torch.Tensor) -> torch.Tensor:
        row = self._rows[seq]
        return (row[pool_ids // self.bp] * self.bp + pool_ids % self.bp).to(torch.int32)

    # -- regions ---------------------------------------------------------------------
    def kda(self, layer, slot): conv, rec = self._kda[layer]; return conv[slot], rec[slot]
    def latent(self, layer): return self._lat[layer]
    def pool_keys(self, layer): return self._pk[layer]
    def pool_scales(self, layer): return self._ps[layer]
    def tail(self, layer, slot): return self._tail[layer][slot]

    def draft_ring(self, slot: int) -> torch.Tensor:
        return self._draft[slot]

    def clear_slot(self, slot: int) -> None:
        """A fresh sequence starts from nothing: the rings it inherits are zeroed
        (a position-addressed ring never reads before it writes, but zero is
        the honest state and the conv history before position 0 IS zero)."""
        for conv, rec in self._kda.values():
            conv[slot].zero_(); rec[slot].zero_()
        for t in self._tail.values():
            t[slot].zero_()
        if self._draft is not None:
            self._draft[slot].zero_()

    def regions(self):
        """(name, storage, bytes per block) for every paged region: the tier's view."""
        for L in self.layers:
            if self.F.is_dsa(L):
                yield f"L{L} latent", self._lat[L].view(torch.uint8), self.B * self.F.kv_lora
                yield f"L{L} pool keys", self._pk[L].view(torch.uint8), self.bp * self.F.idx_dim
                yield f"L{L} pool scales", self._ps[L].view(torch.uint8), self.bp * 4


def _selfcheck() -> None:
    from engine.profiles.glm53 import facts, plan
    F = facts.load()
    layers = list(range(0, 5)); hk = F.kda_heads_local
    # the profile's two numbers, from this file's arithmetic
    per_seq, kv_tok, idx_tok = plan.state_bytes(plan.text_config())
    assert slot_bytes(F, range(F.layers), hk) == per_seq, (slot_bytes(F, range(F.layers), hk), per_seq)
    assert block_bytes(F, range(F.layers)) == 11 * 1_255_680
    nb, ns = 3, 3
    need = nb * block_bytes(F, layers) + ns * slot_bytes(F, layers, hk) + 20 * 256
    arena = Arena(need + (1 << 20))
    c = Glm53Caches(arena, F, layers, hk, nb, ns, max_seqs=2)
    assert c.slots.take(0) == 1                                   # slot 0 is never issued
    c.reserve(0, 2304 + 10)                                       # two blocks: ids come off the free stack top (2, 1)
    row = list(c.blocks.row(0))[:2]
    pos = torch.tensor([0, 2303, 2304, 2313], device=c.device)
    got = c.token_slots(0, pos).tolist()
    assert got == [row[0] * 2304, row[0] * 2304 + 2303, row[1] * 2304, row[1] * 2304 + 9], (got, row)
    pools = c.pool_slots(0, torch.tensor([0, 575, 576], device=c.device)).tolist()
    assert pools == [row[0] * 576, row[0] * 576 + 575, row[1] * 576]
    conv, rec = c.kda(0, 1); assert conv.shape == (3 * hk * 128, 8) and rec.shape == (6, hk, 128, 128)
    assert c.latent(3).shape == (nb * 2304, 512) and c.tail(3, 1).shape == (4, 2, 128)
    assert len(list(c.regions())) == 3
    c.release(0); assert c.blocks.available == nb
    print(f"  caches: block {block_bytes(F, range(F.layers)) / 2**20:.2f} MiB (11 dsa layers), slot {per_seq / 2**20:.1f} MiB; "
          f"block-table translation (token, pool) and per-layer regions on {len(arena.regions)} arena carves OK")


if __name__ == "__main__":
    _selfcheck()
