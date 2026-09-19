"""Qwen3.8-Flash-Next's caches as the served net reads them (profile): paged blocks and slot rings, owned by one arena.

What a sequence carries, from the reference features (engine/modules; engine/profiles/qwen38/composition.py):

    paged, per block of F.block tokens, per QSA layer (12 and the MTP head's)
        K rows      [block, 1, 256] bf16    the rank's KV head, normalised and rotated
        V rows      [block, 1, 256] bf16
        index keys  [block/4, 1, 128] bf16  one pooled, normalised, rotated key per complete group of 4 positions
    slots, per sequence
        GDN, per layer: conv ring [2560, conv-1+K] bf16 raw inputs by position; state ring [K+1, 12, 128, 128] fp32
        QSA, per layer: the raw index-key ring [8, 1, 128] bf16 by position -- the members of the group still open
        PLE: the token-id ring [8] int64 and the conv-input ring [10240, 9+K+1] bf16, both by position

Every value is addressed BY POSITION (GLM-5.3's caches' rule): a rejected draft is overwritten by the next step's writes
at the same positions, and a step reads the state after position ctx-1. Each block is one NVMe transfer unit.

The paged regions are presented to the kernels in the layout they take (engine/kernels/qsa: [pages, page_size, heads,
dim] with a page table of physical pages per request): typed strided views of the block storage, one per layer, so the
kernels never learn the block-major layout. A block's page for the index keys is its F.block/4 records.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lcm, prod

from engine.base.arena import ALIGN
from engine.base.kv import BlockPool, SlotPool
from engine.base.slot_caches import SIZES, SlotCaches, StateField, aligned, blocks_for, region_bytes, snapshots_for, typed_view
from engine.modules.state_rings import StateRings

QSA_KEY_RING = 8            # raw index keys kept per sequence: the open group (3) plus a verify step (K+1), no aliasing
PLE_ID_RING = 8             # token ids kept per sequence: the n-gram's previous 2 plus a verify step (check_rings: K <= 4)


@dataclass(frozen=True)
class CacheLayout:
    block_bytes: int
    slot_bytes: int
    kv_offsets: dict            # layer -> byte offset of its K rows in a block (V rows follow)
    key_offsets: dict           # layer -> byte offset of its index-key records in a block
    fields: tuple

    def nbytes(self, num_blocks: int, max_seqs: int) -> int:
        return region_bytes(num_blocks, max_seqs, self.block_bytes, self.slot_bytes)


def qsa_layers(F, layers, mtp: bool):
    """The attention layers with paged rows: the target's QSA layers in `layers`, then the MTP head's (model layer
    F.layers) when it is served."""
    return [L for L in layers if F.is_qsa(L)] + ([F.layers] if mtp else [])


def check_rings(F) -> None:
    """The fixed rings hold one launch's positions and what they read before them without aliasing a cell: a verify
    step of spec_k+1 tokens after the open QSA group's idx_ratio-1 members (the raw index-key ring), or after the
    n-gram's ngram_size-1 previous ids (the PLE id ring). The MTP head's chain adds one position a launch, after
    those, so it needs no more. The rings derived from spec_k (GDN's, PLE's conv) size themselves."""
    if F.idx_ratio - 1 + F.spec_k + 1 > QSA_KEY_RING:
        raise ValueError(f"a verify step of {F.spec_k + 1} tokens after the open group's {F.idx_ratio - 1} members "
                         f"passes the raw index-key ring of {QSA_KEY_RING} (spec_k <= {QSA_KEY_RING - F.idx_ratio - 1})")
    if F.ngram_size - 1 + F.spec_k + 1 > PLE_ID_RING:
        raise ValueError(f"a verify step of {F.spec_k + 1} tokens after the n-gram's {F.ngram_size - 1} previous ids "
                         f"passes the PLE id ring of {PLE_ID_RING} (spec_k <= {PLE_ID_RING - F.ngram_size - 1})")


def layout(F, layers, *, mtp: bool = True) -> CacheLayout:
    """Declared byte offsets; no CUDA allocation."""
    check_rings(F)
    layers = tuple(layers)
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= L < F.layers for L in layers):
        raise ValueError("cache layers must be nonempty, unique and inside the model")
    if F.block % F.idx_ratio or F.block % 64:
        raise ValueError("a block holds whole QSA groups and whole GDN kernel chunks")
    kv_row = F.kv_heads_local * F.head_dim * 2                          # bytes of one position's K (and of its V)
    key_row = F.idx_dim * 2
    quantum = lcm(4096, kv_row, key_row)
    kv_offsets, key_offsets, fields = {}, {}, []
    paged = state = 0

    def field(name, L, shape, dtype):
        nonlocal state
        state = aligned(state, ALIGN)
        fields.append(StateField(name, L, shape, dtype, state))
        state += prod(shape) * SIZES[dtype]

    for L in qsa_layers(F, layers, mtp):
        paged = aligned(paged, kv_row)
        kv_offsets[L] = paged
        paged += 2 * F.block * kv_row
        paged = aligned(paged, key_row)
        key_offsets[L] = paged
        paged += (F.block // F.idx_ratio) * key_row
        field("keys", L, (QSA_KEY_RING, 1, F.idx_dim), "bf16")
    for L in layers:
        if not F.is_qsa(L):
            field("conv", L, (F.qkv_local, F.conv - 1 + F.spec_k), "bf16")
            field("rec", L, (F.spec_k + 1, F.v_heads_local, F.k_dim, F.v_dim), F.gdn_state_dtype.replace("fp", "f"))
    if any(L in F.ple_layers for L in layers):
        field("ple_ids", -1, (PLE_ID_RING,), "i64")
        field("ple_conv", -1, (F.hc * F.hidden, (F.ple_conv - 1) * F.ngram_size + F.spec_k + 1), "bf16")
    return CacheLayout(aligned(max(1, paged), quantum), aligned(state, ALIGN), kv_offsets, key_offsets, tuple(fields))


def snapshot_layout(F, layers):
    """What a prefix checkpoint at block boundary P keeps: per GDN layer the conv's last conv-1 inputs and the state at
    P-1; PLE's previous ngram_size-1 token ids and its conv's (ple_conv-1)*ngram_size inputs (the conv is dilated by
    ngram_size). A block is whole QSA groups, so the index-key ring holds nothing at P. Returns (nbytes, fields)."""
    fields, at = [], 0

    def field(name, L, shape, dtype):
        nonlocal at
        at = aligned(at, ALIGN)
        fields.append(StateField(name, L, shape, dtype, at))
        at += prod(shape) * SIZES[dtype]

    for L in layers:
        if not F.is_qsa(L):
            field("conv", L, (F.qkv_local, F.conv - 1), "bf16")
            field("rec", L, (F.v_heads_local, F.k_dim, F.v_dim), F.gdn_state_dtype.replace("fp", "f"))
    if any(L in F.ple_layers for L in layers):
        field("ple_ids", -1, (F.ngram_size - 1,), "i64")
        field("ple_conv", -1, (F.hc * F.hidden, (F.ple_conv - 1) * F.ngram_size), "bf16")
    return aligned(max(at, 1), ALIGN), tuple(fields)


def cache_capacity(F, layers, kv_gib: float, max_seqs: int, snapshot_gib: float, *, mtp: bool = True):
    p = layout(F, layers, mtp=mtp)
    return blocks_for(kv_gib, max_seqs, p.block_bytes, p.slot_bytes), snapshots_for(snapshot_gib, snapshot_layout(F, layers)[0])


class Qwen38Caches(SlotCaches, StateRings):
    def __init__(self, arena, F, layers, num_blocks: int, max_seqs: int, snapshots: int = 0, *, mtp: bool = True):
        import torch
        self.F, self.layers, self.mtp = F, tuple(layers), mtp
        self.layout = layout(F, self.layers, mtp=mtp)
        self.snapshot_bytes_n, self._snapshot_fields = snapshot_layout(F, self.layers)
        self.snapshots = snapshots
        self.pool = BlockPool(num_blocks, F.block, max_seqs, num_blocks)
        self.slots = SlotPool(max_seqs + 1)
        p = self.layout
        if aligned(arena.used, ALIGN) + p.nbytes(num_blocks, max_seqs) + snapshots * self.snapshot_bytes_n > arena.nbytes:
            raise MemoryError("arena cannot hold the declared Qwen3.8 caches, block table and prefix snapshots")
        self.paged = arena.carve(num_blocks * p.block_bytes, "qwen38 paged KV")
        self.device = self.paged.device
        self.pool.attach_storage(self.paged, p.block_bytes)
        self.state = arena.carve((max_seqs + 1) * p.slot_bytes, "qwen38 state slots")
        self.block_table = arena.carve(max_seqs * num_blocks * 4, "qwen38 block table").view(torch.int32).view(max_seqs, num_blocks)
        self._fields = {}
        for f in p.fields:
            self._fields[f.name, f.layer] = typed_view(self.state, max_seqs + 1, p.slot_bytes, f)
        self._snap = {}
        if snapshots:
            self.snapshot_store = arena.carve(snapshots * self.snapshot_bytes_n, "qwen38 prefix snapshots")
            for f in self._snapshot_fields:
                self._snap[f.name, f.layer] = typed_view(self.snapshot_store, snapshots, self.snapshot_bytes_n, f)
        self._kv, self._keys = {}, {}
        kv_row = F.kv_heads_local * F.head_dim * 2
        bf = self.paged.view(torch.bfloat16)
        for L, offset in p.kv_offsets.items():
            rows = p.block_bytes // 2
            # [pages, block, kv heads, head_dim]: page stride is the whole block, rows within it are consecutive
            k = bf.as_strided((num_blocks, F.block, F.kv_heads_local, F.head_dim),
                              (rows, F.kv_heads_local * F.head_dim, F.head_dim, 1), bf.storage_offset() + offset // 2)
            v = bf.as_strided((num_blocks, F.block, F.kv_heads_local, F.head_dim),
                              (rows, F.kv_heads_local * F.head_dim, F.head_dim, 1),
                              bf.storage_offset() + (offset + F.block * kv_row) // 2)
            self._kv[L] = (k, v)
        for L, offset in p.key_offsets.items():
            rows = p.block_bytes // 2
            self._keys[L] = bf.as_strided((num_blocks, F.block // F.idx_ratio, 1, F.idx_dim),
                                          (rows, F.idx_dim, F.idx_dim, 1), bf.storage_offset() + offset // 2)
        self.reset()

    def reset(self):
        """Clear contents at boot and check boundaries; ownership is unchanged. PLE's id ring starts DEAD (-1)."""
        super().reset()
        if ("ple_ids", -1) in self._fields:
            self._fields["ple_ids", -1].fill_(-1)

    def reset_slot(self, slot: int):
        super().reset_slot(slot)
        if ("ple_ids", -1) in self._fields:
            self._fields["ple_ids", -1][slot].fill_(-1)

    # -- typed views the net and the lanes read --------------------------------------------------------------------
    def kv(self, layer: int):
        """(K, V) [pages, block, kv heads, head_dim] bf16: the layer's paged rows for the QSA kernels."""
        return self._kv[layer]

    def index_keys(self, layer: int):
        """[pages, block/4, 1, idx_dim] bf16: the layer's pooled index keys, one page of records per block."""
        return self._keys[layer]

    def key_ring(self, layer: int):
        """[slots, 8, 1, idx_dim] bf16: every slot's raw index keys by position % 8 (the compressor-state ring)."""
        return self._fields["keys", layer]

    gdn = StateRings.rings

    def gdn_fields(self, layer: int):
        """Every slot's (conv ring [slots, C, W], state ring [slots, K+1, HV, K, V]): what the row kernels address."""
        return self._fields["conv", layer], self._fields["rec", layer]

    def ple(self, slot: int):
        return self._fields["ple_ids", -1][slot], self._fields["ple_conv", -1][slot]

    def ple_fields(self):
        return self._fields["ple_ids", -1], self._fields["ple_conv", -1]

    def flat_slots(self, layer_rows: int, seq: int, positions):
        """Flat row slots page * layer_rows + offset for the positions of one sequence (qsa_store_cache_rows' address):
        `layer_rows` is the page size of the view written (F.block for K/V, F.block // idx_ratio for index keys)."""
        F = self.F
        per = F.block if layer_rows == F.block else F.block // F.idx_ratio
        pages = self.block_table[seq][(positions // per).long()]
        return (pages * layer_rows + positions % per).to(self.block_table.dtype)

    # -- prefix snapshots at block boundaries --------------------------------------------------------------------
    def checkpoint(self, slot: int, position: int, snap: int) -> None:
        self.save_rings(slot, position, snap)
        F = self.F
        if ("ple_ids", -1) in self._snap:
            ids, conv = self.ple(slot)
            self._snap["ple_ids", -1][snap].copy_(ids.index_select(0, self._ring_cells(position, F.ngram_size - 1, ids.shape[0])))
            span = (F.ple_conv - 1) * F.ngram_size
            self._snap["ple_conv", -1][snap].copy_(conv.index_select(1, self._ring_cells(position, span, conv.shape[1])))

    def restore(self, slot: int, position: int, snap: int) -> None:
        self.load_rings(slot, position, snap)
        F = self.F
        if ("ple_ids", -1) in self._snap:
            ids, conv = self.ple(slot)
            ids.index_copy_(0, self._ring_cells(position, F.ngram_size - 1, ids.shape[0]), self._snap["ple_ids", -1][snap])
            span = (F.ple_conv - 1) * F.ngram_size
            conv.index_copy_(1, self._ring_cells(position, span, conv.shape[1]), self._snap["ple_conv", -1][snap])

    mark_gdn = StateRings.mark_state

    def ring_layers(self) -> tuple:
        """The GDN layers: every layer that is not QSA."""
        return tuple(L for L in self.layers if not self.F.is_qsa(L))

    def mark_ple(self, snap: int, ids, taps) -> None:
        """PLE at a block boundary inside a prefill chunk: the previous ngram_size-1 ids and the conv inputs
        [(ple_conv-1)*ngram_size, 10240] before it."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("a mark needs a declared snapshot")
        self._snap["ple_ids", -1][snap].copy_(ids)
        self._snap["ple_conv", -1][snap].copy_(taps.T)




def caches(tp: int = 4) -> "list[Cache]":
    """The per-sequence and per-token cost table (engine/base/caches) the budget reads: sizes only, no layout."""
    from engine.base.caches import Cache
    from engine.profiles.qwen38.plan import text_config, state_bytes
    c = text_config()
    per_seq, kv_tok, idx_tok = state_bytes(c, tp)
    n_lin = c["layer_types"].count("linear_attention"); n_full = len(c["layer_types"]) - n_lin
    conv_dim = c["linear_key_head_dim"] * c["linear_num_key_heads"] * 2 + c["linear_value_head_dim"] * c["linear_num_value_heads"]
    conv = n_lin * (conv_dim // tp) * (c["linear_conv_kernel_dim"] - 1) * 2
    return [
        Cache("linear conv state", n_lin, 0.0, conv, 0.0, f"[{conv_dim // tp}, {c['linear_conv_kernel_dim'] - 1}] bf16 per GDN layer; a slot, not a page"),
        Cache("linear recurrent state", n_lin, 0.0, per_seq - conv, 0.0,
              f"[{c['linear_num_value_heads'] // tp}, {c['linear_value_head_dim']}, {c['linear_key_head_dim']}] fp32 (mamba_ssm_dtype) per GDN layer"),
        Cache("full-attn kv", n_full, kv_tok, 0.0, 0.0,
              f"1 kv head per rank (2 < TP {tp}, replicated) x {c['head_dim']} x k,v x bf16 per QSA layer -- paged"),
        Cache("qsa compressed keys", n_full, idx_tok, 0.0, 0.0,
              f"[T/{c['indexer_compress_ratio']}, {c['indexer_head_dim']}] bf16 per QSA layer"),
    ]


__all__ = ["QSA_KEY_RING", "PLE_ID_RING", "CacheLayout", "check_rings", "layout", "snapshot_layout", "cache_capacity",
           "qsa_layers",
           "Qwen38Caches", "caches"]


if __name__ == "__main__":
    from engine.base.caches import GIB, total_bytes, max_seq
    from engine.profiles.qwen38.plan import text_config
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--kv-gib", type=float, default=40.0); a = ap.parse_args()
    cs = caches(); cfg = text_config(); S = 131072
    for k in cs:
        at = k.per_batch_per_token * S + k.per_batch + k.per_token * S
        print(f"  {k.name:22s} {k.blocks:>3} layers  {at / GIB:8.4f} GiB @ B=1 S=128K   {k.note}")
    print(f"  total @ B=1 S=128K: {total_bytes(cs, 1, S) / GIB:.3f} GiB")
    print(f"  {a.kv_gib:.0f} GiB buys:")
    for b in (1, 8, 32, 128):
        print(f"    concurrency {b:>3}: {min(max_seq(cs, a.kv_gib, b), cfg['max_position_embeddings']):>9,} tok")
