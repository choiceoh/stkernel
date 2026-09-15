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

from array import array
from dataclasses import dataclass
from math import lcm, prod

from engine.base.arena import ALIGN
from engine.base.kv import BlockPool, SlotPool

QSA_KEY_RING = 8            # raw index keys kept per sequence: the open group (3) plus a verify step (K+1), no aliasing
PLE_ID_RING = 8             # token ids kept per sequence: the n-gram's previous 2 plus a verify step
SIZES = {"f32": 4, "f16": 2, "bf16": 2, "i64": 8}


def aligned(n: int, unit: int) -> int:
    return -(-n // unit) * unit


def field_dtype(name):
    import torch
    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16, "i64": torch.int64}[name]


@dataclass(frozen=True)
class StateField:
    name: str
    layer: int
    shape: tuple
    dtype: str
    offset: int


@dataclass(frozen=True)
class CacheLayout:
    block_bytes: int
    slot_bytes: int
    kv_offsets: dict            # layer -> byte offset of its K rows in a block (V rows follow)
    key_offsets: dict           # layer -> byte offset of its index-key records in a block
    fields: tuple

    def nbytes(self, num_blocks: int, max_seqs: int) -> int:
        return num_blocks * self.block_bytes + (max_seqs + 1) * self.slot_bytes + max_seqs * num_blocks * 4


def qsa_layers(F, layers, mtp: bool):
    """The attention layers with paged rows: the target's QSA layers in `layers`, then the MTP head's (model layer
    F.layers) when it is served."""
    return [L for L in layers if F.is_qsa(L)] + ([F.layers] if mtp else [])


def layout(F, layers, *, mtp: bool = True) -> CacheLayout:
    """Declared byte offsets; no CUDA allocation."""
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
        field("ple_conv", -1, (F.hc * F.hidden, (F.ple_conv - 1) * 3 + F.spec_k + 1), "bf16")
    return CacheLayout(aligned(max(1, paged), quantum), aligned(state, ALIGN), kv_offsets, key_offsets, tuple(fields))


def snapshot_layout(F, layers):
    """What a prefix checkpoint at block boundary P keeps: per GDN layer the conv's last conv-1 inputs and the state at
    P-1; PLE's previous ngram_size-1 token ids and its conv's (ple_conv-1)*3 inputs. A block is whole QSA groups, so
    the index-key ring holds nothing at P. Returns (nbytes, fields)."""
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
        field("ple_conv", -1, (F.hc * F.hidden, (F.ple_conv - 1) * 3), "bf16")
    return aligned(max(at, 1), ALIGN), tuple(fields)


def cache_capacity(F, layers, kv_gib: float, max_seqs: int, snapshot_gib: float, *, mtp: bool = True):
    p = layout(F, layers, mtp=mtp)
    blocks = int((kv_gib * (1 << 30) - (max_seqs + 1) * p.slot_bytes) // (p.block_bytes + max_seqs * 4))
    snapshots = max(9, int(snapshot_gib * (1 << 30)) // snapshot_layout(F, layers)[0])
    return blocks, snapshots


class Qwen38Caches:
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
            self._fields[f.name, f.layer] = self._typed(self.state, max_seqs + 1, p.slot_bytes, f)
        self._snap = {}
        if snapshots:
            self.snapshot_store = arena.carve(snapshots * self.snapshot_bytes_n, "qwen38 prefix snapshots")
            for f in self._snapshot_fields:
                self._snap[f.name, f.layer] = self._typed(self.snapshot_store, snapshots, self.snapshot_bytes_n, f)
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

    @staticmethod
    def _typed(storage, count, stride_bytes, f):
        dtype = field_dtype(f.dtype)
        size = SIZES[f.dtype]
        strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
        base = storage.view(dtype)
        return base.as_strided((count, *f.shape), (stride_bytes // size, *strides), base.storage_offset() + f.offset // size)

    def reset(self):
        """Clear contents at boot and check boundaries; ownership is unchanged. PLE's id ring starts DEAD (-1)."""
        self.paged.zero_()
        self.state.zero_()
        if ("ple_ids", -1) in self._fields:
            self._fields["ple_ids", -1].fill_(-1)
        self.block_table.fill_(-1)
        self._table_blocks = array("i", [0]) * self.pool.max_seqs
        self._table_epochs = array("Q", self.pool.epochs)

    def slot_bytes(self, slot: int):
        if not 0 < slot < self.slots.num_slots:
            raise IndexError("only a real state slot has bytes to move")
        n = self.layout.slot_bytes
        return self.state[slot * n:(slot + 1) * n]

    def reset_slot(self, slot: int):
        self.slot_bytes(slot).zero_()
        if ("ple_ids", -1) in self._fields:
            self._fields["ple_ids", -1][slot].fill_(-1)

    def snapshot_bytes(self, snap: int):
        if not 0 <= snap < self.snapshots:
            raise IndexError("only a declared snapshot has bytes to move")
        n = self.snapshot_bytes_n
        return self.snapshot_store[snap * n:(snap + 1) * n]

    def prepare(self, step):
        """Publish changed block mappings before a step (engine/profiles/glm53/caches.Glm53Caches.prepare's contract:
        rows append within an allocator epoch, so only a new suffix is uploaded)."""
        import torch
        for s in step.segments:
            row = self.pool.row(s.seq)
            if not 0 < s.slot < self.slots.num_slots or self.slots.owner[s.slot] != s.seq:
                raise ValueError(f"seq {s.seq} does not own state slot {s.slot}")
            if s.ctx < 0 or s.length <= 0 or s.ctx + s.length > self.pool.tokens[s.seq]:
                raise ValueError(f"seq {s.seq} step exceeds its reserved context")
            count = self.pool.blocks_for(self.pool.tokens[s.seq])
            previous = self._table_blocks[s.seq]
            epoch = self.pool.epochs[s.seq]
            if epoch == self._table_epochs[s.seq] and count == previous:
                continue
            start = previous if epoch == self._table_epochs[s.seq] else 0
            end = max(count, previous)
            ids = torch.tensor(row[start:end], dtype=torch.int32)
            self.block_table[s.seq, start:end].copy_(ids, non_blocking=False)
            self._table_blocks[s.seq] = count
            self._table_epochs[s.seq] = epoch

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

    def gdn(self, layer: int, slot: int):
        return self._fields["conv", layer][slot], self._fields["rec", layer][slot]

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
    def _ring_cells(self, position: int, count: int, width: int):
        import torch
        return torch.tensor([(position - count + i) % width for i in range(count)], device=self.device)

    def checkpoint(self, slot: int, position: int, snap: int) -> None:
        F = self.F
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("checkpoint needs a real state slot and a declared snapshot")
        if position <= 0 or position % F.block:
            raise ValueError("a checkpoint sits at a block boundary")
        for L in self.layers:
            if F.is_qsa(L):
                continue
            conv, rec = self.gdn(L, slot)
            self._snap["conv", L][snap].copy_(conv.index_select(1, self._ring_cells(position, F.conv - 1, conv.shape[1])))
            self._snap["rec", L][snap].copy_(rec[(position - 1) % rec.shape[0]])
        if ("ple_ids", -1) in self._snap:
            ids, conv = self.ple(slot)
            self._snap["ple_ids", -1][snap].copy_(ids.index_select(0, self._ring_cells(position, F.ngram_size - 1, ids.shape[0])))
            span = (F.ple_conv - 1) * 3
            self._snap["ple_conv", -1][snap].copy_(conv.index_select(1, self._ring_cells(position, span, conv.shape[1])))

    def restore(self, slot: int, position: int, snap: int) -> None:
        F = self.F
        if not 0 <= snap < self.snapshots or not 0 < slot < self.slots.num_slots:
            raise IndexError("restore needs a real state slot and a declared snapshot")
        if position <= 0 or position % F.block:
            raise ValueError("a restore sits at a block boundary")
        for L in self.layers:
            if F.is_qsa(L):
                continue
            conv, rec = self.gdn(L, slot)
            conv.index_copy_(1, self._ring_cells(position, F.conv - 1, conv.shape[1]), self._snap["conv", L][snap])
            rec[(position - 1) % rec.shape[0]].copy_(self._snap["rec", L][snap])
        if ("ple_ids", -1) in self._snap:
            ids, conv = self.ple(slot)
            ids.index_copy_(0, self._ring_cells(position, F.ngram_size - 1, ids.shape[0]), self._snap["ple_ids", -1][snap])
            span = (F.ple_conv - 1) * 3
            conv.index_copy_(1, self._ring_cells(position, span, conv.shape[1]), self._snap["ple_conv", -1][snap])

    def mark_gdn(self, layer: int, snap: int, state, taps) -> None:
        """A block boundary inside a prefill chunk: the state there [HV, K, V] and the conv's conv-1 inputs before it
        [conv-1, C]."""
        if not 0 <= snap < self.snapshots:
            raise IndexError("a mark needs a declared snapshot")
        self._snap["rec", layer][snap].copy_(state)
        self._snap["conv", layer][snap].copy_(taps.T)

    def mark_ple(self, snap: int, ids, taps) -> None:
        """PLE at a block boundary inside a prefill chunk: the previous ngram_size-1 ids and the conv inputs
        [(ple_conv-1)*3, 10240] before it."""
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


__all__ = ["QSA_KEY_RING", "PLE_ID_RING", "CacheLayout", "layout", "snapshot_layout", "cache_capacity", "qsa_layers",
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
