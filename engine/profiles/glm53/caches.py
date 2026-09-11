"""GLM's paged KV and position rings, owned by one arena.

Each physical block contains every DSA layer's latent rows and pooled keys
with scales. That makes a block an NVMe transfer unit without repacking.
Kernel slot ids are absolute rows in typed views of the same byte storage;
latent rows remain contiguous, including for the served MLA pointer API.
KDA and indexer rings live together in a second, slot-major arena region.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lcm, prod

from engine.base.arena import ALIGN
from engine.base.kv import BlockPool, SlotPool


def aligned(n: int, unit: int) -> int:
    return -(-n // unit) * unit


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
    token_offsets: dict
    pool_offsets: dict
    fields: tuple

    def nbytes(self, num_blocks: int, max_seqs: int) -> int:
        # The block table has one row per request id; state slot 0 is null.
        return (num_blocks * self.block_bytes + (max_seqs + 1) * self.slot_bytes
                + max_seqs * num_blocks * 4)


def layout(F, layers, draft=None) -> CacheLayout:
    """Declared byte offsets; no CUDA allocation or model execution."""
    layers = tuple(layers)
    if not layers or len(set(layers)) != len(layers) or any(not 0 <= L < F.layers for L in layers):
        raise ValueError("cache layers must be nonempty, unique and inside the model")
    if F.block <= 0 or F.kpool <= 0 or F.block % F.kpool:
        raise ValueError("cache blocks must contain whole indexer pools")
    record = F.idx_dim + 4                       # fp8 key plus its fp32 scale
    quantum = lcm(4096, F.kv_lora, record)
    token_offsets, pool_offsets, fields = {}, {}, []
    paged = state = 0

    def field(name, L, shape, dtype):
        nonlocal state
        state = aligned(state, ALIGN)
        fields.append(StateField(name, L, shape, dtype, state))
        state += prod(shape) * (4 if dtype == "f32" else 2)

    for L in layers:
        if F.is_dsa(L):
            paged = aligned(paged, F.kv_lora)
            token_offsets[L] = paged
            paged += F.block * F.kv_lora
            paged = aligned(paged, record)
            pool_offsets[L] = paged
            paged += (F.block // F.kpool) * record
            field("tail", L, (F.kpool - 1 + F.spec_k, 2, F.idx_dim), "bf16")
        else:
            field("conv", L, (3 * F.kda_heads_local * F.kda_dim, F.conv - 1 + F.spec_k), "bf16")
            field("rec", L, (F.spec_k + 1, F.kda_heads_local, F.kda_dim, F.kda_dim), "f32")
    if draft is not None:
        if len(draft) != 4 or any(not isinstance(n, int) or n <= 0 for n in draft):
            raise ValueError("draft cache shape must contain four positive dimensions")
        dl, dw, dkv, dd = draft
        field("draft", -1, (dl, 2, dw, dkv, dd), "bf16")
    # A KDA-only slice still has logical blocks for the runner's token ledger.
    return CacheLayout(aligned(max(1, paged), quantum), aligned(state, ALIGN),
                       token_offsets, pool_offsets, tuple(fields))


class Glm53Caches:
    def __init__(self, arena, F, layers, num_blocks: int, max_seqs: int, draft=None):
        import torch

        self.F, self.layers = F, tuple(layers)
        self.layout = layout(F, self.layers, draft)
        self.pool = BlockPool(num_blocks, F.block, max_seqs, num_blocks)
        self.slots = SlotPool(max_seqs + 1)
        p = self.layout
        # Preflight all regions, including alignment at an existing arena cursor.
        if aligned(arena.used, ALIGN) + p.nbytes(num_blocks, max_seqs) > arena.nbytes:
            raise MemoryError("arena cannot hold the declared GLM caches and block table")
        self.paged = arena.carve(num_blocks * p.block_bytes, "glm53 paged KV")
        self.device = self.paged.device
        self.pool.attach_storage(self.paged, p.block_bytes)
        self.state = arena.carve((max_seqs + 1) * p.slot_bytes, "glm53 state slots")
        self.block_table = arena.carve(max_seqs * num_blocks * 4, "glm53 block table").view(torch.int32).view(max_seqs, num_blocks)
        self._fields = {}
        for f in p.fields:
            dtype = torch.float32 if f.dtype == "f32" else torch.bfloat16
            size = 4 if f.dtype == "f32" else 2
            strides = tuple(prod(f.shape[i + 1:]) for i in range(len(f.shape)))
            base = self.state.view(dtype)
            self._fields[f.name, f.layer] = base.as_strided(
                (max_seqs + 1, *f.shape), (p.slot_bytes // size, *strides),
                base.storage_offset() + f.offset // size)
        record = F.idx_dim + 4
        self._latent = self.paged.view(torch.float8_e4m3fn).view(-1, F.kv_lora)
        self._keys = self.paged.as_strided((self.paged.numel() // record, F.idx_dim),
                                         (record, 1)).view(torch.float8_e4m3fn)
        base = self.paged.view(torch.float32)
        self._scales = base.as_strided((self.paged.numel() // record,), (record // 4,),
                                      base.storage_offset() + F.idx_dim // 4)
        self.reset()

    def reset(self):
        """Clear contents at boot/check boundaries; does not change ownership."""
        self.paged.zero_()
        self.state.zero_()
        self.block_table.fill_(-1)

    def reset_slot(self, slot: int):
        if not 0 < slot < self.slots.num_slots:
            raise IndexError("only a real state slot may be reset")
        n = self.layout.slot_bytes
        self.state[slot * n:(slot + 1) * n].zero_()

    def prepare(self, step):
        """Publish host block rows once before a step, after its reservation.

        Segment bounds and slot ownership are checked without device reads.
        Address translation in the model then only gathers from this table.
        """
        import torch

        for s in step.segments:
            row = self.pool.row(s.seq)
            if not 0 < s.slot < self.slots.num_slots or self.slots.owner[s.slot] != s.seq:
                raise ValueError(f"seq {s.seq} does not own state slot {s.slot}")
            if s.ctx < 0 or s.length <= 0 or s.ctx + s.length > self.pool.tokens[s.seq]:
                raise ValueError(f"seq {s.seq} step exceeds its reserved context")
            self.block_table[s.seq].copy_(torch.tensor(row, dtype=torch.int32))

    def kda(self, layer, slot):
        return self._fields["conv", layer][slot], self._fields["rec", layer][slot]

    def tail(self, layer, slot):
        return self._fields["tail", layer][slot]

    def draft_ring(self, slot):
        return self._fields["draft", -1][slot]

    def latent(self, layer):
        return self._latent

    def pool_keys(self, layer):
        return self._keys

    def pool_scales(self, layer):
        return self._scales

    def token_slots(self, layer, seq, positions):
        F, p = self.F, self.layout
        blocks = self.block_table[seq][(positions // F.block).long()]
        return (blocks * (p.block_bytes // F.kv_lora)
                + p.token_offsets[layer] // F.kv_lora + positions % F.block).to(blocks.dtype)

    def pool_slots(self, layer, seq, pool_ids):
        F, p = self.F, self.layout
        per, record = F.block // F.kpool, F.idx_dim + 4
        blocks = self.block_table[seq][(pool_ids // per).long()]
        return (blocks * (p.block_bytes // record)
                + p.pool_offsets[layer] // record + pool_ids % per).to(blocks.dtype)
