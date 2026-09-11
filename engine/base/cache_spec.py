"""What each layer kind needs cached, declared once, sized from the budget (base).

vLLM's `kv_cache_interface` does this by asking every layer for a spec and
then running a 36-second profile to learn how many blocks fit. Here the
profile declares the specs and the budget already said how many bytes KV
gets (D1); this file only turns bytes into block counts and slot counts and
carves the arena (D16). Nothing is discovered.

Two kinds, matching base/kv.py's two allocators:

  PagedSpec   grows with context: bytes per token per layer -> blocks of
              `block_tokens` tokens across all layers that share the pool.
  SlotSpec    fixed per sequence: bytes per sequence per layer -> one slot.

A profile lists both; `plan()` splits a KV budget into (blocks, slots) for a
declared max concurrency and reports what context that buys -- the same
number the profile's caches.py computes, which is the cross-check.
"""
from __future__ import annotations

from dataclasses import dataclass

GIB = 1 << 30


@dataclass(frozen=True)
class PagedSpec:
    name: str
    layers: int
    bytes_per_token: int          # per layer
    source: str


@dataclass(frozen=True)
class SlotSpec:
    name: str
    layers: int
    bytes_per_seq: int            # per layer
    source: str


@dataclass(frozen=True)
class Plan:
    block_tokens: int
    block_bytes: int              # one block, all paged layers together
    num_blocks: int
    slot_bytes: int               # one slot, all slot layers together
    num_slots: int
    paged_gib: float
    slots_gib: float

    def max_context(self, concurrency: int) -> int:
        return (self.num_blocks // concurrency) * self.block_tokens


def plan(paged: "list[PagedSpec]", slots: "list[SlotSpec]", kv_gib: float,
         max_seqs: int, block_tokens: int, sector: int = 4096) -> Plan:
    """Split a KV budget: slots first (fixed by max_seqs), blocks with the rest.

    Block bytes are rounded up to `sector` so a block is a legal O_DIRECT unit
    for the NVMe tier (D16) -- the same rounding kv_tier.py insists on."""
    per_tok = sum(p.bytes_per_token * p.layers for p in paged)
    raw = per_tok * block_tokens
    block_bytes = -(-raw // sector) * sector
    slot_bytes = sum(s.bytes_per_seq * s.layers for s in slots)
    slots_bytes = slot_bytes * (max_seqs + 1)          # +1: slot 0 is the kernels' null block, never issued
    left = kv_gib * GIB - slots_bytes
    if left <= 0:
        raise MemoryError(f"{max_seqs} slots need {slots_bytes / GIB:.2f} GiB, budget is {kv_gib:.2f}")
    return Plan(block_tokens, block_bytes, int(left // block_bytes), slot_bytes, max_seqs + 1,
                (left // block_bytes) * block_bytes / GIB, slots_bytes / GIB)


def carve(arena, p: Plan, pool_cls, slot_cls):
    """Bind the plan to the arena: one KV region, one slot region, two pools."""
    kv_view = arena.carve(p.num_blocks * p.block_bytes, "kv blocks")
    pool = pool_cls(p.num_blocks, p.block_tokens, max_seqs=p.num_slots,
                    max_blocks_per_seq=p.num_blocks)
    pool.attach_storage(kv_view, p.block_bytes)
    slot_view = arena.carve(p.num_slots * p.slot_bytes, "state slots") if p.slot_bytes else None
    return pool, slot_cls(p.num_slots), slot_view


def _selfcheck() -> None:
    # Qwen3.8 from its profile: 12 QSA layers x 12 KiB/tok + indexer; 36 GDN slots.
    from engine.profiles.qwen38.plan import text_config as qcfg, state_bytes as qstate
    c = qcfg(); per_seq, kv_tok, idx_tok = qstate(c)
    n_full = c["layer_types"].count("qwen_sparse_attention") if "qwen_sparse_attention" in c["layer_types"] else len(c["layer_types"]) - c["layer_types"].count("linear_attention")
    paged = [PagedSpec("full-attn kv", n_full, kv_tok // n_full, "profiles/qwen38/caches"),
             PagedSpec("qsa keys", n_full, idx_tok // n_full, "profiles/qwen38/caches")]
    slots = [SlotSpec("gdn state", 1, per_seq, "profiles/qwen38/caches")]
    p = plan(paged, slots, kv_gib=40.0, max_seqs=32, block_tokens=16)
    assert p.block_bytes % 4096 == 0 and p.num_slots == 33          # 32 usable + the null slot
    ctx32 = p.max_context(32)
    # the profile said 40 GiB buys ~100,590 tokens at concurrency 32; blocks round down a little
    assert 90_000 < ctx32 <= 100_590, ctx32
    # GLM: MLA 5.5 KiB + indexer, KDA slots 34.8 MiB
    from engine.profiles.glm53.plan import text_config as gcfg, state_bytes as gstate
    g = gcfg(); gs, gkv, gidx = gstate(g)
    gp = plan([PagedSpec("mla latent fp8", 11, gkv // 11, "glm53/plan"), PagedSpec("indexer", 11, gidx // 11, "glm53/plan")],
              [SlotSpec("kda state", 1, gs, "glm53/plan")], kv_gib=8.73, max_seqs=4, block_tokens=2304)
    assert gp.block_tokens == 2304 and gp.max_context(4) > 300_000, gp
    print(f"  cache_spec: qwen38 40 GiB -> {p.num_blocks:,} blocks x {p.block_bytes} B + 32(+null) slots x {p.slot_bytes / 2**20:.1f} MiB, "
          f"ctx@32 {ctx32:,}; glm53 8.73 GiB -> {gp.num_blocks:,} blocks of 2304 tok, ctx@4 {gp.max_context(4):,} OK")


if __name__ == "__main__":
    _selfcheck()
