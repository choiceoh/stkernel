"""What one GLM-5.3 state slot holds, by field, and what a parked conversation needs of it.

From the repo root (stk-test):
    python3 measurements/park_live_state_20260919/slot_breakdown.py <config.json> <production slot bytes>
config.json: the served meta's (srv2 ~/st-main-d569a915/build/st-glm53-meta/config.json); the slot bytes: a parked
conversation's `extra` in the production tier's manifest (299932672 on 2026-09-19). The drafter ring is the difference.
"""
import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from engine.profiles.glm53.caches import layout, snapshot_layout  # noqa: E402
from engine.profiles.glm53.facts import architecture  # noqa: E402

cfg = json.load(open(sys.argv[1]))
F = architecture(cfg)
prod = int(sys.argv[2])
layers = tuple(range(F.layers))
lay = layout(F, layers, None)
by = defaultdict(int)
size = {"f32": 4, "bf16": 2, "f16": 2}
for f in lay.fields:
    n = 1
    for d in f.shape:
        n *= d
    by[f.name] += n * size[f.dtype]
draft = prod - lay.slot_bytes
print(f"F: layers {F.layers} (kda {len(F.kda_layers)}, dsa {len(F.dsa_layers)}), kda heads/rank {F.kda_heads_local}, "
      f"kda dim {F.kda_dim}, conv {F.conv}, spec_k {F.spec_k}, kpool {F.kpool}, idx dim {F.idx_dim}")
print(f"slot without the drafter ring: {lay.slot_bytes:,} B; production slot {prod:,} B -> drafter ring ~{draft:,} B")
for name, n in sorted(by.items(), key=lambda kv: -kv[1]):
    print(f"  {name:5s} {n:>13,} B  {n / prod:6.1%}")
rec_one = by["rec"] // (F.spec_k + 1)
park = by["conv"] + rec_one + by["tail"] + draft
snap = snapshot_layout(F, layers, None)[0] + draft
print(f"live state only (conv ring + rec[(ctx-1) % {F.spec_k + 1}] + tails + drafter ring): {park:,} B "
      f"= {park / 2**20:.1f} MiB, {prod / park:.2f}x less than the slot")
print(f"prefix snapshot at a block boundary (for comparison): ~{snap:,} B = {snap / 2**20:.1f} MiB")
