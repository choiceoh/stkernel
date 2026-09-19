"""What one Qwen3.8 state slot holds, by field, and what a parked conversation needs of it (served K=3, MTP rows).

From the repo root (stk-test):
    python3 measurements/park_live_state_20260919/qwen38_slot_breakdown.py [spec_k]
The facts are the checkpoint's own config as the repo pins it (probes/qwen38_config.json via probes/engine_qwen38_cells);
#1298 names the served slot: 114,645,248 B at K=3.
"""
import dataclasses
import sys
from collections import defaultdict
from math import prod

sys.path.insert(0, ".")
from engine.base.slot_caches import SIZES  # noqa: E402
from engine.profiles.qwen38.caches import layout, snapshot_layout  # noqa: E402
from probes.engine_qwen38_cells import facts  # noqa: E402

spec_k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
F = dataclasses.replace(facts(), spec_k=spec_k)
layers = tuple(range(F.layers))
lay = layout(F, layers, mtp=True)
by = defaultdict(int)
live = 0
for f in lay.fields:
    size = prod(f.shape) * SIZES[f.dtype]
    by[f.name] += size
    live += size // f.shape[0] if f.name == "rec" else size                 # every rec is a GDN layer's ring
print(f"F: layers {F.layers}, spec_k {F.spec_k}; slot {lay.slot_bytes:,} B")
for name, n in sorted(by.items(), key=lambda kv: -kv[1]):
    print(f"  {name:9s} {n:>13,} B  {n / lay.slot_bytes:6.1%}")
print(f"live state only (rec[(ctx-1) % {F.spec_k + 1}] + every other field): {live:,} B = {live / 2**20:.1f} MiB, "
      f"{lay.slot_bytes / live:.2f}x less than the slot")
print(f"prefix snapshot (for comparison): {snapshot_layout(F, layers)[0]:,} B")
