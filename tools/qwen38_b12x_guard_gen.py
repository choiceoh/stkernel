#!/usr/bin/env python3
"""Generate the Qwen3.8 b12x dynamic-kernel override: stock + unrouted-slot guard.

The stock SM120 dynamic MoE kernel (`_moe_dynamic/generic.py`) indexes its
per-expert routing state with the top-k ids it is handed and never looks at
their sign:

    expert_id = topk_ids[hist_idx]
    atomic_add_global_i32(row_counts + expert_id, 1)        # phase 1
    row = atomic_add_global_i32(expert_write_rows + expert_id, 1)   # phase 2
    phys_tile = expert_tile_base[expert_id] + ...

vLLM's routing contract says a slot may be -1: "unrouted" -- a padded row, or
under expert parallelism an expert that lives on another rank. Every other
backend honours it (`fused_moe/utils.py`: "The kernel uses -1 to represent
invalid topk_ids"). This one turns it into `row_counts[-1]`, four bytes before
the allocation. compute-sanitizer on the exact arguments the Qwen3.8 profile
run hands the kernel (2048 tokens x top-10, all -1):

    Invalid __global__ atomic of size 4 bytes
      at ..._moe_dynamic generic MoEDynamicKernel ...
      Access to 0xf02a3c3d43fc is out of bounds
      and is 4 bytes before the nearest allocation at 0xf02a3c3d4400 of size 2048 bytes

2048 bytes is the [512] int32 histogram. Whether the write faults or lands in
a neighbour is an accident of the allocator's layout: standalone it landed,
in the serving process it faulted -- the cudaErrorIllegalAddress that blocked
every Qwen3.8 b12x boot. Same arguments, valid ids: clean.

The guard is what the other backends do -- a slot with id < 0 contributes
nothing: no histogram count, no packed row, no task. Phase 0 already zeroes
the output, so a token whose every slot is unrouted stays zero, which is what
the EP all-reduce expects from a rank that owns none of its experts. This is
also the expert-parallel path for this model: no dummy expert, no wasted GEMM
rows for the 3/4 of routes a TEP=4 rank does not own.

Two edits, anchored on exact stock text that must occur once, plus one
refusal: the shared-input producer (`share_input_across_experts`, a scalar
input scale) caches 32 route slots per token and has no skip for an unrouted
slot. Qwen3.8 passes per-expert scales and never takes it; rather than carry
an unguarded branch, the override refuses to build it.

    python3 tools/qwen38_b12x_guard_gen.py --stock <image>/_moe_dynamic/generic.py \\
        --out overlay/modules/qwen38_b12x/moe_dynamic_generic.py
    python3 tools/qwen38_b12x_guard_gen.py --stock ... --check <committed file>

The stock file is pinned by SHA-256; a FlashInfer bump that changes the
kernel refuses to generate rather than silently re-anchoring.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

STOCK_SHA256 = "38b112ffe5ca00789b1fa84b7f606431385305d9c835e4adbf2e94ade60655da"
STOCK_VERSION = "flashinfer 0.6.18.dev20260819 (qwen38-fi618:local)"

# (name, old, new). `old` must occur exactly once in the stock text.
EDITS: list[tuple[str, str, str]] = [
    (
        "phase-1 histogram: skip unrouted slots",
        """        hist_idx = flat_tid
        while hist_idx < total_pairs:
            expert_id = topk_ids[hist_idx].to(Int32)
            atomic_add_global_i32(get_ptr_as_int64(row_counts, expert_id), Int32(1))
            hist_idx += flat_stride
""",
        """        hist_idx = flat_tid
        while hist_idx < total_pairs:
            expert_id = topk_ids[hist_idx].to(Int32)
            # qwen38 guard: -1 is vLLM's unrouted slot; it owns no row.
            if expert_id >= Int32(0):
                atomic_add_global_i32(get_ptr_as_int64(row_counts, expert_id), Int32(1))
            hist_idx += flat_stride
""",
    ),
    (
        "phase-2 per-route producer: unrouted slots allocate no row",
        """                        if pair_idx < total_pairs:
                            expert_id = topk_ids[pair_idx].to(Int32)
                            token_idx = pair_idx // num_topk
                            weight = topk_weights[pair_idx].to(cutlass.Float32)

                            if lane_id == Int32(0):
                                row = atomic_add_global_i32(
                                    get_ptr_as_int64(expert_write_rows, expert_id),
                                    Int32(1),
                                )
""",
        """                        # qwen38 guard: an unrouted slot (-1) is skipped whole --
                        # no row, no quantized copy, no task. Phase 0 zeroed the
                        # output, so a token with no routed slot stays zero.
                        route_ok = Int32(0)
                        if pair_idx < total_pairs:
                            expert_id = topk_ids[pair_idx].to(Int32)
                            if expert_id >= Int32(0):
                                route_ok = Int32(1)
                        if route_ok > Int32(0):
                            token_idx = pair_idx // num_topk
                            weight = topk_weights[pair_idx].to(cutlass.Float32)

                            if lane_id == Int32(0):
                                row = atomic_add_global_i32(
                                    get_ptr_as_int64(expert_write_rows, expert_id),
                                    Int32(1),
                                )
""",
    ),
    (
        "refuse the shared-input producer (no unrouted-slot skip there)",
        """        self.share_input_across_experts = share_input_across_experts
""",
        """        self.share_input_across_experts = share_input_across_experts
        if share_input_across_experts:
            raise NotImplementedError(
                "qwen38 b12x guard: the shared-input producer caches 32 route "
                "slots per token and has no skip for an unrouted (-1) slot; "
                "this model passes per-expert input scales and never needs it"
            )
""",
    ),
]

HEADER = '''# GENERATED by tools/qwen38_b12x_guard_gen.py -- do not edit by hand.
#
# FlashInfer's SM120 dynamic MoE kernel, {version},
# stock sha256 {sha}, plus the unrouted-slot guard:
{edits}
#
# Mounted over blackwell_sm12x/_moe_dynamic/generic.py; the on-disk CuTe-DSL
# kernel cache hashes this file, so the guarded build never collides with a
# stock one. Regenerate with --check to prove this file is stock + these edits.
'''


def generate(stock_text: str) -> str:
    digest = hashlib.sha256(stock_text.encode()).hexdigest()
    if digest != STOCK_SHA256:
        raise SystemExit(
            f"ABORT: stock generic.py is {digest[:16]}..., not the pinned "
            f"{STOCK_SHA256[:16]}... ({STOCK_VERSION}). The anchors below were "
            f"written against that file; re-read the kernel before re-pinning.")
    text = stock_text
    for name, old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"ABORT: anchor for '{name}' occurs {n} times, not once")
        text = text.replace(old, new, 1)
    edits = "\n".join(f"#   - {name}" for name, _o, _n in EDITS)
    return HEADER.format(version=STOCK_VERSION, sha=STOCK_SHA256, edits=edits) + text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stock", required=True, help="the image's _moe_dynamic/generic.py")
    ap.add_argument("--out", help="write the override here")
    ap.add_argument("--check", help="verify this committed file equals the generation")
    args = ap.parse_args()
    out = generate(Path(args.stock).read_text())
    if args.check:
        have = Path(args.check).read_text()
        if have != out:
            print(f"MISMATCH: {args.check} is not stock + the pinned edits")
            return 1
        print(f"OK: {args.check} == stock {STOCK_SHA256[:16]}... + {len(EDITS)} edits")
    if args.out:
        Path(args.out).write_text(out)
        print(f"wrote {args.out} ({len(out.splitlines())} lines, {len(EDITS)} edits)")
    if not args.out and not args.check:
        sys.stdout.write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
