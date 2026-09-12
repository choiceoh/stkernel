"""GLM-5.3-Flash's budget lines on the ST engine (profile): the box, declared -- D1.

Every claim on the box carries where it came from; KV is what is left.
Until 2026-09-11 the engine's KV came from vLLM's 40th-boot table (8.73 GiB:
what THAT stack had left after its 8.77 GiB load scratch and 9.17 GiB
profile run, neither of which this engine pays). This file states the ST
engine's own lines: weights and drafter read from the rank files, state
slots and the block table from the cache layout, the workspace as the
ceiling base/runtime_memory enforces (measured peaks from a boot's memory
ledger when one is given), the runtime floor from the ledger, the OS reserve
declared. The remainder is the KV room; `unassigned` is how much of it the
boot does not use at the declared `kv_gib`.

    python3 engine/profiles/glm53/budget.py                      # the box as facts.py sees it
    python3 engine/profiles/glm53/budget.py --ledger /home/choiceoh/glm53-logs/st-dumps/memory-rank0.json
"""
from __future__ import annotations

import json
from pathlib import Path

from engine.base.budget import (GIB, MEASURED, READ, LEDGER, DECLARED, Line, Budget, host_box, rank_weights)
from engine.profiles.glm53 import facts, specs as specs_mod
from engine.profiles.glm53 import drafter as drafter_mod
from engine.profiles.glm53.caches import layout, snapshot_layout, stage_bytes

RUNTIME_FLOOR_GIB = 5.54            # ledger 40th boot table (vLLM): CUDA context + NCCL 16 channels -- re-measure on ST
WORKSPACE_GIB = 12.0                # base/runtime_memory's enforced ceiling for everything outside the arena (#549)
OS_RESERVE_GIB = 4.0               # RuntimeMemory's fixed physical reserve; admission also budgets all workspace
NVME_STAGING_BYTES = 2 * (64 << 20) + 2 * (32 << 20)   # kv_tier: 64 MiB pinned staging + 64 MiB device scratch, and the prefix tier's 32 + 32
SELECT_ROWS_TRANSIENT_NOTE = "indexer selection bounded to 1,024 query rows per pass (net.SELECT_ROWS)"


def ledger_peak(path: "str | Path | None") -> "tuple[float, str] | None":
    """(peak workspace GiB, phase) from a boot's memory ledger (RuntimeMemory.write), or None."""
    if not path or not Path(path).exists():
        return None
    phases = json.loads(Path(path).read_text()).get("phases", [])
    if not phases:
        return None
    top = max(phases, key=lambda r: r.get("peak_workspace_bytes", 0))
    return top.get("peak_workspace_bytes", 0) / GIB, top.get("phase", "?")


def budget(kv_gib: float, max_seqs: int, chunk: int = 6912, box_gib: "float | None" = None,
           ckpt: "str | Path" = facts.CKPT, ranks_dir: "str | Path | None" = None, rank: int = 0,
           drafter_dir: "str | Path | None" = drafter_mod.DRAFTER, ledger: "str | Path | None" = None,
           snapshots: int = 8, draft_tp: int = 1, draft_native: "bool | None" = None) -> Budget:
    """The box, one rank of TP=4. `kv_gib`/`max_seqs` are boot.py's declared values; the table says what they leave."""
    host_total, _ = host_box()
    if box_gib is None:
        box_gib = host_total                                             # GB10: device total == MemTotal (facts.check_box)
    F = facts.load(ckpt)
    from engine.profiles.glm53.weights import MODELOPT_WEIGHT_LAYOUT
    if F.weight_layout == MODELOPT_WEIGHT_LAYOUT:
        from engine.profiles.glm53.modelopt_weights import all_specs
        weight_specs = all_specs(F)
    else:
        weight_specs = specs_mod.all_specs(F)
    rank_file = Path(ranks_dir) / f"rank{rank}of{facts.TP}.safetensors" if ranks_dir else None
    if rank_file is not None and rank_file.exists():
        weights_gib, tensors = rank_weights(rank_file)
        weights_evidence = f"{rank_file.name} header: {tensors:,} tensors"
    else:
        weights_gib = sum(s.nbytes() for s in weight_specs) / GIB
        weights_evidence = f"{F.weight_layout} specs: {len(weight_specs):,} tensors/rank at TP={facts.TP}"
    vision_file = Path(ranks_dir) / "vision.safetensors" if ranks_dir else None
    if vision_file is not None and vision_file.exists():
        vision_gib, vision_evidence = rank_weights(vision_file)[0], f"{vision_file.name} header"
    elif (Path(ckpt) / "processor_config.json").exists():
        from engine.profiles.glm53 import vision as vision_mod
        vision_gib, vision_evidence = sum(s.nbytes() for s in vision_mod.specs(vision_mod.load(ckpt))) / GIB, "vision.specs: whole tower per rank"
    else:
        vision_gib, vision_evidence = 0.0, "no vision tower declared (no processor_config.json)"
    draft_shape, drafter_gib = None, 0.0
    if drafter_dir and (Path(drafter_dir) / "config.json").exists():
        D = drafter_mod.load(drafter_dir)
        if draft_tp <= 0 or D.kv_heads % draft_tp:
            raise ValueError("drafter KV heads must split over the declared TP group")
        native = draft_tp > 1 if draft_native is None else draft_native
        cells = D.window if native else drafter_mod.ring_cells(D)
        draft_shape = (D.layers, cells, D.kv_heads // draft_tp, D.head_dim)
        drafter_gib = sum(s.nbytes() for s in drafter_mod.specs(D)) / GIB
    lay = layout(F, range(F.layers), draft_shape)
    slots_gib = (max_seqs + 1) * lay.slot_bytes / GIB
    snapshot_bytes = snapshot_layout(F, range(F.layers), draft_shape)[0]
    blocks_at_kv = int((kv_gib * GIB - (max_seqs + 1) * lay.slot_bytes) // (lay.block_bytes + max_seqs * 4))
    peak = ledger_peak(ledger)
    workspace_evidence = "base/runtime_memory ceiling: activations, graph pools, kernel scratch; the allocator refuses beyond it"
    if peak is not None:
        workspace_evidence += f"; measured peak {peak[0]:.2f} GiB at {peak[1]} ({Path(ledger).name})"
    else:
        workspace_evidence += f"; no boot ledger given -- vLLM's slope 0.52 GiB/1K puts a {chunk:,}-token chunk at {chunk / 1024 * 0.52:.1f} GiB, {SELECT_ROWS_TRANSIENT_NOTE}"
    lines = [
        Line("reserve for the OS", OS_RESERVE_GIB, DECLARED, "base/runtime_memory: immediately free host/device byte floor"),
        Line("runtime floor (CUDA ctx + NCCL 16ch)", RUNTIME_FLOOR_GIB, LEDGER, "GLM 40th boot table -- re-measure on ST"),
        Line("weights (this rank, TP=4)", weights_gib, READ, weights_evidence),
        Line("drafter weight reservation", drafter_gib, READ, f"drafter.specs: source reservation retained; compute/KV TP={draft_tp}"),
        Line("vision tower (BF16, replicated)", vision_gib, READ, vision_evidence),
        Line(f"state slots ({max_seqs} + null) x {lay.slot_bytes / 2**20:.0f} MiB", slots_gib, READ,
             "caches.layout: KDA conv/recurrent rings (K+1 states), indexer tails, drafter ring"),
        Line(f"prefix snapshots ({snapshots} x {snapshot_bytes / 2**20:.0f} MiB)", snapshots * snapshot_bytes / GIB, READ,
             "caches.snapshot_layout: chunk-boundary position rings for prefix reuse (boot.PREFIX_SNAPSHOTS)"),
        Line("generated-boundary staging", stage_bytes(F, range(F.layers), max_seqs) / GIB, READ,
             "caches.stage_bytes: per-slot recurrent state and convolution history"),
        Line("workspace ceiling (outside the arena)", WORKSPACE_GIB, DECLARED, workspace_evidence),
        Line("NVMe tier staging", NVME_STAGING_BYTES / GIB, DECLARED, "kv_tier: pinned staging + device scratch, conversations and prefix tiers"),
    ]
    b = Budget(box_gib, lines, label=f"GLM-5.3-Flash on ST, one rank of TP={facts.TP}, chunk {chunk:,}, kv_gib {kv_gib} -> {blocks_at_kv:,} blocks")
    b.kv_declared_gib = kv_gib - slots_gib                              # what boot.py actually gives the paged KV + table
    b.paged_gib = blocks_at_kv * lay.block_bytes / GIB
    b.block_bytes, b.slot_bytes, b.block_tokens, b.max_position = lay.block_bytes, lay.slot_bytes, F.block, F.max_position
    return b


def report(b: Budget) -> str:
    """The table, the verdict, and what the declared KV leaves on the table."""
    out = [b.table(), "", "  " + b.verdict()]
    unassigned = b.kv_gib - b.kv_declared_gib
    out.append(f"  declared paged KV {b.kv_declared_gib:.2f} GiB ({b.paged_gib:.2f} in blocks of {b.block_bytes / 2**20:.2f} MiB); "
               f"unassigned {unassigned:+.2f} GiB")
    per_token = b.block_bytes / b.block_tokens
    out.append(f"  what the remainder buys at {per_token:,.0f} B/token (paged, layout) + {b.slot_bytes / 2**20:.0f} MiB/sequence (slot):")
    for n in (1, 4, 8, 32):
        toks = (b.kv_gib * GIB - n * b.slot_bytes) / (n * per_token)
        out.append(f"    concurrency {n:>3}: {int(min(max(toks, 0), b.max_position)):>9,} tok each")
    return "\n".join(out)


def main(argv=None) -> int:
    import argparse
    from engine.profiles.glm53 import boot
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kv-gib", type=float, default=boot.KV_GIB)
    ap.add_argument("--max-seqs", type=int, default=boot.MAX_SEQS)
    ap.add_argument("--chunk", type=int, default=6912)
    ap.add_argument("--box-gib", type=float, default=None)
    ap.add_argument("--ranks", default=str(facts.RANKS))
    ap.add_argument("--ledger", default=None, help="a boot's memory-rankN.json (RuntimeMemory.write)")
    ap.add_argument("--snapshots", type=int, default=boot.PREFIX_SNAPSHOTS)
    a = ap.parse_args(argv)
    b = budget(a.kv_gib, a.max_seqs, a.chunk, a.box_gib, ranks_dir=a.ranks, ledger=a.ledger, snapshots=a.snapshots,
               draft_tp=facts.TP)
    print(report(b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
