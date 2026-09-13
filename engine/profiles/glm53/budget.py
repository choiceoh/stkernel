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
from engine.profiles.glm53.caches import layout, snapshot_layout, stage_bytes, cache_capacity, state_dtype

RUNTIME_FLOOR_GIB = 5.54            # ledger 40th boot table (vLLM): CUDA context + NCCL 16 channels -- re-measure on ST
WORKSPACE_GIB = 12.0                # base/runtime_memory's enforced ceiling for everything outside the arena (#549)
OS_RESERVE_MARGIN_GIB = 2.0        # above the box's own SIGTERM line, so the engine notices first


def os_reserve_gib() -> float:
    """What to keep free for the box, derived from the box's kill line rather than declared.

    It used to be a flat 4.0 GiB. earlyoom on these nodes SIGTERMs at 6 GiB and SIGKILLs at
    4.5, `--prefer python3`, the engine first on purpose -- so the engine's own reserve sat
    BELOW both and its "preparation consumed the OS memory reserve" check could not fire
    first. Fourteen recorded boots reached 1.48-15.74 GiB of free memory and every one of
    them reported healthy; six were under the SIGTERM line (2026-09-12, 45차).
    """
    from engine.base.runtime_memory import oom_floor
    sigterm, _sigkill = oom_floor()
    return max(4.0, sigterm / GIB + OS_RESERVE_MARGIN_GIB)


OS_RESERVE_GIB = os_reserve_gib()  # admission's physical reserve; admission also budgets all workspace
NVME_STAGING_BYTES = 2 * (64 << 20) + 2 * (32 << 20)   # kv_tier: 64 MiB pinned staging + 64 MiB device scratch, and the prefix tier's 32 + 32
SELECT_ROWS_TRANSIENT_NOTE = "indexer selection bounded to 1,024 query rows per pass (net.SELECT_ROWS)"


def ledger_peak(path: "str | Path | None") -> "tuple[float, str] | None":
    """(peak workspace GiB, phase) from a boot's memory ledger (RuntimeMemory.write), or None."""
    report = read_ledger(path)
    phases = (report or {}).get("phases", [])
    if not phases:
        return None
    top = max(phases, key=lambda r: r.get("peak_workspace_bytes", 0))
    return top.get("peak_workspace_bytes", 0) / GIB, top.get("phase", "?")


def read_ledger(source) -> "dict | None":
    """A boot's memory ledger, given as `RuntimeMemory.report()` itself or the path it was written to."""
    if isinstance(source, dict):
        return source
    if not source or not Path(source).exists():
        return None
    return json.loads(Path(source).read_text())


def ledger_measured(source) -> "dict | None":
    """`RuntimeMemory.measured()` out of a ledger: the floor, the prefill peak, the graphs.

    These are the two lines this table used to guess -- a runtime floor carried over from
    vLLM's 40th-boot table and a workspace ceiling declared at 12 GiB with vLLM's activation
    slope quoted underneath it (45차 §50). vLLM gets both from one `profile_run`; ST's boot
    already runs the largest legal prefill and captures every graph under a ledger, so with a
    ledger in hand the lines are measurements of THIS stack and the table says so.

    A ledger written before `measured()` existed still works: the same split is recomputable
    from its rows, and the floor is simply absent, which leaves that line where it was.
    """
    report = read_ledger(source)
    if not report:
        return None
    if report.get("measured"):
        return report["measured"]
    phases, arena = report.get("phases", []), report.get("arena_bytes", 0)
    if not phases:
        # The floor is taken in RuntimeMemory.__init__, before any phase exists. A report
        # handed over at the START of a boot therefore already carries the one line this
        # table used to guess -- and that is the moment anyone reads the table and decides
        # how much KV to ask for. Returning None here is why the first print said 42.77 GiB
        # of KV remained on a box that had 9.26 (2026-09-12).
        return dict(floor_bytes=report["floor_bytes"]) if report.get("floor_bytes") else None
    base = report.get("baseline_reserved_bytes", 0)
    outside = lambda row: row.get("reserved_bytes", 0) - base - arena                    # noqa: E731
    prefill = [row for row in phases if row.get("phase", "").startswith("prefill/")]
    final = phases[-1]
    return dict(floor_bytes=report.get("floor_bytes", 0),
                prefill_peak_bytes=max((r.get("peak_workspace_bytes", 0) for r in prefill), default=0),
                prefill_shapes=[r["phase"] for r in prefill if r.get("phase", "").endswith("/prepared")],
                graph_bytes=max(0, outside(final) - (outside(prefill[-1]) if prefill else 0)),
                peak_workspace_bytes=max(r.get("peak_workspace_bytes", 0) for r in phases),
                retained_workspace_bytes=max(0, outside(final)),
                workspace_limit_bytes=report.get("workspace_limit_bytes", 0),
                at_phase=final.get("phase", "?"))


def budget(kv_gib: float, max_seqs: int, chunk: int = 6912, box_gib: "float | None" = None,
           ckpt: "str | Path" = facts.CKPT, ranks_dir: "str | Path | None" = None, rank: int = 0,
           drafter_dir: "str | Path | None" = drafter_mod.DRAFTER, ledger: "str | Path | None" = None,
           snapshots: "int | None" = None, draft_tp: int = 1, draft_native: "bool | None" = None,
           router_bytes: int = 0, projection_bytes: int = 0, tier_enabled: bool = True, kda_state_dtype: "str | None" = None) -> Budget:
    """The box, one rank of TP=4. `kv_gib`/`max_seqs` are boot.py's declared values; the table says what they leave."""
    host_total, _ = host_box()
    if box_gib is None:
        box_gib = host_total                                             # GB10: device total == MemTotal (facts.check_box)
    F = facts.load(ckpt)
    if kda_state_dtype is not None:
        from dataclasses import replace
        F = replace(F, kda_state_dtype=state_dtype(kda_state_dtype))
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
    draft_shape, drafter_gib, draft_evidence = None, 0.0, "no drafter declared"
    if drafter_dir and (Path(drafter_dir) / "config.json").exists():
        D = drafter_mod.load(drafter_dir)
        if draft_tp <= 0 or D.kv_heads % draft_tp:
            raise ValueError("drafter KV heads must split over the declared TP group")
        native = draft_tp > 1 if draft_native is None else draft_native
        cells = D.window if native else drafter_mod.ring_cells(D)
        draft_shape = (D.layers, cells, D.kv_heads // draft_tp, D.head_dim)
        if native:
            from engine.profiles.glm53.drafter_storage import nbytes as draft_resident_bytes
            drafter_gib = draft_resident_bytes(D, draft_tp, max_seqs) / GIB
            draft_evidence = f"drafter_storage: live packed readers; compute/KV TP={draft_tp}"
        else:
            drafter_gib = sum(s.nbytes() for s in drafter_mod.specs(D)) / GIB
            draft_evidence = f"drafter.specs: source weights; compute/KV TP={draft_tp}"
    lay = layout(F, range(F.layers), draft_shape)
    slots_gib = (max_seqs + 1) * lay.slot_bytes / GIB
    snapshot_bytes = snapshot_layout(F, range(F.layers), draft_shape)[0]
    from engine.profiles.glm53 import boot
    blocks_at_kv, default_snapshots = cache_capacity(
        F, range(F.layers), draft_shape, kv_gib, max_seqs,
        boot.PREFIX_SNAPSHOT_GIB if tier_enabled else boot.PREFIX_UNTIERED_SNAPSHOT_GIB)
    if snapshots is None:
        snapshots = default_snapshots
    m = ledger_measured(ledger)
    ledger_name = Path(ledger).name if isinstance(ledger, (str, Path)) and ledger else "this boot"
    workspace_evidence = ("base/runtime_memory ceiling: activations, graph pools, kernel scratch; the allocator refuses "
                          "beyond it")
    if m and m.get("peak_workspace_bytes"):
        # The LINE stays the enforced ceiling, because that is what the box must be able to
        # absorb: the allocator will hand out every byte of it. What the ledger changes is that
        # the evidence is now this stack's own split instead of vLLM's activation slope, and
        # `report` can say what lowering the ceiling to the measurement would buy.
        workspace_evidence += (f"; {ledger_name} peaked at {m['peak_workspace_bytes'] / GIB:.2f} GiB through "
                               f"{m['at_phase']} -- prefill activations {m['prefill_peak_bytes'] / GIB:.2f} GiB over "
                               f"{len(m.get('prefill_shapes') or [])} qualified shapes, graphs and scratch "
                               f"+{m['graph_bytes'] / GIB:.2f} GiB retained")
    else:
        workspace_evidence += (f"; no boot ledger given -- vLLM's slope 0.52 GiB/1K puts a {chunk:,}-token chunk at "
                               f"{chunk / 1024 * 0.52:.1f} GiB, {SELECT_ROWS_TRANSIENT_NOTE}")
    tenants_line = None
    if m and m.get("floor_bytes"):
        # A cost, not a ceiling: nothing hands this back, so the measurement IS the line.
        floor_gib, floor_source = m["floor_bytes"] / GIB, MEASURED
        floor_evidence = f"{ledger_name}: device in use outside the allocator when the ceiling armed (context + NCCL + one-shot)"
        # ... and split it, because the two halves are different problems. 5.54 GiB is this
        # engine starting up; the rest is what was ALREADY on the box, which no amount of
        # engine work reduces and which the table never showed. On rank 3 that half is
        # 33.50 GiB of a 121.63 GiB box -- larger than the KV it leaves (2026-09-12).
        # The dsv41 profile has carried an "other tenants" line since it was written.
        others = floor_gib - RUNTIME_FLOOR_GIB
        if others > 0.05:
            floor_gib, floor_evidence = RUNTIME_FLOOR_GIB, (
                f"{ledger_name}: the measured floor {m['floor_bytes'] / GIB:.2f} GiB less what this engine's "
                "own start-up costs (GLM 40th boot table)")
            tenants_line = Line("already on this box before us", others, MEASURED,
                                f"{ledger_name} floor minus start-up: other containers, page cache and anything "
                                "else holding pages -- unified memory means they come out of our KV")
    else:
        floor_gib, floor_source = RUNTIME_FLOOR_GIB, LEDGER
        floor_evidence = "GLM 40th boot table (vLLM) -- no ST ledger given"
    lines = [
        Line("reserve for the OS", OS_RESERVE_GIB, DECLARED, "base/runtime_memory: immediately free host/device byte floor"),
        Line("runtime floor (CUDA ctx + NCCL 16ch)", floor_gib, floor_source, floor_evidence),
        *( (tenants_line,) if tenants_line is not None else () ),
        Line("weights (this rank, TP=4)", weights_gib, READ, weights_evidence),
        Line("resident FP32 routers", router_bytes / GIB, READ, "net.router_nbytes: immutable BF16 gate values converted once into the arena"),
        Line("resident decode projection pairs", projection_bytes / GIB, READ, "net.decode_projection_nbytes: indexer pairs copied into the arena after smoothing"),
        Line("drafter weight reservation", drafter_gib, READ, draft_evidence),
        Line("vision tower (BF16, replicated)", vision_gib, READ, vision_evidence),
        Line(f"state slots ({max_seqs} + null) x {lay.slot_bytes / 2**20:.0f} MiB", slots_gib, READ,
             f"caches.layout: KDA {F.kda_state_dtype} recurrent rings (K+1 states), BF16 conv, indexer tails, drafter ring"),
        Line(f"prefix snapshots ({snapshots} x {snapshot_bytes / 2**20:.0f} MiB)", snapshots * snapshot_bytes / GIB, READ,
             f"caches.snapshot_layout: chunk-boundary position rings for prefix reuse; the count follows "
             f"the FP32 baseline's tiered/untiered budget (sharded drafter ring: {'yes' if draft_native else 'no'})"),
        Line("compressed prefix cache and codec", boot.prefix_host_bytes(tier_enabled) / GIB, DECLARED,
             "prefix tier: bounded lossless RAM copies plus chunk workspace, outside the raw arena; compression ratio unmeasured"),
        Line("generated-boundary staging", stage_bytes(F, range(F.layers), max_seqs) / GIB, READ,
             "caches.stage_bytes: per-slot recurrent state and convolution history"),
        Line("workspace ceiling (outside the arena)", WORKSPACE_GIB, DECLARED, workspace_evidence),
        Line("NVMe tier staging", NVME_STAGING_BYTES / GIB, DECLARED, "kv_tier: pinned staging + device scratch, conversations and prefix tiers"),
    ]
    b = Budget(box_gib, lines, label=f"GLM-5.3-Flash on ST, one rank of TP={facts.TP}, chunk {chunk:,}, kv_gib {kv_gib} -> {blocks_at_kv:,} blocks")
    baseline_slots_gib = (max_seqs + 1) * layout(F, range(F.layers), draft_shape, state_storage="fp32").slot_bytes / GIB
    b.kv_declared_gib = kv_gib - baseline_slots_gib                     # FP16 savings stay unassigned, not extra KV
    b.paged_gib = blocks_at_kv * lay.block_bytes / GIB
    b.block_bytes, b.slot_bytes, b.block_tokens, b.max_position = lay.block_bytes, lay.slot_bytes, F.block, F.max_position
    b.measured = m
    return b


def report(b: Budget) -> str:
    """The table, the verdict, and what the declared KV leaves on the table."""
    out = [b.table(), "", "  " + b.verdict()]
    unassigned = b.kv_gib - b.kv_declared_gib
    out.append(f"  declared paged KV {b.kv_declared_gib:.2f} GiB ({b.paged_gib:.2f} in blocks of {b.block_bytes / 2**20:.2f} MiB); "
               f"unassigned {unassigned:+.2f} GiB")
    per_token = b.block_bytes / b.block_tokens
    m = getattr(b, "measured", None)
    if m and m.get("peak_workspace_bytes"):
        # The one line this table could never write from declarations: what the ceiling costs
        # in KV over what the boot actually spends under it (45차 §51).
        headroom = (WORKSPACE_GIB * GIB - m["peak_workspace_bytes"]) / GIB
        out.append(f"  workspace: ceiling {WORKSPACE_GIB:.2f} GiB enforced, this boot peaked at "
                   f"{m['peak_workspace_bytes'] / GIB:.2f} (prefill activations {m['prefill_peak_bytes'] / GIB:.2f}, "
                   f"graphs and scratch +{m['graph_bytes'] / GIB:.2f} retained) -- {headroom:+.2f} GiB of the ceiling "
                   f"unspent, which is what a lower ceiling would return to KV")
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
    ap.add_argument("--snapshots", type=int, default=None,
                    help="resident boundary checkpoints; default follows boot.PREFIX_SNAPSHOT_GIB and this shape")
    a = ap.parse_args(argv)
    b = budget(a.kv_gib, a.max_seqs, a.chunk, a.box_gib, ranks_dir=a.ranks, ledger=a.ledger, snapshots=a.snapshots,
               draft_tp=facts.TP)
    print(report(b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
