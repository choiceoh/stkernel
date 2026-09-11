"""DSv4.1-Flash's budget lines and replication census (profile).
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

from engine.base.budget import (GIB, MEASURED, READ, LEDGER, DECLARED, ESTIMATED,
                                Line, Budget, probe_box, host_box, rank_weights, rank_files)


RUNTIME_FLOOR_GIB = 5.54


ALLOCATOR_SLACK_RATIO = 0.001


CONSTRUCTION_UPPER_GIB = 8.77


ACTIVATION_GIB_PER_1K = 0.520


RESIDUAL_BYTES_PER_TOKEN = 4 * 5120 * 2      # hc_mult x dim, bf16


OS_RESERVE_MULTIPLE = 2.0


PLAN_TP_RESIDENT_GIB = 121.4 - 47.2          # 74.2, vision included


PLAN_TP_RESIDENT_NOVISION_GIB = 74.2 - 0.9   # this fleet serves text only


def for_dsv41(checkpoint: "str | Path", world_size: int = 4,
              box_gib: "float | None" = None,
              tenants_gib: float = 0.0,
              weights_gib: "float | None" = None,
              weights_note: str = "",
              chunk: int = 4096) -> Budget:
    if weights_gib is None:
        files = rank_files(checkpoint, world_size)
        sizes = [rank_weights(f) for f in files]
        weights_gib = max(g for g, _ in sizes)
        tensors = max(n for _, n in sizes)
        spread = weights_gib - min(g for g, _ in sizes)
        weights_line = Line(
            "weights (this rank)", weights_gib, READ,
            f"rank*of{world_size}: {tensors:,} tensors, spread across ranks "
            f"{spread * 1024:.0f} MiB")
    else:
        provenance = READ if "convert.py axes" in weights_note else LEDGER
        weights_line = Line("weights (this rank)", weights_gib, provenance,
                            weights_note or "supplied")

    if box_gib is None:
        box_gib, _free = probe_box()

    host_total, _avail = host_box()
    earlyoom_floor = host_total * 0.05
    reserve = earlyoom_floor * OS_RESERVE_MULTIPLE

    lines = []
    if tenants_gib:
        lines.append(Line("other tenants on this box", tenants_gib, MEASURED,
                          "mem_get_info total-free BEFORE we allocate; unified "
                          "memory means their pages come out of our KV"))
    lines += [
        Line("reserve for the OS", reserve, DECLARED,
             f"{OS_RESERVE_MULTIPLE:g}x earlyoom's 5% floor ({earlyoom_floor:.2f} GiB); "
             "six SIGTERMs happened at that floor, all while serving"),
        Line("runtime floor (CUDA ctx + NCCL)", RUNTIME_FLOOR_GIB, LEDGER,
             "40th boot table: init-device 2.29 + dist 1.00 + 2.25, NCCL 16 channels"),
        weights_line,
        Line("allocator slack", weights_gib * ALLOCATOR_SLACK_RATIO, MEASURED,
             "expandable_segments:True -> 0.1% (16.1% without); torch peak == "
             "tensor bytes exactly in 13 runs, dsv41 has no repack step"),
        Line("module construction (cuBLAS, init)", CONSTRUCTION_UPPER_GIB, ESTIMATED,
             "UPPER BOUND from GLM's 59.17-50.4 with expandable_segments already "
             "on; dsv41 has no pack/quant step so its share is smaller"),
        Line(f"activation @ chunk {chunk:,}",
             chunk / 1024 * ACTIVATION_GIB_PER_1K
             + chunk * RESIDUAL_BYTES_PER_TOKEN / GIB, MEASURED,
             f"{ACTIVATION_GIB_PER_1K} GiB/1K tokens, linear over 1K-8K and "
             "identical across three layer kinds; + the residual stream"),
    ]
    return Budget(box_gib, lines, label=f"DSv4.1-Flash, rank of {world_size}, this box")


_LEVERS = (
    ("mtp experts", lambda k: k.startswith("mtp.") and ".experts." in k,
     "certain", "EP-shard them the way the main 384 routed experts already are"),
    ("vocab (embed + head)", lambda k: k.split(".")[0] in ("embed", "head"),
     "certain", "vocab-parallel: 129,280 rows split four ways"),
    ("vision tower + aligner", lambda k: k.startswith(("vision.", "aligner.")),
     "drop", "drop it: this fleet serves text only"),
    ("attention wq_b / wo_b", lambda k: k.endswith((".wq_b.weight", ".wq_b.scale",
                                                    ".wo_b.weight", ".wo_b.scale"))
     and ".indexer." not in k,
     "candidate", "TP-shard: wq_b [32768,1280] by head, wo_b [5120,8192] by input"),
    ("shared experts", lambda k: ".shared_experts." in k,
     "candidate", "TP-shard the FFN the way every other TP engine does"),
    ("mtp, the rest", lambda k: k.startswith("mtp."), None, None),
)


def _tensor_map(path: "str | Path") -> dict:
    with Path(path).open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return {
        name: (entry["dtype"], tuple(entry["shape"]),
               entry["data_offsets"][1] - entry["data_offsets"][0])
        for name, entry in header.items()
    }


def replication(checkpoint: "str | Path", world_size: int = 4) -> dict:
    """What each rank carries that every other rank carries identically."""
    maps = [_tensor_map(f) for f in rank_files(checkpoint, world_size)]
    shared = set(maps[0])
    for other in maps[1:]:
        shared &= set(other)
    identical = {k for k in shared if all(m[k] == maps[0][k] for m in maps[1:])}

    groups, counts = {}, {}
    for name in identical:
        for label, matches, _kind, _how in _LEVERS:
            if matches(name):
                break
        else:
            label = "per-layer norms and small tensors"
        groups[label] = groups.get(label, 0) + maps[0][name][2]
        counts[label] = counts.get(label, 0) + 1

    total = sum(v[2] for v in maps[0].values())
    return {
        "rank_gib": total / GIB,
        "replicated_gib": sum(groups.values()) / GIB,
        "sharded_gib": (total - sum(groups.values())) / GIB,
        "groups": {k: (v / GIB, counts[k]) for k, v in groups.items()},
        "how": {label: (kind, how) for label, _m, kind, how in _LEVERS if how},
        "world_size": world_size,
    }


def replication_report(info: dict) -> str:
    ws = info["world_size"]
    out = [
        f"  rank total          {info['rank_gib']:>8.2f} GiB",
        f"  sharded             {info['sharded_gib']:>8.2f} GiB",
        f"  replicated          {info['replicated_gib']:>8.2f} GiB   "
        f"(paid {ws}x = {info['replicated_gib'] * ws:.2f} GiB fleet-wide)",
        "",
    ]
    width = max(len(k) for k in info["groups"])
    certain = candidate = 0.0
    for label, (gib, count) in sorted(info["groups"].items(), key=lambda kv: -kv[1][0]):
        entry = info["how"].get(label)
        if entry:
            kind, how = entry
            saving = gib if kind == "drop" else gib * (ws - 1) / ws
            if kind == "candidate":
                candidate += saving
            else:
                certain += saving
            note = f"-> -{saving:.2f} GiB/rank [{kind}]: {how}"
        else:
            note = ("(router gates, norms, MLA compressed side: replicated by "
                    "design, or TP rule unknown)")
        out.append(f"  {label:<{width}}  {gib:>7.3f} GiB  {count:>6,} tensors  {note}")
    out.append("")
    out.append(f"  certain    {certain:>6.2f} GiB/rank -- needs no rule that does not exist")
    out.append(f"  candidate  {candidate:>6.2f} GiB/rank -- needs a TP rule for DSv4.1;")
    out.append(f"  {'':11}dsv41_layers.py has only expert_rank(), so EP-only is why")
    out.append(f"  {'':11}every rank carries the whole attention stack today.")
    out.append(f"  together    {certain + candidate:>5.2f} GiB/rank")
    return "\n".join(out)


def _main(argv: "list[str] | None" = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="/home/choiceoh/models/DeepSeek-V4.1-Flash-tp4")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--box-gib", type=float, default=None,
                        help="override the measured box (for other nodes)")
    parser.add_argument("--exclusive", action="store_true",
                        help="assume the box is ours alone (a real serving node)")
    parser.add_argument("--replication", action="store_true",
                        help="show what every rank carries a copy of")
    parser.add_argument("--repo", default="/home/choiceoh/models/DeepSeek-V4.1-Flash",
                        help="the HF checkpoint, for --layout reference")
    parser.add_argument("--wo-a", choices=("fp8", "bf16"), default="fp8")
    parser.add_argument("--chunk", type=int, default=4096,
                        help="prefill chunk in tokens; the activation line is a "
                             "function of it (engine/shapes.py says what is legal)")
    parser.add_argument("--vision", action="store_true", help="keep the vision tower")
    parser.add_argument("--layout", choices=("disk", "tp", "tp-notext", "reference"),
                        default="disk",
                        help="disk: the rank files as built (dense=replicate, mtp=replicate). "
                             "tp: what `preshard plan --dense tp --mtp ep` says. "
                             "tp-notext: same, minus the vision tower. "
                             "reference: derived here from inference/convert.py's own "
                             "axes over the real shard headers.")
    args = parser.parse_args(argv)

    total, free = (None, None)
    try:
        total, free = probe_box()
    except Exception as exc:  # no GPU here, or no torch
        print(f"  (device probe unavailable: {exc})")
    host_total, host_avail = host_box()

    print("box, measured on this node")
    if total is not None:
        print(f"  device total          {total:>8.2f} GiB")
        print(f"  device free           {free:>8.2f} GiB")
        if abs(total - host_total) < 0.01:
            print(f"  host MemTotal         {host_total:>8.2f} GiB   "
                  "== device total: one pool, so `device free` is the BOX's free,")
            print(f"  {'':22}{'':8}     not the GPU's. Host/device is the wrong model here.")
        else:
            print(f"  host MemTotal         {host_total:>8.2f} GiB")
    else:
        print(f"  host MemTotal         {host_total:>8.2f} GiB")
    print(f"  host MemAvailable     {host_avail:>8.2f} GiB   "
          f"(earlyoom fires at 5% = {host_total * 0.05:.2f} GiB)")
    print()

    tenants = 0.0 if args.exclusive or total is None else total - free
    override, note = None, ""
    if args.layout == "tp":
        override = PLAN_TP_RESIDENT_GIB
        note = "preshard plan --dense tp --mtp ep: 121.4/rank - engram 47.2 on SSD"
    elif args.layout == "reference":
        from engine.profiles.dsv41.placement import rank_plan, resident_gib
        plan = rank_plan(args.repo, args.world_size)
        override = resident_gib(plan, True, args.vision, args.wo_a)
        note = (f"inference/convert.py axes over the real shard headers: "
                f"engram on SSD, vision {'in' if args.vision else 'out'}, wo_a {args.wo_a}")
    elif args.layout == "tp-notext":
        override = PLAN_TP_RESIDENT_NOVISION_GIB
        note = ("preshard plan --dense tp --mtp ep: 121.4/rank - engram 47.2 on SSD "
                "- vision 0.9 (text only)")
    budget = for_dsv41(args.checkpoint, args.world_size, args.box_gib or total,
                       tenants, override, note, args.chunk)
    print(budget.table())
    print()
    print(f"  {budget.verdict()}")

    if budget.kv_gib > 0:
        try:
            from engine.profiles.dsv41.caches import report as kv_report
            print()
            print("what that KV buys -- DSv4.1 caches five buffers, and only four")
            print("of its forty layers produce KV at all (kv_source_layer_ids)")
            print()
            print(kv_report(args.repo, budget.kv_gib))
        except Exception as exc:
            print(f"\n  (kv_plan unavailable: {exc})")

    if args.replication:
        print()
        print("replication -- what every rank carries a copy of")
        print()
        print(replication_report(replication(args.checkpoint, args.world_size)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
