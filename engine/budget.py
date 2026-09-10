"""The box, declared -- not discovered.

vLLM spends 36 s of every boot answering one question: how much is left for KV.
Four boots measured it at 36.1 / 36.2 / 37.2 / 36.5 s -- a spread under one
second, because the answer is the same every time. It gets that answer by
running a synthetic max-shape batch, which costs a 17.7 GiB peak
(``profile-run`` +9.17, ``profile/determine-memory`` +8.56 in the 40th
campaign's boot table). Peak is what earlyoom kills on: six workers died at
``MemAvailable`` 5%, all of them while serving.

So D1 says the engine owns the box. This module is what that means: a budget is
a list of claims on a fixed number, every claim carries where it came from, and
KV is what is left -- known before anything is allocated.

The provenance tag is the point, not decoration:

    measured    read off this machine, right now (mem_get_info, /proc/meminfo)
    read        read out of a file that is a constant (a safetensors header)
    ledger      measured on a previous boot and written down in MEASUREMENTS.md
    declared    a number we choose and then hold ourselves to
    estimated   a guess, usually scaled from a different model

A budget with an ``estimated`` line is not a boot gate. It says so itself.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

GIB = 1 << 30

MEASURED = "measured"
READ = "read"
LEDGER = "ledger"
DECLARED = "declared"
ESTIMATED = "estimated"

_ORDER = (MEASURED, READ, LEDGER, DECLARED, ESTIMATED)


@dataclass(frozen=True)
class Line:
    """One claim on the box."""

    name: str
    gib: float
    source: str
    evidence: str

    def __post_init__(self):
        if self.source not in _ORDER:
            raise ValueError(f"unknown provenance {self.source!r}")


class Budget:
    """A box, the claims on it, and whatever is left for KV."""

    def __init__(self, box_gib: float, lines: "list[Line]", label: str = ""):
        self.box_gib = box_gib
        self.lines = list(lines)
        self.label = label

    @property
    def claimed_gib(self) -> float:
        return sum(line.gib for line in self.lines)

    @property
    def kv_gib(self) -> float:
        return self.box_gib - self.claimed_gib

    @property
    def estimated(self) -> "list[Line]":
        return [line for line in self.lines if line.source == ESTIMATED]

    def is_gate(self) -> bool:
        """A budget can gate a boot only when nothing in it is a guess."""
        return not self.estimated and self.kv_gib > 0

    def table(self) -> str:
        width = max((len(line.name) for line in self.lines), default=0)
        width = max(width, len("box"), len("KV (remainder)"))
        out = [f"{self.label}" if self.label else "", ""]
        out.append(f"  {'box':<{width}}  {self.box_gib:>8.2f} GiB")
        out.append(f"  {'-' * width}  {'-' * 8}")
        for line in self.lines:
            out.append(
                f"  {line.name:<{width}}  {-line.gib:>8.2f} GiB   "
                f"[{line.source}] {line.evidence}"
            )
        out.append(f"  {'-' * width}  {'-' * 8}")
        out.append(f"  {'KV (remainder)':<{width}}  {self.kv_gib:>8.2f} GiB")
        return "\n".join(x for x in out if x is not None)

    def verdict(self) -> str:
        if self.kv_gib <= 0:
            return (
                f"DOES NOT FIT: claims exceed the box by {-self.kv_gib:.2f} GiB. "
                "Something must leave the box before KV gets anything."
            )
        guesses = self.estimated
        if guesses:
            names = ", ".join(line.name for line in guesses)
            total = sum(line.gib for line in guesses)
            return (
                f"NOT A GATE: {len(guesses)} estimated line(s) -- {names} -- "
                f"own {total:.2f} GiB, {total / self.box_gib:.0%} of the box and "
                f"{total / max(self.kv_gib, 1e-9):.1f}x the KV they leave. "
                "This budget's conclusion is theirs, not the measurements'."
            )
        return f"GATE: KV = {self.kv_gib:.2f} GiB, declared before load."


# --------------------------------------------------------------------------
# probes -- each returns a Line whose provenance is honest about its origin
# --------------------------------------------------------------------------

def probe_box() -> "tuple[float, float]":
    """(total device GiB, free GiB after our own context).

    Asking costs a CUDA context, so the difference is a real measurement of
    what a context costs on this node -- and the reason instruments.py asks
    ``is_initialized()`` first when it must NOT pay that cost.
    """
    import torch

    free, total = torch.cuda.mem_get_info()
    return total / GIB, free / GIB


def host_box() -> "tuple[float, float]":
    """(MemTotal GiB, MemAvailable GiB) -- what earlyoom actually counts."""
    fields = {}
    for raw in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = raw.partition(":")
        fields[key] = float(rest.strip().split()[0]) / (1 << 20)
    return fields["MemTotal"], fields["MemAvailable"]


def rank_weights(path: "str | Path") -> "tuple[float, int]":
    """(GiB, tensor count) read out of a safetensors header.

    The header is a constant of the checkpoint, so this is a `read`, not a
    measurement and not an estimate. It is also cheap: one 8-byte read plus
    the header, never the 84 GiB behind it.
    """
    path = Path(path)
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    end = max(entry["data_offsets"][1] for entry in header.values())
    return end / GIB, len(header)


def rank_files(checkpoint: "str | Path", world_size: int) -> "list[Path]":
    root = Path(checkpoint)
    files = [root / f"rank{i}of{world_size}.safetensors" for i in range(world_size)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError(f"presharded ranks missing: {', '.join(missing)}")
    return files


# --------------------------------------------------------------------------
# the DSv4.1 budget
# --------------------------------------------------------------------------

# 40th campaign, boot phase table (GLM-5.3, A9221 boot, PR #504).
# init-device 2.29 + dist-group-ep 1.00 + dist-model-parallel 2.25.
# NCCL is configured with 16 channels; that setting's memory cost lives here.
RUNTIME_FLOOR_GIB = 5.54

# Same table: load-model took 59.17 GiB while vLLM reported 50.4 GiB of
# weights. The 8.77 GiB difference is pack/quant scratch, the drafter, and
# cuBLAS init. Scaling a ratio measured on a different model is a guess, and
# this module is required to say so.
LOAD_SCRATCH_RATIO = 8.77 / 50.4

# Same table again: profile-run peaked at +9.17 GiB on GLM-5.3. DSv4.1 splits
# encoder/decoder (CED), gathers engram rows, and carries three MTP layers, so
# there is no reason its activation peak matches. Another guess, said so.
ACTIVATION_PEAK_GIB = 9.17

# earlyoom sends SIGTERM at MemAvailable 5%. Six workers died there, all while
# serving. Reserving exactly 5% means racing it, so the default is twice the
# floor -- a CHOICE, which is why it prints as `declared` and not as a fact.
# The mechanism that makes the reserve binding (cgroup / oom_score_adj / mlock)
# is still open; see CHARTER.md section 4.
OS_RESERVE_MULTIPLE = 2.0


def for_dsv41(checkpoint: "str | Path", world_size: int = 4,
              box_gib: "float | None" = None,
              tenants_gib: float = 0.0) -> Budget:
    files = rank_files(checkpoint, world_size)
    sizes = [rank_weights(f) for f in files]
    weights_gib = max(g for g, _ in sizes)
    tensors = max(n for _, n in sizes)
    spread = weights_gib - min(g for g, _ in sizes)

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
        Line("weights (this rank)", weights_gib, READ,
             f"rank*of{world_size}: {tensors:,} tensors, spread across ranks "
             f"{spread * 1024:.0f} MiB"),
        Line("load scratch (pack/quant/drafter/cuBLAS)",
             weights_gib * LOAD_SCRATCH_RATIO, ESTIMATED,
             f"GLM ratio {LOAD_SCRATCH_RATIO:.1%} of weights -- NOT measured on dsv41"),
        Line("activation peak", ACTIVATION_PEAK_GIB, ESTIMATED,
             "GLM profile-run +9.17 GiB -- dsv41 has CED, engram, MTP x3; not measured"),
    ]
    return Budget(box_gib, lines, label=f"DSv4.1-Flash, rank of {world_size}, this box")


# --------------------------------------------------------------------------
# replication -- the part of a rank that every other rank also carries
# --------------------------------------------------------------------------

# A tensor identical in name, dtype and shape on all four ranks is paid four
# times. Some of that is correct (norms are tiny and every rank needs them);
# some of it is a presharding choice that can be revisited. The classifier
# below says which is which, and what it would take to stop paying.
# `certain` levers need no rule that does not already exist. `candidate` ones
# need a TP rule for DSv4.1, and dsv41_layers.py has only `expert_rank` -- EP.
# That is the finding: this checkpoint is presharded EP-only, so every rank
# carries the whole attention stack and the whole shared expert.
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
    budget = for_dsv41(args.checkpoint, args.world_size, args.box_gib or total, tenants)
    print(budget.table())
    print()
    print(f"  {budget.verdict()}")

    if args.replication:
        print()
        print("replication -- what every rank carries a copy of")
        print()
        print(replication_report(replication(args.checkpoint, args.world_size)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
