#!/usr/bin/env python3
"""Split the DeepSeek-V4.1-Flash checkpoint per rank, and say whether it fits.

The launcher bind-mounts the checkpoint per node and every rank reads all of it
-- GLM-5.3's boot log reads its whole 184.2 GiB on each of the four workers. At
475.2 GiB that is not a tight fit on this fleet, it is impossible: srv1 has 153
GiB free and the rest of its disk is the RUNNING GLM-5.3's checkpoint and fp8
cache. Sharding the checkpoint on disk is what lets a node hold only what its
rank reads.

`plan` answers the capacity question without writing anything, and without the
download having finished: shard headers are read locally where they exist and
fetched by range request where they do not.

    python3 tools/dsv41_preshard.py plan --repo /home/choiceoh/models/DeepSeek-V4.1-Flash
    python3 tools/dsv41_preshard.py build --rank 0 --out /path/to/rank0
    python3 tools/dsv41_preshard.py verify --rank 0 --out /path/to/rank0

`verify` reads the file BACK -- layout from `__metadata__`, the name set against
the source index, dtype and shape for every tensor, and bytes for a sample
stratified over the placement classes. It used to be `cmd_build` under another
name, so it "passed" by rewriting the file it was meant to be checking; the
selftest now requires it to reject a flipped byte, another rank's file under
this rank's name, and a tensor dropped from the header.

`--mtp` and `--dense` used to reach `plan` only: `build --mtp ep` reported the
saving and wrote a replicate layout. `--mtp ep` is now built; `--dense tp`
ABORTS in `build` rather than write a file the model side cannot read.

## How each tensor is placed

Every tensor must match exactly one rule. An unmatched name ABORTS rather than
falling through to "replicate", because a silent fallback is how a rank ends up
holding 60 GiB it never reads -- or, worse, how the one tensor that needed
splitting quietly did not.

Measured on the real checkpoint (all 48 shard headers, 2026-09-10), TP=4:

    placement      checkpoint     per rank
    engram           188.8 GiB      47.2 GiB    row blocks, ceil(N/W)
    expert           268.9 GiB      67.2 GiB    whole experts, 96/rank/layer
    mtp-expert         6.7 GiB       6.7 GiB    replicated (see --mtp)
    tp:0 + tp:1        7.3 GiB       7.3 GiB    replicated (see --dense)
    mtp-dense + vision + rest
                       3.4 GiB       3.4 GiB
    TOTAL            475.2 GiB     131.9 GiB

srv1 has 153 GiB free, so it fits with 21 GiB spare -- against a 322 GiB
shortfall today. `--mtp ep` takes it to 126.9 and `--dense tp` to 121.4, and
neither is needed to clear the bar.

The 40 MoE layers carry 384 experts each: 15,360 routed experts, 96 per rank.

  engram    the two [384,006,168 x 256] lookup tables, split the way the
            reference `ParallelEngramEmbedding` splits them: CONTIGUOUS ROW
            BLOCKS of `ceil(num_embeddings / world_size)`, rank r owning
            [r*part, (r+1)*part). Ids outside a rank's block are masked, zeroed
            and summed away by its `dist.all_reduce`.
            An earlier version of this file said the split was by the 24
            disjoint prime bucket ranges `EngramLayout` hands out. It is not,
            and a checkpoint split that way could not be loaded by the
            reference at all. probes/dsv41_engram_diff.py holds the row-block
            arithmetic to being bit-identical to that module.
  expert    the 9,984 routed-expert tensors (26 MoE layers x 384 experts).
            Distributed WHOLE, one expert to one rank, which is what expert
            parallelism already does on this fleet (glm53 runs ENABLE_EP=1).
            Nothing is sliced, so nothing depends on guessing a shard axis.
  mtp-*     the DSpark block. Its 128 routed experts per layer are 6.7 of its
            7.4 GiB and divide by four, but this fleet runs its drafter at
            draft_tensor_parallel_size=1 -- a full copy per rank -- so
            replication is the default and `--mtp ep` reports the alternative.
            Splitting a drafter the runtime then replicates would produce
            ranks missing experts they are about to route to.
  vision    the ViT and aligner, 0.9 GiB. Replicated, and not a judgement
            call: glm53 already runs its encoder MM_ENCODER_TP_MODE=data.
  tp:0/tp:1 a real tensor-parallel slice, along the output or input dim.
  replicate every rank keeps a full copy.

The expert and engram placements are facts about the checkpoint. The tp/
replicate split for the ~17 GiB of dense and attention weights is a HYPOTHESIS
until vLLM has DeepSeek-V4.1 model code to compare against, which is why
`--dense replicate` is the default: replicating 17 GiB costs 13 GiB per node
against a 153 GiB budget and cannot be wrong, whereas slicing on the wrong axis
produces a checkpoint that loads and computes garbage. `--dense tp` reports what
the slice would save.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import random
import re
import struct
import sys
import urllib.request
from pathlib import Path

HF_REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main/"

ELEM = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
        "F8_E8M0": 1, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "BOOL": 1}

# (engram_max_ngram_size 4 - 1) n-gram sizes x engram_n_heads 8 = the hash
# columns per token per table. NOT a partition unit -- the partition is by row
# block -- but the per-rank read count is Binomial(this, 1/world), which is what
# the I/O budget is drawn against.
ENGRAM_HASH_COLS = 3 * 8

# (label, pattern). First match wins; order is meaningful.
RULES: list[tuple[str, re.Pattern]] = [
    ("engram", re.compile(r"^layers\.\d+\.engram\.embed\.(weight|scale)$")),
    # the engram projections are tiny and every rank needs them
    ("replicate", re.compile(r"^layers\.\d+\.engram\.")),
    ("expert", re.compile(r"^layers\.\d+\.ffn\.experts\.\d+\.")),
    # The DSpark block's OWN MoE: dspark_n_routed_experts 128 over 3 MTP
    # layers. 6.7 of the block's 7.4 GiB, and it divides four ways -- but
    # whether it MAY be divided is a different question from whether it can,
    # which is what --mtp exists to keep visible. See the note there.
    ("mtp-expert", re.compile(r"^mtp\.\d+\.ffn\.experts\.\d+\.")),
    ("mtp-dense", re.compile(r"^mtp\.")),
    # output-dim slices: q/kv up-projections and the MoE-adjacent w1/w3
    ("tp:0", re.compile(r"^layers\.\d+\.(attn\.(wq_a|wq_b|wkv)|"
                        r"ffn\.shared_experts\.w[13])\.(weight|scale)$")),
    ("tp:0", re.compile(r"^layers\.\d+\.attn\.indexer\.(wq_b|wk|weights_proj)\."
                        r"(weight|scale)$")),
    ("tp:0", re.compile(r"^layers\.\d+\.attn\.compressor\.(wkv|wgate)\.weight$")),
    # input-dim slices: the o-projection tail and w2
    ("tp:1", re.compile(r"^layers\.\d+\.(attn\.wo_b|ffn\.shared_experts\.w2)\."
                        r"(weight|scale)$")),
    ("tp:0", re.compile(r"^layers\.\d+\.attn\.wo_a\.(weight|scale)$")),
    # everything else in a layer is small and shared: norms, gates, the mHC
    # coefficients, the attention sink
    ("replicate", re.compile(r"^layers\.\d+\.")),
    # The vision tower is 0.9 GiB with the aligner and this fleet already runs
    # its encoder data-parallel (glm53's MM_ENCODER_TP_MODE=data), which is
    # replication by another name. Nothing to decide.
    ("vision", re.compile(r"^(vision|aligner)\.")),
    # The LM head is vocab-parallel wherever TP is real, but it is 1.2 GiB and
    # replicating it costs less than being wrong about the axis; --dense tp
    # moves it with the rest.
    ("tp:0", re.compile(r"^head\.weight$")),
    ("replicate", re.compile(r"^(embed\.weight|norm\.weight|"
                             r"image_(start|end|newline))$")),
]


def nbytes(meta: dict) -> int:
    return meta["data_offsets"][1] - meta["data_offsets"][0]


def classify(name: str) -> "str | None":
    for label, pattern in RULES:
        if pattern.match(name):
            return label
    return None


def shard_headers(repo: Path, allow_remote: bool):
    """(shard, header) for every shard in the index, local first."""
    index_path = repo / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
    elif allow_remote:
        # The index is 12 MB and arrives whenever the downloader gets to it;
        # the capacity question does not wait for that.
        index = json.loads(urllib.request.urlopen(
            HF_BASE + "model.safetensors.index.json", timeout=180).read())
    else:
        raise SystemExit(
            f"ABORT: {index_path} is not there and --no-remote was given.")
    shards = sorted(set(index["weight_map"].values()))
    local = remote = 0
    for shard in shards:
        path = repo / shard
        if path.is_file():
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                head = json.loads(fh.read(n))
            local += 1
        elif allow_remote:
            req = urllib.request.Request(HF_BASE + shard,
                                         headers={"Range": "bytes=0-7"})
            n = struct.unpack("<Q", urllib.request.urlopen(req, timeout=60).read())[0]
            req = urllib.request.Request(HF_BASE + shard,
                                         headers={"Range": f"bytes=8-{8 + n - 1}"})
            head = json.loads(urllib.request.urlopen(req, timeout=180).read())
            remote += 1
        else:
            raise SystemExit(
                f"ABORT: {shard} is not downloaded and --no-remote was given. "
                f"The plan would be computed from a partial checkpoint and "
                f"would understate every total.")
        yield shard, head
    print(f"  headers: {local} local, {remote} fetched", file=sys.stderr)


def survey(repo: Path, allow_remote: bool) -> dict:
    """Bytes per placement class, plus the per-rank division of each."""
    totals: dict[str, int] = {}
    experts_seen: set[tuple[int, int]] = set()
    mtp_seen: set[tuple[int, int]] = set()
    unplaced: dict[str, tuple[str, list]] = {}
    for _shard, head in shard_headers(repo, allow_remote):
        for name, meta in head.items():
            if name == "__metadata__":
                continue
            label = classify(name)
            if label is None:
                # Collect them all rather than dying on the first: the useful
                # output is the whole list of names a rule has to cover, not
                # one of them at a time across as many runs as there are gaps.
                unplaced.setdefault(re.sub(r"\.\d+\.", ".N.", name),
                                    (meta["dtype"], list(meta["shape"])))
                continue
            totals[label] = totals.get(label, 0) + nbytes(meta)
            m = re.match(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.", name)
            if m:
                experts_seen.add((int(m.group(1)), int(m.group(2))))
            m = re.match(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.", name)
            if m:
                mtp_seen.add((int(m.group(1)), int(m.group(2))))
    if unplaced:
        lines = "\n".join(
            f"    {n:52s} {d:9s} {sh}" for n, (d, sh) in sorted(unplaced.items()))
        raise SystemExit(
            f"ABORT: {len(unplaced)} tensor name pattern(s) have no placement "
            f"rule:\n{lines}\n"
            f"  Add rules rather than letting them fall through -- an unplaced "
            f"tensor either bloats every rank or silently fails to be split.")
    layers = {l for l, _ in experts_seen}
    per_layer = len({e for l, e in experts_seen if l == min(layers)}) if layers else 0
    mtp_layers = {l for l, _ in mtp_seen}
    mtp_per = (len({e for l, e in mtp_seen if l == min(mtp_layers)})
               if mtp_layers else 0)
    return {"totals": totals, "moe_layers": len(layers), "experts": per_layer,
            "mtp_layers": len(mtp_layers), "mtp_experts": mtp_per}


GIB = float(1 << 30)


def cmd_plan(args) -> int:
    info = survey(Path(args.repo), not args.no_remote)
    totals, world = info["totals"], args.world_size
    # No divisibility condition on the engram side: ceil(N/W) row blocks work
    # for any world size, and the last block is simply short. The experts do
    # have one, because expert parallelism moves whole experts.
    if info["experts"] and info["experts"] % world:
        raise SystemExit(
            f"ABORT: {info['experts']} experts per layer do not divide over "
            f"{world} ranks -- expert parallelism would leave a rank short.")

    dense_tp = args.dense == "tp"
    mtp_ep = args.mtp == "ep"
    split = {"engram", "expert"}
    if dense_tp:
        split |= {"tp:0", "tp:1"}
    if mtp_ep:
        split.add("mtp-expert")
    per_rank: dict[str, float] = {}
    for label, byte_count in sorted(totals.items()):
        per_rank[label] = byte_count / world if label in split else byte_count

    total = sum(totals.values())
    rank_total = sum(per_rank.values())
    print(f"\nDeepSeek-V4.1-Flash, TP={world}, dense={args.dense}, "
          f"mtp={args.mtp}")
    print(f"  {'placement':12s} {'checkpoint':>12s} {'per rank':>12s}")
    print("  " + "-" * 38)
    for label in sorted(totals):
        print(f"  {label:12s} {totals[label] / GIB:9.1f} GiB "
              f"{per_rank[label] / GIB:9.1f} GiB")
    print("  " + "-" * 38)
    print(f"  {'TOTAL':12s} {total / GIB:9.1f} GiB {rank_total / GIB:9.1f} GiB")
    if not mtp_ep and totals.get("mtp-expert"):
        saving = totals["mtp-expert"] * (world - 1) / world / GIB
        print(f"\n  mtp=replicate holds the DSpark block whole on every rank. "
              f"--mtp ep\n  would save {saving:.1f} GiB/rank, and its 128 "
              f"experts do divide by {world} -- but this\n  fleet runs its "
              f"drafter at draft_tensor_parallel_size=1 (a full copy per\n  "
              f"rank) and nothing has established that V4.1's in-checkpoint "
              f"MTP differs.")
    print(f"\n  {info['moe_layers']} MoE layers x {info['experts']} experts "
          f"= {info['moe_layers'] * info['experts']} routed experts, "
          f"{info['experts'] // world} per rank per layer")
    print(f"  today a node needs the whole {total / GIB:.1f} GiB; "
          f"pre-sharded it needs {rank_total / GIB:.1f} GiB "
          f"({total / rank_total:.1f}x less)")

    if args.free:
        print(f"\n  {'node':6s} {'free':>9s} {'needed':>9s}   verdict")
        worst_ok = True
        for entry in args.free:
            node, _, gib = entry.partition("=")
            free = float(gib)
            ok = free >= rank_total / GIB
            worst_ok &= ok
            print(f"  {node:6s} {free:6.0f} GiB {rank_total / GIB:6.1f} GiB   "
                  + ("fits, %.0f GiB spare" % (free - rank_total / GIB) if ok
                     else "SHORT by %.0f GiB" % (rank_total / GIB - free)))
        if not worst_ok:
            print("\n  At least one node cannot hold its rank. Either free "
                  "space there or reduce the per-rank set further "
                  "(--dense tp is the next lever).")
            return 1
    return 0


def rank_of_expert(expert: int, world: int) -> int:
    """Contiguous blocks, so a rank's experts are adjacent in the source."""
    return expert // (384 // world) if world else 0


CHUNK = 64 << 20


def _shard_header(path: Path):
    """(header dict, byte offset where tensor data starts)."""
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        head = json.loads(fh.read(n))
    return head, 8 + n


# The partition is NOT defined here. It is a contract with the model's
# load_weights, so both import it; see expert_rank's docstring for what a
# second implementation costs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "overlay/modules/dsv41_model"))
from dsv41_layers import expert_rank  # noqa: E402


def _wanted(name: str, cfg) -> bool:
    """Does this rank need this tensor?

    Engram is excluded on purpose: it does not live in the rank's safetensors
    at all. tools/dsv41_engram_shard.py writes it as raw row blocks because it
    is read a row at a time off an SSD, not loaded.

    Everything else follows `cfg["mtp"]`. This used to ignore the mode flags
    entirely, so `build --mtp ep` wrote a replicate layout while `plan --mtp
    ep` reported the saving -- a flag that reports one thing and does another
    is worse than an absent flag, because the file it produces is correct-
    looking.
    """
    if ".engram.embed." in name:
        return False
    m = re.match(r"^layers\.\d+\.ffn\.experts\.(\d+)\.", name)
    if m:
        return expert_rank(int(m.group(1)), cfg["experts"],
                           cfg["world"]) == cfg["rank"]
    m = re.match(r"^mtp\.\d+\.ffn\.experts\.(\d+)\.", name)
    if m and cfg["mtp"] == "ep":
        return expert_rank(int(m.group(1)), cfg["mtp_experts"],
                           cfg["world"]) == cfg["rank"]
    return True


def _mode_cfg(args, info) -> dict:
    """The one place a build's layout is decided, so verify can re-derive it."""
    if args.dense == "tp":
        raise SystemExit(
            "ABORT: --dense tp is planned but not built. Splitting the dense "
            "tensors needs a per-parameter axis (wq_b along out, wo_b along "
            "in, and the LM head's vocab axis), and the model side does not "
            "read a split dense tensor yet. Writing one now produces a file "
            "that loads and computes garbage. `plan --dense tp` still shows "
            "what it would save.")
    return {"rank": args.rank, "world": args.world_size,
            "experts": info["experts"], "mtp_experts": info["mtp_experts"],
            "dense": args.dense, "mtp": args.mtp}


def cmd_build(args) -> int:
    """Write one rank's tensors as safetensors, streaming byte ranges.

    Safetensors rather than raw blobs: the rank file is then something an
    ordinary loader opens, and the only custom part left is knowing WHICH file
    to open. Streaming rather than loading, because the source tensors are
    larger than the host.

    Expert names keep their GLOBAL ids -- `layers.5.ffn.experts.100...` stays
    that, it is not renumbered to the rank's fourth expert. A renumbered file
    reads fine and routes wrong, and nothing downstream can tell.
    """
    repo, out = Path(args.repo), Path(args.out)
    if args.rank is None or not 0 <= args.rank < args.world_size:
        raise SystemExit(f"--rank must be in [0, {args.world_size})")
    info = survey(repo, not args.no_remote)
    per_layer = info["experts"]
    if per_layer % args.world_size:
        raise SystemExit(
            f"ABORT: {per_layer} experts per layer do not divide over "
            f"{args.world_size} ranks")
    cfg = _mode_cfg(args, info)

    index = json.loads((repo / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    missing = sorted({s for s in set(weight_map.values())
                      if not (repo / s).is_file()})
    if missing:
        raise SystemExit(
            f"ABORT: {len(missing)} shard(s) not downloaded (e.g. {missing[0]}). "
            f"Building from a partial checkpoint writes zeros where the tensors "
            f"are missing and nothing downstream can detect that.")

    take = [n for n in weight_map if _wanted(n, cfg)]
    heads = {}
    for shard in sorted(set(weight_map[n] for n in take)):
        heads[shard] = _shard_header(repo / shard)

    # Lay the output out first: safetensors offsets are relative to the start
    # of the data section, so every one has to be known before a byte is
    # written.
    header, cursor = {}, 0
    for name in sorted(take):
        meta = heads[weight_map[name]][0][name]
        size = meta["data_offsets"][1] - meta["data_offsets"][0]
        header[name] = {"dtype": meta["dtype"], "shape": meta["shape"],
                        "data_offsets": [cursor, cursor + size]}
        cursor += size
    # safetensors reserves `__metadata__` for str->str side data. Recording
    # the layout here is what lets `verify` -- and a loader -- refuse a file
    # built for a different world size or a different mode. Without it the
    # only evidence is the filename, which a copy or a rename loses.
    header["__metadata__"] = {
        "producer": "tools/dsv41_preshard.py",
        "rank": str(args.rank), "world_size": str(args.world_size),
        "dense": args.dense, "mtp": args.mtp,
        "n_routed_experts": str(info["experts"]),
        "dspark_experts": str(info["mtp_experts"]),
        "engram": "excluded (tools/dsv41_engram_shard.py)",
        "source_index_tensors": str(len(weight_map)),
    }
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-(8 + len(blob))) % 8          # keep the data section 8-aligned
    blob += b" " * pad

    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"rank{args.rank}of{args.world_size}.safetensors"
    written = 0
    with open(dst, "wb") as fo:
        fo.write(struct.pack("<Q", len(blob)))
        fo.write(blob)
        for name in sorted(take):
            shard = weight_map[name]
            src_head, base = heads[shard]
            meta = src_head[name]
            start = base + meta["data_offsets"][0]
            remaining = meta["data_offsets"][1] - meta["data_offsets"][0]
            with open(repo / shard, "rb") as fi:
                fi.seek(start)
                while remaining:
                    block = fi.read(min(CHUNK, remaining))
                    if not block:
                        raise SystemExit(
                            f"ABORT: {shard} ended early inside {name}")
                    fo.write(block)
                    remaining -= len(block)
                    written += len(block)
    print(f"  rank {args.rank}/{args.world_size}: {len(take):,} tensors, "
          f"{written / GIB:.1f} GiB -> {dst.name}")
    print(f"  (engram excluded; tools/dsv41_engram_shard.py writes it as raw "
          f"row blocks for the SSD path)")
    return 0


def cmd_verify(args) -> int:
    """Read a rank file back and hold it to the source checkpoint.

    This used to be `cmd_build` under another name -- `verify` rebuilt the file
    and reported success, which is the one outcome a verifier must never be
    able to produce by itself. What it checks now:

      layout    the file's `__metadata__` must say it was built for THIS rank,
                world size and mode. A rank1of4 file copied over rank0of4's
                name loads without complaint and routes every expert wrong.
      names     the set in the file must EQUAL what `_wanted` selects from the
                source index. Missing leaves a module at its initialization
                values; extra means this rank holds another rank's expert and
                will happily compute with it.
      types     dtype and shape per tensor against the source header.
      bytes     a sample, read from both sides, compared exactly. Sampling is
                stratified over `classify` labels so a whole placement class
                cannot go unsampled -- a uniform sample of 64 out of 24,000
                tensors misses `vision` and `mtp-dense` most of the time.

    Bytes are the expensive part, so `--samples` sets the count; the first
    three checks cover every tensor and cost a header read.
    """
    repo, out = Path(args.repo), Path(args.out)
    dst = out / f"rank{args.rank}of{args.world_size}.safetensors"
    if not dst.is_file():
        raise SystemExit(f"ABORT: {dst} does not exist -- build it first")
    info = survey(repo, not args.no_remote)
    cfg = _mode_cfg(args, info)

    head, base = _shard_header(dst)
    meta = head.pop("__metadata__", None)
    problems = []
    if meta is None:
        print("  layout   NO __metadata__ (built before it was recorded); "
              "falling back to the filename, which a copy does not preserve")
    else:
        want = {"rank": str(args.rank), "world_size": str(args.world_size),
                "dense": args.dense, "mtp": args.mtp}
        bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
        if bad:
            for k, (got, exp) in sorted(bad.items()):
                problems.append(f"layout: {k} is {got!r}, expected {exp!r}")
        else:
            print(f"  layout   rank {meta['rank']}/{meta['world_size']}, "
                  f"dense={meta['dense']}, mtp={meta['mtp']}, "
                  f"{meta['n_routed_experts']} experts/layer")

    index = json.loads((repo / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    want_names = {n for n in weight_map if _wanted(n, cfg)}
    got_names = set(head)
    missing, extra = sorted(want_names - got_names), sorted(got_names - want_names)
    if missing:
        problems.append(f"names: {len(missing)} missing, e.g. {missing[:3]}")
    if extra:
        problems.append(f"names: {len(extra)} not this rank's, e.g. {extra[:3]}")
    if not missing and not extra:
        print(f"  names    {len(got_names):,} tensors, exactly this rank's set")
    leaked = sorted(n for n in got_names if ".engram.embed." in n)
    if leaked:
        problems.append(f"engram: {len(leaked)} table tensor(s) in the rank "
                        f"file, e.g. {leaked[0]}")

    # dtype/shape for every shared name, bytes for a stratified sample
    shared = sorted(want_names & got_names)
    src_heads: dict[str, tuple] = {}

    def src(name):
        shard = weight_map[name]
        if shard not in src_heads:
            src_heads[shard] = _shard_header(repo / shard)
        return src_heads[shard]

    for name in shared:
        sh, _ = src(name)
        a, b = head[name], sh[name]
        if a["dtype"] != b["dtype"] or list(a["shape"]) != list(b["shape"]):
            problems.append(
                f"type: {name} is {a['dtype']}{a['shape']} here, "
                f"{b['dtype']}{b['shape']} in the source")
            if len(problems) > 20:
                break

    buckets: dict[str, list] = collections.defaultdict(list)
    for name in shared:
        buckets[classify(name) or "?"].append(name)
    rng = random.Random(args.seed)
    sample, per_bucket = [], max(1, args.samples // max(1, len(buckets)))
    for label in sorted(buckets):
        names = buckets[label]
        sample += rng.sample(names, min(per_bucket, len(names)))
    checked = 0
    with open(dst, "rb") as fo:
        for name in sample:
            sh, sbase = src(name)
            lo, hi = sh[name]["data_offsets"]
            with open(repo / weight_map[name], "rb") as fi:
                fi.seek(sbase + lo)
                want_bytes = fi.read(hi - lo)
            d_lo, d_hi = head[name]["data_offsets"]
            fo.seek(base + d_lo)
            got_bytes = fo.read(d_hi - d_lo)
            if got_bytes != want_bytes:
                n = next((i for i, (x, y) in enumerate(zip(got_bytes, want_bytes))
                          if x != y), min(len(got_bytes), len(want_bytes)))
                problems.append(f"bytes: {name} differs at offset {n:,} "
                                f"of {hi - lo:,}")
            checked += hi - lo
    print(f"  bytes    {len(sample)} tensors over {len(buckets)} placement "
          f"classes, {checked / GIB:.2f} GiB compared")

    if problems:
        print(f"\n  {len(problems)} PROBLEM(S):")
        for line in problems[:20]:
            print(f"    {line}")
        return 1
    print(f"  VERIFIED {dst.name} ({dst.stat().st_size / GIB:.1f} GiB)")
    return 0


SIDECARS = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "generation_config.json", "chat_template.jinja",
            "preprocessor_config.json")


def _write_index(out: Path, header: dict, shard_name: str, total: int) -> None:
    """A weight_map naming only what this directory actually holds.

    vLLM enumerates the files in the index and materializes every tensor it
    finds before any name filter can run. That is why the engram tables have to
    be absent from the INDEX rather than skipped by the loader: a
    `layers.1.engram.embed.weight` that reaches `get_tensor` is 94 GiB of host
    memory, and a skip_substrs check downstream of it is a check that runs
    after the OOM.
    """
    index = {"metadata": {"total_size": total},
             "weight_map": {name: shard_name for name in sorted(header)}}
    (out / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=1))


def _link_sidecars(repo: Path, out: Path) -> list[str]:
    """config.json and the tokenizer, COPIED so the directory stands alone.

    Copied rather than symlinked because a staged directory gets bind-mounted
    into a container, and a symlink pointing at a host path outside the mount
    resolves to nothing there -- which vLLM reports as "ensure the presence of
    a config.json", i.e. as a missing file rather than as a dangling link.
    They are kilobytes; the shards, which are not, stay symlinked and are made
    RELATIVE for the same reason.
    """
    linked = []
    for name in SIDECARS:
        src = repo / name
        if not src.is_file():
            continue
        dst = out / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil_copy(src, dst)
        linked.append(name)
    return linked


def _relative_symlink(src: Path, dst: Path) -> None:
    """Point dst at src by a relative path where one exists.

    A staged view is mounted somewhere else than it was written. An absolute
    link survives only if the target path exists identically inside the
    container; a relative one survives whenever the whole tree is mounted,
    wherever it lands.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        target = os.path.relpath(src, dst.parent)
    except ValueError:                      # different drives; cannot be relative
        target = str(src)
    os.symlink(target, dst)


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)


def _stage_dir(out: Path, rank, world) -> Path:
    return out if rank is None else out / f"rank{rank}"


def cmd_stage(args) -> int:
    """Write a directory vLLM can load directly. Two modes, one output shape.

    `--rank R`  stage the rank file `build` wrote: its own directory, holding
                a symlink to the weights, an index naming only them, and the
                sidecars. The launcher bind-mounts a per-node path at one
                container path, so every rank names the same --model while
                reading its own weights.
    `--view`    stage a VIEW of the full checkpoint: symlinks to the original
                shards and an index with the engram tables removed. Nothing is
                copied. This exists because the tables cannot be skipped by
                name at load time -- vLLM materializes every tensor the index
                names before any filter runs, and
                `layers.1.engram.embed.weight` is 94 GiB of host memory. The
                only place to remove them is the index.

    `--max-layers N` truncates the view to the first N layers, for a
    real-weights smoke test that fits on one node. It is a TEST artifact and
    says so in its index metadata, so a directory that ends up somewhere it
    should not be can be recognised rather than guessed at.
    """
    repo, out = Path(args.repo), Path(args.out)
    if args.view:
        index = json.loads((repo / "model.safetensors.index.json").read_text())
        weight_map = dict(index["weight_map"])
        # engram tables leave because they are 188.8 GiB the loader would
        # materialize; the vision tower leaves because this derivation has no
        # module for it and the load stops on the first such name.
        drop = re.compile(r"\.engram\.embed\.|^(vision|aligner)\.|"
                          r"^image_(start|end|newline)$")
        dropped_engram = [n for n in weight_map if drop.search(n)]
        for name in dropped_engram:
            del weight_map[name]
        dropped_layers = []
        if args.max_layers:
            for name in list(weight_map):
                m = re.match(r"^layers\.(\d+)\.", name)
                if m and int(m.group(1)) >= args.max_layers:
                    dropped_layers.append(name)
                    del weight_map[name]
        shards = sorted(set(weight_map.values()))
        missing = [s for s in shards if not (repo / s).is_file()]
        if missing:
            raise SystemExit(
                f"ABORT: {len(missing)} shard(s) the view needs are not "
                f"downloaded, e.g. {missing[0]}")
        out.mkdir(parents=True, exist_ok=True)
        for shard in shards:
            dst = out / shard
            _relative_symlink(repo / shard, dst)
        note = ("a VIEW of the full checkpoint with the engram tables removed"
                + (f", truncated to {args.max_layers} layers -- TEST ARTIFACT"
                   if args.max_layers else ""))
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": index["metadata"]["total_size"],
                          "dsv41_view": note},
             "weight_map": weight_map}, indent=1))
        linked = _link_sidecars(repo, out)
        print(f"  staged VIEW {out}")
        print(f"    shards     {len(shards)} symlinked, nothing copied")
        print(f"    index      {len(weight_map):,} tensors "
              f"(-{len(dropped_engram)} engram/vision"
              + (f", -{len(dropped_layers):,} beyond layer "
                 f"{args.max_layers - 1}" if args.max_layers else "") + ")")
        print(f"    sidecars   {', '.join(linked) or 'none found'}")
        if args.max_layers:
            print(f"    NOTE       truncated: a model built from this has "
                  f"{args.max_layers} layers and is a plumbing test, never a "
                  f"quality measurement")
        print(f"    load with  --model {out}"
              + (f" --hf-overrides '{{\"num_hidden_layers\": "
                 f"{args.max_layers}}}'" if args.max_layers else ""))
        return 0

    if args.rank is None:
        raise SystemExit("stage needs --rank R or --view")
    src = out / f"rank{args.rank}of{args.world_size}.safetensors"
    if not src.is_file():
        raise SystemExit(
            f"ABORT: {src} does not exist. Run `build --rank {args.rank}` "
            f"first; staging only writes the index and the sidecars.")
    stage = _stage_dir(out, args.rank, args.world_size)
    stage.mkdir(parents=True, exist_ok=True)
    _relative_symlink(src, stage / src.name)
    header, _ = _shard_header(src)
    header.pop("__metadata__", None)
    total = max(m["data_offsets"][1] for m in header.values())
    _write_index(stage, header, src.name, total)
    linked = _link_sidecars(repo, stage)
    engram = [n for n in header if ".engram.embed." in n]
    print(f"  staged rank {args.rank} at {stage}")
    print(f"    index      {len(header):,} tensors -> {src.name}")
    print(f"    sidecars   {', '.join(linked) or 'none found'}")
    print(f"    engram     {len(engram)} table tensor(s) "
          f"({'none, as intended' if not engram else 'PROBLEM'})")
    if engram:
        return 1
    print(f"    load with  --model {stage}")
    return 0


def cmd_selftest(args) -> int:
    """Build every rank from a synthetic checkpoint and check the partition.

    Three things, and the third is the one a byte comparison alone would miss:
    every tensor readable and byte-exact against the source; every expert
    present on exactly one rank; and expert NAMES unchanged, because a file
    that renumbered them would pass the first two and route to the wrong
    weights forever.
    """
    import shutil
    import tempfile

    world, n_exp, n_lay = args.world_size, 8, 2
    rows, k = 6, 4
    with tempfile.TemporaryDirectory(prefix="preshard-selftest-") as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        tensors, cursor, head = {}, 0, {}
        def add(name, fill):
            nonlocal cursor
            data = bytes([fill]) * (rows * k)
            tensors[name] = data
            head[name] = {"dtype": "F8_E4M3", "shape": [rows, k],
                          "data_offsets": [cursor, cursor + len(data)]}
            cursor += len(data)
        for L in range(n_lay):
            add(f"layers.{L}.attn_norm.weight", 0x11)
            for e in range(n_exp):
                add(f"layers.{L}.ffn.experts.{e}.w1.weight", (L * n_exp + e) & 0xFF)
        for e in range(n_exp):                         # the DSpark block
            add(f"mtp.0.ffn.experts.{e}.w1.weight", 0x40 | e)
        add("mtp.0.attn_norm.weight", 0x12)
        add("layers.0.engram.embed.weight", 0x99)      # must NOT appear
        blob = json.dumps(head, separators=(",", ":")).encode()
        blob += b" " * ((-(8 + len(blob))) % 8)
        shard = repo / "model-00001-of-00001.safetensors"
        with open(shard, "wb") as fh:
            fh.write(struct.pack("<Q", len(blob)))
            fh.write(blob)
            for name in head:
                fh.write(tensors[name])
        (repo / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {n: shard.name for n in head}}))

        out = Path(tmp) / "out"
        seen = {}
        for r in range(world):
            ns = argparse.Namespace(repo=str(repo), out=str(out), rank=r,
                                    world_size=world, dense="replicate",
                                    mtp="replicate", no_remote=True)
            if cmd_build(ns):
                return 1
            path = out / f"rank{r}of{world}.safetensors"
            with open(path, "rb") as fh:
                hl = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(hl))
                body = fh.read()
            h.pop("__metadata__", None)
            for name, meta in h.items():
                lo, hi = meta["data_offsets"]
                if body[lo:hi] != tensors[name]:
                    print(f"  FAIL rank {r}: {name} bytes differ")
                    return 1
                if ".engram.embed." in name:
                    print(f"  FAIL rank {r}: engram leaked into the rank file")
                    return 1
                m = re.match(r"^layers\.\d+\.ffn\.experts\.\d+\.", name)
                if m:
                    if name in seen:
                        print(f"  FAIL: {name} on both rank {seen[name]} and {r}")
                        return 1
                    seen[name] = r
        want = {n for n in tensors if re.match(r"^layers\.\d+\.ffn\.experts\.", n)}
        if seen.keys() != want:
            print(f"  FAIL: {len(seen)} expert tensors placed, {len(want)} exist")
            return 1
        # contiguity is a property the builder relies on and a strided split
        # would satisfy every other check here
        for L in range(n_lay):
            owners = [seen[f"layers.{L}.ffn.experts.{e}.w1.weight"]
                      for e in range(n_exp)]
            if owners != sorted(owners):
                print(f"  FAIL layer {L}: expert owners {owners} are not "
                      f"contiguous blocks -- a strided split routes wrong "
                      f"while passing every byte check")
                return 1
        # -- the verifier must be able to FAIL ---------------------------
        # A verify that cannot fail is what this file shipped with: `verify`
        # was cmd_build under another name, so it "passed" by rewriting the
        # file it was meant to be checking. Each of these is a mutation that
        # every other check in this selftest accepts.
        def verify_rank(r, **kw):
            ns = argparse.Namespace(repo=str(repo), out=str(out), rank=r,
                                    world_size=world, dense="replicate",
                                    mtp="replicate", no_remote=True,
                                    samples=64, seed=1)
            for k_, v in kw.items():
                setattr(ns, k_, v)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cmd_verify(ns)
            return rc, buf.getvalue()

        rc, _ = verify_rank(0)
        if rc:
            print("  FAIL: verify rejects a file this tool just built")
            return 1

        good = out / f"rank0of{world}.safetensors"
        keep = good.read_bytes()
        controls = []

        # 1. one flipped byte in the data section
        i = len(keep) - 1
        good.write_bytes(keep[:i] + bytes([keep[i] ^ 0xFF]))
        controls.append(("a flipped data byte", verify_rank(0)[0]))

        # 2. another rank's file under this rank's name -- same size, same
        #    shapes, every expert wrong
        good.write_bytes((out / f"rank1of{world}.safetensors").read_bytes())
        controls.append(("rank 1's file renamed to rank 0", verify_rank(0)[0]))

        # 3. a tensor removed from the header: the bytes that remain all
        #    match, and only the NAME SET shows it
        hl = struct.unpack("<Q", keep[:8])[0]
        h0 = json.loads(keep[8:8 + hl])
        drop = next(n for n in h0 if ".experts." in n)
        h0.pop(drop)
        nb = json.dumps(h0, separators=(",", ":")).encode()
        nb += b" " * ((-(8 + len(nb))) % 8)
        good.write_bytes(struct.pack("<Q", len(nb)) + nb + keep[8 + hl:])
        controls.append((f"{drop} dropped from the header", verify_rank(0)[0]))

        good.write_bytes(keep)
        for label, rc in controls:
            if rc == 0:
                print(f"  FAIL: verify accepted {label}")
                return 1

        # -- the mode flags must change the FILE, not just the plan --------
        ep_out = Path(tmp) / "out-ep"
        ns = argparse.Namespace(repo=str(repo), out=str(ep_out), rank=0,
                                world_size=world, dense="replicate",
                                mtp="ep", no_remote=True)
        if cmd_build(ns):
            return 1
        ep_head, _ = _shard_header(ep_out / f"rank0of{world}.safetensors")
        ep_head.pop("__metadata__", None)
        base_head, _ = _shard_header(good)
        base_head.pop("__metadata__", None)
        mtp_rep = {n for n in base_head if n.startswith("mtp.") and ".experts." in n}
        mtp_ep = {n for n in ep_head if n.startswith("mtp.") and ".experts." in n}
        if not mtp_rep or mtp_ep >= mtp_rep:
            print(f"  FAIL: --mtp ep kept {len(mtp_ep)} of {len(mtp_rep)} "
                  f"DSpark expert tensors -- the flag did not reach the build")
            return 1
        if verify_rank(0, out=str(ep_out), mtp="replicate")[0] == 0:
            print("  FAIL: verify accepted an --mtp ep file as replicate")
            return 1

        try:
            cmd_build(argparse.Namespace(
                repo=str(repo), out=str(Path(tmp) / "out-tp"), rank=0,
                world_size=world, dense="tp", mtp="replicate", no_remote=True))
        except SystemExit:
            pass
        else:
            print("  FAIL: --dense tp built a file; it is not implemented and "
                  "must abort rather than write a replicate layout")
            return 1

        per = collections.Counter(seen.values())
        print(f"  selftest: {n_lay} layers x {n_exp} experts over {world} ranks")
        print(f"    verify rejects: "
              + ", ".join(label for label, _ in controls))
        print(f"    --mtp ep keeps {len(mtp_ep)}/{len(mtp_rep)} DSpark expert "
              f"tensors; --dense tp aborts")
        print(f"    every tensor byte-exact, engram excluded, expert names kept")
        print(f"    experts per rank {dict(sorted(per.items()))} "
              f"(= {n_lay * n_exp // world} each)")
        shutil.rmtree(out, ignore_errors=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "build", "verify", "stage", "selftest"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default="/home/choiceoh/models/DeepSeek-V4.1-Flash")
        p.add_argument("--world-size", type=int, default=4)
        p.add_argument("--dense", choices=("replicate", "tp"), default="replicate")
        p.add_argument("--mtp", choices=("replicate", "ep"), default="replicate",
                       help="replicate matches this fleet's "
                            "draft_tensor_parallel_size=1; ep shards the "
                            "DSpark block's 128 experts")
        p.add_argument("--no-remote", action="store_true",
                       help="fail instead of range-fetching missing shard headers")
        if name == "plan":
            p.add_argument("--free", action="append", metavar="NODE=GIB",
                           help="check the per-rank total against a node's free "
                                "space, e.g. --free srv1=153")
        elif name == "selftest":
            p.add_argument("--out", default="")
        else:
            p.add_argument("--rank", type=int,
                           required=name in ("build", "verify"))
            p.add_argument("--out", required=True)
            if name == "stage":
                p.add_argument("--view", action="store_true",
                               help="stage a filtered view of the full "
                                    "checkpoint instead of a built rank file")
                p.add_argument("--max-layers", type=int, default=0,
                               help="truncate the view to the first N layers; "
                                    "a test artifact, marked as one")
            if name == "verify":
                p.add_argument("--samples", type=int, default=64,
                               help="tensors to compare byte-for-byte, spread "
                                    "over the placement classes")
                p.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()
    return {"plan": cmd_plan, "build": cmd_build, "stage": cmd_stage,
            "verify": cmd_verify, "selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
