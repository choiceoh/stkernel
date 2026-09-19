"""Qwen3.8's prefill chunk on one GB10, family by family (probe, single-GPU lane).

The fleet has a prompt's time as a whole -- 1,767 tokens in about 0.5 s warm, 4,827 tokens in 3.8 s with their decode,
a 32,256-token chunk's memory pass in 26 s at boot (measurements/qwen38_fleet_boot_20260918, qwen38_boot_window_20260918)
-- and nothing inside it: no record says whether a chunk is its MoE, its GDN recurrence, QSA's scoring and selection,
the hyper-connection mixers or the host waiting on the device between them. The decode step has that census
(probes/engine_qwen38_step.py); this is the prefill's, built on the same harness: the served net for ONE rank of TP=4
from the rank file's own weights, the collectives this rank's own contribution (OneRankComm), the PLE table zeros.

What is measured is what the server runs: adapter.build_model's model, `model.prefill(seq, start, tokens, ...)` -- the
target's uncut forward over the chunk (#1183) and the MTP head's observation of it -- for the prompt's first chunk
(context 0) and its second (context `chunk`: QSA's scoring and selection read what the first one stored). Each chunk
runs three times over fresh sequences: once to compile, once for the wall clock (host and device, no profiler), once
under the CUDA profiler for the device time of each kernel family. Wall less device time is the host's share: the
eager step's launches and its reads of the device (the MoE's dynamic prefill reads its route counts a layer -- carry M4).

Beside production the lane leaves a probe about 4 GiB, so as the step probe does, a chunk is solved from small nets:

    [4, 5, 6, 7]   fixed + 3 GDN + 1 QSA        [4, 5]   fixed + 2 GDN
    [7]            fixed + 1 QSA                [1]      fixed + 1 GDN + PLE

    chunk(48 layers) = fixed + 36 GDN + 12 QSA + PLE         (fixed: embedding, head, closing mixer, the MTP head)

    python3 probes/engine_kernel_check.py --lanes qwen38_prefill --ranks /home/choiceoh/models/st-qwen38-tep4 \\
        --output /cache/qwen38-prefill.json                                       (the queue's single-GPU lane)

`--lanes qwen38_prefill:8192` sets the chunk's tokens (default 4096). The process's device-memory ceiling is the
ticket's budget (ST_PROBE_GIB, which the lane exports: a kernel check's 8) less a GiB, 4 GiB without one.

One rank's compute and launches, random token ids through real weights: the router sees real embeddings, so this
rank's share of the routes is a real one, but it is not a prompt's. No collective is timed. Not a speed claim (D17):
it says what a chunk is made of, so the next prefill work is chosen by a number.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.engine_qwen38_step import (FULL, LAYER_SETS, OneRankComm, ZeroPLETable, counts, extrapolate,  # noqa: E402
                                       family, solve)

CHUNK = 4096
CHUNKS = 2                                # the prompt's first chunk and its second: contexts 0 and CHUNK
MAX_GIB = 4.0                             # this process's device-memory ceiling where the ticket names no budget: the
                                          # lane exports its own as ST_PROBE_GIB, and the ceiling follows it
CONTEXT_GIB = 1.0                         # of a ticket's budget: the CUDA context, cuBLAS, Triton and deep_gemm's modules
TOP = 25                                  # kernels kept a chunk, by device time
SLACK = 128                               # tokens of cache beyond the prompt (two blocks: the MTP head's reach)


def build(ranks: Path, rank: int, layers, *, tokens: int):
    """The served net and caches for one rank over `layers`, no graphs, with room for one sequence of `tokens` ->
    (F, net, caches). The cache is sized from the tokens it must hold, not from a share of memory."""
    from engine.base.arena import Arena
    from engine.base.params import total_bytes
    from engine.profiles.qwen38 import facts, lanes as lane_tables
    from engine.profiles.qwen38.caches import Qwen38Caches, layout, snapshot_layout
    from engine.profiles.qwen38.fleet import rank_loader
    from engine.profiles.qwen38.net import Qwen38Net

    F = facts.load(ranks)
    net = Qwen38Net(F, OneRankComm(rank), lane_tables.served(), layers=list(layers), mtp=True)
    specs = net.specs()
    nb, snapshots, max_seqs = -(-tokens // F.block) + 8, 2, 1
    arena = Arena(total_bytes(specs) + 256 * (len(specs) + 64) + layout(F, net.layers, mtp=True).nbytes(nb, max_seqs)
                  + snapshots * snapshot_layout(F, net.layers)[0])
    loader = rank_loader(ranks / f"rank{rank}of{facts.TP}.safetensors", expected_layout=F.weight_layout)
    net.bind(loader.load([s.name for s in specs], arena=arena))
    if net._ple is not None:
        net.attach_ple(ZeroPLETable(F.ple_rows_per_rank, F.ple_head_dim, float(net._ple_scale)),
                       max_rows=max_seqs * (F.spec_k + 1))
    net.prepare_dense(None)
    return F, net, Qwen38Caches(arena, F, net.layers, nb, max_seqs, snapshots, mtp=True)


def families_of(prof) -> "tuple[dict, dict]":
    """A profile's device kernels -> ({family: [launches, us]}, {kernel: [launches, us]})."""
    kernels, families = {}, {}
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", None)
        if us is None:
            us = getattr(e, "self_cuda_time_total", 0)
        if not us or e.count <= 0:
            continue
        for table, key in ((kernels, e.key), (families, family(e.key))):
            entry = table.setdefault(key, [0, 0.0])
            entry[0] += e.count
            entry[1] += us
    return families, kernels


def prompt_passes(model, caches, F, *, chunk: int, chunks: int, seed: int) -> "list[dict]":
    """The same prompt's chunks three times, a fresh sequence each: compile, wall, profile -> a row a chunk."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    total = chunk * chunks + F.block                         # longer than what is prefilled: no first token is sampled
    ids = torch.randint(0, F.vocab, (total,), generator=torch.Generator(device="cpu").manual_seed(seed)).tolist()
    rows = [{"context": i * chunk, "tokens": chunk} for i in range(chunks)]
    for run in ("compile", "wall", "profile"):
        seq = 0
        slot = caches.slots.take(seq)
        try:
            caches.pool.reserve(seq, total)
            model.add(seq, ids, temperature=0.0)
            model.open(seq, slot)
            for row in rows:
                torch.cuda.synchronize()
                began = time.perf_counter()
                if run == "profile":
                    with profile(activities=[ProfilerActivity.CUDA]) as prof:
                        model.prefill(seq, row["context"], chunk, None, slot)
                        torch.cuda.synchronize()
                    fams, kernels = families_of(prof)
                    row["device_ms"] = round(sum(v[1] for v in kernels.values()) / 1e3, 2)
                    row["launches"] = int(sum(v[0] for v in kernels.values()))
                    row["families"] = {k: {"launches": v[0], "ms": round(v[1] / 1e3, 3)}
                                       for k, v in sorted(fams.items(), key=lambda kv: -kv[1][1])}
                    row["kernels"] = {k[:160]: {"launches": v[0], "ms": round(v[1] / 1e3, 3)}
                                      for k, v in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:TOP]}
                else:
                    model.prefill(seq, row["context"], chunk, None, slot)
                    torch.cuda.synchronize()
                    if run == "wall":
                        row["wall_ms"] = round((time.perf_counter() - began) * 1e3, 2)
        finally:
            model.close(seq)
            caches.pool.release(seq)
            caches.slots.give(slot)
            caches.reset()
    for row in rows:
        row["host_ms"] = round(row["wall_ms"] - row["device_ms"], 2)
    return rows


def ceiling_gib(environ=None) -> float:
    """What this process may allocate on the device: the ticket's budget (ST_PROBE_GIB -- the lane exports its own, a
    kernel check's 8, when the submitter names none) less CONTEXT_GIB for what torch's allocator does not see, else
    MAX_GIB. The decode census's four-layer set peaks at 3.7 GiB with no chunk's activations."""
    value = (os.environ if environ is None else environ).get("ST_PROBE_GIB", "")
    return max(1.0, float(value) - CONTEXT_GIB) if value else MAX_GIB


def measure(ranks: Path, rank: int, layers, *, chunk: int = CHUNK, chunks: int = CHUNKS, max_gib: "float | None" = None) -> dict:
    """One layer set, in this process: the kernel shape bound, the net built, the prompt's chunks -> the build's row."""
    import torch
    max_gib = ceiling_gib() if max_gib is None else max_gib
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, max_gib * (1 << 30) / total))
    from engine.base import kernel_shape
    from engine.profiles.qwen38 import facts
    from engine.profiles.qwen38.adapter import build_model
    _, shape_source = kernel_shape.bind_recorded(ranks, ranks / "config.json", lambda: facts.load(ranks).kernel_shape())
    began = time.perf_counter()
    F, net, caches = build(ranks, rank, layers, tokens=chunk * chunks + SLACK)
    built = time.perf_counter() - began
    model, _ = build_model(net, caches, F, eos_ids=[F.vocab + 7], max_new=16, temperature=0.0, top_p=1.0, seed=0,
                           drafter=True)
    rows = prompt_passes(model, caches, F, chunk=chunk, chunks=chunks, seed=1)
    for row in rows:
        print(json.dumps({"layers": list(layers), "context": row["context"], "tokens": row["tokens"],
                          "wall_ms": row["wall_ms"], "device_ms": row["device_ms"], "host_ms": row["host_ms"],
                          "launches": row["launches"]}), flush=True)
    return {"counts": counts(F, layers), "kernel_shape": shape_source, "free_GiB_at_start": round(free / 2**30, 1),
            "ceiling_GiB": max_gib,
            "build_s": round(built, 1), "peak_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2), "chunks": rows}


def assemble(builds: dict) -> dict:
    """Per chunk: the four unknowns solved from the layer sets -- wall, device and host time, launches, each family's
    device time -- and the 48-layer chunk they add up to, largest family first."""
    out = {}
    contexts = sorted({c["context"] for b in builds.values() for c in b["chunks"]})
    for context in contexts:
        rows = [(b["counts"], next(c for c in b["chunks"] if c["context"] == context)) for b in builds.values()]
        if len(rows) < 4:
            continue
        entry = {"tokens": rows[0][1]["tokens"]}
        for metric in ("wall_ms", "device_ms", "host_ms", "launches"):
            parts = solve([(c, r[metric]) for c, r in rows])
            entry[metric] = {"parts": {u: round(v, 2) for u, v in parts.items()}, "chunk_48": round(extrapolate(parts), 1)}
        fams = {}
        for fam in sorted({f for _, r in rows for f in r["families"]}):
            parts = solve([(c, r["families"].get(fam, {}).get("ms", 0.0)) for c, r in rows])
            fams[fam] = {"parts_ms": {u: round(v, 3) for u, v in parts.items()}, "chunk_48_ms": round(extrapolate(parts), 1)}
        entry["families"] = dict(sorted(fams.items(), key=lambda kv: -kv[1]["chunk_48_ms"]))
        entry["tokens_a_second_one_rank"] = round(entry["tokens"] / max(entry["wall_ms"]["chunk_48"], 1e-9) * 1e3, 1)
        out[f"context {context}"] = entry
    return out


def run(output=None, ranks=None, *, layer_sets=LAYER_SETS, rank: "int | None" = None, chunk: int = CHUNK) -> dict:
    """The lane (probes/engine_kernel_check.py --lanes qwen38_prefill[:tokens]): each layer set in a process of its own
    (a built net's weights stay referenced by the lanes' prepared views), then assembled."""
    import subprocess
    import tempfile
    ranks = Path(ranks or "/home/choiceoh/models/st-qwen38-tep4")
    if rank is None:
        present = sorted(int(p.name[4]) for p in ranks.glob("rank?of4.safetensors"))
        if not present:
            raise SystemExit(f"no rank file under {ranks}")
        rank = present[-1]
    report = {"rank": rank, "layer_sets": [list(s) for s in layer_sets], "chunk": chunk, "chunks": CHUNKS, "full": FULL,
              "builds": {}, "failed": {}}
    with tempfile.TemporaryDirectory() as scratch:
        for layers in layer_sets:
            name = ",".join(map(str, layers))
            out = Path(scratch) / f"{name}.json"
            done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", name, "--ranks", str(ranks),
                                   "--rank", str(rank), "--chunk", str(chunk), "--output", str(out)], cwd=str(ROOT))
            if done.returncode or not out.exists():
                report["failed"][name] = f"rc={done.returncode}"
                continue
            report["builds"][name] = json.loads(out.read_text())
    report["chunk_48"] = assemble(report["builds"])
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: {"wall_ms": v["wall_ms"]["chunk_48"], "device_ms": v["device_ms"]["chunk_48"],
                          "host_ms": v["host_ms"]["chunk_48"], "launches": v["launches"]["chunk_48"],
                          "families_ms": {f: e["chunk_48_ms"] for f, e in v["families"].items()}}
                      for k, v in report["chunk_48"].items()}, indent=1), flush=True)
    if report["failed"]:
        raise RuntimeError(f"layer sets that did not build or run: {report['failed']}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ranks", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--one", default=None, help="one layer set (comma separated), measured in this process")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    a = ap.parse_args()
    if a.one is not None:
        row = measure(Path(a.ranks), a.rank, tuple(int(x) for x in a.one.split(",")), chunk=a.chunk)
        Path(a.output).write_text(json.dumps(row) + "\n")
    else:
        run(a.output, a.ranks, rank=a.rank, chunk=a.chunk)
