"""Compare full-row and incremental GLM block-table publication on CUDA.

Export caches.py at the baseline commit and pass --baseline PATH. The probe
uses the actual Glm53Caches arena and BlockPool with a tiny one-layer DSA shape;
no model, collective or NVMe I/O is timed. Reservation/reuse setup runs outside
each timed sample. Both prepare methods see the same mapping and publication
history; full GPU rows are checked against their host oracle in every case.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.arena import Arena
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.net import Segment, Step
from engine_decode_overhead import summary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(caches, batch, mode, baseline, args):
    pool = caches.pool
    seqs = list(range(batch))
    slots = [caches.slots.take(seq) for seq in seqs]
    ids = torch.zeros(batch, dtype=torch.int64, device="cuda")
    step = Step(ids, tuple(Segment(seq, slot, 0, seq, 1) for seq, slot in zip(seqs, slots)))

    def setup():
        if mode == "unchanged":
            return
        for seq in seqs:
            pool.release(seq)
            pool.reserve(seq, (32 if mode == "reuse_shorter" else 8) * pool.block_size)
        caches.prepare(step)
        for seq in seqs:
            if mode == "reuse_shorter":
                pool.release(seq)
                pool.reserve(seq, 8 * pool.block_size)
            else:
                pool.reserve(seq, pool.block_size)

    def check():
        for seq in seqs:
            assert caches.block_table[seq].tolist() == list(pool.row(seq)), (mode, seq)

    for seq in seqs:
        pool.reserve(seq, 8 * pool.block_size)
    caches.prepare(step)
    fns = [lambda: baseline.prepare(caches, step), lambda: caches.prepare(step)]
    for fn in fns:
        for _ in range(args.warmup):
            setup()
            fn()
        check()
    wall, host, stream, by_round, copy_ops = [[], []], [[], []], [[], []], [[], []], []
    for r in range(args.rounds):
        for which in ((0, 1) if r % 2 == 0 else (1, 0)):
            offset = len(wall[which])
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for _ in range(args.samples):
                setup()
                torch.cuda.synchronize()
                start.record()
                begin = time.perf_counter_ns()
                fns[which]()
                host[which].append((time.perf_counter_ns() - begin) / 1000)
                end.record()
                end.synchronize()
                wall[which].append((time.perf_counter_ns() - begin) / 1000)
                stream[which].append(start.elapsed_time(end) * 1000)
            check()
            by_round[which].append(summary(wall[which][offset:]))
    for fn in fns:
        setup()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
            fn()
        check()
        copy_ops.append([{"operator": event.key, "calls": event.count, "shapes": event.input_shapes}
                         for event in prof.key_averages(group_by_input_shape=True)
                         if event.key in ("aten::copy_", "aten::fill_")])
    for seq, slot in zip(seqs, slots):
        pool.release(seq)
        caches.slots.give(slot)
    return {"batch": batch, "mode": mode, "max_blocks_per_row": pool.max_blocks_per_seq,
            "full_table_matches_host": True,
            **{name: {"wall": summary(wall[i]), "host_call": summary(host[i]),
                      "cuda_stream": summary(stream[i]), "round_wall": by_round[i], "copy_ops": copy_ops[i]}
               for i, name in enumerate(("baseline", "optimized"))}}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples", type=int, default=80)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()
    assert min(args.rounds, args.samples, args.warmup) > 0
    assert torch.cuda.is_available()
    spec = importlib.util.spec_from_file_location("baseline_caches", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    root = Path(__file__).resolve().parents[1]
    report = {"scope": "GLM block-table publication only; setup/model/collectives/I/O excluded",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "platform": platform.platform(),
              "source_sha256": {"baseline_caches.py": digest(args.baseline),
                                **{name: digest(root / name) for name in ("engine/base/kv.py", "engine/profiles/glm53/caches.py")}},
              "protocol": {"rounds": args.rounds, "samples_per_round": args.samples,
                           "warmup": args.warmup, "order": "alternating baseline/optimized then optimized/baseline",
                           "synchronization": "before timing and end event per sample"}, "cases": []}
    F = SimpleNamespace(layers=1, block=16, kpool=4, idx_dim=128, kv_lora=16, spec_k=5, is_dsa=lambda L: True)
    for blocks in (512, 4096):
        arena = Arena(layout(F, [0]).nbytes(blocks, 4))
        caches = Glm53Caches(arena, F, [0], blocks, 4)
        for batch in (1, 4):
            for mode in ("unchanged", "grow_one", "reuse_shorter"):
                item = measure(caches, batch, mode, baseline.Glm53Caches, args)
                report["cases"].append(item)
                print(json.dumps(item), flush=True)
        del caches, arena
        torch.cuda.empty_cache()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
