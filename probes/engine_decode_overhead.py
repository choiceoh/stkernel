"""Paired eager GLM adapter microbenchmarks; no model/collective is executed.

Export the baseline adapter with git show, then pass --baseline PATH. Both
versions use the same installed torch, base sampler, inputs and device. The
decode-prefix probe stops at _forward, so it times the actual input assembly
without mixing in a synthetic model's latency. CUDA event spans include host
launch gaps; neither they nor wall times are a full-model ITL measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.profiles.glm53.adapter import Glm53Engine


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    values = sorted(values)
    return {"median_us": statistics.median(values), "p95_us": values[math.ceil(len(values) * .95) - 1]}


def paired(functions, *, rounds, samples, warmup, cuda=True):
    for fn in functions:
        for _ in range(warmup):
            fn()
    if cuda:
        torch.cuda.synchronize()
    wall, stream, peak, rounds_wall = [[], []], [[], []], [0, 0], [[], []]
    for repeat in range(rounds):
        for which in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            fn = functions[which]
            offset = len(wall[which])
            if cuda:
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                for _ in range(samples):
                    start.record()
                    begin = time.perf_counter_ns()
                    result = fn()
                    end.record()
                    end.synchronize()
                    wall[which].append((time.perf_counter_ns() - begin) / 1000)
                    stream[which].append(start.elapsed_time(end) * 1000)
                    del result
                peak[which] = max(peak[which], torch.cuda.max_memory_allocated() - allocated)
            else:
                begin = time.perf_counter_ns()
                for _ in range(samples):
                    fn()
                wall[which].append((time.perf_counter_ns() - begin) / samples / 1000)
            rounds_wall[which].append(summary(wall[which][offset:]))
    return {name: {"wall": summary(wall[i]),
                   "round_wall": rounds_wall[i],
                   **({"cuda_stream": summary(stream[i]), "peak_extra_cuda_bytes": peak[i]} if cuda else {})}
            for i, name in enumerate(("baseline", "optimized"))}


class Ready(Exception):
    pass


def stop_at_forward(step):
    raise Ready(step)


def decode_prefix(engine, seqs, slots):
    try:
        engine.decode(seqs, None, slots)
    except Ready as ready:
        return ready.args[0]
    raise AssertionError("decode did not reach _forward")


def cpu_ops(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
    torch.cuda.synchronize()
    return {event.key: event.count for event in prof.key_averages()
            if event.key in ("aten::copy_", "aten::cat", "aten::argmax", "aten::multinomial", "aten::_softmax")}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmup", type=int, default=40)
    args = p.parse_args()
    assert min(args.rounds, args.samples, args.warmup) > 0
    assert torch.cuda.is_available(), "this probe measures the target GPU"
    spec = importlib.util.spec_from_file_location("baseline_adapter", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    classes = (baseline.Glm53Engine, Glm53Engine)
    facts, caches = SimpleNamespace(spec_k=5, vocab=154880), SimpleNamespace(device="cuda", draft_ring=lambda slot: None)
    report = {"scope": "eager adapter components only; no model weights or collectives",
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "device": torch.cuda.get_device_name(), "platform": platform.platform(),
              "baseline_sha256": digest(args.baseline),
              "optimized_sha256": digest(Path(__file__).resolve().parents[1] / "engine/profiles/glm53/adapter.py"),
              "protocol": {"rounds": args.rounds, "samples_per_round": args.samples,
                           "warmup": args.warmup, "order": "alternating baseline/optimized then optimized/baseline",
                           "synchronization": "end event per CUDA sample", "host_samples_per_round": 2000},
              "greedy_sampling": [], "decode_input": [], "generation_count": []}
    settings = dict(rounds=args.rounds, samples=args.samples, warmup=args.warmup)
    for rows in (1, 6, 24):
        engines = [cls(None, caches, facts, decodable=facts.vocab) for cls in classes]
        logits = torch.randn(rows, facts.vocab, dtype=torch.bfloat16, device="cuda",
                             generator=torch.Generator(device="cuda").manual_seed(17))
        temps = [0.] * rows
        fns = [lambda engine=engine: engine._sample(logits, temps) for engine in engines]
        assert torch.equal(fns[0](), fns[1]())
        item = {"rows": rows, "vocab": facts.vocab, "dtype": "bfloat16", "temperature": 0, "top_p": 1,
                "tokens_equal": True, **paired(fns, **settings),
                "cpu_ops": {name: cpu_ops(fn) for name, fn in zip(("baseline", "optimized"), fns)}}
        report["greedy_sampling"].append(item)
        print("greedy", rows, json.dumps(item), flush=True)

    class Draft:
        k = 5
        aux_layers = ()
        def propose(self, anchor, position, ring):
            return list(range(anchor + 1, anchor + 6))

    for batch in (1, 4):
        for k in (0, 5):
            engines = [cls(None, caches, facts, drafter=Draft() if k else None) for cls in classes]
            seqs, slots = list(range(batch)), list(range(1, batch + 1))
            for engine in engines:
                engine._forward = stop_at_forward
                for seq in seqs:
                    engine.add(seq, [10 + seq])
                    engine.ctx[seq] = 100 + seq
            fns = [lambda engine=engine: decode_prefix(engine, seqs, slots) for engine in engines]
            a, b = fns[0](), fns[1]()
            assert a.segments == b.segments and torch.equal(a.ids, b.ids)
            item = {"batch": batch, "drafts": k, "tokens_and_segments_equal": True,
                    **paired(fns, **settings),
                    "cpu_ops": {name: cpu_ops(fn) for name, fn in zip(("baseline", "optimized"), fns)}}
            report["decode_input"].append(item)
            print("decode_input", batch, k, json.dumps(item), flush=True)

    for count in (128, 8192, 65536):
        engines = [cls(None, caches, facts) for cls in classes]
        for engine in engines:
            engine.add(0, [1] * 16)
            engine.tokens[0].extend(range(count))
        fns = [lambda: len(engines[0].generated(0)), lambda: engines[1]._generated_count(0)]
        assert fns[0]() == fns[1]() == count
        item = {"generated_tokens": count, **paired(fns, rounds=args.rounds, samples=2000, warmup=40, cuda=False)}
        report["generation_count"].append(item)
        print("generation_count", count, json.dumps(item), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
