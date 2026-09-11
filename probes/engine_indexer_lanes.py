"""Qualify the GLM indexer and time its quantization/expansion helpers.

Run through run_engine_probe.sh in the vLLM-free ST image. --checkpoint supplies the
config.json directory; --rank-file is an existing aligned rank file. Only
layer 3's replicated indexer weights are loaded. Real-weight checks use seeded
synthetic activations; this is not a complete-model generation/quality gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.arena import Arena
from engine.base.comm import Comm, LocalTP
from engine.base.params import bind, total_bytes
from engine.profiles.glm53 import facts, lanes
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.net import Glm53Net, Step
from engine.profiles.glm53.weights import rank_loader
from engine_decode_overhead import paired


def legacy_lanes(table):
    """Supply historical net.py contracts only to an explicitly loaded baseline."""
    from engine.kernels.kpool import expand_pools_and_append_tail
    from engine.kernels.indexer import indexer_slots
    return SimpleNamespace(**vars(table), expand_pools=expand_pools_and_append_tail, indexer_slots=indexer_slots)


def kernel_contracts(ref, fused):
    from engine.modules.sparse_indexer import select_with_tail
    from engine.kernels.kpool import expand_pools_and_append_tail
    g = torch.Generator(device="cuda").manual_seed(71)
    quant_cases = []
    for count in (1, 31, 32, 33, 192, 768, 16384):
        for magnitude in (1e-6, 1., 100.):
            x = (torch.randn(count, 128, device="cuda", generator=g) * magnitude).to(torch.bfloat16)
            a, sa = ref.indexer_quant(x)
            b, sb = fused.indexer_quant(x)
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), (count, magnitude, "fp8 bytes")
            assert torch.equal(sa, sb), (count, magnitude, "scales")
            quant_cases.append({"rows": count, "magnitude": magnitude})
    pattern = torch.stack([torch.zeros(128), torch.ones(128), -torch.ones(128),
                           torch.arange(128) % 2 * 2 - 1, torch.eye(128)[0],
                           torch.linspace(-448, 448, 128)]).to(device="cuda", dtype=torch.bfloat16)
    a, sa = ref.indexer_quant(pattern)
    b, sb = fused.indexer_quant(pattern)
    assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)) and torch.equal(sa, sb)
    lengths = torch.tensor([0, 1, 3, 4, 5, 7, 8, 9, 2047, 2048, 2049, 65535], device="cuda", dtype=torch.int32)
    for groups in (1, 2, 512):
        ids = torch.randint(-1, 17000, (len(lengths), groups), device="cuda", dtype=torch.int32, generator=g)
        ids[:, 0] = 0
        if groups > 1:
            ids[:, 1] = lengths // 4                # incomplete/future pool: must be excluded
        assert torch.equal(select_with_tail(ids, lengths, 4), expand_pools_and_append_tail(ids, lengths, 4))
    return {"fp8_bytes_and_scales_exact": True, "random_quant_cases": quant_cases,
            "pattern_rows": 6, "expansion_exact": True, "expansion_lengths": lengths.tolist(),
            "expansion_group_widths": [1, 2, 512]}


def threaded_dispatch(fused):
    rows = torch.arange(32 * 128, device="cuda").reshape(32, 128).to(torch.bfloat16)
    pools = torch.tensor([[0, -1], [1, 0]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([3, 9], device="cuda", dtype=torch.int32)
    def slots(table):
        out = torch.empty((2, 11), device="cuda", dtype=torch.int32)
        count = torch.empty(2, device="cuda", dtype=torch.int32)
        table.pool_slots(pools, lengths, 4, None, 1, 1, 0, out, count)
        return out, count
    expected = fused.indexer_quant(rows), slots(fused)
    tp = LocalTP(4)
    bound = lanes.served(tp=tp)
    outputs = tp.run(lambda comm: (bound.indexer_quant(rows), slots(bound)))
    for (q, scale), expanded in outputs:
        assert torch.equal(q.view(torch.uint8), expected[0][0].view(torch.uint8))
        assert torch.equal(scale, expected[0][1]) and all(torch.equal(a, b) for a, b in zip(expanded, expected[1]))
    return {"logical_ranks": 4, "main_thread_dispatch_exact": True}


def real_indexer(checkpoint, rank_file, ref, fused, baseline=None):
    loader = rank_loader(rank_file)
    F = facts.load(checkpoint)
    assert F.is_dsa(3)
    net = Glm53Net(F, Comm(4, 1), fused, layers=[3])
    specs = [s for s in net.specs() if s.name.startswith("L3.idx.")]
    blocks, max_seqs = 160, 3
    cache_bytes = layout(F, [3]).nbytes(blocks, max_seqs)
    arena = Arena(total_bytes(specs) + 256 * (len(specs) + 10) + 2 * cache_bytes)
    net.p = bind(specs, loader.load([s.name for s in specs], arena=arena, max_run=32 << 20))
    caches = [Glm53Caches(arena, F, [3], blocks, max_seqs) for _ in range(2)]
    for cache in caches:
        cache.pool.reserve(1, F.block)             # force a nonidentity physical block mapping
        cache.pool.reserve(0, 2080)
        cache.slots.take(0)
    old = replace(fused, indexer_quant=ref.indexer_quant, pool_slots=ref.pool_slots) if baseline is None else legacy_lanes(fused)
    methods = (baseline or Glm53Net._indexer, Glm53Net._indexer)
    g = torch.Generator(device="cuda").manual_seed(23)
    x = torch.randn(2080, F.hidden, device="cuda", generator=g).to(torch.bfloat16)
    qr = torch.randn(2080, F.q_lora, device="cuda", generator=g).to(torch.bfloat16)
    checked = []
    for ctx in (63, 256, 2048):
        for cache in caches:
            cache.reset()
        for phase, start, length in (("prefill", 0, ctx), ("verify", ctx, 6), ("rollback", ctx + 2, 6)):
            xx, qq = x[start:start + length].clone(), qr[start:start + length].clone()
            if phase == "verify":
                xx[2:] += 1
                qq[2:] -= 1
            step = Step.prefill(torch.zeros(length, device="cuda", dtype=torch.int64), start, 0, 1)
            outputs = []
            for table, cache, method in zip((old, fused), caches, methods):
                net.lanes = table
                cache.prepare(step)
                outputs.append(method(net, 3, xx, qq, step, cache))
            assert all(torch.equal(a, b) for a, b in zip(*outputs)), (ctx, phase, "selected slots/counts")
            assert torch.equal(caches[0].paged, caches[1].paged), (ctx, phase, "pool KV bytes/scales")
            assert torch.equal(caches[0].state, caches[1].state), (ctx, phase, "tail ring bytes")
            checked.append({"context": ctx, "phase": phase, "tokens": length})
    measurements = []
    for phase, start, length in (("decode", 2048, 6), ("prefill", 0, 256)):
        step = Step.prefill(torch.zeros(length, device="cuda", dtype=torch.int64), start, 0, 1)
        xx, qq = x[start:start + length], qr[start:start + length]
        for cache in caches:
            cache.prepare(step)
        def run(table, cache, method):
            net.lanes = table
            return method(net, 3, xx, qq, step, cache)
        fns = [lambda table=t, cache=c, method=m: run(table, cache, method)
               for t, c, m in zip((old, fused), caches, methods)]
        measurements.append({"phase": phase, "context": start, "tokens": length,
                             **paired(fns, rounds=5, samples=50, warmup=10)})
        assert all(torch.equal(a, b) for a, b in zip(fns[0](), fns[1]()))
        assert torch.equal(caches[0].paged, caches[1].paged) and torch.equal(caches[0].state, caches[1].state)
    return {"layer": 3, "weights": [s.name for s in specs], "weight_bytes": total_bytes(specs),
            "rank_file": str(rank_file), "config_sha256": hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest(),
            "synthetic_activations": True, "selected_slots_counts_and_cache_bytes_exact": True, "checks": checked,
            "measurement_protocol": {"rounds": 5, "samples_per_round": 50, "warmup": 10}, "measurements": measurements}


def operators(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
    torch.cuda.synchronize()
    gpu = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert gpu, "CUDA profiler captured no device events"
    copies = [e for e in gpu if e.name.startswith(("Memcpy", "Memset"))]
    kernels = [e for e in gpu if not e.name.startswith(("Memcpy", "Memset"))]
    return {"cuda_kernel_count": len(kernels), "cuda_copy_count": len(copies),
            "cuda_kernels": dict(Counter(e.name for e in kernels)),
            "aten_ops": {e.key: e.count for e in prof.key_averages() if e.key.startswith("aten::")}}


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--rank-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert importlib.util.find_spec("vllm") is None, "qualify in the standalone ST runtime"
    ref, fused = lanes.reference(), lanes.served()
    import engine.kernels.kpool as module
    report = {"scope": "fused indexer helpers and real-weight indexer checks; no whole-model quality/ITL claim",
              "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "served_module": module.__file__, "served_module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              "vllm_installed": False,
              "protocol": {"rounds": 5, "samples_per_round": 100, "warmup": 40, "order": "alternating AB/BA"}}
    report["contracts"] = kernel_contracts(ref, fused)
    print("kernel contracts:", json.dumps(report["contracts"]), flush=True)
    report["local_tp"] = threaded_dispatch(fused)
    print("LocalTP:", json.dumps(report["local_tp"]), flush=True)
    report["real_indexer"] = real_indexer(args.checkpoint, args.rank_file, ref, fused)
    print("real indexer:", json.dumps(report["real_indexer"]), flush=True)
    report["measurements"] = []
    to_profile = []
    g = torch.Generator(device="cuda").manual_seed(117)
    for tokens in (1, 6, 24, 512):
        rows = torch.randn(tokens * 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
        pools = torch.randint(-1, 512, (tokens, 512), device="cuda", dtype=torch.int32, generator=g)
        lengths = torch.full((tokens,), 2047, device="cuda", dtype=torch.int32)
        from engine.modules.sparse_indexer import select_with_tail
        from engine.kernels.kpool import expand_pools_and_append_tail
        for name, fns in (("quant", [lambda table=t, rows=rows: table.indexer_quant(rows) for t in (ref, fused)]),
                          ("expand", [lambda fn=fn, pools=pools, lengths=lengths: fn(pools, lengths, 4)
                                      for fn in (select_with_tail, expand_pools_and_append_tail)])):
            item = {"component": name, "tokens": tokens,
                    **paired(fns, rounds=5, samples=100, warmup=40)}
            report["measurements"].append(item)
            to_profile.append((item, fns))
            print("measurement:", json.dumps(item), flush=True)
    # CUDA profiler initialization can change later launch overhead. Collect
    # traces only after every latency measurement, using the same input tensors.
    for item, fns in to_profile:
        item["operators"] = {key: operators(fn) for key, fn in zip(("baseline", "optimized"), fns)}
        print("kernel counts:", item["component"], item["tokens"],
              {key: value["cuda_kernel_count"] for key, value in item["operators"].items()}, flush=True)
    assert not any(n == "vllm" or n.startswith("vllm.") for n in sys.modules)
    report["vllm_loaded"] = False
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
