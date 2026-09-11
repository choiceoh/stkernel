"""Qualify compressed-pool slot finalization on GB10 against the previous path.

Both eager and repeated CUDA-graph samples compare the same input/output
buffers. Real L3 measurements load the exact baseline net.py exported from git.
No full-model latency or four-node quality claim is made by this probe.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from engine.kernels.indexer import pool_slots
from engine.kernels.indexer import indexer_slots as old_slots
from engine.kernels.kpool import expand_pools_and_append_tail
from engine.modules.sparse_indexer import pool_slots as reference
from engine_decode_overhead import paired
from engine_indexer_lanes import operators, real_indexer
from engine_indexer_slots import load_baseline, local_tp


def previous(ids, lengths, pool, *args):
    old_slots(expand_pools_and_append_tail(ids, lengths, pool), *args)


def capture(fn, repetitions):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(repetitions):
            fn()
    return graph


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-net', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--rank-file', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert torch.cuda.get_device_capability() == (12, 1)
    assert importlib.util.find_spec('vllm') is None
    props = torch.cuda.get_device_properties(0)
    report = dict(scope='pool-slot kernel and optional real-weight L3 indexer; not full-model ITL',
                  torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                  device=props.name, sm_count=props.multi_processor_count,
                  max_threads_per_sm=props.max_threads_per_multi_processor,
                  shared_memory_per_sm=props.shared_memory_per_multiprocessor,
                  protocol=dict(rounds=5, eager_samples=100, graph_samples=50,
                                graph_repetitions=100, order='alternating AB/BA'), measurements=[])
    if args.baseline_net:
        assert args.checkpoint and args.rank_file
        from engine.profiles.glm53 import lanes
        ref, fused = lanes.reference(), lanes.served()
        report['local_tp'] = local_tp(fused)
        report['baseline_net_sha256'] = hashlib.sha256(args.baseline_net.read_bytes()).hexdigest()
        report['real_indexer'] = real_indexer(args.checkpoint, args.rank_file, ref, fused, load_baseline(args.baseline_net))
        print('real indexer:', json.dumps(report['real_indexer']), flush=True)
    g = torch.Generator(device='cuda').manual_seed(442)
    table = torch.randperm(4096, device='cuda', generator=g).int()
    profiles = []
    for rows in (1, 6, 24, 256, 512):
        # A long-context selection with nontrivial token order, all tail phases,
        # and invalid/future pools on the shorter rows. No timing includes RNG.
        ids = torch.rand((rows, 4096), device='cuda', generator=g).topk(512, dim=1).indices.int()
        lengths = 8192 + torch.arange(rows, device='cuda', dtype=torch.int32) % 4
        outputs = [(torch.empty((rows, 2051), device='cuda', dtype=torch.int32),
                    torch.empty(rows, device='cuda', dtype=torch.int32)) for _ in range(3)]
        functions = [lambda fn=fn, ids=ids, lengths=lengths, out=out, counts=counts:
                     fn(ids, lengths, 4, table, 64, 2112, 512, out, counts)
                     for fn, (out, counts) in zip((previous, pool_slots, reference), outputs)]
        for fn in functions:
            fn()
        assert all(torch.equal(a, b) for output in outputs[1:] for a, b in zip(outputs[0], output))
        fns = functions[:2]
        item = dict(rows=rows, groups=512, width=2051, slots_and_counts_exact=True,
                    eager=paired(fns, rounds=5, samples=100, warmup=40))
        graphs = [capture(fn, 100) for fn in fns]
        timing = paired([graph.replay for graph in graphs], rounds=5, samples=50, warmup=10)
        item['graph_100_calls'] = timing
        item['graph_device_us_per_call'] = {name: timing[name]['cuda_stream']['median_us'] / 100
                                             for name in ('baseline', 'optimized')}
        report['measurements'].append(item)
        profiles.append((item, fns))
        print('measurement:', json.dumps(item), flush=True)
        del graphs
    # CUPTI can alter later launch timing. Initialize it after all timed cases.
    for item, fns in profiles:
        item['operators'] = {name: operators(fn) for name, fn in zip(('baseline', 'optimized'), fns)}
    assert not any(name == 'vllm' or name.startswith('vllm.') for name in sys.modules)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
