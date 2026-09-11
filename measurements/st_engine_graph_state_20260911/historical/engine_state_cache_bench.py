"""Paired old/new graph state transfers and real GLM graph steps.

Export the baseline decode_graphs.py from git. Both variants share weights,
inputs, streams and communicator; only GraphCaches changes. Resets are queued
outside the timed spans. Each sample batch is queued before synchronization
to reduce host entry jitter between ranks. Repeat in independent processes.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from engine.base.comm import Comm
from engine.profiles.glm53.decode_graphs import DeviceStep, GraphCaches
from engine.profiles.glm53.net import Step
from engine_graph_profile import IsolatedRank, build_slice, capture, load_baseline


def paired(functions, reset, comm, rounds, samples):
    results = [[], []]
    resources = [[], []]
    for round_id in range(rounds):
        row = [None, None]
        for i in ((0, 1) if round_id % 2 == 0 else (1, 0)):
            comm.barrier()
            for _ in range(5):
                reset()
                functions[i]()
            torch.cuda.synchronize()
            stat = Path('/sys/fs/cgroup/cpu.stat')
            resource_before = stat.read_text() if stat.exists() else None
            wall_start = time.perf_counter()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(samples)]
            for start, end in events:
                reset()
                start.record()
                functions[i]()
                end.record()
            events[-1][1].synchronize()
            row[i] = [a.elapsed_time(b)*1000 for a,b in events]
            results[i].append(row[i])
            resources[i].append(dict(cpu_before=resource_before,
                                     cpu_after=stat.read_text() if stat.exists() else None,
                                     wall_seconds=time.perf_counter()-wall_start))
        print('round', round_id, [statistics.median(v) for v in row], flush=True)
    return {name: dict(median_us=statistics.median(sum(values, [])),
                       rounds_median_us=[statistics.median(v) for v in values],
                       samples_us=values, resources=resource)
            for name, values, resource in zip(('baseline', 'optimized'), results, resources)}


@torch.inference_mode()
def run(args, comm):
    baseline = load_baseline(args.baseline)
    F, net, caches, arena = build_slice(args, comm)
    report = dict(scope='paired real-weight layers 0 and 3, embedding and head; not full-model ITL',
                  rank=comm.rank, distributed=args.distributed, results=[],
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                  source_sha256={str(f.relative_to(Path(__file__).resolve().parents[1])):
                                 hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in (Path(__file__).resolve().parents[1] / 'engine').rglob('*.py')})
    for ctx in args.contexts:
        for tokens in (1, 6):
            capacity = max(4096, 1 << (ctx + tokens - 1).bit_length())
            inputs, graphs, outputs = [], [], []
            for cls in (baseline.GraphCaches, GraphCaches):
                seqs = torch.zeros(1, device='cuda', dtype=torch.int64)
                slots = torch.ones(1, device='cuda', dtype=torch.int64)
                step = DeviceStep(torch.zeros(tokens, device='cuda', dtype=torch.int64),
                                  torch.zeros(1, device='cuda', dtype=torch.int64), tokens)
                scratch = cls(caches, seqs, slots, capacity,
                              *(() if cls is baseline.GraphCaches else (step,)))
                def forward(scratch=scratch, step=step):
                    scratch.gather()
                    h = net.forward(step, scratch)
                    logits = net.head(h)
                    scratch.commit()
                    scratch.fields.clear()
                    del scratch.block_table
                    return h, logits
                graph, out = capture(forward)
                inputs.append(step); graphs.append(graph); outputs.append(out)
            caches.reset()
            slot = caches.slots.take(0)
            caches.pool.reserve(0, ctx + tokens)
            values = torch.randint(0, 30000, (ctx + tokens,), device='cuda')
            for start in range(0, ctx, 256):
                pre = Step.prefill(values[start:min(ctx, start+256)], start, 0, slot)
                caches.prepare(pre); net.forward(pre, caches)
            step = Step.decode([(values[ctx:], ctx, 0, slot)])
            caches.prepare(step)
            for inp in inputs:
                inp.ids.copy_(step.ids); inp.contexts.fill_(ctx)
            before = caches.state.clone(), caches.paged.clone()
            def reset():
                caches.state.copy_(before[0]); caches.paged.copy_(before[1])
            snapshots = []
            for graph, out in zip(graphs, outputs):
                reset(); graph.replay()
                snapshots.append(tuple(x.clone() for x in (*out, caches.state, caches.paged)))
            torch.cuda.synchronize()
            exact = [torch.equal(a, b) for a, b in zip(*snapshots)]
            assert all(exact), (ctx, tokens, exact)
            del snapshots
            result = dict(context=ctx, tokens=tokens, capacity=capacity,
                          hidden_logits_state_paged_exact=exact,
                          timing=paired([g.replay for g in graphs], reset, comm, args.rounds, args.samples))
            report['results'].append(result)
            print('case', ctx, tokens, {k:v['median_us'] for k,v in result['timing'].items()}, flush=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            for graph in graphs:
                graph.reset()
            del graphs, outputs, inputs, before, out, scratch, graph
            caches.slots.give(slot); caches.pool.release(0)
            gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', required=True)
    parser.add_argument('--ckpt-meta', required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--contexts', type=int, nargs='+', default=[256, 32768])
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comm = Comm.init(world=4, timeout_s=600) if args.distributed else IsolatedRank()
    try:
        run(args, comm)
    finally:
        comm.close()


if __name__ == '__main__':
    main()
