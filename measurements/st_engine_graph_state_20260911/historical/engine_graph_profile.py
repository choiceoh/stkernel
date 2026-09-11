"""Bounded real-weight GLM layer-slice profiling, with optional TP4 collectives.

Uninstrumented CUDA graphs establish latency. A separate graph with external
timing events attributes the same work to disjoint stages; its overhead is
reported. This is not a full-model serving/ITL or generation-quality benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from engine.base.arena import Arena
from engine.base.comm import Comm
from engine.base.params import total_bytes
from engine.profiles.glm53 import facts
from engine.profiles.glm53.caches import Glm53Caches, layout
from engine.profiles.glm53.decode_graphs import DeviceStep, GraphCaches
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.net import Glm53Net, Step
from engine.profiles.glm53.weights import rank_loader


class IsolatedRank:
    world_size = 4
    rank = 0

    def all_reduce(self, x):
        return x

    def all_gather(self, x, dim=-1):
        return x

    def barrier(self):
        pass

    def close(self):
        pass


class Stages:
    def __init__(self):
        self.active = False
        self.entries, self.stack = [], []

    @contextmanager
    def stage(self, name):
        if not self.active:
            yield
            return
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        entry = dict(name=name, start=start, end=end, children=[])
        if self.stack:
            self.stack[-1]['children'].append(entry)
        self.entries.append(entry)
        self.stack.append(entry)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.stack.pop()

    def wrap(self, name, fn):
        def call(*args, **kwargs):
            with self.stage(name):
                return fn(*args, **kwargs)
        return call

    def read(self):
        values = defaultdict(float)
        for e in self.entries:
            elapsed = e['start'].elapsed_time(e['end']) * 1000
            children = sum(c['start'].elapsed_time(c['end']) * 1000 for c in e['children'])
            values[e['name']] += elapsed - children
        return dict(values)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def load_baseline(path):
    spec = importlib.util.spec_from_file_location('state_cache_baseline', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, samples, stages=None, reset=lambda: None):
    for _ in range(5):
        reset()
        fn()
    torch.cuda.synchronize()
    times, breakdown = [], defaultdict(list)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(samples):
        reset()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
        if stages:
            for name, us in stages.read().items():
                breakdown[name].append(us)
    return dict(median_us=statistics.median(times),
                p95_us=sorted(times)[min(len(times)-1, int(len(times)*.95))],
                samples_us=times,
                stages_us={k: statistics.median(v) for k, v in breakdown.items()})


@torch.inference_mode()
def build_slice(args, comm):
    torch.manual_seed(13)
    F = facts.load(args.ckpt_meta)
    net = Glm53Net(F, comm, served(), [0, 3])
    specs = net.specs()
    p = layout(F, net.layers)
    max_capacity = max(4096, 1 << (max(args.contexts) + 6 - 1).bit_length())
    blocks = (max_capacity + F.block - 1) // F.block + 2
    arena = Arena(total_bytes(specs) + p.nbytes(blocks, 2) + 256 * (len(specs) + 64))
    loader = rank_loader(Path(args.ranks) / f'rank{comm.rank}of4.safetensors')
    net.bind(loader.load([s.name for s in specs], arena=arena))
    caches = Glm53Caches(arena, F, net.layers, blocks, 2)
    return F, net, caches, arena


@torch.inference_mode()
def run(args, comm):
    F, net, caches, arena = build_slice(args, comm)
    cache_class = load_baseline(args.baseline).GraphCaches if args.baseline else GraphCaches
    stages = Stages()
    for attr in ('embed', 'head', '_hc_pre', '_kda', '_dsa', '_indexer', '_dense', '_moe'):
        setattr(net, attr, stages.wrap(attr, getattr(net, attr)))
    for attr in ('all_reduce', 'all_gather'):
        setattr(comm, attr, stages.wrap(attr, getattr(comm, attr)))
    net.lanes = replace(net.lanes, mhc_post=stages.wrap('mhc_post', net.lanes.mhc_post))
    report = dict(scope='real-weight layers 0 and 3 plus embedding/head; not full-model ITL',
                  distributed=args.distributed, rank=comm.rank, torch=torch.__version__,
                  device=torch.cuda.get_device_name(), arena_bytes=arena.nbytes,
                  state_slot_bytes=caches.layout.slot_bytes, results=[],
                  source_sha256={str(f.relative_to(Path(__file__).resolve().parents[1])):
                                 hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in (Path(__file__).resolve().parents[1] / 'engine').rglob('*.py')})
    if args.baseline:
        report['baseline_sha256'] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    print('loaded', comm.rank, arena.nbytes, flush=True)
    for ctx in args.contexts:
        for tokens in (() if args.prefill_only else (1, 6)):
            capacity = 4096
            while capacity < ctx + tokens:
                capacity *= 2
            seqs = torch.tensor([0], device='cuda')
            slots = torch.tensor([1], device='cuda')
            step = DeviceStep(torch.zeros(tokens, dtype=torch.int64, device='cuda'),
                              torch.zeros(1, dtype=torch.int64, device='cuda'), tokens)
            scratch = cache_class(caches, seqs, slots, capacity, *(() if args.baseline else (step,)))

            def forward():
                with stages.stage('step_other'):
                    with stages.stage('state_gather'):
                        scratch.gather()
                    h = net.forward(step, scratch)
                    logits = net.head(h)
                    with stages.stage('state_commit'):
                        scratch.commit()
                    scratch.fields.clear()
                    del scratch.block_table
                    return h, logits

            plain, _ = capture(forward)
            # Warm up outside event recording, then record one set of events.
            stages.active = True
            instrumented = torch.cuda.CUDAGraph()
            with torch.cuda.graph(instrumented):
                outputs = forward()
            stages.active = False
            caches.reset()
            slot = caches.slots.take(0)
            assert slot == 1
            caches.pool.reserve(0, ctx + tokens)
            values = torch.randint(0, 30000, (ctx + tokens,), device='cuda')
            for start in range(0, ctx, 256):
                prefix = Step.prefill(values[start:min(ctx, start+256)], start, 0, slot)
                caches.prepare(prefix)
                net.forward(prefix, caches)
            actual = Step.decode([(values[ctx:], ctx, 0, slot)])
            caches.prepare(actual)
            step.ids.copy_(actual.ids)
            step.contexts.fill_(ctx)
            before = caches.state.clone(), caches.paged.clone()
            def reset():
                caches.state.copy_(before[0])
                caches.paged.copy_(before[1])
            comm.barrier()
            baseline = measure(plain.replay, args.samples, reset=reset)
            comm.barrier()
            attributed = measure(instrumented.replay, args.samples, stages, reset)
            result = dict(context=ctx, tokens=tokens, capacity=capacity,
                          graph=baseline, instrumented=attributed,
                          instrumentation_ratio=attributed['median_us']/baseline['median_us'])
            report['results'].append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            plain.reset(); instrumented.reset()
            del plain, instrumented, outputs, before
            stages.entries.clear()
            caches.slots.give(slot); caches.pool.release(0)
    if args.prefill_only:
        report['scope'] = ('real-weight layers 0 and 3, eager 256-token prefill and last-token head; '
                           'event spans include host dispatch gaps; not full-model TTFT')
        for ctx in args.contexts:
            caches.reset()
            slot = caches.slots.take(0)
            caches.pool.reserve(0, ctx + 256)
            values = torch.randint(0, 30000, (ctx + 256,), device='cuda')
            for start in range(0, ctx, 256):
                prefix = Step.prefill(values[start:min(ctx, start+256)], start, 0, slot)
                caches.prepare(prefix); net.forward(prefix, caches)
            step = Step.prefill(values[ctx:], ctx, 0, slot)
            caches.prepare(step)
            before = caches.state.clone(), caches.paged.clone()
            def reset():
                caches.state.copy_(before[0]); caches.paged.copy_(before[1])
            def forward():
                stages.entries.clear()
                with stages.stage('step_other'):
                    return net.head(net.forward(step, caches)[-1:])
            comm.barrier()
            plain = measure(forward, args.samples, reset=reset)
            stages.active = True
            comm.barrier()
            attributed = measure(forward, args.samples, stages, reset)
            stages.active = False
            result = dict(context=ctx, tokens=256, eager=plain, instrumented_eager=attributed,
                          instrumentation_ratio=attributed['median_us']/plain['median_us'])
            report['results'].append(result)
            print(json.dumps(result), flush=True)
            args.output.write_text(json.dumps(report, indent=2)+'\n')
            stages.entries.clear()
            caches.slots.give(slot); caches.pool.release(0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', required=True)
    parser.add_argument('--ckpt-meta', required=True)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--contexts', nargs='+', type=int, default=[256, 32768])
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--prefill-only', action='store_true')
    parser.add_argument('--baseline', type=Path, help='exported pre-change decode_graphs.py for baseline attribution')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    comm = Comm.init(world=4, timeout_s=600) if args.distributed else IsolatedRank()
    try:
        run(args, comm)
    finally:
        comm.close()


if __name__ == '__main__':
    main()
