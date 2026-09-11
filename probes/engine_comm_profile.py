"""Bounded TP4 NCCL latency and rank-wait diagnostics, without model weights.

Chain timings include PyTorch/NCCL stream dependencies and graph scheduling.
They are not wire latency, full-model latency, or exclusive-fleet qualification.
An injected GPU delay illustrates peer waiting without comparing clocks across
devices. Launch exactly four processes with the production RoCE environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from engine.base.comm import Comm


def stats(values):
    return dict(median_us=statistics.median(values),
                p95_us=sorted(values)[min(len(values)-1, int(len(values)*.95))], samples_us=values)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def measure(graph, comm, samples, chain, spans=()):
    comm.barrier()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    comm.barrier()
    def cpu_stats():
        path = Path('/sys/fs/cgroup/cpu.stat')
        return dict(line.split() for line in path.read_text().splitlines()) if path.exists() else {}
    before = cpu_stats()
    elapsed, local_spans = [], [[] for _ in spans]
    for _ in range(samples):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end)*1000/chain)
        for values, pairs in zip(local_spans, spans):
            values.append(sum(a.elapsed_time(b)*1000 for a, b in pairs)/len(pairs))
    comm.barrier()
    after = cpu_stats()
    return dict(per_operation=stats(elapsed), spans=[stats(s) for s in local_spans],
                cpu_delta={k: int(after[k])-int(before[k]) for k in before})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--chain', type=int, default=32)
    args = parser.parse_args()
    if not 5 <= args.samples <= 50 or not 8 <= args.chain <= 64:
        parser.error('bounded diagnostic requires samples 5..50 and chain 8..64')
    torch.cuda.set_device(0)
    comm = Comm.init(world=4, timeout_s=45)
    root = Path(__file__).resolve().parents[1]
    report = dict(scope=__doc__, rank=comm.rank, host=socket.gethostname(),
                  torch=torch.__version__, nccl=torch.cuda.nccl.version(),
                  device=torch.cuda.get_device_name(), chain=args.chain,
                  env={k: v for k, v in os.environ.items() if k.startswith(('NCCL_', 'TORCH_NCCL_'))},
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in (
                      'engine/base/comm.py', 'engine/profiles/glm53/net.py',
                      'launchers/start-st-glm53.sh', 'probes/engine_comm_profile.py')},
                  results=[], injected_wait=[])
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    try:
        cases = [('max', n, torch.int64) for n in (1, 6, 24)]
        cases += [('sum', n*4096, torch.bfloat16) for n in (1, 6, 24, 256, 1024)]
        for operation, elements, dtype in cases:
            x = torch.full((elements,), comm.rank+1, dtype=dtype, device='cuda')
            reduce = comm.all_reduce_max if operation == 'max' else comm.all_reduce
            reduce(x)
            expected = 4 if operation == 'max' else 10
            assert torch.equal(x, torch.full_like(x, expected))
            # Zero is stable across arbitrarily many in-place sum replays.
            # Nonzero rank-dependent values were checked independently above.
            x.zero_()
            def chain():
                for _ in range(args.chain):
                    reduce(x)
            graph, _ = capture(chain)
            timing = measure(graph, comm, args.samples, args.chain)
            assert torch.equal(x, torch.zeros_like(x))
            result = dict(operation=operation, elements=elements, bytes=elements*x.element_size(),
                          dtype=str(dtype), exact=True, **timing)
            report['results'].append(result)
            print(operation, result['bytes'], timing['per_operation']['median_us'], flush=True)
            save()
            graph.reset()
            del graph, x

        # Each reduction closes the preceding iteration. A one-rank delay
        # before the next reduction makes peer wait visible in local event
        # durations; CUDA event timestamps are never compared across GPUs.
        for delayed_rank in (-1, 3):
            x = torch.zeros(4096, dtype=torch.bfloat16, device='cuda')
            comm_pairs, delay_pairs = [], []
            collecting = False
            def event():
                return torch.cuda.Event(enable_timing=True, external=True)
            def chain():
                comm.all_reduce(x)  # arrival alignment, outside attributed spans
                for _ in range(args.chain):
                    a, b, c = event(), event(), event()
                    a.record()
                    if comm.rank == delayed_rank:
                        torch.cuda._sleep(200_000)
                    b.record()
                    comm.all_reduce(x)
                    c.record()
                    if collecting:
                        delay_pairs.append((a, b))
                        comm_pairs.append((b, c))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    chain()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            collecting = True
            with torch.cuda.graph(graph):
                chain()
            timing = measure(graph, comm, args.samples, args.chain, (delay_pairs, comm_pairs))
            assert torch.equal(x, torch.zeros_like(x))
            report['injected_wait'].append(dict(delayed_rank=delayed_rank, sleep_cycles=200_000,
                                                exact=True, **timing))
            print('injected_wait', delayed_rank, [s['median_us'] for s in timing['spans']], flush=True)
            save()
            graph.reset()
        report['passed'] = True
        save()
    finally:
        comm.close()


if __name__ == '__main__':
    main()
