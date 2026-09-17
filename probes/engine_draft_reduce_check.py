"""Exact packet arithmetic: --gpu uses local rank buffers; --tp4 uses an owned fleet."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--gpu', action='store_true')
    mode.add_argument('--tp4', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from engine.kernels.draft_conv import tap_mix, tap_add_norm
    from engine.kernels.draft_reduce import packet_tap_mix, packet_tap_add_norm
    comm = None
    if args.tp4:
        from engine.base.comm import Comm
        comm = Comm.init()
        if comm.world_size != 4:
            raise RuntimeError('--tp4 requires four participating ranks')
        comm.prepare_oneshot()
    rank = comm.rank if comm else 0
    path = args.output.with_name(args.output.stem + f'.rank{rank}.json') if comm else args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='RUNNING', device=torch.cuda.get_device_name(), torch=torch.__version__,
                  cuda=torch.version.cuda, rank=rank, transport_tested=bool(comm), cells=[],
                  scope='exact consumer arithmetic and graph replay; no throughput measurement',
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in
                                 ('engine/kernels/draft_reduce.py', 'engine/profiles/glm53/drafter.py',
                                  'probes/engine_draft_reduce_check.py')})
    def save():
        path.write_text(json.dumps(report, indent=2) + '\n')
    def equal(actual, expected):
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert torch.equal(a.view(torch.int16), b.view(torch.int16))
    graph = None
    try:
        for rows, group, taps in ((8, 256, 2), (16, 256, 2), (24, 256, 2), (32, 256, 2),
                                  (8, 16, 2), (16, 64, 4)):
            torch.manual_seed(1700 + rows)
            buffers = [torch.randn(rows, 4096, device='cuda', dtype=torch.bfloat16) for _ in range(4)]
            other = [torch.empty_like(x) for x in buffers]
            descriptor = torch.tensor([x.data_ptr() for x in buffers], device='cuda', dtype=torch.int64)
            delta = torch.randn(rows, 2, taps, 4096 // group, device='cuda', dtype=torch.bfloat16)[:, 1]
            base = torch.randn(taps, 4096, device='cuda', dtype=torch.bfloat16)
            residual = torch.randn(rows, 4104, device='cuda', dtype=torch.bfloat16)[:, 4:4100]
            weight = torch.randn(4096, device='cuda', dtype=torch.bfloat16)
            def mixed(x, desc):
                return packet_tap_mix(x, desc, delta, base, group, 8)
            def normalized(x, desc):
                return packet_tap_add_norm(x, desc, delta, base, residual, weight, 1e-6, group, 8)
            def candidate():
                if comm:
                    # One consumer per exchange, exactly as each serving boundary.
                    a = comm.transport.exchange(buffers[rank]).consume(mixed)
                    b = comm.transport.exchange(buffers[rank]).consume(normalized)
                    return (a, *b)
                return (mixed(buffers[0], descriptor), *normalized(buffers[0], descriptor))
            def reference(values):
                if comm:
                    total = comm.all_reduce(buffers[rank])
                else:
                    total = ((values[0].float() + values[1].float()) + values[2].float()) + values[3].float()
                    total = total.bfloat16()
                return (tap_mix(total, delta, base, group, 8),
                        *tap_add_norm(total, delta, base, residual, weight, 1e-6, group, 8))
            for scale in (0., .01, 1., 64.):
                for x in buffers:
                    x.normal_().mul_(scale)
                expected = reference(buffers)
                equal(candidate(), expected)
            for x, value in zip(buffers, (16777216., -16777216., 1., 1.)):
                x.fill_(value)
            equal(candidate(), reference(buffers))
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                candidate()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = candidate()
            for i in range(4):
                values = buffers if comm or i % 2 == 0 else other
                for x in values:
                    x.normal_().mul_(i + 1)
                if not comm:
                    descriptor.copy_(torch.tensor([x.data_ptr() for x in values], device='cuda'))
                expected = reference(values)
                graph.replay()
                equal(actual, expected)
            graph.reset()
            graph = None
            cell = dict(rows=rows, group=group, taps=taps, bit_exact=True, graph_replays=4,
                        descriptor_retargeted=not bool(comm))
            report['cells'].append(cell); save(); print(json.dumps(cell), flush=True)
        if comm:
            comm.transport.assert_consumed()
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'], report['error'] = 'FAIL', repr(exc)
        raise
    finally:
        save()
        try:
            if graph is not None:
                graph.reset()
        finally:
            if comm:
                comm.close()


if __name__ == '__main__':
    main()
