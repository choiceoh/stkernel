"""Explicit cuBLASLt producer-to-output comparison; never run by a model forward.

Use --gpu only with an owned idle GPU. No reservations, service control or
network calls are made here. SM120 permits --numerics-only, never timing.
Synthetic weights are kernel evidence, not acceptance or serving speed proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def shape(value):
    try:
        m, n, k = map(int, value.lower().split('x'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('shape must be MxNxK') from error
    if min(m, n, k) <= 0 or n % 128 or k % 128:
        raise argparse.ArgumentTypeError('positive shape with N and K aligned to 128 required')
    return m, n, k


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='explicit permission to execute on the owned GPU')
    parser.add_argument('--numerics-only', action='store_true', help='no algorithm timing or speed verdict')
    parser.add_argument('--shape', type=shape, action='append')
    parser.add_argument('--producer', choices=('bf16', 'packets'), default='bf16')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.gpu:
        parser.error('execution requires --gpu and an owned idle GPU; this tool does not acquire one')
    shapes = args.shape or [(8, 4096, 20480), (8, 38784, 4096), (512, 6144, 4096)]
    if args.producer == 'packets' and any(k != 4096 or m < 128 for m, _, k in shapes):
        parser.error('packet producer needs M >= 128 and K = 4096; specify matching --shape values')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense import fp8, mxfp8
    from engine.kernels.dense.cublaslt import PreparedProjection, _build, WORKSPACE_LIMIT
    from engine.kernels.deep_gemm import _initialize
    _initialize()
    from deep_gemm import fp8_gemm_nt
    device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    capability = prop.major, prop.minor
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError('numerical probe only supports the declared SM120/SM121 targets')
    if not args.numerics_only and (capability != (12, 1) or prop.multi_processor_count != 48):
        raise RuntimeError('timing is restricted to the GB10 fleet target')
    report = dict(status='RUNNING', gpu_used=True, numerics_only=args.numerics_only,
                  scope='synthetic kernel/producer evidence; no serving or acceptance verdict',
                  device=prop.name, capability=capability, sms=prop.multi_processor_count,
                  torch=torch.__version__, cuda=torch.version.cuda, cublaslt=_build().version(), cells=[])
    root = Path(__file__).resolve().parents[1]
    sources = ('engine/kernels/dense/cublaslt.cpp', 'engine/kernels/dense/cublaslt.py',
               'engine/kernels/dense/mxfp8.py', 'engine/kernels/dense/fp8.py',
               'engine/kernels/prefill_collectives/consumer.py', 'probes/engine_cublaslt_check.py')
    report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for index, (m, n, k) in enumerate(shapes):
            torch.manual_seed(1907 + index)
            weight = FP8Linear(torch.randn(n, k, device='cuda', dtype=torch.bfloat16)*.02).weight
            mx_weight = mxfp8.pack_weight_scales(weight[1], n, k)
            if args.producer == 'bf16':
                source = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                def producer(mx, out):
                    return (mxfp8 if mx else fp8).quantize(source, out=out)
            else:
                from engine.kernels.prefill_collectives import PrefillCollectives
                from engine.kernels.prefill_collectives.consumer import quantize_gather
                local = (m+3)//4
                packets = []
                for rank in range(4):
                    x = torch.randn(local, k, device='cuda', dtype=torch.bfloat16)*2**rank
                    packets.append(PrefillCollectives.pack(x, x.numel())[0])
                source = torch.cat(packets)
                def producer(mx, out):
                    return quantize_gather(source, local, real_rows=m, mx=mx, out=out)
            baseline = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            candidate = torch.empty_like(baseline)
            q0, s0 = producer(False, None)
            q1, s1 = producer(True, None)
            if not torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)):
                raise RuntimeError('producer FP8 byte mismatch')
            fp8_gemm_nt((q0, s0), weight, baseline)
            if args.numerics_only:
                native = _build()
                context = native.Context(device, *capability)
                plan = native.Plan(context, m, n, k, WORKSPACE_LIMIT)
                algorithms = plan.candidates()
                scratch = torch.empty(max((c['workspace'] for c in algorithms), default=0), device='cuda', dtype=torch.uint8)
                checked = []
                for c in algorithms:
                    plan.run(c['index'], q1, weight[0], s1, mx_weight, candidate, scratch)
                    relative = ((candidate.float()-baseline.float()).norm()/baseline.float().norm().clamp_min(1e-30)).item()
                    checked.append(dict(c, relative_l2=relative,
                                        numerics=bool(torch.allclose(candidate, baseline, rtol=.01, atol=.001))))
                record = dict(shape=(m, n, k), producer=args.producer, checked=checked,
                              status='PASS' if any(c['numerics'] for c in checked) else 'NO_QUALIFIED_ALGORITHM')
            else:
                prepared = PreparedProjection(weight, m, producer, args.producer, mx_weight=mx_weight)
                prepared(producer, out=candidate)
                if not torch.allclose(candidate, baseline, rtol=.01, atol=.001):
                    raise RuntimeError('prepared output differs from the matched DeepGEMM baseline')
                # Capture on a separate stream: scratch must be owned by that
                # stream and stay alive across changed-input graph replays.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    prepared(producer, out=candidate)
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph, stream=stream):
                    prepared(producer, out=candidate)
                try:
                    for multiplier in (.5, 2.):
                        if args.producer == 'bf16':
                            source.mul_(multiplier)
                        else:
                            # Vary valid packet values without touching the
                            # padding or transport scales.
                            stride = source.numel()//4
                            for rank in range(4):
                                view = source[rank*stride:rank*stride+local*k].view(torch.float8_e4m3fn)
                                view.copy_((view.float()*multiplier).to(torch.float8_e4m3fn))
                        graph.replay()
                        fp8_gemm_nt(producer(False, None), weight, baseline)
                        if not torch.allclose(candidate, baseline, rtol=.01, atol=.001):
                            raise RuntimeError('changed-input graph replay differs from matched baseline')
                finally:
                    graph.reset()
                record = dict(prepared.record, status='PASS', graph_replays=2)
            record['extra_weight_scale_bytes'] = mx_weight.numel()
            record['activation_scale_bytes'] = s1.numel()
            report['cells'].append(record)
            save()
        report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['cells']) else 'NO_QUALIFIED_ALGORITHM'
    except BaseException as error:
        report.update(status='FAIL', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
