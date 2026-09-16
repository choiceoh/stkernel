"""Explicit cuBLASLt producer-to-output comparison; never run by a model forward.

Use --gpu only with an owned idle GPU. No reservations, service control or
network calls are made here. SM120 timing requires an explicit probe target;
its results are not GB10 evidence. Synthetic inputs do not establish acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def packed_weight(path, n, k):
    """Read an explicit, immutable FP8 PackStore artifact without requantizing it."""
    import torch
    blob = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
    if not isinstance(blob, dict) or not isinstance(blob.get('identity'), dict):
        raise ValueError('FP8 pack must contain a PackStore identity and tensors')
    identity = blob['identity']
    q, scale = blob.get('q'), blob.get('scale')
    if (identity.get('kind') != 'fp8' or not isinstance(q, torch.Tensor)
            or not isinstance(scale, torch.Tensor) or q.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.float32 or tuple(q.shape) != (n, k)
            or tuple(scale.shape) != (n // 128, k // 128)
            or not q.is_contiguous() or not scale.is_contiguous()):
        raise ValueError('FP8 pack does not match the requested padded weight shape')
    digest = hashlib.sha256()
    for value in (q, scale):
        digest.update(value.view(torch.uint8).numpy())
    return (q, scale), dict(identity=identity, sha256=digest.hexdigest(), path=str(path))


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
    parser.add_argument('--packed-weight', type=Path,
                        help='explicit FP8 PackStore artifact; all shapes must use its padded N,K')
    parser.add_argument('--timing-target', choices=('gb10', 'sm120-probe'), default='gb10',
                        help='SM120 measurements are device-specific probe evidence, never GB10 admission')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.gpu:
        parser.error('execution requires --gpu and an owned idle GPU; this tool does not acquire one')
    shapes = args.shape or [(8, 4096, 20480), (8, 38784, 4096), (512, 6144, 4096)]
    if args.producer == 'packets' and any(k != 4096 or m < 128 for m, _, k in shapes):
        parser.error('packet producer needs M >= 128 and K = 4096; specify matching --shape values')
    if args.packed_weight and len({(n, k) for _, n, k in shapes}) != 1:
        parser.error('--packed-weight requires one common N,K across all --shape values')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    fixed_weight, fixed_record = (packed_weight(args.packed_weight, *shapes[0][1:])
                                  if args.packed_weight else (None, None))
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense import fp8, mxfp8
    from engine.kernels.dense.cublaslt import BF16Producer, PacketProducer, PreparedProjection, _build, WORKSPACE_LIMIT
    from engine.kernels.deep_gemm import _initialize
    _initialize()
    from deep_gemm import fp8_gemm_nt
    device = torch.cuda.current_device()
    prop = torch.cuda.get_device_properties(device)
    capability = prop.major, prop.minor
    if capability not in ((12, 0), (12, 1)):
        raise RuntimeError('numerical probe only supports the declared SM120/SM121 targets')
    if not args.numerics_only:
        from engine.kernels.dense.cublaslt import require_timing_target
        require_timing_target(dict(capability=capability, sms=prop.multi_processor_count), args.timing_target)
    from dataclasses import replace
    from engine.base.kernel_shape import MEASURED, Device, bind
    bind(replace(MEASURED, device=Device(capability=capability, sms=prop.multi_processor_count)))
    report = dict(status='RUNNING', gpu_used=True, numerics_only=args.numerics_only,
                  scope=('packed model weight; synthetic inputs; no serving or acceptance verdict'
                         if args.packed_weight else
                         'synthetic kernel/producer evidence; no serving or acceptance verdict'),
                  device=prop.name, capability=capability, sms=prop.multi_processor_count,
                  timing_target=args.timing_target, gb10_admission=False,
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
        if fixed_weight is not None:
            fixed_weight = tuple(value.to('cuda') for value in fixed_weight)
            fixed_scales = mxfp8.pack_weight_scales(fixed_weight[1], *shapes[0][1:])
        for index, (m, n, k) in enumerate(shapes):
            torch.manual_seed(1907 + index)
            if args.packed_weight:
                weight, weight_record, mx_weight = fixed_weight, fixed_record, fixed_scales
            else:
                weight = FP8Linear(torch.randn(n, k, device='cuda', dtype=torch.bfloat16)*.02).weight
                weight_record = dict(kind='synthetic', seed=1907 + index)
                mx_weight = mxfp8.pack_weight_scales(weight[1], n, k)
            if args.producer == 'bf16':
                source = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
                producer = BF16Producer(source)
            else:
                from engine.kernels.prefill_collectives import PrefillCollectives
                local = (m+3)//4
                packets = []
                for rank in range(4):
                    x = torch.randn(local, k, device='cuda', dtype=torch.bfloat16)*2**rank
                    packets.append(PrefillCollectives.pack(x, x.numel())[0])
                source = torch.cat(packets)
                producer = PacketProducer(source, local, real_rows=m)
            baseline = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            candidate = torch.empty_like(baseline)
            q0, s0 = producer(False, None)
            q1, s1 = producer(True, None)
            if not torch.equal(q0.view(torch.uint8), q1.view(torch.uint8)):
                raise RuntimeError('producer FP8 byte mismatch')
            reference_scales = s1.clone()
            producer_checks = []
            for warps in (1, 2, 4):
                # Numerical-only cards exercise every producer layout too,
                # without entering the GB10-only algorithm timing path.
                bound_producer = producer.bind(True, out=(q1, s1), num_warps=warps)
                bound_producer()
                same_q = torch.equal(q0.view(torch.uint8), q1.view(torch.uint8))
                same_s = torch.equal(reference_scales, s1)
                if not same_q or not same_s:
                    raise RuntimeError(f'producer warp {warps} changed activation or scale bytes')
                producer_checks.append(dict(warps=warps, identical_fp8=bool(same_q), identical_scales=bool(same_s)))
            fp8_gemm_nt((q0, s0), weight, baseline)
            if args.numerics_only:
                native = _build()
                context = native.Context(device, *capability)
                plan = native.Plan(context, m, n, k, WORKSPACE_LIMIT, s1, mx_weight)
                algorithms = plan.candidates()
                scratch = torch.empty(max((c['workspace'] for c in algorithms), default=0), device='cuda', dtype=torch.uint8)
                checked = []
                for c in algorithms:
                    bound = plan.bind(c['index'], q1, weight[0], s1, mx_weight, candidate, scratch)
                    bound.run()
                    relative = ((candidate.float()-baseline.float()).norm()/baseline.float().norm().clamp_min(1e-30)).item()
                    checked.append(dict(c, relative_l2=relative,
                                        numerics=bool(torch.allclose(candidate, baseline, rtol=.01, atol=.001))))
                record = dict(shape=(m, n, k), producer=args.producer, checked=checked, search=plan.statistics(),
                              status='PASS' if any(c['numerics'] for c in checked) else 'NO_QUALIFIED_ALGORITHM')
            else:
                prepared = PreparedProjection(weight, m, producer, args.producer, mx_weight=mx_weight,
                                              timing_target=args.timing_target)
                execution = prepared.bind(producer, out=candidate)
                execution()
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                execution()
                torch.cuda.synchronize()
                allocation_delta = torch.cuda.max_memory_allocated() - resident
                if prepared.choice.index is not None and allocation_delta:
                    raise RuntimeError('bound cuBLAS execution allocated additional Torch GPU storage')
                if not torch.allclose(candidate, baseline, rtol=.01, atol=.001):
                    raise RuntimeError('prepared output differs from the matched DeepGEMM baseline')
                # Capture on a separate stream. The binding owns private
                # scratch and immutable descriptors across graph replays.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    execution()
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph, stream=stream):
                    execution()
                try:
                    for multiplier in (.5, 2.):
                        if args.producer == 'bf16':
                            # Power-of-two rescaling alone leaves FP8 values
                            # unchanged. Replace values to catch a stale Q as
                            # well as stale scale pointers on graph replay.
                            source.normal_().mul_(multiplier)
                        else:
                            # Vary valid packet values without touching the
                            # padding or transport scales.
                            stride = source.numel()//4
                            for rank in range(4):
                                view = source[rank*stride:rank*stride+local*k].view(torch.float8_e4m3fn)
                                fresh = torch.randn(view.numel(), device=view.device)*(multiplier+rank)
                                view.copy_(fresh.to(torch.float8_e4m3fn))
                        graph.replay()
                        fp8_gemm_nt(producer(False, None), weight, baseline)
                        if not torch.allclose(candidate, baseline, rtol=.01, atol=.001):
                            raise RuntimeError('changed-input graph replay differs from matched baseline')
                finally:
                    graph.reset()
                record = dict(prepared.record, status='PASS', graph_replays=2, bound_allocation_delta=allocation_delta,
                              bound_workspace_bytes=0 if execution.workspace is None else execution.workspace.numel())
            record['extra_weight_scale_bytes'] = mx_weight.numel()
            record['weight'] = weight_record
            record['input'] = dict(kind='synthetic', seed=1907 + index)
            record['producer_warp_checks'] = producer_checks
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
