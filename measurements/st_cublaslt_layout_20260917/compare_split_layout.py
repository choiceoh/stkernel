"""Owned-GPU split geometry/weight-layout experiment, including full producer cost."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--timing-target', choices=('gb10', 'sm120-probe'), default='gb10')
    ap.add_argument('--kind', choices=('fc', 'head'), default='fc')
    ap.add_argument('--rows', type=int, nargs='+', default=[8, 16, 64, 512, 2304])
    ap.add_argument('--parts', type=int, nargs='+', default=[2, 4, 5, 8, 10, 16])
    ap.add_argument('--packed-weight', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu:
        ap.error('explicit --gpu and an owned idle device are required')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import triton
    from engine.kernels.dense import FP8Linear, mxfp8
    from engine.kernels.dense.cublaslt import _build, _measure, require_timing_target
    from engine.kernels.dense.cublaslt_serving import _candidate
    from engine.kernels.dense.cublaslt_split import _quantize, _reduce
    from probes.engine_cublaslt_check import packed_weight
    from engine.kernels.deep_gemm import _initialize
    _initialize()
    prop = torch.cuda.get_device_properties(0)
    require_timing_target(dict(capability=(prop.major, prop.minor), sms=prop.multi_processor_count), args.timing_target)
    n, k = (4096, 20480) if args.kind == 'fc' else (38784, 4096)
    weight, receipt = packed_weight(args.packed_weight, n, k)
    weight = tuple(t.cuda() for t in weight)
    baseline = FP8Linear(weight[0], quantized=weight)
    baseline.prepare_cublas(split_decode=args.kind == 'fc')
    deep = FP8Linear(weight[0], quantized=weight)
    native = _build()
    context = native.Context(0, prop.major, prop.minor)
    report = dict(status='RUNNING', scope='split layout experiment; real weights, synthetic activations; no engine/acceptance proof',
                  torch=torch.__version__, cuda=torch.version.cuda, cublaslt=native.version(), device=prop.name,
                  weight=receipt, cells=[])
    paths = ('engine/kernels/dense/cublaslt.cpp', 'engine/kernels/dense/cublaslt_split.py',
             'engine/kernels/dense/cublaslt_serving.py', 'probes/engine_cublaslt_layout_check.py')
    root = Path(__file__).resolve().parents[1]
    report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    def save(): args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for m in args.rows:
            torch.manual_seed(2718+m)
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            ref, old, out = (torch.empty(m, n, device='cuda', dtype=torch.bfloat16) for _ in range(3))
            def base(): return baseline(x, out=old, decode=args.kind == 'fc' and m <= 64)
            def deep_run(): return deep(x, out=ref)
            base(); deep_run()
            for parts in args.parts:
                if k % (parts*128):
                    continue
                kp = k//parts
                ws = weight[1].reshape(n//128, parts, kp//128).permute(1, 0, 2).contiguous()
                ws = torch.stack([mxfp8.pack_weight_scales(ws[p], n, kp) for p in range(parts)])
                query = torch.full((parts, mxfp8.scale_bytes(m, kp)), 127, device='cuda', dtype=torch.uint8)
                original_view = weight[0].reshape(n, parts, kp).permute(1, 0, 2)
                for strided in (False, True):
                    w = original_view if strided else original_view.contiguous()
                    plan = native.Plan(context, m, n, kp, 0, query, ws, parts, True, strided)
                    cell = dict(shape=[m, n, k], parts=parts, strided=strided, search=plan.statistics())
                    if not plan.candidates():
                        cell['status'] = 'UNSUPPORTED'
                    else:
                        choice = _candidate(plan.candidates())
                        workspace = torch.empty(0, device='cuda', dtype=torch.uint8)
                        def trial():
                            q = torch.empty((parts, m, kp), device='cuda', dtype=torch.float8_e4m3fn)
                            s = torch.empty_like(query)
                            partial = torch.empty((parts, m, n), device='cuda', dtype=torch.float32)
                            _quantize[(triton.cdiv(m, 4), k//128)](x, q, s.view(torch.int32), m, k, parts,
                                                                bool(m % 128), num_warps=1)
                            plan.run(choice['index'], q, w, s, ws, partial, workspace)
                            _reduce[(triton.cdiv(m*n, 256),)](partial, out, m*n, parts, num_warps=4)
                        trial()
                        torch.testing.assert_close(out, ref, rtol=.01, atol=.001)
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            trial()
                            cell['base_trial_trial_base_ms'] = [_measure(fn, None, repeats=2) for fn in (base, trial, trial, base)]
                            cell['deep_ms'] = _measure(deep_run, None, repeats=2)
                        torch.cuda.current_stream().wait_stream(stream)
                        for scale in (.5, 2.):
                            x.normal_().mul_(scale); deep_run(); trial()
                            torch.testing.assert_close(out, ref, rtol=.01, atol=.001)
                        cell.update(status='PASS', choice=choice, extra_weight_bytes=ws.numel()+(0 if strided else w.numel()))
                    report['cells'].append(cell); save(); print(json.dumps(cell), flush=True)
                    del plan, w
                del query, ws, original_view
        report['status'] = 'PASS'
    except BaseException as error:
        report.update(status='FAIL', error=repr(error)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
