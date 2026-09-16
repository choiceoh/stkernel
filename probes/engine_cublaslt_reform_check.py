"""Matched serving A/B against the frozen PR1071 reader; no fleet operations."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--timing-target', choices=('gb10', 'sm120-probe'), default='gb10')
    ap.add_argument('--kind', choices=('head', 'fc'), required=True)
    ap.add_argument('--packed-weight', type=Path, required=True)
    ap.add_argument('--baseline-reader', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu:
        ap.error('explicit --gpu and an owned idle device are required')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from engine.kernels.dense import FP8Linear, mxfp8
    from engine.kernels.dense.cublaslt import _build, _measure, require_timing_target
    from engine.kernels.dense.cublaslt_serving import weight_nbytes
    from engine.kernels.common.norm_rope import norm
    from probes.engine_cublaslt_check import packed_weight
    spec = importlib.util.spec_from_file_location('engine.kernels.dense._baseline_reader', args.baseline_reader)
    before = importlib.util.module_from_spec(spec); spec.loader.exec_module(before)
    prop = torch.cuda.get_device_properties(0)
    require_timing_target(dict(capability=(prop.major, prop.minor), sms=prop.multi_processor_count), args.timing_target)
    n, k = (38784, 4096) if args.kind == 'head' else (4096, 20480)
    weight, receipt = packed_weight(args.packed_weight, n, k)
    weight = tuple(t.cuda() for t in weight)
    baseline = FP8Linear(weight[0], quantized=weight)
    baseline.cublas = before.Reader(weight, split_decode=args.kind == 'fc')
    candidate = FP8Linear(weight[0], quantized=weight)
    storage = torch.empty(weight_nbytes(n, k, split_decode=args.kind == 'fc'), device='cuda', dtype=torch.uint8)
    candidate.prepare_cublas(split_decode=args.kind == 'fc', storage=storage)
    report = dict(status='RUNNING', device=prop.name, capability=[prop.major, prop.minor], sms=prop.multi_processor_count,
                  torch=torch.__version__, cuda=torch.version.cuda, cublaslt=_build().version(), weight=receipt,
                  scope='actual serving calls; real weights, synthetic inputs; no GB10/TP4/acceptance claim',
                  baseline_sha256=hashlib.sha256(args.baseline_reader.read_bytes()).hexdigest(), preparation=[], cells=[])
    paths = ('engine/kernels/dense/__init__.py', 'engine/kernels/dense/cublaslt.cpp',
             'engine/kernels/dense/cublaslt_serving.py', 'engine/kernels/dense/cublaslt_split.py',
             'engine/profiles/glm53/drafter.py', 'engine/profiles/glm53/cublas.py',
             'probes/engine_cublaslt_reform_check.py')
    report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    def save(): args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        native = _build()
        for parts, (seed, _) in candidate.cublas.seeds.items():
            kp = k//parts
            scales = torch.full((mxfp8.scale_bytes(8,kp),) if parts == 1 else (parts,mxfp8.scale_bytes(8,kp)),
                                127,device='cuda',dtype=torch.uint8)
            ws = candidate.cublas.mx_weight if parts == 1 else candidate.cublas.split_weight.scales
            def prepare(fast):
                return native.Plan(candidate.cublas.context,8,n,kp,0,scales,ws,parts,parts != 1,fast,
                                   candidate.cublas.split_weight.padding if parts != 1 else 0)
            prepare(False);prepare(True);torch.cuda.synchronize()
            timings = []
            for fast in (False,True,True,False)*3:
                started=time.perf_counter_ns();plan=prepare(fast)
                timings.append(dict(preferred=fast,ms=(time.perf_counter_ns()-started)/1e6,statistics=plan.statistics()))
            report['preparation'].append(dict(parts=parts,samples=timings))
        shapes = ([(m,False,False) for m in (1,7,8,14,16,24,32,56,64,256)] if args.kind == 'head' else
                  [(m,True,bias) for m in (1,7,8,16,24,32,64) for bias in (False,True)]
                  +[(m,False,False) for m in (1,128,512,2304)])
        for m,decode,has_bias in shapes:
            torch.manual_seed(2718+m)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            nw=torch.randn(n,device='cuda',dtype=torch.bfloat16)
            bias=torch.randn(n,device='cuda',dtype=torch.float32)*.01 if has_bias else None
            old,out=(torch.empty(m,n,device='cuda',dtype=torch.bfloat16) for _ in range(2))
            reference=None
            def base():
                nonlocal reference
                result=baseline(x,out=old,decode=decode)
                reference=norm(result,nw,1e-6,bias=bias) if decode else result
            def trial():
                candidate(x,out=out,decode=decode,normalization=(nw,1e-6,bias) if decode else None)
            base();trial()
            torch.testing.assert_close(out,reference,rtol=0,atol=0)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):trial()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):trial()
            try:
                for scale in (.5,2.):
                    x.normal_().mul_(scale);base();torch.cuda.synchronize()
                    allocated=torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                    graph.replay();torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()['allocated_bytes.all.allocated']==allocated
                    torch.testing.assert_close(out,reference,rtol=0,atol=0)
            finally:graph.reset()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                _measure(base,None);_measure(trial,None)
                brackets=[[_measure(fn,None) for fn in (base,trial,trial,base)] for _ in range(2)]
            cell=dict(shape=[m,n,k],decode=decode,bias=has_bias,bit_exact=True,changed_input_graph_replays=2,
                      graph_allocation_bytes=0,base_trial_trial_base_ms=brackets)
            report['cells'].append(cell);save();print(json.dumps(cell),flush=True)
        report.update(status='PASS',reader=candidate.cublas.report())
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:save()


if __name__=='__main__':
    main()
