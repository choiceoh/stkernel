"""Exercise the actual FP8Linear cuBLAS default with real packs; no fleet calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--timing-target', choices=('gb10','sm120-probe'), default='gb10')
    ap.add_argument('--kind', choices=('head','fc'), required=True)
    ap.add_argument('--packed-weight', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu:
        ap.error('requires explicit --gpu on the owned GPU')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense.cublaslt import _measure, require_timing_target
    from engine.kernels.dense.cublaslt_serving import weight_nbytes
    from probes.engine_cublaslt_check import packed_weight
    from engine.kernels.deep_gemm import _initialize
    _initialize()
    import deep_gemm
    prop = torch.cuda.get_device_properties(0)
    require_timing_target(dict(capability=(prop.major, prop.minor), sms=prop.multi_processor_count), args.timing_target)
    n, k, logical = (38784,4096,38720) if args.kind=='head' else (4096,20480,4096)
    weight, receipt = packed_weight(args.packed_weight, n, k)
    weight = tuple(t.cuda() for t in weight)
    baseline = FP8Linear(weight[0][:logical], quantized=weight)
    serving = FP8Linear(weight[0][:logical], quantized=weight)
    storage = torch.empty(weight_nbytes(n,k,split_decode=args.kind=='fc'), device='cuda', dtype=torch.uint8)
    serving.prepare_cublas(split_decode=args.kind=='fc', storage=storage)
    shapes = ([(m,False) for m in (1,7,8,14,16,21,24,28,32,56,64,256)] if args.kind=='head' else
              [(m,True) for m in (1,7,8,16,24,32,64)]+[(m,False) for m in (1,128,512,2304)])
    report = dict(status='RUNNING', device=prop.name, capability=[prop.major,prop.minor], sms=prop.multi_processor_count,
                  cuda=torch.version.cuda, torch=torch.__version__, timing_target=args.timing_target,
                  scope='actual serving reader; real packed weights, synthetic activations; no TP4/acceptance verdict',
                  weight=receipt, cells=[])
    root=Path(__file__).resolve().parents[1]
    sources=('engine/kernels/dense/__init__.py','engine/kernels/dense/cublaslt.cpp','engine/kernels/dense/cublaslt_serving.py',
             'engine/kernels/dense/cublaslt_split.py','engine/kernels/dense/mxfp8.py','probes/engine_cublaslt_serving_check.py')
    report['source_sha256']={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}
    def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for m,decode in shapes:
            torch.manual_seed(2718+m)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            ref=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
            out=torch.empty_like(ref)
            def base(): return baseline(x,out=ref)
            def trial(): return serving(x,out=out,decode=decode)
            base()
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with patch.object(deep_gemm,'fp8_gemm_nt',side_effect=AssertionError('DeepGEMM used by cuBLAS default')):
                trial()
                torch.testing.assert_close(out,ref,rtol=.01,atol=.001)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream): trial()
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph,stream=stream): trial()
            try:
                for scale in (.5,2.):
                    x.normal_().mul_(scale)
                    base()
                    torch.cuda.synchronize()
                    before=torch.cuda.memory_stats()['allocated_bytes.all.allocated']
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.cuda.memory_stats()['allocated_bytes.all.allocated']==before
                    torch.testing.assert_close(out,ref,rtol=.01,atol=.001)
            finally: graph.reset()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                samples=[_measure(fn,None) for fn in (base,trial,trial,base)]
            cell=dict(shape=[m,n,k],decode=decode,numerics=True,changed_input_graph_replays=2,
                      graph_allocation_bytes=0,deep_serving_serving_deep_ms=samples)
            report['cells'].append(cell);save();print(json.dumps(cell),flush=True)
        report.update(status='PASS',reader=serving.cublas.report())
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally: save()


if __name__=='__main__':
    main()
