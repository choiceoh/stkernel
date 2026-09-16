"""Owned-GPU experiment: include the cost of restoring a swapped GEMM output."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import triton
import triton.language as tl
from engine.kernels.dense import mxfp8, fp8
from engine.kernels.dense.cublaslt import _measure, WORKSPACE_LIMIT
from engine.kernels.deep_gemm import _initialize
from probes.engine_cublaslt_check import packed_weight


@triton.jit
def restore_kernel(X, Y, M: tl.constexpr, MP: tl.constexpr, N: tl.constexpr):
    row = tl.arange(0, MP)
    col = tl.program_id(0) * 128 + tl.arange(0, 128)
    values = tl.load(X + col[None, :] * MP + row[:, None], col[None, :] < N, 0)
    tl.store(Y + row[:, None] * N + col[None, :], values,
             (row[:, None] < M) & (col[None, :] < N))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--packed-weight', type=Path, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--rows', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_cuda_sources
    from engine.kernels.native_root import build_root
    source = Path(__file__).with_name('transpose_plan.cpp')
    flags, links = ['-O2', '-std=c++17'], ['-lcublasLt']
    key, directory, sources = prepare_cuda_sources(build_root('cublaslt-layout'), [source],
                                                    (flags, links, torch.__version__, torch.version.cuda))
    swapped_native = load(name='st_cublaslt_layout_'+key, sources=list(sources), extra_cflags=flags,
                          extra_ldflags=links, with_cuda=True, build_directory=str(directory))
    native = swapped_native
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError('this explicit experiment is restricted to SM120')
    contexts = native.Context(0, *capability), swapped_native.Context(0, *capability)
    _initialize()
    from deep_gemm import fp8_gemm_nt
    weight, receipt = packed_weight(args.packed_weight, args.n, args.k)
    weight = tuple(t.to('cuda') for t in weight)
    sw = mxfp8.pack_weight_scales(weight[1], args.n, args.k)
    report = dict(device=torch.cuda.get_device_name(), scope='SM120 probe only', weight=receipt,
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (source, Path(__file__), Path('engine/kernels/dense/cublaslt.cpp'))},
                  cells=[], status='RUNNING')
    def save():
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for m in args.rows:
            torch.manual_seed(2700+m)
            n, k = args.n, args.k
            mp = triton.next_power_of_2(max(8, m))
            padded = torch.zeros(mp, k, device='cuda', dtype=torch.bfloat16)
            x = padded[:m]
            x.normal_()
            reference, direct, alternate = (torch.empty(m, n, device='cuda', dtype=torch.bfloat16) for _ in range(3))
            raw = torch.empty(n, mp, device='cuda', dtype=torch.bfloat16)
            q0, s0 = fp8.quantize(x)
            def base():
                fp8.quantize(x, out=(q0, s0))
                fp8_gemm_nt((q0, s0), weight, reference)
            base()
            qp, sp = mxfp8.quantize(padded)
            qd, sd = mxfp8.quantize(x)
            assert torch.equal(q0.view(torch.uint8), qp[:m].view(torch.uint8))
            plans = (native.Plan(contexts[0], m, n, k, WORKSPACE_LIMIT, sd, sw),
                     swapped_native.Plan(contexts[1], n, mp, k, WORKSPACE_LIMIT, sw, sp))
            cell = dict(shape=[m,n,k], padded_rows=mp, layouts=[])
            for label, plan, quant_source, q, s, out in (
                    ('direct', plans[0], x, qd, sd, direct),
                    ('swapped', plans[1], padded, qp, sp, raw)):
                candidates = plan.candidates()
                scratch = torch.empty(max((c['workspace'] for c in candidates),default=0),device='cuda',dtype=torch.uint8)
                layout = dict(layout=label, search=plan.statistics(), screened=[], brackets=[])
                ranked=[]
                runs={}
                for warps in (4,1,2):
                    produce=mxfp8.bind_quantize(quant_source,out=(q,s),num_warps=warps)
                    for candidate in candidates:
                        index=candidate['index']
                        binding=(plan.bind(index,q,weight[0],s,sw,out,scratch) if label=='direct' else
                                 plan.bind(index,weight[0],q,sw,s,out,scratch))
                        def fn(produce=produce,binding=binding,label=label):
                            produce()
                            binding.run()
                            if label=='swapped':
                                restore_kernel[(triton.cdiv(n,128),)](raw,alternate,m,mp,n,num_warps=4)
                        fn()
                        result=direct if label=='direct' else alternate
                        numeric=bool(torch.allclose(result,reference,rtol=.01,atol=.001))
                        ms=_measure(fn,None,repeats=2) if numeric else None
                        layout['screened'].append(dict(candidate,warps=warps,numerics=numeric,milliseconds=ms))
                        if numeric:
                            ranked.append((ms,index,warps)); runs[index,warps]=fn
                if ranked:
                    _,index,warps=min(ranked)
                    fn=runs[index,warps]
                    times=[_measure(base,None),_measure(fn,None),_measure(fn,None),_measure(base,None)]
                    layout['brackets'].append(dict(index=index,warps=warps,B_A_A_B=times))
                    layout['selected']=dict(index=index,warps=warps)
                    layout['fn']=fn
                cell['layouts'].append(layout)
            if all('fn' in layout for layout in cell['layouts']):
                d,a=(layout['fn'] for layout in cell['layouts'])
                cell['direct_swapped_swapped_direct']=[_measure(d,None),_measure(a,None),_measure(a,None),_measure(d,None)]
                for factor in (.5,2.):
                    x.normal_().mul_(factor)
                    base(); d(); a()
                    torch.testing.assert_close(direct,reference,rtol=.01,atol=.001)
                    torch.testing.assert_close(alternate,reference,rtol=.01,atol=.001)
            for layout in cell['layouts']:
                layout.pop('fn',None)
            report['cells'].append(cell); save()
            print(json.dumps(cell),flush=True)
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error)); raise
    finally:
        save()


if __name__=='__main__':
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
