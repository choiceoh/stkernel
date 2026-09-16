"""Explicit prefill experiment: larger producer tiles and reversed GEMM operands."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
import triton.language as tl
from engine.kernels.dense import mxfp8


@triton.jit
def quantize(X, Q, S, M: tl.constexpr, K: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0)*R + tl.arange(0, R)
    group = tl.program_id(1)
    col = group*128 + tl.arange(0, 128)
    x = tl.load(X+row[:, None]*K+col[None, :], row[:, None] < M, 0).to(tl.float32)
    scale, inverse = mxfp8._power2_scale(tl.maximum(tl.max(tl.abs(x), 1), 1e-4))
    tl.store(Q+row[:, None]*K+col[None, :], (x*inverse[:, None]).to(tl.float8e4nv), row[:, None] < M)
    tl.store(S+mxfp8._word_offset(row, group, K//128), mxfp8._scale_word(scale), row < M)
    if M % 128:
        if tl.program_id(0) == tl.num_programs(0)-1:
            pad = M//128*128+tl.arange(0, 128)
            tl.store(S+mxfp8._word_offset(pad, group, K//128), 0x7F7F7F7F, pad >= M)


@triton.jit
def transpose(X, Y, M: tl.constexpr, N: tl.constexpr):
    r = tl.program_id(0)*32+tl.arange(0, 32)
    c = tl.program_id(1)*32+tl.arange(0, 32)
    v = tl.load(X+c[None, :]*M+r[:, None], (r[:, None] < M) & (c[None, :] < N), 0)
    tl.store(Y+r[:, None]*N+c[None, :], v, (r[:, None] < M) & (c[None, :] < N))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--packed-weight', type=Path, required=True)
    ap.add_argument('--rows', type=int, nargs='+', default=[128, 512, 2304])
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if not args.gpu or torch.cuda.get_device_capability() != (12, 0):
        ap.error('explicit --gpu on the owned SM120 probe required')
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense.cublaslt import _build, _measure, WORKSPACE_LIMIT
    from probes.engine_cublaslt_check import packed_weight
    from engine.kernels.deep_gemm import _initialize
    _initialize()
    native = _build(); context = native.Context(0, 12, 0)
    n, k = 4096, 20480
    weight, receipt = packed_weight(args.packed_weight, n, k)
    weight = tuple(t.cuda() for t in weight)
    ws = mxfp8.pack_weight_scales(weight[1], n, k)
    base = FP8Linear(weight[0], quantized=weight); base.prepare_cublas()
    deep = FP8Linear(weight[0], quantized=weight)
    report = dict(status='RUNNING', device=torch.cuda.get_device_name(), weight=receipt, cells=[],
                  scope='SM120 experiment, real weights and synthetic activations, no engine proof')
    paths = ('engine/kernels/dense/cublaslt.cpp', 'engine/kernels/dense/mxfp8.py', 'probes/engine_cublaslt_prefill_check.py')
    root = Path(__file__).resolve().parents[1]
    report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    def save(): args.output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    try:
        for m in args.rows:
            torch.manual_seed(2718+m)
            x = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
            old, ref, out = (torch.empty(m, n, device='cuda', dtype=torch.bfloat16) for _ in range(3))
            q, s = mxfp8.quantize(x, num_warps=1)
            def baseline(): base(x, out=old)
            baseline(); deep(x, out=ref)
            cell = dict(shape=[m,n,k], producers=[], layouts=[])
            for rows, warps in ((4,1),(8,2),(16,4),(32,4),(32,8),(64,4),(64,8)):
                def produce(): quantize[(triton.cdiv(m,rows), k//128)](x,q,s.view(torch.int32),m,k,rows,num_warps=warps)
                qref,sref = mxfp8.quantize(x,num_warps=1)
                produce()
                assert torch.equal(q.view(torch.uint8),qref.view(torch.uint8)) and torch.equal(s,sref)
                _measure(produce,None,repeats=2)
                cell['producers'].append(dict(rows=rows,warps=warps,ms=_measure(produce,None,repeats=4)))
            best=min(cell['producers'],key=lambda a:a['ms'])
            for swapped in (False,True):
                plan=native.Plan(context,n if swapped else m,m if swapped else n,k,WORKSPACE_LIMIT,ws if swapped else s,s if swapped else ws)
                raw=torch.empty(n,m,device='cuda',dtype=torch.bfloat16) if swapped else out
                layout=dict(swapped=swapped,trials=[])
                for choice in plan.candidates():
                    scratch=torch.empty(choice['workspace'],device='cuda',dtype=torch.uint8)
                    for tiled in (False,True):
                        def trial():
                            if tiled:
                                quantize[(triton.cdiv(m,best['rows']),k//128)](x,q,s.view(torch.int32),m,k,best['rows'],num_warps=best['warps'])
                            else:
                                mxfp8.quantize(x,out=(q,s),num_warps=1)
                            if swapped:
                                plan.run(choice['index'],weight[0],q,ws,s,raw,scratch)
                                transpose[(triton.cdiv(m,32),triton.cdiv(n,32))](raw,out,m,n,num_warps=4)
                            else:
                                plan.run(choice['index'],q,weight[0],s,ws,out,scratch)
                        trial();torch.testing.assert_close(out,ref,rtol=.01,atol=.001)
                        _measure(trial,None,repeats=2);_measure(baseline,None,repeats=2)
                        times=[_measure(fn,None,repeats=2) for fn in (baseline,trial,trial,baseline)]
                        layout['trials'].append(dict(choice,tiled=tiled,base_trial_trial_base_ms=times))
                cell['layouts'].append(layout)
            report['cells'].append(cell);save();print(json.dumps(cell),flush=True)
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error));raise
    finally:save()


if __name__=='__main__':
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
