"""Explicit SM120 trial: parallel FP32 cuBLAS partials, then one BF16 reduction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import triton
import triton.language as tl
from engine.kernels.dense import fp8, mxfp8
from engine.kernels.dense.cublaslt import _build, _measure, WORKSPACE_LIMIT
from engine.kernels.deep_gemm import _initialize
from probes.engine_cublaslt_check import packed_weight


@triton.jit
def quantize_split(X,Q,S,M:tl.constexpr,K:tl.constexpr,P:tl.constexpr,SCALAR:tl.constexpr):
    row=tl.program_id(0)*4+tl.arange(0,4)
    group=tl.program_id(1)
    part=group//(K//P//128)
    within=group%(K//P//128)
    col=group*128+tl.arange(0,128)
    x=tl.load(X+row[:,None]*K+col[None,:],row[:,None]<M,0).to(tl.float32)
    scale,inv=mxfp8._power2_scale(tl.maximum(tl.max(tl.abs(x),1),1e-4))
    tl.store(Q+part*M*(K//P)+row[:,None]*(K//P)+within*128+tl.arange(0,128)[None,:],
             (x*inv[:,None]).to(tl.float8e4nv),row[:,None]<M)
    words=mxfp8._scale_word(scale)
    base=part*tl.cdiv(M,128)*(K//P//128)*128
    if SCALAR:
        for i in tl.static_range(4):
            word=tl.sum(tl.where(tl.arange(0,4)==i,words,0),0)
            r=tl.program_id(0)*4+i
            tl.store(S+base+mxfp8._word_offset(r,within,K//P//128),word,r<M)
    else:
        tl.store(S+base+mxfp8._word_offset(row,within,K//P//128),words,row<M)


@triton.jit
def reduce_split(X,Y,SIZE:tl.constexpr,P:tl.constexpr):
    i=tl.program_id(0)*256+tl.arange(0,256)
    value=tl.full((256,),0,tl.float32)
    for p in tl.static_range(P):
        value+=tl.load(X+p*SIZE+i,i<SIZE,0)
    tl.store(Y+i,value.to(tl.bfloat16),i<SIZE)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parts',type=int,nargs='+',default=[2,4,5,8,10])
    parser.add_argument('--rows',type=int,nargs='+',default=[8,16])
    args=parser.parse_args()
    if torch.cuda.get_device_capability()!=(12,0):
        raise RuntimeError('explicit SM120 experiment only')
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_cuda_sources
    from engine.kernels.native_root import build_root
    source=Path(__file__).with_name('batch_plan.cpp')
    flags,links=['-O2','-std=c++17'],['-lcublasLt']
    key,directory,sources=prepare_cuda_sources(build_root('cublaslt-batch'),[source],(flags,links,torch.__version__,torch.version.cuda))
    split_native=load(name='st_cublaslt_batch_'+key,sources=list(sources),extra_cflags=flags,extra_ldflags=links,with_cuda=True,build_directory=str(directory))
    native=_build(); context=native.Context(0,12,0); split_context=split_native.Context(0,12,0)
    _initialize()
    from deep_gemm import fp8_gemm_nt
    n,k=4096,20480
    weight,receipt=packed_weight(Path('/work/packs/e29b30a8de75e0e3e6fc6895abd857dac5a74910db589ab28f5306e98fc50009.pt'),n,k)
    weight=tuple(x.cuda() for x in weight)
    sw=mxfp8.pack_weight_scales(weight[1],n,k)
    report=dict(status='RUNNING',scope='SM120 experiment only; real FC pack and synthetic activations',weight=receipt,
                device=torch.cuda.get_device_name(),cublaslt=native.version(),cells=[],
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,Path(__file__))})
    def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    try:
        for m in args.rows:
            torch.manual_seed(3100+m)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)
            ref,out,base_out=(torch.empty(m,n,device='cuda',dtype=torch.bfloat16) for _ in range(3))
            q0,s0=fp8.quantize(x)
            def deep():
                fp8.quantize(x,out=(q0,s0)); fp8_gemm_nt((q0,s0),weight,ref)
            deep()
            producer=mxfp8.bind_quantize(x,num_warps=2)
            q,s=producer()
            plan=native.Plan(context,m,n,k,WORKSPACE_LIMIT,s,sw)
            candidate=plan.candidates()[0]
            scratch=torch.empty(candidate['workspace'],device='cuda',dtype=torch.uint8)
            bound=plan.bind(candidate['index'],q,weight[0],s,sw,base_out,scratch)
            def direct(): producer(); bound.run()
            direct()
            for parts in args.parts:
                assert k%(parts*128)==0
                kp=k//parts
                w=weight[0].reshape(n,parts,kp).permute(1,0,2).contiguous()
                scales=weight[1].reshape(n//128,parts,kp//128).permute(1,0,2).contiguous()
                ws=torch.stack([mxfp8.pack_weight_scales(scales[p],n,kp) for p in range(parts)])
                qparts=torch.empty(parts,m,kp,device='cuda',dtype=torch.float8_e4m3fn)
                sparts=torch.full((parts,mxfp8.scale_bytes(m,kp)),127,device='cuda',dtype=torch.uint8)
                partial=torch.empty(parts,m,n,device='cuda',dtype=torch.float32)
                words=sparts.view(torch.int32)
                streams=[torch.cuda.Stream() for _ in range(parts)]
                ready=torch.cuda.Event(); done=[torch.cuda.Event() for _ in streams]
                pplan=split_native.Plan(split_context,m,n,kp,WORKSPACE_LIMIT,sparts,ws,parts)
                candidates=pplan.candidates()
                cell=dict(shape=[m,n,k],parts=parts,search=pplan.statistics(),candidates=len(candidates),trials=[])
                def produce(): quantize_split[(triton.cdiv(m,4),k//128)](x,qparts,words,m,k,parts,True,num_warps=1)
                produce()
                restored=qparts.permute(1,0,2).reshape(m,k)
                assert torch.equal(restored.view(torch.uint8),q0.view(torch.uint8))
                ranked=[]; runners={}
                for c in candidates:
                    workspace=torch.empty(c['workspace'],device='cuda',dtype=torch.uint8)
                    binding=pplan.bind(c['index'],qparts,w,sparts,ws,partial,workspace)
                    def run(binding=binding):
                        produce(); binding.run()
                        reduce_split[(triton.cdiv(m*n,256),)](partial,out,m*n,parts,num_warps=4)
                    run()
                    numeric=bool(torch.allclose(out,ref,rtol=.01,atol=.001))
                    ms=_measure(run,None,repeats=2) if numeric else None
                    cell['trials'].append(dict(c,numerics=numeric,milliseconds=ms))
                    if numeric: ranked.append((ms,c['index'])); runners[c['index']]=run
                if ranked:
                    _,index=min(ranked); run=runners[index]
                    cell['choice']=index
                    cell['direct_split_split_direct']=[_measure(direct,None),_measure(run,None),_measure(run,None),_measure(direct,None)]
                    cell['deep_split_split_deep']=[_measure(deep,None),_measure(run,None),_measure(run,None),_measure(deep,None)]
                    for scale in (.5,2.):
                        x.normal_().mul_(scale); deep(); run()
                        torch.testing.assert_close(out,ref,rtol=.01,atol=.001)
                    cell['changed_inputs']=2
                report['cells'].append(cell); save(); print(json.dumps(cell),flush=True)
        report['status']='PASS'
    except BaseException as error:
        report.update(status='FAIL',error=repr(error)); raise
    finally: save()


if __name__=='__main__':
    with torch.cuda.stream(torch.cuda.Stream()): main()
