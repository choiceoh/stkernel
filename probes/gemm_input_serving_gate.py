#!/usr/bin/env python3
"""Same-build oracle, allocator/replay and timing gate for serving input reuse."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
from gemm_input_reuse import FLAGS, ROOT


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir',type=Path,default=Path('/build'))
    ap.add_argument('--out',type=Path,default=Path('/evidence/production-gate.json'))
    ap.add_argument('--samples',type=int,default=32)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    os.environ.update(MAX_JOBS='1',VLLM_GLM53_MK_PDL='1')
    import torch
    from torch.utils.cpp_extension import load
    src=ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.cu'
    sha=hashlib.sha256(src.read_bytes()).hexdigest()
    build=args.build_dir/sha[:16];build.mkdir(parents=True,exist_ok=True)
    ext=load(name='input_serving_'+sha[:16],sources=[str(src)],extra_cuda_cflags=FLAGS,
             build_directory=str(build),verbose=False)
    spec=importlib.util.spec_from_file_location('input_serving_driver',ROOT/'overlay/modules/glm53_megakernel/glm53_megakernel.py')
    mk=importlib.util.module_from_spec(spec);spec.loader.exec_module(mk);mk._EXT=ext
    assert ext.probe_device()[:3]==[12,1,48]
    torch.backends.cuda.matmul.allow_tf32=False
    result={'source_sha256':sha,'flags':FLAGS,'gates':[],'timings':[],'status':'RUNNING'}
    def save():args.out.write_text(json.dumps(result,indent=2)+'\n')
    args.out.parent.mkdir(parents=True,exist_ok=True);save()
    flush=torch.zeros(64*1024*1024,dtype=torch.uint8,device='cuda')
    shapes=((6,6528,4096,False),(6,4096,512,False),(6,6528,4096,True),
            (6,4096,512,True),(1,6528,4096,False),(8,6528,4096,False),
            (6,6144,4096,False),(6,4096,4096,False),(16,1024,4096,False))
    retained=[]
    for m,n,k,bg in shapes:
        torch.manual_seed(847+m+n+k)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.3
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.05
        pack=mk.build_mk_weight_w4(w);del w
        wr=mk.mk_w4_dequant(pack[0],pack[1],n,pack[2],pack[3] if len(pack)>3 else None).float()
        graphs={};ys={};plans={}
        for mode in (0,1):
            ext.set_gemm_input(mode);ext.set_gemm2(0)
            plans[mode]=ext.gemm_input_plan(m,n,k,bg,False)
            for _ in range(2):mk._gemm_call(x,pack,n,bg=bg)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):ys[mode]=mk._gemm_call(x,pack,n,bg=bg)
            graphs[mode]=graph
        active=m==6 and not bg and (n,k) in ((6528,4096),(4096,512))
        assert bool(plans[1][0])==active
        assert not ext.gemm_input_plan(m,n,k,bg,True)[0], 'low-rank path must fall back'
        for case in ('random','zero','tiny','wide','random2'):
            if case=='zero':x.zero_()
            else:x.normal_().mul_({'tiny':1e-20,'wide':1e3}.get(case,.3))
            # Freed temporary storage must not corrupt either captured graph.
            poison=torch.empty(512*1024,dtype=torch.uint8,device='cuda');poison.fill_(0xA5);del poison
            for mode in (1,0,1):graphs[mode].replay()
            ref=mk._mk_quant_x_ref(x)@wr.T
            torch.cuda.synchronize()
            for mode in (0,1):
                rel,over=mk._exact_gate(ys[mode],ref)
                finite=bool(torch.isfinite(ys[mode]).all())
                row={'shape':[m,n,k],'bg':bg,'mode':mode,'case':case,'relative':rel,'over_ulp':over,'finite':finite}
                result['gates'].append(row);save()
                assert finite and rel<=1e-3 and over==0,row
            assert torch.equal(ys[0],ys[1]), ('output bits changed',m,n,k,bg,case)
        x.normal_().mul_(.3)
        if active and not args.check_only:
            for cache in ('cold','warm'):
                times={0:[],1:[]}
                for rep in range(args.samples):
                    for mode in ((0,1) if rep%2==0 else (1,0)):
                        for _ in range(2):graphs[mode].replay()
                        if cache=='cold':flush.sum()  # read eviction; no dirty 64 MiB writeback
                        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                        a.record();graphs[mode].replay();b.record();b.synchronize()
                        times[mode].append(a.elapsed_time(b)*1000)
                base,cand=median(times[0]),median(times[1])
                row={'shape':[m,n,k],'cache':cache,'plans':plans,'baseline_us':base,'candidate_us':cand,
                     'reduction_pct':100*(base-cand)/base,'raw_us':times}
                result['timings'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k!='raw_us'}),flush=True)
        if active:retained.append((x,wr,ys,graphs))
    # Switch between the two live captured candidates repeatedly. Each must
    # retain its own scratch/output despite intervening shapes and allocations.
    for rep in range(20):
        for x,wr,ys,graphs in retained:
            x.normal_().mul_(.3);graphs[1].replay()
        for x,wr,ys,graphs in reversed(retained):
            ref=mk._mk_quant_x_ref(x)@wr.T
            rel,over=mk._exact_gate(ys[1],ref)
            assert rel<=1e-3 and over==0,(rep,rel,over)
    ext.set_gemm_input(1)
    assert mk._selftest_input_reuse()
    result['alternating_graph_replays']=40
    result['boot_gate']=True;result['status']='PASS';save()
    print('PASS serving-source numerics, fallback, scratch lifetime and replay gates',flush=True)


if __name__=='__main__':main()
