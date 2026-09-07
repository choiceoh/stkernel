#!/usr/bin/env python3
"""Local M64 numerical/lifetime cases under memcheck or racecheck.

Run only through the owned offline TP4 runner after both transport gates pass.
This supplements TP4 numerics; it does not measure serving latency.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.update(VLLM_GLM53_B12X_PREFILL_M64='1',VLLM_GLM53_B12X_STATIC_V2='t',
                  VLLM_GLM53_B12X_PREFILL_REUSE='0',VLLM_GLM53_B12X_PREFILL_FC1_N128='0')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group()
    mode.add_argument('--int8',action='store_true')
    mode.add_argument('--reuse-diagnostic',action='store_true',help='Collect repeated local controls; never numerical or serving acceptance')
    args=ap.parse_args()
    # Initialize the CUDA driver before PyTorch creates a runtime context.
    # Keep sanitizer API reporting enabled and check every requested status.
    from cuda.bindings import driver
    status,version=driver.cuDriverGetVersion()
    assert int(status)==0,status
    initialized=driver.cuInit(0)
    assert int(initialized[0])==0,initialized
    print(json.dumps(dict(kind='CUDA_DRIVER_INITIALIZED_BEFORE_TORCH',driver_version=version)),flush=True)
    import torch
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
    from b12x_static_probe import expert_set
    assert torch.cuda.get_device_capability()==(12,1)
    manifest=Path('/repo/build/glm53/manifest.tsv');provenance={}
    for line in manifest.read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,target,*_=line.split('\t')
        actual=hashlib.sha256(Path(target).read_bytes()).hexdigest()
        assert actual==hashlib.sha256((manifest.parent/name).read_bytes()).hexdigest(),name
        provenance[name]=actual
    md._STATIC_V2_OVERRIDE=md._parse_glm53_static_v2('t',probe=True)
    w13,sf13,w2,sf2=expert_set(torch.Generator().manual_seed(73209))
    md.tile_expert_weights_inplace(w13,w2)
    wrapper=B12xMoEWrapper(num_experts=288,num_local_experts=288,top_k=8,
        hidden_size=4096,intermediate_size=512,use_cuda_graph=True,max_num_tokens=8192,
        activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.)
    candidate=wrapper._prefill_m64_workspace
    assert candidate.tile_m==64 and wrapper._dynamic_workspace.tile_m==128
    ones=torch.ones(288,device='cuda');results=[]
    def case_factory(rows,skew):
        gen=torch.Generator(device='cuda').manual_seed(9211+rows)
        x=torch.randn(rows,4096,generator=gen,device='cuda',dtype=torch.bfloat16)*.5
        def call(enabled):
            wrapper._prefill_m64_workspace=candidate if enabled else None
            ids=((x[:,0].float().abs()*1024).int()[:,None]+torch.arange(8,device='cuda'))%(8 if skew else 288)
            scales=torch.softmax(x[:,:8].float(),dim=1);out=torch.empty_like(x)
            return wrapper.run(x,w13,sf13,w2,sf2,ids.to(torch.int32),scales,
                w1_alpha=ones,w2_alpha=ones,fc2_input_scale=ones,out=out)
        return x,call
    if args.reuse_diagnostic:
        from glm53_moe_m64_reuse_diagnostic import run
        run(torch=torch,case_factory=case_factory,provenance=provenance)
        return
    def compare(a,b,repeat):
        a,b,r=(v.float() for v in (a,b,repeat))
        assert all(bool(torch.isfinite(v).all()) for v in (a,b,r))
        norm=b.norm(dim=1).clamp_min(1e-6);peak=b.abs().amax(dim=1).clamp_min(1e-6)
        err=(a-b).norm(dim=1)/norm;noise=(r-b).norm(dim=1)/norm
        worst=(a-b).abs().amax(dim=1)/peak;npeak=(r-b).abs().amax(dim=1)/peak
        bad=(err>torch.maximum(3*noise,torch.full_like(noise,.02))) | (worst>torch.maximum(3*npeak,torch.full_like(npeak,.04)))
        assert not bool(bad.any()),dict(bad_rows=int(bad.sum()),l2=float(err.max()),peak=float(worst.max()))
    for rows in (6144,6912,8192):
        for skew in (False,True):
            x,call=case_factory(rows,skew)
            b=call(False);repeat=call(False);control=call(False);a=call(True)
            compare(control,b,repeat);compare(a,b,repeat)
            retained=a.clone();x.mul_(-.75)
            trash=[torch.empty_like(x) for _ in range(3)];del trash
            b=call(False);repeat=call(False)
            for _ in range(3):compare(call(True),b,repeat)
            assert torch.equal(a,retained),'retained output changed'
            torch.cuda.synchronize()
            assert any('glm53_prefill_m64_v3' in key for key in md._DYNAMIC_KERNEL_CACHE)
            results.append(dict(rows=rows,skew=skew,bad_rows=0))
            print(json.dumps(results[-1]),flush=True)
    extra={}
    if args.int8:
        from vllm.distributed.device_communicators import glm53_prefill_collectives as h
        from glm53_prefill_int8_sanitize import run
        extra['int8']=run(torch,h)
    print(json.dumps(dict(verdict='MOE_M64_SANITIZER_CASES_PASS',provenance=provenance,results=results,**extra)),flush=True)

if __name__=='__main__':main()
