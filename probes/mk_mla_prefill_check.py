#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MK_SEG_MLA on PREFILL shapes: numerics vs the torch twin + rate vs FlashInfer."""
import importlib.util, os, sys, time
os.environ.setdefault("VLLM_GLM53_MEGAKERNEL","1"); os.environ.setdefault("VLLM_GLM53_MK_MLA","1")
sys.path.insert(0,"/usr/local/lib/python3.12/dist-packages")
import torch
spec=importlib.util.spec_from_file_location("mk","/repo/overlay/modules/glm53_megakernel/glm53_megakernel.py")
mk=importlib.util.module_from_spec(spec); spec.loader.exec_module(mk); mk._build()
dev="cuda"; D,H,W=mk.MLA_D,mk.MLA_H,2048; NS=300000
torch.manual_seed(0)
cache=(torch.randn(NS,D,device=dev)*0.4).to(torch.float8_e4m3fn)
def us(fn,n=3,reps=5):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(reps*n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/(reps*n)*1e6
print("splits policy:", {T: mk.mla_splits(T) for T in (8,32,64,128,1024,8192)})
for T in (256, 2048, 8192):
    q=(torch.randn(T,H,D,device=dev)*0.3).to(torch.bfloat16)
    slots=torch.randint(0,NS,(T,W),dtype=torch.int32,device=dev)
    lens=torch.randint(W//2,W+1,(T,),dtype=torch.int32,device=dev)
    t=us(lambda: mk.mla_decode(q,cache.view(torch.uint8),slots,lens,D**-0.5,1.0))
    got=mk.mla_decode(q,cache.view(torch.uint8),slots,lens,D**-0.5,1.0)
    sub=min(T,64)
    ref=mk.mla_decode_ref(q[:sub],cache,slots[:sub],lens[:sub],D**-0.5,1.0)
    torch.cuda.synchronize()
    rel=mk._rel_err(got[:sub].float(), ref.float())
    gb=float(lens.sum().item())*D/(t*1e-6)/1e9
    print(f"T={T:5d}: {t/1000:8.2f} ms  {gb:5.0f} GB/s  rel(vs torch twin) {rel:.2e}  splits={mk.mla_splits(T)}")
