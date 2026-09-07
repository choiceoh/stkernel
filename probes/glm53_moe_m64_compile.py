#!/usr/bin/env python3
"""Compile actual M128/M64 dispatcher paths for SM121 without CUDA access.

This uses the repository's existing fake-pointer compile API. It proves tracing,
TMA/layout consistency and ptxas compilation only, never numerics or speed.
"""
import json
import os
import time
os.environ.update(CUTE_DSL_ARCH='sm_121a',VLLM_GLM53_B12X_PREFILL_M64='1',
                  VLLM_GLM53_B12X_PREFILL_REUSE='0',VLLM_GLM53_B12X_PREFILL_FC1_N128='0')
import torch
# Match b12x_static_compile_check.py: target metadata, no real CUDA context.
torch.cuda.is_available=lambda:True
torch.cuda.get_device_capability=lambda *a,**k:(12,1)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md
md.get_num_sm=lambda dev=None:48
md.get_max_active_clusters=lambda n=1:48


def main():
    records=[]
    for tile_m in (128,64):
        md._DYNAMIC_KERNEL_CACHE.clear();start=time.monotonic()
        record=dict(tile_m=tile_m,compiled=False,gpu_execution=False)
        try:
            compiled,mac=md._get_dynamic_kernel(288,8192,4096,512,8,
                (65536//tile_m+287)*tile_m,tile_m=tile_m,tiled=True,
                activation='swigluoai_uninterleave',swiglu_alpha=1.,swiglu_beta=0.,swiglu_limit=10.)
            if tile_m==64 and not any('glm53_prefill_m64_v3' in key for key in md._DYNAMIC_KERNEL_CACHE):
                raise RuntimeError('requested M64 port did not compile')
            record.update(compiled=True,mac=mac)
        except Exception as exc:
            record.update(error_type=type(exc).__name__,error=str(exc))
        record['elapsed_s']=time.monotonic()-start
        records.append(record);print(json.dumps(record),flush=True)
    passed=all(r['compiled'] for r in records)
    print(json.dumps(dict(verdict='M64_CPU_COMPILE_PASS' if passed else 'M64_CPU_COMPILE_FAIL',results=records)),flush=True)
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
