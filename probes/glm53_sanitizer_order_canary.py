"""Deliberately bad, isolated CuTe kernels verify sanitizer detection after driver-first init."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUTE_DSL_ARCH','sm_121a')


def kernels():
    import cutlass
    from cutlass import cute
    import cutlass.utils

    @cute.kernel
    def out_of_bounds(output:cute.Tensor):
        tid,_,_=cute.arch.thread_idx()
        output[tid+64]=cutlass.Float32(tid)

    @cute.kernel
    def shared_race(output:cute.Tensor):
        tid,_,_=cute.arch.thread_idx()
        shared=cutlass.utils.SmemAllocator().allocate_tensor(cutlass.Float32,cute.make_layout((1,)),byte_alignment=4)
        shared[0]=cutlass.Float32(tid)
        cute.arch.sync_threads()
        output[tid]=shared[0]

    @cute.jit
    def launch_memory(output:cute.Tensor):
        out_of_bounds(output).launch(grid=(1,1,1),block=(32,1,1))

    @cute.jit
    def launch_race(output:cute.Tensor):
        shared_race(output).launch(grid=(1,1,1),block=(64,1,1))
    return launch_memory,launch_race


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compile-only',action='store_true')
    ap.add_argument('--tool',choices=('memcheck','racecheck'))
    args=ap.parse_args()
    if not args.compile_only:
        if os.environ.get('GLM53_SANITIZER_CANARY')!='1' or not args.tool:
            ap.error('only the isolated owned sanitizer runner may execute these deliberately bad kernels')
        from cuda.bindings import driver
        for name,arguments in (('cuDriverGetVersion',()),('cuInit',(0,)),('cuDeviceGetCount',())):
            result=getattr(driver,name)(*arguments);assert int(result[0])==0
        # Exact device allocations make the intended OOB boundary observable.
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING']='1'
    import cutlass
    from cutlass import cute
    memory,race=kernels()
    if args.compile_only:
        fake=cute.runtime.make_fake_compact_tensor(cutlass.Float32,(64,),stride_order=(0,),assumed_align=16)
        for function in (memory,race):cute.compile(function,fake)
        print(json.dumps(dict(kind='SANITIZER_CANARY_CPU_COMPILE_PASS',gpu_execution=False)),flush=True)
        return
    import torch
    from cutlass.cute.runtime import from_dlpack
    value=torch.zeros(64,device='cuda',dtype=torch.float32)
    tensor=from_dlpack(value,assumed_align=16)
    compiled=cute.compile(memory if args.tool=='memcheck' else race,tensor)
    print(json.dumps(dict(kind='INTENTIONAL_BAD_KERNEL_BEGIN',tool=args.tool,driver_first=True)),flush=True)
    try:
        compiled(tensor);torch.cuda.synchronize()
        error=None
    except RuntimeError as exc:
        error=str(exc)
    print(json.dumps(dict(kind='INTENTIONAL_BAD_KERNEL_END',tool=args.tool,error=error,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        serving_gate=False,numerical_acceptance=False)),flush=True)


if __name__=='__main__':main()
