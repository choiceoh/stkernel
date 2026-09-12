"""CPU-only shared-memory layout audit for the current seven-row MoE tile."""
import json
import os

os.environ.setdefault('CUTE_DSL_ARCH', 'sm_121a')
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from engine.kernels.b12x.moe_static_kernel_v4 import MoEStaticKernelV4


def main():
    assert not torch.cuda.is_initialized()
    for fc1, fc2, compact in ((2, 2, False), (1, 2, False), (2, 1, False), (1, 1, False), (1, 1, True)):
        k = MoEStaticKernelV4(sf_vec_size=16, output_tile_count_n=4,
                              fc1_stages=fc1, fc2_stages=fc2, decode_reform=True,
                              decode_compact=compact,
                              reform_sf_pack=True, activation='swigluoai_uninterleave',
                              swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        k.a_dtype = k.b_dtype = cutlass.Float4E2M1FN
        k.sf_dtype = cutlass.Float8E4M3FN
        k.a_layout = k.b_layout = k.c_layout = utils.LayoutEnum.ROW_MAJOR

        @cute.jit
        def layout(dummy: cute.Tensor):
            k._setup_attributes(4096)
            print(json.dumps(dict(fc1=fc1, fc2=fc2, compact=compact, smem_bytes=k.smem_bytes,
                                  threads=k.threads_per_cta,
                                  scope='CPU layout only; no GPU residency or speed proof')))

        dummy = cute.runtime.make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
        cute.compile(layout, dummy)
    assert not torch.cuda.is_initialized()


if __name__ == '__main__':
    main()
