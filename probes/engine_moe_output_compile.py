"""Compile tensor finalization and the full native packet path without CUDA access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.moe_output import _finish
    records = []
    for rows in (1, 8, 16, 24, 32):
        source = ASTSource(fn=_finish,
                           signature=dict(Acc='*fp32', Shared='*bf16', Destination='*bf16'),
                           constexprs=dict(COUNT=rows*4096, BLOCK=256))
        kernel = triton.compile(source, target=GPUTarget('cuda', 121, 32),
                                 options=dict(num_warps=4, enable_fp_fusion=False))
        ptx = kernel.asm['ptx']
        if 'cvt.rn.bf16' not in ptx or 'add.rn.f32' not in ptx:
            raise RuntimeError('compiler removed an explicit rounding boundary')
        records.append(dict(rows=rows, shared_bytes=kernel.metadata.shared,
                            cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest(),
                            ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest()))
    from engine.kernels.oneshot import build
    from tests.test_engine_direct_producer_cuda import build_oracle
    native = {name: dict(module=module.__name__,
                        sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
              for name, module in (('oneshot', build()), ('producer_oracle', build_oracle()))}
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    names = ('engine/kernels/moe_output.py', 'engine/kernels/b12x/b12x_moe.py',
             'engine/kernels/b12x/moe_dispatch.py', 'engine/profiles/glm53/lanes.py',
             'engine/kernels/dense/shared_mlp.py', 'engine/profiles/glm53/net.py',
             'engine/profiles/glm53/direct_mhc.py', 'engine/kernels/oneshot/__init__.py',
             'engine/kernels/oneshot/dsv4_oneshot_ar.cu', 'probes/oneshot_producer_oracle.cu')
    result = dict(status='PASS', gpu_used=False, scope='SM121 compilation only; GPU numerics/timing pending',
                  torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__,
                  kernels=records, native=native, source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in names})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
