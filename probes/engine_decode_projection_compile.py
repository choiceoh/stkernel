"""Compile Triton projection/reduction kernels for SM121 with no device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.decode_projection import _kda_pair
    from probes.engine_moe_scatter import _reduce_routes
    records = []
    for rows in (1, 6, 7, 14, 21, 28):
        for stride0, stride1 in ((6416, 6416), (128, 6416)):
            source = ASTSource(fn=_kda_pair, signature={name: '*bf16' for name in ('X0', 'X1', 'W0', 'W1', 'Y')},
                               constexprs=dict(M=rows, XS0=stride0, XS1=stride1, BM=16, BN=64))
            kernel = triton.compile(source, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
            records.append(dict(kernel='kda_pair', rows=rows, strides=[stride0, stride1],
                                shared_bytes=kernel.metadata.shared, status='PASS'))
    source = ASTSource(fn=_reduce_routes, signature=dict(Partial='*fp32', Out='*fp32'),
                       constexprs=dict(K=4096, PARTS=32, BLOCK=128))
    kernel = triton.compile(source, target=GPUTarget('cuda', 121, 32),
                           options=dict(num_warps=4, enable_fp_fusion=False))
    records.append(dict(kernel='route_reduce', shared_bytes=kernel.metadata.shared, status='PASS'))
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    report = dict(status='PASS', gpu_used=False, kernels=records,
                  scope='native PTXAS compile only; GPU numerics and timing pending',
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ('engine/kernels/decode_projection.py', 'probes/engine_moe_scatter.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
