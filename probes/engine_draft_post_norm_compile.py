"""Compile post-convolution residual RMS for SM121 without GPU access."""
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
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('offline compile requires hidden GPUs and no device nodes')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.draft_conv import _taps_add_norm
    cells = []
    for width, block, group, taps in ((4096, 6, 256, 2), (4096, 7, 256, 2),
                                      (4096, 8, 256, 2), (4096, 8, 16, 2),
                                      (4096, 8, 64, 4), (32, 7, 8, 2), (32, 8, 4, 2)):
        groups = width // group
        constants = dict(sX=width, sDr=2*taps*groups, sDt=groups, sDg=1,
                         sR=width, sO=width, width=width, BLOCK=block, GROUP=group,
                         T=taps, BC=triton.next_power_of_2(width), EPS=1e-6)
        signature = {name: '*bf16' for name in ('X', 'DELTA', 'BASE', 'RES', 'W', 'TOTAL', 'OUT')}
        kernel = triton.compile(ASTSource(_taps_add_norm, signature, constexprs=constants),
                                target=GPUTarget('cuda', 121, 32),
                                options=dict(num_warps=4 if width <= 1024 else 8))
        cells.append(dict(width=width, block=block, group=group, taps=taps,
                          shared_bytes=kernel.metadata.shared,
                          ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest(),
                          cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    if torch.cuda.is_initialized():
        raise RuntimeError('offline compile initialized CUDA')
    paths = ('engine/kernels/draft_conv.py', 'probes/engine_draft_post_norm_compile.py')
    report = dict(status='PASS', gpu_used=False, target='sm_121', torch=torch.__version__,
                  cuda=torch.version.cuda, triton=triton.__version__, cells=cells,
                  source_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
