"""Compile packet consumers for SM121 with GPUs hidden."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('offline compile requires hidden GPUs')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.draft_reduce import _mix, _mix_add_norm
    cells = []
    for group, taps in ((256, 2), (16, 2), (64, 4)):
        groups = 4096 // group
        for normalized, fn in ((False, _mix), (True, _mix_add_norm)):
            constants = dict(WIDTH=4096, BLOCK=8, GROUP=group, T=taps,
                             DR=2*taps*groups, DT=groups, DG=1, BC=4096 if normalized else 512)
            pointers = ['DELTA', 'BASE', 'OUT']
            if normalized:
                constants.update(SR=4104, EPS=1e-6)
                pointers += ['RES', 'W', 'TOTAL']
            signature = dict(PACKETS='*i64', **{name: '*bf16' for name in pointers})
            kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                     target=GPUTarget('cuda', 121, 32),
                                     options=dict(num_warps=8 if normalized else 4))
            cells.append(dict(group=group, taps=taps, normalized=normalized,
                              shared_bytes=kernel.metadata.shared,
                              ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest(),
                              cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    assert not torch.cuda.is_initialized()
    report = dict(status='PASS', gpu_used=False, target='sm_121', cells=cells,
                  source_sha256=hashlib.sha256((root/'engine/kernels/draft_reduce.py').read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
