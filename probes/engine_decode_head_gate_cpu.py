"""Execute the actual partial kernel in Triton's CPU interpreter.

This checks address coverage and the reduction formula, not SM121 rounding,
timing or graph replay. Run with GPUs hidden and TRITON_INTERPRET=1.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if (os.environ.get('TRITON_INTERPRET') != '1' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void'):
        raise RuntimeError('CPU interpreter requires TRITON_INTERPRET=1 with CUDA hidden')
    import torch
    from engine.kernels.indexer_gate import _gate_partials, ROWS
    torch.set_num_threads(1)
    torch.manual_seed(91432)
    w = torch.randn(32, 4096)*.02
    results = []
    for rows in ROWS:
        for stride in (4096, 4104):
            source = torch.randn(rows, stride).bfloat16()
            x = source if stride == 4096 else source[:, 4:4100]
            storage = torch.full((rows*16*32 + 32,), float('nan'))
            p = storage[16:-16].view(rows, 16, 32)
            for factor in (1., 0., -16.):
                x.copy_(torch.randn_like(x)*factor)
                p.fill_(float('nan'))
                _gate_partials[(16, 8, rows//8)](x, w, p, stride, num_warps=4, enable_fp_fusion=False)
                assert p.isfinite().all()
                assert storage[:16].isnan().all() and storage[-16:].isnan().all()
                ref = x.double() @ w.double().T
                error = ((p.double().sum(1)-ref).abs().amax(1)/ref.abs().amax(1).clamp_min(1e-12)).max().item()
                assert error < 1e-5, error
                results.append(dict(rows=rows, input_stride=stride, factor=factor, row_relative_error=error))
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(status='PASS', gpu_used=False, cases=results,
                         source_sha256=hashlib.sha256(Path('engine/kernels/indexer_gate.py').read_bytes()).hexdigest(),
                         scope='actual Triton CPU interpreter; GPU arithmetic and timing unmeasured')), flush=True)


if __name__ == '__main__':
    main()
