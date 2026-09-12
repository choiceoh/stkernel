"""Compile the new integer decode kernels for SM121 without a CUDA device."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from triton.compiler import ASTSource
    from triton.backends.compiler import GPUTarget
    from engine.kernels.decode_commit import _advance
    from engine.kernels.vocab_candidates import _argmax_partials, _argmax_finish
    variants = []
    for k in (0, 1, 5, 8):
        for accepted in (False, True):
            signature = {name: '*i64' for name in _advance.arg_names[:_advance.arg_names.index('K')]}
            signature.update(ALIVE='*i1', DONE='*i1')
            constants = dict(K=k, E=3, PICK_STRIDE=k+1, DRAFT_STRIDE=k,
                             HAS_ACCEPTED=accepted, BT=triton.next_power_of_2(k+1), BE=4)
            variants.append((f'commit-{k}-{int(accepted)}', _advance, signature, constants))
    for dtype in ('bf16', 'fp16', 'fp32'):
        variants.append((f'argmax-{dtype}', _argmax_partials, {'X': '*'+dtype, 'OUT': '*i64'},
                         dict(ROW_STRIDE=38720, COL_STRIDE=1, VALID=38720, START=3*38720, PARTS=38, BLOCK=1024)))
    variants.append(('argmax-finish', _argmax_finish, {'PARTIALS':'*i64', 'OUT':'*i64'}, dict(PARTS=38, BLOCK=64)))
    report = dict(evidence='device compilation only', gpu_used=False, triton=triton.__version__, variants=[])
    args.output.mkdir(parents=True, exist_ok=True)
    for name, fn, signature, constants in variants:
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                target=GPUTarget('cuda', 121, 32), options={'num_warps':4})
        (args.output/(name+'.ptx')).write_text(kernel.asm['ptx'])
        (args.output/(name+'.cubin')).write_bytes(kernel.asm['cubin'])
        report['variants'].append(dict(name=name, shared_bytes=kernel.metadata.shared,
                                       cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    assert not torch.cuda.is_initialized()
    report['source_sha256'] = {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in
        ('engine/kernels/decode_commit.py', 'engine/kernels/vocab_candidates.py')}
    report['status'] = 'PASS'
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
