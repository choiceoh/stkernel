"""Compile direct tree key-bank copies and ring convolution for SM121 offline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia import compiler as nvidia_compiler
from triton.compiler import ASTSource

from engine.kernels.indexer import _tree_key_bank
from engine.kernels.kda.tree import _conv
from probes.engine_tree_dataflow_compile import cubin_usage


def run(output):
    if torch.cuda.is_initialized():
        raise RuntimeError('offline compilation must not inherit CUDA')
    output.mkdir(parents=True, exist_ok=True)
    signature = {p: '*u8' for p in ('KEYS', 'PRIVATE', 'OUT')}
    signature.update({p: '*i32' for p in ('SCALES', 'PRIVATE_S', 'TABLE', 'OUT_S')})
    signature.update({p: 'i32' for p in ('prefix', 'total', 'key_s0', 'scale_s0', 'private_s0',
                                       'private_scale_s0', 'table_s0', 'stride', 'offset')})
    variants = [('paged-key-bank', _tree_key_bank, signature, dict(PER=16, D=128, B=64))]
    for channels in (193, 6144):
        for ring in (0, 10, 11):
            for dtype in ('bf16', 'fp32'):
                signature = {p: '*bf16' for p in ('X', 'HISTORY', 'OUT')}
                signature.update(W='*'+dtype, PATH='*i32', context='i32')
                variants.append((f'conv-c{channels}-ring{ring}-{dtype}', _conv, signature,
                                 dict(C=channels, TAPS=4, B=256, RING=ring)))
    ptxas = Path(nvidia_compiler.__file__).parent/'bin/ptxas'
    report = dict(scope='offline SM121 code generation; no device execution or speed claim',
                  gpu_used=False, torch=torch.__version__, triton=triton.__version__,
                  ptxas=subprocess.check_output([str(ptxas), '--version'], text=True), variants=[],
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                      ('engine/kernels/indexer.py', 'engine/kernels/kda/tree.py',
                       'probes/engine_tree_bank_compile.py', 'probes/engine_tree_dataflow_compile.py')})
    for name, fn, signature, constants in variants:
        print('compile '+name, flush=True)
        # Only pointers are aligned. Page/record strides remain runtime values.
        attrs = {(fn.arg_names.index(p),): [('tt.divisibility', 16)]
                 for p, dtype in signature.items() if dtype.startswith('*')}
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
            target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4, enable_fp_fusion=fn is not _conv))
        ptx, cubin = kernel.asm['ptx'], output/(name+'.cubin')
        cubin.write_bytes(kernel.asm['cubin'])
        (output/(name+'.ptx')).write_text(ptx)
        usage = cubin_usage(cubin)
        if ('.target sm_121a' not in ptx or usage['stack_bytes'] or usage['local_instructions']
                or any(op in ptx for op in ('atom.', 'nanosleep', 'ld.local', 'st.local'))):
            raise AssertionError(f'{name}: wrong target, spills or unexpected synchronization: {usage}')
        report['variants'].append(dict(name=name, constants=constants, shared_bytes=kernel.metadata.shared,
            **usage, cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest()))
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise AssertionError('compilation initialized CUDA')
    (output/'compile.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
