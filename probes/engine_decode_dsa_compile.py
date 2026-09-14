"""No-GPU full dense extension build and DSA/head-gate SM121 PTXAS compile."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--build-dir', type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('compile needs CUDA_VISIBLE_DEVICES= and NVIDIA_VISIBLE_DEVICES=void')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from torch.utils.cpp_extension import load
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.mla.decode_inputs import _latent_norm_write
    from engine.kernels.indexer_gate import _gate_partials
    from engine.kernels.decode_projection import _indexer_boundary
    from engine.kernels.common.native_cache import prepare_sources
    python_source = root / 'engine/kernels/dense/__init__.py'
    body = next(n for n in ast.parse(python_source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'extension')
    flags = next(ast.literal_eval(n.value) for n in body.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'flags' for t in n.targets))
    key, directory, sources = prepare_sources(args.build_dir, [root / 'engine/kernels/dense/kernels.cu'],
                                              (flags, torch.__version__, torch.version.cuda))
    load(name='st_dense_' + key, sources=list(sources), extra_cuda_cflags=flags,
         build_directory=str(directory), verbose=True)
    records = []
    for xs, ls, ts0, ts1, offset in ((2048, 512, 176, 1, 0), (2056, 520, 352, 2, 7680)):
        src = ASTSource(fn=_latent_norm_write, signature=dict(X='*bf16', W='*bf16', LATENT='*fp8e4nv',
                        TABLE='*i32', CTX='*i64', EPS='fp32'),
                        constexprs=dict(XS=xs, LS=ls, TS0=ts0, TS1=ts1, BLOCK=768,
                                        BLOCK_STRIDE=8448, OFFSET=offset, TOKENS=8, D=512))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=4))
        records.append(dict(input_stride=xs, latent_stride=ls, block=768, block_stride=8448,
                            layer_offset=offset, shared_bytes=kernel.metadata.shared,
                            status='PASS'))
    head_records = []
    for stride in (4096, 4104):
        src = ASTSource(fn=_gate_partials, signature=dict(X='*bf16', W='*fp32', P='*fp32'), constexprs=dict(XS=stride))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                               options=dict(num_warps=4, enable_fp_fusion=False))
        head_records.append(dict(kernel='gate_partials', input_stride=stride, shared_bytes=kernel.metadata.shared,
                                 status='PASS'))
    for splits in (1, 16):
        src = ASTSource(fn=_indexer_boundary,
                        signature=dict(Q='*bf16', K='*bf16', W='*fp32', NW='*fp32', NB='*fp32',
                                       Q8='*fp8e4nv', KO='*bf16', WE='*fp32'),
                        constexprs=dict(NH=32, KS=256, SCALE=128 ** -.5 * 32 ** -.5, HEAD_SPLITS=splits))
        kernel = triton.compile(src, target=GPUTarget('cuda', 121, 32),
                               options=dict(num_warps=1, enable_fp_fusion=False))
        head_records.append(dict(kernel='indexer_boundary', head_splits=splits, shared_bytes=kernel.metadata.shared,
                                 status='PASS'))
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    files = ('engine/kernels/dense/kernels.cu', 'engine/kernels/dense/query_pair.py', 'engine/kernels/mla/decode_inputs.py',
             'engine/kernels/indexer_gate.py', 'engine/kernels/decode_projection.py')
    result = dict(status='PASS', gpu_used=False, torch=torch.__version__, triton=triton.__version__,
                  cuda=torch.version.cuda, flags=flags, native_cache_key=key, latent_kernels=records, head_gate_kernels=head_records,
                  source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
                  scope='full native build and PTXAS only; GPU bytes/replay/timing pending')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
