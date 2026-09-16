"""Compile/load the cuBLASLt binding and lower MX producers without a GPU."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('offline compile requires hidden GPUs and no device nodes')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from engine.kernels.dense.cublaslt import _build
    from engine.kernels.dense.mxfp8 import _quantize, _quantize_bound, _pack_weights
    from engine.kernels.dense.cublaslt_split import _quantize as _split_quantize, _reduce as _split_reduce, _reduce_norm
    from engine.kernels.dense.cublaslt_serving import _activation_scales
    from engine.kernels.prefill_collectives.consumer import _quantize_gather_mx, _quantize_gather_mx_bound
    started = time.monotonic()
    native = _build()
    kernels = []
    variants = [
        (f'quantize-k{k}-tiled{int(tiled)}-pad{int(pad)}', _quantize,
         dict(X='*bf16', Q='*fp8e4nv', S='*i32', M='i32'), dict(K=k, G=k//128, TILED=tiled, PAD=pad))
        for k in (128, 384, 4096, 20480) for tiled in (False, True) for pad in (False, True)]
    variants += [('weight-scales', _pack_weights, dict(S='*fp32', Out='*i32'), {}),
                 *[(f'packet-mx-pad{int(pad)}', _quantize_gather_mx,
                  dict(Packed='*fp8e4nv', Scales='*fp32', Q='*fp8e4nv', S='*i32', M='i32',
                       LOCAL_N='i32', PAYLOAD_BYTES='i32'), dict(K=4096, G=32, PACK_BLOCK=2048, TILED=True, PAD=pad))
                   for pad in (False, True)]]
    variants += [(f'bound-m{m}-k{k}', _quantize_bound, dict(X='*bf16', Q='*fp8e4nv', S='*i32'), dict(M=m, K=k))
                 for m, k in ((1, 4096), (8, 4096), (8, 20480), (128, 4096), (129, 4096))]
    variants += [(f'bound-packet-m{m}', _quantize_gather_mx_bound,
                  dict(Packed='*fp8e4nv', Scales='*fp32', Q='*fp8e4nv', S='*i32'),
                  dict(M=m, LOCAL_N=local*4096, PAYLOAD_BYTES=((local*4096 + local*8 + 127)//128)*128, PACK_BLOCK=2048))
                 for m, local in ((128, 32), (129, 33))]
    configurations = [(name, fn, signature, constants, warps)
                      for name, fn, signature, constants in variants
                      for warps in ((4,) if name == 'weight-scales' else (1, 2, 4))]
    configurations += [(f'split-producer-m{m}', _split_quantize, dict(X='*bf16', Q='*fp8e4nv', S='*i32'),
                        dict(M=m, K=20480, P=5, PAD=False), 1) for m in (8, 16)]
    configurations += [(f'split-reduction-m{m}', _split_reduce, dict(X='*fp32', Y='*bf16'),
                        dict(SIZE=m*4096, P=5), 4) for m in (8, 16)]
    configurations += [(f'serving-split-producer-m{m}', _split_quantize,
                        dict(X='*bf16', Q='*fp8e4nv', S='*i32'), dict(M=m, K=20480, P=5, PAD=True), 1)
                       for m in (1, 7, 8, 16, 24, 32, 64)]
    configurations += [(f'activation-scales-m{m}-g{g}', _activation_scales, dict(S='*fp32', MX='*i32'),
                        dict(M=m, G=g), 1) for m in (7, 128, 512) for g in (32, 160)]
    configurations += [(f'split-norm-m{m}-bias{int(bias)}', _reduce_norm,
                        dict(X='*fp32', W='*bf16', BIAS='*fp32' if bias else '*bf16', Y='*bf16'),
                        dict(M=m, N=4096, P=5, EPS=1e-6, HAS_BIAS=bias, BN=4096), 8)
                       for m in (1, 7, 8, 16, 24, 32, 64) for bias in (False, True)]
    for name, fn, signature, constants, warps in configurations:
        if 'SCALAR_SCALE' in fn.arg_names:
            constants = dict(constants, SCALAR_SCALE=warps == 1)
        kernel = triton.compile(ASTSource(fn, signature, constexprs=constants),
                                target=GPUTarget('cuda', 121, 32), options=dict(num_warps=warps))
        ptx = kernel.asm['ptx']
        expensive = re.findall(r'\b(?:lg2|ex2|div|rcp)\.[\w.]*f32', ptx)
        # The FP8 producers avoid log/exp/div; RMS keeps the established IEEE
        # division (also present in common.norm_rope) to preserve its rounding.
        if name != 'weight-scales' and not name.startswith('split-norm-') and expensive:
            raise RuntimeError(f'{name} still has log/exp/div/reciprocal instructions: {expensive}')
        integer_divisions = re.findall(r'\b(?:div|rem)\.[su]32', ptx)
        if name.startswith('bound-') and integer_divisions:
            raise RuntimeError(f'{name} kept dynamic integer division: {integer_divisions}')
        barriers = re.findall(r'\bbar\.(?:warp\.)?sync\b', ptx)
        if warps == 1 and (kernel.metadata.shared or barriers):
            raise RuntimeError(f'{name} retained shared storage or a CTA barrier with one warp')
        kernels.append(dict(name=name, num_warps=warps, shared_bytes=kernel.metadata.shared,
                            cta_barriers=len(barriers), expensive_arithmetic=expensive,
                            integer_divisions=integer_divisions,
                            ptx_sha256=hashlib.sha256(kernel.asm['ptx'].encode()).hexdigest(),
                            cubin_sha256=hashlib.sha256(kernel.asm['cubin']).hexdigest()))
    if torch.cuda.is_initialized():
        raise RuntimeError('binding build initialized CUDA')
    binary = Path(native.__file__)
    report = dict(status='PASS', gpu_used=False, seconds=time.monotonic()-started,
                  torch=torch.__version__, cuda=torch.version.cuda, cublaslt=native.version(),
                  binary=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  triton=triton.__version__, kernels=kernels,
                  scope='host binding compile/dlopen and SM121 producer compilation; no cuBLAS GPU execution')
    root = Path(__file__).resolve().parents[1]
    paths = ('engine/kernels/dense/cublaslt.cpp', 'engine/kernels/dense/cublaslt.py',
             'engine/kernels/dense/mxfp8.py', 'engine/kernels/dense/cublaslt_split.py',
             'engine/kernels/dense/cublaslt_serving.py', 'engine/kernels/dense/fp8.py',
             'engine/kernels/prefill_collectives/consumer.py', 'probes/engine_cublaslt_compile.py')
    report['source_sha256'] = {p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
