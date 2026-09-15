"""Compile the production dense extension without a GPU or model boot."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--forward-pipeline', action='store_true', help='also report the ordered-K and joined-query cubins')
    ap.add_argument('--rows16', action='store_true', help='also report the sixteen-row C=2 cubins')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void':
        raise RuntimeError('this compile requires CUDA hidden')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.common.native_cache import prepare_cuda_sources
    tree = ast.parse((root/'engine/kernels/dense/__init__.py').read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    flags = next(ast.literal_eval(n.value) for n in builder.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'flags' for t in n.targets))
    source = root/'engine/kernels/dense/kernels.cu'
    key, directory, sources = prepare_cuda_sources(args.build_dir, [source], (flags, torch.__version__, torch.version.cuda))
    native = load(name='st_dense_'+key, sources=list(sources), extra_cuda_cflags=flags,
                  build_directory=str(directory), verbose=True)
    assert callable(native.run_query_pair) and callable(native.run_gemm_bound_input)
    if args.rows16:
        assert callable(native.rows16_info)
    usage = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', native.__file__], text=True)
    entries = []
    for block in re.split(r'(?m)^\s*Function(?:\s+|:)', usage)[1:]:
        if 'mk_gemm_input_cta3_kernel' in block and any(f'Li{k}E' in block.splitlines()[0] for k in (12, 16, 24)):
            entries.append(block.strip())
    if len(entries) != 5:
        raise RuntimeError(f'expected five new K-block/direct specializations; got {len(entries)}')
    registers = [b.strip() for b in re.split(r'(?m)^\s*Function(?:\s+|:)', usage)[1:]
                 if re.search(r'mk_gemm_input_cta_kernelILi0ELb[01]ELb1ELi(?:16|24|32)ELi[23]E', b.splitlines()[0])] if args.forward_pipeline else []
    if args.forward_pipeline and len(registers) != 8:
        raise RuntimeError(f'expected eight ordered/direct specializations; got {len(registers)}')
    queries = [b.strip() for b in re.split(r'(?m)^\s*Function(?:\s+|:)', usage)[1:]
               if 'mk_query_pair_kernel' in b.splitlines()[0]] if args.forward_pipeline else []
    if args.forward_pipeline and len(queries) != 1:
        raise RuntimeError(f'expected one joined-query specialization; got {len(queries)}')
    rows16 = [b.strip() for b in re.split(r'(?m)^\s*Function(?:\s+|:)', usage)[1:]
              if re.search(r'mk_gemm_rows16_kernel', b.splitlines()[0])] if args.rows16 else []
    if args.rows16 and len(rows16) != 7:
        raise RuntimeError(f'expected four matrix and three TX output sixteen-row specializations; got {len(rows16)}')
    assert not torch.cuda.is_initialized()
    result = dict(status='PASS', gpu_used=False, torch=torch.__version__, cuda=torch.version.cuda,
                  cache_key=key, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  flags=flags, new_native_specializations=entries, ordered_specializations=registers,
                  query_specializations=queries, rows16_specializations=rows16,
                  scope='production native compile/load and resource usage; not GPU execution or timing')
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
