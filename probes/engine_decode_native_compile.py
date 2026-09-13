"""Build the complete native dense extension and report mHC resources, no GPU."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def mhc_resources(resources):
    # CUDA 13 prints "Function <symbol>:", while older dumps used
    # "Function : <symbol>". Match the symbol before inspecting its usage.
    return [dict(kernel=name.strip().rstrip(':'), usage=usage.strip())
            for name, usage in re.findall(r'^\s*Function\s+(?::\s*)?([^\n]+)\n([^\n]*)',
                                           resources, flags=re.MULTILINE)
            if 'mk_mhc_ar_' in name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('this compile gate requires CUDA_VISIBLE_DEVICES=')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from torch.utils.cpp_extension import load
    from engine.kernels.native_cache import prepare_sources
    if torch.cuda.is_initialized():
        raise RuntimeError('a GPU was already initialized')
    directory = root / 'engine/kernels/dense'
    tree = ast.parse((directory / '__init__.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extension')
    flags_node = next(n.value for n in function.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'flags' for t in n.targets))
    flags = ast.literal_eval(flags_node)
    key, build, sources = prepare_sources(args.build_root, [directory / 'kernels.cu'],
                                          (flags, torch.__version__, torch.version.cuda))
    extension = load(name='st_dense_' + key, sources=list(sources), extra_cuda_cflags=flags,
                     build_directory=str(build), verbose=False)
    for name in ('run_mhc', 'run_gemm', 'run_smlp2'):
        if not callable(getattr(extension, name, None)):
            raise RuntimeError(f'the full extension did not bind {name}')
    if torch.cuda.is_initialized():
        raise RuntimeError('extension loading unexpectedly initialized a GPU')
    resources = subprocess.check_output(['cuobjdump', '--dump-resource-usage', extension.__file__], text=True)
    sections = mhc_resources(resources)
    if not sections:
        raise RuntimeError('the native mHC kernels were not emitted')
    report = dict(status='PASS', scope='full CUDA/Torch build and resources only; GPU numerics and timing pending',
                  gpu_used=False, torch=torch.__version__, cuda=torch.version.cuda, flags=flags,
                  extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256((directory / 'kernels.cu').read_bytes()).hexdigest(),
                  mhc_resources=sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.with_suffix('.resources.txt').write_text(resources)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
