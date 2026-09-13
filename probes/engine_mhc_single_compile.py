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
    if 'single_token_grid' not in (extension.run_mhc.__doc__ or ''):
        raise RuntimeError('the full extension did not bind the candidate entry point')
    for entry, keywords in ((extension.run_gemm, ('pack_rows', 'short_input')),
                            (extension.run_smlp2, ('direct_down',))):
        if any(word not in (entry.__doc__ or '') for word in keywords):
            raise RuntimeError('the full extension did not bind the batched decode candidates')
    if torch.cuda.is_initialized():
        raise RuntimeError('extension loading unexpectedly initialized a GPU')
    resources = subprocess.check_output(['cuobjdump', '--dump-resource-usage', extension.__file__], text=True)
    sections = mhc_resources(resources)
    if not any('mk_mhc_ar_single_kernel' in section['kernel'] for section in sections):
        raise RuntimeError('the candidate kernel was not emitted')
    decode_resources = [dict(kernel=name.strip().rstrip(':'), usage=usage.strip())
                        for name, usage in re.findall(r'^\s*Function\s+(?::\s*)?([^\n]+)\n([^\n]*)',
                                                       resources, flags=re.MULTILINE)
                        if 'mk_input_pack_kernel' in name or 'mk_gemm_input_cta3_kernel' in name]
    if len(decode_resources) != 8:
        raise RuntimeError(f'expected three pack and five CTA instantiations, got {len(decode_resources)}')
    report = dict(status='PASS', scope='full CUDA/Torch build and resources only; GPU numerics and timing pending',
                  gpu_used=False, torch=torch.__version__, cuda=torch.version.cuda, flags=flags,
                  extension_sha256=hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256((directory / 'kernels.cu').read_bytes()).hexdigest(),
                  mhc_resources=sections, decode_resources=decode_resources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.with_suffix('.resources.txt').write_text(resources)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
