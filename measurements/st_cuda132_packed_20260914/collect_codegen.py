"""Inspect already-built native binaries and the CPU image's DeepGEMM headers.

No compilation, CUDA context, kernel execution or timing comparison occurs.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('inspection requires CUDA hidden and no device nodes')
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from probes.engine_moe_sf6_compile import instruction_opcodes
    records = []
    for name in ('dense', 'mla'):
        build = json.loads((args.output_dir / (name + '.json')).read_text())
        binary = Path(build['binary'])
        if sha(binary) != build['binary_sha256']:
            raise RuntimeError('compiled binary changed since its compile receipt')
        sass = subprocess.check_output(['cuobjdump', '--dump-sass', str(binary)], text=True)
        count = Counter(instruction_opcodes(sass))['VIADD.U8x4']
        if not count:
            raise RuntimeError('native CUDA extension has no packed-byte additions')
        records.append(dict(extension=name, binary_sha256=sha(binary),
                            sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),
                            native_byte_add_sites=count))
    package = Path(importlib.util.find_spec('deep_gemm').origin).parent
    header = package / 'include/deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh'
    lines = header.read_text().splitlines()
    needles = ('kSplitKFactor = 1', '// SF-MAJOR PATH:',
               'sm120_mma::fp8_mma_block_scaled', '// persistent loop')
    snippets = {}
    for needle in needles:
        at = next(i for i, line in enumerate(lines) if needle in line)
        snippets[needle] = dict(line=at+1, text=lines[at].strip())
    source_files = ('b12x/moe_dispatch.py', 'b12x/moe_w4a16_fp4_helpers.py',
                    'b12x/moe_static_kernel_v4.py', 'b12x/moe_dynamic_gated_sf6_words.py',
                    'dense/kernels.cu', 'mla/glm53_megakernel.cu')
    report = dict(gpu_used=False, status='PASS', native=records,
                  source_sha256={name: sha(root / 'engine/kernels' / name) for name in source_files},
                  deep_gemm=dict(header=str(header.relative_to(package)), sha256=sha(header),
                                 evidence=snippets,
                                 scope='available implementation features; no dispatch or latency claim'),
                  scope='native instruction emission only; no performance verdict')
    (args.output_dir / 'codegen.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
