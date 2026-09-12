"""Compile actual MLA bodies and report SM121 instructions without a GPU.

This uses the same extraction as the GPU comparison probe. It validates device
compilation, not the Torch binding, GPU numerical results, or performance.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('engine/kernels/mla/glm53_megakernel.cu'))
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--cuda-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('mla_gpu_probe', Path(__file__).with_name('engine_mla_hardware_check.py'))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    nvcc = args.cuda_root / 'bin/nvcc'
    version = subprocess.check_output([str(nvcc), '--version'], text=True).strip()
    report = {'gpu_used': False, 'target': 'sm_121a', 'nvcc': version,
              'harness_sha256': sha256(Path(__file__)),
              'extraction_sha256': sha256(Path(probe.__file__)), 'variants': {}}
    variants = [('baseline', args.baseline)] if args.baseline else []
    variants.append(('current', args.source))
    for name, source in variants:
        cu, cubin = args.output / (name + '.cu'), args.output / (name + '.cubin')
        cu.write_text(probe.cuda_source(source.read_text()).rstrip() + '\n')
        command = [str(nvcc), '-O2', '-std=c++17', '-arch=sm_121a', '--cubin',
                   '--ptxas-options=-v', str(cu), '-o', str(cubin)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
        log = compiled.stdout + compiled.stderr
        (args.output / (name + '-nvcc.log')).write_text(log)
        if compiled.returncode:
            raise RuntimeError(log)
        sass = subprocess.check_output([str(args.cuda_root / 'bin/cuobjdump'), '--dump-sass', str(cubin)], text=True)
        (args.output / (name + '.sass')).write_text(sass)
        kernels = {}
        for function, body in re.findall(r'Function : (\S+)\n(.*?)(?=Function :|\Z)', sass, re.S):
            if not function.startswith('_Z13mk_mla_kernel'):
                continue
            opcodes = Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+\d*\s+)?([A-Z][A-Z0-9_]*)(?=[.;\s])', body))
            resources = re.search(r"Compiling entry function '" + re.escape(function) + r"'.*?(?=ptxas info\s+: Compiling entry function|\Z)", log, re.S).group()
            stores, loads = map(int, re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', resources).groups())
            kernels['cluster' if 'ILb1E' in function else 'ordinary'] = {
                'registers': int(re.search(r'Used (\d+) registers', resources).group(1)),
                'spill_store_bytes': stores, 'spill_load_bytes': loads,
                'static_instruction_sites': sum(opcodes.values()),
                'opcode_sites': dict(sorted(opcodes.items())),
            }
        assert set(kernels) == {'ordinary', 'cluster'}, kernels
        report['variants'][name] = {'source_sha256': sha256(source),
                                    'extracted_sha256': sha256(cu),
                                    'cubin_sha256': sha256(cubin),
                                    'command': command, 'kernels': kernels}
        print(name, json.dumps({key: {k:v for k,v in values.items() if k != 'opcode_sites'} for key,values in kernels.items()}), flush=True)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
