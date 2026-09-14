"""Compare actual MLA decode/prefill code under one CUDA toolkit without a GPU.

Run this separately with CUDA 13.0 and 13.2. The extracted conversion probe
also compiles the adjacent, strided and packed helpers from the serving file.
Instruction counts are static code-generation evidence, never latency results.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('engine/kernels/mla/glm53_megakernel.cu'))
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--cuda-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('CPU compile requires CUDA hidden and no NVIDIA device nodes')
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location('mla_body', Path(__file__).with_name('engine_mla_hardware_check.py'))
    body = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(body)
    nvcc, objdump = (args.cuda_root/'bin'/name for name in ('nvcc', 'cuobjdump'))
    version = subprocess.check_output([str(nvcc), '--version'], text=True).strip()
    report = dict(gpu_used=False, target='sm_121a', nvcc=version,
                  harness_sha256=digest(Path(__file__)), extraction_sha256=digest(Path(body.__file__)),
                  scope='device compile and static instructions; no GPU numerics or timing', variants={})
    for name, path in (('baseline', args.baseline), ('candidate', args.source)):
        source = path.read_text()
        prefill = source[source.index('// Large-M prefill:'):source.index('void mk_run_mla_prefill32(')]
        cu = args.output/(name+'.cu')
        cu.write_text(body.cuda_source(source) + '\n' + prefill)
        for extension, mode in (('ptx', '--ptx'), ('cubin', '--cubin')):
            command = [str(nvcc), '-O2', '-std=c++17', '-arch=sm_121a', mode,
                       '--ptxas-options=-v', str(cu), '-o', str(cu.with_suffix('.'+extension))]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            log = result.stdout+result.stderr
            (args.output/(name+'-'+extension+'.log')).write_text(log)
            if result.returncode:
                raise RuntimeError(log)
        ptx, cubin = cu.with_suffix('.ptx'), cu.with_suffix('.cubin')
        sass = subprocess.check_output([str(objdump), '--dump-sass', str(cubin)], text=True)
        (args.output/(name+'.sass')).write_text(sass)
        kernels = {}
        for symbol, instructions in re.findall(r'Function : (\S+)\n(.*?)(?=Function :|\Z)', sass, re.S):
            label = ('prefill32' if 'mk_mla_prefill32_kernel' in symbol else
                     'cluster' if 'mk_mla_kernelILb1E' in symbol else
                     'decode' if 'mk_mla_kernelILb0E' in symbol else
                     'conversion' if 'convert_pairs' in symbol else None)
            if label is None:
                continue
            resource = re.search(r"Compiling entry function '"+re.escape(symbol)+r"'.*?(?=ptxas info\s+: Compiling entry function|\Z)", log, re.S).group()
            spills = re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', resource)
            opcodes = Counter(re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?[A-Z]+\d*\s+)?([A-Z][A-Z0-9_]*)(?=[.;\s])', instructions))
            kernels[label] = dict(registers=int(re.search(r'Used (\d+) registers', resource).group(1)),
                spill_store_bytes=int(spills.group(1)), spill_load_bytes=int(spills.group(2)),
                static_instruction_sites=sum(opcodes.values()), opcode_sites=dict(sorted(opcodes.items())))
        if set(kernels) != {'decode', 'cluster', 'prefill32', 'conversion'}:
            raise RuntimeError('missing compiled serving body: '+str(kernels.keys()))
        report['variants'][name] = dict(source_sha256=digest(path), extracted_sha256=digest(cu),
            cubin_sha256=digest(cubin), ptx_sha256=digest(ptx), kernels=kernels,
            native_bf16_pair_sites=ptx.read_text().count('cvt.rn.bf16x2.e4m3x2'))
        print(name, json.dumps({k: {a:b for a,b in v.items() if a!='opcode_sites'} for k,v in kernels.items()}), flush=True)
    (args.output/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
