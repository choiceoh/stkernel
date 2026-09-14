"""Offline SM121 tree-address and verbatim MLA compilation; no CUDA context."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sysconfig

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.backends.nvidia import compiler as nvidia_compiler

from engine.kernels.indexer import _pool_slots
from engine.kernels.mla.prefill_absorb import _absorb
from probes.engine_mla_hardware_check import cuda_source


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_source(source):
    return cuda_source(source) + '''
extern "C" int launch_tree(MKMlaArgs args, cudaStream_t stream, bool cluster) {
  cudaFuncSetAttribute(mk_mla_kernel<false,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_SMEM);
  cudaFuncSetAttribute(mk_mla_kernel<true,true>, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_SMEM);
  cudaLaunchConfig_t cfg{};
  cfg.gridDim=dim3(args.grid); cfg.blockDim=dim3(MK_THREADS);
  cfg.dynamicSmemBytes=MLA_SMEM; cfg.stream=stream;
  if (cluster) return cudaLaunchKernelEx(&cfg, mk_mla_kernel<true,true>, args);
  return cudaLaunchKernelEx(&cfg, mk_mla_kernel<false,true>, args);
}
'''


def run(cuda_root, output, baseline):
    if torch.cuda.is_initialized():
        raise RuntimeError('offline compilation must not inherit CUDA')
    output.mkdir(parents=True, exist_ok=True)
    source = Path('engine/kernels/mla/glm53_megakernel.cu')
    nvcc = cuda_root/'bin/nvcc'
    dump = Path(nvidia_compiler.__file__).parent/'bin/cuobjdump'
    report = dict(gpu_used=False, scope='CPU model and SM121 code generation only',
                  nvcc=subprocess.check_output([str(nvcc), '--version'], text=True),
                  torch=torch.__version__, triton=triton.__version__, native={}, triton_kernels=[])
    native_ops = {}
    for name, code in (('baseline', cuda_source(baseline.read_text())), ('candidate', tree_source(source.read_text()))):
        cu, cubin = output/(name+'.cu'), output/(name+'.cubin')
        cu.write_text(code)
        done = subprocess.run([str(nvcc), '-O2', '-std=c++17', '-arch=sm_121a', '--cubin',
                               '--ptxas-options=-v', str(cu), '-o', str(cubin)], capture_output=True, text=True)
        log = done.stdout+done.stderr
        (output/(name+'.log')).write_text(log)
        if done.returncode:
            raise RuntimeError(log)
        sass = subprocess.check_output([str(dump), '--dump-sass', str(cubin)], text=True)
        (output/(name+'.sass')).write_text(sass)
        kernels, native_ops[name] = {}, {}
        for symbol, body in re.findall(r'Function : (\S+)\n(.*?)(?=Function :|\Z)', sass, re.S):
            if 'mk_mla_kernelI' not in symbol:
                continue
            flags = re.findall(r'Lb([01])E', symbol)
            key = ('cluster' if flags[0] == '1' else 'ordinary') + ('-tree' if flags[1:] == ['1'] else '')
            resources = re.search(r"Compiling entry function '"+re.escape(symbol)+r"'.*?(?=ptxas info\s+: Compiling entry function|\Z)", log, re.S)[0]
            stores, loads = map(int, re.search(r'(\d+) bytes spill stores, (\d+) bytes spill loads', resources).groups())
            # Compare actual encoded instruction words, not source lines or PTX.
            words = re.findall(r'/\* (0x[0-9a-f]+) \*/', body)
            native_ops[name][key] = words
            kernels[key] = dict(registers=int(re.search(r'Used (\d+) registers', resources)[1]),
                spill_store_bytes=stores, spill_load_bytes=loads,
                instruction_words=len(words), instruction_sha256=hashlib.sha256(''.join(words).encode()).hexdigest())
            if stores or loads or not words:
                raise AssertionError(f'MLA spills or missing disassembly: {name}/{key}')
        expected = {'ordinary', 'cluster'} | ({'ordinary-tree', 'cluster-tree'} if name == 'candidate' else set())
        if set(kernels) != expected:
            raise AssertionError(kernels)
        report['native'][name] = dict(source_sha256=sha(baseline if name == 'baseline' else source),
            cubin_sha256=sha(cubin), kernels=kernels)
        print(name, json.dumps(kernels), flush=True)
    report['ordinary_instruction_words_unchanged'] = all(native_ops['baseline'][k] == native_ops['candidate'][k]
                                                        for k in ('ordinary', 'cluster'))
    if not report['ordinary_instruction_words_unchanged']:
        raise AssertionError('ordinary MLA machine instructions changed')
    signatures = {p: '*i32' for p in ('ids', 'lengths', 'table', 'out', 'counts')}
    signatures['paths'] = '*i64'
    constants = dict(groups=512, id_s0=512, id_s1=1, len_s0=1, table_s0=1,
        out_s0=2051, out_s1=1, count_s0=1, block_size=64, block_stride=32768, layer_offset=1024,
        POOL=4, MAPPED=False, BLOCK=512, table_s1=0, TOKENS=1, PATH_WIDTH=8)
    signatures['context'] = 'i32'
    variants = [('tree-slots', _pool_slots, signatures, constants, 4, 1)]
    for rows in (1, 8, 15, 32):
        for transpose in (False, True):
            inner, outer = (512, 256) if transpose else (256, 512)
            variants.append((f'absorb-m{rows}-transpose{int(transpose)}', _absorb,
                {'X': '*bf16', 'W': '*bf16', 'Y': '*bf16', 'ROWS': 'i32'},
                dict(HEADS=16, INPUT=inner, OUTPUT=outer, WH=512*512, WR=512,
                     TRANSPOSE=transpose, BM=16 if rows <= 16 else 32, BN=64, BK=64), 4, 2))
    for name, fn, signature, const, warps, stages in variants:
        kernel = triton.compile(ASTSource(fn, signature, constexprs=const), target=GPUTarget('cuda', 121, 32),
                               options=dict(num_warps=warps, num_stages=stages))
        cubin = output/(name+'.cubin')
        cubin.write_bytes(kernel.asm['cubin'])
        resources = subprocess.check_output([str(dump), '--dump-resource-usage', str(cubin)], text=True)
        sass = subprocess.check_output([str(dump), '--dump-sass', str(cubin)], text=True)
        stack = int(re.search(r'\bSTACK:(\d+)', resources)[1])
        local = len(re.findall(r'\b(?:LDL|STL)\b', sass))
        if stack or local:
            raise AssertionError(f'{name}: stack={stack}, local instructions={local}')
        report['triton_kernels'].append(dict(name=name, registers=int(re.search(r'\bREG:(\d+)', resources)[1]),
            stack_bytes=stack, local_instructions=local, shared_bytes=kernel.metadata.shared,
            cubin_sha256=sha(cubin)))
    report['source_sha256'] = {str(p): sha(p) for p in [source, Path(__file__),
        Path('engine/kernels/indexer.py'), Path('engine/kernels/mla/decode_absorb.py'),
        Path('engine/kernels/mla/prefill_absorb.py'), Path('probes/engine_mla_hardware_check.py')]}
    # Compile the actual Torch binding as well as extracted device bodies.
    # CPU Torch omits its generated CUDA export header; use c10's documented
    # opt-out. This is an object-only check, without CUDA linking/execution.
    include = Path(torch.__file__).parent/'include'
    obj = output/'full-mla.o'
    command = [str(nvcc), '-O2', '-std=c++20', '-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE',
        '-gencode', 'arch=compute_121a,code=sm_121a', '-c', '-I'+str(include),
        '-I'+str(include/'torch/csrc/api/include'), '-I'+sysconfig.get_paths()['include'],
        str(source), '-o', str(obj)]
    done = subprocess.run(command, capture_output=True, text=True)
    (output/'full-mla.log').write_text(done.stdout+done.stderr)
    if done.returncode:
        raise RuntimeError(done.stdout+done.stderr)
    report['full_translation_unit'] = dict(command=command, exit_code=done.returncode,
        object_sha256=sha(obj), linked=False, note='CPU Torch headers with c10 CUDA export-header opt-out')
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise AssertionError('offline compilation initialized CUDA')
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.cuda_root, args.output, args.baseline)
