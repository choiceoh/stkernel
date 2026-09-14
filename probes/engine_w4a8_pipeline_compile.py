"""Compare assembled W4A8 stages to a pinned revision without a CUDA context."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.backends.nvidia import compiler as nvidia_compiler
from triton.compiler import ASTSource

from engine.kernels import w4a8_pipeline as current
from engine.modules.w4a8_dataflow import W4A8PipelinePlan
from probes.engine_tree_dataflow_compile import cubin_usage


def run(output, baseline):
    if torch.cuda.is_initialized():
        raise RuntimeError('offline compilation must not inherit CUDA')
    output.mkdir(parents=True, exist_ok=True)
    source = 'engine/kernels/w4a8_pipeline.py'
    sha = subprocess.check_output(['git', 'rev-parse', baseline], text=True).strip()
    old_source = output/'baseline.py'
    old_source.write_bytes(subprocess.check_output(['git', 'show', f'{sha}:{source}']))
    spec = importlib.util.spec_from_file_location('w4a8_compile_baseline', old_source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    dump = Path(nvidia_compiler.__file__).parent/'bin/cuobjdump'
    report = dict(scope='offline SM121 code generation; not latency or serving performance',
                  gpu_used=False, baseline_commit=sha, torch=torch.__version__, triton=triton.__version__,
                  ptxas=nvidia_compiler.get_ptxas_version(121),
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                      (source, 'engine/kernels/tile_dataflow.py', 'engine/modules/w4a8_dataflow.py')},
                  baseline_source_sha256=hashlib.sha256(old_source.read_bytes()).hexdigest(), variants=[], storage=[])
    shapes = [(m, 4096, 3072) for m in (1, 4, 8, 15, 16, 17, 24, 31, 32)]
    shapes += [(1, 128, 128), (17, 384, 256), (32, 8192, 8192)]
    for rows, hidden, intermediate in shapes:
        for arm, module in (('baseline', previous), ('candidate', current)):
            stages = ('input', 'gate_up', 'down') if arm == 'candidate' else ('gate_up', 'down')
            for stage in stages:
                fn = getattr(module, '_'+stage)
                constants = dict(M=rows, H=hidden)
                signature = dict(X='*bf16', XQ='*u8', XS='*fp32')
                if stage != 'input':
                    constants.update(I=intermediate, BM=max(16, triton.next_power_of_2(rows)))
                    signature = dict(U='*u8', US='*fp32', W='*u8', S='*i8', RS='*fp32')
                    if stage == 'gate_up':
                        constants['LIMIT'] = 10.
                        signature.update(dict(X='*bf16') if arm == 'baseline' else dict(XQ='*u8', XS='*fp32'))
                    else:
                        signature['OUT'] = '*bf16'
                # The candidate explicitly aligns every workspace plane. The
                # historical small-shape baseline did not; model its actual
                # OUT alignment instead of inventing a vectorization promise.
                alignment = {p: 16 for p in signature}
                if arm == 'baseline' and stage == 'down':
                    off = rows*(intermediate+intermediate//128*4)
                    alignment['OUT'] = min(16, off & -off)
                attrs = {(fn.arg_names.index(p),): [('tt.divisibility', a)] for p, a in alignment.items()}
                options = dict(num_warps=4 if stage == 'input' else 8, num_stages=1, enable_fp_fusion=False)
                if arm == 'candidate' and stage != 'input':
                    options['arch'] = current.MMA_ARCH
                kernel = triton.compile(ASTSource(fn, signature, constexprs=constants, attrs=attrs),
                    target=GPUTarget('cuda', 121, 32),
                    options=options)
                name = f'{arm}-{stage}-m{rows}-h{hidden}-i{intermediate}'
                cubin, ptx = output/(name+'.cubin'), kernel.asm['ptx']
                cubin.write_bytes(kernel.asm['cubin'])
                (output/(name+'.ptx')).write_text(ptx)
                sass = subprocess.check_output([str(dump), '--dump-sass', str(cubin)], text=True)
                (output/(name+'.sass')).write_text(sass)
                usage = cubin_usage(cubin)
                if arm == 'candidate' and (usage['stack_bytes'] or usage['local_instructions'] or
                        any(op in ptx for op in ('atom.', 'nanosleep', 'ld.local', 'st.local'))):
                    raise AssertionError(f'candidate spills or synchronizes through a queue: {name}: {usage}')
                if arm == 'candidate' and stage != 'input' and ('e4m3.e4m3' not in ptx or 'e2m1.e2m1' in ptx or 'f16.f16' in ptx):
                    raise AssertionError('must retain FP8 MMA over expanded W4')
                if '.target sm_121a' not in ptx:
                    raise AssertionError('lowering selection must not change the actual hardware target')
                if stage == 'input' and not re.search(r'cvt\.rn\..*e4m3', ptx):
                    raise AssertionError('shared input must retain round-to-nearest FP8 conversion')
                record = dict(arm=arm, stage=stage, rows=rows, hidden=hidden, intermediate=intermediate,
                              shared_bytes=kernel.metadata.shared, num_warps=kernel.metadata.num_warps,
                              lowering_arch=options.get('arch', 'sm121'),
                              mma_operand_types=sorted(set(re.findall(r'mma\.sync[^\n]*?row\.col\.f32\.([^.]+\.[^.]+)\.f32', ptx))),
                              cubin_sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
                              static_instruction_words=len(re.findall(r'/\* (0x[0-9a-f]+) \*/', sass)), **usage)
                report['variants'].append(record)
                print(json.dumps(record), flush=True)
        p = W4A8PipelinePlan(rows, hidden, intermediate)
        report['storage'].append(dict(rows=rows, hidden=hidden, intermediate=intermediate,
            input_quantization_repetitions_before=intermediate//128, input_quantization_repetitions_after=1,
            scratch_bytes_before=rows*(intermediate+intermediate//128*4+hidden*2),
            scratch_bytes_after=p.scratch_bytes, launches_before=2, launches_after=3))
    report['cuda_initialized'] = torch.cuda.is_initialized()
    if report['cuda_initialized']:
        raise AssertionError('offline compilation initialized CUDA')
    (output/'compile.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', default='de8bfff6092c759cb5bca1d47270e68f38f7e03c')
    args = parser.parse_args()
    run(args.output, args.baseline)
