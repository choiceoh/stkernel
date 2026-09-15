"""Compile the KDA output norm's o_proj pack writer for SM121 without CUDA: C1 and sixteen-row layouts.

`--baseline FILE` also compiles `_output_norm_pack` from another copy of engine/kernels/kda/output.py and
requires its C1 specialization to lower to the same PTX as this tree's WIDE=False one.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def compile_pack(fn, weight, wide):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    constexprs = dict(ROWS=256 if wide else 128, HEADS=16, D=128, EPS=1e-5, BD=128, BR=1)  # tokens x heads programs
    if wide is not None:
        constexprs['WIDE'] = wide
    source = ASTSource(fn=fn, signature=dict(X='*bf16', G='*bf16', W='*' + weight, Y='*bf16',
                                             WORDS='*i32', SCALES='*fp32'), constexprs=constexprs)
    return triton.compile(source, target=GPUTarget('cuda', 121, 32), options=dict(num_warps=1, enable_fp_fusion=False))


def resources(cubin):
    """cuobjdump's resource line for the compiled binary: registers, stack, shared and local bytes."""
    with tempfile.NamedTemporaryFile(suffix='.cubin') as f:
        f.write(cubin)
        f.flush()
        usage = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--dump-resource-usage', f.name], text=True)
    found = re.search(r'REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)', usage)
    return dict(zip(('registers', 'stack', 'shared', 'local'), map(int, found.groups()))) if found else dict(raw=usage[-400:])


def body(ptx):
    """The instruction section without locations or comments: debug sections carry source paths and line tables,
    which a moved line or another file name changes without changing the program."""
    lines = []
    for line in ptx.splitlines():
        if re.match(r'\s*\.section\s+\.debug', line):
            break
        line = re.sub(r'//.*$', '', line).rstrip()
        if line.strip() and not line.strip().startswith(('.loc', '.file')):
            lines.append(line)
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, help='another output.py whose C1 pack writer must lower identically')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    import triton
    from engine.kernels.kda.output import _output_norm_pack
    records = []
    for weight in ('bf16', 'fp32'):
        for wide in (False, True):
            kernel = compile_pack(_output_norm_pack, weight, wide)
            ptx = kernel.asm['ptx']
            registers = sorted({int(m) for m in re.findall(r'\.reg\w*\s*\.\w+\s*(\d+)', ptx)})
            records.append(dict(weight=weight, layout='sixteen-row wide' if wide else 'C1', shared_bytes=kernel.metadata.shared,
                                resources=resources(kernel.asm['cubin']),
                                metadata={k: v for k, v in kernel.metadata._asdict().items()
                                          if isinstance(v, (int, float, str, bool)) and k not in ('hash',)},
                                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                                body_sha256=hashlib.sha256(body(ptx).encode()).hexdigest(),
                                ptx_lines=len(ptx.splitlines())))
    baseline = None
    if args.baseline:
        spec = importlib.util.spec_from_file_location('baseline_output', args.baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline = []
        for weight in ('bf16', 'fp32'):
            ptx = compile_pack(module._output_norm_pack, weight, None).asm['ptx']
            mine = next(r for r in records if r['weight'] == weight and r['layout'] == 'C1')
            same = hashlib.sha256(body(ptx).encode()).hexdigest() == mine['body_sha256']
            baseline.append(dict(weight=weight, body_sha256=hashlib.sha256(body(ptx).encode()).hexdigest(), same_c1_program=same))
            if not same:
                raise RuntimeError(f'the C1 pack writer ({weight} weight) no longer lowers to the baseline program')
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    result = dict(status='PASS', gpu_used=False, torch=torch.__version__, triton=triton.__version__, kernels=records,
                  baseline=baseline, scope='SM121 PTX compilation only; GPU numerics and timing are probes/engine_producer_pack.py',
                  source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in ('engine/kernels/kda/output.py',)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
