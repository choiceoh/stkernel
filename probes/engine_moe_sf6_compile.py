"""Compile C1 SF6/activation storage and controls in an existing CPU-only ST image."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import subprocess
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sass', action='store_true', help='also disassemble and count native instructions')
    parser.add_argument('--activation-store', action='store_true', help='compare packed activation stores')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    if args.sass and shutil.which('nvdisasm') is None:
        raise RuntimeError('native instruction checks require nvdisasm on PATH')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    os.environ['CUTE_DSL_KEEP'] = 'ptx,cubin'
    os.environ['CUTE_DSL_DUMP_DIR'] = str(args.output.parent / 'cute')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    records, selected = [], {}
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        from engine.kernels.b12x.moe_static_kernel_v4 import MoEStaticKernelV4
        setup = MoEStaticKernelV4._setup_attributes

        def checked_setup(owner, hidden_size):
            setup(owner, hidden_size)
            selected.update(smem_bytes=owner.smem_bytes,
                            smem_capacity=owner.smem_capacity,
                            separate=owner.sf6_separate, word_expand=owner.sf6_word_expand,
                            packed_activation_store=owner.packed_activation_store)

        def builder(module, name, build, **kwargs):
            start = time.monotonic()
            kernel = build()  # Actual CuTe lowering, PTXAS and TVM-FFI; no CUDA context.
            record = dict(selected, kernel=name, status='PASS', seconds=time.monotonic()-start)
            cubin = kernel.__cubin__
            if not isinstance(cubin, bytes):
                # The CUDA-dialect pipeline lowers its fatbin into an LLVM
                # byte string, while __cubin__ returns an unused dump path.
                # Decode that retained binary without loading a CUDA library.
                from cutlass.base_dsl.jit_executor import get_escaped_cubin_bytes
                payloads = re.findall(r'llvm\.mlir\.global[^\n]*@\w+_binary\("([^"\n]*)"\)',
                                      str(kernel.ir_module))
                if len(payloads) != 1:
                    raise RuntimeError(f'expected one retained native binary, got {len(payloads)}')
                cubin = get_escaped_cubin_bytes(payloads[0].encode())
            artifact = args.output.parent / (name + '.fatbin')
            artifact.write_bytes(cubin)
            resources = subprocess.run(['cuobjdump', '--dump-resource-usage', str(artifact)],
                                       check=True, capture_output=True, text=True).stdout
            record.update(native_binary_sha256=hashlib.sha256(cubin).hexdigest(), resources=resources)
            if args.sass:
                sass = subprocess.run(['cuobjdump', '--dump-sass', str(artifact)],
                                      check=True, capture_output=True, text=True).stdout
                instructions = re.findall(
                    r'^\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)(?:\s|;)',
                    sass, re.M)
                if not instructions:
                    raise RuntimeError('no native instructions were captured')
                artifact.with_suffix('.sass').write_text(sass)
                record.update(sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),
                    static_instructions=len(instructions), opcode_counts=dict(sorted(Counter(instructions).items())))
            records.append(record)
            print(json.dumps(record), flush=True)
            return kernel

        cases = ([(1, True, True, True), (7, True, True, True), (8, True, True, True),
                  (8, True, True, False), (16, True, True, True), (32, True, True, True)]
                 if args.activation_store else
                 [(1, True, True, True), (7, True, True, True), (8, True, True, True),
                  (8, True, False, True), (8, False, False, True),
                  (16, True, True, True), (32, True, True, True)])
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder), \
                patch.object(MoEStaticKernelV4, '_setup_attributes', checked_setup):
            for rows, separate, word_expand, activation_store in cases:
                selected.clear()
                selected.update(rows=rows, requested_separate=separate, requested_word_expand=word_expand,
                                requested_activation_store=activation_store)
                config = dict(md._parse_glm53_static_v2('t,r,sf6'),
                              sf6_separate=separate, sf6_word_expand=word_expand,
                              packed_activation_store=activation_store)
                try:
                    config = md._static_v2_decode_config(config, rows)
                    md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, rows*8,
                        config=config, mac_override=48, activation='swigluoai_uninterleave',
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                except Exception as exc:
                    record = dict(selected, status='FAIL', error=repr(exc))
                    records.append(record)
                    print(json.dumps(record), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    passed = len(records) == len(cases) and all(r['status'] == 'PASS' for r in records)
    report = dict(status='PASS' if passed else 'FAIL', gpu_used=False, kernels=records,
                  nvdisasm_version=(subprocess.check_output(['nvdisasm', '--version'], text=True).strip()
                                    if args.sass else None),
                  scope='native compile and layout checks; GPU numerics/replay/timing pending',
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ('engine/kernels/b12x/moe_dispatch.py',
                                   'engine/kernels/b12x/moe_static_kernel_v4.py',
                                   'engine/kernels/b12x/moe_static_common.py',
                                   'engine/kernels/b12x/moe_static_kernel_v5.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise RuntimeError('one or more native SF6 handles failed')


if __name__ == '__main__':
    main()
