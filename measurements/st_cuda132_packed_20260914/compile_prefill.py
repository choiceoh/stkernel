"""Compile the two served SF6 word-producer families with no GPU access."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('CPU-only compile requires CUDA hidden and no device nodes')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    import torch
    from probes.engine_moe_sf6_compile import instruction_opcodes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md

        def builder(module, name, build, **kwargs):
            start = time.monotonic()
            kernel = build()
            binary = kernel.__cubin__
            if not isinstance(binary, bytes):
                from cutlass.base_dsl.jit_executor import get_escaped_cubin_bytes
                payload = re.findall(r'llvm\.mlir\.global[^\n]*@\w+_binary\("([^"\n]*)"\)',
                                     str(kernel.ir_module))
                if len(payload) != 1:
                    raise RuntimeError('missing compiled CUDA binary')
                binary = get_escaped_cubin_bytes(payload[0].encode())
            artifact = args.output.parent / (name + '.fatbin')
            artifact.write_bytes(binary)
            sass = subprocess.check_output(['cuobjdump', '--dump-sass', str(artifact)], text=True)
            resources = subprocess.check_output(['cuobjdump', '--dump-resource-usage', str(artifact)], text=True)
            counts = dict(sorted(Counter(instruction_opcodes(sass)).items()))
            if counts.get('VIADD.U8x4', 0) == 0:
                raise RuntimeError('served SF6 producer did not emit the native byte add')
            record = dict(status='PASS', kernel=name, seconds=time.monotonic()-start,
                          binary_sha256=hashlib.sha256(binary).hexdigest(),
                          sass_sha256=hashlib.sha256(sass.encode()).hexdigest(),
                          native_byte_add_sites=counts['VIADD.U8x4'], resources=resources)
            records.append(record)
            artifact.with_suffix('.sass').write_text(sass)
            print(json.dumps(record), flush=True)
            return kernel

        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
            for rows, q0 in ((65, True), (8193, False)):
                md._get_dynamic_kernel(288, rows, 4096, 512, 8, rows,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., tiled=True, reform_sf_pack=True,
                    tile_m=128, _tp_sf6_q0_override=q0)
    if torch.cuda.is_initialized() or len(records) != 2:
        raise RuntimeError('expected two compiled families and no CUDA context')
    args.output.write_text(json.dumps(dict(status='PASS', gpu_used=False, kernels=records,
        scope='CPU lowering, native instructions and resources only'), indent=2)+'\n')


if __name__ == '__main__':
    main()
