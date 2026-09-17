"""Inspect the compiled input-reuse cells without creating a CUDA context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--control-only', action='store_true', help='also works on the pre-change source')
    parser.add_argument('--modes', default='0,1,2,3,4', help='comma-separated selectors, or default for the serving selection')
    args = parser.parse_args()
    modes = (0,) if args.control_only else tuple(None if m == 'default' else int(m) for m in args.modes.split(','))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or list(Path('/dev').glob('nvidia*')):
        raise RuntimeError('CPU compile requires no exposed CUDA devices')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    import torch
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md

        def build_resources(module, name, build, **kwargs):
            kernel = build()
            binary = kernel.__cubin__
            if not isinstance(binary, bytes):
                from cutlass.base_dsl.jit_executor import get_escaped_cubin_bytes
                payload = re.findall(r'llvm\.mlir\.global[^\n]*@\w+_binary\("([^"\n]*)"\)', str(kernel.ir_module))
                if len(payload) != 1:
                    raise RuntimeError('missing compiled CUDA binary')
                binary = get_escaped_cubin_bytes(payload[0].encode())
            artifact = args.output.parent / (name + '.fatbin')
            artifact.write_bytes(binary)
            usage = subprocess.check_output(['cuobjdump', '--dump-resource-usage', str(artifact)], text=True)
            sass = subprocess.check_output(['cuobjdump', '--dump-sass', str(artifact)], text=True)
            artifact.with_suffix('.sass').write_text(sass)
            record = dict(kernel=name, binary_sha256=hashlib.sha256(binary).hexdigest(),
                          sass_sha256=hashlib.sha256(sass.encode()).hexdigest(), resources=usage)
            records.append(record)
            print(json.dumps(record), flush=True)
            return kernel

        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', build_resources):
            for rows in (8, 16):
                for mode in modes:
                    cfg = dict(md._parse_glm53_static_v2('t,r,sf6,batch'), input_vec16=True)
                    if mode is not None:
                        cfg['input_reuse'] = mode
                    md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, rows*8, config=cfg,
                        mac_override=48, w13_chunk=256, activation='swigluoai_uninterleave',
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                    records[-1].update(rows=rows, input_reuse='default' if mode is None else mode)
    if torch.cuda.is_initialized() or len(records) != 2 * len(modes):
        raise RuntimeError('missing cells or unexpectedly initialized CUDA context')
    sources = ('engine/kernels/b12x/moe_static_kernel_v4.py', 'engine/kernels/b12x/moe_dispatch.py')
    args.output.write_text(json.dumps(dict(gpu_used=False, scope='static native resources only', kernels=records,
        sources={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources}), indent=2)+'\n')


if __name__ == '__main__':
    main()
