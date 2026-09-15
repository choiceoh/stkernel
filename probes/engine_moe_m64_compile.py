"""Compile the private M64 prefill MoE kernel for sm_121a with no device.

The device capability is patched, not present: `_get_dynamic_kernel` gates the TP SF6 Q0
family on `get_device_capability() == (12, 1)`, and this box's card is sm_120. Nothing here
opens a CUDA context -- sm_121a is a compile target for ptxas, and the check below asserts
that no context was created.

    bash bench/compile_sm121a.sh probes/engine_moe_m64_compile.py --output /evidence --dump

It compiles two arms of the same shape -- the pinned M128 control and the private M64
candidate -- so a change to either is caught where it is cheap. On the box this was written
for that is about three seconds an arm; the GPU queue is twenty minutes a round trip.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dump', action='store_true', help='keep PTX/cubin and read resources')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUTE_DSL_ARCH', 'sm_121a')

    import torch
    from engine.kernels.b12x import moe_dispatch as md

    report = {'status': 'RUNNING', 'arch': os.environ['CUTE_DSL_ARCH'], 'variants': []}
    md.get_num_sm = lambda *a: 48                    # GB10
    md.get_max_active_clusters = lambda *a: 48

    if args.dump:
        original_build = md.build_and_load_cute_dsl_kernel

        def build_reader(module, name, build, **kwargs):
            dest = args.output / 'cute' / name
            dest.mkdir(parents=True, exist_ok=True)
            original_compile = md.cute.compile

            def compile(*a, **kw):
                kw['options'] = kw.get('options', '') + \
                    f' --keep-ptx --keep-cubin --dump-dir={dest}'
                return original_compile(*a, **kw)
            md.cute.compile = compile
            try:
                return build()
            finally:
                md.cute.compile = original_compile
        md.build_and_load_cute_dsl_kernel = build_reader

    md.configure_static_v2('t,r,sf6')
    md.configure_tp_sf6_q0(True)

    shape = dict(activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                 swiglu_limit=10., tiled=True, reform_sf_pack=True)
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        for label, tile_m, tile64 in (('m128_control', 128, False), ('m64_candidate', 64, True)):
            rows = 2304
            md._get_dynamic_kernel(288, rows, 4096, 512, 8, rows, tile_m=tile_m,
                                   _prefill_tile64=tile64, **shape)
            report['variants'].append(dict(kind=label, rows=rows, tile_m=tile_m,
                                           key=repr(list(md._DYNAMIC_KERNEL_CACHE)[-1])))
            print('compiled', label, flush=True)

    if args.dump:
        cuobjdump = '/usr/local/cuda/bin/cuobjdump'
        for cubin in sorted((args.output / 'cute').rglob('*.cubin')):
            usage = subprocess.check_output([cuobjdump, '--dump-resource-usage', str(cubin)],
                                            text=True)
            report.setdefault('resources', []).append(
                dict(path=str(cubin.relative_to(args.output)), usage=usage))

    report['cuda_initialized'] = bool(torch.cuda.is_initialized())
    report['status'] = 'PASS' if not report['cuda_initialized'] else 'FAIL'
    (args.output / 'result.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'resources'}), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
