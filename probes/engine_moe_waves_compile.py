"""Compile the two actual resident-wave dispatch handles with no CUDA device."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('the compile gate requires CUDA_VISIBLE_DEVICES=')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        config = md._parse_glm53_static_v2('t,r,sf6')
        handles, records = [], []
        selected = [False]
        kernel_class = md.MoEStaticKernelV5
        def constructor(**kwargs):
            kernel = kernel_class(**kwargs)
            if kernel.even != selected[0]:
                raise RuntimeError('the dispatcher did not forward the wave option')
            return kernel
        def builder(module, name, build, **kwargs):
            kernel = build()
            records.append(dict(name=name, even=selected[0]))
            return kernel
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'MoEStaticKernelV5', constructor), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
            for enabled in (False, True, False, True):
                selected[0] = enabled
                handle = md._get_static_kernel_v2(288, 288, 7, 4096, 512, 8, 128,
                    config=dict(config, even=enabled), mac_override=48,
                    activation='swigluoai_uninterleave', swiglu_alpha=1.,
                    swiglu_beta=0., swiglu_limit=10.)
                handles.append(handle)
        if len(records) != 2 or handles[0] is not handles[2] or handles[1] is not handles[3]:
            raise RuntimeError('the two dispatch handles did not compile and cache independently')
        if handles[0] is handles[1] or torch.cuda.is_initialized():
            raise RuntimeError('aliased handles or unexpected CUDA initialization')
    report = dict(status='PASS', gpu_used=False, scope='two native SM121 compile handles only',
                  torch=torch.__version__, kernels=records,
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                     for name in ('engine/kernels/b12x/moe_dispatch.py',
                                  'engine/kernels/b12x/moe_static_kernel_v4.py',
                                  'engine/kernels/b12x/moe_static_kernel_v5.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
