"""Compile declared MoE capacity candidates without opening a CUDA device."""
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
        raise RuntimeError('CPU compilation requires CUDA_VISIBLE_DEVICES=')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    rows, instances = [], []
    report = dict(gpu_used=False, torch=torch.__version__, cases=rows,
                  scope='native CuTe compilation and shared-memory feasibility only')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        base = md._parse_glm53_static_v2('t,r,sf6')
        kernel_class = md.MoEStaticKernelV5
        def constructor(**kwargs):
            kernel = kernel_class(**kwargs)
            instances.append(kernel)
            return kernel
        def builder(module, name, build, **kwargs):
            return build()
        cases = [('served-m7', 7, base),
                 ('moe_stage_fc1', 7, dict(base, fc1=3, fc2=1)),
                 ('moe_stage_fc1_shared', 7, dict(base, fc1=3, fc2=1, probe_shared_epilogue=True)),
                 ('moe_stage_fc2', 7, dict(base, fc1=1, fc2=3))]
        cases += [('moe_batch', m, dict(base, probe_batch_reform=True)) for m in (14, 21, 28)]
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'MoEStaticKernelV5', constructor), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
            for name, m, config in cases:
                row = dict(candidate=name, rows=m, status='RUNNING')
                try:
                    handle = md._get_static_kernel_v2(288, 288, m, 4096, 512, 8, m*8,
                        config=config, mac_override=48, activation='swigluoai_uninterleave',
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                    repeated = md._get_static_kernel_v2(288, 288, m, 4096, 512, 8, m*8,
                        config=config, mac_override=48, activation='swigluoai_uninterleave',
                        swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
                    if handle is not repeated:
                        raise RuntimeError('same configuration did not reuse its handle')
                    kernel = instances[-1]
                    row.update(status='PASS', smem_bytes=kernel.smem_bytes,
                               tile_m=kernel.tile_m, fc1=kernel.fc1_stages, fc2=kernel.fc2_stages,
                               shared_epilogue=kernel.shared_epilogue)
                except Exception as exc:
                    row.update(status='FAIL', error=str(exc))
                rows.append(row)
                args.output.write_text(json.dumps(report, indent=2)+'\n')
                print(json.dumps(row), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU compile opened a CUDA device')
    report.update(status='PASS' if all(row['status']=='PASS' for row in rows) else 'PARTIAL',
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                 for name in ('engine/kernels/b12x/moe_dispatch.py',
                                              'engine/kernels/b12x/moe_static_kernel_v4.py',
                                              'engine/kernels/b12x/moe_static_kernel_v5.py')})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
