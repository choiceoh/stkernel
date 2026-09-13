"""Compile the real private MoE scatter ABI and coordinate checks without a GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    records, selected = [], {}
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        base = md._parse_glm53_static_v2('t,r,sf6')
        def builder(module, name, build, **kwargs):
            start = time.monotonic()
            kernel = build()  # Real CuTe lowering, PTXAS and TVM-FFI, no cached substitute.
            record = dict(selected, kernel=name, status='PASS', seconds=time.monotonic()-start)
            records.append(record)
            print(json.dumps(record), flush=True)
            return kernel
        cases = [(rows, route, direct) for rows in (7, 14, 21, 28)
                 for route, direct in ((False, False), (True, False), (False, True), (True, True))]
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
            for rows, route, direct in cases:
                selected.update(rows=rows, route_scatter=route, direct_scatter=direct)
                config = dict(base, probe_route_scatter=route, probe_direct_scatter=direct)
                try:
                    md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, 128,
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
                  scope='native compile and static output coordinate coverage; numerics/timing pending',
                  source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                      for name in ('engine/kernels/b12x/moe_dispatch.py', 'engine/kernels/b12x/moe_static_kernel_v4.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    if not passed:
        raise RuntimeError('one or more native scatter handles failed')


if __name__ == '__main__':
    main()
