"""Compile both long-prefill MoE input ABIs without initializing CUDA."""
import argparse
import json
import os
from pathlib import Path
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', default='t,r,sf6,batch,q0,as1',
                        help='Compile the actual serving recipe; omit as1 for the incident control')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('use CUDA_VISIBLE_DEVICES= and a container without GPUs')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    import torch
    records = []
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        from engine.profiles.glm53.lanes import parse_moe_static
        spec, q0 = parse_moe_static(args.recipe)
        md.configure_static_v2(spec)
        md.configure_tp_sf6_q0(q0)
        def build(module, name, callback, **kwargs):
            start = time.monotonic()
            result = callback()
            records.append(dict(packets=packets, kernel=name, seconds=time.monotonic()-start))
            print(json.dumps(records[-1]), flush=True)
            return result
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', build):
            for packets in (False, True):
                md._get_dynamic_kernel(288, 32256, 4096, 512, 8, 32256*8,
                    activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0.,
                    swiglu_limit=10., tiled=True, reform_sf_pack=True, tile_m=128,
                    _prefill_packets=packets)
    if torch.cuda.is_initialized() or len(records) != 2:
        raise RuntimeError('both real ABIs must compile without CUDA initialization')
    args.output.write_text(json.dumps(dict(status='PASS', gpu_used=False, recipe=args.recipe,
                                         kernels=records), indent=2)+'\n')


if __name__ == '__main__':
    main()
