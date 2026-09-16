"""Compile the served b12x MoE kernels for every tile-major w13 chunk on the CPU; no CUDA context.

The chunk (moe_static_kernel_v5.TILED_W13_CHUNKS) is a storage layout: the kernels keep their tiles and
arithmetic, and only the 4-D weight tensor they compile against changes. This proves every served handle
traces, lays out its TMA descriptors over that tensor and passes ptxas for each chunk:

  static   the served recipe t,r,sf6,batch at 8 rows (C=1), 16 rows (C=2) and 32 rows (the t tile a short
           static prefill takes), and the stamped 16-row tile the GPU cells read their timeline from; at the 512
           chunk also the arms the GPU cells compare it with: the C=2 tile with two FC2 slots (plain and
           stamped), the stamped C=1 tile and the probe-only xa / xs timing cells
  dynamic  the served prefill classes: Q0 words (m=2304), long SF6 words (m=16384) and its FFN packets

It proves nothing about numerics or speed (probes/engine_moe_c2_cells.py on the single-GPU lane).

    CUDA_VISIBLE_DEVICES= CUTE_DSL_ARCH=sm_121a PYTHONPATH=/repo python3 probes/engine_moe_chunk_compile.py --output /out/chunks.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

SOURCES = ('engine/kernels/b12x/moe_dispatch.py', 'engine/kernels/b12x/moe_static_kernel_v4.py',
           'engine/kernels/b12x/moe_static_kernel_v5.py', 'engine/kernels/b12x/moe_static_common.py',
           'engine/kernels/b12x/moe_dynamic_gated_sf6.py', 'engine/kernels/b12x/moe_dynamic_gated_sf6_q0.py',
           'engine/kernels/b12x/moe_dynamic_gated_sf6_q0_words.py', 'engine/kernels/b12x/moe_dynamic_gated_sf6_prefill.py',
           'engine/kernels/b12x/moe_dynamic_prefill_packets.py', 'probes/engine_moe_chunk_compile.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunks', default='512,256')
    parser.add_argument('--only', default='static,dynamic')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('compile requires CUDA_VISIBLE_DEVICES=')
    os.environ['CUTE_DSL_ARCH'] = 'sm_121a'
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    chunks = [int(c) for c in args.chunks.split(',')]
    only = set(args.only.split(','))
    records = []
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(torch.cuda, 'get_device_capability', return_value=(12, 1)):
        from engine.kernels.b12x import moe_dispatch as md
        md.configure_static_v2('t,r,sf6,batch')
        md.configure_tp_sf6_q0(True)
        selected = {}

        def builder(module, name, build, **kwargs):
            start = time.monotonic()
            build()  # the actual CuTe lowering, ptxas and TVM-FFI build
            records.append(dict(selected, name=name, status='PASS', seconds=round(time.monotonic() - start, 2)))
            print(json.dumps(records[-1]), flush=True)

        glm = dict(activation='swigluoai_uninterleave', swiglu_alpha=1., swiglu_beta=0., swiglu_limit=10.)
        cases = []
        if 'static' in only:
            for chunk in chunks:
                for rows, stamps in ((8, False), (16, False), (32, False), (16, True)):
                    cases.append(('static', rows, chunk, dict(stamps=stamps)))
            for rows, extra in ((16, dict(c2_fc2_prefetch=False)), (16, dict(c2_fc2_prefetch=False, stamps=True)),
                                (8, dict(stamps=True)), (16, dict(spec='xa')), (16, dict(spec='xs'))):
                cases.append(('static', rows, 512, extra))
            # the l<n> prefetch cells over the 256 chunk, the only storage whose reform box is one contiguous run
            for rows, cell in ((16, 'l2'), (16, 'l4'), (16, 'l8'), (8, 'l4'), (16, 'lf4'), (8, 'lf4'), (16, 'z'), (8, 'z')):
                cases.append(('static', rows, 256, dict(spec=cell)))
            # short-prefill static row counts: the t tile at 12 rows, and the M16 reform for every row count
            for chunk in chunks:
                cases.append(('static', 12, chunk, dict(stamps=False)))
                for rows in (12, 32):
                    cases.append(('static', rows, chunk, dict(reform_every_static=True)))
        if 'dynamic' in only:
            for chunk in chunks:
                for m, packets in ((2304, False), (16384, False), (16384, True)):
                    cases.append(('dynamic', m, chunk, dict(packets=packets)))
        with patch.object(md, 'get_num_sm', return_value=48), \
                patch.object(md, 'get_max_active_clusters', return_value=48), \
                patch.object(md, 'build_and_load_cute_dsl_kernel', builder):
            for kind, rows, chunk, extra in cases:
                selected.clear()
                selected.update(kind=kind, rows=rows, chunk=chunk, **extra)
                before = len(records)
                try:
                    if kind == 'static':
                        spec = 't,r,sf6,batch' + (',' + extra['spec'] if 'spec' in extra else '')
                        config = dict(md._parse_glm53_static_v2(spec, probe=True),
                                      **{k: v for k, v in extra.items() if k != 'spec'})
                        md._get_static_kernel_v2(288, 288, rows, 4096, 512, 8, rows * 8, config=config,
                                                 mac_override=48, w13_chunk=chunk, **glm)
                    else:
                        md._get_dynamic_kernel(288, rows, 4096, 512, 8, rows * 8, tiled=True, reform_sf_pack=True,
                                               tile_m=128, _prefill_packets=extra['packets'], w13_chunk=chunk, **glm)
                    if len(records) == before:
                        raise RuntimeError('the dispatcher returned a cached handle instead of building one')
                except Exception as exc:
                    records.append(dict(selected, status='FAIL', error=f'{type(exc).__name__}: {exc}'[:2000],
                                        traceback=traceback.format_exc()[-3000:]))
                    print(json.dumps({k: v for k, v in records[-1].items() if k != 'traceback'}), flush=True)
    if torch.cuda.is_initialized():
        raise RuntimeError('compile initialized CUDA')
    passed = len(records) == len(cases) and all(r['status'] == 'PASS' for r in records)
    report = dict(status='PASS' if passed else 'FAIL', gpu_used=False, chunks=chunks, kernels=records,
                  scope='native compile of the served handles per w13 chunk; GPU numerics and timing pending',
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(status=report['status'], compiled=sum(r['status'] == 'PASS' for r in records),
                          cases=len(cases))), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
