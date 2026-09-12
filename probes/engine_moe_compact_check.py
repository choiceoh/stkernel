"""Seven-row native MoE residency experiment; never a consumer speed claim.

The 96-CTA cooperative launch is refused before execution unless CUDA's
occupancy API confirms that this exact cubin can keep all CTAs resident.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import time

ARTIFACTS = Path('/cache/st-moe-compact-proof') / str(time.time_ns())
ARTIFACTS.mkdir(parents=True)
os.environ.update(CUTE_DSL_ARCH='sm_121a', CUTE_DSL_KEEP='ptx,cubin',
                  CUTE_DSL_DUMP_DIR=str(ARTIFACTS), CUTE_DSL_DISABLE_FILE_CACHING='1')

import torch
from engine.kernels.b12x import moe_dispatch as md
from engine.modules.nvfp4_sf import swizzle_sf
from engine.profiles.glm53.lanes import served
from probes.engine_decode_fusions import _capture, _time


def report(**row):
    print(json.dumps(row), flush=True)


def checked(result):
    if int(result[0]):
        raise RuntimeError(str(result[0]))
    return result[1] if len(result) == 2 else result[1:]


def occupancy(since, dynamic_smem, mac):
    from cuda.bindings import driver as cu
    cubins = [p for p in ARTIFACTS.rglob('*.cubin') if p.stat().st_mtime >= since]
    assert len(cubins) == 1, ('need the exact fresh compiled cubin', [str(p) for p in cubins])
    cubin = cubins[0]
    entries = {name for p in ARTIFACTS.rglob('*.ptx') if p.stat().st_mtime >= since
               for name in re.findall(r'\.entry\s+([\w$]+)\s*\(', p.read_text())}
    assert len(entries) == 1, entries
    module = checked(cu.cuModuleLoad(str(cubin).encode()))
    try:
        function = checked(cu.cuModuleGetFunction(module, next(iter(entries)).encode()))
        attr = cu.CUfunction_attribute
        checked(cu.cuFuncSetAttribute(function, attr.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, dynamic_smem))
        blocks = checked(cu.cuOccupancyMaxActiveBlocksPerMultiprocessor(function, 160, dynamic_smem))
        registers = checked(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_NUM_REGS, function))
        local = checked(cu.cuFuncGetAttribute(attr.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function))
        sms = torch.cuda.get_device_properties(0).multi_processor_count
        report(lane='moe_compact_residency', cubin=str(cubin), sha256=hashlib.sha256(cubin.read_bytes()).hexdigest(),
               dynamic_smem=dynamic_smem, registers=registers, local_bytes=local,
               blocks_per_sm=blocks, sms=sms, requested_ctas=mac)
        assert mac <= blocks * sms, 'cooperative grid exceeds actual cubin residency; do not launch'
    finally:
        checked(cu.cuModuleUnload(module))


def main(router_ranks=None):
    assert torch.cuda.get_device_capability() == (12, 1)
    torch.manual_seed(91613)
    if router_ranks:
        from probes.engine_decode_fusions import tensorcore_router
        tensorcore_router(lambda lane, **values: report(lane=lane, **values), ranks=router_ranks)
    native = served(moe_static='t,r,sf6,q0')
    cfg = md._parse_glm53_static_v2('t,r,sf6', probe=True)
    original_get = md._get_static_kernel_v2
    original_name = md._disk_kernel_name
    before = md._STATIC_V2_OVERRIDE
    # New outer names force each requested cubin to be emitted for its gate.
    md._disk_kernel_name = lambda prefix, key: original_name(prefix+'_compact_'+ARTIFACTS.name, key)
    arm = 'B'
    checked_keys = set()
    keepalive = []

    def get(*args, **kwargs):
        compact = arm != 'B'
        mac = 96 if arm == 'C96' else 48
        config = dict(kwargs['config'], fc1=1 if compact else 2, fc2=1 if compact else 2,
                      decode_compact=compact)
        kwargs.update(config=config, mac_override=mac)
        key = (args[2], arm, config['reform_sf_pack'])
        since = time.time()
        result = original_get(*args, **kwargs)
        if key not in checked_keys:
            if compact:
                occupancy(since, 44032, mac)
            checked_keys.add(key)
        keepalive.append(result)
        return result

    md._get_static_kernel_v2 = get
    experts, hidden, width, rows = 288, 4096, 512, 7
    w13 = torch.randint(0, 256, (experts, 2*width, hidden//2), device='cuda', dtype=torch.uint8)
    w2 = torch.randint(0, 256, (experts, hidden, width//2), device='cuda', dtype=torch.uint8)
    s13 = torch.stack([swizzle_sf((torch.rand(2*width, hidden//16, device='cuda')*.04+.01).to(torch.float8_e4m3fn)) for _ in range(experts)])
    s2 = torch.stack([swizzle_sf((torch.rand(hidden, width//16, device='cuda')*.04+.01).to(torch.float8_e4m3fn)) for _ in range(experts)])
    x = torch.randn(rows, hidden, device='cuda', dtype=torch.bfloat16)*.3
    ids = torch.arange(rows*8, device='cuda', dtype=torch.int32).reshape(rows, 8)
    weights = torch.rand(rows, 8, device='cuda'); weights.div_(weights.sum(1, keepdim=True))
    try:
        for packed in (True, False):
            md._STATIC_V2_OVERRIDE = dict(cfg, reform_sf_pack=packed)
            for unique in ((8, 32, 56) if packed else (32,)):
                rotations = min(8, experts//unique)
                route_sets = [(torch.arange(rows*8, device='cuda', dtype=torch.int32)%unique + i*unique).reshape(rows,8)
                              for i in range(rotations)]
                graphs, outputs = [], []
                try:
                    for arm in ('B', 'C48', 'C96'):
                        graph, output = _capture(lambda: [native.moe(x, route, weights, w13, s13, w2, s2, 10.)
                                                         for route in route_sets])
                        graphs.append(graph); outputs.append(output)
                    max_error = 0.
                    for i in range(5):
                        x.normal_().mul_(.1+i*.2)
                        for route in route_sets:
                            route.add_(11).remainder_(experts)
                        if i == 4:
                            weights.zero_()
                        for graph in graphs:
                            graph.replay()
                        torch.cuda.synchronize()
                        for candidate in outputs[1:]:
                            for actual, expected in zip(candidate, outputs[0]):
                                error = (actual.float()-expected.float()).abs().max().item()
                                limit = max(1e-5, expected.float().abs().max().item()*.002)
                                assert torch.isfinite(actual).all().item() and error <= limit, (packed, unique, i, error, limit)
                                if i == 4:
                                    assert torch.count_nonzero(actual).item() == 0
                                max_error = max(max_error, error)
                    weights.uniform_(); weights.div_(weights.sum(1, keepdim=True))
                    timings = [dict(arm=label, ms=_time(graphs[i], iterations=32)/rotations) for label,i in
                               (('B',0),('C48',1),('C96',2),('C96',2),('C48',1),('B',0))]
                    report(lane='moe_compact_timing', rows=rows, unique=unique, rotations=rotations,
                           sf6=packed, max_error=max_error, changed_graph_inputs=True, exact_zero=True,
                           measurements=timings, scope='same packs, synthetic routes, captured; not consumer speed')
                finally:
                    for graph in graphs:
                        graph.reset()
        report(passed=True)
    finally:
        md._get_static_kernel_v2 = original_get
        md._disk_kernel_name = original_name
        md._STATIC_V2_OVERRIDE = before


if __name__ == '__main__':
    main()
