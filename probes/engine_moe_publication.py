"""Identical-input M8/M16 gate over real weights, before and after scatter.

Frontend bytes are compared by (expert, token), independent of route row order.
An isolated FC2 K128 slice and one nonzero route expose each contribution
without atomic-sum ambiguity. The ordinary full sum retains a separate repeat
noise check. No arithmetic or instrumentation is added to serving kernels.
"""
import dataclasses
from unittest.mock import patch

import torch


def frontend_snapshot(args):
    active = int(args[14].item())
    counts, experts, tokens = (args[i].cpu() for i in (13, 15, 22))
    rows = tokens.shape[1]
    keys, local_rows = [], []
    for local in range(active):
        for row in range(int(counts[local])):
            keys.append((int(experts[local]), int(tokens[local, row])))
            local_rows.append((local, row))
    assert len(keys) == len(set(keys)), 'duplicate expert/token route'
    coords = torch.tensor(local_rows, device=args[5].device, dtype=torch.int64)
    local, row = coords.unbind(1)
    packed = args[5].view(288, rows, 2048)[local, row].cpu()
    scales = args[6].view(288, -1)
    block = torch.arange(256, device=coords.device)[None, :]
    row = row[:, None]
    offsets = ((row // 128) * 32768 + (block // 4) * 512 + (row % 32) * 16
               + ((row % 128) // 32) * 4 + block % 4)
    scale = scales[local[:, None], offsets].cpu()
    return {key: (packed[i], scale[i]) for i, key in enumerate(keys)}


def assert_frontend_equal(single, pair, token_start):
    mismatched = []
    for (expert, token), values in single.items():
        other = pair[(expert, token + token_start)]
        if any(not torch.equal(a, b) for a, b in zip(values, other)):
            mismatched.append((expert, token))
    assert not mismatched, f'packed A/SFA differ: {mismatched[:8]}'
    return len(single)


def check(report, layers):
    from engine.kernels.b12x import moe_dispatch as md
    from probes import engine_moe_c2_cells as cells
    from probes.engine_decode_fusions import _capture

    cfg = md._parse_glm53_static_v2('t,r,sf6,batch,as1')
    chunk = md._w13_tile_chunk()
    get_kernel = md._get_static_kernel_v2
    captured, configs = [], {}

    class Observe:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getattr__(self, name):
            return getattr(self.kernel, name)

        def __call__(self, *args):
            result = self.kernel(*args)
            captured.append(args)
            return result

    def observe(*args, **kwargs):
        kernel, mac = get_kernel(*args, **kwargs)
        selected = md._static_v2_decode_config(kwargs['config'], args[2])
        configs[int(args[2])] = md._static_v2_input_reuse_config(selected, *args[:7])
        return Observe(kernel), mac

    for layer in layers:
        original_scale = layer.q13
        original_view = layer.views[chunk]
        x = torch.empty((16, 4096), dtype=torch.bfloat16, device='cuda')
        ids = torch.empty((16, 8), dtype=torch.int32, device='cuda')
        routes = torch.empty((16, 8), dtype=torch.float32, device='cuda')
        outputs = {name: torch.empty((8 if name != 'pair' else 16, 4096),
                                    dtype=torch.float32, device='cuda')
                   for name in ('first', 'second', 'pair')}
        selectors = {'first': slice(0, 8), 'second': slice(8, 16), 'pair': slice(0, 16)}

        def invoke(name):
            part = selectors[name]
            out = outputs[name]

            def finalize(accumulator):
                out.copy_(accumulator)
                return out

            return layer.moe(chunk, x[part], ids[part], routes[part], finalize=finalize)

        def evaluate(*, inspect_frontend=False, exact=False):
            snapshots = {}
            for name in ('pair', 'first', 'second'):
                outputs[name].fill_(float('nan'))
                invoke(name)
                torch.cuda.synchronize()
                assert bool(torch.isfinite(outputs[name]).all()), name
                if inspect_frontend:
                    snapshots[name] = frontend_snapshot(captured[-1])
                captured.clear()
            single = torch.cat((outputs['first'], outputs['second']))
            pair = outputs['pair'].clone()
            if inspect_frontend:
                assert len(snapshots['pair']) == 16 * 8
                checked = sum(assert_frontend_equal(snapshots[name], snapshots['pair'], offset)
                              for name, offset in (('first', 0), ('second', 8)))
                assert checked == 16 * 8
            difference = cells.fp32_noise(pair, single)
            if exact:
                assert torch.equal(pair, single), difference
            else:
                assert difference['fp32_max_ulps'] <= cells.MAX_ULPS, difference
            # A second execution separates the width comparison from each
            # width's own unspecified FP32 atomic addition order.
            for name in ('second', 'first', 'pair'):
                invoke(name)
            torch.cuda.synchronize()
            noise = dict(single=cells.fp32_noise(torch.cat((outputs['first'], outputs['second'])), single),
                         pair=cells.fp32_noise(outputs['pair'], pair))
            assert all(v['fp32_max_ulps'] <= cells.MAX_ULPS for v in noise.values())
            captured.clear()
            return dict(difference=difference, repeat_noise=noise)

        graphs = {}
        try:
            with patch.object(md, '_STATIC_V2_OVERRIDE', cfg), \
                    patch.object(md, '_get_static_kernel_v2', observe):
                for seed in (918, 919):
                    x.copy_(cells.grouped(16, 2, .7, seed))
                    for routing in ('shared', 'disjoint', 'router'):
                        if routing == 'shared':
                            ids.copy_(torch.arange(8, device='cuda').expand(16, 8))
                            routes.fill_(1. / 8)
                        elif routing == 'disjoint':
                            ids.copy_(torch.arange(128, device='cuda').reshape(16, 8))
                            routes.fill_(1. / 8)
                        else:
                            selected, weights = layer.route(x)
                            ids.copy_(selected); routes.copy_(weights)
                        for scales in ('unit', 'unequal'):
                            layer.q13 = (original_scale if scales == 'unit' else
                                         torch.linspace(.25, 1.75, 288, device='cuda'))
                            result = evaluate(inspect_frontend=True)
                            report('publication_width', layer=layer.L, seed=seed,
                                   routing=routing, input_scales=scales,
                                   compared_routes=128, **result)

                # Keep the same real inputs, routing and FC1 weights. Zeroing
                # the other three down-projection K tiles isolates one native
                # FC2 contribution, including FC1, activation, FP4 search,
                # BF16 rounding and route weighting, before the final sum.
                layer.q13 = original_scale
                saved_routes = routes.clone()
                down = original_view.w2_tiled_storage
                assert tuple(down.shape) == (288, 4, 4096, 64)
                isolated = torch.zeros_like(down)
                layer.views[chunk] = dataclasses.replace(
                    original_view, w2_tiled_storage=isolated,
                    down_fp4=isolated.view(torch.float4_e2m1fn_x2).permute(2, 3, 1, 0))
                for k_tile in range(4):
                    isolated.zero_()
                    isolated[:, k_tile].copy_(down[:, k_tile])
                    for route_slot in (0, 7):
                        routes.zero_()
                        routes[:, route_slot].copy_(saved_routes[:, route_slot])
                        result = evaluate(exact=True)
                        report('publication_contribution', layer=layer.L, k_tile=k_tile,
                               route_slot=route_slot, bit_exact=True, **result)

                # Reuse the actual captured kernels with changed payloads,
                # then exercise zero routes after poisoning the outputs.
                layer.views[chunk] = original_view
                routes.copy_(saved_routes)
                for name in outputs:
                    graphs[name], _ = _capture(lambda name=name: invoke(name))
                captured.clear()
                for replay in range(4):
                    x.copy_(cells.grouped(16, 2, .7, 1920 + replay))
                    selected, weights = layer.route(x)
                    ids.copy_(selected); routes.copy_(weights)
                    if replay == 3:
                        routes.zero_()
                    for name in ('pair', 'second', 'first'):
                        outputs[name].fill_(float('nan'))
                        graphs[name].replay()
                    torch.cuda.synchronize()
                    single = torch.cat((outputs['first'], outputs['second']))
                    assert all(bool(torch.isfinite(out).all()) for out in outputs.values())
                    difference = cells.fp32_noise(outputs['pair'], single)
                    assert difference['fp32_max_ulps'] <= cells.MAX_ULPS, difference
                    if replay == 3:
                        assert all(int(torch.count_nonzero(out)) == 0 for out in outputs.values())
                    report('publication_graph', layer=layer.L, replay=replay,
                           zero_routes=replay == 3, **difference)
                assert configs[8]['input_reuse'] == 3
                assert configs[16]['input_reuse'] == 4
                assert configs[16]['c2_direct_scatter'] and configs[16]['c2_fc2_prefetch']
                report('publication_paths', layer=layer.L, configurations=configs)
        finally:
            for graph in graphs.values():
                graph.reset()
            layer.q13 = original_scale
            layer.views[chunk] = original_view
            captured.clear()
