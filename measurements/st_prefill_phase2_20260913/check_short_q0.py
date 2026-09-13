"""Actual-weight short-prefill Q0 scalar/word controls; no baseline engine."""
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import torch
from engine.base.loader import RankLoader
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.kernels.b12x import moe_dispatch as md
from check_long_q0 import oracle


def main():
    torch.set_num_threads(4)
    started = time.monotonic()
    judge, lanes = oracle(), served(moe_static='t,r,sf6,q0')
    loader = RankLoader(sys.argv[1])
    report, branches, selected_rows = [], set(), set()
    selector, launch = md._short_prefill_q0_word_unpack, md.launch_sm120_dynamic_moe
    def witnessed_selector(**kwargs):
        selected = selector(**kwargs)
        if selected:
            selected_rows.add(kwargs['m'])
        return selected

    # Reuse the existing numerical, route-byte, changed-input and graph
    # oracle. Its binary selector chooses word decoding here; both controls
    # retain the same original BF16 GEMM/scatter arithmetic and storage.
    def selected_launch(**kwargs):
        candidate = kwargs.pop('_tp_sf6_q0_override')
        with patch.object(md, '_short_prefill_q0_word_unpack',
                          witnessed_selector if candidate else lambda **kw: False):
            return launch(**kwargs, _tp_sf6_q0_override=True)

    for layer in (3, 44):
        prefix = f'L{layer}.moe.'
        suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
        params = loader.load([prefix + key for key in suffixes])
        p = {key: params[prefix+key] for key in suffixes}
        scales = ModelOptScales.bind(*(p[k] for k in suffixes[4:]), experts=288, device=p['w13'].device)
        views_seen, original = [], md._get_weight_views
        def capture(*args, **kwargs):
            views = original(*args, **kwargs)
            views_seen.append(views)
            return views
        with patch.object(md, '_get_weight_views', capture):
            lanes.moe_prepare(*(p[k] for k in suffixes[:4]), 8, 10., scales=scales)
        views = views_seen[-1]
        packed = bool(views.reform_scales.enabled)
        branches.add(packed)
        experts = SimpleNamespace(_sf6_weight_views=views, g1_alphas=scales.input13,
                                  _fc2_input_scale=scales.input2)
        workspace = md.allocate_sm120_dynamic_workspace(
            state_E=288, weight_E=288, routed_rows=8192*8, k=4096, n=512,
            num_topk=8, device=p['w13'].device, activation='swigluoai_uninterleave',
            quant_mode='nvfp4', tile_m=128)
        with patch.object(md, 'launch_sm120_dynamic_moe', selected_launch):
            for case in (('balanced2121',2121,'balanced'),
                         ('concentrated2128',2128,'concentrated'),
                         ('duplicate8192',8192,'duplicate'),
                         ('zeros2128',2128,'zeros')):
                row = dict(layer=layer, sf6=packed, selection='word decoding versus original Q0 producer with the same FP32 scatter')
                report.append(row)
                judge._case(torch, md, p['w13'].device, experts, workspace, case, row)
                print(json.dumps(row), flush=True)
        del workspace
    if branches != {False, True}:
        raise AssertionError('both actual packed SF6 and raw-scale fallback must execute')
    pairs = []
    for key in md._DYNAMIC_KERNEL_CACHE:
        if key[-1] == 'short_prefill_q0_words_v1':
            if key[:-1] not in md._DYNAMIC_KERNEL_CACHE:
                raise AssertionError('candidate lacks a same-geometry scalar control')
            pairs.append(repr(key))
    if not pairs or selected_rows != {2121,2128,8192}:
        raise AssertionError(f'missing compiled word path or executed row count: {selected_rows}')
    print(json.dumps(dict(passed=True, seconds=time.monotonic()-started,
                         cases=report, candidate_keys=pairs, selected_rows=sorted(selected_rows),
                         scope='same MoE math, actual rank0 weights, eager/graph and route bytes; no throughput verdict')),
          flush=True)


if __name__ == '__main__':
    main()
