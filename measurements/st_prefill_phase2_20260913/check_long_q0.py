"""Native TP4 MoE canary above the old Q0 ceiling, without a baseline boot.

Uses actual rank-0 ModelOpt weights and the existing route/graph/numerical
oracle. Cold reference kernels are numerical controls, never consumer scores.
"""
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.base.loader import RankLoader
from engine.profiles.glm53.lanes import served
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.kernels.b12x import moe_dispatch as md


def oracle():
    package = '_long_q0_canary'
    module = ModuleType(package)
    module.__path__ = []
    sys.modules[package] = module
    for name in ('glm53_ep_local_selftest', 'glm53_tp_sf6_q0_selftest'):
        path = ROOT / 'overlay/modules/glm53_moe' / (name + '.py')
        spec = importlib.util.spec_from_file_location(package + '.' + name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loaded
        spec.loader.exec_module(loaded)
    return loaded


def main():
    torch.set_num_threads(4)
    start = time.monotonic()
    judge = oracle()
    layers = served(moe_static='t,r,sf6,q0')
    loader = RankLoader(sys.argv[1])
    results, branches = [], set()
    for layer in (3, 44):
        prefix = f'L{layer}.moe.'
        suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
        params = loader.load([prefix + key for key in suffixes])
        p = {key: params[prefix + key] for key in suffixes}
        scales = ModelOptScales.bind(p['w13_alpha'], p['a13_scale'], p['w2_alpha'], p['a2_scale'],
                                    experts=288, device=p['w13'].device)
        captured = []
        original = md._get_weight_views

        def capture(*args, **kwargs):
            views = original(*args, **kwargs)
            captured.append(views)
            return views

        md._get_weight_views = capture
        try:
            layers.moe_prepare(*(p[key] for key in suffixes[:4]), 8, 10., scales=scales)
        finally:
            md._get_weight_views = original
        views = captured[-1]
        packed = bool(views.reform_scales.enabled)
        branches.add(packed)
        experts = SimpleNamespace(_sf6_weight_views=views, g1_alphas=scales.input13,
                                  _fc2_input_scale=scales.input2)
        workspace = md.allocate_sm120_dynamic_workspace(
            state_E=288, weight_E=288, routed_rows=16384 * 8, k=4096, n=512, num_topk=8,
            device=p['w13'].device, activation='swigluoai_uninterleave', quant_mode='nvfp4', tile_m=128)
        # All eight routes are retained, including duplicate/zero-weight routes.
        for case in (('balanced9216', 9216, 'balanced'),
                     ('concentrated13824', 13824, 'concentrated'),
                     ('duplicate16384', 16384, 'duplicate')):
            result = dict(layer=layer, sf6=packed)
            results.append(result)
            judge._case(torch, md, p['w13'].device, experts, workspace, case, result)
            print(json.dumps(result), flush=True)
        del workspace
    if branches != {False, True}:
        raise AssertionError('must cover both SF6 and raw-scale fallback with actual weights')
    report = dict(passed=True, seconds=time.monotonic() - start, cases=results,
                  device=torch.cuda.get_device_name(), torch=torch.__version__,
                  scope='actual rank0 layers 3 and 44; unchanged .02/.04 and stock-noise numerical bounds; not consumer throughput')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
