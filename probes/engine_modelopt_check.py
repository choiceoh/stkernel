"""Judge NVIDIA preshard serving against original per-projection checkpoint bytes.

CPU checks the reference lane and dense model wiring. CUDA additionally checks
the b12x lane, repeat stability and CUDA graph replay. Only one layer's packed
weights is resident; the oracle dequantizes only the selected experts. This is
a component check, not a full-model quality or TP collective qualification.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.base.checkpoint import Checkpoint
from engine.modules import moe
from engine.profiles.glm53 import facts, lanes, modelopt_weights
from engine.profiles.glm53.modelopt_scales import ModelOptScales
from engine.profiles.glm53.net import Glm53Net
from engine.profiles.glm53.weights import rank_loader


def relative(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6))


def source_weights(checkpoint, layer, experts, rank, device, dense):
    """Read and split raw projections without the preshard builder or SF swizzle."""
    out = {}
    for expert in experts:
        prefix = f'model.language_model.layers.{layer}.mlp.'
        if not dense:
            prefix += f'experts.{expert}.'
        keys = [prefix + p + '_proj.' + s for p in ('up', 'gate', 'down')
                for s in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')]
        raw = checkpoint.load(keys, max_run=64 << 20)
        projections = {}
        for p in ('up', 'gate', 'down'):
            tensors = [raw[prefix + p + '_proj.' + s]
                       for s in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')]
            axis = 1 if p == 'down' else 0
            packed, sf = (t.chunk(4, axis)[rank].contiguous().to(device) for t in tensors[:2])
            wg, ag = (t.to(device).reshape(()) for t in tensors[2:])
            projections[p] = (moe.dequant_nvfp4(packed, sf, wg), ag)
        out[expert] = projections
    return out


def oracle(x, ids, route, weights, quant, limit, *, gpu_epilogue=False):
    out = torch.zeros_like(x, dtype=torch.float32)
    for expert, projections in weights.items():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        if not rows.numel():
            continue
        def linear(values, name):
            weight, ag = projections[name]
            packed, sf = quant(values, ag)
            return moe.dequant_nvfp4_act(packed, sf, ag) @ weight.T
        up, gate = (linear(x[rows], name) for name in ('up', 'gate'))
        hidden = lanes.swiglu_clamped(gate, up, limit)
        y = linear(hidden, 'down')
        # The CPU reference rounds each expert output to BF16. The SM121
        # decode epilogue accumulates weighted contributions in FP32.
        if not gpu_epilogue:
            y = y.bfloat16().float()
        out.index_add_(0, rows, y * route[rows, slots, None])
    return out.bfloat16()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True)
    ap.add_argument('--ranks', required=True)
    ap.add_argument('--rank', type=int, choices=range(4), default=0)
    ap.add_argument('--layers', type=int, nargs='+', default=[0, 1, 2, 3, 44])
    ap.add_argument('--tokens', type=int, nargs='+', default=[1, 6, 129])
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    ap.add_argument('--moe-static', default='stock')
    ap.add_argument('--repeats', type=int, default=8)
    ap.add_argument('--oracle-rows', type=int, default=32)
    ap.add_argument('--out')
    a = ap.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(9391)
    gpu = a.device == 'cuda'
    if gpu:
        torch.backends.cuda.matmul.allow_tf32 = False
        from probes.engine_moe_real_check import hardware_quant
        quant = lambda x, ag: hardware_quant(x, ag.reciprocal(), multiplier=True)
    else:
        quant = moe.quant_nvfp4_act
    F = facts.load(a.ranks)
    checkpoint = Checkpoint(a.source)
    loader = rank_loader(Path(a.ranks) / f'rank{a.rank}of4.safetensors', expected_layout=F.weight_layout)
    rows_out = []
    for layer in a.layers:
        lane = lanes.served(moe_static=a.moe_static) if gpu else lanes.reference()
        dense = not F.is_moe(layer)
        contracts = modelopt_weights.quant_specs(F, layer)
        got = loader.load([s.name for s in contracts], device=a.device, max_run=64 << 20)
        prefix = f'L{layer}.' + ('mlp.' if dense else 'moe.')
        w13, sf13, w2, sf2 = (got[prefix + s] for s in ('w13', 'w13_sf', 'w2', 'w2_sf'))
        scales = ModelOptScales.bind(*(got[prefix + s] for s in ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')),
                                    experts=w13.shape[0], device=w13.device)
        experts = [0] if dense else [0, 1, 2, 31, 63, 127, 191, 287]
        source = source_weights(checkpoint, layer, experts, a.rank, a.device, dense)
        if dense:
            comm = SimpleNamespace(world_size=4, rank=a.rank, all_reduce=lambda x: x)
            net = Glm53Net(F, comm, lane, layers=[layer])
            # Restrict storage to this MLP; exercise the real bind and fixed E=1
            # routing without allocating unrelated embeddings or attention.
            net.specs = lambda: contracts
            net.bind(got)
        elif lane.moe_prepare:
            lane.moe_prepare(w13, sf13, w2, sf2, 8, F.swiglu_limit, scales=scales)
        for tokens in a.tokens:
            x = torch.randn(tokens, F.hidden, device=a.device, dtype=torch.bfloat16) * .5
            ids = torch.tensor(experts, device=a.device, dtype=torch.int32).repeat(tokens, 1)
            route = torch.ones(tokens, len(experts), device=a.device)
            if not dense:
                route = torch.rand_like(route)
                route[::2, 0] = 0.  # Zero route weights must contribute nothing.
                route /= route.sum(-1, keepdim=True)
                route *= F.routed_scale
            call = (lambda: net._dense(layer, x)) if dense else (
                lambda: lane.moe(x, ids, route, w13, sf13, w2, sf2, F.swiglu_limit, scales=scales))
            for _ in range(3 if gpu else 0):
                call()
            actual = call().clone()
            sample = torch.linspace(0, tokens - 1, min(tokens, a.oracle_rows), device=a.device).long()
            expected = oracle(x[sample], ids[sample], route[sample], source, quant, F.swiglu_limit, gpu_epilogue=gpu)
            row = dict(layer=layer, rank=a.rank, tokens=tokens, oracle_rows=len(sample), dense=dense,
                       relative=relative(actual[sample], expected), finite=bool(torch.isfinite(actual).all()))
            if gpu:
                repeats = torch.stack([call().clone() for _ in range(a.repeats)])
                row['repeat_relative'] = relative(repeats, actual)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = call()
                owners = lane.graph_resources() if lane.graph_resources else None
                graph_runs = []
                for _ in range(a.repeats):
                    graph.replay()
                    graph_runs.append(captured.clone())
                row['graph_relative'] = relative(torch.stack(graph_runs), actual)
                graph.reset()
                del owners, captured, graph_runs, repeats
            rows_out.append(row)
            print(json.dumps(row), flush=True)
            assert row['finite'] and row['relative'] <= (.02 if gpu else 1e-6), row
            if gpu:
                assert row['repeat_relative'] <= .001 and row['graph_relative'] <= .001, row
        if dense:
            del net
        del source, got, w13, sf13, w2, sf2, scales, lane, call
        if gpu:
            from engine.kernels.b12x.moe_dispatch import clear_sm120_moe_caches
            clear_sm120_moe_caches()  # every graph above is already reset
    result = dict(passed=True, device=a.device, layout=F.weight_layout, moe_static=a.moe_static, checks=rows_out)
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
