"""Qualify M0/M1 hot routes on one GB10; cold completion/serving remain unimplemented."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch


def compare_frontend(ordinary, prepared, ids):
    """The ordinary atomic row allocation may permute both experts and rows."""
    import torch
    ws = prepared.workspace
    count = int(ordinary.active_expert_count.cpu()[0])
    expert_map = ordinary.weight_expert_ids[:count].cpu().tolist()
    counts = ordinary.row_counts[:count].cpu().tolist()
    indices = {expert: local for local, expert in enumerate(expert_map)}
    groups = torch.arange(256, device=ids.device)
    ordinary_sf = ordinary.packed_input_scale.reshape(288, -1)
    mixed_sf = ws.packed_input_scale.reshape(288, -1)
    def sf_row(row):
        return (row//128)*32768 + (groups//4)*512 + (row%32)*16 + ((row//32)%4)*4 + groups%4
    for source in prepared.plan.sources[:prepared.plan.decode_routes]:
        local, row, kind, token, slot = source
        old_local = indices[prepared.plan.experts[local]]
        matches = (ordinary.token_map[old_local, :counts[old_local]] == token).nonzero().flatten()
        if len(matches) != 1:
            raise RuntimeError('ordinary decode route did not map to exactly one source token')
        old_row = int(matches[0])
        for a, b in ((ordinary.packed_input[old_local, old_row], ws.packed_input[local, row]),
                     (ordinary_sf[old_local, sf_row(old_row)], mixed_sf[local, sf_row(row)]),
                     (ordinary.token_weights[old_local, old_row], ws.token_weights[local, row])):
            if not torch.equal(a, b):
                raise RuntimeError('ordinary and explicit producer FP4/SFA/route weights differ')


def measure(args, report):
    import torch
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.b12x.moe_mixed import PreparedMixedExperts
    from engine.kernels.glm_pointwise import route_weights
    from engine.kernels.prefill_router import router_logits
    from engine.modules.mixed_experts import ExpertInvocation
    from engine.profiles.glm53 import facts
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_graph_profile import rank_on_this_node
    from probes.engine_ffn_packets_check import sha_tensor
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('the mixed-route component requires GB10/SM121')
    budget = 8 << 30
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    root = Path(args.ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent/root
    rank = rank_on_this_node(str(root))
    path, prefix = root/f'rank{rank}of4.safetensors', 'L3.moe.'
    model = facts.load(args.ckpt_meta)
    if (model.topk_experts, model.swiglu_limit) != (8, 10.):
        raise RuntimeError('checkpoint does not declare the fixed GLM arithmetic')
    names = ('w13', 'w13_sf', 'w2', 'w2_sf', 'gate', 'bias')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    loader = rank_loader(path)
    keys = set(loader.keys())
    separate = all(prefix+s in keys for s in scale_names)
    if not separate and any(prefix+s in keys for s in scale_names):
        raise RuntimeError('partial ModelOpt scale contract')
    tensors = loader.load([prefix+s for s in names+(scale_names if separate else ())], device='cuda')
    report.update(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        rank_file=str(path), memory_budget_bytes=budget,
        weight_sha256={k: sha_tensor(v) for k, v in tensors.items()},
        routed_scale=model.routed_scale, model_config_sha256=hashlib.sha256((Path(args.ckpt_meta)/'config.json').read_bytes()).hexdigest())
    pack = [tensors[prefix+s] for s in names[:4]]
    scales = ModelOptScales.bind(*(tensors[prefix+s] for s in scale_names), experts=288,
                                 device=pack[0].device) if separate else None
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*pack, 8, 10., scales=scales)
    torch.manual_seed(89514)
    report['cases'] = []
    cells = [(d, p, False) for d in (8, 32) for p in (9216, 9240, 32768)]
    cells += [(d, 32768, True) for d in (8, 32)]
    for epoch, (rows, prefill_rows, real_router) in enumerate(cells):
        identity = ExpertInvocation(3, epoch, 1, epoch)
        decode = (torch.randn(rows, 4096, device='cuda')*.5).bfloat16()
        prefill = (torch.randn(prefill_rows, 4096, device='cuda')*.5).bfloat16()
        if real_router:
            ids, routes = route_weights(router_logits(decode, tensors[prefix+'gate']), tensors[prefix+'bias'], 8, model.routed_scale)
            pids, proutes = route_weights(router_logits(prefill, tensors[prefix+'gate']), tensors[prefix+'bias'], 8, model.routed_scale)
        else:
            ids = (torch.arange(rows*8, device='cuda').view(rows, 8)%288).int()
            pids = (torch.arange(prefill_rows*8, device='cuda').view(prefill_rows, 8)%288).int()
            routes = torch.rand(rows, 8, device='cuda')
            proutes = torch.rand(prefill_rows, 8, device='cuda')
            routes.mul_(model.routed_scale/routes.sum(-1, keepdim=True))
            proutes.mul_(model.routed_scale/proutes.sum(-1, keepdim=True))
            routes[0, 0] = 0.
            proutes[0, 1] = 0.  # still computed/overwritten; never removed from the route set
        observed = {}
        original = md.launch_sm120_static_moe
        def capture(**kw):
            observed.update(kw)
            return original(**kw)
        def native():
            return lane.moe(decode, ids, routes, *pack, 10., scales=scales)
        with patch.object(md, 'launch_sm120_static_moe', capture):
            native_output = native().clone()
        if not observed or observed['num_tokens'] != rows:
            raise RuntimeError('native baseline did not execute the static body')
        if observed.get('input_scales_are_reciprocal', False) or not observed.get('fast_math', True):
            raise RuntimeError('native baseline did not use the producer quantization contract')
        owned = dict(weights=observed['weights'], input_scale=observed['input_gs'],
                     down_scale=observed['down_input_scale'], identity=identity)
        # Full host preparation is reported separately: copies, route histogram,
        # validation, metadata/workspace allocation, and cold-route enumeration.
        owners, preparation = [], []
        for quota in (0, 128):
            torch.cuda.synchronize()
            start = time.perf_counter()
            owner = PreparedMixedExperts(decode, prefill, ids, pids, routes, proutes,
                                         **owned, hot_route_quota=quota)
            torch.cuda.synchronize()
            preparation.append((time.perf_counter()-start)*1000)
            owners.append(owner)
        base, mixed = owners
        frozen = [sha_tensor(v) for v in (decode, prefill, ids, pids, routes, proutes)]
        expected = base.run(identity).clone()
        mixed.partials.fill_(float('nan'))
        actual = mixed.run(identity)
        if not torch.isfinite(actual).all() or not torch.equal(actual[:rows*8], expected):
            raise RuntimeError('mixed decode route/part outputs changed or retained poison')
        compare_frontend(observed['workspace'], mixed, ids)
        # Price/review the route reducer separately from the served FP32 atomics.
        decoded = actual[:rows*8].reshape(rows, 32, 4096).sum(1).bfloat16()
        denominator = native_output.float().abs().max().clamp_min(1e-8)
        relative = float((decoded.float()-native_output.float()).abs().max()/denominator)
        variance = float((native().float()-native_output.float()).abs().max()/denominator)
        if max(relative, variance) > .001:
            raise RuntimeError(f'private reducer exceeded fixed native tolerance: {relative=}, {variance=}')
        # Every moved prefill route is checked against a separate source window
        # using the SAME M16/M32 body and exact prepared weights/scales.
        windows = {}
        for index, source in enumerate(mixed.plan.sources[rows*8:], rows*8):
            local, dest, kind, token, slot = source
            first = min((token//rows)*rows, prefill_rows-rows)
            windows.setdefault(first, []).append((index, token-first, slot))
        for first, selected in windows.items():
            reference = PreparedMixedExperts(prefill[first:first+rows], decode[:1],
                pids[first:first+rows], ids[:1], proutes[first:first+rows], routes[:1],
                **owned, hot_route_quota=0)
            partial = reference.run(identity)
            for index, token, slot in selected:
                if not torch.equal(actual[index], partial[token*8+slot]):
                    raise RuntimeError('moved prefill route contribution changed')
            del reference, partial
        # A second, unequal expert-scale fixture must preserve decode FP4/SFA.
        unequal = dict(observed, input_gs=torch.exp2(torch.arange(288, device='cuda')%7-3).float())
        original(**unequal)
        check = PreparedMixedExperts(decode, prefill, ids, pids, routes, proutes,
            **dict(owned, input_scale=unequal['input_gs']), hot_route_quota=128)
        check.run(identity)
        compare_frontend(observed['workspace'], check, ids)
        del check
        # All arms are warmed before samples. No cold/shared work is included:
        # these samples can reject hot decode interference, never prove a gain.
        def run_owner(owner):
            output = owner.run(identity)
            return output[:rows*8].reshape(rows, 32, 4096).sum(1).bfloat16()
        run_owner(base); run_owner(mixed); native()
        times = []
        for _ in range(args.samples//2):
            for arm in ('native', 'split', 'mixed', 'mixed', 'split', 'native'):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                output = native() if arm == 'native' else run_owner(base if arm == 'split' else mixed)
                end.record(); end.synchronize()
                times.append(dict(arm=arm, ms=begin.elapsed_time(end)))
        if frozen != [sha_tensor(v) for v in (decode, prefill, ids, pids, routes, proutes)]:
            raise RuntimeError('mixed component mutated an input owner')
        try:
            mixed.run(replace(identity, slot_generation=2))
        except ValueError:
            pass
        else:
            raise RuntimeError('stale slot generation executed')
        routes.add_(0.)  # even a value-preserving in-place write invalidates the descriptor
        try:
            mixed.run(identity)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('mutated route owner executed')
        cell = dict(rows=rows, prefill_rows=prefill_rows,
            routing='real L3 router on synthetic activations' if real_router else 'controlled tile-boundary fixture',
            work=mixed.plan.work(), decode_counts=mixed.plan.decode_counts, hot_counts=mixed.plan.hot_counts,
            prefill_counts=mixed.plan.prefill_counts,
            prepare_ms=preparation, preparation_includes_first_use_compile=(epoch == 0 or rows == 32 and epoch == 3),
            timings=times, decode_route_parts_byte_exact=True, hot_route_parts_byte_exact=True,
            fp4_sfa_byte_exact=True, unequal_expert_scale_frontend_byte_exact=True,
            native_relative_max=relative, native_repeat_relative_max=variance,
            source_sha256=frozen, cold_complete=False, shared_complete=False,
            serving_speedup_proven=False, scratch_peak_bytes=torch.cuda.max_memory_allocated())
        report['cases'].append(cell)
        print(json.dumps(cell), flush=True)
        del owners, base, mixed, owner, actual, expected, output, observed, unequal, owned
    report.update(status='PASS', gpu_used=True, default_enabled=False)


def main():
    from engine.profiles.glm53 import facts
    from probes.engine_mixed_experts_compile import fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', default=str(facts.RANKS))
    parser.add_argument('--ckpt-meta', default=str(facts.CKPT))
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('/cache/mixed-experts.json'))
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 2:
        parser.error('--samples must be even and at least four')
    sources = fingerprint()
    sources['probes/engine_mixed_experts_check.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(status='FAIL', scope='M0 work accounting and M1 hot component only; '
        'excludes cold-route M128 execution, shared expert, rank agreement and serving scheduler; '
        'no TTFT, tok/s, quality, acceptance or end-to-end speed verdict',
        source_sha256=sources, image=os.environ.get('ST_IMAGE'))
    try:
        measure(args, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
