"""M1b full routed/shared component qualification; no serving speed verdict."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch


def output_error(actual, expected):
    """Fixed BF16 component gate, with bounded scratch even for 32K rows."""
    import math
    import torch
    if actual.shape != expected.shape:
        raise RuntimeError('completion output shape changed')
    max_delta = max_ref = sum_delta = sum_ref = 0.
    for a, b in zip(actual.split(2048), expected.split(2048)):
        if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
            raise RuntimeError('completion output is nonfinite')
        ref, delta = b.float(), a.float()-b.float()
        max_delta = max(max_delta, float(delta.abs().max()))
        max_ref = max(max_ref, float(ref.abs().max()))
        sum_delta += float(delta.square().sum(dtype=torch.float64))
        sum_ref += float(ref.square().sum(dtype=torch.float64))
    count = max(1, actual.numel())
    result = dict(relative_max=max_delta/max(max_ref, 1e-8),
        relative_rms=math.sqrt(sum_delta/count)/max(math.sqrt(sum_ref/count), 1e-8))
    if result['relative_max'] > .02 or result['relative_rms'] > .004:
        raise RuntimeError(f'completion exceeded the fixed BF16 component tolerance: {result}')
    return result


def compare_cold_frontend(ordinary, owner):
    """Every remaining route, matched by expert and original token, byte exact."""
    import torch
    new, plan = owner.workspace, owner.cold
    counts = ordinary.row_counts.cpu().tolist()
    bases = ordinary.expert_tile_base.cpu().tolist()
    if sum(counts) != len(owner.plan.prefill)*8:
        raise RuntimeError('native prefill omitted routes')
    groups = torch.arange(256, device=owner.sources.device)[None, :]
    def sf_indices(rows):
        r = rows[:, None]
        return (r//128)*32768 + (groups//4)*512 + (r%32)*16 + ((r//32)%4)*4 + groups%4
    cursor = 0
    for expert, count in enumerate(plan.counts):
        if not count:
            continue
        source = owner.sources[cursor:cursor+count]
        cursor += count
        old_start = bases[expert]*128
        tokens, order = ordinary.token_map[old_start:old_start+counts[expert]].sort()
        lookup = torch.searchsorted(tokens, source[:, 2].contiguous())
        if bool((lookup >= len(tokens)).any()) or not torch.equal(tokens[lookup], source[:, 2]):
            raise RuntimeError('a cold route did not retain its original token')
        old_rows, new_rows = old_start+order[lookup], source[:, 1].long()
        for a, b in ((ordinary.packed_input.view(-1, 2048)[old_rows], new.packed_input.view(-1, 2048)[new_rows]),
                     (ordinary.scale_flat[sf_indices(old_rows)], new.scale_flat[sf_indices(new_rows)]),
                     (ordinary.token_weights[old_rows], new.token_weights[new_rows])):
            if not torch.equal(a, b):
                raise RuntimeError('cold FP4/SFA or route weight bytes changed')


def measure(args, report):
    import torch
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.b12x.moe_mixed import PreparedMixedExperts
    from engine.kernels.b12x.moe_mixed_completion import PreparedMixedCompletion, shared_ffn
    from engine.kernels.glm_pointwise import route_weights
    from engine.kernels.prefill_router import router_logits
    from engine.modules.mixed_experts import ExpertInvocation
    from engine.profiles.glm53 import facts
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_graph_profile import rank_on_this_node
    from probes.engine_ffn_packets_check import sha_tensor
    from probes.engine_mixed_experts_check import compare_frontend
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('M1b qualification requires GB10/SM121')
    budget = 8 << 30
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    root = Path(args.ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent/root
    rank, prefix = rank_on_this_node(str(root)), 'L3.moe.'
    path = root/f'rank{rank}of4.safetensors'
    model = facts.load(args.ckpt_meta)
    if (model.topk_experts, model.swiglu_limit) != (8, 10.):
        raise RuntimeError('checkpoint does not declare fixed GLM arithmetic')
    names = ('w13', 'w13_sf', 'w2', 'w2_sf', 'gate', 'bias', 'sh_gate_up', 'sh_down')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    loader = rank_loader(path)
    keys = set(loader.keys())
    separate = all(prefix+s in keys for s in scale_names)
    if not separate and any(prefix+s in keys for s in scale_names):
        raise RuntimeError('partial ModelOpt scale contract')
    tensors = loader.load([prefix+s for s in names+(scale_names if separate else ())], device='cuda')
    report.update(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        rank_file=str(path), memory_budget_bytes=budget,
        weight_sha256={k: sha_tensor(v) for k, v in tensors.items()}, routed_scale=model.routed_scale,
        model_config_sha256=hashlib.sha256((Path(args.ckpt_meta)/'config.json').read_bytes()).hexdigest())
    pack = [tensors[prefix+s] for s in names[:4]]
    scales = ModelOptScales.bind(*(tensors[prefix+s] for s in scale_names), experts=288,
                                 device=pack[0].device) if separate else None
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*pack, 8, 10., scales=scales)
    shared = tuple(tensors[prefix+s] for s in ('sh_gate_up', 'sh_down'))
    torch.manual_seed(89515)
    cases = report.setdefault('cases', [])
    cells = [(d, p, False) for d in (8, 32) for p in (9216, 9240, 32768)]
    cells += [(d, 32768, True) for d in (8, 32)]
    for epoch, (d, p, real_router) in enumerate(cells):
        identity = ExpertInvocation(3, epoch, 1, epoch)
        decode = (torch.randn(d, 4096, device='cuda')*.5).bfloat16()
        prefill = (torch.randn(p, 4096, device='cuda')*.5).bfloat16()
        if real_router:
            ids, routes = route_weights(router_logits(decode, tensors[prefix+'gate']), tensors[prefix+'bias'], 8, model.routed_scale)
            pids, proutes = route_weights(router_logits(prefill, tensors[prefix+'gate']), tensors[prefix+'bias'], 8, model.routed_scale)
        else:
            ids = (torch.arange(d*8, device='cuda').view(d, 8)%288).int()
            pids = (torch.arange(p*8, device='cuda').view(p, 8)%288).int()
            routes, proutes = torch.rand(d, 8, device='cuda'), torch.rand(p, 8, device='cuda')
            routes.mul_(model.routed_scale/routes.sum(-1, keepdim=True))
            proutes.mul_(model.routed_scale/proutes.sum(-1, keepdim=True))
            routes[0, 0] = proutes[0, 1] = 0.
        frozen = [sha_tensor(t) for t in (decode, prefill, ids, pids, routes, proutes, *shared)]
        observed_d, observed_p = {}, {}
        original_d, original_p = md.launch_sm120_static_moe, md.launch_sm120_dynamic_moe
        def native_d():
            return lane.moe(decode, ids, routes, *pack, 10., scales=scales) + shared_ffn(decode, shared)
        def native_p():
            return lane.moe(prefill, pids, proutes, *pack, 10., scales=scales) + shared_ffn(prefill, shared)
        def capture_d(**kw):
            observed_d.update(kw); return original_d(**kw)
        def capture_p(**kw):
            observed_p.update(kw); return original_p(**kw)
        with patch.object(md, 'launch_sm120_static_moe', capture_d), \
                patch.object(md, 'launch_sm120_dynamic_moe', capture_p):
            ref_d, ref_p = native_d().clone(), native_p().clone()
        if not observed_d or not observed_p or observed_p['workspace'].tile_m != 128:
            raise RuntimeError('baseline did not execute static decode and dynamic M128 prefill')
        for kw in (observed_d, observed_p):
            if (kw.get('input_scales_are_reciprocal', False) or not kw.get('fast_math', True)
                    or kw['weights'].w1_storage.data_ptr() != observed_d['weights'].w1_storage.data_ptr()):
                raise RuntimeError('baseline packed weights or quantizer contract changed')
        owned = dict(weights=observed_d['weights'], input_scale=observed_d['input_gs'],
            down_scale=observed_d['down_input_scale'], shared_up=shared[0], shared_down=shared[1], identity=identity)
        owners, preparation = [], []
        for quota in (0, 128):
            torch.cuda.synchronize(); start = time.perf_counter()
            owner = PreparedMixedCompletion(decode, prefill, ids, pids, routes, proutes,
                                             **owned, hot_route_quota=quota)
            torch.cuda.synchronize()
            preparation.append(dict(hot_quota=quota, wall_ms=(time.perf_counter()-start)*1000))
            owners.append(owner)
        base, mixed = owners
        base.begin(identity); base_out = base.finish(identity).clone()
        expected = base.hot.partials[:d*32].clone()
        mixed.hot.partials.fill_(float('nan')); mixed.cold_output.fill_(float('nan'))
        actual_d = mixed.begin(identity)
        try:
            mixed.prefill_result(identity)
        except RuntimeError:
            pass
        else:
            raise RuntimeError('partial prefill was published before cold/shared work')
        actual_p = mixed.finish(identity)
        mixed.prefill_ready.synchronize()
        if not torch.isfinite(actual_d).all() or not torch.equal(mixed.hot.partials[:d*32], expected):
            raise RuntimeError('decode route parts changed or retained poison')
        compare_frontend(observed_d['workspace'], mixed.hot, ids)
        compare_cold_frontend(observed_p['workspace'], mixed)
        routed_decode = mixed.hot.partials[:d*32].reshape(d, 32, 4096).sum(1).bfloat16()
        native_routed = observed_d['scatter_output']
        decode_relative = float((routed_decode.float()-native_routed.float()).abs().max()
                                / native_routed.float().abs().max().clamp_min(1e-8))
        if decode_relative > .001:
            raise RuntimeError(f'decode reducer exceeded its fixed .001 native gate: {decode_relative}')
        errors = dict(native_prefill_repeat=output_error(native_p(), ref_p),
            prepared_split=output_error(base_out, ref_p), mixed_prefill=output_error(actual_p, ref_p),
            mixed_vs_split=output_error(actual_p, base_out), decode=output_error(actual_d, ref_d))
        # Moved routes retain the same M16/M32 arithmetic as a separately
        # prepared source window. The full M128 comparison above is a distinct
        # numerical gate, including different intermediate rounding boundaries.
        windows = {}
        for index, source in enumerate(mixed.plan.sources[d*8:], d*8):
            token, slot = source[3:]
            first = min((token//d)*d, p-d)
            windows.setdefault(first, []).append((index, token-first, slot))
        for first, selected in windows.items():
            reference = PreparedMixedExperts(prefill[first:first+d], decode[:1], pids[first:first+d], ids[:1],
                proutes[first:first+d], routes[:1], **{k: v for k, v in owned.items() if not k.startswith('shared_')},
                hot_route_quota=0)
            parts = reference.run(identity)
            for index, token, slot in selected:
                if not torch.equal(mixed.hot.partials.view(-1, 4, 4096)[index], parts[token*8+slot]):
                    raise RuntimeError('moved route changed its source-window contribution')
            del reference, parts
        # One smaller quota checks continuation across many more launches.
        if (d, p, real_router) == (8, 9240, False):
            check = PreparedMixedCompletion(decode, prefill, ids, pids, routes, proutes,
                **owned, hot_route_quota=128, cold_task_quota=7)
            check.begin(identity)
            errors['seven_task_continuation'] = output_error(check.finish(identity), actual_p)
            del check
        for owner in owners:
            owner.begin(identity); owner.finish(identity)
        native_d(); native_p(); torch.cuda.synchronize()
        times = []
        for _ in range(args.samples//2):
            for arm in ('native', 'split', 'mixed', 'mixed', 'split', 'native'):
                start, dec, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
                wall = time.perf_counter(); start.record()
                if arm == 'native':
                    native_d(); dec.record(); native_p()
                else:
                    owner = base if arm == 'split' else mixed
                    owner.begin(identity); dec.record(); owner.finish(identity)
                end.record(); end.synchronize()
                times.append(dict(arm=arm, decode_ready_ms=start.elapsed_time(dec),
                    prefill_complete_ms=start.elapsed_time(end), wall_ms=(time.perf_counter()-wall)*1000))
        if frozen != [sha_tensor(t) for t in (decode, prefill, ids, pids, routes, proutes, *shared)]:
            raise RuntimeError('completion mutated a source or shared weight owner')
        try:
            mixed.begin(replace(identity, epoch=epoch+1))
        except ValueError:
            pass
        else:
            raise RuntimeError('stale completion generation executed')
        cell = dict(decode_rows=d, prefill_rows=p,
            routing='real L3 router on synthetic activations' if real_router else 'controlled tile-boundary fixture',
            work=mixed.plan.work(), cold_task_quota=mixed.cold.task_quota,
            cold_windows=len(mixed.cold.windows), prepare=preparation, preparation_includes_first_use_compile=epoch in (0, 3),
            source_sha256=frozen, errors=errors, timings=times,
            decode_route_parts_byte_exact=True, hot_route_parts_byte_exact=True, cold_fp4_sfa_byte_exact=True,
            native_decode_relative_max=decode_relative,
            cold_complete=True, shared_complete=True, shared_recipe='BF16 linear / clamped SwiGLU / BF16 linear',
            serving_speedup_proven=False, scratch_peak_bytes=torch.cuda.max_memory_allocated())
        cases.append(cell); print(json.dumps(cell), flush=True)
        del owners, owner, base, mixed, actual_d, actual_p, base_out, expected, observed_d, observed_p, owned, ref_d, ref_p
    report.update(status='PASS', gpu_used=True, default_enabled=False)


def main():
    from engine.profiles.glm53 import facts
    from probes.engine_mixed_completion_compile import fingerprint
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', default=str(facts.RANKS))
    parser.add_argument('--ckpt-meta', default=str(facts.CKPT))
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--output', type=Path, default=Path('/cache/mixed-completion.json'))
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 2:
        parser.error('--samples must be even and at least four')
    sources = fingerprint()
    sources['probes/engine_mixed_completion_check.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = dict(status='FAIL', source_sha256=sources, image=os.environ.get('ST_IMAGE'),
        scope='M1b eager one-rank component with full routed and explicit BF16 shared completion; '
              'CPU route preparation is measured separately. Excludes served dense reader selection, '
              'TP4 collectives, arrival-trace scheduling, graph replay and cancellation. '
              'No TTFT, tok/s, quality, acceptance or serving-speed verdict.')
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
