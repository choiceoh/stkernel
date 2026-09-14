"""M2 one-GB10 served-reader and cancellation gate; TP4 host proof is separate."""
import argparse
import cProfile
from contextlib import contextmanager, ExitStack
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def fingerprint():
    from probes.engine_mixed_completion_compile import fingerprint as core
    paths = ('engine/base/comm.py', 'engine/modules/mixed_tickets.py',
        'engine/profiles/glm53/mixed_shared.py', 'engine/profiles/glm53/net.py',
        'engine/profiles/glm53/lanes.py', 'engine/profiles/glm53/modelopt_scales.py',
        'engine/kernels/dense/__init__.py', 'engine/kernels/dense/shared_mlp.py',
        'engine/kernels/dense/kernels.cu', 'engine/kernels/dense/fp8.py',
        'probes/engine_mixed_tickets_check.py', 'probes/engine_mixed_completion_check.py',
        'probes/engine_mixed_plan_bench.py',
        'probes/engine_mixed_prepare_bench.py',
        'tests/test_engine_mixed_tickets.py', 'tests/test_engine_mixed_shared.py')
    root = Path(__file__).resolve().parents[1]
    return dict(core(), **{p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


@contextmanager
def planning_arm(arm):
    """Probe-only A/B: identical readers/kernels, fresh host preparation per arm."""
    from unittest.mock import patch
    from engine.kernels.b12x import moe_mixed
    from engine.modules import mixed_tickets
    from probes.engine_mixed_plan_bench import legacy_plan, legacy_signature
    from probes.engine_mixed_prepare_bench import previous_routes, torch_check_values, BlockingMetadata
    stages = {}
    def timed(name, fn):
        def call(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                if name == 'value_check_ms' and result is not None:
                    from types import SimpleNamespace
                    return SimpleNamespace(wait=timed('value_check_wait_ms', result.wait))
                return result
            finally:
                stages[name] = (time.perf_counter()-start)*1000
        return call
    def scalar_routes(decode, prefill, *, cold_task_quota=None, **options):
        from engine.modules.mixed_completion import plan_cold
        plan = legacy_plan(decode, prefill, **options)
        return plan, None if cold_task_quota is None else plan_cold(plan, task_quota=cold_task_quota)
    old = arm in ('legacy', 'packed_v1')
    planner = scalar_routes if arm == 'legacy' else previous_routes if old else moe_mixed.prepare_routes
    metadata_type = BlockingMetadata if old else moe_mixed.MixedMetadata
    class MeasuredMetadata:
        def __init__(self, *args, **kwargs):
            self.owner = metadata_type(*args, **kwargs)
        def __getitem__(self, name):
            start = time.perf_counter()
            try:
                return self.owner[name]
            finally:
                stages['metadata_reads_ms'] = stages.get('metadata_reads_ms', 0.) + (time.perf_counter()-start)*1000
    with ExitStack() as stack:
        for module, name, field, fn in (
                (moe_mixed, 'prepare_routes', 'route_plan_ms', planner),
                (moe_mixed, 'check_values', 'value_check_ms', torch_check_values if old else moe_mixed.check_values),
                (moe_mixed, 'MixedMetadata', 'metadata_prepare_ms', MeasuredMetadata),
                (mixed_tickets, 'signature', 'agreement_ms',
                 legacy_signature if arm == 'legacy' else mixed_tickets.signature)):
            stack.enter_context(patch.object(module, name, timed(field, fn)))
        yield stages


def measure(args, report):
    import torch
    from engine.base.comm import Comm
    from engine.kernels.dense import DenseLinear
    from engine.kernels.dense.shared_mlp import SharedMLP, SharedOverlap
    from engine.modules.mixed_experts import ExpertInvocation
    from engine.modules.mixed_tickets import MixedLayerScheduler
    from engine.profiles.glm53 import facts
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.net import Glm53Net
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_graph_profile import rank_on_this_node
    from probes.engine_ffn_packets_check import sha_tensor
    from probes.engine_mixed_completion_check import output_error
    from probes.engine_mixed_prepare_bench import value_check_gate, cold_token_gate

    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('mixed ticket qualification requires GB10/SM121')
    budget = 8 << 30
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    report['value_check_gate'] = value_check_gate()
    report['cold_token_gate'] = cold_token_gate()
    root = Path(args.ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent/root
    rank, prefix = rank_on_this_node(str(root)), 'L3.moe.'
    path = root/f'rank{rank}of4.safetensors'
    F = facts.load(args.ckpt_meta)
    if (F.hidden, F.experts, F.topk_experts, F.moe_inter_local, F.spec_k, F.swiglu_limit) != (4096, 288, 8, 512, 7, 10.):
        raise RuntimeError('mixed ticket gate requires the fixed GLM TP4 arithmetic')
    names = ('w13', 'w13_sf', 'w2', 'w2_sf', 'gate', 'bias', 'sh_gate_up', 'sh_down')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    loader = rank_loader(path)
    keys = set(loader.keys())
    separate = all(prefix+s in keys for s in scale_names)
    if not separate and any(prefix+s in keys for s in scale_names):
        raise RuntimeError('partial ModelOpt scale contract')
    weights = loader.load([prefix+s for s in names+(scale_names if separate else ())], device='cuda')
    report.update(device=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        rank_file=str(path), memory_budget_bytes=budget, weight_sha256={k: sha_tensor(v) for k, v in weights.items()},
        model_config_sha256=hashlib.sha256((Path(args.ckpt_meta)/'config.json').read_bytes()).hexdigest())
    pack = [weights[prefix+s] for s in names[:4]]
    scales = ModelOptScales.bind(*(weights[prefix+s] for s in scale_names), experts=288,
        device=pack[0].device) if separate else None
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*pack, 8, 10., scales=scales)
    # Only L3 FFN resources fit this component's budget. The profile metadata
    # describes one TP4 weight shard; no collective uses this metadata Comm.
    # Actual bind() scale forwarding is covered by the CPU integration test.
    net = Glm53Net(F, Comm(4, rank), lane, layers=[3])
    net.p = weights
    bound = dict(zip(('w13', 'w13_sf', 'w2', 'w2_sf'), pack), limit=10., scales=scales)
    net._experts[3] = partial(lane.moe, **bound)
    net._mixed_experts[3] = partial(lane.moe_mixed_prepare, **bound)
    net._router_weights[3] = weights[prefix+'gate'].float()
    for suffix in ('sh_gate_up', 'sh_down'):
        net.dense[prefix+suffix] = DenseLinear(weights[prefix+suffix], name=prefix+suffix)
    net.shared_mlp[3] = SharedMLP(net.dense[prefix+'sh_gate_up'], net.dense[prefix+'sh_down'], 10.)
    net.shared_overlap = SharedOverlap(pack[0].device)
    shared_packs = tuple(t for reader in net.dense.values() for t in
        (*[t for p in reader.packs for t in (p.data, p.scale, p.rowscale)], *reader.fp8.weight))
    frozen_packs = [sha_tensor(t) for t in shared_packs]
    report.update(shared_pack_sha256=frozen_packs,
        shared_recipe='same DenseLinear W4/FP8 packs in all arms; RTN from checkpoint BF16, no GPTQ store',
        communication='world=1 identity, one real TP4 weight shard; no NCCL or four-GPU result')
    torch.manual_seed(89516)
    scheduler, consumer = MixedLayerScheduler(3, Comm()), torch.cuda.Stream()
    cases, generation = report.setdefault('cases', []), 0
    for d, p in ((8, 9240), (32, 9240), (8, 32768), (32, 32768)):
        x = (torch.randn(d, 4096, device='cuda')*.5).bfloat16()
        pref = (torch.randn(p, 4096, device='cuda')*.5).bfloat16()
        frozen = [sha_tensor(t) for t in (x, pref)]
        def native():
            return (net._moe(3, x, reduce=lambda value: value),
                    net._moe(3, pref, reduce=lambda value: value))
        ref_d, ref_p = (t.clone() for t in native())
        torch.cuda.synchronize()
        cell = dict(decode_rows=d, prefill_rows=p, source_sha256=frozen,
            routing='actual profile L3 router on synthetic activations', samples=[], native_samples=[])
        def measure_native(sample):
            # The adoption baseline is the actual homogeneous FFN, including
            # routing/shared work and the same decode-output synchronization.
            torch.cuda.synchronize(); wall = time.perf_counter()
            actual_d = net._moe(3, x, reduce=lambda value: value)
            torch.cuda.synchronize(); decoded = time.perf_counter()
            actual_p = net._moe(3, pref, reduce=lambda value: value)
            torch.cuda.synchronize(); completed = time.perf_counter()
            cell['native_samples'].append(dict(sample=sample,
                decode_ready_wall_ms=(decoded-wall)*1000,
                prefill_complete_wall_ms=(completed-wall)*1000,
                includes_first_use_compile=sample == 0,
                errors=dict(decode=output_error(actual_d, ref_d), prefill=output_error(actual_p, ref_p))))
        # Every mixed sample prepares fresh routes and storage. Report all host
        # planning/admission and first-use compilation, never amortize them as
        # if dynamic arrivals reused a fixed input/route histogram.
        for sample in range(args.samples):
            if sample % 2 == 0:
                measure_native(sample)
            for quota in (0, 128) if sample % 2 == 0 else (128, 0):
                baseline = 'packed_v1' if args.compare_preparation else 'legacy'
                arms = ((baseline, 'packed_v2') if sample % 2 == 0 else ('packed_v2', baseline)) \
                    if args.compare_planning or args.compare_preparation else ('packed_v2',)
                for arm in arms:
                    generation += 1
                    identity = ExpertInvocation(3, generation, generation, generation)
                    with planning_arm(arm) as stages:
                        torch.cuda.synchronize(); wall = time.perf_counter()
                        key = net.submit_mixed_ffn(scheduler, x, pref, identity=identity,
                            request=f'cell-{d}-{p}-{generation}', slot=0, hot_route_quota=quota)
                        prepared = time.perf_counter()
                    scheduler.begin(key)
                    actual_d, decode_event = scheduler.result(key, prefill=False)
                    decode_event.synchronize(); decoded = time.perf_counter()
                    windows = 0
                    if args.drain_cold:
                        scheduler.drain(key)
                        windows = 1
                    else:
                        while True:
                            windows += 1
                            if scheduler.advance(key):
                                break
                    scheduler.finish(key)
                    actual_p, prefill_event = scheduler.result(key, prefill=True)
                    prefill_event.synchronize(); completed = time.perf_counter()
                    # Exercise a real foreign-stream consumer and its last-reader
                    # fence. The scheduler retains both outputs until it finishes.
                    with torch.cuda.stream(consumer):
                        consumer.wait_event(decode_event); consumer.wait_event(prefill_event)
                        copy_d, copy_p = actual_d.clone(), actual_p.clone()
                        used = torch.cuda.Event(); used.record(consumer)
                    scheduler.release(key, consumer_fence=used)
                    used.synchronize(); torch.cuda.synchronize()
                    if not scheduler.reap(key):
                        raise RuntimeError('fully drained ticket did not retire')
                    errors = dict(decode=output_error(copy_d, ref_d), prefill=output_error(copy_p, ref_p))
                    cell['samples'].append(dict(hot_quota=quota, planning_arm=arm, planning_stages_ms=stages,
                        cold_windows=windows, cold_dispatch='drain' if args.drain_cold else 'bounded', errors=errors,
                        prepare_admit_wall_ms=(prepared-wall)*1000,
                        decode_ready_wall_ms=(decoded-wall)*1000, prefill_complete_wall_ms=(completed-wall)*1000,
                        includes_first_use_compile=sample == 0))
                    del actual_d, actual_p, copy_d, copy_p
            if sample % 2:
                measure_native(sample)
            again_d, again_p = native()
            cell['native_repeat_error'] = dict(decode=output_error(again_d, ref_d), prefill=output_error(again_p, ref_p))
            del again_d, again_p
        if d == 8 and p == 9240:
            for phase in ('queued', 'decode', 'cold'):
                generation += 1
                key = net.submit_mixed_ffn(scheduler, x, pref,
                    identity=ExpertInvocation(3, generation, generation, generation),
                    request='cancel-'+phase, slot=0)
                if phase != 'queued': scheduler.begin(key)
                if phase == 'cold': scheduler.advance(key)
                scheduler.cancel(key)
                torch.cuda.synchronize()
                if not scheduler.reap(key):
                    raise RuntimeError('cancelled readers did not drain')
                try:
                    scheduler.begin(key)
                except RuntimeError:
                    pass
                else:
                    raise RuntimeError('retired ticket dispatched into a reused slot')
            cell['cancellation_phases'] = ['queued', 'decode', 'cold']
        if d == 8 and p == 32768:
            # A separate, warm preparation sample attributes Python/C++ host
            # time (including waits). Never include profiling overhead in A/B.
            generation += 1
            profile = cProfile.Profile()
            profile.enable()
            try:
                key = net.submit_mixed_ffn(scheduler, x, pref,
                    identity=ExpertInvocation(3, generation, generation, generation),
                    request='preparation-profile', slot=0)
            finally:
                profile.disable()
            # Separate attribution invocation: event overhead is not part of
            # the native/mixed wall-time samples above.
            owner = scheduler._entries[key.slot].owner
            timed_events = {}
            def event_call(name, fn):
                def call(*args):
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record(); result = fn(*args); end.record()
                    timed_events[name] = (start, end)
                    return result
                return call
            owner.producer = event_call('cold_pack_ms', owner.producer)
            owner.compiled = event_call('cold_compute_ms', owner.compiled)
            event_call('decode_ms', scheduler.begin)(key)
            event_call('cold_drain_ms', scheduler.drain)(key)
            event_call('prefill_finish_ms', scheduler.finish)(key)
            scheduler.release(key)
            torch.cuda.synchronize()
            if not scheduler.reap(key):
                raise RuntimeError('profiled preparation did not retire')
            report['gpu_phase_profile'] = {name: start.elapsed_time(end)
                for name, (start, end) in timed_events.items()}
            import pstats
            entries = pstats.Stats(profile).stats
            report['preparation_profile'] = [dict(file=f, line=line, function=name,
                primitive_calls=v[0], total_calls=v[1], self_ms=v[2]*1000, cumulative_ms=v[3]*1000)
                for (f, line, name), v in sorted(entries.items(), key=lambda item: item[1][3], reverse=True)[:50]]
        if frozen != [sha_tensor(t) for t in (x, pref)]:
            raise RuntimeError('ticket changed its prepared source')
        cases.append(cell); print(json.dumps(cell), flush=True)
        del x, pref, ref_d, ref_p
    if frozen_packs != [sha_tensor(t) for t in shared_packs]:
        raise RuntimeError('ticket changed the served shared weight packs')
    report.update(status='PASS', gpu_used=True, scratch_peak_bytes=torch.cuda.max_memory_allocated(),
        default_enabled=False, serving_speedup_proven=False)


def main():
    from engine.profiles.glm53 import facts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', default=str(facts.RANKS))
    parser.add_argument('--ckpt-meta', default=str(facts.CKPT))
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--compare-planning', action='store_true',
        help='alternate pre-optimization scalar/JSON preparation and packed planning on the same build')
    parser.add_argument('--compare-preparation', action='store_true',
        help='compare previous packed components with joint planning, fused checks and one metadata upload')
    parser.add_argument('--drain-cold', action='store_true',
        help='explicitly drain cold work in one launch after decode; no interleaved decode arrival is promised')
    parser.add_argument('--output', type=Path, default=Path('/cache/mixed-tickets.json'))
    args = parser.parse_args()
    if args.samples < 2 or args.samples % 2:
        parser.error('--samples must be even and at least two')
    if args.compare_planning and args.compare_preparation:
        parser.error('choose one preparation baseline')
    report = dict(status='FAIL', source_sha256=fingerprint(), image=os.environ.get('ST_IMAGE'),
        compare_planning=args.compare_planning,
        compare_preparation=args.compare_preparation,
        drain_cold=args.drain_cold,
        scope='M2 eager one-rank FFN component, real served shared readers, ticket retirement and foreign-stream '
              'consumers. Includes fresh route planning/allocation/admission in wall timings. '
              'No full model, TP4 NCCL, arrival trace, graph, TTFT, tok/s, quality or acceptance verdict.')
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
