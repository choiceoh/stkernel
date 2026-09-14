"""Same-weight FFN packet qualification on one GB10; no NIC or serving-speed claim."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time
import traceback
import unittest
from unittest.mock import patch


def sha_tensor(value):
    return hashlib.sha256(value.contiguous().view(__import__('torch').uint8).cpu().numpy().tobytes()).hexdigest()


def frontend(workspace, rows):
    """Compare by (expert, token), since atomic route packing permutes physical rows."""
    import torch
    counts = workspace.row_counts.cpu().tolist()
    bases = workspace.expert_tile_base.cpu().tolist()
    if sum(counts) != rows*8:
        raise RuntimeError('route histogram must contain precisely the real top-8 rows')
    a = workspace.packed_input.view(-1, 2048)
    sf = workspace.packed_input_scale.view(-1)
    groups = torch.arange(256, device=a.device)[None, :]
    result = []
    for expert, count in enumerate(counts):
        start = bases[expert]*128
        tokens, order = workspace.token_map[start:start+count].sort()
        if count and (int(tokens[0]) < 0 or int(tokens[-1]) >= rows or torch.unique(tokens).numel() != count):
            raise RuntimeError('padded or duplicate tokens entered the expert routes')
        physical = start + order
        # The inherited SF6 Q0 producer's group16 scale storage address.
        r = physical[:, None]
        index = (r >> 7)*(64*512) + (r & 31)*16 + ((r >> 5) & 3)*4 + (groups >> 2)*512 + (groups & 3)
        result.append(dict(expert=expert, count=count, tokens=sha_tensor(tokens),
            route_weights=sha_tensor(workspace.token_weights[physical]),
            fp4=sha_tensor(a[physical]), group16_scales=sha_tensor(sf[index])))
    return result


def output_error(actual, expected):
    import torch
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise RuntimeError('FFN output is nonfinite')
    delta, reference = actual.float()-expected.float(), expected.float()
    result = dict(relative_max=float(delta.abs().max()/reference.abs().max().clamp_min(1e-8)),
                  relative_rms=float(delta.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-8)))
    # BF16 atomic scatter can reorder between launches. Frontend bytes and
    # shared output are checked exactly; this fixed component tolerance does
    # not qualify model acceptance, generated tokens or serving quality.
    if result['relative_max'] > .02 or result['relative_rms'] > .004:
        raise RuntimeError(f'cross-launch BF16 scatter exceeded the declared component tolerance: {result}')
    return result


def paired_summary(samples):
    """Keep every B/A/A/B cycle; a favorable median is not a latency verdict."""
    if len(samples) != 2 or len(samples[0]) != len(samples[1]) or len(samples[0]) < 4 or len(samples[0]) % 2:
        raise ValueError('paired timings require two equally sized arms and complete B/A/A/B cycles')
    means = [statistics.mean(v) for v in samples]
    cycles = [[statistics.mean(v[i:i+2]) for v in samples] for i in range(0, len(samples[0]), 2)]
    return dict(mean_ms=means, mean_change_pct=100*(means[1]/means[0]-1),
                cycle_mean_ms=cycles, cycle_change_pct=[100*(a/b-1) for b, a in cycles])


class RoutedFixture:
    """Four senders for numerics; one sender's preparation in each timed arm.

    Peer preparation is cached to model concurrent remote ranks. Concatenation
    stands in for all-gather on this single GPU; this is not NIC latency proof.
    """
    def __init__(self, rows, gate, bias, scale, rank):
        import torch
        from engine.kernels.glm_pointwise import route_weights
        from engine.kernels.prefill_collectives import PrefillCollectives
        from engine.kernels.prefill_collectives.routes import pack_routed
        from engine.kernels.prefill_router import router_shard_logits
        from engine.modules.prefill_packets import PacketBatch, PacketGeometry
        self.g = PacketGeometry(rows, (rows+3)//4)
        self.rg = PacketGeometry(rows, self.g.local_rows, routed=True)
        self.rank, self.source, self.payloads = rank, [], [[], []]
        self.local_logits = []
        def select(x):
            return route_weights(router_shard_logits(x, gate), bias, 8, scale)
        self.select = select
        torch.manual_seed(rows)
        for r in range(4):
            x = torch.randn((self.g.local_rows, 4096), device='cuda', dtype=torch.bfloat16)*(2**r)
            x[0].zero_()
            x[-1].copy_(torch.exp2(torch.arange(32, device='cuda') % 16-8).repeat_interleave(128))
            if r == 3 and self.g.padded_rows != rows:
                x[-(self.g.padded_rows-rows):].zero_()
            self.source.append(x)
            plain = PrefillCollectives.pack(x, x.numel())[0]
            def inspect(roundtrip):
                logits = router_shard_logits(roundtrip, gate)
                self.local_logits.append(logits)
                return route_weights(logits, bias, 8, scale)
            routed = pack_routed(x, self.rg, inspect)
            if not torch.equal(plain, routed[:self.g.stride]):
                raise RuntimeError('routed sender changed FP8 activation values, scales or alignment gap')
            self.payloads[0].append(plain)
            self.payloads[1].append(routed)
        self.sender_logits = torch.cat(self.local_logits)[:rows]
        del self.local_logits
        self.batches = [PacketBatch(torch.cat(payloads), geometry)
                        for payloads, geometry in zip(self.payloads, (self.g, self.rg))]

    def exchange(self, routed):
        import torch
        from engine.kernels.prefill_collectives import PrefillCollectives
        from engine.kernels.prefill_collectives.routes import pack_routed
        from engine.modules.prefill_packets import PacketBatch
        x = self.source[self.rank]
        own = pack_routed(x, self.rg, self.select) if routed else PrefillCollectives.pack(x, x.numel())[0]
        payloads = list(self.payloads[int(routed)])
        payloads[self.rank] = own
        return PacketBatch(torch.cat(payloads), self.rg if routed else self.g)


def unpack_batch(batch):
    import torch
    from engine.kernels.prefill_collectives.kernels import _unpack_gather
    g = batch.geometry
    x = torch.empty((g.padded_rows, 4096), device='cuda', dtype=torch.bfloat16)
    _unpack_gather[(x.numel()//g.block,)](batch.received.view(torch.float8_e4m3fn),
        batch.received.view(torch.float32), x, g.local_elements, g.stride, BLOCK=g.block)
    return x[:g.rows]


def measure_router(args, report):
    """Sender ownership exactness and a complete local route preparation bracket."""
    import torch
    import triton
    from engine.kernels.glm_pointwise import route_weights
    from engine.kernels.prefill_collectives.consumer import quantize_gather
    from engine.kernels.prefill_collectives.routes import packet_routes
    from engine.kernels.prefill_router import router_logits
    from engine.profiles.glm53 import facts
    from engine.profiles.glm53.weights import rank_loader
    from probes.engine_graph_profile import rank_on_this_node
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('sender routing requires GB10/SM121')
    budget = 8 << 30
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    root = Path(args.ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent/root
    rank = rank_on_this_node(str(root))
    path = root/f'rank{rank}of4.safetensors'
    loaded = rank_loader(path).load(['L3.moe.gate', 'L3.moe.bias'], device='cuda')
    gate, bias = loaded['L3.moe.gate'], loaded['L3.moe.bias']
    model = facts.load(args.ckpt_meta)
    report.update(device=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__,
        cuda=torch.version.cuda, memory_budget_bytes=budget,
        weights=dict(rank=rank, rank_file=str(path), source_sha256={k: sha_tensor(v) for k,v in loaded.items()}),
        routed_scale=model.routed_scale, cases=[])
    for rows in (8193, 8194, 8195, 9216, 32768):
        report['active_case'] = dict(rows=rows)
        fixture = RoutedFixture(rows, gate, bias, model.routed_scale, rank)
        expected_logits = router_logits(unpack_batch(fixture.batches[0]), gate)
        expected = route_weights(expected_logits, bias, 8, model.routed_scale)
        actual = packet_routes(fixture.batches[1])
        if sha_tensor(expected_logits) != sha_tensor(fixture.sender_logits):
            raise RuntimeError('sender shard GEMM changed long-prefill logits')
        if [sha_tensor(v) for v in expected] != [sha_tensor(v) for v in actual]:
            raise RuntimeError('route metadata changed top-8 IDs or FP32 weights')
        shared = [quantize_gather(b.received, b.geometry.local_rows, real_rows=rows,
                                  routed=b.geometry.routed) for b in fixture.batches]
        if [sha_tensor(v) for v in shared[0]] != [sha_tensor(v) for v in shared[1]]:
            raise RuntimeError('routed packet stride changed shared quantization')
        def pipeline(routed):
            batch = fixture.exchange(routed)
            return (packet_routes(batch) if routed else
                    route_weights(router_logits(unpack_batch(batch), gate), bias, 8, model.routed_scale))
        for arm in (False, True):
            pipeline(arm)
        times, wall_times = [[], []], [[], []]
        for _ in range(args.samples//2):
            for arm in (False, True, True, False):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record(); output = pipeline(arm); end.record(); end.synchronize()
                times[int(arm)].append(start.elapsed_time(end))
                wall_times[int(arm)].append((time.perf_counter()-wall)*1000)
                del output
        cell = dict(rows=rows, status='EXACT', logits_byte_exact=True, routes_byte_exact=True,
                    shared_byte_exact=True, route_sha256=[sha_tensor(v) for v in actual],
                    workspace=fixture.rg.workspace(), milliseconds=times, wall_milliseconds=wall_times,
                    paired_device=paired_summary(times), paired_wall=paired_summary(wall_times))
        report['cases'].append(cell)
        print(json.dumps(cell), flush=True)
    report.pop('active_case', None)
    report.update(status='PASS', gpu_used=True, default_enabled=False,
        max_allocated_bytes=torch.cuda.max_memory_allocated())


def measure(args, report):
    import torch
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense.fp8 import quantize
    from engine.kernels.glm_pointwise import route_weights, swiglu_clamped
    from engine.kernels.prefill_collectives import BLOCK
    from engine.kernels.prefill_collectives.consumer import quantize_gather
    from engine.kernels.prefill_collectives.kernels import _unpack_gather
    from engine.kernels.prefill_router import router_logits
    from engine.kernels.prefill_collectives.routes import packet_routes
    from engine.modules.prefill_packets import PacketBatch, PacketGeometry
    from engine.profiles.glm53.lanes import served
    from engine.profiles.glm53.modelopt_scales import ModelOptScales
    from engine.profiles.glm53.weights import rank_loader
    from engine.profiles.glm53 import facts
    from probes.engine_graph_profile import rank_on_this_node
    from tests.test_engine_prefill_fp8_consumer import PrefillConsumerTests

    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('this qualification requires GB10/SM121')
    budget = 8 << 30
    torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory)
    suite = unittest.defaultTestLoader.loadTestsFromNames((
        'tests.test_engine_ffn_packets', 'tests.test_engine_prefill_fp8_consumer'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful() or result.skipped:
        raise RuntimeError('packet CPU/transport/router qualification failed or skipped')
    report.update(unit_tests=result.testsRun, device=torch.cuda.get_device_name(),
                  torch=torch.__version__, cuda=torch.version.cuda, memory_budget_bytes=budget)

    root = Path(args.ranks)
    if not root.is_absolute():
        root = facts.RANKS.parent/root
    rank, prefix = rank_on_this_node(str(root)), 'L3.moe.'
    path = root/f'rank{rank}of4.safetensors'
    model = facts.load(args.ckpt_meta)
    if model.swiglu_limit != 10. or model.topk_experts != 8:
        raise RuntimeError('checkpoint does not declare the fixed GLM FFN arithmetic')
    report['model'] = dict(metadata=args.ckpt_meta, routed_scale=model.routed_scale,
        config_sha256=hashlib.sha256((Path(args.ckpt_meta)/'config.json').read_bytes()).hexdigest())
    loader = rank_loader(path)
    suffixes = ('w13', 'w13_sf', 'w2', 'w2_sf', 'gate', 'bias', 'sh_gate_up', 'sh_down')
    scale_names = ('w13_alpha', 'a13_scale', 'w2_alpha', 'a2_scale')
    keys = set(loader.keys())
    separate = all(prefix+s in keys for s in scale_names)
    if not separate and any(prefix+s in keys for s in scale_names):
        raise RuntimeError('partial ModelOpt scale contract')
    loaded = loader.load([prefix+s for s in suffixes+(scale_names if separate else ())], device='cuda')
    report['weights'] = dict(rank=rank, rank_file=str(path), scales='ModelOpt' if separate else 'folded',
        source_sha256={k: sha_tensor(v) for k, v in loaded.items()})
    weights = [loaded[prefix+s] for s in suffixes[:4]]
    scales = (ModelOptScales.bind(*(loaded[prefix+s] for s in scale_names), experts=288,
                                 device=weights[0].device) if separate else None)
    lane = served(moe_static='t,r,sf6')
    lane.moe_prepare(*weights, 8, 10., scales=scales)
    bound = dict(w13=weights[0], w13_sf=weights[1], w2=weights[2], w2_sf=weights[3], limit=10., scales=scales)
    expert = partial(lane.moe, **bound)
    packet_expert = partial(lane.moe_packets, **bound)
    gate, bias = (loaded[prefix+s] for s in ('gate', 'bias'))
    shared_up, shared_down = (FP8Linear(loaded[prefix+s]) for s in ('sh_gate_up', 'sh_down'))
    cases = report.setdefault('cases', [])

    for rows in (8193, 8194, 8195, 9216, 32768):
        report['active_case'] = dict(rows=rows, phase='capability')
        if not lane.moe_packets_supported(rows, **bound):
            raise RuntimeError('prepared real weights refused the packet consumer; no skipped cell is a pass')
        fixture = RoutedFixture(rows, gate, bias, model.routed_scale, rank)
        g, batch = fixture.rg, fixture.batches[1]
        input_hash = sha_tensor(batch.received)

        def unpack():
            return unpack_batch(fixture.batches[0])

        expected_logits = router_logits(unpack(), gate)
        if sha_tensor(expected_logits) != sha_tensor(fixture.sender_logits):
            raise RuntimeError('sender shard GEMM changed long-prefill logits')
        del expected_logits

        def ffn(packets, *, prepare=False):
            packet = fixture.exchange(packets) if prepare else fixture.batches[int(packets)]
            x = None if packets else unpack_batch(packet)
            if packets:
                ids, routes = packet_routes(packet)
            else:
                ids, routes = route_weights(router_logits(x, gate), bias, 8, model.routed_scale)
            routed = packet_expert(packet, ids, routes) if packets else expert(x, ids, routes)
            q, s = (quantize_gather(packet.received, g.local_rows, real_rows=rows, routed=True)
                    if packets else quantize(x))
            up = shared_up.project_quantized(q, s)
            a, b = up.chunk(2, -1)
            shared = shared_down(swiglu_clamped(a, b, 10.))
            return routed+shared, (ids, routes, q, s, shared)

        observed = {}
        original = md.launch_sm120_dynamic_moe
        def capture_launch(**kw):
            observed.update(kw)
            return original(**kw)

        # Warm both actual variants before comparing or timing them.
        report['active_case']['phase'] = 'warmup'
        for arm in (False, True):
            ffn(arm)
        with patch.object(md, 'launch_sm120_dynamic_moe', capture_launch):
            base, base_parts = ffn(False)
        if not observed or observed.get('_packet_input') is not None:
            raise RuntimeError('baseline did not execute the ordinary persistent dynamic kernel')
        workspace = observed['workspace']
        baseline_frontend = frontend(workspace, rows)
        hashes = [sha_tensor(v) for v in base_parts]
        del base_parts
        with patch.object(md, 'launch_sm120_dynamic_moe', capture_launch):
            candidate, candidate_parts = ffn(True)
        if observed.get('_packet_input') is not batch or observed['workspace'] is not workspace:
            raise RuntimeError('candidate did not execute the explicit packet ABI on the same workspace')
        report['active_case']['phase'] = 'byte-comparison'
        candidate_frontend = frontend(workspace, rows)
        names = ('route_ids', 'route_weights', 'shared_q', 'shared_scales', 'shared_output')
        changed_parts = [name for name, actual, expected in zip(names, candidate_parts, hashes)
                         if sha_tensor(actual) != expected]
        changed_experts = [actual['expert'] for actual, expected in zip(candidate_frontend, baseline_frontend)
                           if actual != expected]
        if changed_parts or changed_experts:
            raise RuntimeError(f'packet bytes changed: parts={changed_parts}, experts={changed_experts}')
        del candidate_parts
        errors = output_error(candidate, base)
        repeat, parts = ffn(False)
        variance = output_error(repeat, base)
        del repeat, parts, base, candidate

        # Force unequal per-expert input scales even for folded checkpoint
        # packs. Both arms use the same synthetic scales and same epilogue.
        report['active_case']['phase'] = 'unequal-scales'
        launch_args = dict(observed)
        launch_args['input_gs'] = torch.exp2(torch.arange(288, device='cuda') % 7 - 3).float()
        x = unpack()
        direct = []
        front = []
        for packets in (False, True):
            launch_args.update(a=None if packets else x, _packet_input=batch if packets else None)
            direct.append(original(**launch_args).clone())
            front.append(frontend(workspace, rows))
        if front[0] != front[1]:
            raise RuntimeError('per-expert unequal input scale FP4 boundary changed')
        unequal_error = output_error(direct[1], direct[0])
        del x, direct, front, launch_args

        times, wall_times = [[], []], [[], []]
        report['active_case']['phase'] = 'timing'
        for _ in range(args.samples//2):
            for arm in (False, True, True, False):  # same-build B/A/A/B
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter()
                start.record()
                outputs = ffn(arm, prepare=True)
                end.record(); end.synchronize()
                wall_times[int(arm)].append((time.perf_counter()-wall)*1000)
                times[int(arm)].append(start.elapsed_time(end))
                del outputs
        if sha_tensor(batch.received) != input_hash:
            raise RuntimeError('a consumer modified its packet owner')
        medians = [statistics.median(v) for v in times]
        wall_medians = [statistics.median(v) for v in wall_times]
        cell = dict(rows=rows, real_weight_frontend_byte_exact=True, shared_byte_exact=True,
            unequal_expert_scales_byte_exact=True, errors=errors, baseline_variance=variance,
            unequal_scales_error=unequal_error, output_part_sha256=hashes,
            packet_sha256=input_hash, workspace=g.workspace(), milliseconds=times,
            median_ms=medians, change_pct=100*(medians[1]/medians[0]-1),
            wall_milliseconds=wall_times, wall_median_ms=wall_medians,
            wall_change_pct=100*(wall_medians[1]/wall_medians[0]-1),
            paired_device=paired_summary(times), paired_wall=paired_summary(wall_times))
        cases.append(cell)
        print(json.dumps(cell), flush=True)
        observed.clear()
    report.pop('active_case', None)
    report.update(status='PASS', gpu_used=True, default_enabled=False,
                  max_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    from engine.profiles.glm53 import facts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', default=str(facts.RANKS))
    parser.add_argument('--ckpt-meta', default=str(facts.CKPT))
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--router-only', action='store_true', help='sender routing preparation and exactness; no full FFN qualification')
    parser.add_argument('--output', type=Path, default=Path('/cache/ffn-packets.json'))
    args = parser.parse_args()
    if args.samples < 4 or args.samples % 2:
        parser.error('--samples must be even and at least four')
    from probes.engine_ffn_packets_compile import fingerprint
    sources = fingerprint()
    root = Path(__file__).resolve().parents[1]
    for p in ('probes/engine_ffn_packets_check.py', 'tests/test_engine_ffn_packets.py',
              'tests/test_engine_prefill_fp8_consumer.py', 'engine/kernels/glm_pointwise.py',
              'engine/profiles/glm53/modelopt_scales.py'):
        sources[p] = hashlib.sha256((root/p).read_bytes()).hexdigest()
    report = dict(status='FAIL', scope='one GB10, one sender pack and routes plus emulated all-gather through full L3 FFN output; '
                  'synthetic activations, actual rank weights; other three senders prepared outside timing; '
                  'torch.cat substitutes for communication; excludes NIC/reduce-scatter, '
                  'TTFT, generation tok/s and model quality/acceptance', source_sha256=sources,
                  image=os.environ.get('ST_IMAGE'), started=time.time())
    try:
        if args.router_only:
            report['scope'] = ('one GB10 actual L3 router weights, synthetic inputs, sender pack/route and receiver routes; '
                'one sender preparation timed, peers prepared outside timing, torch.cat emulates all-gather; '
                'excludes expert/shared GEMMs, NIC and serving performance')
            measure_router(args, report)
        else:
            measure(args, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
