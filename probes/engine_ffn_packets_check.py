"""Same-weight FFN packet qualification on one GB10; no NIC or serving-speed claim."""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
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


def measure(args, report):
    import torch
    from engine.kernels.b12x import moe_dispatch as md
    from engine.kernels.dense import FP8Linear
    from engine.kernels.dense.fp8 import quantize
    from engine.kernels.glm_pointwise import route_weights, swiglu_clamped
    from engine.kernels.prefill_collectives import BLOCK
    from engine.kernels.prefill_collectives.consumer import quantize_gather
    from engine.kernels.prefill_collectives.kernels import _unpack_gather
    from engine.kernels.prefill_router import router_logits, router_packet_logits
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
        if not lane.moe_packets_supported(rows, **bound):
            raise RuntimeError('prepared real weights refused the packet consumer; no skipped cell is a pass')
        g = PacketGeometry(rows, (rows+3)//4)
        batch = PacketBatch(PrefillConsumerTests.received(g.local_rows, seed=rows), g)
        input_hash = sha_tensor(batch.received)

        def unpack():
            x = torch.empty((g.padded_rows, 4096), device='cuda', dtype=torch.bfloat16)
            _unpack_gather[(x.numel()//BLOCK,)](batch.received.view(torch.float8_e4m3fn),
                batch.received.view(torch.float32), x, g.local_elements, g.stride, BLOCK=BLOCK)
            return x[:rows]

        def ffn(packets):
            x = None if packets else unpack()
            logits = router_packet_logits(batch, gate) if packets else router_logits(x, gate)
            ids, routes = route_weights(logits, bias, 8, 2.5)
            routed = packet_expert(batch, ids, routes) if packets else expert(x, ids, routes)
            q, s = quantize_gather(batch.received, g.local_rows, real_rows=rows) if packets else quantize(x)
            up = shared_up.project_quantized(q, s)
            a, b = up.chunk(2, -1)
            shared = shared_down(swiglu_clamped(a, b, 10.))
            return routed+shared, (logits, ids, routes, q, s, shared)

        observed = {}
        original = md.launch_sm120_dynamic_moe
        def capture_launch(**kw):
            observed.update(kw)
            return original(**kw)

        # Warm both actual variants before comparing or timing them.
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
        if frontend(workspace, rows) != baseline_frontend or [sha_tensor(v) for v in candidate_parts] != hashes:
            raise RuntimeError('router, routes, per-expert FP4/scales or shared consumer changed')
        del candidate_parts
        errors = output_error(candidate, base)
        repeat, parts = ffn(False)
        variance = output_error(repeat, base)
        del repeat, parts, base, candidate

        # Force unequal per-expert input scales even for folded checkpoint
        # packs. Both arms use the same synthetic scales and same epilogue.
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

        times = [[], []]
        for _ in range(args.samples//2):
            for arm in (False, True, True, False):  # same-build B/A/A/B
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                outputs = ffn(arm)
                end.record(); end.synchronize()
                times[int(arm)].append(start.elapsed_time(end))
                del outputs
        if sha_tensor(batch.received) != input_hash:
            raise RuntimeError('a consumer modified its packet owner')
        medians = [statistics.median(v) for v in times]
        cell = dict(rows=rows, real_weight_frontend_byte_exact=True, shared_byte_exact=True,
            unequal_expert_scales_byte_exact=True, errors=errors, baseline_variance=variance,
            unequal_scales_error=unequal_error, output_part_sha256=hashes,
            packet_sha256=input_hash, workspace=g.workspace(), milliseconds=times,
            median_ms=medians, change_pct=100*(medians[1]/medians[0]-1))
        cases.append(cell)
        print(json.dumps(cell), flush=True)
        observed.clear()
    report.update(status='PASS', gpu_used=True, default_enabled=False,
                  max_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    from engine.profiles.glm53 import facts
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranks', default=str(facts.RANKS))
    parser.add_argument('--samples', type=int, default=8)
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
    report = dict(status='FAIL', scope='one GB10, received FP8 packets through full L3 FFN output; '
                  'synthetic activations, actual rank weights; excludes pack/all-gather/reduce-scatter, NIC, '
                  'TTFT, generation tok/s and model quality/acceptance', source_sha256=sources,
                  image=os.environ.get('ST_IMAGE'), started=time.time())
    try:
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
